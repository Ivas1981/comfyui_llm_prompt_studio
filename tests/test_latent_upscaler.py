"""Tests for the model-based latent upscaler.

The ttl-nn ``.pt`` weights ship with the project, so we can validate that the
reconstructed ``_TtlLatentResizer`` architecture actually loads them (proving the
key names match) and that a forward pass scales the latent by the requested factor.
"""

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PT = os.path.join(ROOT, "models", "upscale_models", "sdxl_resizer.pt")

from comfyui_llm_prompt_studio.nodes import _latent_upscaler as lu


def test_ttl_loads_real_weights_and_scales():
    if not os.path.isfile(PT):
        import pytest
        pytest.skip("sdxl_resizer.pt not present")
    model = lu._TtlLatentResizer.load_model(PT, torch.device("cpu"), torch.float32)
    latent = torch.zeros((1, 4, 32, 32))
    with torch.no_grad():
        out = model(0.13025 * latent, scale=2.0) / 0.13025
    assert out.shape == (1, 4, 64, 64)


def test_ttl_state_dict_keys_match():
    if not os.path.isfile(PT):
        import pytest
        pytest.skip("sdxl_resizer.pt not present")
    sd = torch.load(PT, map_location="cpu", weights_only=True)
    # from_state_dict builds the exact architecture and load_state_dict(strict=True)
    model = lu._TtlLatentResizer.from_state_dict(sd)
    assert model is not None


def test_project_local_helpers():
    ups = lu.project_local_upscale_models()
    assert any(f.endswith(".safetensors") for f in ups)
    assert any(f.endswith(".pt") for f in ups)
    bbox = lu.project_local_ultralytics_bbox()
    assert any(f.endswith(".pt") for f in bbox)


def test_city96_net_scales_by_factor():
    net = lu._LatentUpscalerNet(2)
    x = torch.zeros((1, 4, 16, 16))
    with torch.no_grad():
        out = net(x)
    assert out.shape == (1, 4, 32, 32)


def test_model_native_scale_parses_filename():
    # Baked scale must win over the requested factor (the bug that produced "каша").
    assert lu.model_native_scale("latent-upscaler-v2.1_SDxl-x1.5.safetensors", 2.0) == 1.5
    assert lu.model_native_scale("latent-upscaler-v2.1_SDv1-x2.0.safetensors", 2.0) == 2.0
    assert lu.model_native_scale("latent-upscaler-v2.1_SDxl-x1.25.safetensors", 2.0) == 1.25
    # No scale in the name (e.g. ttl .pt) falls back to the requested factor.
    assert lu.model_native_scale("sdxl_resizer.pt", 2.0) == 2.0
    assert lu.model_native_scale("some-model.safetensors", 2.0) == 2.0


def test_latent_upscale_with_model_uses_native_scale(tmp_path, monkeypatch):
    # A 1.5x checkpoint must upscale 1.5x even when the caller requests factor=2.0.
    net = lu._LatentUpscalerNet(1.5)
    import safetensors.torch as sf
    path = tmp_path / "fake-x1.5.safetensors"
    sf.save_file({k: v.clone() for k, v in net.state_dict().items()}, str(path))

    monkeypatch.setattr(lu, "_resolve_path", lambda name: str(path))
    latent = {"samples": torch.zeros((1, 4, 32, 32))}
    out = lu.latent_upscale_with_model(latent, "fake-x1.5.safetensors", 2.0)
    # 32 * 1.5 = 48, not 64 (which a 2.0 factor would have produced).
    assert out["samples"].shape[2] == 48
    assert out["samples"].shape[3] == 48


# -- memory guards (OOM in the resizer's global attention) --------------------
def test_chunked_attention_matches_full_attention(monkeypatch):
    # The chunked path exists only to bound the [B,N,N] score matrix; the result must be
    # numerically identical to a single scaled_dot_product_attention call.
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 96, 16) for _ in range(3))
    reference = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    monkeypatch.setattr(lu, "_ATTN_SCORE_BUDGET", 4 * 1024)  # forces several chunks
    chunked = lu._attention(q, k, v)
    assert torch.allclose(reference, chunked, atol=1e-6)


def test_tile_positions_cover_the_whole_axis():
    assert lu._tile_positions(30, 64, 8) == [0]
    pos = lu._tile_positions(100, 40, 8)
    assert pos[0] == 0 and pos[-1] == 60          # last tile is flush with the edge
    assert all(b - a <= 32 for a, b in zip(pos, pos[1:]))


def test_tiled_forward_is_exact_for_a_local_op():
    # nearest upsample is purely local, so overlapping tiles blend identical values and
    # the tiled result must match the untiled one exactly.
    torch.manual_seed(0)
    x = torch.randn(1, 4, 40, 56)
    fn = lambda t: torch.nn.functional.interpolate(t, scale_factor=2.0, mode="nearest")
    tiled = lu._tiled_forward(fn, x, 2.0, tile=16)
    assert tiled.shape == (1, 4, 80, 112)
    assert torch.allclose(tiled, fn(x), atol=1e-5)


def test_tiled_forward_covers_output_for_fractional_scale():
    # A fractional scale (City96 x1.5, or a non-integer ttl factor) makes the net emit a size
    # that round(h*scale) does not match - so the output size must be read from the net itself,
    # otherwise a strip is left uncovered (black row/column in the hires latent) or the tiles
    # are written to the wrong (too-large) buffer. Use a floor-based net, like F.interpolate /
    # City96 Upsample, which is exactly the case round() used to get wrong.
    x = torch.ones(1, 4, 45, 45)
    fn = lambda t: torch.nn.functional.interpolate(t, scale_factor=1.5, mode="nearest")
    tiled = lu._tiled_forward(fn, x, 1.5, tile=24)
    whole = fn(x)
    assert tiled.shape == whole.shape            # size derived from the net, not round()
    assert torch.allclose(tiled, whole)          # every output cell is written and matches
    assert torch.all(tiled > 0.5)                 # no uncovered (zero-weight) cells


def test_tiled_forward_tile_equal_to_overlap_still_covers_everything():
    # hires_latent_upscale_tile=8 with the default _TILE_OVERLAP=8 used to collapse the
    # stepping window (step = max(1, tile-overlap) -> 1) into one tile per latent cell -
    # tens of thousands of tiny forwards for a small latent, millions for a 1024x768 one,
    # i.e. an effective hang that looked like a broken/"one tile" output. The effective
    # overlap is now capped at tile//2, so the whole frame is still covered.
    x = torch.randn(1, 4, 96, 128)
    fn = lambda t: torch.nn.functional.interpolate(t, scale_factor=1.25, mode="nearest")
    tiled = lu._tiled_forward(fn, x, 1.25, tile=8)   # overlap defaults to _TILE_OVERLAP=8
    whole = fn(x)
    assert tiled.shape == whole.shape
    assert torch.isfinite(tiled).all()
    assert tiled.abs().max() > 0                       # full frame written, not a single tile


def test_tiled_model_upscale_matches_untiled(tmp_path, monkeypatch):
    net = lu._LatentUpscalerNet(2)
    import safetensors.torch as sf
    path = tmp_path / "fake-x2.0.safetensors"
    sf.save_file({k: v.clone() for k, v in net.state_dict().items()}, str(path))
    monkeypatch.setattr(lu, "_resolve_path", lambda name: str(path))

    torch.manual_seed(0)
    latent = {"samples": torch.randn((1, 4, 48, 48))}
    plain = lu.latent_upscale_with_model(latent, "fake-x2.0.safetensors", 2.0)
    tiled = lu.latent_upscale_with_model(latent, "fake-x2.0.safetensors", 2.0, tile=16)
    assert tiled["samples"].shape == plain["samples"].shape == (1, 4, 96, 96)
    # Convs see less context at tile borders, so only the overall result must stay close.
    assert (tiled["samples"] - plain["samples"]).abs().mean() < 0.1


def test_auto_tile_is_bounded_and_aligned():
    cpu = torch.device("cpu")
    tile_x2 = lu._tile_edge(lu._TILE_TOKENS_CPU, 2.0)
    tile_x4 = lu._tile_edge(lu._TILE_TOKENS_CPU, 4.0)
    assert tile_x2 % 8 == 0 and tile_x2 >= lu._MIN_TILE
    assert tile_x4 <= tile_x2                      # bigger scale -> smaller input tile
    # The conv net has no attention, so it is allowed much larger tiles.
    assert lu._tile_edge(lu._TILE_TOKENS_CONV, 2.0) > tile_x2
    assert lu._tile_edge(64, 8.0) == lu._MIN_TILE   # never below the floor


def test_plan_tile_prefers_whole_latent_until_the_budget_is_hit():
    cpu = torch.device("cpu")
    # A 3072px target (384x384 latent) is over the CPU budget and gets tiled...
    assert lu._plan_tile(cpu, True, 2.0, 384 * 384) > 0
    # ...while a 2048px one (128x128 -> 256x256) still runs in one piece (best quality).
    assert lu._plan_tile(cpu, True, 2.0, 256 * 256) == 0
    # force_tiles (the OOM retry) tiles regardless of size.
    assert lu._plan_tile(cpu, True, 2.0, 32 * 32, force_tiles=True) > 0
    # The attention-free conv net is never tiled at these sizes.
    assert lu._plan_tile(cpu, False, 2.0, 512 * 512) == 0


def test_attention_resizer_tiles_without_growing_memory():
    # Small randomly-initialised ttl-shaped net (one attention block per stage): the tiled
    # run must produce the same shape as the untiled one.
    torch.manual_seed(0)
    model = lu._TtlLatentResizer(2, {1}, 2, {1})
    x = torch.randn(1, 4, 40, 40)
    with torch.no_grad():
        plain = model(x, scale=2.0)
        tiled = lu._tiled_forward(lambda t: model(t, scale=2.0), x, 2.0, tile=16)
    assert plain.shape == tiled.shape == (1, 4, 80, 80)


def test_attention_resizer_tiles_fractional_scale():
    # Same idea with a non-integer scale: the tiled output must still cover the full frame
    # and match the untiled one (the output size is taken from the net via the probe).
    torch.manual_seed(0)
    model = lu._TtlLatentResizer(2, {1}, 2, {1})
    x = torch.randn(1, 4, 45, 45)
    with torch.no_grad():
        plain = model(x, scale=1.5)
        tiled = lu._tiled_forward(lambda t: model(t, scale=1.5), x, 1.5, tile=24)
    assert plain.shape == tiled.shape == (1, 4, 68, 68)
    # Tiling changes normalisation/attention statistics, so the values only approximately
    # match the whole-latent run - the important guarantee is full coverage (no gaps).
    assert torch.isfinite(tiled).all()
    assert tiled.abs().max() > 0


def test_oom_detection_recognises_cpu_allocator_failure():
    err = RuntimeError(
        "[enforce fail at alloc_cpu.cpp:117] data. DefaultCPUAllocator: not enough "
        "memory: you tried to allocate 3600000000 bytes.")
    assert lu._is_oom(err)
    assert lu._is_oom(RuntimeError("CUDA out of memory."))
    assert lu._is_oom(RuntimeError("DefaultCPUAllocator: can't allocate memory: ..."))
    assert not lu._is_oom(ValueError("unsupported extension"))
    # Shape/key mismatches must surface as themselves, not as a retried "OOM".
    assert not lu._is_oom(RuntimeError("Error(s) in loading state_dict: size mismatch"))


def test_oom_detection_never_swallows_a_user_interrupt(monkeypatch):
    class _Interrupt(Exception):
        pass

    class _MM:
        InterruptProcessingException = _Interrupt
        OOM_EXCEPTION = Exception          # ComfyUI's fallback on some backends

    monkeypatch.setattr(lu, "_model_management", lambda: _MM)
    # OOM_EXCEPTION == Exception must not turn every error into an OOM...
    assert not lu._is_oom(ValueError("boom"))
    # ...and ComfyUI's stop button must propagate even if the message mentions memory.
    assert not lu._is_oom(_Interrupt("out of memory"))
    assert lu._is_oom(RuntimeError("CUDA out of memory."))


def test_memory_estimate_counts_one_attention_chunk():
    # The estimate feeds model_management.free_memory, which unloads the diffusion model
    # when the request does not fit, so it must not be padded.
    est = lu._estimate_bytes(64 * 64, True)
    activations = 64 * 64 * 128 * 4 * 8
    assert est == 64 * 1024 * 1024 + activations + lu._ATTN_SCORE_BUDGET
    # The attention-free conv net needs no score budget at all.
    assert lu._estimate_bytes(64 * 64, False) < est


def test_device_request_covers_only_the_first_pass(tmp_path, monkeypatch):
    net = lu._LatentUpscalerNet(2)
    import safetensors.torch as sf
    path = tmp_path / "fake-x2.0.safetensors"
    sf.save_file({k: v.clone() for k, v in net.state_dict().items()}, str(path))
    monkeypatch.setattr(lu, "_resolve_path", lambda name: str(path))

    asked = []
    monkeypatch.setattr(lu, "_select_device",
                        lambda latent, needed: asked.append(needed) or torch.device("cpu"))
    latent = {"samples": torch.zeros((1, 4, 16, 16))}
    lu.latent_upscale_with_model(latent, "fake-x2.0.safetensors", 2.0, iterations=3)
    # 3 iterations compound to 64x the tokens, but only one pass is requested up front.
    assert asked == [lu._estimate_bytes(32 * 32, False)]


def test_tiled_forward_reuses_the_blend_mask(monkeypatch):
    built = []
    real_mask = lu._feather_mask

    def counting_mask(h, w, overlap, device, dtype):
        built.append((h, w))
        return real_mask(h, w, overlap, device, dtype)

    monkeypatch.setattr(lu, "_feather_mask", counting_mask)
    x = torch.ones(1, 4, 96, 96)
    fn = lambda t: torch.nn.functional.interpolate(t, scale_factor=2.0, mode="nearest")
    lu._tiled_forward(fn, x, 2.0, tile=32)
    # 9 tiles of identical shape must share a single mask.
    assert built == [(64, 64)]


def test_upscale_retries_with_tiles_then_reports_persistent_oom(tmp_path, monkeypatch):
    net = lu._LatentUpscalerNet(2)
    import safetensors.torch as sf
    path = tmp_path / "fake-x2.0.safetensors"
    sf.save_file({k: v.clone() for k, v in net.state_dict().items()}, str(path))
    monkeypatch.setattr(lu, "_resolve_path", lambda name: str(path))
    # Whole latent fits the budget, but tiles stay small once the retry forces them.
    monkeypatch.setattr(lu, "_token_limits", lambda dev, attn: (10 ** 9, 64 * 64))
    monkeypatch.setattr(lu, "_MIN_TILE", 32)

    calls = []

    def boom(x):
        calls.append(tuple(x.shape[-2:]))
        raise RuntimeError("DefaultCPUAllocator: not enough memory")

    monkeypatch.setattr(lu._LatentUpscalerNet, "forward", lambda self, x: boom(x))
    latent = {"samples": torch.zeros((1, 4, 48, 48))}
    try:
        lu.latent_upscale_with_model(latent, "fake-x2.0.safetensors", 2.0)
        assert False, "expected RuntimeError after all retries"
    except RuntimeError as e:
        assert "ran out of memory even tiled" in str(e)
    # First the whole latent (best quality), then progressively smaller tiles. _tiled_forward
    # also probes the net on a 1-wide/1-tall strip to learn its true output size; those probe
    # calls (one dimension == 1) are incidental and excluded from the tile-size assertions.
    real_calls = [c for c in calls if min(c) > 1]
    assert real_calls[0] == (48, 48)
    assert len(real_calls) >= 2 and real_calls[-1] != (48, 48)
    assert max(real_calls[-1]) <= 32

