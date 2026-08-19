"""Model-based latent upscaler (no third-party sampler packs, fully offline).

Supports two weight formats, auto-selected by file extension:
  * ``.safetensors``  -> City96 / SD-Latent-Upscaler Conv2d net
  * ``.pt`` / ``.pth`` -> ttl-nn ``latent_resizer`` net

Weights are searched first in the project-local ``models/upscale_models`` folder
(then optionally via ComfyUI's ``folder_paths``). No network access.
"""

import logging
import math
import os
import re

import torch
import torch.nn as nn

try:
    import safetensors.torch as sf
except Exception:  # pragma: no cover - safetensors is a ComfyUI dependency
    sf = None

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger("llm_prompt_studio")

# --- memory / compute limits ------------------------------------------------
# The ttl-nn resizer uses full (global) self-attention, whose score matrix is
# O(tokens^2): a 200x150 latent (1600x1200 px) needs 200*150*200*150*4 = 3.6 GB in
# fp32 - exactly the "DefaultCPUAllocator: not enough memory: you tried to allocate
# 3600000000 bytes" crash. Two guards keep it bounded:
#   * attention is always evaluated in query chunks, so the score matrix never exceeds
#     _ATTN_SCORE_BUDGET no matter how big the latent is, and
#   * really huge latents are additionally processed in overlapping tiles, which bounds
#     the quadratic attention *compute* too.
# Tiling is the second choice: GroupNorm/attention then see only a tile's statistics, so
# the result drifts from the whole-latent one. Anything up to the _UNTILED_TOKENS_*
# budgets therefore runs in one piece.
_ATTN_SCORE_BUDGET = 128 * 1024 * 1024   # max bytes of attention scores held at once
_UNTILED_TOKENS_CPU = 64 * 1024          # 256x256 latent (2048 px); ~400 MB RAM, ~50 s
_UNTILED_TOKENS_GPU = 384 * 384          # 3072 px hires target
_TILE_TOKENS_CPU = 128 * 128             # per-tile output tokens once tiling is needed
_TILE_TOKENS_GPU = 256 * 256
_UNTILED_TOKENS_CONV = 4 * 1024 * 1024   # City96 conv net: no attention, cheap per token
_TILE_TOKENS_CONV = 1024 * 1024
_TILE_OVERLAP = 8                        # latent cells shared by neighbouring tiles
_MIN_TILE = 32                           # never tile below this (seams/quality)


def _attention(q, k, v):
    """Chunked ``scaled_dot_product_attention`` with a bounded score-matrix peak.

    ``q/k/v`` are ``[B, N, C]``. SDPA may pick a memory-efficient kernel, but the CPU
    math fallback materializes the full ``[B, N, N]`` score matrix, so the query dim is
    split into chunks small enough to stay inside ``_ATTN_SCORE_BUDGET``.
    """
    b, n, _ = q.shape
    per_query_row = max(1, b * n * q.element_size())
    chunk = max(1, int(_ATTN_SCORE_BUDGET // per_query_row))
    if chunk >= n:
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)
    out = torch.empty_like(q)
    for i in range(0, n, chunk):
        out[:, i:i + chunk] = torch.nn.functional.scaled_dot_product_attention(
            q[:, i:i + chunk], k, v)
    return out


# ---------------------------------------------------------------------------
# City96 / SD-Latent-Upscaler architecture (tiny Conv2d net, in=out=4 channels)
# ---------------------------------------------------------------------------
class _LatentUpscalerNet(nn.Module):
    def __init__(self, factor=2):
        super().__init__()
        head = [
            nn.Conv2d(4, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=factor, mode="nearest"),
            nn.ReLU(inplace=True),
        ]
        core = []
        for _ in range(16):
            core.append(nn.Conv2d(64, 64, 3, padding=1, bias=True))
            core.append(nn.ReLU(inplace=True))
        tail = [nn.Conv2d(64, 4, 3, padding=1, bias=True)]
        self.sequential = nn.Sequential(*head, *core, *tail)

    def forward(self, x):
        return self.sequential(x)


# ---------------------------------------------------------------------------
# ttl-nn latent resizer architecture
# ---------------------------------------------------------------------------
class _ResBlockEmb(nn.Module):
    def __init__(self, channels=128, emb_channels=32):
        super().__init__()
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, channels, bias=True),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Dropout(0.0),
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
        )
        self.skip_connection = nn.Identity()

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb)
        h = h + emb_out[:, :, None, None]
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class _AttnBlock(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.q = nn.Conv2d(channels, channels, 1, bias=True)
        self.k = nn.Conv2d(channels, channels, 1, bias=True)
        self.v = nn.Conv2d(channels, channels, 1, bias=True)
        self.proj_out = nn.Conv2d(channels, channels, 1, bias=True)

    def forward(self, x):
        b, c, h, w = x.shape
        xn = self.norm(x)
        q = self.q(xn).reshape(b, c, h * w).permute(0, 2, 1)
        k = self.k(xn).reshape(b, c, h * w).permute(0, 2, 1)
        v = self.v(xn).reshape(b, c, h * w).permute(0, 2, 1)
        out = _attention(q, k, v)
        out = out.permute(0, 2, 1).reshape(b, c, h, w)
        out = self.proj_out(out)
        return x + out


class _TtlLatentResizer(nn.Module):
    def __init__(self, n_in, attn_in, n_out, attn_out, channels=128, emb_channels=32):
        super().__init__()
        self.conv_in = nn.Conv2d(4, channels, 3, padding=1, bias=True)
        self.embed = nn.Sequential(
            nn.Linear(1, emb_channels, bias=True),
            nn.SiLU(),
            nn.Linear(emb_channels, emb_channels, bias=True),
        )
        self.in_blocks = nn.ModuleList()
        for i in range(n_in):
            self.in_blocks.append(
                _AttnBlock(channels) if i in attn_in else _ResBlockEmb(channels, emb_channels))
        self.out_blocks = nn.ModuleList()
        for i in range(n_out):
            self.out_blocks.append(
                _AttnBlock(channels) if i in attn_out else _ResBlockEmb(channels, emb_channels))
        self.norm_out = nn.GroupNorm(32, channels)
        self.conv_out = nn.Conv2d(channels, 4, 3, padding=1, bias=True)

    def forward(self, x, scale=2.0):
        emb = self.embed(
            torch.tensor([scale - 1.0], device=x.device, dtype=x.dtype).unsqueeze(0))
        h = self.conv_in(x)
        for blk in self.in_blocks:
            h = blk(h) if isinstance(blk, _AttnBlock) else blk(h, emb)
        _, _, hh, ww = h.shape
        h = torch.nn.functional.interpolate(
            h, size=(int(round(hh * scale)), int(round(ww * scale))), mode="bilinear")
        for blk in self.out_blocks:
            h = blk(h) if isinstance(blk, _AttnBlock) else blk(h, emb)
        h = self.norm_out(h)
        h = torch.nn.functional.silu(h)
        h = self.conv_out(h)
        return h

    @classmethod
    def from_state_dict(cls, sd):
        def block_info(prefix):
            idxs = set()
            attn = set()
            for key in sd:
                if not key.startswith(prefix + "."):
                    continue
                rest = key[len(prefix) + 1:]
                first = rest.split(".")[0]
                if first.isdigit():
                    idxs.add(int(first))
                if ".q.weight" in rest:
                    attn.add(int(first))
            n = max(idxs) + 1 if idxs else 0
            return n, attn

        n_in, attn_in = block_info("in_blocks")
        n_out, attn_out = block_info("out_blocks")
        model = cls(n_in, attn_in, n_out, attn_out)
        model.load_state_dict(sd, strict=True)
        return model

    @classmethod
    def load_model(cls, path, device, dtype=torch.float32):
        sd = torch.load(path, map_location=device, weights_only=True)
        if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]
        model = cls.from_state_dict(sd)
        return model.to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Tiled execution + device selection (keeps big latents inside RAM/VRAM)
# ---------------------------------------------------------------------------
def _tile_positions(size, tile, overlap):
    """Start offsets covering ``size`` with ``tile``-sized, ``overlap``-sharing windows."""
    if size <= tile:
        return [0]
    step = max(1, tile - overlap)
    pos = list(range(0, size - tile + 1, step))
    if pos[-1] != size - tile:
        pos.append(size - tile)
    return pos


def _feather_mask(h, w, overlap, device, dtype):
    """Linear-ramp blend mask so tile borders cross-fade instead of showing seams."""
    m = torch.ones((1, 1, h, w), device=device, dtype=dtype)
    ry = max(1, min(int(overlap), h // 2))
    rx = max(1, min(int(overlap), w // 2))
    ramp_y = torch.linspace(1.0 / (ry + 1), 1.0, ry, device=device, dtype=dtype)
    ramp_x = torch.linspace(1.0 / (rx + 1), 1.0, rx, device=device, dtype=dtype)
    m[:, :, :ry, :] *= ramp_y.view(1, 1, -1, 1)
    m[:, :, h - ry:, :] *= ramp_y.flip(0).view(1, 1, -1, 1)
    m[:, :, :, :rx] *= ramp_x.view(1, 1, 1, -1)
    m[:, :, :, w - rx:] *= ramp_x.flip(0).view(1, 1, 1, -1)
    return m


def _tiled_forward(fn, x, scale, tile, overlap=_TILE_OVERLAP, out_channels=4):
    """Apply ``fn`` (an upscaling net) tile-by-tile and blend the results.

    Peak memory (and quadratic attention cost) then depends on ``tile`` instead of the
    full latent size. Edge tiles are flushed against the output border so no row/column
    is left uncovered when ``scale`` is fractional.
    """
    b, _, h, w = x.shape
    # The output size must come from the upscaling net itself, not from ``round(h*scale)``:
    # a fractional scale (City96 ``x1.5``, or a non-integer ttl factor) can make the net emit
    # a size that round() does not match. With a too-large buffer the last tile is flushed
    # one cell past the net's true edge, leaving an uncovered strip; with a too-small one the
    # tile writes would fall out of bounds. Probe the net on thin strips (far cheaper than a
    # full forward) to read its real output dimensions - our nets size each axis independently,
    # so a 1-wide strip gives the true output height and a 1-tall strip the true output width.
    # If even that tiny forward cannot run (pathological OOM), fall back to round() so we still
    # produce output.
    try:
        out_h = max(1, int(fn(x[:, :, :, :1]).shape[-2]))
        out_w = max(1, int(fn(x[:, :, :1, :]).shape[-1]))
    except Exception:  # pragma: no cover - only when the net cannot process a 1-wide strip
        out_h = max(1, int(round(h * scale)))
        out_w = max(1, int(round(w * scale)))
    acc = torch.zeros((b, out_channels, out_h, out_w), device=x.device, dtype=x.dtype)
    weight = torch.zeros((b, 1, out_h, out_w), device=x.device, dtype=x.dtype)
    out_overlap = max(1, int(round(overlap * scale)))
    masks = {}  # interior tiles share a shape, so build each blend mask only once
    for y in _tile_positions(h, tile, overlap):
        th = min(tile, h - y)
        for xx in _tile_positions(w, tile, overlap):
            tw = min(tile, w - xx)
            piece = fn(x[:, :, y:y + th, xx:xx + tw])
            ph, pw = piece.shape[-2], piece.shape[-1]
            oy = max(0, min(int(round(y * scale)), out_h - ph))
            ox = max(0, min(int(round(xx * scale)), out_w - pw))
            if y + th >= h:                      # last row: flush with the bottom edge
                oy = max(0, out_h - ph)
            if xx + tw >= w:                     # last column: flush with the right edge
                ox = max(0, out_w - pw)
            ph = min(ph, out_h - oy)
            pw = min(pw, out_w - ox)
            piece = piece[:, :, :ph, :pw]
            mask = masks.get((ph, pw))
            if mask is None:
                mask = _feather_mask(ph, pw, out_overlap, x.device, x.dtype)
                masks[(ph, pw)] = mask
            acc[:, :, oy:oy + ph, ox:ox + pw] += piece * mask
            weight[:, :, oy:oy + ph, ox:ox + pw] += mask
    return acc / weight.clamp(min=1e-6)


def _model_management():
    try:
        import comfy.model_management as mm
        return mm
    except Exception:  # pragma: no cover - headless tests have no ComfyUI
        return None


def _is_oom(exc):
    """True for out-of-memory failures (CUDA/XPU/CPU allocator), false for real bugs.

    Type checks come first; the message check is only for the CPU allocator, which has no
    dedicated exception type ("DefaultCPUAllocator: not enough memory" / "can't allocate
    memory"). A user interrupt must never be mistaken for an OOM, otherwise the retry
    ladder would keep working after the stop button.
    """
    mm = _model_management()
    interrupt = getattr(mm, "InterruptProcessingException", None)
    if interrupt is not None and isinstance(exc, interrupt):
        return False
    oom_type = getattr(mm, "OOM_EXCEPTION", None)
    # ComfyUI falls back to ``OOM_EXCEPTION = Exception`` on some backends; that would
    # match every error, so ignore it and rely on the concrete types/messages instead.
    if oom_type not in (None, Exception) and isinstance(exc, oom_type):
        return True
    cuda_oom = getattr(torch.cuda, "OutOfMemoryError", None)
    if cuda_oom is not None and isinstance(exc, cuda_oom):
        return True
    if not isinstance(exc, (RuntimeError, MemoryError)):
        return False
    msg = str(exc).lower()
    return ("out of memory" in msg or "not enough memory" in msg
            or "can't allocate memory" in msg or "cannot allocate memory" in msg)


def _empty_cache():
    mm = _model_management()
    try:
        if mm is not None:
            mm.soft_empty_cache()
        elif torch.cuda.is_available():  # pragma: no cover - no GPU in tests
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover - best effort only
        pass


def _estimate_bytes(out_tokens, has_attention):
    """Rough working-set estimate for ONE upscale pass (weights + activations).

    Deliberately tight: ``_select_device`` feeds it to ``model_management.free_memory``,
    which unloads the diffusion model when the request does not fit - and the hires
    sampling pass needs that model back immediately afterwards. Attention is chunked, so
    only one chunk of scores is ever live.
    """
    channels = 128 if has_attention else 64
    cap = _UNTILED_TOKENS_GPU if has_attention else _UNTILED_TOKENS_CONV
    tokens = min(int(out_tokens), cap)
    weights = 64 * 1024 * 1024
    activations = tokens * channels * 4 * 8
    scores = _ATTN_SCORE_BUDGET if has_attention else 0
    return weights + activations + scores


def _select_device(latent, needed_bytes):
    """Prefer ComfyUI's compute device (freeing VRAM first), else stay on the latent's."""
    mm = _model_management()
    if mm is None:
        return _latent_device(latent)
    try:
        device = mm.get_torch_device()
        if device.type == "cpu":
            return device
        mm.free_memory(needed_bytes, device)
        free = mm.get_free_memory(device)
    except Exception:  # pragma: no cover - unusual model_management builds
        return _latent_device(latent)
    if free < needed_bytes:
        logger.info(
            "latent upscale: only %.0f MB free on %s (need ~%.0f MB), running on CPU",
            free / 1048576.0, device, needed_bytes / 1048576.0)
        return torch.device("cpu")
    return device


def _token_limits(device, has_attention):
    """``(max tokens for one whole-latent pass, per-tile tokens when tiling)`` for a device.

    Token counts are output-side (post-upscale) latent cells.
    """
    if not has_attention:
        return _UNTILED_TOKENS_CONV, _TILE_TOKENS_CONV
    if device.type == "cpu":
        return _UNTILED_TOKENS_CPU, _TILE_TOKENS_CPU
    mm = _model_management()
    try:
        free = mm.get_free_memory(device) if mm is not None else 0
    except Exception:  # pragma: no cover - best effort only
        free = 0
    if not free:
        return _UNTILED_TOKENS_GPU, _TILE_TOKENS_GPU
    # ~4 KB of activations per output token, plus room for the attention chunks.
    fits = int(max(0, free - 2 * _ATTN_SCORE_BUDGET) // (128 * 4 * 8))
    return (min(_UNTILED_TOKENS_GPU, max(_UNTILED_TOKENS_CPU, fits)),
            min(_TILE_TOKENS_GPU, max(_TILE_TOKENS_CPU, fits)))


def _tile_edge(tile_tokens, scale):
    """Input-side tile edge (latent cells) whose upscaled tile stays under the budget."""
    tile = int(math.sqrt(max(1.0, float(tile_tokens))) / max(1.0, float(scale)))
    return max(_MIN_TILE, (tile // 8) * 8)


def _plan_tile(device, has_attention, scale, out_tokens, force_tiles=False):
    """Tile edge for one pass, or ``0`` to run the whole latent at once (best quality)."""
    untiled, tile_tokens = _token_limits(device, has_attention)
    if not force_tiles and out_tokens <= untiled:
        return 0
    return _tile_edge(tile_tokens, scale)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------
# City96 / SD-Latent-Upscaler .safetensors checkpoints bake the upscale factor into
# the architecture (the `Upsample(scale_factor=...)` layer), so the *real* scale is the
# one the checkpoint was trained for, not whatever the caller passes. The scale is encoded
# in the filename, e.g. "latent-upscaler-v2.1_SDxl-x1.5.safetensors" -> 1.5. If we rebuild
# the net with a different factor the convs receive the wrong feature flow and the output is
# garbage ("каша"). Parse it so the net and the target size use the model's native scale.
_SCALE_RE = re.compile(r"[xX](\d+(?:\.\d+)?)")


def model_native_scale(model_name, default=2.0):
    """Best-effort native upscale scale of a latent-upscale model from its filename.

    Returns the parsed ``xN.N`` scale (>= 1.0) or ``default`` when not detectable.
    """
    m = _SCALE_RE.search(str(model_name))
    if m:
        try:
            v = float(m.group(1))
            if v >= 1.0:
                return v
        except ValueError:
            pass
    return default


def project_local_upscale_models():
    d = os.path.join(PROJECT_ROOT, "models", "upscale_models")
    if not os.path.isdir(d):
        return []
    exts = (".safetensors", ".pt", ".pth")
    return sorted(f for f in os.listdir(d) if f.lower().endswith(exts))


def project_local_ultralytics_bbox():
    d = os.path.join(PROJECT_ROOT, "models", "ultralytics", "bbox")
    if not os.path.isdir(d):
        return []
    exts = (".pt", ".pth")
    return sorted(f for f in os.listdir(d) if f.lower().endswith(exts))


def _resolve_path(model_name):
    local = os.path.join(PROJECT_ROOT, "models", "upscale_models", model_name)
    if os.path.isfile(local):
        return local
    try:
        import folder_paths
        p = folder_paths.get_full_path("upscale_models", model_name)
        if p and os.path.isfile(p):
            return p
    except Exception:
        pass
    return None


def _latent_device(latent):
    t = latent.get("samples")
    return t.device if hasattr(t, "device") else torch.device("cpu")


def latent_upscale_with_model(latent, model_name, factor, iterations=1, tile=0):
    """Upscale a latent ``dict`` (returns a new dict) by ``factor``.

    Applies a City96/ttl-nn net directly to the latent samples - no VAE round-trip, no
    third-party upscaler packs. The net is loaded once and the upscale is applied
    ``iterations`` times (compounding by the net's scale each pass).

    The pass runs on ComfyUI's compute device when there is room. Attention is always
    chunked, and latents beyond the per-device token budget are additionally processed in
    overlapping tiles, so a big hires target no longer blows up with "not enough memory"
    in the resizer's attention. ``tile`` = 0 picks that automatically; a positive value
    forces that tile edge (in latent cells).
    """
    path = _resolve_path(model_name)
    if not path:
        raise FileNotFoundError(
            "Latent upscale model not found: %s (looked in project "
            "models/upscale_models and ComfyUI upscale_models)" % model_name)

    lowered = model_name.lower()
    is_ttl = lowered.endswith(".pt") or lowered.endswith(".pth")
    if lowered.endswith(".safetensors"):
        if sf is None:
            raise RuntimeError(
                "safetensors is required to load .safetensors latent upscalers")
        # The checkpoint's native scale is fixed by its architecture; use it (falling back
        # to the requested factor) so the Upsample layer matches the trained conv weights.
        scale = model_native_scale(model_name, factor)
        net = _LatentUpscalerNet(scale)
        net.load_state_dict(sf.load_file(path), strict=True)
    elif is_ttl:
        # The ttl resizer takes the scale as an input, so the requested factor applies.
        scale = float(factor)
        net = _TtlLatentResizer.load_model(path, torch.device("cpu"), torch.float32)
    else:
        raise ValueError("Unsupported latent upscale model extension: %s" % model_name)

    samples = latent["samples"]
    iters = max(1, int(iterations))
    src_h, src_w = samples.shape[2], samples.shape[3]
    # Only the first pass is used for the VRAM request: asking for the fully compounded
    # size would free (i.e. unload) far more than one pass needs.
    first_pass_tokens = int(src_h * src_w * scale * scale)
    device = _select_device(latent, _estimate_bytes(first_pass_tokens, is_ttl))

    def _apply(net_on_device, x):
        if is_ttl:
            return net_on_device(0.13025 * x, scale=scale) / 0.13025
        return net_on_device(x)

    def _run(dev, forced_tile=0, force_tiles=False):
        net_on_device = net.to(device=dev)
        dtype = next(net_on_device.parameters()).dtype
        with torch.no_grad():
            out = samples.to(device=dev, dtype=dtype)
            for _ in range(iters):
                h, w = out.shape[2], out.shape[3]
                # Decided per pass: iterations compound, so a later pass may need tiles
                # even when the first one fitted whole.
                pass_tile = forced_tile or _plan_tile(
                    dev, is_ttl, scale, int(h * w * scale * scale), force_tiles)
                if pass_tile and (h > pass_tile or w > pass_tile):
                    logger.info("latent upscale: %dx%d latent on %s in %d-cell tiles",
                                w, h, dev, pass_tile)
                    out = _tiled_forward(lambda t: _apply(net_on_device, t), out, scale,
                                         pass_tile, out_channels=out.shape[1])
                else:
                    out = _apply(net_on_device, out)
            return out.to(device=samples.device, dtype=samples.dtype)

    # Whole-latent first (attention is chunked, so memory stays bounded); fall back to
    # progressively smaller tiles and finally to CPU only if that runs out of memory.
    if int(tile) > 0:
        attempts = [(device, int(tile), False)]
    else:
        attempts = [(device, 0, False), (device, 0, True)]
        _, tile_tokens = _token_limits(device, is_ttl)
        edge = _tile_edge(tile_tokens, scale)
        if edge > _MIN_TILE:
            attempts.append((device, max(_MIN_TILE, edge // 2), True))
    if device.type != "cpu":
        cpu = torch.device("cpu")
        attempts.append((cpu, 0, True))

    last_error = None
    for dev, forced_tile, force_tiles in attempts:
        try:
            out = _run(dev, forced_tile, force_tiles)
            break
        except Exception as exc:  # noqa: BLE001 - re-raised unless it is an OOM
            if not _is_oom(exc):
                raise
            last_error = exc
            logger.warning(
                "latent upscale ran out of memory on %s (tile=%s); retrying smaller",
                dev, forced_tile or ("auto tiles" if force_tiles else "whole latent"))
            _empty_cache()
    else:
        raise RuntimeError(
            "Latent upscale model '%s' ran out of memory even tiled. Lower the hires "
            "upscale factor/iterations, or switch hires_upscale_type to plain 'latent'."
            % model_name) from last_error

    result = dict(latent)
    result["samples"] = out
    return result

