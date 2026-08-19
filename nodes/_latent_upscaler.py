"""Model-based latent upscaler (no third-party sampler packs, fully offline).

Supports two weight formats, auto-selected by file extension:
  * ``.safetensors``  -> City96 / SD-Latent-Upscaler Conv2d net
  * ``.pt`` / ``.pth`` -> ttl-nn ``latent_resizer`` net

Weights are searched first in the project-local ``models/upscale_models`` folder
(then optionally via ComfyUI's ``folder_paths``). No network access.
"""

import os

import torch
import torch.nn as nn

try:
    import safetensors.torch as sf
except Exception:  # pragma: no cover - safetensors is a ComfyUI dependency
    sf = None

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
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
# Public helpers
# ---------------------------------------------------------------------------
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


def latent_upscale_with_model(latent, model_name, factor, iterations=1):
    """Upscale a latent ``dict`` (returns a new dict) by ``factor``.

    Applies a City96/ttl-nn Conv2d net directly to the latent samples - no VAE
    round-trip, no third-party upscaler packs. The net is loaded once and the
    upscale is applied ``iterations`` times (compounding by ``factor`` each pass).
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
        net = _LatentUpscalerNet(factor)
        net.load_state_dict(sf.load_file(path), strict=True)
    elif is_ttl:
        net = _TtlLatentResizer.load_model(path, torch.device("cpu"), torch.float32)
    else:
        raise ValueError("Unsupported latent upscale model extension: %s" % model_name)

    device = _latent_device(latent)
    net = net.to(device=device)
    dtype = next(net.parameters()).dtype
    iters = max(1, int(iterations))
    with torch.no_grad():
        out = latent["samples"].to(device=device, dtype=dtype)
        for _ in range(iters):
            if is_ttl:
                out = net(0.13025 * out, scale=factor) / 0.13025
            else:
                out = net(out)
        out = out.to(device=latent["samples"].device, dtype=latent["samples"].dtype)

    result = dict(latent)
    result["samples"] = out
    return result

