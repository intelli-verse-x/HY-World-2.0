"""
Real-ESRGAN super-resolution stage (structure-preserving, coherence-safe).

The tiled 20B-edit refine proved not pixel-aligned (ghosting/hallucination), losing the
A/B coherence parameter. Real-ESRGAN is a dedicated SR CNN: it adds genuine high-frequency
detail while preserving the exact composition (no hallucinated content, no ghosting), which
is what the beats-current bar needs.

HONEST PROVENANCE: this upscales the native 1952x960 HY-Pano base by the model's factor
(x4) then resamples to the target (default 3840x1920 = ~2x). Output is recorded as
"native 1952x960 + Real-ESRGAN x4->resample", NOT as native/4K.

Loads via `spandrel` (robust .pth architecture detection). Weight cached on the mounted
volume so subsequent worlds/pods skip the download. 360deg seam handled by wrap-padding.
Importable `esrgan_super_resolve()` + CLI. Never prints secrets.
"""

import argparse
import functools
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "hyworld2" / "panogen"))

print = functools.partial(print, flush=True)  # noqa: A001

WEIGHT_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.1.0/RealESRGAN_x4plus.pth"
)
# Cache on the mounted network volume (MODEL_ROOT=/models/hf-cache -> /models/esrgan).
_MODEL_ROOT = os.environ.get("MODEL_ROOT", "/models/hf-cache")
_CACHE_DIR = os.environ.get("ESRGAN_CACHE", _MODEL_ROOT.rsplit("/hf-cache", 1)[0] + "/esrgan")
WEIGHT_PATH = os.path.join(_CACHE_DIR, "RealESRGAN_x4plus.pth")


def _ensure_deps() -> None:
    try:
        import spandrel  # noqa: F401
    except ImportError:
        print("[esrgan] installing spandrel")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", "spandrel"]
        )


def _ensure_weight() -> str:
    if not os.path.exists(WEIGHT_PATH):
        os.makedirs(_CACHE_DIR, exist_ok=True)
        print(f"[esrgan] downloading RealESRGAN_x4plus.pth -> {WEIGHT_PATH}")
        tmp = WEIGHT_PATH + ".part"
        urllib.request.urlretrieve(WEIGHT_URL, tmp)
        os.replace(tmp, WEIGHT_PATH)
    else:
        print(f"[esrgan] using cached weight {WEIGHT_PATH}")
    return WEIGHT_PATH


def load_model():
    _ensure_deps()
    from spandrel import ModelLoader
    model = ModelLoader().load_from_file(_ensure_weight())
    net = model.model.eval().to("cuda").half()
    scale = int(getattr(model, "scale", 4))
    print(f"[esrgan] model loaded (x{scale})")
    return net, scale


def _forward_tiled(net, t: torch.Tensor, scale: int, tile: int = 640, ov: int = 32) -> torch.Tensor:
    """Tiled forward with overlap to bound VRAM; blends overlaps to hide tile seams."""
    _, _, h, w = t.shape
    out = torch.zeros((1, 3, h * scale, w * scale), dtype=t.dtype, device=t.device)
    wsum = torch.zeros((1, 1, h * scale, w * scale), dtype=t.dtype, device=t.device)
    ramp = torch.ones(tile * scale, dtype=t.dtype, device=t.device)
    if ov > 0:
        r = torch.linspace(0, 1, ov * scale, dtype=t.dtype, device=t.device)
        ramp[: ov * scale] = r
        ramp[-ov * scale:] = r.flip(0)
    stride = tile - ov
    ys = list(range(0, max(1, h - ov), stride))
    xs = list(range(0, max(1, w - ov), stride))
    for y in ys:
        for x in xs:
            y1, x1 = min(y + tile, h), min(x + tile, w)
            y0, x0 = max(0, y1 - tile), max(0, x1 - tile)
            with torch.no_grad():
                o = net(t[:, :, y0:y1, x0:x1])
            th, tw = o.shape[2], o.shape[3]
            m = (ramp[:th, None] * ramp[None, :tw]).unsqueeze(0).unsqueeze(0)
            out[:, :, y0 * scale:y0 * scale + th, x0 * scale:x0 * scale + tw] += o * m
            wsum[:, :, y0 * scale:y0 * scale + th, x0 * scale:x0 * scale + tw] += m
    return out / wsum.clamp_min(1e-6)


def esrgan_super_resolve(
    base_img: Image.Image,
    target_w: int = 3840,
    target_h: int = 1920,
    *,
    net=None,
    scale: int | None = None,
    wrap: int = 64,
) -> Image.Image:
    """Structure-preserving SR of an equirect pano to (target_w, target_h)."""
    from pipeline_with_qwen_image import circular_blend_edges

    if net is None:
        net, scale = load_model()
    arr = np.asarray(base_img.convert("RGB"))
    h, w, _ = arr.shape
    # Wrap-pad the 360deg seam so the SR net sees continuous context across the edge.
    padded = np.concatenate([arr[:, w - wrap:], arr, arr[:, :wrap]], axis=1)
    t = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0).float().div(255).to("cuda").half()
    try:
        with torch.no_grad():
            out = net(t)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        out = _forward_tiled(net, t, scale)
    out = out.clamp(0, 1).mul(255).round().byte().cpu().permute(0, 2, 3, 1).numpy()[0]
    # Crop the (scaled) wrap padding, then resample to the exact target.
    out = out[:, wrap * scale: out.shape[1] - wrap * scale]
    up = Image.fromarray(out)
    if up.size != (target_w, target_h):
        up = up.resize((target_w, target_h), Image.LANCZOS)
    up = circular_blend_edges(up, 32)
    # circular_blend_edges trims the seam width; restore exact 2:1 target dims.
    if up.size != (target_w, target_h):
        up = up.resize((target_w, target_h), Image.LANCZOS)
    return up


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--width", type=int, default=3840)
    ap.add_argument("--height", type=int, default=1920)
    args = ap.parse_args()
    base = Image.open(args.input).convert("RGB")
    out = esrgan_super_resolve(base, args.width, args.height)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.save(args.output)
    print(f"[esrgan] saved {args.output} ({args.width}x{args.height})")


if __name__ == "__main__":
    main()
