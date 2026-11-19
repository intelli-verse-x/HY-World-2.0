"""
Tiled equirectangular super-resolution (Stage 0b-SR).

The probe proved that generating a *wide* equirect canvas above the HY-Pano LoRA's
~1MP training resolution makes the DiT duplicate its composition (mirrored motifs).
This module gets genuine higher-res WITHOUT duplication by refining the coherent
1920x960 base panorama in overlapping, normal-aspect TILES:

  base pano (coherent, 1952x960)
    -> Lanczos upscale to target (e.g. 3840x1920)   [structure preserved]
    -> for each overlapping tile (a normal-aspect crop, in-distribution):
         run Qwen-Image-Edit-2509 (NO pano LoRA) as an image-conditioned refiner
         to add genuine high-frequency detail/texture the upscale can't invent
    -> feather-blend tiles back with horizontal 360deg wrap
    -> circular edge blend

Because every tile is a normal-aspect, in-distribution image (not a wide pano),
the base edit model refines faithfully and never tiles/duplicates the composition.
Cross-tile consistency comes from (a) conditioning each tile on the same upscaled
base and (b) overlap feathering + wrap.

CLI and importable `tiled_super_resolve()`. Never prints secrets.
"""

import argparse
import functools
import signal
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

print = functools.partial(print, flush=True)  # noqa: A001  stream through tee even if killed


class _TileTimeout(Exception):
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "hyworld2" / "panogen"))

_REFINE_POSITIVE = (
    "ultra sharp, high resolution, crisp fine detail, clean textures, "
    "photographic clarity, preserve the exact composition and content"
)
_REFINE_NEGATIVE = (
    "blurry, soft, low resolution, oversmoothed, jpeg artifacts, duplicated objects, "
    "extra buildings, warped geometry, distorted text, watermark"
)


def _feather_mask(tw: int, th: int, ox: int, oy: int) -> np.ndarray:
    """Linear ramp on the overlap borders; flat 1.0 in the tile interior."""
    mx = np.ones(tw, dtype=np.float32)
    my = np.ones(th, dtype=np.float32)
    if ox > 0:
        ramp = np.linspace(0.0, 1.0, ox, dtype=np.float32)
        mx[:ox] = ramp
        mx[-ox:] = ramp[::-1]
    if oy > 0:
        ramp = np.linspace(0.0, 1.0, oy, dtype=np.float32)
        my[:oy] = ramp
        my[-oy:] = ramp[::-1]
    return np.clip(np.outer(my, mx), 1e-3, 1.0)


def _load_refiner():
    from qwen_image import PanoDiffusionPipeline

    print("[tiledsr] loading Qwen-Image-Edit-2509 base refiner (no pano LoRA)")
    pipe = PanoDiffusionPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.bfloat16
    )
    # On a 140GB H200 the ~40GB bf16 transformer + Qwen2.5-VL encoder (~60GB total)
    # fit fully on-device with headroom, so load resident on GPU: ~10-15s/tile and no
    # CPU-offload host-RAM accretion (which OOM-killed multi-tile runs on 80GB pods).
    # Fall back to sequential offload if the card is too small.
    try:
        pipe.to("cuda")
        print("[tiledsr] refiner resident on GPU (no offload)")
    except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:  # noqa
        print(f"[tiledsr] full-GPU load failed ({str(exc)[:120]}); using CPU offload")
        torch.cuda.empty_cache()
        pipe.enable_model_cpu_offload()
    return pipe


def tiled_super_resolve(
    base_img: Image.Image,
    target_w: int,
    target_h: int,
    prompt: str,
    *,
    pipe=None,
    tile: int = 1280,
    overlap: int = 320,
    steps: int = 14,
    seed: int = 4201,
    refine_blend: float = 0.6,
    pole_protect: float = 0.16,
) -> Image.Image:
    """Refine `base_img` up to (target_w, target_h) via wrap-aware tiled img2img.

    The Qwen edit model has no denoise-strength knob and over-edits, so refined tiles are
    blended over the coherent Lanczos base (`refine_blend`) to add sharpness without
    hallucinating new content, and the equirect poles (top/bottom `pole_protect` band) are
    left as the base since extreme spherical distortion there confuses the edit model.
    """
    if pipe is None:
        pipe = _load_refiner()

    from pipeline_with_qwen_image import circular_blend_edges

    up = base_img.convert("RGB").resize((target_w, target_h), Image.LANCZOS)
    src = np.asarray(up).astype(np.float32)
    out_canvas = src.copy()  # coherent base everywhere; refined tiles composite over it
    acc = np.zeros_like(src)
    wacc = np.zeros((target_h, target_w, 1), dtype=np.float32)

    stride = tile - overlap
    xs = list(range(0, target_w, stride))
    # Restrict vertical tiling to the horizon band (protect the distorted poles).
    y_lo = int(target_h * pole_protect)
    y_hi = int(target_h * (1.0 - pole_protect)) - tile
    ys = list(range(y_lo, max(y_lo, y_hi) + 1, stride))
    if not ys:
        ys = [max(0, (target_h - tile) // 2)]
    if ys[-1] < y_hi:
        ys.append(max(y_lo, y_hi))
    ys = [max(0, min(y, target_h - tile)) for y in ys]
    ys = sorted(set(ys))

    positive = (prompt.strip() + ", " + _REFINE_POSITIVE).strip(", ")
    total = len(ys) * len(xs)
    print(f"[tiledsr] target {target_w}x{target_h}, {total} tiles ({len(xs)}x{len(ys)}), "
          f"tile={tile} overlap={overlap} steps={steps}")

    def _on_alarm(signum, frame):
        raise _TileTimeout()
    signal.signal(signal.SIGALRM, _on_alarm)

    n = 0
    for yi in ys:
        for x0 in xs:
            # Horizontal wrap: roll the source so the tile is contiguous, crop, refine,
            # then unroll on paste. This makes the 360deg seam tile like any interior tile.
            rolled = np.roll(src, -x0, axis=1)
            tile_np = rolled[yi:yi + tile, 0:tile]
            tile_img = Image.fromarray(tile_np.astype(np.uint8))
            import time as _t
            t0 = _t.time()
            signal.alarm(420)  # a single tile must never hang past the pod deadline
            out = pipe(
                image=tile_img,
                prompt=positive,
                negative_prompt=_REFINE_NEGATIVE,
                generator=torch.Generator(device="cpu").manual_seed(seed + n),
                true_cfg_scale=4.0,
                num_inference_steps=steps,
                guidance_scale=1.0,
                num_images_per_prompt=1,
                height=tile,
                width=tile,
            ).images[0]
            signal.alarm(0)
            ref_arr = np.asarray(out.resize((tile, tile), Image.LANCZOS)).astype(np.float32)
            base_tile = tile_np.astype(np.float32).copy()
            del rolled, tile_np, tile_img, out
            import gc
            import resource
            gc.collect()
            torch.cuda.empty_cache()
            rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
            print(f"[tiledsr] tile {n+1}/{total} @ x={x0} y={yi} in {_t.time()-t0:.0f}s "
                  f"(peakRSS {rss_gb:.1f}GB)")
            # Blend refined detail over the coherent base to add sharpness without the
            # edit model's hallucinated content.
            blended_tile = refine_blend * ref_arr + (1.0 - refine_blend) * base_tile
            mask = _feather_mask(tile, tile, overlap, overlap)[:, :, None]

            contrib = np.zeros_like(src)
            wc = np.zeros((target_h, target_w, 1), dtype=np.float32)
            contrib[yi:yi + tile, 0:tile] = blended_tile * mask
            wc[yi:yi + tile, 0:tile] = mask
            acc += np.roll(contrib, x0, axis=1)
            wacc += np.roll(wc, x0, axis=1)
            n += 1

    # Composite the refined horizon band over the coherent base; poles stay base.
    covered = wacc[:, :, 0] > 1e-3
    refined = acc / np.clip(wacc, 1e-3, None)
    out_canvas[covered] = refined[covered]
    result = Image.fromarray(np.clip(out_canvas, 0, 255).astype(np.uint8))
    return circular_blend_edges(result, 32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--width", type=int, default=3840)
    ap.add_argument("--height", type=int, default=1920)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--tile", type=int, default=1280)
    ap.add_argument("--overlap", type=int, default=320)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--seed", type=int, default=4201)
    args = ap.parse_args()

    base = Image.open(args.input).convert("RGB")
    out = tiled_super_resolve(
        base, args.width, args.height, args.prompt,
        tile=args.tile, overlap=args.overlap, steps=args.steps, seed=args.seed,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.save(args.output)
    print(f"[tiledsr] saved {args.output} ({args.width}x{args.height})")


if __name__ == "__main__":
    main()
