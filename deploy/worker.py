"""HY-World 2.0 worldgen queue worker.

Consumes JSON jobs {jobId, prompt, ...} from Redis list `pipeline:signal:worldgen`,
runs text -> seed image -> 360 panorama (HY-Pano-2.0 Qwen backend) -> perspective
views -> WorldMirror-2.0 3DGS reconstruction, and uploads splat artifacts to
s3://$S3_BUCKET/worldgen/{jobId}/.

Designed for a single 24GB GPU (L4/A10G) using 4-bit quantization for the
20B Qwen-Image-Edit backbone; QUANT_MODE=bf16 for 48GB+ cards (L40S).
"""

import gc
import io
import json
import math
import os
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- config
REDIS_HOST = os.environ.get("REDIS_HOST", "content-factory-redis.aicart.svc.cluster.local")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD") or None
QUEUE_NAME = os.environ.get("QUEUE_NAME", "pipeline:signal:worldgen")
PROCESSING_LIST = os.environ.get("PROCESSING_LIST", "pipeline:worldgen:processing")
S3_BUCKET = os.environ.get("S3_BUCKET", "intelliverse-world-templates")
S3_PREFIX = os.environ.get("S3_PREFIX", "worldgen")
WORK_DIR = Path(os.environ.get("WORK_DIR", "/tmp/worldgen"))
DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK", "")
QUANT_MODE = os.environ.get("QUANT_MODE", "auto")  # auto | 4bit | mixed | bf16
MAX_ATTEMPTS = int(os.environ.get("MAX_ATTEMPTS", "2"))
BLPOP_TIMEOUT = int(os.environ.get("BLPOP_TIMEOUT", "120"))

PANO_MODEL = os.environ.get("PANO_MODEL", "Qwen/Qwen-Image-Edit-2509")
PANO_LORA = os.environ.get("PANO_LORA", "tencent/HY-World-2.0")
SEED_MODEL = os.environ.get("SEED_MODEL", "stabilityai/sdxl-turbo")
MIRROR_MODEL = os.environ.get("MIRROR_MODEL", "tencent/HY-World-2.0")

PANO_H = int(os.environ.get("PANO_H", "960"))
PANO_W = int(os.environ.get("PANO_W", "1952"))
PANO_STEPS = int(os.environ.get("PANO_STEPS", "40"))          # upstream default 40
PANO_TRUE_CFG = float(os.environ.get("PANO_TRUE_CFG", "7.5"))  # upstream default 7.5
VIEW_SIZE = int(os.environ.get("VIEW_SIZE", "960"))
VIEW_FOV_DEG = float(os.environ.get("VIEW_FOV_DEG", "70"))
RECON_TARGET_SIZE = int(os.environ.get("RECON_TARGET_SIZE", "952"))  # upstream default 952
GS_MAX_POINTS = int(os.environ.get("GS_MAX_POINTS", "2000000"))
# HD tier: keep up to this many gaussians in world-hd.splat (desktop);
# world.splat stays capped at GS_MAX_POINTS for mobile. 0 disables HD export.
GS_MAX_POINTS_HD = int(os.environ.get("GS_MAX_POINTS_HD", "0"))
# Real-ESRGAN x4plus upscale of the panorama before reconstruction
# (1 = off, 2/4 = target factor). Sharper input -> sharper gaussians.
PANO_UPSCALE = int(os.environ.get("PANO_UPSCALE", "1"))
ESRGAN_WEIGHTS_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
)
# WorldMirror's raw gaussian scales are tuned for gsplat's rasterizer and
# render as sub-pixel specks in .splat viewers (salt-and-pepper "mush").
# Inflating them ~1.8x closes the gaps into continuous surfaces.
SPLAT_SCALE_MULT = float(os.environ.get("SPLAT_SCALE_MULT", "1.8"))

# Per-job overrides: job payload may carry {"params": {<key>: <value>}} for
# any of these keys, so quality experiments don't need a redeploy per run.
_PARAM_DEFAULTS = {
    "pano_steps": PANO_STEPS,
    "pano_true_cfg": PANO_TRUE_CFG,
    "pano_upscale": PANO_UPSCALE,
    "view_size": VIEW_SIZE,
    "recon_target_size": RECON_TARGET_SIZE,
    "gs_max_points": GS_MAX_POINTS,
    "gs_max_points_hd": GS_MAX_POINTS_HD,
    "splat_scale_mult": SPLAT_SCALE_MULT,
    # WorldMirror masks: sky segmentation and depth/normal edge filtering can
    # produce false-positive holes on indoor scenes (dark glass, complex
    # silhouettes) — toggleable per experiment.
    "sky_mask": 1,
    "edge_mask": 1,
    "dense_floor": 0,
    # Two-pass reconstruction: high-res detail pass + coarse coverage pass
    # (dense floor rig); coverage splats fill voxels the detail pass left
    # empty (occlusion/shadow holes that show as black voids).
    "two_pass": 0,
    "coverage_view_size": 960,
    "coverage_recon_size": 952,
    # Coverage fill below the horizon only (floor/shadow holes) — full-sphere
    # fill trades sharpness for milky mid-distance haze (verifier HD round 3).
    "fill_floor_only": 1,
    # Panorama-shell backfill: one backdrop splat per 1-degree angular bin at
    # r97 of that bin, so no ray from spawn escapes to empty black.
    "shell_fill": 1,
    "shell_bins": 360,
    # Fringing/noise culls applied at export: drop near-transparent gaussians
    # (neon bloom past sign borders) and clamp runaway scales.
    "opacity_floor": 0.0,
    "scale_clamp": 0.0,
}


def job_params(job: dict) -> dict:
    cfg = dict(_PARAM_DEFAULTS)
    overrides = job.get("params") or {}
    for k, v in overrides.items():
        if k in cfg:
            cfg[k] = type(cfg[k])(v)
    return cfg

VIEWER_BASE = "https://worlds.quizverse.world"
S3_HTTP_BASE = f"https://{S3_BUCKET}.s3.us-east-1.amazonaws.com"

_shutdown = {"flag": False, "current_job_raw": None}


def log(msg):
    print(f"[worker {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def discord(msg):
    if not DISCORD_WEBHOOK:
        return
    try:
        import requests
        requests.post(DISCORD_WEBHOOK, json={"content": msg[:1900]}, timeout=10)
    except Exception as e:
        log(f"discord post failed: {e}")


# ---------------------------------------------------------------- stages
def free_gpu():
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def gen_seed_image(prompt: str, out_path: Path):
    """Stage A: text -> seed perspective image with SDXL-Turbo."""
    import torch
    from diffusers import AutoPipelineForText2Image

    t0 = time.time()
    pipe = AutoPipelineForText2Image.from_pretrained(
        SEED_MODEL, torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")
    img = pipe(
        prompt=prompt,
        num_inference_steps=6,
        guidance_scale=0.0,
        height=576,
        width=1024,
    ).images[0]
    img.save(out_path)
    del pipe
    free_gpu()
    log(f"seed image done in {time.time()-t0:.0f}s -> {out_path}")
    return out_path


def resolved_quant_mode():
    # "auto": pick the best mode the local GPU can hold. "mixed" (bf16
    # transformer ~40GB + nf4 text encoder ~5GB) needs a 48GB card (L40S);
    # 24GB cards (L4/A10G) fall back to full 4-bit.
    if QUANT_MODE != "auto":
        return QUANT_MODE
    import torch
    vram_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    return "mixed" if vram_gib >= 44 else "4bit"


def gen_panorama(prompt: str, seed_path: Path, out_path: Path,
                 steps: int = None, true_cfg: float = None):
    """Stage B: seed image -> 360 equirectangular panorama (HY-Pano-2.0 Qwen)."""
    import torch

    sys.path.insert(0, "/app/hyworld2/panogen")
    from pipeline_with_qwen_image import HunyuanPanoPipeline
    from qwen_image import PanoDiffusionPipeline

    quant = resolved_quant_mode()
    t0 = time.time()
    kwargs = {"torch_dtype": torch.bfloat16}
    if quant in ("4bit", "mixed"):
        from diffusers import PipelineQuantizationConfig
        # "mixed": only the 7B text encoder is quantized; the 20B transformer
        # (which paints the pixels and dominates output fidelity) stays bf16.
        # 40GB (transformer bf16) + ~5GB (TE nf4) fits a 48GB L40S without
        # CPU offload, so no 90GB+ host RAM requirement.
        components = ["transformer", "text_encoder"] if quant == "4bit" else ["text_encoder"]
        kwargs["quantization_config"] = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.bfloat16,
            },
            components_to_quantize=components,
        )
    pipe = PanoDiffusionPipeline.from_pretrained(PANO_MODEL, **kwargs)
    if quant == "4bit":
        pipe.to("cuda")
    else:
        # mixed/bf16: keep only the active component on the GPU. With
        # everything resident, nf4 TE + bf16 transformer ≈ 45GB and OOMs the
        # L40S's 44.4GiB usable; offload swaps the TE out before denoising.
        pipe.enable_model_cpu_offload()
    try:
        pipe.vae.enable_tiling()
    except Exception:
        pass
    log(f"pano base loaded in {time.time()-t0:.0f}s, loading LoRA...")
    pipe.load_lora_weights(
        PANO_LORA, subfolder="HY-Pano-2.0",
        weight_name="pytorch_lora_weights.safetensors",
    )
    hy = HunyuanPanoPipeline(pipe)
    t1 = time.time()
    pano = hy(
        str(seed_path),
        prompt=prompt,
        seed=42,
        height=PANO_H,
        width=PANO_W,
        num_inference_steps=steps if steps is not None else PANO_STEPS,
        guidance_scale=1.0,
        true_cfg_scale=true_cfg if true_cfg is not None else PANO_TRUE_CFG,
    )
    pano.save(out_path)
    del hy, pipe
    free_gpu()
    log(f"panorama done: load={t1-t0:.0f}s infer={time.time()-t1:.0f}s -> {out_path}")
    return out_path


class _RRDB(object):
    """Minimal Real-ESRGAN x4plus (RRDBNet) inference, no basicsr dependency
    (basicsr is incompatible with torchvision>=0.17)."""

    @staticmethod
    def build(num_feat=64, num_block=23, num_grow_ch=32):
        import torch.nn as nn

        class ResidualDenseBlock(nn.Module):
            def __init__(self, nf, gc):
                super().__init__()
                self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
                self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
                self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
                self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
                self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
                self.lrelu = nn.LeakyReLU(0.2, inplace=True)

            def forward(self, x):
                import torch
                x1 = self.lrelu(self.conv1(x))
                x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
                x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
                x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
                x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
                return x5 * 0.2 + x

        class RRDBBlock(nn.Module):
            def __init__(self, nf, gc):
                super().__init__()
                self.rdb1 = ResidualDenseBlock(nf, gc)
                self.rdb2 = ResidualDenseBlock(nf, gc)
                self.rdb3 = ResidualDenseBlock(nf, gc)

            def forward(self, x):
                return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x

        class RRDBNet(nn.Module):
            def __init__(self):
                super().__init__()
                import torch.nn.functional  # noqa: F401
                self.conv_first = nn.Conv2d(3, num_feat, 3, 1, 1)
                self.body = nn.Sequential(*[RRDBBlock(num_feat, num_grow_ch) for _ in range(num_block)])
                self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
                self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
                self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
                self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
                self.conv_last = nn.Conv2d(num_feat, 3, 3, 1, 1)
                self.lrelu = nn.LeakyReLU(0.2, inplace=True)

            def forward(self, x):
                import torch.nn.functional as F
                feat = self.conv_first(x)
                feat = feat + self.conv_body(self.body(feat))
                feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
                feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
                return self.conv_last(self.lrelu(self.conv_hr(feat)))

        return RRDBNet()


def upscale_panorama(pano_path: Path, factor: int, out_path: Path):
    """Real-ESRGAN x4plus tiled upscale, then resample to `factor`x.

    Runs on the worker's own GPU between the pano and reconstruction stages
    (the fleet realesrgan service targets an L4 nodepool that no longer
    schedules, and its HTTP contract is video-oriented)."""
    import cv2
    import torch

    t0 = time.time()
    wpath = Path(os.environ.get("HF_HOME", "/models/hf-cache")) / "realesrgan_x4plus.pth"
    if not wpath.is_file():
        import urllib.request
        urllib.request.urlretrieve(ESRGAN_WEIGHTS_URL, wpath)
    state = torch.load(str(wpath), map_location="cpu", weights_only=True)
    state = state.get("params_ema", state.get("params", state))
    model = _RRDB.build()
    model.load_state_dict(state, strict=True)
    model = model.half().to("cuda").eval()

    img = cv2.imread(str(pano_path), cv2.IMREAD_COLOR)  # BGR
    h, w = img.shape[:2]
    x = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1).half().div(255.0)

    tile, pad = 512, 16
    out = torch.zeros(3, h * 4, w * 4, dtype=torch.float32)
    with torch.no_grad():
        for ty in range(0, h, tile):
            for tx in range(0, w, tile):
                y0, y1 = max(ty - pad, 0), min(ty + tile + pad, h)
                x0, x1 = max(tx - pad, 0), min(tx + tile + pad, w)
                patch = x[:, y0:y1, x0:x1].unsqueeze(0).to("cuda")
                up = model(patch)[0].float().cpu()
                iy0, ix0 = (ty - y0) * 4, (tx - x0) * 4
                cy = min(tile, h - ty) * 4
                cx = min(tile, w - tx) * 4
                out[:, ty * 4:ty * 4 + cy, tx * 4:tx * 4 + cx] = \
                    up[:, iy0:iy0 + cy, ix0:ix0 + cx]
    del model
    free_gpu()

    res = (out.clamp(0, 1) * 255.0).byte().permute(1, 2, 0).numpy()[:, :, ::-1]
    if factor != 4:
        res = cv2.resize(res, (w * factor, h * factor), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_path), res)
    log(f"pano upscaled x{factor} ({w}x{h} -> {w*factor}x{h*factor}) in {time.time()-t0:.0f}s")
    return out_path


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def render_views(pano_path: Path, views_dir: Path, view_size: int = None,
                 dense_floor: bool = False):
    """Stage C: equirect panorama -> pinhole views + camera prior JSON.

    OpenCV convention: +x right, +y down, +z forward. World frame = camera 0.
    """
    import cv2

    views_dir.mkdir(parents=True, exist_ok=True)
    pano = cv2.imread(str(pano_path), cv2.IMREAD_COLOR)
    ph, pw = pano.shape[:2]

    size = view_size if view_size is not None else VIEW_SIZE
    f = 0.5 * size / math.tan(math.radians(VIEW_FOV_DEG) / 2)
    cx = cy = size / 2.0
    K = [[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]]

    rig = []  # (yaw_deg, pitch_deg)
    rig += [(y, 0.0) for y in range(0, 360, 45)]          # 8 equator views
    rig += [(y, 35.0) for y in range(0, 360, 90)]         # 4 up
    rig += [(y + 45, -35.0) for y in range(0, 360, 90)]   # 4 down (offset yaw)
    if dense_floor:
        # Floor coverage: 4 down views leave grazing-angle gaps that render as
        # black voids near the camera. Add a steep ring + nadir.
        rig += [(y, -65.0) for y in range(0, 360, 120)]   # 3 steep down
        rig += [(0.0, -88.0)]                             # nadir

    # Pixel-direction grid in camera frame
    u, v = np.meshgrid(np.arange(size, dtype=np.float64) + 0.5,
                       np.arange(size, dtype=np.float64) + 0.5)
    d_cam = np.stack([(u - cx) / f, (v - cy) / f, np.ones_like(u)], axis=-1)
    d_cam /= np.linalg.norm(d_cam, axis=-1, keepdims=True)

    cams = {"num_cameras": len(rig), "extrinsics": [], "intrinsics": []}
    for i, (yaw, pitch) in enumerate(rig):
        R = _rot_y(math.radians(yaw)) @ _rot_x(math.radians(-pitch))
        d = d_cam @ R.T  # world directions
        lon = np.arctan2(d[..., 0], d[..., 2])
        elev = np.arcsin(np.clip(-d[..., 1], -1, 1))
        map_x = ((lon / (2 * math.pi) + 0.5) * pw).astype(np.float32)
        map_y = ((0.5 - elev / math.pi) * ph).astype(np.float32)
        view = cv2.remap(pano, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_WRAP)
        name = f"image_{i:04d}"
        cv2.imwrite(str(views_dir / f"{name}.png"), view)
        c2w = np.eye(4)
        c2w[:3, :3] = R
        cams["extrinsics"].append({"camera_id": name, "matrix": c2w.tolist()})
        cams["intrinsics"].append({"camera_id": name, "matrix": K})

    cam_path = views_dir.parent / "camera_prior.json"
    with open(cam_path, "w") as fjson:
        json.dump(cams, fjson)
    log(f"rendered {len(rig)} views -> {views_dir}")
    return views_dir, cam_path


_mirror_pipe = None


def run_worldmirror(views_dir: Path, cam_path: Path, out_dir: Path,
                    target_size: int = None, gs_cap: int = None,
                    sky_mask: bool = True, edge_mask: bool = True):
    """Stage D: multi-view images -> 3DGS via WorldMirror-2.0 (kept resident)."""
    global _mirror_pipe
    from hyworld2.worldrecon.pipeline import WorldMirrorPipeline

    if _mirror_pipe is None:
        _mirror_pipe = WorldMirrorPipeline.from_pretrained(
            MIRROR_MODEL, subfolder="HY-WorldMirror-2.0", enable_bf16=True,
        )
    t0 = time.time()
    result = _mirror_pipe(
        str(views_dir),
        strict_output_path=str(out_dir),
        target_size=target_size if target_size is not None else RECON_TARGET_SIZE,
        save_depth=False,
        save_normal=False,
        save_points=False,
        save_camera=True,
        save_gs=True,
        apply_sky_mask=sky_mask,
        apply_edge_mask=edge_mask,
        compress_gs_max_points=gs_cap if gs_cap is not None else GS_MAX_POINTS,
        prior_cam_path=str(cam_path),
        log_time=True,
    )
    if result is None:
        raise RuntimeError("WorldMirror pipeline returned None (skipped input)")
    ply = out_dir / "gaussians.ply"
    if not ply.is_file() or ply.stat().st_size < 1_000_000:
        raise RuntimeError(f"gaussians.ply missing or too small: {ply}")
    log(f"worldmirror done in {time.time()-t0:.0f}s -> {ply} ({ply.stat().st_size/1e6:.0f}MB)")
    return ply


def merge_fill_ply(detail_ply: Path, coverage_ply: Path, out_ply: Path,
                   voxel: float = 0.02, floor_only: bool = False):
    """Union of detail splats + coverage splats that land in voxels the
    detail pass left empty. Fills recon occlusion holes without doubling
    density where the detail pass already has geometry. floor_only keeps
    only below-horizon coverage splats (Y-down space): full-sphere fill
    hazes mid-distance detail the detail pass rendered sharply."""
    from plyfile import PlyData, PlyElement

    d = PlyData.read(str(detail_ply))["vertex"].data
    c = PlyData.read(str(coverage_ply))["vertex"].data
    if floor_only:
        c = c[c["y"] > -0.1]

    def keys(v):
        p = np.stack([v["x"], v["y"], v["z"]], axis=1)
        return np.floor(p / voxel).astype(np.int64)

    dk = keys(d)
    ck = keys(c)
    lo = np.minimum(dk.min(0), ck.min(0))
    dk -= lo
    ck -= lo
    dims = np.maximum(dk.max(0), ck.max(0)) + 1
    flat_d = (dk[:, 0] * dims[1] + dk[:, 1]) * dims[2] + dk[:, 2]
    flat_c = (ck[:, 0] * dims[1] + ck[:, 1]) * dims[2] + ck[:, 2]
    fill = ~np.isin(flat_c, np.unique(flat_d))
    merged = np.concatenate([d, c[fill]])
    PlyData([PlyElement.describe(merged, "vertex")]).write(str(out_ply))
    log(f"two-pass merge: detail={len(d)} + fill={int(fill.sum())}/{len(c)} -> {len(merged)}")
    return out_ply


_SPLAT_REC = np.dtype([("p", "<f4", 3), ("s", "<f4", 3),
                       ("c", "u1", 4), ("r", "u1", 4)])


def shell_records(rec, bins_x: int = 360, rpad: float = 1.03, scale_k: float = 1.6):
    """Panorama-shell backfill: one backdrop splat per angular bin (equirect,
    ~1 degree) at r97 of the bin, colored like the bin's far band; empty bins
    dilated from neighbors. Guarantees no ray from spawn hits empty black."""
    p = rec["p"].astype(np.float64)
    r = np.linalg.norm(p, axis=1)
    ok = r > 1e-6
    p, r, c = p[ok], r[ok], rec["c"][ok]

    bx, by = bins_x, bins_x // 2
    theta = np.arctan2(p[:, 0], p[:, 2])
    phi = np.arcsin(np.clip(p[:, 1] / r, -1, 1))
    ix = np.clip(((theta + np.pi) / (2 * np.pi) * bx).astype(int), 0, bx - 1)
    iy = np.clip(((phi + np.pi / 2) / np.pi * by).astype(int), 0, by - 1)
    flat = iy * bx + ix

    nb = bx * by
    r_out = np.full(nb, np.nan)
    col = np.zeros((nb, 3))
    order = np.argsort(flat, kind="stable")
    flat_s, r_s = flat[order], r[order]
    c_s = c[order][:, :3].astype(np.float64)
    bounds = np.searchsorted(flat_s, np.arange(nb + 1))
    for b in range(nb):
        lo, hi = bounds[b], bounds[b + 1]
        if hi - lo < 3:
            continue
        rb = r_s[lo:hi]
        r97 = np.percentile(rb, 97)
        far = rb >= r97 * 0.8
        r_out[b] = r97
        col[b] = np.median(c_s[lo:hi][far], axis=0)

    grid_r = r_out.reshape(by, bx)
    grid_c = col.reshape(by, bx, 3)
    for _ in range(max(bx, by)):
        empty = np.isnan(grid_r)
        if not empty.any():
            break
        acc_r = np.zeros((by, bx))
        acc_c = np.zeros((by, bx, 3))
        cnt = np.zeros((by, bx))
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            sr = np.roll(grid_r, dx, axis=1) if dy == 0 else np.full_like(grid_r, np.nan)
            sc = np.roll(grid_c, dx, axis=1) if dy == 0 else np.zeros_like(grid_c)
            if dy != 0:
                sr[max(dy, 0):by + min(dy, 0)] = grid_r[max(-dy, 0):by + min(-dy, 0)]
                sc[max(dy, 0):by + min(dy, 0)] = grid_c[max(-dy, 0):by + min(-dy, 0)]
            good = ~np.isnan(sr)
            acc_r[good] += sr[good]
            acc_c[good] += sc[good]
            cnt[good] += 1
        f = empty & (cnt > 0)
        grid_r[f] = acc_r[f] / cnt[f]
        grid_c[f] = acc_c[f] / cnt[f][:, None]

    yy, xx = np.mgrid[0:by, 0:bx]
    th = (xx + 0.5) / bx * 2 * np.pi - np.pi
    ph = (yy + 0.5) / by * np.pi - np.pi / 2
    rr = grid_r * rpad
    dirs = np.stack([np.sin(th) * np.cos(ph), np.sin(ph), np.cos(th) * np.cos(ph)], axis=-1)
    shell = np.zeros(nb, dtype=_SPLAT_REC)
    shell["p"] = (dirs * rr[..., None]).reshape(nb, 3).astype(np.float32)
    shell["s"] = (rr.reshape(nb, 1) * (2 * np.pi / bx) * scale_k).astype(np.float32)
    shell["c"][:, :3] = np.clip(grid_c.reshape(nb, 3), 0, 255).astype(np.uint8)
    shell["c"][:, 3] = 255
    shell["r"] = (255, 128, 128, 128)  # identity quat (w=1)
    return shell


def ply_to_splat(ply_path: Path, splat_path: Path,
                 scale_mult: float = None, max_points: int = 0,
                 shell_bins: int = 0, opacity_floor: float = 0.0,
                 scale_clamp: float = 0.0):
    """Vectorized 3DGS .ply -> antimatter15 .splat conversion.

    Records are sorted by volume*opacity importance; `max_points` truncates to
    the top-N, which is how the mobile tier is derived from the HD set."""
    from plyfile import PlyData

    if scale_mult is None:
        scale_mult = SPLAT_SCALE_MULT
    v = PlyData.read(str(ply_path))["vertex"].data
    if opacity_floor > 0.0:
        # Cull near-transparent gaussians: they render as color bloom past
        # real geometry borders (neon-sign fringing).
        keep = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float64))) >= opacity_floor
        log(f"opacity_floor {opacity_floor}: culled {int((~keep).sum())}/{len(v)}")
        v = v[keep]
    n = len(v)
    opac = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float64)))
    vol = np.exp(v["scale_0"].astype(np.float64) + v["scale_1"] + v["scale_2"])
    order = np.argsort(-(vol * opac))

    SH_C0 = 0.28209479177387814
    rec = np.zeros(n, dtype=_SPLAT_REC)
    rec["p"] = np.stack([v["x"], v["y"], v["z"]], axis=1)
    rec["s"] = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1)) * scale_mult
    if scale_clamp > 0.0:
        rec["s"] = np.minimum(rec["s"], scale_clamp)
    rgb = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1) * SH_C0 + 0.5
    rec["c"][:, :3] = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    rec["c"][:, 3] = np.clip(opac * 255.0, 0, 255).astype(np.uint8)
    quat = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype(np.float64)
    quat /= (np.linalg.norm(quat, axis=1, keepdims=True) + 1e-12)
    rec["r"] = np.clip(quat * 128.0 + 128.0, 0, 255).astype(np.uint8)

    rec = rec[order]
    if max_points and n > max_points:
        # Uniform random subsample (not top-importance): keeping only the
        # largest splats collapses fine detail and skews the extent
        # distribution the viewer uses for framing.
        keep = np.sort(np.random.default_rng(42).choice(n, size=max_points, replace=False))
        rec = rec[keep]
    if shell_bins > 0:
        shell = shell_records(rec, bins_x=shell_bins)
        rec = np.concatenate([rec, shell])
    with open(splat_path, "wb") as fh:
        fh.write(rec.tobytes())
    log(f"splat written: {splat_path} ({splat_path.stat().st_size/1e6:.0f}MB, {len(rec)} gaussians)")
    return splat_path


def upload_artifacts(job_id: str, files: dict, prefix: str = None):
    """Upload {s3_key_suffix: local_path} to the job prefix. Returns url map."""
    import boto3

    s3 = boto3.client("s3", region_name="us-east-1")
    ctype = {".png": "image/png", ".jpg": "image/jpeg", ".json": "application/json",
             ".splat": "application/octet-stream", ".ply": "application/octet-stream"}
    urls = {}
    for suffix, path in files.items():
        key = f"{prefix or S3_PREFIX}/{job_id}/{suffix}"
        ext = Path(suffix).suffix
        s3.upload_file(str(path), S3_BUCKET, key,
                       ExtraArgs={"ContentType": ctype.get(ext, "application/octet-stream")})
        urls[suffix] = f"{S3_HTTP_BASE}/{key}"
        log(f"uploaded {key} ({Path(path).stat().st_size/1e6:.1f}MB)")
    return urls


def process_job(job: dict):
    job_id = job["jobId"]
    prompt = job["prompt"]
    t_start = time.time()
    jdir = WORK_DIR / job_id
    jdir.mkdir(parents=True, exist_ok=True)

    cfg = job_params(job)
    log(f"job params: {cfg}")
    seed_path = jdir / "seed.png"
    pano_path = jdir / "panorama.png"
    views_dir = jdir / "views"
    recon_dir = jdir / "recon"
    splat_path = jdir / "world.splat"
    hd_path = jdir / "world-hd.splat"

    gen_seed_image(prompt, seed_path)
    gen_panorama(prompt, seed_path, pano_path,
                 steps=cfg["pano_steps"], true_cfg=cfg["pano_true_cfg"])
    recon_input = pano_path
    if cfg["pano_upscale"] > 1:
        recon_input = jdir / "panorama-up.png"
        upscale_panorama(pano_path, cfg["pano_upscale"], recon_input)

    views_dir, cam_path = render_views(recon_input, views_dir,
                                       view_size=cfg["view_size"],
                                       dense_floor=bool(cfg["dense_floor"]))
    hd = cfg["gs_max_points_hd"] > cfg["gs_max_points"]
    gs_cap = cfg["gs_max_points_hd"] if hd else cfg["gs_max_points"]
    ply_path = run_worldmirror(
        views_dir, cam_path, recon_dir,
        target_size=cfg["recon_target_size"],
        gs_cap=gs_cap,
        sky_mask=bool(cfg["sky_mask"]), edge_mask=bool(cfg["edge_mask"]))

    if cfg["two_pass"]:
        # Coverage pass: base-res recon with a dense-floor rig and no masks;
        # its splats patch voxels the detail pass left as black holes.
        cov_views = jdir / "views-cov"
        cov_recon = jdir / "recon-cov"
        cov_views, cov_cam = render_views(recon_input, cov_views,
                                          view_size=cfg["coverage_view_size"],
                                          dense_floor=True)
        cov_ply = run_worldmirror(cov_views, cov_cam, cov_recon,
                                  target_size=cfg["coverage_recon_size"],
                                  gs_cap=cfg["gs_max_points"],
                                  sky_mask=False, edge_mask=False)
        merged_ply = jdir / "merged.ply"
        ply_path = merge_fill_ply(ply_path, cov_ply, merged_ply,
                                  floor_only=bool(cfg["fill_floor_only"]))
    export_kw = dict(scale_mult=cfg["splat_scale_mult"],
                     shell_bins=cfg["shell_bins"] if cfg["shell_fill"] else 0,
                     opacity_floor=cfg["opacity_floor"],
                     scale_clamp=cfg["scale_clamp"])
    if hd:
        ply_to_splat(ply_path, hd_path, **export_kw)
        ply_to_splat(ply_path, splat_path, max_points=cfg["gs_max_points"], **export_kw)
    else:
        ply_to_splat(ply_path, splat_path, **export_kw)

    # preview = downscaled panorama
    from PIL import Image
    preview = jdir / "preview.png"
    im = Image.open(pano_path)
    im.thumbnail((1600, 800))
    im.save(preview)

    elapsed = time.time() - t_start
    meta = {
        "jobId": job_id, "prompt": prompt, "elapsedSec": round(elapsed),
        "splatBytes": splat_path.stat().st_size,
        "generator": "hy-world-2.0 (HY-Pano-2.0-Qwen + WorldMirror-2.0)",
        "finishedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "params": cfg,
        # Viewer camera override (splat-local space): all views are rendered
        # from the panorama center at the origin looking +Z. Without this the
        # viewer guesses inside/outside from the radial histogram, which
        # misfires on deep (non-shell) reconstructions.
        "camera": {"position": [0.0, 0.0, 0.0], "target": [0.0, 0.0, 0.12]},
    }
    if hd:
        meta["hdSplatBytes"] = hd_path.stat().st_size

    uploads = {
        "world.splat": splat_path,
        "panorama.png": pano_path,
        "preview.png": preview,
        "seed.png": seed_path,
    }
    if hd:
        uploads["world-hd.splat"] = hd_path

    meta_path = jdir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    uploads["meta.json"] = meta_path
    # Experiments upload to a non-listed staging prefix (the production picker
    # auto-lists worldgen/); canonical worlds pass s3Prefix=null.
    urls = upload_artifacts(job_id, uploads, prefix=job.get("s3Prefix"))
    meta["urls"] = urls
    viewer = f"{VIEWER_BASE}?src={urls['world.splat']}"
    log(f"JOB DONE {job_id} in {elapsed/60:.1f} min — {viewer}")
    discord(f"✅ worldgen `{job_id}` done in {elapsed/60:.0f}m — {viewer}")

    # cleanup local
    import shutil
    shutil.rmtree(jdir, ignore_errors=True)
    return meta


# ---------------------------------------------------------------- main loop
def main():
    import redis
    import torch

    log(f"starting: queue={QUEUE_NAME} quant={QUANT_MODE} "
        f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD,
                    decode_responses=True, socket_keepalive=True,
                    health_check_interval=30, retry_on_timeout=True,
                    socket_connect_timeout=10)
    r.ping()
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    def on_term(signum, frame):
        # Requeue in-flight job explicitly: the finally block below LREMs it
        # from PROCESSING_LIST during interpreter shutdown, so leaving it
        # there is not enough.
        _shutdown["flag"] = True
        raw = _shutdown["current_job_raw"]
        if raw:
            try:
                r.lpush(QUEUE_NAME, raw)
                log("SIGTERM: requeued in-flight job")
            except Exception:
                pass
        sys.exit(0)

    signal.signal(signal.SIGTERM, on_term)

    # Reclaim jobs a previous (killed/interrupted) worker left in flight.
    # Safe with maxReplicaCount=1.
    stale = r.lrange(PROCESSING_LIST, 0, -1)
    if stale:
        log(f"reclaiming {len(stale)} stale in-flight job(s)")
        pipe = r.pipeline()
        for raw in stale:
            pipe.lrem(PROCESSING_LIST, 0, raw)
            pipe.lpush(QUEUE_NAME, raw)
        pipe.execute()

    while not _shutdown["flag"]:
        try:
            raw = r.blmove(QUEUE_NAME, PROCESSING_LIST, BLPOP_TIMEOUT, "LEFT", "RIGHT")
        except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError) as e:
            log(f"redis connection error ({e}); reconnecting in 5s")
            time.sleep(5)
            continue
        if raw is None:
            continue
        try:
            job = json.loads(raw)
        except Exception:
            log(f"bad payload dropped: {raw[:200]}")
            r.lrem(PROCESSING_LIST, 0, raw)
            continue
        job_id = job.get("jobId", "unknown")
        attempts = int(job.get("attempts", 0))
        _shutdown["current_job_raw"] = raw
        log(f"picked job {job_id} (attempt {attempts + 1})")
        try:
            meta = process_job(job)
            r.set(f"worldgen:done:{job_id}", json.dumps(meta), ex=7 * 24 * 3600)
        except SystemExit:
            raise
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            log(f"JOB FAILED {job_id}: {err}\n{traceback.format_exc()}")
            free_gpu()
            if attempts + 1 < MAX_ATTEMPTS:
                job["attempts"] = attempts + 1
                r.rpush(QUEUE_NAME, json.dumps(job))
                log(f"requeued {job_id} (attempt {attempts + 1}/{MAX_ATTEMPTS})")
            else:
                r.set(f"worldgen:failed:{job_id}",
                      json.dumps({"error": err, "trace": traceback.format_exc()[-3000:]}),
                      ex=7 * 24 * 3600)
                discord(f"❌ worldgen `{job_id}` failed after {MAX_ATTEMPTS} attempts: {err[:300]}")
        finally:
            _shutdown["current_job_raw"] = None
            r.lrem(PROCESSING_LIST, 0, raw)

    log("shutdown")


if __name__ == "__main__":
    main()
