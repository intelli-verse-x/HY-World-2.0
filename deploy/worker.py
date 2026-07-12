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
QUANT_MODE = os.environ.get("QUANT_MODE", "auto")  # auto | 4bit | bf16
MAX_ATTEMPTS = int(os.environ.get("MAX_ATTEMPTS", "2"))
BLPOP_TIMEOUT = int(os.environ.get("BLPOP_TIMEOUT", "120"))

PANO_MODEL = os.environ.get("PANO_MODEL", "Qwen/Qwen-Image-Edit-2509")
PANO_LORA = os.environ.get("PANO_LORA", "tencent/HY-World-2.0")
SEED_MODEL = os.environ.get("SEED_MODEL", "stabilityai/sdxl-turbo")
MIRROR_MODEL = os.environ.get("MIRROR_MODEL", "tencent/HY-World-2.0")

PANO_H = int(os.environ.get("PANO_H", "960"))
PANO_W = int(os.environ.get("PANO_W", "1952"))
PANO_STEPS = int(os.environ.get("PANO_STEPS", "28"))
VIEW_SIZE = int(os.environ.get("VIEW_SIZE", "768"))
VIEW_FOV_DEG = float(os.environ.get("VIEW_FOV_DEG", "70"))
RECON_TARGET_SIZE = int(os.environ.get("RECON_TARGET_SIZE", "756"))
GS_MAX_POINTS = int(os.environ.get("GS_MAX_POINTS", "2000000"))

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
    # "auto" resolves to 4bit: bf16 (~57GB of weights) needs either >57GB VRAM
    # or >57GB host RAM for CPU offload; neither fits g6e.2xl/g6.4xl nodes.
    return "4bit" if QUANT_MODE == "auto" else QUANT_MODE


def gen_panorama(prompt: str, seed_path: Path, out_path: Path):
    """Stage B: seed image -> 360 equirectangular panorama (HY-Pano-2.0 Qwen)."""
    import torch

    sys.path.insert(0, "/app/hyworld2/panogen")
    from pipeline_with_qwen_image import HunyuanPanoPipeline
    from qwen_image import PanoDiffusionPipeline

    quant = resolved_quant_mode()
    t0 = time.time()
    kwargs = {"torch_dtype": torch.bfloat16}
    if quant == "4bit":
        from diffusers import PipelineQuantizationConfig
        kwargs["quantization_config"] = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.bfloat16,
            },
            components_to_quantize=["transformer", "text_encoder"],
        )
    pipe = PanoDiffusionPipeline.from_pretrained(PANO_MODEL, **kwargs)
    if quant != "4bit":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
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
        num_inference_steps=PANO_STEPS,
        guidance_scale=1.0,
        true_cfg_scale=4.0,
    )
    pano.save(out_path)
    del hy, pipe
    free_gpu()
    log(f"panorama done: load={t1-t0:.0f}s infer={time.time()-t1:.0f}s -> {out_path}")
    return out_path


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def render_views(pano_path: Path, views_dir: Path):
    """Stage C: equirect panorama -> pinhole views + camera prior JSON.

    OpenCV convention: +x right, +y down, +z forward. World frame = camera 0.
    """
    import cv2

    views_dir.mkdir(parents=True, exist_ok=True)
    pano = cv2.imread(str(pano_path), cv2.IMREAD_COLOR)
    ph, pw = pano.shape[:2]

    size = VIEW_SIZE
    f = 0.5 * size / math.tan(math.radians(VIEW_FOV_DEG) / 2)
    cx = cy = size / 2.0
    K = [[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]]

    rig = []  # (yaw_deg, pitch_deg)
    rig += [(y, 0.0) for y in range(0, 360, 45)]          # 8 equator views
    rig += [(y, 35.0) for y in range(0, 360, 90)]         # 4 up
    rig += [(y + 45, -35.0) for y in range(0, 360, 90)]   # 4 down (offset yaw)

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


def run_worldmirror(views_dir: Path, cam_path: Path, out_dir: Path):
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
        target_size=RECON_TARGET_SIZE,
        save_depth=False,
        save_normal=False,
        save_points=False,
        save_camera=True,
        save_gs=True,
        apply_sky_mask=True,
        apply_edge_mask=True,
        compress_gs_max_points=GS_MAX_POINTS,
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


def ply_to_splat(ply_path: Path, splat_path: Path):
    """Vectorized 3DGS .ply -> antimatter15 .splat conversion."""
    from plyfile import PlyData

    v = PlyData.read(str(ply_path))["vertex"].data
    n = len(v)
    opac = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float64)))
    vol = np.exp(v["scale_0"].astype(np.float64) + v["scale_1"] + v["scale_2"])
    order = np.argsort(-(vol * opac))

    SH_C0 = 0.28209479177387814
    rec = np.zeros(n, dtype=[("p", "<f4", 3), ("s", "<f4", 3),
                             ("c", "u1", 4), ("r", "u1", 4)])
    rec["p"] = np.stack([v["x"], v["y"], v["z"]], axis=1)
    rec["s"] = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1))
    rgb = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1) * SH_C0 + 0.5
    rec["c"][:, :3] = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    rec["c"][:, 3] = np.clip(opac * 255.0, 0, 255).astype(np.uint8)
    quat = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype(np.float64)
    quat /= (np.linalg.norm(quat, axis=1, keepdims=True) + 1e-12)
    rec["r"] = np.clip(quat * 128.0 + 128.0, 0, 255).astype(np.uint8)

    rec = rec[order]
    with open(splat_path, "wb") as fh:
        fh.write(rec.tobytes())
    log(f"splat written: {splat_path} ({splat_path.stat().st_size/1e6:.0f}MB, {n} gaussians)")
    return splat_path


def upload_artifacts(job_id: str, files: dict):
    """Upload {s3_key_suffix: local_path} to the job prefix. Returns url map."""
    import boto3

    s3 = boto3.client("s3", region_name="us-east-1")
    ctype = {".png": "image/png", ".jpg": "image/jpeg", ".json": "application/json",
             ".splat": "application/octet-stream", ".ply": "application/octet-stream"}
    urls = {}
    for suffix, path in files.items():
        key = f"{S3_PREFIX}/{job_id}/{suffix}"
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

    seed_path = jdir / "seed.png"
    pano_path = jdir / "panorama.png"
    views_dir = jdir / "views"
    recon_dir = jdir / "recon"
    splat_path = jdir / "world.splat"

    gen_seed_image(prompt, seed_path)
    gen_panorama(prompt, seed_path, pano_path)
    views_dir, cam_path = render_views(pano_path, views_dir)
    ply_path = run_worldmirror(views_dir, cam_path, recon_dir)
    ply_to_splat(ply_path, splat_path)

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
    }
    meta_path = jdir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    urls = upload_artifacts(job_id, {
        "world.splat": splat_path,
        "panorama.png": pano_path,
        "preview.png": preview,
        "seed.png": seed_path,
        "meta.json": meta_path,
    })
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
        # In-flight payload stays on PROCESSING_LIST; the next pod reclaims it.
        _shutdown["flag"] = True
        log("SIGTERM: exiting (in-flight job left on processing list for reclaim)")
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
