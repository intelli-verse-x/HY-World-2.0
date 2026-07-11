"""
HY-World 2.0 world-template generation queue worker.

Consumes JSON jobs from the Redis list `pipeline:signal:worldgen` on
content-factory-redis (same pattern as the other aicart GPU workers), runs the
full generate-once/play-many pipeline on all local GPUs, and uploads the
resulting 3D world artifacts (3DGS .ply/.spz, mesh, panorama, preview renders)
to S3.

Job payload:
    {
      "jobId": "smoke-001",
      "prompt": "a cozy ancient library interior ...",
      "style": "photorealistic",              # optional, appended to prompt
      "sceneType": "indoor",                  # optional: indoor|outdoor (default indoor)
      "outputS3Prefix": "worldgen/smoke-001", # optional, default worldgen/{jobId}
      "seedImageS3": "s3://bucket/key.png"    # optional: skip text->image stage
    }

Pipeline stages (see hyworld2/worldgen/README.md):
    0a. Seed image      — text -> perspective image via litellm image API (no GPU)
    0b. Panorama        — Qwen-Image-Edit-2509 + HY-Pano-2.0 LoRA (CPU-offloaded)
    1.  traj_generate   — VLM-guided trajectory planning (1 GPU, VLM via litellm)
    2.  traj_render     — torchrun point-cloud rendering (all GPUs)
    3.  video_gen       — WorldStereo-2 DMD keyframe generation (all GPUs, FSDP)
    4.  gen_gs_data     — 3DGS training data prep (all GPUs)
    5.  world_gs_trainer— 3DGS optimization + ply/spz/mesh export (all GPUs)

Reliability: jobs are moved signal-list -> processing-list (BRPOPLPUSH) so the
KEDA scaler keeps the replica alive while a job runs; a heartbeat key with TTL
lets a replacement pod detect and resume/clear stale jobs after spot loss.
"""

import base64
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import boto3
import redis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [WORLDGEN] - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class Config:
    REDIS_HOST = os.environ.get("REDIS_HOST", "content-factory-redis.aicart.svc.cluster.local")
    REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
    REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD") or None

    QUEUE = os.environ.get("WORLDGEN_QUEUE", "pipeline:signal:worldgen")
    PROCESSING_QUEUE = os.environ.get("WORLDGEN_PROCESSING_QUEUE", "pipeline:worldgen:processing")
    DONE_QUEUE = os.environ.get("WORLDGEN_DONE_QUEUE", "pipeline:done:worldgen")
    STATUS_KEY_PREFIX = "pipeline:worldgen:status:"
    HEARTBEAT_KEY_PREFIX = "pipeline:worldgen:heartbeat:"
    HEARTBEAT_TTL = 300  # seconds

    S3_BUCKET = os.environ.get("AWS_S3_BUCKET_NAME", "intelli-verse-x-media")
    S3_OUTPUT_BASE = os.environ.get("S3_OUTPUT_BASE", "worldgen")
    AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

    # litellm gateway: used for both the seed image (text->image) and the VLM
    # calls inside traj_generate/traj_render (OpenAI-compatible chat).
    LLM_ADDR = os.environ.get("LLM_ADDR", "litellm.aicart.svc.cluster.local")
    LLM_PORT = int(os.environ.get("LLM_PORT", "80"))
    VLM_MODEL = os.environ.get("VLM_MODEL", "gemini/gemini-2.5-flash")
    IMAGE_MODEL = os.environ.get("IMAGE_MODEL", "gemini/gemini-3.1-flash-image")
    VLM_API_KEY = os.environ.get("VLM_API_KEY", "EMPTY")

    REPO_DIR = Path(os.environ.get("REPO_DIR", "/app"))
    WORLDGEN_DIR = REPO_DIR / "hyworld2" / "worldgen"
    SCENES_DIR = Path(os.environ.get("SCENES_DIR", "/workspace/scenes"))

    NGPU = int(os.environ.get("NGPU", "0")) or None  # None -> autodetect
    MAX_JOB_SECONDS = int(os.environ.get("MAX_JOB_SECONDS", "16200"))  # 4.5 h hard cap
    PANO_WIDTH = int(os.environ.get("PANO_WIDTH", "1952"))
    PANO_HEIGHT = int(os.environ.get("PANO_HEIGHT", "960"))
    GS_MAX_STEPS = int(os.environ.get("GS_MAX_STEPS", "2000"))  # x4 GPUs per upstream README


def n_gpus() -> int:
    if Config.NGPU:
        return Config.NGPU
    import torch
    return max(torch.cuda.device_count(), 1)


def redis_client() -> redis.Redis:
    return redis.Redis(
        host=Config.REDIS_HOST,
        port=Config.REDIS_PORT,
        password=Config.REDIS_PASSWORD,
        decode_responses=True,
        socket_keepalive=True,
    )


def s3_client():
    return boto3.client("s3", region_name=Config.AWS_REGION)


# --------------------------------------------------------------------------
# Stage 0a: seed image via litellm image API (no GPU time burned)
# --------------------------------------------------------------------------
def generate_seed_image(prompt: str, out_path: Path) -> None:
    import httpx

    url = f"http://{Config.LLM_ADDR}:{Config.LLM_PORT}/v1/images/generations"
    logger.info(f"Seed image via {Config.IMAGE_MODEL} ...")
    resp = httpx.post(
        url,
        headers={"Authorization": f"Bearer {Config.VLM_API_KEY}"},
        json={"model": Config.IMAGE_MODEL, "prompt": prompt, "n": 1, "size": "1024x1024"},
        timeout=300,
    )
    resp.raise_for_status()
    data = resp.json()["data"][0]
    if data.get("b64_json"):
        out_path.write_bytes(base64.b64decode(data["b64_json"]))
    else:
        img = httpx.get(data["url"], timeout=120)
        img.raise_for_status()
        out_path.write_bytes(img.content)
    logger.info(f"Seed image saved: {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")


# --------------------------------------------------------------------------
# Stage 0b: panorama via Qwen-Image-Edit-2509 + HY-Pano-2.0 LoRA
# --------------------------------------------------------------------------
def generate_panorama(seed_image: Path, prompt: str, out_path: Path) -> None:
    """Run in a subprocess so all GPU memory is guaranteed released after."""
    script = Config.REPO_DIR / "deploy" / "worldgen" / "run_pano.py"
    cmd = [
        sys.executable, str(script),
        "--image", str(seed_image),
        "--prompt", prompt,
        "--save", str(out_path),
        "--width", str(Config.PANO_WIDTH),
        "--height", str(Config.PANO_HEIGHT),
    ]
    run_stage("panorama", cmd, cwd=Config.REPO_DIR / "hyworld2" / "panogen")


# --------------------------------------------------------------------------
# Subprocess plumbing for pipeline stages
# --------------------------------------------------------------------------
_DEADLINE = 0.0


def remaining_seconds() -> int:
    return max(int(_DEADLINE - time.time()), 60)


def run_stage(name: str, cmd: list, cwd: Path, extra_env: dict | None = None) -> None:
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(Config.REPO_DIR))
    env["VLM_API_KEY"] = Config.VLM_API_KEY
    if extra_env:
        env.update(extra_env)
    logger.info(f"[stage:{name}] {' '.join(map(str, cmd))}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, timeout=remaining_seconds())
    if proc.returncode != 0:
        raise RuntimeError(f"stage {name} failed with exit code {proc.returncode}")
    logger.info(f"[stage:{name}] done in {time.time()-t0:.0f}s")


def torchrun(nproc: int, script: str, *args: str) -> list:
    return ["torchrun", "--standalone", f"--nproc_per_node={nproc}", script, *args]


# --------------------------------------------------------------------------
# S3 upload
# --------------------------------------------------------------------------
UPLOAD_EXT_CONTENT_TYPES = {
    ".ply": "application/octet-stream",
    ".spz": "application/octet-stream",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".mp4": "video/mp4",
    ".json": "application/json",
    ".pt": "application/octet-stream",
    ".obj": "model/obj",
    ".glb": "model/gltf-binary",
}


def upload_dir(s3, local: Path, bucket: str, prefix: str, manifest: list) -> None:
    for p in sorted(local.rglob("*")):
        if not p.is_file():
            continue
        key = f"{prefix}/{p.relative_to(local)}"
        ctype = UPLOAD_EXT_CONTENT_TYPES.get(p.suffix.lower(), "application/octet-stream")
        s3.upload_file(str(p), bucket, key, ExtraArgs={"ContentType": ctype})
        manifest.append({"key": key, "bytes": p.stat().st_size})
        logger.info(f"uploaded s3://{bucket}/{key} ({p.stat().st_size/1e6:.1f} MB)")


def upload_outputs(job_id: str, scene_dir: Path, result_dir: Path, prefix: str) -> list:
    s3 = s3_client()
    manifest: list = []
    bucket = Config.S3_BUCKET

    # Final 3DGS + mesh outputs
    if result_dir.exists():
        upload_dir(s3, result_dir, bucket, f"{prefix}/gs", manifest)

    # Panorama + scene metadata
    for fname in ("panorama.png", "seed_image.png", "meta_info.json", "objects.json"):
        p = scene_dir / fname
        if p.exists():
            key = f"{prefix}/{fname}"
            s3.upload_file(str(p), bucket, key)
            manifest.append({"key": key, "bytes": p.stat().st_size})

    # A few preview renders (generated keyframe videos), not the whole tree
    previews = sorted(scene_dir.glob("render_results/*/traj*/worldstereo-*_result.mp4"))[:6]
    for p in previews:
        key = f"{prefix}/previews/{p.parent.parent.name}_{p.parent.name}.mp4"
        s3.upload_file(str(p), bucket, key, ExtraArgs={"ContentType": "video/mp4"})
        manifest.append({"key": key, "bytes": p.stat().st_size})

    # Global point cloud / mesh intermediates are useful for engine import
    for rel in ("render_results/global_pcd.ply", "render_results/global_mesh.ply"):
        p = scene_dir / rel
        if p.exists():
            key = f"{prefix}/{Path(rel).name}"
            s3.upload_file(str(p), bucket, key)
            manifest.append({"key": key, "bytes": p.stat().st_size})

    manifest_key = f"{prefix}/manifest.json"
    s3.put_object(
        Bucket=bucket,
        Key=manifest_key,
        Body=json.dumps({"jobId": job_id, "files": manifest}, indent=2),
        ContentType="application/json",
    )
    return manifest


# --------------------------------------------------------------------------
# Job processing
# --------------------------------------------------------------------------
def process_job(r: redis.Redis, raw_job: str) -> None:
    global _DEADLINE
    job = json.loads(raw_job)
    job_id = job.get("jobId") or str(uuid.uuid4())
    prompt = job["prompt"]
    style = job.get("style", "")
    scene_type = job.get("sceneType", "indoor")
    prefix = job.get("outputS3Prefix") or f"{Config.S3_OUTPUT_BASE}/{job_id}"
    prefix = prefix.strip("/")
    full_prompt = f"{prompt}, {style}" if style else prompt

    _DEADLINE = time.time() + Config.MAX_JOB_SECONDS
    status_key = Config.STATUS_KEY_PREFIX + job_id
    hb_key = Config.HEARTBEAT_KEY_PREFIX + job_id

    def set_status(stage: str, state: str = "running", **extra):
        r.set(status_key, json.dumps({
            "jobId": job_id, "state": state, "stage": stage,
            "updatedAt": int(time.time()), **extra,
        }))

    stop_hb = threading.Event()

    def heartbeat():
        while not stop_hb.is_set():
            r.set(hb_key, str(int(time.time())), ex=Config.HEARTBEAT_TTL)
            stop_hb.wait(60)

    hb_thread = threading.Thread(target=heartbeat, daemon=True)
    hb_thread.start()

    scene_dir = Config.SCENES_DIR / job_id
    result_dir = scene_dir / "gs_result"
    ngpu = n_gpus()
    wg = Config.WORLDGEN_DIR
    t_start = time.time()
    logger.info(f"=== JOB {job_id} start | ngpu={ngpu} | prompt: {full_prompt[:120]}")

    try:
        scene_dir.mkdir(parents=True, exist_ok=True)
        (scene_dir / "meta_info.json").write_text(json.dumps({"scene_type": scene_type}))

        # Stage 0a: seed image
        set_status("seed_image")
        seed_path = scene_dir / "seed_image.png"
        if not seed_path.exists():
            if job.get("seedImageS3"):
                b, k = job["seedImageS3"].replace("s3://", "").split("/", 1)
                s3_client().download_file(b, k, str(seed_path))
            else:
                generate_seed_image(full_prompt, seed_path)

        # Stage 0b: panorama
        set_status("panorama")
        pano_path = scene_dir / "panorama.png"
        if not pano_path.exists():
            generate_panorama(seed_path, full_prompt, pano_path)

        vlm_args = [
            "--llm_addr", Config.LLM_ADDR,
            "--llm_port", str(Config.LLM_PORT),
            "--llm_name", Config.VLM_MODEL,
        ]

        # Stage 1: trajectory planning (single process)
        set_status("traj_generate")
        run_stage("traj_generate", [
            sys.executable, "traj_generate.py",
            "--target_path", str(scene_dir), *vlm_args,
            "--apply_nav_traj", "--apply_up_route", "--apply_recon_iteration",
            "--force_vlm", "--skip_exist",
        ], cwd=wg)

        # Stage 2: trajectory rendering (multi-GPU)
        set_status("traj_render")
        run_stage("traj_render", torchrun(
            ngpu, "traj_render.py",
            "--target_path", str(scene_dir), *vlm_args, "--skip_exist",
        ), cwd=wg)

        # Stage 3: world expansion (multi-GPU + FSDP)
        set_status("video_gen")
        run_stage("video_gen", torchrun(
            ngpu, "video_gen.py",
            "--target_path", str(scene_dir), "--fsdp", "--skip_exist",
        ), cwd=wg)

        # Stage 4: 3DGS training data
        set_status("gen_gs_data")
        run_stage("gen_gs_data", torchrun(
            ngpu, "gen_gs_data.py",
            "--root_path", str(scene_dir), "--save_normal", "--split_sky",
        ), cwd=wg)

        # Stage 5: 3DGS training + export (upstream flags, steps scaled for ngpu)
        set_status("gs_train")
        steps = str(Config.GS_MAX_STEPS)
        run_stage("gs_train", [
            sys.executable, "-m", "world_gs_trainer", "default",
            "--data_dir", str(scene_dir / "gs_data"), "--result_dir", str(result_dir),
            "--max_steps", steps, "--save_steps", steps, "--eval_steps", steps,
            "--ply_steps", steps, "--save_ply", "--convert_to_spz", "--disable_video",
            "--use_scale_regularization", "--antialiased",
            "--depth_loss", "--normal_loss", "--sky_depth_from_pcd",
            "--use_mask_gaussian", "--mask_export_stochastic",
            "--no-mask-export-anchor-protection", "--use_anchor_protection", "--export_mesh",
            "--strategy.refine-start-iter", "150", "--strategy.refine-stop-iter", "750",
            "--strategy.refine-every", "100", "--strategy.refine-scale2d-stop-iter", "750",
            "--strategy.reset-every", "99990", "--strategy.grow-grad2d", "0.0001",
            "--strategy.prune-scale3d", "0.1",
        ], cwd=wg)

        # Upload
        set_status("upload")
        manifest = upload_outputs(job_id, scene_dir, result_dir, prefix)
        elapsed = int(time.time() - t_start)
        result = {
            "jobId": job_id, "state": "done", "elapsedSeconds": elapsed,
            "s3Prefix": f"s3://{Config.S3_BUCKET}/{prefix}/",
            "files": len(manifest),
        }
        set_status("done", state="done", **result)
        r.rpush(Config.DONE_QUEUE, json.dumps(result))
        logger.info(f"=== JOB {job_id} DONE in {elapsed}s -> s3://{Config.S3_BUCKET}/{prefix}/")

        # Free scratch space for the next job (outputs are in S3 now)
        shutil.rmtree(scene_dir, ignore_errors=True)

    except Exception as e:
        elapsed = int(time.time() - t_start)
        logger.error(f"=== JOB {job_id} FAILED after {elapsed}s: {e}", exc_info=True)
        set_status("failed", state="failed", error=str(e)[:2000], elapsedSeconds=elapsed)
        r.rpush(Config.DONE_QUEUE, json.dumps({"jobId": job_id, "state": "failed", "error": str(e)[:2000]}))
    finally:
        stop_hb.set()
        hb_thread.join(timeout=5)
        r.delete(hb_key)
        # Always clear the processing entry so KEDA can scale back to zero.
        r.lrem(Config.PROCESSING_QUEUE, 0, raw_job)


def recover_stale_jobs(r: redis.Redis) -> list:
    """Return processing-list entries with no live heartbeat (spot loss etc.)."""
    stale = []
    for raw in r.lrange(Config.PROCESSING_QUEUE, 0, -1):
        try:
            job_id = json.loads(raw).get("jobId", "")
        except (ValueError, AttributeError):
            r.lrem(Config.PROCESSING_QUEUE, 0, raw)
            continue
        if not r.exists(Config.HEARTBEAT_KEY_PREFIX + job_id):
            stale.append(raw)
    return stale


def main() -> None:
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    r = redis_client()
    r.ping()
    logger.info(
        f"worker up | queue={Config.QUEUE} | gpus={n_gpus()} | "
        f"bucket={Config.S3_BUCKET} | scenes={Config.SCENES_DIR}"
    )

    for raw in recover_stale_jobs(r):
        logger.warning(f"resuming stale job from processing list: {raw[:120]}")
        process_job(r, raw)

    while True:
        raw = r.brpoplpush(Config.QUEUE, Config.PROCESSING_QUEUE, timeout=30)
        if raw is None:
            continue
        process_job(r, raw)


if __name__ == "__main__":
    main()
