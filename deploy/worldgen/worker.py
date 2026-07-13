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
import hashlib
import json
import logging
import os
import shlex
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

    QUEUE = os.environ.get("WORLDGEN_QUEUE", "pipeline:signal:worldgen-full")
    PROCESSING_QUEUE = os.environ.get(
        "WORLDGEN_PROCESSING_QUEUE", "pipeline:worldgen-full:processing"
    )
    DONE_QUEUE = os.environ.get("WORLDGEN_DONE_QUEUE", "pipeline:done:worldgen-full")
    STATUS_KEY_PREFIX = "pipeline:worldgen-full:status:"
    HEARTBEAT_KEY_PREFIX = "pipeline:worldgen-full:heartbeat:"
    HEARTBEAT_TTL = 300  # seconds

    S3_BUCKET = os.environ.get("AWS_S3_BUCKET_NAME", "intelli-verse-x-media")
    S3_OUTPUT_BASE = os.environ.get("S3_OUTPUT_BASE", "worldgen")
    S3_CHECKPOINT_BASE = os.environ.get("S3_CHECKPOINT_BASE", "worldgen-full-checkpoints")
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
    SOURCE_COMMIT = os.environ.get("SOURCE_COMMIT", "unknown")
    IMAGE_URI = os.environ.get("IMAGE_URI", "unknown")
    INSTANCE_TYPE = os.environ.get("INSTANCE_TYPE", "unknown")
    INSTANCE_HOURLY_USD = float(os.environ.get("INSTANCE_HOURLY_USD", "3.40"))
    DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK", "")
    ALLOW_PRODUCTION_PROMOTION = os.environ.get("ALLOW_PRODUCTION_PROMOTION", "0") == "1"


STAGE_MODELS = {
    "seed_image": ["gemini/gemini-3.1-flash-image"],
    "panorama": ["Qwen/Qwen-Image-Edit-2509", "tencent/HY-World-2.0/HY-Pano-2.0"],
    "traj_generate": [
        "gemini/gemini-2.5-flash", "DiffusionWave/sam3",
        "Ruicheng/moge-2-vitl-normal", "naver-iv/zim-anything-vitl",
    ],
    "traj_render": ["gemini/gemini-2.5-flash", "Ruicheng/moge-2-vitl-normal"],
    "video_gen": [
        "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        "hanshanxue/WorldStereo/worldstereo-memory-dmd",
        "tencent/HY-World-2.0/HY-WorldMirror-2.0",
    ],
    "gen_gs_data": ["Ruicheng/moge-2-vitl-normal"],
    "gs_train": ["gsplat_maskgaussian", "HY-World 2.0 3DGS trainer"],
}


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def post_discord(message: str) -> None:
    if not Config.DISCORD_WEBHOOK:
        return
    try:
        import httpx
        httpx.post(Config.DISCORD_WEBHOOK, json={"content": message[:1900]}, timeout=20).raise_for_status()
    except Exception as exc:
        logger.warning("Discord notification failed: %s", exc)


def checkpoint_prefix(job_id: str) -> str:
    return f"{Config.S3_CHECKPOINT_BASE}/{job_id}"


def restore_checkpoint(s3, job_id: str, scene_dir: Path, result_dir: Path) -> int:
    """Restore the latest durable stage snapshot into an empty/replacement PVC."""
    prefix = f"{checkpoint_prefix(job_id)}/current/"
    restored = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=Config.S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(prefix):]
            if not rel:
                continue
            if rel.startswith("result/"):
                dst = result_dir / rel.removeprefix("result/")
            else:
                dst = scene_dir / rel.removeprefix("scene/")
            dst.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(Config.S3_BUCKET, obj["Key"], str(dst))
            restored += 1
    if restored:
        logger.info("restored %d checkpoint files for %s", restored, job_id)
    return restored


def stage_manifest(s3, job_id: str, stage: str) -> dict | None:
    try:
        obj = s3.get_object(
            Bucket=Config.S3_BUCKET,
            Key=f"{checkpoint_prefix(job_id)}/stages/{stage}.json",
        )
        return json.loads(obj["Body"].read())
    except s3.exceptions.NoSuchKey:
        return None
    except Exception as exc:
        if getattr(exc, "response", {}).get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            return None
        raise


def _checkpoint_files(scene_dir: Path, result_dir: Path):
    for root_name, root in (("scene", scene_dir), ("result", result_dir)):
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file():
                if root_name == "scene" and result_dir in path.parents:
                    continue
                yield root_name, root, path


def persist_checkpoint(
    s3, job_id: str, stage: str, scene_dir: Path, result_dir: Path,
    started_at: float, validation: dict, prompt_meta: dict,
) -> dict:
    """Incrementally upload stage state, then atomically publish its manifest."""
    prefix = checkpoint_prefix(job_id)
    files = []
    for root_name, root, path in _checkpoint_files(scene_dir, result_dir):
        rel = path.relative_to(root).as_posix()
        key = f"{prefix}/current/{root_name}/{rel}"
        digest = sha256_file(path)
        unchanged = False
        try:
            head = s3.head_object(Bucket=Config.S3_BUCKET, Key=key)
            unchanged = head.get("Metadata", {}).get("sha256") == digest
        except Exception:
            pass
        if not unchanged:
            s3.upload_file(
                str(path), Config.S3_BUCKET, key,
                ExtraArgs={"Metadata": {"sha256": digest, "stage": stage}},
            )
        files.append({"path": f"{root_name}/{rel}", "bytes": path.stat().st_size, "sha256": digest})

    elapsed = time.time() - started_at
    gpu_count = n_gpus()
    manifest = {
        "schemaVersion": 1,
        "jobId": job_id,
        "stage": stage,
        "status": "succeeded",
        "startedAt": int(started_at),
        "finishedAt": int(time.time()),
        "elapsedSeconds": round(elapsed, 3),
        "sourceCommit": Config.SOURCE_COMMIT,
        "image": Config.IMAGE_URI,
        "models": STAGE_MODELS.get(stage, []),
        "prompt": prompt_meta,
        "compute": {
            "instanceType": Config.INSTANCE_TYPE,
            "gpuCount": gpu_count,
            "estimatedCostUsd": round(elapsed / 3600 * Config.INSTANCE_HOURLY_USD, 4),
        },
        "validation": validation,
        "files": files,
    }
    body = json.dumps(manifest, indent=2).encode()
    manifest["manifestSha256"] = hashlib.sha256(body).hexdigest()
    s3.put_object(
        Bucket=Config.S3_BUCKET,
        Key=f"{prefix}/stages/{stage}.json",
        Body=json.dumps(manifest, indent=2),
        ContentType="application/json",
    )
    return manifest


def validate_stage(stage: str, scene_dir: Path, result_dir: Path) -> dict:
    """Fail closed when a stage did not produce its minimum upstream contract."""
    if stage == "seed_image":
        files = [scene_dir / "seed_image.png"]
    elif stage == "panorama":
        files = [scene_dir / "panorama.png"]
    elif stage == "traj_generate":
        files = list(scene_dir.glob("render_results/**/camera.json"))
        if not files:
            raise RuntimeError("trajectory contract failed: no camera.json paths")
        return {
            "trajectoryFiles": len(files),
            "objectsJson": (scene_dir / "objects.json").exists(),
            "navmeshPresent": (scene_dir / "navmesh").exists(),
            "upRoutes": len(list(scene_dir.glob("render_results/**/up*"))),
        }
    elif stage == "traj_render":
        files = list(scene_dir.glob("render_results/**/render.mp4"))
    elif stage == "video_gen":
        files = list(scene_dir.glob("render_results/**/*worldstereo*-result.mp4"))
        if not files:
            files = list(scene_dir.glob("render_results/**/*worldstereo*_result.mp4"))
    elif stage == "gen_gs_data":
        files = list((scene_dir / "gs_data").rglob("*")) if (scene_dir / "gs_data").exists() else []
        files = [p for p in files if p.is_file()]
    elif stage == "gs_train":
        files = list(result_dir.rglob("*.ply")) + list(result_dir.rglob("*.spz"))
    else:
        files = []
    if not files or any(not path.exists() or path.stat().st_size == 0 for path in files):
        raise RuntimeError(f"{stage} contract failed: required non-empty outputs missing")
    metrics = {"fileCount": len(files), "totalBytes": sum(path.stat().st_size for path in files)}
    if stage == "panorama":
        from PIL import Image
        with Image.open(files[0]) as image:
            metrics["nativeResolution"] = list(image.size)
    return metrics


def validate_landmark_visibility(scene_dir: Path, landmarks: list[dict]) -> dict:
    """Use the configured VLM to fail worlds whose five authored landmarks vanished."""
    from PIL import Image, ImageDraw
    import httpx

    frames = sorted(scene_dir.glob("render_results/**/start_frame.png"))[:16]
    if len(frames) < 5:
        raise RuntimeError(f"landmark visibility contract failed: only {len(frames)} source views")
    thumbs = []
    labels = []
    for index, frame in enumerate(frames):
        image = Image.open(frame).convert("RGB")
        image.thumbnail((512, 320))
        thumbs.append(image.copy())
        labels.append(f"frame-{index:02d}:{frame.relative_to(scene_dir)}")
    cols = 4
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 512, rows * 350), "black")
    draw = ImageDraw.Draw(sheet)
    for index, image in enumerate(thumbs):
        x, y = (index % cols) * 512, (index // cols) * 350
        sheet.paste(image, (x, y))
        draw.text((x + 8, y + 324), f"frame-{index:02d}", fill="white")
    sheet_path = scene_dir / "landmark-contact-sheet.jpg"
    sheet.save(sheet_path, quality=90)
    encoded = base64.b64encode(sheet_path.read_bytes()).decode()
    requested = [{"id": item["id"], "name": item["name"]} for item in landmarks]
    instruction = (
        "Inspect this contact sheet of geometry-only world-generation views. "
        "For each requested landmark, mark visible=true only when its physical scene geometry "
        "is clearly recognizable without HUD text or overlays. Return strict JSON only as "
        '{"landmarks":[{"id":"...","visible":true,"frameIds":["frame-00"]}]} '
        f"Requested landmarks: {json.dumps(requested)}"
    )
    response = httpx.post(
        f"http://{Config.LLM_ADDR}:{Config.LLM_PORT}/v1/chat/completions",
        headers={"Authorization": f"Bearer {Config.VLM_API_KEY}"},
        json={
            "model": Config.VLM_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
                ],
            }],
            "temperature": 0,
        },
        timeout=180,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"].strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0]
    verdict = json.loads(content)
    by_id = {entry["id"]: entry for entry in verdict.get("landmarks", [])}
    mapping = []
    for landmark in landmarks:
        seen = by_id.get(landmark["id"], {})
        frame_ids = seen.get("frameIds") or []
        mapping.append({
            **landmark,
            "trajectoryVisibility": frame_ids,
            "visibleWithoutOverlay": bool(seen.get("visible") and frame_ids),
            "reconstructedRegion": None,
        })
    artifact = {
        "schemaVersion": 1,
        "frameIndex": labels,
        "landmarks": mapping,
        "allFiveVisible": all(item["visibleWithoutOverlay"] for item in mapping),
        "validator": Config.VLM_MODEL,
    }
    (scene_dir / "landmark-map.json").write_text(json.dumps(artifact, indent=2))
    if not artifact["allFiveVisible"]:
        missing = [item["id"] for item in mapping if not item["visibleWithoutOverlay"]]
        raise RuntimeError(f"landmark visibility contract failed: not independently visible: {missing}")
    return artifact


def finalize_landmark_mapping(scene_dir: Path, result_dir: Path) -> dict:
    """Bind visible landmark evidence to the trained 3D artifact and story IDs."""
    path = scene_dir / "landmark-map.json"
    mapping = json.loads(path.read_text())
    gs_files = [
        str(item.relative_to(result_dir))
        for item in sorted(result_dir.rglob("*"))
        if item.is_file() and item.suffix.lower() in (".ply", ".spz", ".obj", ".glb")
    ]
    if not gs_files:
        raise RuntimeError("landmark mapping contract failed: no trained geometry artifact")
    for landmark in mapping["landmarks"]:
        landmark["reconstructedRegion"] = {
            "geometryArtifacts": gs_files,
            "visibilityEvidenceFrames": landmark["trajectoryVisibility"],
            "anchorPolicy": "geometry landmark; derive checkpoint anchor from trained depth/camera evidence",
        }
    mapping["trainedGeometryPresent"] = True
    path.write_text(json.dumps(mapping, indent=2))
    return mapping


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
def generate_panorama(seed_image: Path, prompt: str, out_path: Path, seed: int = 42) -> None:
    """Run in a subprocess so all GPU memory is guaranteed released after."""
    script = Config.REPO_DIR / "deploy" / "worldgen" / "run_pano.py"
    cmd = [
        sys.executable, str(script),
        "--image", str(seed_image),
        "--prompt", prompt,
        "--save", str(out_path),
        "--width", str(Config.PANO_WIDTH),
        "--height", str(Config.PANO_HEIGHT),
        "--seed", str(seed),
    ]
    run_stage("panorama", cmd, cwd=Config.REPO_DIR / "hyworld2" / "panogen")


# --------------------------------------------------------------------------
# Subprocess plumbing for pipeline stages
# --------------------------------------------------------------------------
_DEADLINE = 0.0


def remaining_seconds() -> int:
    return max(int(_DEADLINE - time.time()), 60)


def run_stage(
    name: str, cmd: list, cwd: Path, extra_env: dict | None = None,
    log_path: Path | None = None,
) -> None:
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(Config.REPO_DIR))
    env["VLM_API_KEY"] = Config.VLM_API_KEY
    if extra_env:
        env.update(extra_env)
    logger.info(f"[stage:{name}] {' '.join(map(str, cmd))}")
    t0 = time.time()
    actual_cmd = cmd
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        quoted = " ".join(shlex.quote(str(part)) for part in cmd)
        actual_cmd = ["bash", "-o", "pipefail", "-c", f"{quoted} 2>&1 | tee {shlex.quote(str(log_path))}"]
    proc = subprocess.run(actual_cmd, cwd=str(cwd), env=env, timeout=remaining_seconds())
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


def publish_provenance_manifest(s3, job_id: str, output_prefix: str, prompt_meta: dict) -> dict:
    stages = []
    for stage in STAGE_MODELS:
        manifest = stage_manifest(s3, job_id, stage)
        if manifest:
            stages.append({
                "stage": stage,
                "manifestKey": f"{checkpoint_prefix(job_id)}/stages/{stage}.json",
                "manifestSha256": manifest.get("manifestSha256"),
                "elapsedSeconds": manifest.get("elapsedSeconds"),
                "estimatedCostUsd": manifest.get("compute", {}).get("estimatedCostUsd"),
                "validation": manifest.get("validation"),
            })
    provenance = {
        "schemaVersion": 1,
        "jobId": job_id,
        "pipeline": [
            "HY-Pano 2.0",
            "WorldNav trajectory planning/rendering",
            "WorldStereo 2.0 + WorldMirror memory alignment",
            "multi-view 3DGS training/export",
        ],
        "sourceCommit": Config.SOURCE_COMMIT,
        "image": Config.IMAGE_URI,
        "prompt": prompt_meta,
        "nativePanoramaResolution": [Config.PANO_WIDTH, Config.PANO_HEIGHT],
        "upscaledPanoramaResolution": None,
        "truthfulResolutionNote": (
            "HY-Pano Qwen backend native output; no super-resolution label or synthetic 4K claim"
        ),
        "stages": stages,
        "totalEstimatedCostUsd": round(sum(item.get("estimatedCostUsd") or 0 for item in stages), 4),
        "v15RollbackPrefix": "s3://intelliverse-world-templates/worldgen/",
        "promotionStatus": "staging-only-pending-independent-gates",
    }
    s3.put_object(
        Bucket=Config.S3_BUCKET,
        Key=f"{output_prefix}/provenance-manifest.json",
        Body=json.dumps(provenance, indent=2),
        ContentType="application/json",
    )
    return provenance


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
    if not Config.ALLOW_PRODUCTION_PROMOTION and not prefix.startswith("worldgen-full-staging/"):
        prefix = f"worldgen-full-staging/{job_id}"
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
    s3 = s3_client()
    landmarks = job.get("landmarks") or []
    prompt_words = full_prompt.split()
    contract_errors = []
    planned_max_cost = Config.MAX_JOB_SECONDS / 3600 * Config.INSTANCE_HOURLY_USD
    budget_usd = float(job.get("budgetUsd", planned_max_cost))
    if planned_max_cost > budget_usd:
        contract_errors.append(
            f"planned max GPU cost ${planned_max_cost:.2f} exceeds job budget ${budget_usd:.2f}"
        )
    if len(prompt_words) > 77:
        contract_errors.append(f"prompt has {len(prompt_words)} whitespace tokens (>77)")
    if len(landmarks) != 5:
        contract_errors.append(f"exactly 5 landmarks required, got {len(landmarks)}")
    required_landmark_fields = {"id", "name", "promptPhrase", "checkpointId"}
    for landmark in landmarks:
        missing = required_landmark_fields - set(landmark)
        if missing:
            contract_errors.append(f"landmark {landmark.get('id', '?')} missing fields: {sorted(missing)}")
        elif landmark["promptPhrase"].lower() not in full_prompt.lower():
            contract_errors.append(
                f"landmark {landmark['id']} phrase not present in prompt: {landmark['promptPhrase']}"
            )
    if contract_errors:
        error = "authoring contract failed: " + "; ".join(contract_errors)
        logger.error(error)
        set_status("authoring_contract", state="failed", error=error)
        r.rpush(Config.DONE_QUEUE, json.dumps({"jobId": job_id, "state": "failed", "error": error}))
        stop_hb.set()
        hb_thread.join(timeout=5)
        r.delete(hb_key)
        r.lrem(Config.PROCESSING_QUEUE, 0, raw_job)
        return
    prompt_meta = {
        "text": full_prompt,
        "whitespaceTokenCount": len(prompt_words),
        "maxAuthoringTokens": 77,
        "clipConditioned": False,
        "conditioners": [
            "Gemini image API (seed)",
            "Qwen-Image-Edit tokenizer (HY-Pano)",
            "Gemini VLM (WorldNav captions)",
        ],
        "landmarks": landmarks,
        "seed": job.get("seed", 42),
        "jobBudgetUsd": budget_usd,
        "plannedMaxGpuCostUsd": round(planned_max_cost, 2),
    }
    logger.info(f"=== JOB {job_id} start | ngpu={ngpu} | prompt: {full_prompt[:120]}")

    try:
        scene_dir.mkdir(parents=True, exist_ok=True)
        restored = restore_checkpoint(s3, job_id, scene_dir, result_dir)
        (scene_dir / "meta_info.json").write_text(json.dumps({
            "scene_type": scene_type,
            "job_id": job_id,
            "prompt": prompt_meta,
            "source_commit": Config.SOURCE_COMMIT,
            "image": Config.IMAGE_URI,
        }, indent=2))

        def completed(stage: str) -> bool:
            manifest = stage_manifest(s3, job_id, stage)
            return bool(manifest and manifest.get("status") == "succeeded")

        def checkpoint(stage: str, started_at: float, extra_validation: dict | None = None) -> dict:
            validation = validate_stage(stage, scene_dir, result_dir)
            if extra_validation:
                validation.update(extra_validation)
            manifest = persist_checkpoint(
                s3, job_id, stage, scene_dir, result_dir,
                started_at, validation, prompt_meta,
            )
            set_status(stage, state="checkpointed", validation=validation)
            post_discord(
                f"HY-World full-stack `{job_id}` stage `{stage}` checkpointed "
                f"({manifest['elapsedSeconds']:.0f}s, ${manifest['compute']['estimatedCostUsd']:.2f} est.)"
            )
            return manifest

        logger.info("checkpoint restore files=%d", restored)

        # Stage 0a: seed image
        set_status("seed_image")
        seed_path = scene_dir / "seed_image.png"
        if not completed("seed_image"):
            stage_start = time.time()
            if job.get("seedImageS3"):
                b, k = job["seedImageS3"].replace("s3://", "").split("/", 1)
                s3.download_file(b, k, str(seed_path))
            else:
                generate_seed_image(full_prompt, seed_path)
            checkpoint("seed_image", stage_start)

        # Stage 0b: panorama
        set_status("panorama")
        pano_path = scene_dir / "panorama.png"
        if not completed("panorama"):
            stage_start = time.time()
            generate_panorama(seed_path, full_prompt, pano_path, seed=int(job.get("seed", 42)))
            checkpoint("panorama", stage_start)

        vlm_args = [
            "--llm_addr", Config.LLM_ADDR,
            "--llm_port", str(Config.LLM_PORT),
            "--llm_name", Config.VLM_MODEL,
        ]

        # Stage 1: trajectory planning (single process)
        set_status("traj_generate")
        if not completed("traj_generate"):
            stage_start = time.time()
            run_stage("traj_generate", [
                sys.executable, "traj_generate.py",
                "--target_path", str(scene_dir), *vlm_args,
                "--apply_nav_traj", "--apply_up_route", "--apply_recon_iteration",
                "--force_vlm", "--skip_exist",
            ], cwd=wg, log_path=scene_dir / "logs/traj_generate.log")
            checkpoint("traj_generate", stage_start)

        # Stage 2: trajectory rendering (multi-GPU)
        set_status("traj_render")
        if not completed("traj_render"):
            stage_start = time.time()
            run_stage("traj_render", torchrun(
                ngpu, "traj_render.py",
                "--target_path", str(scene_dir), *vlm_args, "--skip_exist",
            ), cwd=wg, log_path=scene_dir / "logs/traj_render.log")
            checkpoint("traj_render", stage_start)

        # Stage 3: world expansion (multi-GPU + FSDP)
        set_status("video_gen")
        if not completed("video_gen"):
            stage_start = time.time()
            run_stage("video_gen", torchrun(
                ngpu, "video_gen.py",
                "--target_path", str(scene_dir), "--fsdp", "--skip_exist",
            ), cwd=wg, log_path=scene_dir / "logs/video_gen.log")
            landmark_validation = validate_landmark_visibility(scene_dir, landmarks)
            checkpoint("video_gen", stage_start, {
                "allFiveLandmarksVisible": landmark_validation["allFiveVisible"],
                "landmarkMap": "scene/landmark-map.json",
            })

        # Stage 4: 3DGS training data
        set_status("gen_gs_data")
        if not completed("gen_gs_data"):
            stage_start = time.time()
            run_stage("gen_gs_data", torchrun(
                ngpu, "gen_gs_data.py",
                "--root_path", str(scene_dir), "--save_normal", "--split_sky",
            ), cwd=wg, log_path=scene_dir / "logs/gen_gs_data.log")
            checkpoint("gen_gs_data", stage_start)

        # Stage 5: 3DGS training + export (upstream flags, steps scaled for ngpu)
        set_status("gs_train")
        steps = str(Config.GS_MAX_STEPS)
        if not completed("gs_train"):
            stage_start = time.time()
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
            ], cwd=wg, log_path=scene_dir / "logs/gs_train.log")
            final_landmarks = finalize_landmark_mapping(scene_dir, result_dir)
            checkpoint("gs_train", stage_start, {
                "allFiveLandmarksMapped": all(
                    item.get("reconstructedRegion") for item in final_landmarks["landmarks"]
                ),
                "landmarkMap": "scene/landmark-map.json",
            })

        # Upload
        set_status("upload")
        manifest = upload_outputs(job_id, scene_dir, result_dir, prefix)
        provenance = publish_provenance_manifest(s3, job_id, prefix, prompt_meta)
        elapsed = int(time.time() - t_start)
        result = {
            "jobId": job_id, "state": "done", "elapsedSeconds": elapsed,
            "s3Prefix": f"s3://{Config.S3_BUCKET}/{prefix}/",
            "files": len(manifest),
            "estimatedCostUsd": provenance["totalEstimatedCostUsd"],
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
        failure = {
            "jobId": job_id,
            "status": "failed",
            "failedAt": int(time.time()),
            "elapsedSeconds": elapsed,
            "error": str(e)[:2000],
            "sourceCommit": Config.SOURCE_COMMIT,
            "image": Config.IMAGE_URI,
        }
        try:
            s3.put_object(
                Bucket=Config.S3_BUCKET,
                Key=f"{checkpoint_prefix(job_id)}/failures/{int(time.time())}.json",
                Body=json.dumps(failure, indent=2),
                ContentType="application/json",
            )
        except Exception:
            logger.exception("failed to persist failure manifest")
        post_discord(f"HY-World full-stack `{job_id}` FAILED after {elapsed}s: {str(e)[:500]}")
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
