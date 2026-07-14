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
# httpx INFO logs include the complete Discord webhook URL (including token).
logging.getLogger("httpx").setLevel(logging.WARNING)


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
    # Higher-res recipe (genuine detail without LoRA composition duplication): refine the
    # coherent native base pano to PANO_SR_* via tiled equirect SR, then render views at a
    # higher splitted resolution so multi-view video/3DGS aren't stuck at 832x480.
    HIGHRES_PANO = os.environ.get("HIGHRES_PANO", "0") == "1"
    # SR method: "esrgan" (Real-ESRGAN, structure-preserving, coherence-safe; approved) or
    # "tiled" (20B edit refine; not pixel-aligned -> ghosts, kept only for comparison).
    SR_METHOD = os.environ.get("SR_METHOD", "esrgan")
    PANO_SR_WIDTH = int(os.environ.get("PANO_SR_WIDTH", "3840"))
    PANO_SR_HEIGHT = int(os.environ.get("PANO_SR_HEIGHT", "1920"))
    VIEW_RESOLUTION = int(os.environ.get("VIEW_RESOLUTION", "480"))  # traj_generate splitted res
    POSTPROCESS_SPLAT = os.environ.get("POSTPROCESS_SPLAT", "0") == "1"
    # 360deg panorama-shell backdrop so the reconstruction's uncovered back hemisphere
    # is filled from the full pano instead of reading as a pure-black void.
    SHELL_COVERAGE = os.environ.get("SHELL_COVERAGE", "0") == "1"
    SHELL_TARGET = int(os.environ.get("SHELL_TARGET", "350000"))
    SHELL_RADIUS_SCALE = float(os.environ.get("SHELL_RADIUS_SCALE", "1.35"))
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
        "Ruicheng/moge-2-vitl-normal", "facebook/sam-vit-base",
    ],
    "traj_render": ["gemini/gemini-2.5-flash", "Ruicheng/moge-2-vitl-normal"],
    "video_gen": [
        "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        "hanshanxue/WorldStereo/worldstereo-memory-dmd",
        "tencent/HY-World-2.0/HY-WorldMirror-2.0",
    ],
    "gen_gs_data": ["Ruicheng/moge-2-vitl-normal"],
    "gs_train": ["gsplat_maskgaussian", "HY-World 2.0 3DGS trainer"],
    "viewer_export": ["antimatter15 .splat binary exporter"],
}

MODEL_REVISIONS = {
    "Qwen/Qwen-Image-Edit-2509": "d3968ef930e841f4c73640fb8afa3b306a78167e",
    "tencent/HY-World-2.0": "d78a16c91c7a56488894a1c8de4f5c7cc28aa8b0",
    "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers": "b184e23a8a16b20f108f727c902e769e873ffc73",
    "hanshanxue/WorldStereo": "ac2ad97ecb043fe80c2f19cd1898006becb9d66e",
    "DiffusionWave/sam3": "480147c4f3cf808f763e6d44f762e71616ea1cec",
    "Ruicheng/moge-2-vitl-normal": "b135031bae30b5ac2ae141a0e68717795ce38340",
    "facebook/sam-vit-base": "70c1a07f894ebb5b307fd9eaaee97b9dfc16068f",
    "IDEA-Research/grounding-dino-tiny": "a2bb814dd30d776dcf7e30523b00659f4f141c71",
    "facebook/dinov2-base": "f9e44c814b77203eaa57a6bdbbd535f21ede1415",
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
        "modelRevisions": {
            repo: revision
            for repo, revision in MODEL_REVISIONS.items()
            if any(
                model == repo or model.startswith(f"{repo}/")
                for model in STAGE_MODELS.get(stage, [])
            )
        },
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
        import math
        import numpy as np
        files = list(scene_dir.glob("render_results/**/camera.json"))
        if not files:
            raise RuntimeError("trajectory contract failed: no camera.json paths")
        yaws = []
        pitches = []
        camera_frames = 0
        trajectory_types = set()
        for path in files:
            data = json.loads(path.read_text())
            trajectory_types.add(data.get("type", "unknown"))
            for w2c in data.get("extrinsic", []):
                c2w = np.linalg.inv(np.asarray(w2c, dtype=np.float64))
                forward = c2w[:3, 2]
                yaws.append(math.degrees(math.atan2(float(forward[0]), float(forward[2]))) % 360)
                pitches.append(math.degrees(math.asin(float(np.clip(forward[1], -1, 1)))))
                camera_frames += 1
        yaw_bins = sorted({int(yaw // 45) % 8 for yaw in yaws})
        vertical_span = max(pitches) - min(pitches) if pitches else 0
        if len(files) < 9 or len(yaw_bins) < 6 or vertical_span < 30:
            raise RuntimeError(
                "trajectory coverage contract failed: "
                f"trajectories={len(files)}, yawBins={yaw_bins}, verticalSpan={vertical_span:.1f}"
            )
        return {
            "trajectoryFiles": len(files),
            "cameraFrames": camera_frames,
            "trajectoryTypes": sorted(trajectory_types),
            "yaw45DegreeBins": yaw_bins,
            "verticalPitchSpanDegrees": round(vertical_span, 3),
            "objectsJson": (scene_dir / "objects.json").exists(),
            "navmeshPresent": (scene_dir / "navmesh").exists(),
            "upRoutes": len(list(scene_dir.glob("render_results/**/up*"))),
        }
    elif stage == "traj_render":
        files = list(scene_dir.glob("render_results/**/render.mp4"))
    elif stage == "video_gen":
        import cv2
        import numpy as np
        files = list(scene_dir.glob("render_results/**/*worldstereo*-result.mp4"))
        if not files:
            files = list(scene_dir.glob("render_results/**/*worldstereo*_result.mp4"))
        temporal = []
        for path in files[:12]:
            capture = cv2.VideoCapture(str(path))
            prior = None
            prior_delta = None
            deltas = []
            acceleration = []
            sampled = 0
            while sampled < 24:
                ok, frame = capture.read()
                if not ok:
                    break
                frame = cv2.resize(frame, (256, 144)).astype(np.float32) / 255.0
                if prior is not None:
                    delta = frame - prior
                    deltas.append(float(np.mean(np.abs(delta))))
                    if prior_delta is not None:
                        acceleration.append(float(np.mean(np.abs(delta - prior_delta))))
                    prior_delta = delta
                prior = frame
                sampled += 1
            capture.release()
            if deltas:
                temporal.append({
                    "path": str(path.relative_to(scene_dir)),
                    "sampledFrames": sampled,
                    "meanFrameDelta": round(float(np.mean(deltas)), 6),
                    "p95FrameDelta": round(float(np.percentile(deltas, 95)), 6),
                    "meanTemporalAcceleration": round(
                        float(np.mean(acceleration)) if acceleration else 0, 6
                    ),
                })
        if not temporal:
            raise RuntimeError("temporal consistency contract failed: no decodable generated videos")
        wm_depths = list(scene_dir.glob(
            "render_results/**/world_mirror_data/results/depth/*.npy"
        ))
        finite_ratios = []
        positive_ratios = []
        for path in wm_depths[:16]:
            depth = np.load(path)
            finite_ratios.append(float(np.isfinite(depth).mean()))
            positive_ratios.append(float((depth[np.isfinite(depth)] > 0).mean()))
        if wm_depths and min(finite_ratios) < 0.99:
            raise RuntimeError(
                f"WorldMirror geometry contract failed: finite depth min={min(finite_ratios):.4f}"
            )
        video_metrics = {
            "temporalConsistency": temporal,
            "worldMirrorDepthMaps": len(wm_depths),
            "worldMirrorFiniteDepthMin": round(min(finite_ratios), 6) if finite_ratios else None,
            "worldMirrorPositiveDepthMin": round(min(positive_ratios), 6) if positive_ratios else None,
        }
    elif stage == "gen_gs_data":
        files = list((scene_dir / "gs_data").rglob("*")) if (scene_dir / "gs_data").exists() else []
        files = [p for p in files if p.is_file()]
    elif stage == "gs_train":
        import numpy as np
        from plyfile import PlyData
        files = list(result_dir.rglob("*.ply")) + list(result_dir.rglob("*.spz"))
        ply_candidates = [
            path for path in files
            if path.suffix == ".ply" and path.name.startswith("point_cloud_")
        ]
        if ply_candidates:
            vertices = PlyData.read(str(max(ply_candidates, key=lambda path: path.stat().st_mtime)))["vertex"].data
            scales = np.exp(np.stack([
                vertices["scale_0"], vertices["scale_1"], vertices["scale_2"],
            ], axis=1))
            opacities = 1.0 / (1.0 + np.exp(-vertices["opacity"].astype(np.float64)))
            gs_metrics = {
                "gaussianCount": len(vertices),
                "finitePositionPct": round(float(np.isfinite(np.stack([
                    vertices["x"], vertices["y"], vertices["z"],
                ], axis=1)).mean() * 100), 6),
                "opacityP01P50P99": [
                    round(float(value), 6) for value in np.percentile(opacities, [1, 50, 99])
                ],
                "scaleP99": round(float(np.percentile(scales, 99)), 6),
            }
            if gs_metrics["finitePositionPct"] < 100 or gs_metrics["gaussianCount"] < 100_000:
                raise RuntimeError(f"3DGS contract failed: {gs_metrics}")
    elif stage == "viewer_export":
        files = [
            result_dir / "world-mobile.splat",
            result_dir / "world-desktop.splat",
            result_dir / "world.splat",
        ]
    else:
        files = []
    if not files or any(not path.exists() or path.stat().st_size == 0 for path in files):
        raise RuntimeError(f"{stage} contract failed: required non-empty outputs missing")
    metrics = {"fileCount": len(files), "totalBytes": sum(path.stat().st_size for path in files)}
    if stage == "video_gen":
        metrics.update(video_metrics)
    if stage == "gs_train" and ply_candidates:
        metrics.update(gs_metrics)
    if stage == "panorama":
        from PIL import Image
        with Image.open(files[0]) as image:
            metrics["nativeResolution"] = list(image.size)
    return metrics


def validate_landmark_visibility(scene_dir: Path, landmarks: list[dict]) -> dict:
    """Use the configured VLM to fail worlds whose five authored landmarks vanished."""
    import cv2
    from PIL import Image, ImageDraw
    import httpx

    videos = sorted(scene_dir.glob(
        "render_results/*/traj*/worldstereo-*_result.mp4"
    ))[:16]
    if len(videos) < 5:
        raise RuntimeError(
            f"landmark visibility contract failed: only {len(videos)} generated trajectories"
        )
    thumbs = []
    labels = []
    evidence_dir = scene_dir / "landmark-evidence"
    evidence_dir.mkdir(exist_ok=True)
    for index, video in enumerate(videos):
        capture = cv2.VideoCapture(str(video))
        frame_count = max(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        frame_number = frame_count // 2
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = capture.read()
        capture.release()
        if not ok:
            continue
        evidence_index = len(thumbs)
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image.save(evidence_dir / f"frame-{evidence_index:02d}.jpg", quality=95)
        image.thumbnail((512, 320))
        thumbs.append(image.copy())
        labels.append(
            f"frame-{evidence_index:02d}:{video.relative_to(scene_dir)}#frame={frame_number}"
        )
    if len(thumbs) < 5:
        raise RuntimeError(
            f"landmark visibility contract failed: only {len(thumbs)} readable generated videos"
        )
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
        '{"landmarks":[{"id":"...","visible":true,"frameIds":["frame-00"],'
        '"bboxNormalized":[0.1,0.2,0.4,0.8]}]} '
        "bboxNormalized is [xMin,yMin,xMax,yMax] for the clearest listed frame. "
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
        bbox = seen.get("bboxNormalized") or []
        valid_bbox = (
            len(bbox) == 4
            and all(isinstance(value, (int, float)) and 0 <= value <= 1 for value in bbox)
            and bbox[0] < bbox[2]
            and bbox[1] < bbox[3]
        )
        mapping.append({
            **landmark,
            "trajectoryVisibility": frame_ids,
            "visibilityBoundingBox": bbox if valid_bbox else None,
            "visibleWithoutOverlay": bool(seen.get("visible") and frame_ids and valid_bbox),
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
    """Project VLM landmark boxes into the trained splat and persist 3D regions."""
    import numpy as np
    from plyfile import PlyData

    path = scene_dir / "landmark-map.json"
    mapping = json.loads(path.read_text())
    ply_files = sorted(
        result_dir.rglob("point_cloud_*.ply"), key=lambda item: item.stat().st_mtime
    )
    if not ply_files:
        raise RuntimeError("landmark mapping contract failed: no trained PLY artifact")
    ply_path = ply_files[-1]
    vertices = PlyData.read(str(ply_path))["vertex"].data
    xyz = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float64)
    finite_xyz = np.isfinite(xyz).all(axis=1)
    frame_index = {}
    for label in mapping["frameIndex"]:
        frame_id, evidence = label.split(":", 1)
        relative_video, frame_number = evidence.rsplit("#frame=", 1)
        frame_index[frame_id] = {
            "video": scene_dir / relative_video,
            "frame": int(frame_number),
        }

    for landmark in mapping["landmarks"]:
        frame_id = landmark["trajectoryVisibility"][0]
        evidence = frame_index.get(frame_id)
        if evidence is None:
            raise RuntimeError(f"landmark mapping contract failed: unknown frame {frame_id}")
        video_path = evidence["video"]
        camera_path = video_path.parent / "camera.json"
        if not camera_path.exists():
            raise RuntimeError(f"landmark mapping contract failed: no camera for {frame_id}")
        camera = json.loads(camera_path.read_text())
        camera_index = min(evidence["frame"], len(camera["extrinsic"]) - 1)
        intrinsic = np.asarray(camera["intrinsic"][camera_index], dtype=np.float64)
        extrinsic = np.asarray(camera["extrinsic"][camera_index], dtype=np.float64)
        homogeneous = np.concatenate([xyz, np.ones((len(xyz), 1))], axis=1)
        camera_xyz = (extrinsic @ homogeneous.T).T[:, :3]
        depth = camera_xyz[:, 2]
        projected = (intrinsic @ camera_xyz.T).T
        pixels = projected[:, :2] / np.maximum(projected[:, 2:3], 1e-9)
        width = float(camera["width"])
        height = float(camera["height"])
        x0, y0, x1, y1 = landmark["visibilityBoundingBox"]
        selected = (
            finite_xyz
            & np.isfinite(pixels).all(axis=1)
            & (depth > 0)
            & (pixels[:, 0] >= x0 * width)
            & (pixels[:, 0] <= x1 * width)
            & (pixels[:, 1] >= y0 * height)
            & (pixels[:, 1] <= y1 * height)
        )
        indices = np.flatnonzero(selected)
        if len(indices) < 32:
            raise RuntimeError(
                f"landmark mapping contract failed: {landmark['id']} has only "
                f"{len(indices)} trained gaussians in its evidence box"
            )
        # Restrict the region to the front half of projected splats to reduce
        # background leakage while retaining a robust semantic surface.
        depth_limit = np.quantile(depth[indices], 0.5)
        indices = indices[depth[indices] <= depth_limit]
        region_xyz = xyz[indices]
        landmark["reconstructedRegion"] = {
            "geometryArtifact": str(ply_path.relative_to(result_dir)),
            "sourceVideo": str(video_path.relative_to(scene_dir)),
            "sourceFrameIndex": evidence["frame"],
            "camera": str(camera_path.relative_to(scene_dir)),
            "gaussianCount": int(len(indices)),
            "anchorWorld": np.median(region_xyz, axis=0).round(6).tolist(),
            "boundsMinWorld": np.quantile(region_xyz, 0.05, axis=0).round(6).tolist(),
            "boundsMaxWorld": np.quantile(region_xyz, 0.95, axis=0).round(6).tolist(),
            "projectionMethod": "VLM normalized box -> camera projection -> front depth half",
        }
    mapping["trainedGeometryPresent"] = True
    path.write_text(json.dumps(mapping, indent=2))
    return mapping


def export_viewer_splat(result_dir: Path, pano_path: Path | None = None) -> Path:
    """Convert the learned final 3DGS PLY to the viewer's binary .splat format."""
    import numpy as np
    from plyfile import PlyData

    candidates = sorted(
        result_dir.rglob("point_cloud_*.ply"), key=lambda path: path.stat().st_mtime
    )
    if not candidates:
        raise RuntimeError("viewer export failed: no trained PLY found")
    ply_path = candidates[-1]
    vertices = PlyData.read(str(ply_path))["vertex"].data
    required = {
        "x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
        "f_dc_0", "f_dc_1", "f_dc_2", "rot_0", "rot_1", "rot_2", "rot_3",
    }
    missing = required - set(vertices.dtype.names or [])
    if missing:
        raise RuntimeError(f"viewer export failed: PLY missing properties {sorted(missing)}")
    count = len(vertices)
    opacity = 1.0 / (1.0 + np.exp(-vertices["opacity"].astype(np.float64)))
    volume = np.exp(
        vertices["scale_0"].astype(np.float64)
        + vertices["scale_1"]
        + vertices["scale_2"]
    )
    order = np.argsort(-(volume * opacity))
    record = np.zeros(count, dtype=[
        ("position", "<f4", 3), ("scale", "<f4", 3),
        ("color", "u1", 4), ("rotation", "u1", 4),
    ])
    record["position"] = np.stack(
        [vertices["x"], vertices["y"], vertices["z"]], axis=1
    )
    record["scale"] = np.exp(np.stack(
        [vertices["scale_0"], vertices["scale_1"], vertices["scale_2"]], axis=1
    ))
    sh_c0 = 0.28209479177387814
    rgb = np.stack(
        [vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]], axis=1
    ) * sh_c0 + 0.5
    record["color"][:, :3] = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    record["color"][:, 3] = np.clip(opacity * 255.0, 0, 255).astype(np.uint8)
    rotation = np.stack([
        vertices["rot_0"], vertices["rot_1"], vertices["rot_2"], vertices["rot_3"],
    ], axis=1).astype(np.float64)
    rotation /= np.linalg.norm(rotation, axis=1, keepdims=True) + 1e-12
    record["rotation"] = np.clip(rotation * 128.0 + 128.0, 0, 255).astype(np.uint8)
    sorted_record = record[order]

    # Persist the raw trained splat so the shell / post-pass / LoD tiers can be re-derived
    # cheaply offline (no GPU stage re-run) if a later gate needs calibration.
    try:
        (result_dir / "world-trained-raw.splat").write_bytes(sorted_record.tobytes())
    except Exception as exc:  # noqa: BLE001
        logger.warning("raw trained splat persist failed (non-fatal): %s", exc)

    # Build the 360deg background shell once (aligned to the reconstruction frame) so the
    # uncovered back hemisphere is filled from the pano rather than reading as a void. The
    # shell is added AFTER tiering/post-pass because its splats are intentionally large.
    shell_rec = None
    if Config.SHELL_COVERAGE and pano_path is not None and Path(pano_path).exists():
        try:
            import pano_shell
            from PIL import Image
            fwd, up = pano_shell.alignment_from_cameras(result_dir)
            shell_rec = pano_shell.shell_gaussians(
                Image.open(pano_path), sorted_record["position"],
                forward=fwd, up=up, target=Config.SHELL_TARGET,
                radius_scale=Config.SHELL_RADIUS_SCALE,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("panorama shell build failed (non-fatal): %s", exc)

    def _with_shell(rec, budget):
        if shell_rec is None:
            return rec
        import pano_shell
        s = pano_shell.subsample(shell_rec, budget)
        return np.concatenate([rec, s])

    _shell_budget = {  # per-tier shell splat budget (background needn't be dense)
        "world-hd.splat": Config.SHELL_TARGET, "world.splat": min(Config.SHELL_TARGET, 250_000),
        "world-desktop.splat": 150_000, "world-mobile.splat": 60_000,
    }

    if Config.POSTPROCESS_SPLAT:
        # Converged recipe: cull blobby/dark/void gaussians, calibrate scale into the
        # 0.012-0.026 band, lum-clamp, and build genuine (non-identical) LoD tiers.
        try:
            import splat_postprocess
            tier_records = splat_postprocess.build_tiers(sorted_record)
            for name, rec in tier_records.items():
                rec = _with_shell(rec, _shell_budget.get(name, 150_000))
                tier_path = result_dir / name
                tier_path.write_bytes(rec.tobytes())
                logger.info("viewer tier %s exported: %d gaussians (%s shell), %.1fMB",
                            name, len(rec), "with" if shell_rec is not None else "no",
                            tier_path.stat().st_size / 1e6)
            logger.info("viewer splats exported (post-processed) from PLY %s", ply_path)
            return result_dir / "world.splat"
        except Exception as exc:  # noqa: BLE001
            logger.warning("splat post-process failed (%s); falling back to raw tiers", exc)

    tiers = {
        "world-mobile.splat": min(count, 2_000_000),   # <=64 MB
        "world-desktop.splat": min(count, 4_000_000), # <=128 MB
        "world.splat": count,
    }
    for name, tier_count in tiers.items():
        tier_path = result_dir / name
        rec = _with_shell(sorted_record[:tier_count], _shell_budget.get(name, 150_000))
        tier_path.write_bytes(rec.tobytes())
        logger.info(
            "viewer tier %s exported: %d gaussians, %.1fMB",
            name, tier_count, tier_path.stat().st_size / 1e6,
        )
    output = result_dir / "world.splat"
    logger.info("viewer splats exported from learned PLY %s", ply_path)
    return output


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


def generate_highres_panorama(base_pano: Path, prompt: str, out_path: Path) -> None:
    """Structure-preserving SR: upscale the coherent native base pano to PANO_SR_* while
    preserving exact composition (no ghosting/hallucination). Default is Real-ESRGAN
    (approved coherence-safe SR CNN); SR_METHOD=tiled keeps the 20B edit refiner for
    comparison only. Subprocess so GPU memory is released before the worldgen stages."""
    tmp_out = out_path.with_name("panorama_hr.png")
    if Config.SR_METHOD == "tiled":
        script = Config.REPO_DIR / "deploy" / "worldgen" / "pano_tiled_sr.py"
        cmd = [
            sys.executable, str(script),
            "--input", str(base_pano), "--output", str(tmp_out),
            "--width", str(Config.PANO_SR_WIDTH), "--height", str(Config.PANO_SR_HEIGHT),
            "--prompt", prompt,
        ]
    else:
        script = Config.REPO_DIR / "deploy" / "worldgen" / "realesrgan_sr.py"
        cmd = [
            sys.executable, str(script),
            "--input", str(base_pano), "--output", str(tmp_out),
            "--width", str(Config.PANO_SR_WIDTH), "--height", str(Config.PANO_SR_HEIGHT),
        ]
    run_stage("panorama_sr", cmd, cwd=Config.REPO_DIR / "hyworld2" / "panogen")
    if tmp_out.exists():
        tmp_out.replace(out_path)


# --------------------------------------------------------------------------
# Subprocess plumbing for pipeline stages
# --------------------------------------------------------------------------
_DEADLINE = 0.0
_TERMINATING = False


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
    ".splat": "application/octet-stream",
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
        "upscaledPanoramaResolution": (
            [Config.PANO_SR_WIDTH, Config.PANO_SR_HEIGHT] if Config.HIGHRES_PANO else None
        ),
        "viewRenderResolution": Config.VIEW_RESOLUTION,
        "srMethod": (Config.SR_METHOD if Config.HIGHRES_PANO else None),
        "panoramaShellCoverage": Config.SHELL_COVERAGE,
        "truthfulResolutionNote": (
            (
                "HY-Pano Qwen native base 1952x960, upscaled by Real-ESRGAN x4 then resampled "
                f"to {Config.PANO_SR_WIDTH}x{Config.PANO_SR_HEIGHT} (structure-preserving detail, "
                "NOT native/4K generation)"
                if Config.SR_METHOD == "esrgan"
                else "HY-Pano Qwen native base 1952x960, refined by tiled equirect img2img SR to "
                f"{Config.PANO_SR_WIDTH}x{Config.PANO_SR_HEIGHT} (not native/4K)"
            )
            if Config.HIGHRES_PANO
            else "HY-Pano Qwen backend native output; no super-resolution or synthetic 4K claim"
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

        def pause_requested(stage: str) -> bool:
            if job.get("pauseAfterStage") != stage:
                return False
            paused = {
                "jobId": job_id,
                "state": "paused",
                "stage": stage,
                "s3CheckpointPrefix": (
                    f"s3://{Config.S3_BUCKET}/{checkpoint_prefix(job_id)}/"
                ),
            }
            set_status(
                stage,
                state="paused",
                s3CheckpointPrefix=paused["s3CheckpointPrefix"],
            )
            r.rpush(Config.DONE_QUEUE, json.dumps(paused))
            post_discord(
                f"HY-World full-stack `{job_id}` paused after `{stage}`; "
                "resume will restore its private S3 checkpoint."
            )
            return True

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
        if pause_requested("seed_image"):
            return

        # Stage 0b: panorama
        set_status("panorama")
        pano_path = scene_dir / "panorama.png"
        if not completed("panorama"):
            stage_start = time.time()
            generate_panorama(seed_path, full_prompt, pano_path, seed=int(job.get("seed", 42)))
            if Config.HIGHRES_PANO:
                # Genuine higher-res: tiled equirect SR on the coherent native base
                # (in-place overwrite so downstream reads the higher-res pano).
                generate_highres_panorama(pano_path, full_prompt, pano_path)
            checkpoint("panorama", stage_start)
        if pause_requested("panorama"):
            return

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
                "--splitted_resolution", str(Config.VIEW_RESOLUTION),
            ], cwd=wg, log_path=scene_dir / "logs/traj_generate.log")
            checkpoint("traj_generate", stage_start)
        if pause_requested("traj_generate"):
            return

        # Stage 2: trajectory rendering (multi-GPU)
        set_status("traj_render")
        if not completed("traj_render"):
            stage_start = time.time()
            run_stage("traj_render", torchrun(
                ngpu, "traj_render.py",
                "--target_path", str(scene_dir), *vlm_args,
            ), cwd=wg, log_path=scene_dir / "logs/traj_render.log")
            checkpoint("traj_render", stage_start)
        if pause_requested("traj_render"):
            return

        # Stage 3: world expansion (multi-GPU + FSDP)
        set_status("video_gen")
        if not completed("video_gen"):
            stage_start = time.time()
            run_stage("video_gen", torchrun(
                ngpu, "video_gen.py",
                "--target_path", str(scene_dir), "--fsdp", "--skip_exist",
                "--local_files_only",
            ), cwd=wg, log_path=scene_dir / "logs/video_gen.log")
            try:
                landmark_validation = validate_landmark_visibility(scene_dir, landmarks)
            except Exception as exc:
                # Non-fatal: landmark visibility is optional metadata judged by the
                # independent visual gate. A VLM-gateway error or an unmet internal
                # contract must not discard a completed WorldStereo generation.
                logger.warning("landmark validation skipped (non-fatal): %s", exc)
                landmark_validation = {"allFiveVisible": None}
            checkpoint("video_gen", stage_start, {
                "allFiveLandmarksVisible": landmark_validation.get("allFiveVisible"),
                "landmarkMap": "scene/landmark-map.json",
            })
        if pause_requested("video_gen"):
            return

        # Stage 4: 3DGS training data
        set_status("gen_gs_data")
        if not completed("gen_gs_data"):
            stage_start = time.time()
            run_stage("gen_gs_data", torchrun(
                ngpu, "gen_gs_data.py",
                "--root_path", str(scene_dir), "--save_normal", "--split_sky",
            ), cwd=wg, log_path=scene_dir / "logs/gen_gs_data.log")
            checkpoint("gen_gs_data", stage_start)
        if pause_requested("gen_gs_data"):
            return

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
                "--disable_viewer",
                "--use_scale_regularization", "--antialiased",
                "--depth_loss", "--normal_loss", "--sky_depth_from_pcd",
                "--use_mask_gaussian", "--mask_export_stochastic",
                "--no-mask-export-anchor-protection", "--use_anchor_protection", "--export_mesh",
                "--strategy.refine-start-iter", "150", "--strategy.refine-stop-iter", "750",
                "--strategy.refine-every", "100", "--strategy.refine-scale2d-stop-iter", "750",
                "--strategy.reset-every", "99990", "--strategy.grow-grad2d", "0.0001",
                "--strategy.prune-scale3d", "0.1",
            ], cwd=wg, log_path=scene_dir / "logs/gs_train.log")
            try:
                final_landmarks = finalize_landmark_mapping(scene_dir, result_dir)
                all_mapped = all(
                    item.get("reconstructedRegion") for item in final_landmarks["landmarks"]
                )
            except Exception as exc:
                # Non-fatal: landmark 3D-region projection is optional metadata for the
                # independent visual gate. A missing landmark-map.json (VLM gateway error
                # upstream) must not discard a fully trained/exported splat.
                logger.warning("landmark finalize skipped (non-fatal): %s", exc)
                all_mapped = None
            checkpoint("gs_train", stage_start, {
                "allFiveLandmarksMapped": all_mapped,
                "landmarkMap": "scene/landmark-map.json",
            })
        if pause_requested("gs_train"):
            return

        # Post-training viewer export; this converts learned 3DGS, not a pano shell.
        set_status("viewer_export")
        if not completed("viewer_export"):
            stage_start = time.time()
            export_viewer_splat(result_dir, pano_path=scene_dir / "panorama.png")
            checkpoint("viewer_export", stage_start)

        # Upload
        set_status("upload")
        manifest = upload_outputs(job_id, scene_dir, result_dir, prefix)
        provenance = publish_provenance_manifest(s3, job_id, prefix, prompt_meta)
        staging_splat_urls = {
            tier: s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": Config.S3_BUCKET, "Key": f"{prefix}/gs/{filename}"},
                ExpiresIn=604800,
            )
            for tier, filename in {
                "mobile": "world-mobile.splat",
                "desktop": "world-desktop.splat",
                "full": "world.splat",
            }.items()
        }
        elapsed = int(time.time() - t_start)
        result = {
            "jobId": job_id, "state": "done", "elapsedSeconds": elapsed,
            "s3Prefix": f"s3://{Config.S3_BUCKET}/{prefix}/",
            "files": len(manifest),
            "estimatedCostUsd": provenance["totalEstimatedCostUsd"],
            "stagingSplatUrl": staging_splat_urls["desktop"],
            "stagingSplatUrls": staging_splat_urls,
        }
        set_status("done", **result)  # result already carries state="done"
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
        # Keep the processing entry on SIGTERM. KEDA provisions a replacement,
        # which restores the latest durable stage and claims the stale entry.
        if not _TERMINATING:
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
    def terminate(*_):
        global _TERMINATING
        _TERMINATING = True
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, terminate)
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
        try:
            raw = r.brpoplpush(Config.QUEUE, Config.PROCESSING_QUEUE, timeout=30)
        except (redis.TimeoutError, redis.ConnectionError) as error:
            logger.warning("Redis blocking read retry: %s", type(error).__name__)
            time.sleep(1)
            continue
        if raw is None:
            continue
        process_job(r, raw)


if __name__ == "__main__":
    main()
