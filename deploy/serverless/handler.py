#!/usr/bin/env python3
"""RunPod Serverless handler for the HY-World light (4090) tier.

Request -> stage -> artifact to S3. Designed for the flex 4090 worker whose
weights are baked into the image (see Dockerfile / weights-bake-manifest.json),
so it scales to zero and cold-starts by loading models from local layers.

Input schema (job["input"]):
    {
      "jobId": "nm-a",                # required
      "stage": "traj_generate",       # one of LIGHT_STAGES (default traj_generate)
      "prompt": "...",                # for seed/panorama inputs (optional per stage)
      "sceneS3": "s3://.../scene/",   # optional: prior scene state to restore
      "outputS3Prefix": "worldgen-serverless/{jobId}",
      "width": 1920, "height": 960    # resolution guardrail inputs
    }

GUARDRAIL (A100-only-for-4K): this worker refuses 4K jobs and the heavy stages
(panorama diffusion, WorldStereo video_gen) with a structured error telling the
caller to use the A100 pod path. A 24 GB 4090 never silently OOMs on 4K.
"""
from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

# Stages this 24 GB light tier is allowed to run.
LIGHT_STAGES = {"traj_generate", "traj_render", "gen_gs_data", "gs_train"}
# Stages that require 80 GB / multi-GPU — always routed to the A100 pod.
HEAVY_STAGES = {"panorama", "video_gen"}
# 4K guardrail: total pixels at/above this route to A100.
FOURK_PIXELS = 3840 * 2160
ACTIVITY_FILE = os.environ.get("ACTIVITY_FILE", "/tmp/worldgen-activity")


def _touch_activity() -> None:
    try:
        Path(ACTIVITY_FILE).touch()
    except OSError:
        pass


def _guardrail(stage: str, width: int, height: int) -> dict | None:
    """Return an error dict if this job must not run on the 24 GB light tier."""
    if stage in HEAVY_STAGES:
        return {
            "error": "stage_requires_a100",
            "message": f"Stage '{stage}' needs an 80 GB GPU (WorldStereo/Qwen). "
                       "Route to the RunPod pod A100 path (runpod_launch.py).",
            "route": "a100-pod",
        }
    if width * height >= FOURK_PIXELS:
        return {
            "error": "resolution_requires_a100",
            "message": f"{width}x{height} >= 4K exceeds the 24 GB 4090 budget. "
                       "A100-only-for-4K guardrail: route to the A100 pod path.",
            "route": "a100-pod",
        }
    if stage not in LIGHT_STAGES:
        return {
            "error": "unknown_stage",
            "message": f"Stage '{stage}' is not a light-tier stage. "
                       f"Allowed: {sorted(LIGHT_STAGES)}.",
            "route": "reject",
        }
    return None


def _run_light_stage(job_id: str, stage: str, scene_dir: Path, inp: dict) -> dict:
    """Execute a light stage by reusing the pod worker's building blocks.

    Imports are done lazily so the handler can be imported (and unit-guarded)
    on a machine without CUDA/redis. On the real worker these resolve to the
    baked pipeline in /app.
    """
    import sys
    sys.path.insert(0, "/app/deploy/worldgen")
    sys.path.insert(0, "/app/hyworld2/worldgen")
    import worker  # noqa: E402  (the pod worker module; provides run_stage etc.)

    wg = Path("/app/hyworld2/worldgen")
    log_path = scene_dir / f"logs/{stage}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if stage == "traj_generate":
        worker.run_stage("traj_generate", [
            "python", "traj_generate.py",
            "--scene_dir", str(scene_dir),
        ], cwd=wg, log_path=log_path)
    elif stage == "traj_render":
        worker.run_stage("traj_render", [
            "python", "traj_render.py", "--scene_dir", str(scene_dir),
        ], cwd=wg, log_path=log_path)
    elif stage == "gen_gs_data":
        worker.run_stage("gen_gs_data", [
            "python", "gen_gs_data.py", "--scene_dir", str(scene_dir),
        ], cwd=wg, log_path=log_path)
    elif stage == "gs_train":
        worker.run_stage("gs_train", [
            "python", "world_gs_trainer.py", "--scene_dir", str(scene_dir),
        ], cwd=wg, log_path=log_path)

    prefix = inp.get("outputS3Prefix") or f"worldgen-serverless/{job_id}"
    result_dir = scene_dir / "result"
    artifacts = worker.upload_outputs(job_id, scene_dir, result_dir, prefix)
    return {"stage": stage, "outputS3Prefix": prefix, "artifacts": artifacts}


def handler(job: dict) -> dict:
    started = time.time()
    inp = (job or {}).get("input") or {}
    job_id = inp.get("jobId") or job.get("id") or f"srv-{int(started)}"
    stage = inp.get("stage", "traj_generate")
    width = int(inp.get("width", 1920))
    height = int(inp.get("height", 960))

    guard = _guardrail(stage, width, height)
    if guard is not None:
        return {**guard, "jobId": job_id, "stage": stage}

    _touch_activity()
    scene_dir = Path(os.environ.get("SCENES_DIR", "/workspace/scenes")) / job_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Keep the activity marker fresh for a colocated idle watchdog.
        import threading
        stop = threading.Event()

        def _beat():
            while not stop.wait(30):
                _touch_activity()

        threading.Thread(target=_beat, daemon=True).start()
        out = _run_light_stage(job_id, stage, scene_dir, inp)
        stop.set()
        out.update({"jobId": job_id, "ok": True, "seconds": round(time.time() - started, 1)})
        return out
    except Exception as exc:  # structured failure, never leak secrets
        return {
            "jobId": job_id,
            "stage": stage,
            "ok": False,
            "error": "stage_failed",
            "message": str(exc)[:500],
            "trace": traceback.format_exc(limit=3)[-1500:],
            "seconds": round(time.time() - started, 1),
        }


if __name__ == "__main__":
    # Local sanity path: `python handler.py --selftest` exercises the guardrail
    # without any GPU / RunPod SDK (used by the smoke test).
    import sys
    if "--selftest" in sys.argv:
        cases = [
            {"input": {"jobId": "t1", "stage": "video_gen"}},
            {"input": {"jobId": "t2", "stage": "traj_generate", "width": 3840, "height": 2160}},
            {"input": {"jobId": "t3", "stage": "bogus"}},
        ]
        for c in cases:
            print(json.dumps(handler(c)))
        sys.exit(0)

    import runpod  # provided by the base image / pip
    runpod.serverless.start({"handler": handler})
