"""Validation probe for the Real-ESRGAN SR stage.

Pulls the coherent 1952x960 base pano, runs Real-ESRGAN -> 3840x1920, and uploads the
result + a Lanczos baseline for A/B. Confirms the SR adds detail while preserving exact
composition (no ghosting/hallucination) before spending on a full higher-res world run.
"""

import functools
import json
import os
import time
import traceback
from pathlib import Path

import boto3
from PIL import Image

from realesrgan_sr import esrgan_super_resolve

print = functools.partial(print, flush=True)  # noqa: A001

BUCKET = os.environ["MODEL_BUCKET"]
REGION = os.environ.get("AWS_REGION", "us-east-1")
BASE_KEY = os.environ.get("PROBE_BASE_KEY", "worldgen-full-ops/probe/pano-res/pano_1952x960.png")
OUT_PREFIX = os.environ.get("PROBE_OUT_PREFIX", "worldgen-full-ops/probe/esrgan")


def main() -> None:
    s3 = boto3.client("s3", region_name=REGION)
    base_local = "/tmp/base_pano.png"
    s3.download_file(BUCKET, BASE_KEY, base_local)
    base = Image.open(base_local).convert("RGB")
    print(f"[esrgan-probe] base {base.size} from s3://{BUCKET}/{BASE_KEY}")

    base.resize((3840, 1920), Image.LANCZOS).save("/tmp/lanczos.png")
    s3.upload_file("/tmp/lanczos.png", BUCKET, f"{OUT_PREFIX}/lanczos_3840x1920.png")

    t0 = time.time()
    try:
        out = esrgan_super_resolve(base, 3840, 1920)
    except Exception:
        print("[esrgan-probe] SR FAILED:\n" + traceback.format_exc())
        raise
    elapsed = time.time() - t0
    out.save("/tmp/esrgan.png")
    s3.upload_file("/tmp/esrgan.png", BUCKET, f"{OUT_PREFIX}/esrgan_3840x1920.png")

    report = {"baseKey": BASE_KEY, "seconds": round(elapsed, 1),
              "output": f"{OUT_PREFIX}/esrgan_3840x1920.png"}
    Path("/tmp/esrgan-report.json").write_text(json.dumps(report, indent=2))
    s3.upload_file("/tmp/esrgan-report.json", BUCKET, f"{OUT_PREFIX}/report.json")
    print(f"[esrgan-probe] done in {elapsed:.1f}s -> s3://{BUCKET}/{OUT_PREFIX}/")


if __name__ == "__main__":
    main()
