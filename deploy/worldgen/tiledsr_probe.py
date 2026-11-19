"""
Validation probe for the tiled equirect SR pipeline.

Pulls the coherent 1952x960 base pano produced by the resolution probe, runs the
tiled super-resolver to 3840x1920, and uploads both the tiled-SR result and a
plain-Lanczos upscale of the same base for A/B. The goal is to confirm visually
that tiled SR adds genuine detail WITHOUT reintroducing composition duplication
or visible tile seams, before spending on a full higher-res world run.
"""

import functools
import json
import os
import time
import traceback
from pathlib import Path

import boto3
from PIL import Image

from pano_tiled_sr import tiled_super_resolve  # same dir on the pod

print = functools.partial(print, flush=True)  # noqa: A001

BUCKET = os.environ["MODEL_BUCKET"]
REGION = os.environ.get("AWS_REGION", "us-east-1")
BASE_KEY = os.environ.get("PROBE_BASE_KEY", "worldgen-full-ops/probe/pano-res/pano_1952x960.png")
OUT_PREFIX = os.environ.get("PROBE_OUT_PREFIX", "worldgen-full-ops/probe/tiled-sr")
PROMPT = os.environ.get(
    "PROBE_PROMPT",
    "Night market alley, food stalls, ICE-blue hanging lantern, neon shop signs, "
    "wet stone paving, dense cyberpunk Asian market at blue hour",
)


def main() -> None:
    s3 = boto3.client("s3", region_name=REGION)
    base_local = "/tmp/base_pano.png"
    s3.download_file(BUCKET, BASE_KEY, base_local)
    base = Image.open(base_local).convert("RGB")
    print(f"[tiledsr-probe] base {base.size} from s3://{BUCKET}/{BASE_KEY}")

    lanczos = base.resize((3840, 1920), Image.LANCZOS)
    lanczos.save("/tmp/lanczos_3840x1920.png")
    s3.upload_file("/tmp/lanczos_3840x1920.png", BUCKET, f"{OUT_PREFIX}/lanczos_3840x1920.png")

    t0 = time.time()
    try:
        out = tiled_super_resolve(base, 3840, 1920, PROMPT, tile=1024, overlap=256, steps=20)
    except Exception:
        print("[tiledsr-probe] tiled SR FAILED:\n" + traceback.format_exc())
        raise
    elapsed = time.time() - t0
    out.save("/tmp/tiledsr_3840x1920.png")
    s3.upload_file("/tmp/tiledsr_3840x1920.png", BUCKET, f"{OUT_PREFIX}/tiledsr_3840x1920.png")

    report = {
        "baseKey": BASE_KEY,
        "target": "3840x1920",
        "tiledSrSeconds": round(elapsed, 1),
        "outputs": {
            "tiledSr": f"{OUT_PREFIX}/tiledsr_3840x1920.png",
            "lanczos": f"{OUT_PREFIX}/lanczos_3840x1920.png",
        },
    }
    Path("/tmp/tiledsr-report.json").write_text(json.dumps(report, indent=2))
    s3.upload_file("/tmp/tiledsr-report.json", BUCKET, f"{OUT_PREFIX}/report.json")
    print(f"[tiledsr-probe] done in {elapsed:.1f}s -> s3://{BUCKET}/{OUT_PREFIX}/")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
