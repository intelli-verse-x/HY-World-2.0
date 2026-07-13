#!/usr/bin/env python3
"""Download the light WorldNav vision tier into the image build context so the
Dockerfile can COPY it into an image layer (weights BAKED IN, not mounted).

Reads deploy/serverless/weights-bake-manifest.json and pulls each `baked: true`
model snapshot (pinned revision) into $OUT (default /opt/hf-cache, HF hub cache
layout). Runs at IMAGE BUILD time, so the resulting layers ship the weights and
the serverless worker cold-starts by loading from local disk — keeping the
effective serverless cost near the warm rate.

No secrets required for the baked tier (all ungated). If a model ever needs
auth, pass HF_TOKEN in the build env; it is used only by huggingface_hub and is
never written into a layer.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

MANIFEST = Path(__file__).with_name("weights-bake-manifest.json")


def main() -> int:
    out = Path(os.environ.get("OUT", "/opt/hf-cache"))
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text())
    bake = manifest["bake"]

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed in build stage", file=sys.stderr)
        return 1

    total = 0.0
    for repo_id, spec in bake.items():
        if not spec.get("baked"):
            continue
        rev = spec.get("revision")
        print(f"[bake] {repo_id}@{(rev or 'main')[:12]} (~{spec.get('approxGb', '?')} GB)")
        snapshot_download(
            repo_id=repo_id,
            revision=rev,
            cache_dir=str(out),
            token=os.environ.get("HF_TOKEN") or None,
        )
        total += float(spec.get("approxGb", 0) or 0)
    print(f"[bake] done; ~{total:.1f} GB staged under {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
