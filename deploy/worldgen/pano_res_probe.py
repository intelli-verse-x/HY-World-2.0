"""
HY-Pano higher-resolution feasibility probe (Stage 0b feasibility only).

Loads the HY-Pano 2.0 (Qwen-Image-Edit + LoRA) pipeline ONCE and generates the
same seed image at several equirectangular resolutions, recording wall-clock and
peak GPU memory per resolution and uploading the outputs for visual inspection.

This exists to answer, empirically and cheaply, whether HY-Pano can natively
produce genuinely higher-resolution panoramas (the founder's "break the 1952x960
ceiling" objective) or whether it degrades into DiT extrapolation artifacts
outside the LoRA's trained resolution — and what VRAM/time a higher-res run costs.

It never prints secrets. Reads the seed image and writes outputs via the pod's
injected AWS credential chain.
"""

import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "hyworld2" / "panogen"))

from pipeline_with_qwen_image import (  # noqa: E402
    GENERAL_NEGATIVE_PROMPT,
    GENERAL_POSITIVE_PREFIX,
    GENERAL_POSITIVE_SUFFIX,
    circular_blend_edges,
)
from qwen_image import PanoDiffusionPipeline  # noqa: E402

import boto3  # noqa: E402
from PIL import Image  # noqa: E402

BUCKET = os.environ["MODEL_BUCKET"]
REGION = os.environ.get("AWS_REGION", "us-east-1")
SEED_KEY = os.environ.get("PROBE_SEED_KEY", "worldgen-full-staging/full-nm-a/seed_image.png")
OUT_PREFIX = os.environ.get("PROBE_OUT_PREFIX", "worldgen-full-ops/probe/pano-res")
STEPS = int(os.environ.get("PROBE_STEPS", "40"))
PROMPT = os.environ.get(
    "PROBE_PROMPT",
    "Night market alley, food stalls, ICE-blue hanging lantern, neon shop signs, "
    "and wet stone paving define a dense cyberpunk Asian market at blue hour.",
)
# (width, height) — all divisible by 16 for the VAE/patch grid. Baseline first as a
# reference, then progressively higher native targets.
RESOLUTIONS = [
    (1952, 960),
    (2688, 1344),
    (3840, 1920),
]


def main() -> None:
    s3 = boto3.client("s3", region_name=REGION)
    seed_path = "/tmp/probe_seed.png"
    s3.download_file(BUCKET, SEED_KEY, seed_path)
    print(f"[probe] seed image downloaded from s3://{BUCKET}/{SEED_KEY}")

    print("[probe] loading Qwen-Image-Edit-2509 + HY-Pano-2.0 LoRA (CPU offload)")
    pipe = PanoDiffusionPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.bfloat16
    )
    pipe.load_lora_weights(
        "tencent/HY-World-2.0",
        subfolder="HY-Pano-2.0",
        weight_name="pytorch_lora_weights.safetensors",
        torch_dtype=torch.bfloat16,
    )
    pipe.fuse_lora(lora_scale=1.0)
    pipe.unload_lora_weights()
    pipe.enable_model_cpu_offload()
    print("[probe] pipeline ready")

    image = Image.open(seed_path).convert("RGB")
    positive = (GENERAL_POSITIVE_PREFIX + PROMPT + GENERAL_POSITIVE_SUFFIX).strip()

    report = {
        "seedKey": SEED_KEY,
        "steps": STEPS,
        "gpu": torch.cuda.get_device_name(0),
        "gpuTotalMemGiB": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
        "results": [],
    }

    for (w, h) in RESOLUTIONS:
        entry = {"width": w, "height": h, "megapixels": round(w * h / 1e6, 2)}
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            out = pipe(
                image=image,
                prompt=positive,
                negative_prompt=GENERAL_NEGATIVE_PROMPT,
                generator=torch.Generator(device="cpu").manual_seed(4201),
                true_cfg_scale=7.5,
                num_inference_steps=STEPS,
                guidance_scale=1.0,
                num_images_per_prompt=1,
                height=h,
                width=w,
            ).images[0]
            out = circular_blend_edges(out, 32)
            elapsed = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 2**30
            fname = f"pano_{w}x{h}.png"
            local = f"/tmp/{fname}"
            out.save(local)
            s3.upload_file(local, BUCKET, f"{OUT_PREFIX}/{fname}")
            entry.update({
                "ok": True,
                "seconds": round(elapsed, 1),
                "peakVramGiB": round(peak, 1),
                "outputKey": f"{OUT_PREFIX}/{fname}",
            })
            print(f"[probe] {w}x{h} OK: {elapsed:.1f}s, peak {peak:.1f} GiB -> {fname}")
        except Exception as exc:  # noqa: BLE001
            entry.update({"ok": False, "error": str(exc)[:500]})
            print(f"[probe] {w}x{h} FAILED: {exc}")
        report["results"].append(entry)

    Path("/tmp/pano-res-report.json").write_text(json.dumps(report, indent=2))
    s3.upload_file("/tmp/pano-res-report.json", BUCKET, f"{OUT_PREFIX}/report.json")
    print(f"[probe] report uploaded to s3://{BUCKET}/{OUT_PREFIX}/report.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
