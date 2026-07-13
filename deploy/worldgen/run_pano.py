"""
Panorama generation subprocess (Stage 0b).

Wraps the HY-Pano-2.0 Qwen-Image-Edit backend
(hyworld2/panogen/pipeline_with_qwen_image.py) but loads the pipeline with
sequential CPU offload instead of `.to("cuda")`: the Qwen-Image-Edit-2509
transformer is ~20B params (~40 GB bf16) and does not fit a single 24 GB
A10G/L4, while the host has plenty of RAM. Runs as its own process so all GPU
memory is released before the multi-GPU worldgen stages start.
"""

import argparse
import sys
from pathlib import Path

import torch

# Import the panogen module (this file is run with cwd=hyworld2/panogen)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "hyworld2" / "panogen"))

from pipeline_with_qwen_image import (  # noqa: E402
    GENERAL_NEGATIVE_PROMPT,
    GENERAL_POSITIVE_PREFIX,
    GENERAL_POSITIVE_SUFFIX,
    circular_blend_edges,
)
from qwen_image import PanoDiffusionPipeline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--save")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--width", type=int, default=1952)
    ap.add_argument("--height", type=int, default=960)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--base-model", default="Qwen/Qwen-Image-Edit-2509")
    ap.add_argument("--lora-path", default="tencent/HY-World-2.0")
    ap.add_argument("--lora-subfolder", default="HY-Pano-2.0")
    args = ap.parse_args()

    print(f"[pano] loading {args.base_model} (CPU-offload mode)")
    pipe = PanoDiffusionPipeline.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
    pipe.load_lora_weights(
        args.lora_path,
        subfolder=args.lora_subfolder,
        weight_name="pytorch_lora_weights.safetensors",
        torch_dtype=torch.bfloat16,
    )
    # Materialize the HY-Pano adapter into the base transformer before adding
    # Accelerate's CPU-offload hooks. Leaving PEFT LoRA modules live across
    # offload boundaries can strand adapter tensors after the first denoising
    # step; on L40S this surfaced as an illegal CUDA memory access in
    # peft.tuners.lora.layer during step two.
    pipe.fuse_lora(lora_scale=1.0)
    pipe.unload_lora_weights()
    print("[pano] LoRA fused; enabling model CPU offload")
    pipe.enable_model_cpu_offload()
    if args.preflight_only:
        print("[pano] model-load preflight OK")
        return
    if not args.image or not args.save:
        ap.error("--image and --save are required unless --preflight-only is used")

    from PIL import Image

    image = Image.open(args.image).convert("RGB")
    positive = (GENERAL_POSITIVE_PREFIX + args.prompt + GENERAL_POSITIVE_SUFFIX).strip()

    print(f"[pano] generating {args.width}x{args.height}, {args.steps} steps")
    out = pipe(
        image=image,
        prompt=positive,
        negative_prompt=GENERAL_NEGATIVE_PROMPT,
        generator=torch.Generator(device="cpu").manual_seed(args.seed),
        true_cfg_scale=7.5,
        num_inference_steps=args.steps,
        guidance_scale=1.0,
        num_images_per_prompt=1,
        height=args.height,
        width=args.width,
    ).images[0]

    out = circular_blend_edges(out, 32)
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    out.save(args.save)
    print(f"[pano] saved {args.save}")


if __name__ == "__main__":
    main()
