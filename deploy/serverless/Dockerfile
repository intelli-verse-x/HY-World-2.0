# HY-World 2.0 — RunPod Serverless 4090 flex worker (weights BAKED INTO IMAGE)
#
# Founder's target pattern: a Serverless 4090 flex worker with model weights
# baked into the container image (NOT a network volume). Baking keeps cold-start
# cost near the warm rate (weights load from local layers, not the network),
# gives a real API, pays $0 idle (flex scales to zero), and bursts under load.
#
# TIERING: a 4090 is 24 GB, so this image bakes ONLY the light WorldNav vision
# tier (~5.8 GB, ungated) — see weights-bake-manifest.json. The heavy Qwen
# editor / Wan base / WorldStereo DMD / gated tencent weights stay on the A100
# pod path (encrypted network volume). This worker serves the light stages
# (WorldNav trajectory; 3DGS train <=1080p) as a burst-scaling API and REFUSES
# 4K / heavy stages (routing them to the A100 pod) via the handler guardrail.
#
# ---- Stage 1: bake weights into a dedicated layer ----
FROM python:3.11-slim AS weights
ENV HF_HUB_ENABLE_HF_TRANSFER=1
RUN pip install --no-cache-dir "huggingface_hub[hf_transfer]"
WORKDIR /build
COPY deploy/serverless/weights-bake-manifest.json deploy/serverless/weights-bake-manifest.json
COPY deploy/serverless/bake_weights.py deploy/serverless/bake_weights.py
# HF_TOKEN is only needed if a baked model becomes gated; ungated tier needs none.
ARG HF_TOKEN=""
RUN OUT=/opt/hf-cache HF_TOKEN="$HF_TOKEN" python deploy/serverless/bake_weights.py

# ---- Stage 2: runtime (matches the pod worker's CUDA/toolchain pins) ----
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="8.9+PTX" \
    MAX_JOBS=4 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    HF_HOME=/opt/hf-cache \
    MODEL_ROOT=/opt/hf-cache \
    SERVERLESS_TIER=light-4090

RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake ninja-build build-essential \
        ffmpeg libgl1 libglib2.0-0 libgomp1 libegl1 libx11-6 \
        curl ca-certificates redis-tools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements_git.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --extra-index-url https://miropsota.github.io/torch_packages_builder \
        "pytorch3d==0.7.9+d9839a9pt2.7.1cu128"
RUN grep -v 'pytorch3d' requirements_git.txt > /tmp/req_git.txt \
    && pip install --no-cache-dir --no-build-isolation -r /tmp/req_git.txt
RUN pip install --no-cache-dir redis boto3 httpx "huggingface_hub[hf_transfer]" runpod \
    && pip install --no-cache-dir --no-deps --force-reinstall "onnxruntime-gpu==1.22.0"

# Repo code (pipeline + worker building blocks the handler reuses).
COPY . /app

# gsplat_maskgaussian for sm89 (RTX 4090). Built here so the 3DGS light stage
# runs natively without a JIT-at-launch stall on the serverless worker.
RUN mkdir -p /app/hyworld2/worldgen/third_party/gsplat_maskgaussian/gsplat/cuda/csrc/third_party \
    && git clone --depth 1 https://github.com/g-truc/glm.git \
       /app/hyworld2/worldgen/third_party/gsplat_maskgaussian/gsplat/cuda/csrc/third_party/glm \
    && cd /app/hyworld2/worldgen/third_party/gsplat_maskgaussian \
    && TORCH_CUDA_ARCH_LIST="8.9" pip install --no-cache-dir --no-build-isolation -e . || true

# ---- Bring in the baked weights layer ----
COPY --from=weights /opt/hf-cache /opt/hf-cache
# traj_generate resolves SAM/GroundingDINO/SAM3/MoGe via the hardcoded
# ~/.cache/huggingface/hub dir; point it at the baked cache (offline).
# bake_weights.py writes the HF hub cache directly under /opt/hf-cache (models--*),
# so HF_HUB_CACHE points there and traj_generate's hardcoded
# ~/.cache/huggingface/hub is symlinked to it for offline resolution.
RUN mkdir -p /root/.cache/huggingface && ln -sfn /opt/hf-cache /root/.cache/huggingface/hub

ENV PYTHONPATH=/app \
    HF_HUB_CACHE=/opt/hf-cache \
    SAM_BOX_REPO_ID=facebook/sam-vit-base \
    SAM3_REPO_ID=DiffusionWave/sam3

# RunPod Serverless entrypoint: starts the handler loop (runpod.serverless.start).
CMD ["python", "/app/deploy/serverless/handler.py"]
