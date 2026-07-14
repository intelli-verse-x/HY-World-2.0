# HY-World 2.0 worldgen queue worker
#
# Base: PyTorch 2.7.1 + CUDA 12.8 + cuDNN 9 devel (matches upstream pins in
# requirements.txt; devel is required because gsplat_maskgaussian, fused-ssim
# and the navmesh extension compile CUDA/C++ from source).
#
# CUDA archs: 8.6 = A10G (g5), 8.9 = L4 (g6), 9.0 = H100 (portable RunPod runner).
# flash-attn is intentionally NOT installed: models/attention.py falls back to
# PyTorch SDPA automatically, which works on both architectures.

FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0+PTX" \
    MAX_JOBS=4 \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake ninja-build build-essential \
        ffmpeg libgl1 libglib2.0-0 libgomp1 libegl1 libx11-6 \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Python deps (order matters: base reqs -> prebuilt pytorch3d wheel ->
# ---- source-built git deps -> CUDA extensions) ----
COPY requirements.txt requirements_git.txt ./

RUN pip install --no-cache-dir -r requirements.txt

# pytorch3d: use the MiroPsota prebuilt wheel for torch 2.7.1 + cu128 instead
# of the git source build in requirements_git.txt (hours -> seconds).
RUN pip install --no-cache-dir --extra-index-url https://miropsota.github.io/torch_packages_builder \
        "pytorch3d==0.7.9+d9839a9pt2.7.1cu128"

# Remaining git deps (pytorch3d line excluded — handled above).
RUN grep -v 'pytorch3d' requirements_git.txt > /tmp/req_git.txt \
    && pip install --no-cache-dir --no-build-isolation -r /tmp/req_git.txt

# Worker-only deps. ORT 1.22 is the CUDA-12 build; newer wheels currently link
# libcudart.so.13 and cannot load in the upstream CUDA-12.8 base image.
RUN pip install --no-cache-dir redis boto3 httpx "huggingface_hub[hf_transfer]" \
    && pip install --no-cache-dir --no-deps --force-reinstall "onnxruntime-gpu==1.22.0"

# ---- Repo code ----
COPY . /app

# gsplat_maskgaussian (custom gsplat with MaskGaussian) — CUDA build
RUN mkdir -p /app/hyworld2/worldgen/third_party/gsplat_maskgaussian/gsplat/cuda/csrc/third_party \
    && git clone --depth 1 https://github.com/g-truc/glm.git \
       /app/hyworld2/worldgen/third_party/gsplat_maskgaussian/gsplat/cuda/csrc/third_party/glm \
    && cd /app/hyworld2/worldgen/third_party/gsplat_maskgaussian \
    && pip install --no-cache-dir --no-build-isolation -e .

# recastnavigation (git submodule; fetched here in case the build context
# was created without submodules) + navmesh pybind11 extension
RUN if [ ! -f /app/hyworld2/worldgen/third_party/recastnavigation/Recast/Source/Recast.cpp ]; then \
        rm -rf /app/hyworld2/worldgen/third_party/recastnavigation && \
        git clone --depth 1 https://github.com/recastnavigation/recastnavigation.git \
            /app/hyworld2/worldgen/third_party/recastnavigation; \
    fi \
    && cd /app/hyworld2/worldgen/third_party/navmesh \
    && RECAST_PATH=/app/hyworld2/worldgen/third_party/recastnavigation \
       pip install --no-cache-dir --no-build-isolation .

ENV PYTHONPATH=/app

CMD ["python", "/app/deploy/worldgen/worker.py"]
