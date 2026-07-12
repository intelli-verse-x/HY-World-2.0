# HY-World 2.0 worldgen queue worker
# Text -> seed (SDXL-Turbo) -> 360 pano (HY-Pano-2.0 Qwen backend, 4-bit on 24GB)
# -> perspective views -> WorldMirror-2.0 3DGS -> .splat to S3.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/hf-cache \
    PYTHONPATH=/app

# gcc: triton (via bitsandbytes) JIT-compiles its CUDA utils at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 curl gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        diffusers==0.36.0 \
        "transformers==4.57.1" \
        accelerate \
        peft \
        bitsandbytes \
        safetensors \
        "huggingface_hub[hf_transfer]" \
        einops \
        omegaconf \
        onnxruntime \
        opencv-python-headless \
        pillow \
        plyfile \
        matplotlib \
        scipy \
        tqdm \
        trimesh \
        requests \
        redis \
        boto3

WORKDIR /app
COPY hyworld2 /app/hyworld2
COPY deploy/worker.py /app/worker.py
COPY deploy/LICENSE-NOTE.md /app/LICENSE-NOTE.md

CMD ["python", "/app/worker.py"]
