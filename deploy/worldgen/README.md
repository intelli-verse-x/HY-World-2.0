# HY-World 2.0 — world-template generation pipeline (Intelliverse)

Generate-once/play-many: a text prompt goes in, a persistent 3D world
(3DGS `.ply`/`.spz` + mesh + previews) comes out in S3, ready for engine
import. Runs on the existing aicart GPU-worker pattern: Redis signal queue +
KEDA scale-to-zero + Karpenter spot GPU nodes.

## Architecture

```
LPUSH pipeline:signal:worldgen {job JSON}          (content-factory-redis)
        │  KEDA worldgen-scaler (0 -> 1)
        ▼
hy-world-worker  (4x 24GB GPU spot node: g6.12xlarge / g5.12xlarge)
  0a. seed image        litellm images API (gemini) — no GPU
  0b. panorama          Qwen-Image-Edit-2509 + HY-Pano-2.0 LoRA (CPU offload)
  1. traj_generate      VLM + SAM3 + Apache-2.0 SAM + MoGe (1 GPU)
  2. traj_render        torchrun x4 point-cloud rendering
  3. video_gen          WorldStereo-2 DMD, FSDP over 4 GPUs
  4. gen_gs_data        3DGS training data
  5. world_gs_trainer   3DGS optimization, ply/spz/mesh export
        │
        ▼
s3://intelliverse-world-templates/worldgen/{jobId}/   (gs/, previews/, panorama.png,
                                                       manifest.json)
RPUSH pipeline:done:worldgen {result JSON}
```

Weights (~190 GB) live on the `scratch-mirror-hy-world` gp3 PVC (HF cache
layout), hydrated from the S3 mirror
`s3://intelli-verse-x-media/models/hy-world/hf` by an initContainer — the
same pattern as vllm-chat's `s3-model-sync`.

## Enqueue a world-generation job

```bash
REDIS_PW=$(kubectl -n aicart get secret keda-redis-auth -o jsonpath='{.data.password}' | base64 -d)
kubectl -n aicart run enqueue-worldgen --rm -i --restart=Never \
  --image=public.ecr.aws/docker/library/redis:7 -- \
  redis-cli -h content-factory-redis.aicart.svc.cluster.local -a "$REDIS_PW" \
  LPUSH pipeline:signal:worldgen \
  '{"jobId":"library-001","prompt":"a cozy ancient library interior with tall bookshelves, warm lamplight, reading alcoves and a central podium","style":"photorealistic","sceneType":"indoor"}'
```

Job fields: `jobId`, `prompt` (required), `style`, `sceneType`
(`indoor`|`outdoor`, default indoor), `outputS3Prefix` (default
`worldgen/{jobId}`), `seedImageS3` (optional, skips the text->image stage).

Watch: `kubectl -n aicart logs -f deploy/hy-world-worker` ·
status key `pipeline:worldgen:status:{jobId}` · completion list
`pipeline:done:worldgen`.

## Deploy from scratch

```bash
kubectl apply -f k8s/nodepool.yaml -f k8s/pvc.yaml

# 1. stage weights (CPU job, ~1 h: HF -> PVC -> S3 mirror)
kubectl apply -f k8s/stage-weights-job.yaml

# 2. build + push image in-cluster (rootless BuildKit; local arm64 Docker
#    can't compile the CUDA extensions in reasonable time)
git archive HEAD | gzip > /tmp/context.tar.gz     # or tar the worktree
aws s3 cp /tmp/context.tar.gz s3://intelli-verse-x-media/models/hy-world/build/context.tar.gz
kubectl -n aicart create secret docker-registry hy-world-ecr-push \
  --docker-server=970547373533.dkr.ecr.us-east-1.amazonaws.com \
  --docker-username=AWS --docker-password="$(aws ecr get-login-password)"
kubectl apply -f k8s/build-job.yaml

# 3. worker + scaler (replicas stay 0 until a job arrives)
kubectl apply -f k8s/deployment.yaml -f k8s/scaledobject.yaml
```

## Cost & runtime

One world ≈ 2.5–4.5 h on a g6.12xlarge spot node (~$2.8–4/h, us-east-1)
≈ **$7–15/world**. `MAX_JOB_SECONDS` (default 16200 s) hard-caps a job.
The KEDA scaler holds the replica while `pipeline:worldgen:processing` is
non-empty and scales to zero ~10 min after the queue drains.

## Spot interruption

Jobs are moved to `pipeline:worldgen:processing` (BRPOPLPUSH) with a
heartbeat key (TTL 300 s). If the node is reclaimed, KEDA keeps the
deployment at 1, the replacement pod finds the stale entry (dead heartbeat)
and re-runs it; stages pass `--skip_exist` and scene state lives on the PVC,
so completed stages are not redone.

## Local modifications vs upstream

- `requirements.txt`: `cupy` -> `cupy-cuda12x` (prebuilt CUDA 12 wheel).
- `hyworld2/worldgen/{traj_generate,video_gen}.py`,
  `src/retrieval_wm.py`: SAM3 repo id overridable via `SAM3_REPO_ID`
  (facebook/sam3 is HF-gated; deployment uses the ungated mirror
  `DiffusionWave/sam3`, byte-identical safetensors).
- `traj_generate.py`, `src/vlm_utils.py`: OpenAI client key from
  `VLM_API_KEY` env (upstream hardcodes "EMPTY"; we route VLM calls through
  litellm, which requires auth).
- pytorch3d installed from MiroPsota prebuilt wheels instead of the git
  source build in `requirements_git.txt`.
- flash-attn intentionally omitted (A10G/L4): `models/attention.py` falls
  back to PyTorch SDPA.

See `LICENSE-NOTE.md` (repo root) for the Tencent community-license
constraints (EU/UK/KR geo-block, <1M MAU) — fine for internal template
generation, revisit before shipping generated worlds to end users.
