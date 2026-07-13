# HY-World Serverless 4090 flex worker (baked weights)

The founder's target pattern: a **RunPod Serverless 4090 flex** worker with model
weights **baked into the container image** (not a network volume). Baking keeps
cold-start cost near the warm rate, gives a real API, pays **$0 when idle** (flex
scales to zero), and bursts under load.

## What runs here (light tier)

A 4090 is 24 GB, so this image only serves the **light** stages and bakes only
the light, ungated weights (~5.8 GB):

| Baked (ungated) | Stage it serves |
| --- | --- |
| `facebook/sam-vit-base`, `IDEA-Research/grounding-dino-tiny`, `Ruicheng/moge-2-vitl-normal`, `facebook/dinov2-base`, `DiffusionWave/sam3` | WorldNav (`traj_generate`), 3DGS train (`gs_train`) at ≤1080p |

Heavy stages (panorama diffusion, WorldStereo `video_gen`) and **any 4K job** are
**refused** by the handler with `route: a100-pod` — the A100-only-for-4K guardrail.
See `../worldgen/runpod_launch.py` (PR #7) for the A100 pod path and
`weights-bake-manifest.json` for the bake/no-bake rationale.

## Files

- `Dockerfile` — multi-stage; stage 1 bakes weights into a layer, stage 2 is the runtime.
- `bake_weights.py` — pulls the pinned, `baked:true` snapshots at build time.
- `handler.py` — RunPod serverless handler: request → light stage → artifacts to S3, with the 4K/heavy guardrail.
- `endpoint-config.json` — flex endpoint spec (`workersMin: 0` → scale-to-zero).
- `deploy_endpoint.py` — creates the endpoint; reads the RunPod key from SSM at runtime.
- `build_and_push.sh` — build + push the baked image to private ECR.

## Deploy

```bash
# 1. build + push the baked image (prefer an x86/in-cluster builder for CUDA)
bash deploy/serverless/build_and_push.sh

# 2. validate auth + config without creating anything (safe while the pod runs)
python deploy/serverless/deploy_endpoint.py --dry-run

# 3. create the flex endpoint (scale-to-zero; $0 idle)
python deploy/serverless/deploy_endpoint.py \
  --image 970547373533.dkr.ecr.us-east-1.amazonaws.com/hy-world-serverless-4090@sha256:...
```

## Idle guarantee

Serverless flex satisfies the 15-min idle requirement **natively**: with
`workersMin: 0` a worker scales to zero after `idleTimeout` and bills **$0**
while no request is in flight. No watchdog is needed on serverless (the watchdog
is for pods + Vast). `executionTimeoutMs` caps a single runaway request.
