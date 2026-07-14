# HY-World Serverless 4090 flex worker (baked weights)

The founder's target pattern: a **RunPod Serverless 4090 flex** worker with model
weights **baked into the container image** (not a network volume). Baking keeps
cold-start cost near the warm rate, gives a real API, pays **$0 when idle** (flex
scales to zero), and bursts under load.

## What runs here (light tier)

> **Built image (2026-07-13):**
> `970547373533.dkr.ecr.us-east-1.amazonaws.com/hy-world-serverless-4090`
> tags `baked-light` / `git-cdbc688`,
> digest `sha256:02bc38b3b6b4ac8cc599d6159c70d6537b95e024e0ba4cdd70177640fcd3bc1e`,
> `linux/amd64`, ~21.6 GB compressed, **11.4 GB weights baked in** and verified
> offline (`HF_HUB_OFFLINE=1`). Pull/registry-auth per provider: see
> `../../docs/gpu-cost-strategy-runbook.md` → "Built serverless image (ECR)".

A 4090 is 24 GB, so this image only serves the **light** stages and bakes only
the light, ungated weights (11.4 GB on disk):

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
# 1. build + push the baked image — ONLY on change; already built+pushed.
#    x86 host:  bash deploy/serverless/build_and_push.sh
#    arm64/mac: in-cluster amd64 BuildKit (CUDA won't compile under qemu);
#               mirror deploy/worldgen/k8s/build-job.yaml with dockerfile dir
#               deploy/serverless and output repo hy-world-serverless-4090.

# 2. validate auth + config without creating anything (safe anytime)
python deploy/serverless/deploy_endpoint.py --dry-run

# 3. create the flex endpoint (scale-to-zero; $0 idle) against the verified digest
python deploy/serverless/deploy_endpoint.py \
  --image 970547373533.dkr.ecr.us-east-1.amazonaws.com/hy-world-serverless-4090@sha256:02bc38b3b6b4ac8cc599d6159c70d6537b95e024e0ba4cdd70177640fcd3bc1e \
  --registry-auth-id <ecrCredId>   # ECR is private; 12h token via `aws ecr get-login-password`
```

## Idle guarantee

Serverless flex satisfies the 15-min idle requirement **natively**: with
`workersMin: 0` a worker scales to zero after `idleTimeout` and bills **$0**
while no request is in flight. No watchdog is needed on serverless (the watchdog
is for pods + Vast). `executionTimeoutMs` caps a single runaway request.
