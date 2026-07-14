# HY-World GPU cost strategy — runbook

Serverless-first, tiered by weight. Pay ~$0 idle, get a real burst API, and only
reach for 80 GB silicon when the work needs it. Companion to the decision canvas
`hy-world-gpu-serverless-first-cost-strategy.canvas.tsx` (VRAM, pricing,
break-even, scenarios with cited 2026 sources).

## TL;DR

| Need | Use | Command |
| --- | --- | --- |
| Bursty API / light stage per request | **RunPod Serverless 4090 flex** (baked weights) | `python deploy/serverless/deploy_endpoint.py --image <ecr-digest>` |
| Ad-hoc experiment (interruptible) | **Vast.ai spot 4090** | `bash deploy/vast/vast-launch.sh` |
| Single hero world (full 4-stage) | **RunPod pod A100/H100 80 GB** | `python deploy/worldgen/runpod_launch.py --job <job.json> --network-volume-id <vol>` |
| Full 5-world batch | **RunPod pod A100 + persistent volume** | same, reuse `--network-volume-id` |
| 4K push | **A100 80 GB ONLY** (guardrail) | pod A100 (sustained) or serverless A100 endpoint (bursty) |

## Cost snapshot (retrieved 2026-07-13 — verify before big spend)

- RunPod Serverless flex: 4090 `$0.00031/s` (~$1.12/hr) · A100 `$0.00076/s` (~$2.74/hr) · H100 `$0.00116/s` (~$4.18/hr) — **$0 idle** (scale-to-zero).
- RunPod pods (secure): 4090 `$0.69/hr` · A100 `$1.49/hr` · H100 `$2.69/hr`.
- Vast.ai spot 4090: `$0.16–0.28/hr` (marketplace, fluctuates).
- AWS on-demand fallback: `g6.12xlarge` (4×L4) `$4.60/hr` · `p4de.24xlarge` (8×A100) `$27.45/hr`.
- Break-even (pod beats flex serverless): **A100 ≈ 13 h/day**, 4090 secure ≈ 15 h/day, 4090 spot ≈ 7 h/day.

Rule of thumb: **< ~13 h/day of A100 work → serverless; sustained/batch → pod.**

## Built serverless image (ECR) — weights baked IN the layers

Built + pushed 2026-07-13. This is the image RunPod Serverless **and** Vast.ai pull
directly; the light WorldNav/3DGS weight tier is baked into the image layers (no
network volume, no runtime HF fetch — verified with `HF_HUB_OFFLINE=1`).

| Field | Value |
| --- | --- |
| Repo | `970547373533.dkr.ecr.us-east-1.amazonaws.com/hy-world-serverless-4090` |
| Tags | `baked-light`, `git-cdbc688` |
| Digest | `sha256:02bc38b3b6b4ac8cc599d6159c70d6537b95e024e0ba4cdd70177640fcd3bc1e` |
| Size (compressed, download) | **~21.6 GB** across 26 layers (largest ~10.6 GB site-packages, ~4.2 GB CUDA base, weights ~5–6 GB) |
| Baked weights | **11.4 GB on disk**, 5 repos: `facebook/sam-vit-base`, `IDEA-Research/grounding-dino-tiny`, `Ruicheng/moge-2-vitl-normal`, `facebook/dinov2-base`, `DiffusionWave/sam3` |
| Base | `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel`, `gsplat_maskgaussian` compiled for **sm89** (RTX 4090) |
| Arch | `linux/amd64` (RunPod/Vast 4090s are x86_64) |

**Honest sizing note:** ~21.6 GB is large — it's the `-devel` CUDA base + compiled
extensions + 11 GB weights. Acceptable for a scale-to-zero flex worker (pulled once
per cold node, then FlashBoot-cached), but the base could be slimmed to `-runtime`
in a follow-up if cold-pull latency matters. The **A100 heavy tier is deliberately
NOT baked** (Qwen-Image-Edit ~58 GB + Wan2.1-I2V ~90 GB + WorldStereo DMD + gated
`tencent/HY-World-2.0` ≈ 180 GB+, and gated licensing) — it stays on the PR #7
network-volume pod path. See `deploy/serverless/weights-bake-manifest.json`.

**How it was built:** in-cluster amd64 BuildKit (aicart ns), the same pattern as
`deploy/worldgen/k8s/build-job.yaml`. A macOS/arm64 laptop cannot compile the CUDA
extensions in reasonable time, so `build_and_push.sh` (local `docker build`) is only
for x86 hosts; on arm64 use the in-cluster job.

### Registry auth (ECR is private)
ECR auth tokens last **12 h**; fetch at runtime, never store:
```bash
aws ecr get-login-password --region us-east-1   # username is always "AWS"
```

### Pull from each provider
```text
# RunPod Serverless endpoint — add an ECR container-registry credential in the
# RunPod console (Settings → Container Registry Auth) or via deploy_endpoint.py:
python deploy/serverless/deploy_endpoint.py \
  --image 970547373533.dkr.ecr.us-east-1.amazonaws.com/hy-world-serverless-4090@sha256:02bc38b3b6b4ac8cc599d6159c70d6537b95e024e0ba4cdd70177640fcd3bc1e \
  --registry-auth-id <ecrCredId>      # created from the 12h ECR token above
# (endpoint = workersMin:0 flex → $0 idle; tear down = delete the endpoint)

# RunPod pod (heavy path already does this in runpod_launch.py via ecr_registry_auth())
# uses the same 12h token as imageAuth.

# Vast.ai — pass private-registry creds at instance create:
vastai create instance <OFFER_ID> \
  --image 970547373533.dkr.ecr.us-east-1.amazonaws.com/hy-world-serverless-4090:baked-light \
  --login "-u AWS -p $(aws ecr get-login-password --region us-east-1) 970547373533.dkr.ecr.us-east-1.amazonaws.com" \
  --onstart-cmd "$(cat deploy/vast/onstart.sh)"
# deploy/vast/vast-launch.sh wires this up (add --login for the private image).
```

### Prove it's pullable + runs with weights baked (no GPU needed)
```bash
# Any amd64 host / k8s node with ECR read (EC2 node role or the 12h token):
docker run --rm --entrypoint bash \
  970547373533.dkr.ecr.us-east-1.amazonaws.com/hy-world-serverless-4090:baked-light -lc '
    python /app/deploy/serverless/handler.py --selftest        # guardrail routing
    HF_HUB_OFFLINE=1 python -c "from huggingface_hub import snapshot_download as d;
      [print(r, d(r).startswith(\"/opt/hf-cache\")) for r in
       [\"facebook/sam-vit-base\",\"Ruicheng/moge-2-vitl-normal\",\"DiffusionWave/sam3\"]]"'
```

## Scenario → exact launch

### 1. Bursty API serving / light stage (WorldNav, 3DGS ≤1080p)
RunPod Serverless 4090 flex, weights baked in the image (near-warm cold start, $0 idle).
```bash
# image already built+pushed (see "Built serverless image" above); rebuild only on change:
#   x86 host:  bash deploy/serverless/build_and_push.sh
#   arm64/mac: in-cluster BuildKit job (deploy/worldgen/k8s/build-job.yaml pattern)
python deploy/serverless/deploy_endpoint.py --dry-run     # validate (safe anytime)
python deploy/serverless/deploy_endpoint.py \
  --image 970547373533.dkr.ecr.us-east-1.amazonaws.com/hy-world-serverless-4090@sha256:02bc38b3b6b4ac8cc599d6159c70d6537b95e024e0ba4cdd70177640fcd3bc1e \
  --registry-auth-id <ecrCredId>
# invoke: POST https://api.runpod.ai/v2/<endpointId>/run  {"input":{"jobId":"x","stage":"traj_generate"}}
```

### 2. Ad-hoc experiment (interruptible, cheapest)
```bash
pip install vastai && export VAST_API_KEY=...   # read at runtime, never logged
MAX_DPH=0.35 IDLE_SECONDS=900 bash deploy/vast/vast-launch.sh
```

### 3. Single hero world / 4-stage (heavy — needs 80 GB)
Use the existing PR #7 pod path (unchanged). The in-pod dual watchdog + the new
continuous idle watchdog both cap billing.
```bash
python deploy/worldgen/runpod_launch.py --job deploy/worldgen/jobs/full-nm-a.json \
  --network-volume-id <encrypted-vol-id>          # stage weights once, reuse across worlds
```

### 4. Full 5-world batch
Same as (3) with the persistent `--network-volume-id` reused → high utilization,
so a pod beats serverless (above the ~13 h/day break-even).

### 5. 4K push — A100-only guardrail
4K needs 40–80 GB. The 24 GB 4090 must **never** attempt it. Enforcement:
- **Serverless handler** (`deploy/serverless/handler.py`) refuses `width*height ≥ 3840×2160` and the heavy stages, returning `route: a100-pod`.
- Run 4K on a pod A100 (`runpod_launch.py`, GPU preference already A100/H100 first) or a serverless **A100** endpoint (edit `endpoint-config.json` gpuTypeIds → A100).

## A100-only-for-4K guardrail (why + where)

| Where | Behavior |
| --- | --- |
| `handler.py` `_guardrail()` | Rejects 4K + `panorama`/`video_gen` on the 24 GB tier with a structured `route: a100-pod` error (self-tested). |
| `weights-bake-manifest.json` | Documents that heavy/gated weights are NOT baked into the 4090 image. |
| `runpod_launch.py` (PR #7) | `GPU_PREFERENCE` already targets 80 GB A100/H100 for the heavy pod path. |

## Idle auto-shutdown guarantee (15 min, every provider)

| Provider / mode | Mechanism | 15-min idle |
| --- | --- | --- |
| RunPod Serverless flex | Native scale-to-zero (`workersMin: 0`, `idleTimeout`) → $0 idle | **Native** |
| RunPod pod | `deploy/worldgen/idle_watchdog.sh` (`PROVIDER=runpod`) → `DELETE /v1/pods/{id}` | `IDLE_SECONDS=900` |
| Vast.ai spot/on-demand | `onstart.sh` runs `idle_watchdog.sh` (`PROVIDER=vast`) → Vast API destroy | `IDLE_SECONDS=900` |
| AWS EC2 fallback | existing `capacity_lifecycle.py` + `watchdog.yaml` empty-node cleanup | Yes |

`idle_watchdog.sh` **generalizes PR #7's** post-job idle *sleep* into a
**continuous** monitor (GPU utilization + `/tmp/worldgen-activity` marker +
optional Redis processing queue). It reuses the same layered teardown
(REST DELETE → `runpodctl`/`vastai` → `poweroff`) so a host that goes quiet
mid-session — not only after a clean completion — is still reclaimed.

Arm it on any pod entrypoint:
```bash
PROVIDER=runpod IDLE_SECONDS=900 deploy/worldgen/idle_watchdog.sh &
# and while a stage runs, refresh the marker:
while working; do touch /tmp/worldgen-activity; sleep 60; done
```

### Verify shutdown behavior cheaply
The watchdog is verifiable without GPU spend by driving it with a fake
`nvidia-smi` and a short idle window (used in CI / locally):
```bash
# stub an idle GPU + 6s window; expect a poweroff/teardown attempt in ~6-8s
PATH="/tmp/fakebin:$PATH" PROVIDER=generic IDLE_SECONDS=6 POLL_SECONDS=2 \
  GRACE_SECONDS=0 deploy/worldgen/idle_watchdog.sh
```
See the PR description for the recorded dry-run.

## Live RunPod endpoint — ready to deploy

The image is built, pushed, and container-verified (guardrail + offline weights +
gsplat). A live serverless deploy was intentionally **not** left running to avoid a
~21.6 GB cold-pull billing a worker and to keep $0 idle. To bring it up and tear it
down:
```bash
# 1) create an ECR container-registry credential in RunPod (12h token; console or API)
# 2) deploy against the verified digest
python deploy/serverless/deploy_endpoint.py \
  --image 970547373533.dkr.ecr.us-east-1.amazonaws.com/hy-world-serverless-4090@sha256:02bc38b3b6b4ac8cc599d6159c70d6537b95e024e0ba4cdd70177640fcd3bc1e \
  --registry-auth-id <ecrCredId>
# 3) smoke: POST /v2/<endpointId>/run {"input":{"jobId":"smoke","stage":"traj_generate"}}
# 4) tear down: delete the endpoint (workersMin:0 already = $0 idle)
```

## Coordination / safety

- This work lives on branch `ops/serverless-gpu-cost-strategy` (base `main`),
  isolated from PR #7 `ops/runpod-portable-runner`. The build ran only after the
  worldgen pipeline finished and freed the shared builder (0 RunPod pods), so there
  was no contention; no pods/volumes were touched.
- Built in-cluster with dedicated, disposable resources (`hy-world-serverless-build`
  job + `hy-world-serverless-build-ws` PVC + `hy-world-serverless-ecr-push` secret),
  all deleted after verification. Nothing the gate/pipeline agents read was modified.
- Secrets (RunPod key `/intelliverse/worldgen/runpod-api-key`, Vast key) are read
  at runtime only and never printed or baked into images.
- A serverless deploy is safe to run with `--dry-run` anytime; the live flex
  endpoint (`workersMin: 0`) is independent of the pod and costs $0 until called.
- Known limitation: this IAM user lacks `ecr:BatchDeleteImage`, so two superseded
  build SHAs (`git-4b80bd2`, `git-54b98da` — earlier offline-resolution bugs, now
  fixed) remain in the repo. They are untagged from `baked-light`; the lifecycle
  policy (keep last 10 tagged) governs them, or a founder can prune them.
