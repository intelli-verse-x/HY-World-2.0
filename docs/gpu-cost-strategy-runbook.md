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

## Scenario → exact launch

### 1. Bursty API serving / light stage (WorldNav, 3DGS ≤1080p)
RunPod Serverless 4090 flex, weights baked in the image (near-warm cold start, $0 idle).
```bash
bash deploy/serverless/build_and_push.sh                 # build+push baked image (x86 builder)
python deploy/serverless/deploy_endpoint.py --dry-run     # validate (safe anytime)
python deploy/serverless/deploy_endpoint.py --image <ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com/hy-world-serverless-4090@sha256:...
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

## Coordination / safety

- This work lives on branch `ops/serverless-gpu-cost-strategy` (base `main`),
  isolated from PR #7 `ops/runpod-portable-runner` and its **running H100 pod +
  network volume** — do not reconfigure those.
- Secrets (RunPod key `/intelliverse/worldgen/runpod-api-key`, Vast key) are read
  at runtime only and never printed or baked into images.
- A serverless deploy is safe to run with `--dry-run` anytime; the live flex
  endpoint (`workersMin: 0`) is independent of the pod and costs $0 until called.
