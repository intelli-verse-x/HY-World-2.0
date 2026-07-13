#!/usr/bin/env bash
# Launch a Vast.ai SPOT (interruptible) RTX 4090 for ad-hoc HY-World experiments.
#
# Cheapest raw compute in the strategy (~$0.16-0.28/GPU-hr, 2026-07-13). Spot is
# preemptible, so this is for fault-tolerant, checkpointed experiments only —
# NOT hero worlds. Every instance boots the 15-min idle watchdog so a forgotten
# experiment can never bill indefinitely.
#
# Prereqs: `pip install vastai`; `vastai set api-key <KEY>` (or export VAST_API_KEY).
# The API key is read from the environment at runtime and is NEVER printed here.
set -euo pipefail

IMAGE="${IMAGE:-pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel}"
DISK_GB="${DISK_GB:-80}"
MAX_DPH="${MAX_DPH:-0.35}"          # don't bid above this $/hr
IDLE_SECONDS="${IDLE_SECONDS:-900}" # 15-min idle auto-shutdown (hard requirement)
GPU_NAME="${GPU_NAME:-RTX_4090}"
ONSTART="$(dirname "$0")/onstart.sh"

command -v vastai >/dev/null 2>&1 || { echo "install vastai: pip install vastai"; exit 1; }
: "${VAST_API_KEY:?export VAST_API_KEY (scoped key; read at runtime, never logged)}"

echo "[vast] searching cheapest interruptible ${GPU_NAME} (<= \$${MAX_DPH}/hr)…"
OFFER_ID="$(vastai search offers \
  "gpu_name=${GPU_NAME} num_gpus=1 rentable=true dph_total<=${MAX_DPH} disk_space>=${DISK_GB}" \
  --order 'dph_total asc' --raw 2>/dev/null | python3 -c '
import sys, json
offers = json.load(sys.stdin)
if not offers:
    sys.exit("no matching offer under price cap")
o = offers[0]
print(o["id"])
sys.stderr.write(f"[vast] picked offer {o[\"id\"]}: {o.get(\"gpu_name\")} @ ${o.get(\"dph_total\"):.3f}/hr, {o.get(\"geolocation\")}\n")
')"

echo "[vast] creating INTERRUPTIBLE instance from offer ${OFFER_ID}…"
# --bid_price makes it a spot/interruptible rental. The onstart installs and
# arms the idle watchdog; VAST_API_KEY is injected as env so the watchdog can
# self-destroy the instance (billing stops) after 15 min idle.
vastai create instance "$OFFER_ID" \
  --image "$IMAGE" \
  --disk "$DISK_GB" \
  --bid_price "$MAX_DPH" \
  --env "-e IDLE_SECONDS=${IDLE_SECONDS} -e PROVIDER=vast -e VAST_API_KEY=${VAST_API_KEY}" \
  --onstart "$ONSTART" \
  --raw

echo "[vast] instance requested. Watch: vastai show instances"
echo "[vast] idle watchdog will destroy the instance after ${IDLE_SECONDS}s idle."
