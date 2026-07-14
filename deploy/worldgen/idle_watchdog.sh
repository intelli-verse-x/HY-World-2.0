#!/usr/bin/env bash
# Provider-agnostic 15-minute idle auto-shutdown watchdog.
#
# GUARANTEE: after $IDLE_SECONDS (default 900 = 15 min) of continuous GPU idle
# with no active worldgen job, this self-terminates the host so it can never
# bill indefinitely — on RunPod pods AND Vast.ai instances (and a generic
# poweroff fallback for anything else).
#
# This GENERALIZES the PR #7 (`ops/runpod-portable-runner`) design, which armed
# an absolute hard-deadline plus a fixed post-job idle *sleep*. Here the idle
# window is a CONTINUOUS monitor (GPU utilization + a job-activity marker + an
# optional Redis processing queue), so a host that goes quiet mid-session — not
# just after a clean job completion — is still reclaimed. Source this from an
# entrypoint, or run it as a background process:
#
#     PROVIDER=runpod IDLE_SECONDS=900 ./idle_watchdog.sh &
#
# It NEVER prints secret values. RUNPOD_API_KEY / VAST_API_KEY are read from the
# environment only when a teardown call is actually made.
set -uo pipefail

IDLE_SECONDS="${IDLE_SECONDS:-900}"          # 15-minute idle shutdown (hard requirement)
POLL_SECONDS="${POLL_SECONDS:-30}"           # how often to sample activity
UTIL_THRESHOLD="${UTIL_THRESHOLD:-5}"        # GPU % below which we consider "idle"
GRACE_SECONDS="${GRACE_SECONDS:-120}"        # startup grace before the clock can arm
ACTIVITY_FILE="${ACTIVITY_FILE:-/tmp/worldgen-activity}"  # touch this while working
PROVIDER="${PROVIDER:-auto}"                 # runpod | vast | generic | auto
LOG_PREFIX="[idle-watchdog]"

log() { echo "$LOG_PREFIX $*"; }

detect_provider() {
  if [[ "$PROVIDER" != "auto" ]]; then echo "$PROVIDER"; return; fi
  if [[ -n "${RUNPOD_POD_ID:-}" ]]; then echo runpod; return; fi
  if [[ -n "${VAST_CONTAINERLABEL:-}" || -n "${CONTAINER_ID:-}" || -n "${VAST_INSTANCE_ID:-}" ]]; then echo vast; return; fi
  echo generic
}

# ---- Teardown paths (best-effort, layered fallbacks) ----
terminate_runpod() {
  local pid="${RUNPOD_POD_ID:-}"
  log "terminating RunPod pod ${pid:-<self>}"
  if [[ -n "${RUNPOD_API_KEY:-}" && -n "$pid" ]]; then
    curl -fsS -X DELETE "https://rest.runpod.io/v1/pods/${pid}" \
      -H "Authorization: Bearer ${RUNPOD_API_KEY}" >/dev/null 2>&1 && return 0
    python3 -c 'import os,urllib.request as u;u.urlopen(u.Request("https://rest.runpod.io/v1/pods/"+os.environ["RUNPOD_POD_ID"],method="DELETE",headers={"Authorization":"Bearer "+os.environ["RUNPOD_API_KEY"]}),timeout=20)' >/dev/null 2>&1 && return 0
  fi
  command -v runpodctl >/dev/null 2>&1 && runpodctl remove pod "$pid" >/dev/null 2>&1 && return 0
  return 1
}

terminate_vast() {
  # Vast injects CONTAINER_ID (the instance id) into the container env. Destroy
  # the whole instance so billing stops — a plain poweroff does NOT stop a Vast
  # bill on its own for many host configs.
  local iid="${VAST_INSTANCE_ID:-${CONTAINER_ID:-}}"
  log "terminating Vast.ai instance ${iid:-<self>}"
  if [[ -n "${VAST_API_KEY:-}" && -n "$iid" ]]; then
    curl -fsS -X DELETE "https://console.vast.ai/api/v0/instances/${iid}/" \
      -H "Authorization: Bearer ${VAST_API_KEY}" \
      -H "Content-Type: application/json" -d '{}' >/dev/null 2>&1 && return 0
    if command -v vastai >/dev/null 2>&1; then
      vastai destroy instance "$iid" --api-key "$VAST_API_KEY" >/dev/null 2>&1 && return 0
    fi
  fi
  return 1
}

terminate_self() {
  local provider; provider="$(detect_provider)"
  case "$provider" in
    runpod) terminate_runpod && exit 0 ;;
    vast)   terminate_vast   && exit 0 ;;
  esac
  # Generic / last-resort fallback: hard power off (also catches the case where
  # the API teardown above silently failed).
  log "API teardown unavailable/failed; forcing poweroff"
  poweroff -f >/dev/null 2>&1 || shutdown -h now >/dev/null 2>&1 || kill -9 1 || true
  exit 0
}

gpu_busy() {
  # Returns 0 (busy) if ANY visible GPU is at/above the utilization threshold.
  local max
  max="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1)"
  [[ -z "$max" ]] && return 1            # no GPU readable -> treat as idle
  [[ "$max" =~ ^[0-9]+$ ]] || return 1
  (( max >= UTIL_THRESHOLD ))
}

marker_active() {
  # A stage that is running should `touch $ACTIVITY_FILE` periodically. If the
  # marker was updated within IDLE_SECONDS, the host is considered active even
  # if the GPU momentarily dipped (e.g. between stages / CPU-bound sub-steps).
  [[ -f "$ACTIVITY_FILE" ]] || return 1
  local now mtime age
  now="$(date +%s)"; mtime="$(stat -c %Y "$ACTIVITY_FILE" 2>/dev/null || stat -f %m "$ACTIVITY_FILE" 2>/dev/null || echo 0)"
  age=$(( now - mtime ))
  (( age < IDLE_SECONDS ))
}

redis_busy() {
  # Optional: if a local Redis processing queue is reachable, a non-empty
  # processing list means a job is in flight.
  command -v redis-cli >/dev/null 2>&1 || return 1
  local q="${WORLDGEN_PROCESSING_QUEUE:-pipeline:worldgen-full:processing}"
  local n; n="$(redis-cli LLEN "$q" 2>/dev/null || echo 0)"
  [[ "$n" =~ ^[0-9]+$ ]] && (( n > 0 ))
}

main() {
  local provider; provider="$(detect_provider)"
  log "armed: provider=$provider idle=${IDLE_SECONDS}s poll=${POLL_SECONDS}s util>=${UTIL_THRESHOLD}% grace=${GRACE_SECONDS}s"
  sleep "$GRACE_SECONDS"
  local idle_for=0
  while true; do
    if gpu_busy || marker_active || redis_busy; then
      (( idle_for != 0 )) && log "activity resumed; idle timer reset"
      idle_for=0
    else
      idle_for=$(( idle_for + POLL_SECONDS ))
      if (( idle_for % 120 == 0 )); then log "idle ${idle_for}s / ${IDLE_SECONDS}s"; fi
      if (( idle_for >= IDLE_SECONDS )); then
        log "idle threshold reached (${idle_for}s >= ${IDLE_SECONDS}s); shutting down"
        terminate_self
      fi
    fi
    sleep "$POLL_SECONDS"
  done
}

main "$@"
