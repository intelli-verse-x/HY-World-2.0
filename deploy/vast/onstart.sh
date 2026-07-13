#!/usr/bin/env bash
# Vast.ai onstart: arm the 15-min idle auto-shutdown, then leave the box ready
# for ad-hoc experiments. Runs inside the Vast instance at boot.
#
# It fetches the provider-agnostic idle_watchdog.sh (same script used by the
# RunPod pod path) and runs it with PROVIDER=vast. The watchdog destroys the
# instance via the Vast API after $IDLE_SECONDS idle. Vast injects CONTAINER_ID
# (the instance id); VAST_API_KEY arrives as instance env (never logged).
set -uo pipefail

IDLE_SECONDS="${IDLE_SECONDS:-900}"
WATCHDOG_URL="${WATCHDOG_URL:-https://raw.githubusercontent.com/intelli-verse-x/HY-World-2.0/ops/serverless-gpu-cost-strategy/deploy/worldgen/idle_watchdog.sh}"
WD=/opt/idle_watchdog.sh

# Prefer a baked copy if the experiment image already ships it; else fetch.
if [[ -f /app/deploy/worldgen/idle_watchdog.sh ]]; then
  cp /app/deploy/worldgen/idle_watchdog.sh "$WD"
else
  curl -fsSL "$WATCHDOG_URL" -o "$WD" || { echo "[onstart] could not fetch watchdog"; }
fi

if [[ -f "$WD" ]]; then
  chmod +x "$WD"
  echo "[onstart] arming idle watchdog (PROVIDER=vast, IDLE=${IDLE_SECONDS}s)"
  PROVIDER=vast IDLE_SECONDS="$IDLE_SECONDS" ACTIVITY_FILE=/tmp/worldgen-activity \
    nohup "$WD" >/var/log/idle_watchdog.log 2>&1 &
else
  # Fail-safe: even without the watchdog, never let a forgotten box run forever.
  echo "[onstart] watchdog missing; arming absolute ${IDLE_SECONDS}s+4h failsafe poweroff"
  ( sleep 14400; poweroff -f || kill -9 1 ) &
fi

echo "[onstart] ready. While a stage runs, touch /tmp/worldgen-activity to hold the box."
echo "[onstart] example: while training; do touch /tmp/worldgen-activity; sleep 60; done"
