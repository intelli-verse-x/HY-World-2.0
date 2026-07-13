# Vast.ai spot 4090 — ad-hoc experiments

Cheapest raw compute in the strategy (~$0.16–0.28/GPU-hr, retrieved 2026-07-13).
**Spot/interruptible**, so use it only for fault-tolerant, checkpointed
experiments — never hero worlds or the 5-world batch.

## Files

- `vast-launch.sh` — finds the cheapest interruptible 4090 under a price cap and creates the instance.
- `onstart.sh` — boot hook that arms the **15-min idle watchdog** (`../worldgen/idle_watchdog.sh`, `PROVIDER=vast`).

## Use

```bash
pip install vastai
export VAST_API_KEY=...        # scoped key; read at runtime, never logged
MAX_DPH=0.35 IDLE_SECONDS=900 bash deploy/vast/vast-launch.sh
vastai show instances
```

## Idle guarantee

`onstart.sh` runs the provider-agnostic `idle_watchdog.sh` with `PROVIDER=vast`.
After `IDLE_SECONDS` (900) of GPU idle + no `/tmp/worldgen-activity` marker, the
watchdog **destroys the instance via the Vast API** (a plain `poweroff` does not
reliably stop a Vast bill), with a `poweroff` + 4h absolute failsafe as backup.

While a stage runs, keep the box alive by refreshing the marker:

```bash
while training; do touch /tmp/worldgen-activity; sleep 60; done
```
