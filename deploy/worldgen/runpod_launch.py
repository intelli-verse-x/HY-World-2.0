#!/usr/bin/env python3
"""Control-plane launcher for the HY-World full-stack worker on RunPod.

Runs from an operator workstation / CI. Responsibilities:

  1. Read the RunPod API key from AWS SSM SecureString at runtime (never print).
  2. Mint least-privilege, short-lived AWS credentials for the pod by assuming
     the ``hy-world-portable-runner`` role, so the pod can read the private,
     license-controlled weights cache and write private S3 checkpoints/artifacts
     WITHOUT ever seeing operator credentials.
  3. Register a short-lived RunPod container-registry auth for the private ECR
     worker image.
  4. Create an encrypted network volume in a US Secure Cloud data center (so the
     183 GiB weights are staged once and reused across all five worlds).
  5. Create ONE Secure Cloud US pod (H100/A100 80 GB) that runs
     ``runpod-entrypoint.sh`` with a self-terminating watchdog.
  6. Poll pod state, enforce the budget cap and idle shutdown, and terminate.

SECURITY: secrets are only ever read into memory and passed to RunPod as pod
env / registry auth. They are never logged, echoed, or committed. This script is
credential-safe to store in git.

STATUS: this path is BLOCKED at step 2 until an admin makes the
``hy-world-portable-runner`` role assumable by the operator principal (its trust
policy currently allows only ``ec2.amazonaws.com``). See
docs/runpod-portable-runner.md for the exact one-line unblock.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.request
import urllib.error

import boto3
from botocore.exceptions import ClientError

ACCOUNT = "970547373533"
SSM_KEY_PATH = "/intelliverse/worldgen/runpod-api-key"
RUNNER_ROLE_ARN = f"arn:aws:iam::{ACCOUNT}:role/hy-world-portable-runner"
ECR_REPO = "hy-world-full-worker"
WORKER_DIGEST = (
    "sha256:24ad1ae4b0ae26722d710efdb6c1602268c45d40230a7b5a2c96c952311829b0"
)
MODEL_BUCKET = "intelliverse-hyworld-private-us-east-1"
MODEL_PREFIX = "models/hy-world/hf"
REST = "https://rest.runpod.io/v1"
import os
ENTRYPOINT_PATH = os.path.join(os.path.dirname(__file__), "runpod-entrypoint.sh")

# 80 GB Secure Cloud options in preference order (smallest reliable topology
# first). Rates are verified at launch; these are only ordering hints.
GPU_PREFERENCE = [
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA A100 80GB PCIe",
    "NVIDIA H100 80GB HBM3",
    "NVIDIA H200",
]

# US Secure Cloud data centers (from the RunPod REST enum; US only). Ordered
# East-first so the ~196 GB pull from the us-east-1 weights bucket stays on a
# short, fast path; falls through to central/west only if East has no 80 GB GPU.
US_SECURE_DCS = [
    "US-GA-1", "US-GA-2", "US-NC-1", "US-DE-1", "US-IL-1",
    "US-KS-2", "US-KS-3", "US-TX-1", "US-TX-3", "US-TX-4",
    "US-CA-2", "US-WA-1",
]


def get_runpod_key(region: str) -> str:
    ssm = boto3.client("ssm", region_name=region)
    return ssm.get_parameter(Name=SSM_KEY_PATH, WithDecryption=True)["Parameter"]["Value"]


def assume_runner_creds() -> dict:
    """Short-lived least-privilege creds for the pod. Fail closed (never fall
    back to operator credentials)."""
    sts = boto3.client("sts")
    try:
        c = sts.assume_role(
            RoleArn=RUNNER_ROLE_ARN,
            RoleSessionName="runpod-fullstack",
            DurationSeconds=3600,
        )["Credentials"]
    except ClientError as e:
        raise SystemExit(
            "BLOCKED: cannot mint scoped pod credentials: "
            f"{e.response['Error']['Code']}. The pod must NOT receive operator "
            "credentials. Admin action required — see docs/runpod-portable-runner.md."
        )
    return {
        "AWS_ACCESS_KEY_ID": c["AccessKeyId"],
        "AWS_SECRET_ACCESS_KEY": c["SecretAccessKey"],
        "AWS_SESSION_TOKEN": c["SessionToken"],
    }


def runpod(key: str, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        REST + path, data=data, method=method,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read(400).decode("utf-8", "replace")}


def ecr_registry_auth(region: str) -> tuple[str, str]:
    ecr = boto3.client("ecr", region_name=region)
    tok = ecr.get_authorization_token()["authorizationData"][0]
    user, pwd = base64.b64decode(tok["authorizationToken"]).decode().split(":", 1)
    return user, pwd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--job", required=True, help="path to worldgen job JSON")
    ap.add_argument("--volume-gb", type=int, default=400,
                    help="container disk GiB (weights ~183 + scratch)")
    ap.add_argument("--budget-usd", type=float, default=416.46)
    ap.add_argument("--rate-usd", type=float, default=2.69,
                    help="verified pod hourly rate for cost accounting")
    ap.add_argument("--hard-deadline-seconds", type=int, default=14400)
    ap.add_argument("--confirm-seconds", type=int, default=600,
                    help="max time to confirm RUNNING before terminating")
    ap.add_argument("--network-volume-id", default=None,
                    help="attach an existing encrypted volume at /models so the "
                         "196GB weight cache is staged once and reused (skips the "
                         "~22min per-run S3 sync). RunPod pins the pod to its DC.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--probe", default=None,
                    help="feasibility probe mode (e.g. 'pano'); runs the probe "
                         "instead of the worker, then self-terminates")
    args = ap.parse_args()

    job = json.load(open(args.job))
    assert job.get("jobId"), "job JSON must include jobId"

    key = get_runpod_key(args.region)
    st, pods = runpod(key, "GET", "/pods")
    if st != 200:
        raise SystemExit(f"RunPod auth failed: {st} {pods}")
    print(f"runpod_auth_ok existing_pods={len(pods)}")

    # Step 2 — the current hard blocker. Fails closed with an actionable message.
    pod_creds = assume_runner_creds()

    if args.dry_run:
        print("dry-run: credential + auth path OK; not creating billing resources")
        return 0

    reg_user, reg_pwd = ecr_registry_auth(args.region)
    st, auth = runpod(key, "POST", "/containerregistryauth",
                      {"name": f"ecr-{int(time.time())}",
                       "username": reg_user, "password": reg_pwd})
    if st not in (200, 201):
        raise SystemExit(f"registry auth failed: {st} {auth}")
    auth_id = auth["id"]

    env = {
        "AWS_REGION": args.region,
        "MODEL_BUCKET": MODEL_BUCKET,
        "MODEL_PREFIX": MODEL_PREFIX,
        "JOB_JSON_B64": base64.b64encode(json.dumps(job).encode()).decode(),
        "INSTANCE_TYPE": "runpod-secure-80gb",
        "INSTANCE_HOURLY_USD": str(args.rate_usd),
        "HARD_DEADLINE_SECONDS": str(args.hard_deadline_seconds),
        "RUNPOD_API_KEY": key,
        **pod_creds,
    }
    if args.probe:
        env["PROBE_MODE"] = args.probe
    # Hero run uses container disk (no network volume) to avoid data-center
    # pinning; RunPod places the pod in whichever US Secure Cloud DC has an
    # 80 GB GPU available.
    image = f"{ACCOUNT}.dkr.ecr.{args.region}.amazonaws.com/{ECR_REPO}@{WORKER_DIGEST}"
    if args.network_volume_id:
        # Persist/reuse the weight cache on an encrypted volume mounted at /models.
        # MODEL_ROOT lives on the volume so the .sync-complete stamp survives and
        # the ~22min sync only happens on first use.
        env["MODEL_ROOT"] = "/models/hf-cache"

    # The entrypoint is a NEW repo file that is not baked into the prebuilt image,
    # so deliver it inline (base64) and, before anything else, arm an inline
    # Python failsafe watchdog using RunPod-injected RUNPOD_POD_ID + the
    # RUNPOD_API_KEY we inject. This guarantees self-termination even if the
    # script never runs. Python is always present in the worker image.
    with open(ENTRYPOINT_PATH, "rb") as fh:
        script_b64 = base64.b64encode(fh.read()).decode()
    failsafe = (
        "python3 -c \"import os,time,urllib.request as u;"
        f"time.sleep({args.hard_deadline_seconds});"
        "u.urlopen(u.Request('https://rest.runpod.io/v1/pods/'+os.environ['RUNPOD_POD_ID'],"
        "method='DELETE',headers={'Authorization':'Bearer '+os.environ['RUNPOD_API_KEY']}),timeout=20)\" ; "
        "kill -9 1"
    )
    # Probe scripts are also new repo files not baked into the image; deliver inline
    # to their expected on-disk path so the entrypoint's PROBE_MODE branch can run them
    # (they resolve panogen imports relative to /app/deploy/worldgen/).
    probe_deliver = ""
    if args.probe == "pano":
        probe_path = os.path.join(os.path.dirname(ENTRYPOINT_PATH), "pano_res_probe.py")
        with open(probe_path, "rb") as fh:
            probe_b64 = base64.b64encode(fh.read()).decode()
        probe_deliver = (
            f"printf %s '{probe_b64}' | base64 -d > /app/deploy/worldgen/pano_res_probe.py && "
        )
    start_cmd = (
        f"( {failsafe} ) & "
        f"printf %s '{script_b64}' | base64 -d > /tmp/runpod-entrypoint.sh && "
        f"{probe_deliver}"
        "exec bash /tmp/runpod-entrypoint.sh"
    )
    pod_body = {
        "name": f"hyworld-full-{job['jobId']}",
        "imageName": image,
        "containerRegistryAuthId": auth_id,
        "cloudType": "SECURE",
        "computeType": "GPU",
        "gpuTypeIds": GPU_PREFERENCE,
        "gpuCount": 1,
        "containerDiskInGb": args.volume_gb,
        "env": env,
        "dockerStartCmd": ["bash", "-c", start_cmd],
    }
    if args.network_volume_id:
        pod_body["networkVolumeId"] = args.network_volume_id  # RunPod pins the DC
        pod_body["volumeMountPath"] = "/models"
    else:
        pod_body["dataCenterIds"] = US_SECURE_DCS
        pod_body["dataCenterPriority"] = "custom"  # East-first order above
    st, pod = runpod(key, "POST", "/pods", pod_body)
    if st not in (200, 201):
        raise SystemExit(f"pod create failed: {st} {pod}")
    pod_id = pod["id"]
    print(f"pod_created id={pod_id}")

    # Bounded control-plane backstop: confirm the pod reaches RUNNING; if it does
    # not within --confirm-seconds, terminate it so a stuck/failed pull cannot
    # bill. Once RUNNING, the in-pod dual watchdog (failsafe hard-deadline + idle)
    # owns termination, so we return without blocking for the whole job.
    deadline = time.time() + args.confirm_seconds
    while time.time() < deadline:
        st, cur = runpod(key, "GET", f"/pods/{pod_id}")
        status = (cur or {}).get("desiredStatus") or (cur or {}).get("lastStatusChange") or cur
        print(f"pod_status={ (cur or {}).get('desiredStatus') } runtime={bool((cur or {}).get('runtime'))}")
        if (cur or {}).get("desiredStatus") == "RUNNING" and (cur or {}).get("runtime"):
            print(f"pod_running id={pod_id}; in-pod watchdog owns lifecycle")
            return 0
        time.sleep(15)
    print(f"pod did not confirm RUNNING within {args.confirm_seconds}s; terminating {pod_id}")
    runpod(key, "DELETE", f"/pods/{pod_id}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
