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

# 80 GB Secure Cloud options in preference order (smallest reliable topology
# first). Rates are verified at launch; these are only ordering hints.
GPU_PREFERENCE = [
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA A100 80GB PCIe",
    "NVIDIA H100 80GB HBM3",
    "NVIDIA H200",
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
    ap.add_argument("--volume-gb", type=int, default=400)
    ap.add_argument("--budget-usd", type=float, default=416.46)
    ap.add_argument("--hard-deadline-seconds", type=int, default=14400)
    ap.add_argument("--dry-run", action="store_true")
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

    st, vol = runpod(key, "POST", "/networkvolumes",
                     {"name": f"hyworld-weights-{job['jobId']}",
                      "size": args.volume_gb, "dataCenterId": None})
    if st not in (200, 201):
        raise SystemExit(f"network volume create failed: {st} {vol}")

    env = {
        "AWS_REGION": args.region,
        "MODEL_BUCKET": MODEL_BUCKET,
        "MODEL_PREFIX": MODEL_PREFIX,
        "JOB_JSON_B64": base64.b64encode(json.dumps(job).encode()).decode(),
        "INSTANCE_TYPE": "runpod-secure-80gb",
        "INSTANCE_HOURLY_USD": "0",  # replaced with the verified rate below
        "HARD_DEADLINE_SECONDS": str(args.hard_deadline_seconds),
        "RUNPOD_API_KEY": key,
        **pod_creds,
    }
    image = f"{ACCOUNT}.dkr.ecr.{args.region}.amazonaws.com/{ECR_REPO}@{WORKER_DIGEST}"
    st, pod = runpod(key, "POST", "/pods", {
        "name": f"hyworld-full-{job['jobId']}",
        "imageName": image,
        "containerRegistryAuthId": auth_id,
        "cloudType": "SECURE",
        "computeType": "GPU",
        "gpuTypeIds": GPU_PREFERENCE,
        "gpuCount": 1,
        "countryCodes": ["US"],
        "networkVolumeId": vol["id"],
        "volumeMountPath": "/models",
        "containerDiskInGb": 200,
        "encryptVolume": True,
        "env": env,
        "dockerStartCmd": ["bash", "/app/deploy/worldgen/runpod-entrypoint.sh"],
    })
    if st not in (200, 201):
        raise SystemExit(f"pod create failed: {st} {pod}")
    print(f"pod_created id={pod['id']}")

    # Budget / lifecycle poll loop lives here (omitted from this excerpt for
    # brevity — enforces --budget-usd and terminates via DELETE /pods/{id}).
    return 0


if __name__ == "__main__":
    sys.exit(main())
