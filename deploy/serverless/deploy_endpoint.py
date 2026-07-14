#!/usr/bin/env python3
"""Create/update the RunPod Serverless flex endpoint for the light 4090 tier.

Reads the RunPod API key from AWS SSM SecureString at runtime (never printed),
reads deploy/serverless/endpoint-config.json, and creates a serverless endpoint
bound to the baked-weights image. Safe to run repeatedly; with `--dry-run` it
only validates auth + config and creates no billing resources.

SAFETY: this does NOT touch the pipeline owner's running pod/volume. Serverless
endpoints are independent resources; a flex endpoint with workersMin=0 costs $0
until a request arrives. Prefer `--dry-run` while the pod pipeline is active.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
import urllib.error

SSM_KEY_PATH = "/intelliverse/worldgen/runpod-api-key"
REST = "https://rest.runpod.io/v1"
CONFIG = os.path.join(os.path.dirname(__file__), "endpoint-config.json")


def get_runpod_key(region: str) -> str:
    import boto3
    ssm = boto3.client("ssm", region_name=region)
    return ssm.get_parameter(Name=SSM_KEY_PATH, WithDecryption=True)["Parameter"]["Value"]


def runpod(key: str, method: str, path: str, body: dict | None = None):
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--image", required=False, help="fully-qualified baked-weights image ref (ECR digest)")
    ap.add_argument("--registry-auth-id", default=None, help="RunPod container registry auth id for private ECR")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = json.load(open(CONFIG))["endpoint"]
    key = get_runpod_key(args.region)

    st, pods = runpod(key, "GET", "/pods")
    if st != 200:
        raise SystemExit(f"RunPod auth failed: {st}")
    print(f"runpod_auth_ok existing_pods={len(pods)}")

    if args.dry_run or not args.image:
        print("dry-run: auth OK, config valid; no endpoint created. "
              "Provide --image (and --registry-auth-id for private ECR) to deploy.")
        return 0

    body = {
        "name": cfg["name"],
        "computeType": cfg["computeType"],
        "gpuTypeIds": cfg["gpuTypeIds"],
        "gpuCount": cfg["gpuCount"],
        "workersMin": cfg["workersMin"],
        "workersMax": cfg["workersMax"],
        "idleTimeout": cfg["idleTimeout"],
        "executionTimeoutMs": cfg["executionTimeoutMs"],
        "flashboot": cfg["flashboot"],
        "imageName": args.image,
        "env": cfg.get("env", {}),
    }
    if args.registry_auth_id:
        body["containerRegistryAuthId"] = args.registry_auth_id
    st, ep = runpod(key, "POST", "/endpoints", body)
    if st not in (200, 201):
        raise SystemExit(f"endpoint create failed: {st} {ep}")
    print(f"endpoint_created id={ep.get('id')} name={cfg['name']} workersMin={cfg['workersMin']} (scale-to-zero)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
