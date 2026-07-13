"""ODCR lifecycle controller for the isolated HY-World full-stack pool.

Runs every five minutes. It creates at most one tagged p5.4xlarge reservation
when work exists, and cancels that reservation only after the Redis signal and
processing queues have both stayed empty for 15 minutes. Before cancellation
it scales the worker to zero, waits for worker pods to disappear, requests
deletion of the dedicated NodeClaims, and leaves a durable audit state in the
private HY-World bucket.
"""

from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import boto3
import redis
from botocore.exceptions import ClientError


REGION = os.environ.get("AWS_REGION", "us-east-1")
INSTANCE_TYPE = os.environ.get("INSTANCE_TYPE", "p5.4xlarge")
ZONES = tuple(
    zone.strip()
    for zone in os.environ.get(
        "CAPACITY_ZONES", "us-east-1a,us-east-1b,us-east-1c"
    ).split(",")
    if zone.strip()
)
IDLE_SECONDS = int(os.environ.get("IDLE_SECONDS", "900"))
BUCKET = os.environ["AWS_S3_BUCKET_NAME"]
STATE_KEY = os.environ.get(
    "LIFECYCLE_STATE_KEY", "worldgen-full-ops/capacity-lifecycle.json"
)
QUEUE = os.environ.get("WORLDGEN_QUEUE", "pipeline:signal:worldgen-full")
PROCESSING_QUEUE = os.environ.get(
    "WORLDGEN_PROCESSING_QUEUE", "pipeline:worldgen-full:processing"
)
DEPLOYMENT = os.environ.get("WORKER_DEPLOYMENT", "hy-world-full-worker")
NODEPOOL = os.environ.get("WORLDGEN_NODEPOOL", "worldgen-fullstack-p5")
NAMESPACE = os.environ.get("POD_NAMESPACE", "aicart")
HOURLY_USD = float(os.environ.get("INSTANCE_HOURLY_USD", "6.88"))
DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK", "")

ec2 = boto3.client("ec2", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)
r = redis.Redis(
    host=os.environ.get(
        "REDIS_HOST", "content-factory-redis.aicart.svc.cluster.local"
    ),
    port=int(os.environ.get("REDIS_PORT", "6379")),
    password=os.environ.get("REDIS_PASSWORD") or None,
    decode_responses=True,
    socket_connect_timeout=5,
    socket_timeout=10,
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(event: str, **fields: object) -> None:
    print(json.dumps({"at": now_iso(), "event": event, **fields}, sort_keys=True))


def alert(message: str) -> None:
    log("alert", message=message)
    if not DISCORD_WEBHOOK:
        return
    request = urllib.request.Request(
        DISCORD_WEBHOOK,
        data=json.dumps({"content": f"HY-WORLD CAPACITY: {message}"}).encode(),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "HY-World-Capacity-Lifecycle/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10):
            pass
    except Exception as error:
        log("alert_failed", error=type(error).__name__)


def load_state() -> dict:
    try:
        response = s3.get_object(Bucket=BUCKET, Key=STATE_KEY)
        return json.loads(response["Body"].read())
    except s3.exceptions.NoSuchKey:
        return {}
    except Exception as error:
        if getattr(error, "response", {}).get("Error", {}).get("Code") in {
            "NoSuchKey",
            "404",
        }:
            return {}
        raise


def save_state(state: dict) -> None:
    state["updatedAt"] = now_iso()
    s3.put_object(
        Bucket=BUCKET,
        Key=STATE_KEY,
        Body=json.dumps(state, indent=2).encode(),
        ContentType="application/json",
    )


def active_reservations() -> list[dict]:
    response = ec2.describe_capacity_reservations(
        Filters=[
            {"Name": "state", "Values": ["active"]},
            {"Name": "tag:ManagedBy", "Values": ["hy-world-fullstack"]},
            {"Name": "tag:Workload", "Values": ["worldgen-fullstack"]},
        ]
    )
    return [
        reservation
        for reservation in response.get("CapacityReservations", [])
        if reservation.get("InstanceType") == INSTANCE_TYPE
    ]


def create_reservation() -> dict | None:
    for zone in ZONES:
        try:
            response = ec2.create_capacity_reservation(
                InstanceType=INSTANCE_TYPE,
                InstancePlatform="Linux/UNIX",
                AvailabilityZone=zone,
                InstanceCount=1,
                InstanceMatchCriteria="open",
                Tenancy="default",
                EbsOptimized=True,
                TagSpecifications=[
                    {
                        "ResourceType": "capacity-reservation",
                        "Tags": [
                            {"Key": "ManagedBy", "Value": "hy-world-fullstack"},
                            {"Key": "Workload", "Value": "worldgen-fullstack"},
                        ],
                    }
                ],
            )
            reservation = response["CapacityReservation"]
            alert(
                f"created {reservation['CapacityReservationId']} for "
                f"{INSTANCE_TYPE} in {zone}; ${HOURLY_USD:.2f}/hour"
            )
            return reservation
        except ClientError as error:
            code = error.response["Error"]["Code"]
            log("reservation_attempt_failed", zone=zone, code=code)
            if code not in {
                "InsufficientInstanceCapacity",
                "InsufficientCapacity",
            }:
                raise
    alert(f"no {INSTANCE_TYPE} ODCR capacity in {','.join(ZONES)}")
    return None


def kube_request(path: str, method: str = "GET", body: dict | None = None) -> dict:
    token = open(
        "/var/run/secrets/kubernetes.io/serviceaccount/token", encoding="utf-8"
    ).read()
    ca_path = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
    request = urllib.request.Request(
        f"https://kubernetes.default.svc{path}",
        data=json.dumps(body).encode() if body is not None else None,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/merge-patch+json",
        },
        method=method,
    )
    context = ssl.create_default_context(cafile=ca_path)
    try:
        with urllib.request.urlopen(request, timeout=15, context=context) as response:
            payload = response.read()
            return json.loads(payload) if payload else {}
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"Kubernetes {method} {path}: {error.code} {detail}") from error


def scale_worker_zero() -> None:
    kube_request(
        f"/apis/apps/v1/namespaces/{NAMESPACE}/deployments/{DEPLOYMENT}/scale",
        method="PATCH",
        body={"spec": {"replicas": 0}},
    )


def worker_pods_exist() -> bool:
    selector = urllib.parse.quote("app=hy-world-full-worker")
    response = kube_request(
        f"/api/v1/namespaces/{NAMESPACE}/pods?labelSelector={selector}"
    )
    return any(
        not pod.get("metadata", {}).get("deletionTimestamp")
        for pod in response.get("items", [])
    )


def delete_nodeclaims() -> None:
    selector = urllib.parse.quote(f"karpenter.sh/nodepool={NODEPOOL}")
    response = kube_request(
        f"/apis/karpenter.sh/v1/nodeclaims?labelSelector={selector}"
    )
    for claim in response.get("items", []):
        name = claim["metadata"]["name"]
        kube_request(f"/apis/karpenter.sh/v1/nodeclaims/{name}", method="DELETE")
        log("nodeclaim_delete_requested", name=name)


def drain_idle_capacity(reservations: list[dict]) -> bool:
    alert("15-minute idle threshold reached; scaling worker to zero")
    scale_worker_zero()
    deadline = time.time() + 120
    while time.time() < deadline and worker_pods_exist():
        time.sleep(5)
    if worker_pods_exist():
        alert("reservation retained: worker pod did not terminate safely")
        return False
    delete_nodeclaims()
    for reservation in reservations:
        reservation_id = reservation["CapacityReservationId"]
        alert(f"canceling idle reservation {reservation_id}")
        ec2.cancel_capacity_reservation(CapacityReservationId=reservation_id)
        alert(f"canceled idle reservation {reservation_id}")
    return True


def main() -> None:
    r.ping()
    queued = r.llen(QUEUE)
    processing = r.llen(PROCESSING_QUEUE)
    reservations = active_reservations()
    state = load_state()
    log(
        "snapshot",
        queued=queued,
        processing=processing,
        reservations=[item["CapacityReservationId"] for item in reservations],
    )

    if queued or processing:
        if state.get("idleSince"):
            state.pop("idleSince", None)
            save_state(state)
        if not reservations:
            reservation = create_reservation()
            if reservation:
                save_state(
                    {
                        "reservationId": reservation["CapacityReservationId"],
                        "reservationZone": reservation["AvailabilityZone"],
                        "hourlyUsd": HOURLY_USD,
                    }
                )
        return

    if not reservations:
        if state.get("idleSince") or state.get("reservationId"):
            save_state({"lastObservedNoReservationAt": now_iso()})
        log("idle_no_reservation")
        return

    now = int(time.time())
    idle_since = int(state.get("idleSince") or now)
    if "idleSince" not in state:
        state["idleSince"] = idle_since
        save_state(state)
        alert("queues and active stages empty; 15-minute cancellation timer started")
        return

    idle_for = now - idle_since
    log("idle", seconds=idle_for)
    if idle_for < IDLE_SECONDS:
        return
    if not drain_idle_capacity(reservations):
        return
    save_state({"idleSince": now, "lastCanceledAt": now_iso()})


if __name__ == "__main__":
    main()
