"""
360deg panorama-shell coverage.

The four-stage reconstruction only covers the ~180-220deg frontal wedge the WorldNav
trajectory swept; the back hemisphere (yaw 135/180/225) is a pure-black void, which loses
the coverage A/B parameter to current production. The full 360deg pano DOES contain the
back-hemisphere content, so we reproject it onto a large background sphere of gaussians:
every azimuth/elevation gets a colored backdrop, killing the void while the real trained
geometry (in front) occludes the shell where it exists.

Shell gaussians are produced in the viewer's .splat record dtype and are added AFTER the
splat post-pass/tiers (they are intentionally large background splats; the oversized-tail
cull would otherwise remove them). Orientation is aligned to the reconstruction frame via
the saved camera trajectory (pano-center -> camera0 forward, world up = mean camera up).
"""

import functools
import json
from pathlib import Path

import numpy as np
from PIL import Image

print = functools.partial(print, flush=True)  # noqa: A001

SPLAT_DTYPE = [
    ("position", "<f4", 3), ("scale", "<f4", 3),
    ("color", "u1", 4), ("rotation", "u1", 4),
]
_IDENTITY_ROT = np.array([128, 128, 128, 255], dtype=np.uint8)  # packed (0,0,0,1)

# Reconstruction convention (from production spatial.json): OpenCV frame of the first view,
# +Z forward, +Y down => world up is -Y. Views were rendered from the pano center at the
# origin looking +Z, so pano-center maps to +Z and the shell aligns with these defaults even
# when a per-frame cameras.json is absent.
DEFAULT_FORWARD = np.array([0.0, 0.0, 1.0])
DEFAULT_UP = np.array([0.0, -1.0, 0.0])


def alignment_from_cameras(result_dir: Path):
    """Recover (forward, up, centroid_hint) from the saved camera trajectory json.

    Returns (forward, up) unit vectors in the reconstruction world frame, or (None, None)
    if no usable camera file is found (caller falls back to canonical axes)."""
    cams = None
    for name in ("cameras.json", "render_results/cameras.json", "camera.json"):
        p = result_dir / name
        if p.exists():
            try:
                cams = json.loads(p.read_text())
            except Exception:  # noqa: BLE001
                continue
            break
    if cams is None:
        return None, None
    # Accept a few shapes: list of {position, rotation|R|extrinsic} or {"extrinsic": [...]}.
    entries = cams if isinstance(cams, list) else cams.get("frames") or cams.get("cameras") or []
    fwds, ups = [], []
    for e in entries:
        R = None
        if isinstance(e, dict):
            if "extrinsic" in e:
                w2c = np.asarray(e["extrinsic"], dtype=np.float64).reshape(4, 4)
                R = np.linalg.inv(w2c)[:3, :3]
            elif "rotation" in e:
                R = np.asarray(e["rotation"], dtype=np.float64).reshape(3, 3)
        if R is None:
            continue
        # OpenCV/COLMAP c2w: forward = +Z col, up = -Y col.
        fwds.append(R[:, 2])
        ups.append(-R[:, 1])
    if not fwds:
        return None, None
    fwd = np.asarray(fwds[0], dtype=np.float64)
    up = np.mean(np.asarray(ups, dtype=np.float64), axis=0)
    fwd /= np.linalg.norm(fwd) + 1e-9
    up /= np.linalg.norm(up) + 1e-9
    # Re-orthogonalize forward against up.
    fwd = fwd - up * float(fwd @ up)
    fwd /= np.linalg.norm(fwd) + 1e-9
    return fwd, up


def _rotation_matrix(forward, up):
    """World<-canonical rotation with canonical +Z=forward, +Y=up, +X=right."""
    f = np.asarray(forward, dtype=np.float64)
    u = np.asarray(up, dtype=np.float64)
    right = np.cross(u, f)
    right /= np.linalg.norm(right) + 1e-9
    u2 = np.cross(f, right)
    return np.stack([right, u2, f], axis=1)  # columns map canonical axes -> world


def shell_gaussians(
    pano: Image.Image,
    positions: np.ndarray,
    *,
    forward=None,
    up=None,
    target: int = 350_000,
    radius_scale: float = 1.35,
    opacity: int = 235,
    scale_overlap: float = 1.6,
) -> np.ndarray:
    """Build background shell gaussians reprojecting `pano` onto a sphere around the scene."""
    centroid = np.median(positions, axis=0).astype(np.float64)
    d = np.linalg.norm(positions.astype(np.float64) - centroid, axis=1)
    extent = float(np.percentile(d, 95)) if len(d) else 1.0
    radius = max(extent * radius_scale, 1e-3)

    W, H = pano.size
    # Choose a lon/lat grid sized for ~target samples with 2:1 equirect aspect.
    n_lat = max(64, int(round(np.sqrt(target / 2.0))))
    n_lon = 2 * n_lat
    us = (np.arange(n_lon) + 0.5) / n_lon
    vs = (np.arange(n_lat) + 0.5) / n_lat
    uu, vv = np.meshgrid(us, vs)
    lon = (uu - 0.5) * 2.0 * np.pi
    lat = (0.5 - vv) * np.pi
    clat = np.cos(lat)
    dirs = np.stack([clat * np.sin(lon), np.sin(lat), clat * np.cos(lon)], axis=-1)
    dirs = dirs.reshape(-1, 3)

    if forward is None or up is None:
        forward, up = DEFAULT_FORWARD, DEFAULT_UP
    R = _rotation_matrix(forward, up)
    dirs = dirs @ R.T

    pano_rgb = np.asarray(pano.convert("RGB"))
    px = np.clip((uu * W).astype(np.int64), 0, W - 1).reshape(-1)
    py = np.clip((vv * H).astype(np.int64), 0, H - 1).reshape(-1)
    colors = pano_rgb[py, px]  # (N,3)

    n = dirs.shape[0]
    rec = np.zeros(n, dtype=SPLAT_DTYPE)
    rec["position"] = (centroid[None, :] + radius * dirs).astype(np.float32)
    # Isotropic scale sized so neighbours overlap: chord of one lon cell at the equator.
    cell = radius * (2.0 * np.pi / n_lon) * scale_overlap
    rec["scale"] = np.full((n, 3), cell, dtype=np.float32)
    rec["color"][:, :3] = colors.astype(np.uint8)
    rec["color"][:, 3] = np.uint8(opacity)
    rec["rotation"] = _IDENTITY_ROT
    print(f"[shell] {n} gaussians, radius={radius:.3f} (extent-p95 {extent:.3f}), "
          f"cell={cell:.4f}, fwd={np.round(forward,2)} up={np.round(up,2)}")
    return rec


def subsample(rec: np.ndarray, target: int) -> np.ndarray:
    if len(rec) <= target:
        return rec
    idx = np.linspace(0, len(rec) - 1, target).astype(np.int64)
    return rec[idx]
