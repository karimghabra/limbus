"""Validation of a flow analysis against a synthetic burst's ground truth.

The flow notebook (notebooks/flow.py, from the flow analysis branch) writes, for the burst it analysed,
flow_outputs/<burst>/velocities.csv (one row per centreline segment: its red-cell velocity in px/frame, positive from
its first point to its last, and a status) and flow_maps.npz (the perfusion map and the centreline points, in the
reference frame of the fixation it analysed). compare_flow() puts those segments on the truth:

1. The fixation's reference frame differs from the truth's frame 0 by the eye's motion. The translation between
   them is found by phase correlation of the perfused map with the truth's frame-0 vessel map (rotation in a
   fixation is a few hundredths of a degree, under a pixel at the border).
2. Along every measured segment, the truth's cycle-averaged projected red-cell velocity (masks.npz,
   mean_velocity_frame0) is sampled at the segment's points, projected on the segment's own direction and averaged:
   that is the velocity the segment should have measured, in the same px/frame and sign convention. The truth
   segment under most of its points names the vessel and its kind.

The result is one row per measured segment with both velocities, the true vessel kind, diameter and depth.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np


def _registration(perfused, truth_edges):
    """Translation (dx, dy) that maps fixation-frame pixels onto truth frame-0 pixels."""
    a = cv2.GaussianBlur(perfused.astype(np.float32), (0, 0), 2.0)
    b = cv2.GaussianBlur((truth_edges > 0).astype(np.float32), (0, 0), 2.0)
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    a, b = a[:h, :w], b[:h, :w]
    window = cv2.createHanningWindow((w, h), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(a - a.mean(), b - b.mean(), window)
    return float(dx), float(dy), float(response)


def compare_flow(burst_folder, flow_folder):
    """Rows comparing every measured segment with the truth, and the registration found."""
    burst_folder, flow_folder = Path(burst_folder), Path(flow_folder)
    masks = np.load(burst_folder / "truth" / "masks.npz")
    graph = json.loads((burst_folder / "truth" / "graph.json").read_text(encoding="utf-8"))
    edges0 = masks["edge_frame0"].astype(np.int64)
    velocity0 = masks["mean_velocity_frame0"].astype(np.float32)          # (2, h, w) px/frame, x and y
    maps = np.load(flow_folder / "flow_maps.npz")
    dx, dy, response = _registration(maps["perfused"], edges0)
    points = maps["segment_points"]
    with (flow_folder / "velocities.csv").open(newline="", encoding="utf-8") as handle:
        measured = {int(row["segment"]): row for row in csv.DictReader(handle)}
    h, w = edges0.shape
    # a little tolerance: the nearest truth vessel pixel within 2 px of each centreline point
    grown = edges0.copy()
    for _ in range(2):
        grown = np.where(grown > 0, grown, cv2.dilate(grown.astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int64))
    vx = velocity0[0].copy()
    vy = velocity0[1].copy()
    for _ in range(2):
        empty = edges0 == 0
        vx = np.where(empty & (grown > 0), cv2.dilate(np.abs(vx), np.ones((3, 3), np.uint8)) * np.sign(cv2.blur(vx, (3, 3))), vx)
        vy = np.where(empty & (grown > 0), cv2.dilate(np.abs(vy), np.ones((3, 3), np.uint8)) * np.sign(cv2.blur(vy, (3, 3))), vy)
    rows = []
    for segment, row in measured.items():
        select = points[:, 0] == segment
        if select.sum() < 5:
            continue
        rc = points[select, 1:]
        x = rc[:, 1] + dx
        y = rc[:, 0] + dy
        xi = np.clip(np.round(x).astype(int), 0, w - 1)
        yi = np.clip(np.round(y).astype(int), 0, h - 1)
        ids = grown[yi, xi]
        on = ids > 0
        if on.mean() < 0.5:
            rows.append(dict(segment=segment, status=row["status"], v_measured=float(row["v_px_frame"]), v_true=np.nan,
                             kind="none", diameter_um=np.nan, depth_um=np.nan, on_truth=float(on.mean())))
            continue
        tangent = np.gradient(np.stack([x, y], 1), axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
        along = vx[yi, xi] * tangent[:, 0] + vy[yi, xi] * tangent[:, 1]
        v_true = float(np.median(along[on]))
        edge = int(np.bincount(ids[on]).argmax()) - 1
        info = graph["edges"][edge]
        rows.append(dict(segment=segment, status=row["status"], v_measured=float(row["v_px_frame"]), v_true=v_true,
                         kind=info["kind"], diameter_um=info["diameter_um"], depth_um=info["depth_mean_um"],
                         on_truth=float(on.mean())))
    return rows, dict(dx=dx, dy=dy, response=response)


def summarise(rows, statuses=("confirmed", "consistent"), search_limit=20.0):
    """Agreement of the measured velocities with the truth over the segments with the given statuses.

    search_limit is the largest speed the flow analysis searches (px/frame; flow.py's MAX_VELOCITY): faster vessels
    alias, so the agreement is also given over the segments whose true speed is within it."""
    from scipy.stats import spearmanr

    use = [r for r in rows if r["status"] in statuses and np.isfinite(r["v_true"])]
    if not use:
        return {"segments": 0}
    m = np.array([r["v_measured"] for r in use])
    t = np.array([r["v_true"] for r in use])
    moving = np.abs(t) > 0.3
    reach = moving & (np.abs(t) <= search_limit)

    def agreement(select):
        if select.sum() < 3:
            return {}
        speed_m, speed_t = np.abs(m[select]), np.abs(t[select])
        return {
            "direction agrees (share)": float(np.mean(np.sign(m[select]) == np.sign(t[select]))),
            "median relative speed error": float(np.median(np.abs(speed_m - speed_t) / speed_t)),
            "speed ratio measured/true (median)": float(np.median(speed_m / speed_t)),
            "speed rank correlation (Spearman)": float(spearmanr(speed_m, speed_t)[0]),
            "speed correlation (Pearson)": float(np.corrcoef(speed_m, speed_t)[0, 1]),
        }

    out = {
        "segments": len(use),
        "on a vessel with flow (true speed > 0.3 px/frame)": int(moving.sum()),
        f"of those, true speed within the search limit ({search_limit:g} px/frame)": int(reach.sum()),
        "median true speed (px/frame)": float(np.median(np.abs(t))),
        "median velocity error (px/frame)": float(np.median(np.abs(m - t))),
    }
    out.update({f"all: {k}": v for k, v in agreement(moving).items()})
    out.update({f"within the limit: {k}": v for k, v in agreement(reach).items()})
    return out
