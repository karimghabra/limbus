"""Per-burst processing: gate -> vessel masks -> groupwise registration ->
metrics -> outputs. Raw burst folders are only ever read.
"""
import csv
import json
import os
import platform
import shutil
import time
from datetime import datetime, timezone

import cv2
import numpy as np
import tifffile

from . import __version__
from .bursts import has_image_data, load_burst, read_frame
from .metrics import (closure_error, mask_consensus, motion_metrics,
                      residual_motion, rotation_diagnostic)
from .quality import frame_stats, gate
from .register import build_templates, register_groupwise, shift
from .report import write_qc_png
from .vessels import (contrast_constant, envelope, glare_mask, to_working,
                      vesselness, working_sigmas)


def _working_scale(burst, params):
    return params.working_scale if burst.height >= params.min_height_to_downscale else 1.0


def _skip(out_dir, burst, reason, started):
    record = {
        "status": "skipped",
        "skip_reason": reason,
        "burst": {"name": burst.name, "path": burst.path, "frames": len(burst),
                  "pixel_format": burst.pixel_format},
        "processing": {"finished_utc": datetime.now(timezone.utc).isoformat(),
                       "seconds": round(time.time() - started, 2),
                       "stabilize_version": __version__},
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    return record


def process_burst(path, out_root, params, log=print):
    started = time.time()
    burst = load_burst(path)
    out_dir = os.path.join(out_root, burst.name)
    n = len(burst)
    log(f"{burst.name}: {n} frames {burst.width}x{burst.height} "
        f"{burst.pixel_format or '?'} @ {burst.fps:.1f} fps")

    if n < params.min_frames:
        return _skip(out_dir, burst, f"only {n} frames (< {params.min_frames})", started)
    if not has_image_data(burst):
        return _skip(out_dir, burst, "all pixel values are zero (no image data)", started)

    scale = _working_scale(burst, params)
    work_dir = os.path.join(out_dir, "_work")
    os.makedirs(work_dir, exist_ok=True)
    try:
        return _process(burst, out_dir, work_dir, scale, params, log, started)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _process(burst, out_dir, work_dir, scale, params, log, started):
    n = len(burst)
    fs = burst.full_scale

    # ---- pass 1: frame statistics and the quality gate (§3) -----------------
    t0 = time.time()
    log_sharp = np.zeros(n)
    means = np.zeros(n)
    clipped = np.zeros(n)
    for i, f in enumerate(burst.files):
        log_sharp[i], means[i], clipped[i] = frame_stats(
            to_working(read_frame(f), scale), fs, params)
    good, reasons, z_sharp, z_mean = gate(log_sharp, means, params)
    frames = np.flatnonzero(good)
    log(f"  gate: kept {frames.size}/{n} frames ({time.time() - t0:.0f}s)")
    if frames.size < params.min_frames:
        return _skip(out_dir, burst,
                     f"only {frames.size} frames passed the quality gate", started)

    # ---- burst-wide vesselness constants (§5, §6) ---------------------------
    ref_img = to_working(read_frame(burst.files[frames[frames.size // 2]]), scale)
    hw = ref_img.shape
    sigmas = working_sigmas(params, scale, hw[0])
    c = contrast_constant(ref_img, sigmas, params)
    sample = frames[np.linspace(0, frames.size - 1,
                                min(params.threshold_sample_frames, frames.size)).astype(int)]
    sample_v = []
    for i in sample:
        img = to_working(read_frame(burst.files[i]), scale)
        v = vesselness(img, sigmas, c, params)
        v[glare_mask(img, fs, params, scale)] = 0
        sample_v.append(v)
    threshold = float(np.percentile(np.stack(sample_v), params.envelope_percentile))

    # ---- pass 2: vesselness and envelope for every frame, on disk ----------
    t0 = time.time()
    V = np.lib.format.open_memmap(os.path.join(work_dir, "V.npy"), "w+",
                                  np.float32, (n,) + hw)
    M = np.lib.format.open_memmap(os.path.join(work_dir, "M.npy"), "w+",
                                  np.uint8, (n,) + hw)
    # O: what each frame actually observed. Glare-hidden pixels are unknown,
    # not 'not vessel' — and because glare is fixed to the camera, after
    # alignment its hole falls on different tissue in every frame (§4, §9)
    O = np.lib.format.open_memmap(os.path.join(work_dir, "O.npy"), "w+",
                                  np.uint8, (n,) + hw)
    for i, f in enumerate(burst.files):
        img = to_working(read_frame(f), scale)
        v = vesselness(img, sigmas, c, params)
        glare = glare_mask(img, fs, params, scale)
        v[glare] = 0
        V[i] = v
        M[i] = envelope(v, threshold)
        O[i] = ~glare
    V.flush()
    M.flush()
    O.flush()
    log(f"  vessel masks: sigmas {[round(s / scale, 1) for s in sigmas]} px, "
        f"{time.time() - t0:.0f}s")

    # ---- registration (§8) ---------------------------------------------------
    t0 = time.time()
    reg = register_groupwise(V, M, good, params, scale, O=O, log=log)
    traj_ws = reg["traj"]
    traj_px = traj_ws / scale
    registered = reg["registered"] & good
    used = np.flatnonzero(registered)
    log(f"  registration: {reg['iterations']} iterations, "
        f"{used.size} frames registered ({time.time() - t0:.0f}s)")

    # ---- metrics (§9, §10) ---------------------------------------------------
    t0 = time.time()
    motion = motion_metrics(traj_px, burst.times_s, registered, params)
    before, _ = mask_consensus(M, traj_ws, used, params, aligned=False, O=O)
    after, consensus = mask_consensus(M, traj_ws, used, params, aligned=True, O=O)
    _, template_v, _ = build_templates(V, M, traj_ws, used, O)
    residual = residual_motion(V, traj_ws, used, scale, params)
    closure = closure_error(V, used, scale, params)
    rotation = rotation_diagnostic(V, traj_ws, template_v, used, scale, params)
    log(f"  metrics: vessel overlap {before['overlap']:.3f} -> {after['overlap']:.3f}, "
        f"Dice {before['dice']:.3f} -> {after['dice']:.3f} ({time.time() - t0:.0f}s)")

    # ---- pass 3: raw and stabilized projections at full resolution ----------
    t0 = time.time()
    shape = (burst.height, burst.width)
    raw_sum = np.zeros(shape, np.float64)
    st_sum = np.zeros(shape, np.float64)
    st_sq = np.zeros(shape, np.float64)
    cover = np.zeros(shape, np.float64)
    for i in used:
        img = read_frame(burst.files[i]).astype(np.float32)
        raw_sum += img
        # a single resampling pass from the original frame: warping repeatedly
        # would blur the fine detail vessel analysis needs
        w = shift(img, traj_px[i]).astype(np.float64)
        # leave glare out: fixed to the camera, it would otherwise be smeared
        # across the tissue by the alignment (§4)
        seen = (shift((~glare_mask(img, fs, params, 1.0)).astype(np.float32),
                      traj_px[i]) >= 0.5)
        st_sum += w * seen
        st_sq += w * w * seen
        cover += seen
    valid = cover >= params.coverage_min_fraction * max(used.size, 1)
    mean_raw = (raw_sum / max(used.size, 1)).astype(np.float32)
    mean_st = np.full(shape, np.nan, np.float32)
    std_st = np.full(shape, np.nan, np.float32)
    mean_st[valid] = st_sum[valid] / cover[valid]
    std_st[valid] = np.sqrt(np.maximum(
        st_sq[valid] / cover[valid] - (st_sum[valid] / cover[valid]) ** 2, 0))
    log(f"  projections ({time.time() - t0:.0f}s)")

    # ---- outputs -------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    tifffile.imwrite(os.path.join(out_dir, "mean_raw.tif"), mean_raw)
    tifffile.imwrite(os.path.join(out_dir, "mean_stabilized.tif"), mean_st)
    tifffile.imwrite(os.path.join(out_dir, "std_stabilized.tif"), std_st)
    mask_full = cv2.resize(consensus.astype(np.uint8) * 255, (burst.width, burst.height),
                           interpolation=cv2.INTER_NEAREST)
    tifffile.imwrite(os.path.join(out_dir, "consensus_mask.tif"), mask_full)

    with open(os.path.join(out_dir, "transforms.csv"), "w", newline="",
              encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["index", "filename", "time_s", "dx_px", "dy_px",
                     "coarse_response", "fine_response", "passed_gate",
                     "gate_reason", "registered", "sharpness_z", "brightness_z",
                     "clipped_fraction"])
        for i, fpath in enumerate(burst.files):
            wr.writerow([i, os.path.basename(fpath), f"{burst.times_s[i]:.6f}",
                         f"{traj_px[i, 0]:.4f}", f"{traj_px[i, 1]:.4f}",
                         f"{reg['coarse_response'][i]:.4f}",
                         f"{reg['fine_response'][i]:.4f}", int(good[i]),
                         reasons[i], int(registered[i]), f"{z_sharp[i]:.3f}",
                         f"{z_mean[i]:.3f}", f"{clipped[i]:.5f}"])

    rejected = {}
    for r in reasons:
        if r:
            rejected[r] = rejected.get(r, 0) + 1
    record = {
        "status": "ok",
        "stability_index": after["overlap"],
        "usable_fraction": float(used.size / n),
        "burst": {
            "name": burst.name, "path": burst.path, "frames": n,
            "width": burst.width, "height": burst.height,
            "pixel_format": burst.pixel_format, "bit_depth": burst.bit_depth,
            "fps": burst.fps, "roi_top_row": burst.offset_y,
            "duration_s": float(burst.times_s[-1]) if n else 0.0,
        },
        "frames": {"total": n, "passed_gate": int(good.sum()),
                   "rejected_by_gate": rejected, "registered": int(used.size)},
        "motion": motion,
        "quality": {"before": before, "after": after, **residual, **closure},
        "diagnostics": {**rotation, "iterations": reg["iterations"],
                        "convergence_rms_px": reg["convergence_px"]},
        "processing": {
            "working_scale": scale,
            "sigmas_px": [round(s / scale, 2) for s in sigmas],
            "contrast_constant": c,
            "envelope_threshold": threshold,
            "params": params.to_dict(),
            "seconds": round(time.time() - started, 1),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "stabilize_version": __version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "tifffile": tifffile.__version__,
        },
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    write_qc_png(os.path.join(out_dir, "qc.png"), mean_raw, mean_st, traj_px,
                 burst.times_s, registered, good, record)
    log(f"  done: stability index {after['overlap']:.3f}, "
        f"usable {100 * used.size / n:.0f}% ({time.time() - started:.0f}s total)")
    return record
