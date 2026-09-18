"""Non-rigid refinement — EXPERIMENTAL (METHODS.md §13).

Translation alignment leaves the corners of wide-field bursts doubled: the
eye rotates and its distance to the camera changes slightly between
fixations, so one shift can't align the whole field. Starting from the
translation result, each frame's remaining deformation is measured patch by
patch against a template (NoRMCorre-style: Pnevmatikakis & Giovannucci,
J Neurosci Methods 2017) and modelled as

    a robust per-frame affine   (rotation, magnification, shear)
  + a bounded, smoothed local residual at the patch centres

Iterated with template rebuilding, seeded by a single-pose reference. Frames
that still disagree with the template are rejected. Every original frame is
resampled once, with its combined field.

Experimental: it reached a single mode on real bursts, but has not yet been
validated against ground truth the way translation has (§12).
"""
import csv
import json
import os
import shutil
import time
from datetime import datetime, timezone

import cv2
import numpy as np
import tifffile

from . import __version__
from .bursts import load_burst, read_frame
from .config import NonrigidParams, Params
from .fields import FieldEvaluator, Grid, save_fields, to_full_resolution
from .metrics import mask_consensus, residual_motion, rotation_diagnostic
from .pipeline import process_burst
from .quality import robust_z
from .register import phase_correlate, shift
from .report import write_qc_png
from .vessels import glare_mask

METHOD = "nonrigid"
OUTPUT_FILES = ("mean_raw.tif", "mean_stabilized.tif", "std_stabilized.tif",
                "consensus_mask.tif", "transforms.csv", "qc.png")


def process_burst_nonrigid(path, out_base, params=None, nparams=None, log=print):
    """Translation into <out_base>/translation/<burst>, then non-rigid
    refinement into <out_base>/nonrigid/<burst>."""
    params = params or Params()
    nparams = nparams or NonrigidParams()
    started = time.time()
    name = os.path.basename(os.path.normpath(path))
    tr_dir = os.path.join(out_base, "translation", name)
    out_dir = os.path.join(out_base, METHOD, name)
    # a crash part-way must not leave an older result that looks current
    stale = os.path.join(out_dir, "metrics.json")
    if os.path.exists(stale):
        os.remove(stale)

    log("stage 1/2: translation")
    rec = process_burst(path, os.path.join(out_base, "translation"), params, log=log,
                        keep_work=True, method="translation")
    work_dir = rec.pop("_work_dir", None)
    try:
        if rec.get("status") != "ok":
            return _write_skip(out_dir, rec, rec.get("skip_reason", ""), params, nparams, started)
        log("stage 2/2: non-rigid refinement (experimental)")
        return _refine(load_burst(path), rec, tr_dir, work_dir, out_dir, params, nparams,
                       log, started)
    finally:
        if work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)


# ---- helpers ---------------------------------------------------------------------

def _processing(params, nparams, started):
    return {"finished_utc": datetime.now(timezone.utc).isoformat(),
            "seconds": round(time.time() - started, 1),
            "stabilize_version": __version__,
            "params_hash": nparams.params_hash(params),
            "nonrigid_params": nparams.to_dict()}


def _write_skip(out_dir, rec, reason, params, nparams, started):
    record = {"status": "skipped", "skip_reason": reason, "method": METHOD,
              "experimental": True, "burst": rec.get("burst", {}),
              "processing": {**rec.get("processing", {}),
                             **_processing(params, nparams, started)}}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    return record


def _nanmedian3(a):
    """3x3 median over the finite neighbours; fills gaps from their neighbours."""
    out = a.copy()
    rows, cols = a.shape
    for r in range(rows):
        for c in range(cols):
            block = a[max(0, r - 1):r + 2, max(0, c - 1):c + 2]
            if np.isfinite(block).any():
                out[r, c] = np.nanmedian(block)
    return out


def fit_robust(nx, ny, dx, dy, wts, model):
    """Robust (IRLS, Huber) fit of patch displacements, as a 2x3 matrix A with
    d(x) = A @ [x, y, 1]. 'affine' has 6 unknowns (rotation, magnification,
    shear, translation); 'similarity' has 4 (rotation, uniform magnification,
    translation) for grids too short to pin down shear."""
    x, y = nx.ravel().astype(np.float64), ny.ravel().astype(np.float64)
    n = x.size
    one, zero = np.ones(n), np.zeros(n)
    d = np.concatenate([dx.ravel(), dy.ravel()]).astype(np.float64)
    if model == "affine":
        X = np.vstack([np.stack([x, y, one, zero, zero, zero], 1),
                       np.stack([zero, zero, zero, x, y, one], 1)])
    else:   # dx = a x - b y + tx,  dy = b x + a y + ty
        X = np.vstack([np.stack([x, -y, one, zero], 1),
                       np.stack([y, x, zero, one], 1)])
    w0 = wts.ravel().astype(np.float64)
    wv = w0.copy()
    for _ in range(4):
        sw = np.sqrt(np.concatenate([wv, wv]))
        coef, *_ = np.linalg.lstsq(X * sw[:, None], d * sw, rcond=None)
        fit = X @ coef
        r = np.hypot(d[:n] - fit[:n], d[n:] - fit[n:])
        k = 1.5 * max(np.median(r[wv > 0]) if (wv > 0).any() else 1.0, 0.25)
        wv = w0 * np.where(r <= k, 1.0, k / np.maximum(r, 1e-9))
    if model == "affine":
        return coef.reshape(2, 3)
    a, b, tx, ty = coef
    return np.array([[a, -b, tx], [b, a, ty]])


def compose(A_f, L_f, A_u, l_u, nx, ny, grid):
    """Field of 'apply f, then correct by u': g(x) = u(x) + f(x + u(x)).

    With f = M_f x + b_f + L_f and u = M_u x + b_u + l_u:
        affine  M = M_u + M_f + M_f M_u,   b = b_u + b_f + M_f b_u   (exact)
        local   l_u + M_f l_u + L_f(x + u(x)), sampled at the grid nodes
    """
    Mf, bf = A_f[:, :2], A_f[:, 2]
    Mu, bu = A_u[:, :2], A_u[:, 2]
    A = np.empty((2, 3))
    A[:, :2] = Mu + Mf + Mf @ Mu
    A[:, 2] = bu + bf + Mf @ bu
    ux = A_u[0, 0] * nx + A_u[0, 1] * ny + A_u[0, 2] + l_u[..., 0]
    uy = A_u[1, 0] * nx + A_u[1, 1] * ny + A_u[1, 2] + l_u[..., 1]
    qx = ((nx + ux - grid.x0) / grid.stride).astype(np.float32)
    qy = ((ny + uy - grid.y0) / grid.stride).astype(np.float32)
    L_at = np.stack([cv2.remap(np.ascontiguousarray(L_f[..., k]), qx, qy, cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE) for k in (0, 1)], -1)
    L = l_u + l_u @ Mf.T + L_at
    return A, L.astype(np.float32)


def tile_spread(V, frames, warp, grid_shape, scale):
    """Per-tile residual motion spread (full-res px): after alignment, each
    tile of each frame is registered to the same tile of the mean; the
    per-frame common shift is removed and the SD across frames is reported
    per tile. Near 0 everywhere = one pose; large in the corners = doubling."""
    rows, cols = grid_shape
    h, w = V.shape[1:]
    th, tw = h // rows, w // cols
    if th < 32 or tw < 32 or len(frames) < 2:
        return None
    mean = np.zeros((h, w), np.float32)
    for i in frames:
        mean += warp(V[i], i)
    mean /= len(frames)
    win = cv2.createHanningWindow((tw, th), cv2.CV_32F)
    res = np.zeros((len(frames), rows, cols, 2))
    for n, i in enumerate(frames):
        a = warp(V[i], i)
        for r in range(rows):
            for c in range(cols):
                ys, xs = slice(r * th, (r + 1) * th), slice(c * tw, (c + 1) * tw)
                (ddx, ddy), _ = phase_correlate(mean[ys, xs], a[ys, xs], win)
                res[n, r, c] = (ddx / scale, ddy / scale)
    det = res - np.median(res.reshape(len(frames), -1, 2), axis=1)[:, None, None, :]
    return np.hypot(*np.std(np.clip(det, -30, 30), axis=0).transpose(2, 0, 1))


# ---- the refinement ------------------------------------------------------------

def _refine(burst, rec_t, tr_dir, work_dir, out_dir, params, np_, log, started):
    S = float(rec_t["processing"]["working_scale"])
    V = np.load(os.path.join(work_dir, "V.npy"), mmap_mode="r")
    M = np.load(os.path.join(work_dir, "M.npy"), mmap_mode="r")
    O = np.load(os.path.join(work_dir, "O.npy"), mmap_mode="r")
    with open(os.path.join(tr_dir, "transforms.csv"), encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    traj_px = np.array([[float(r["dx_px"]), float(r["dy_px"])] for r in rows])
    traj_ws = traj_px * S
    good = np.array([r["passed_gate"] == "1" for r in rows])
    idx = np.flatnonzero([r["registered"] == "1" for r in rows])
    m = idx.size
    h, w = V.shape[1:]

    # ---- patch grid ----------------------------------------------------------
    patch = max(2, int(round(np_.patch_px * S / 2)) * 2)
    stride = max(1, int(round(np_.stride_px * S)))
    ys = list(range(0, h - patch + 1, stride))
    xs = list(range(0, w - patch + 1, stride))
    R, C = len(ys), len(xs)
    if R < 2 or C < 2:
        return _fallback(burst, rec_t, tr_dir, out_dir, idx, traj_px, params, np_, started,
                         f"patch grid {R}x{C} can't measure deformation "
                         f"({patch / S:.0f} px patches on a {burst.width}x{burst.height} frame)",
                         log)
    # two rows of patches barely constrain vertical gradients: shear and
    # anisotropic scale would be fitted to noise, so only rotation and
    # uniform magnification are allowed
    model = "affine" if R >= 3 and C >= 3 else "similarity"
    grid = Grid(xs[0] + (patch - 1) / 2, ys[0] + (patch - 1) / 2, stride, R, C)
    ev = FieldEvaluator((h, w), grid)
    NX, NY = grid.nodes()
    win = cv2.createHanningWindow((patch, patch), cv2.CV_32F)
    max_dev = np_.max_local_px * S
    log(f"  patch grid {R}x{C} ({patch / S:.0f} px patches, stride {stride / S:.0f} px), "
        f"model {model}, {m} frames")

    A = np.zeros((m, 2, 3))
    A[:, :, 2] = traj_ws[idx]
    L = np.zeros((m, R, C, 2), np.float32)

    def aligned(k, img):
        fx, fy = ev.dense(A[k], L[k])
        return ev.warp(img, fx, fy)

    # ---- single-pose reference -----------------------------------------------
    # A template from all translation-aligned frames is itself doubled at the
    # corners, and registering against a doubled template is ambiguous: either
    # copy is a plausible match, pulling frames toward the average of the
    # poses instead of one of them. So seed with the most self-consistent run
    # of consecutive frames — one fixation, one pose.
    t0 = time.time()
    K = min(np_.reference_frames, m)
    small = np.stack([aligned(k, V[idx[k]])[::4, ::4].ravel() for k in range(m)])
    best_start, best_score = 0, -np.inf
    for s0 in range(0, m - K + 1, 3):
        block = small[s0:s0 + K]
        mean = block.mean(0)
        zm = (mean - mean.mean()) / (mean.std() + 1e-12)
        zb = (block - block.mean(1, keepdims=True)) / (block.std(1, keepdims=True) + 1e-12)
        score = float(np.mean(zb @ zm) / zm.size)
        if score > best_score:
            best_start, best_score = s0, score
    ref = np.arange(best_start, best_start + K)
    del small
    log(f"  reference: frames {idx[ref[0]]}-{idx[ref[-1]]} "
        f"(t {burst.times_s[idx[ref[0]]]:.2f}-{burst.times_s[idx[ref[-1]]]:.2f} s), "
        f"NCC to their mean {best_score:.3f} ({time.time() - t0:.0f}s)")

    # ---- iterate: template -> patch shifts -> affine + local -> compose -------
    history = []
    for it in range(1, np_.max_iterations + 1):
        t0 = time.time()
        members = ref if it <= np_.reference_iterations else range(m)
        acc = np.zeros((h, w), np.float32)
        cov = np.zeros((h, w), np.float32)
        for k in members:
            fx, fy = ev.dense(A[k], L[k])
            acc += ev.warp(V[idx[k]], fx, fy)
            cov += ev.warp(O[idx[k]], fx, fy) >= 0.5
        T = np.where(cov > 0, acc / np.maximum(cov, 1), 0).astype(np.float32)
        tp = [[T[y0:y0 + patch, x0:x0 + patch] for x0 in xs] for y0 in ys]

        changes, confs = np.zeros(m), []
        for k in range(m):
            a = aligned(k, V[idx[k]])
            rdx = np.full((R, C), np.nan, np.float32)
            rdy = np.full((R, C), np.nan, np.float32)
            wts = np.zeros((R, C), np.float32)
            for r, y0 in enumerate(ys):
                for c, x0 in enumerate(xs):
                    ap = a[y0:y0 + patch, x0:x0 + patch]
                    if tp[r][c].max() <= 0 or ap.max() <= 0:
                        continue
                    # copies inside: phaseCorrelate windows its inputs in place,
                    # which would erode the shared template patch frame by frame
                    (ddx, ddy), resp = phase_correlate(tp[r][c], ap, win)
                    if (resp >= np_.patch_min_response and abs(ddx) < patch / 4
                            and abs(ddy) < patch / 4):
                        rdx[r, c], rdy[r, c], wts[r, c] = ddx, ddy, resp
            valid = wts > 0
            confs.append(float(np.median(wts[valid])) if valid.any() else 0.0)
            if valid.sum() < np_.min_patches:
                continue                     # too little structure: keep the field
            frame_model = model
            if model == "affine" and (np.count_nonzero(valid.any(1)) < 2
                                      or np.count_nonzero(valid.any(0)) < 2):
                frame_model = "similarity"   # valid patches all in one row/column
            A_u = fit_robust(NX, NY, np.nan_to_num(rdx), np.nan_to_num(rdy), wts, frame_model)
            ax = A_u[0, 0] * NX + A_u[0, 1] * NY + A_u[0, 2]
            ay = A_u[1, 0] * NX + A_u[1, 1] * NY + A_u[1, 2]
            # local residual beyond the affine: bounded, gap-filled, smoothed
            lx = np.clip(rdx - ax, -max_dev, max_dev)
            ly = np.clip(rdy - ay, -max_dev, max_dev)
            lx[~valid] = np.nan
            ly[~valid] = np.nan
            l_u = np.stack([np.nan_to_num(_nanmedian3(lx)),
                            np.nan_to_num(_nanmedian3(ly))], -1).astype(np.float32)
            ufx, ufy = ev.dense(A_u, l_u)
            changes[k] = float(np.sqrt(np.mean(ufx ** 2 + ufy ** 2)))
            A[k], L[k] = compose(A[k], L[k], A_u, l_u, NX, NY, grid)
        changes /= S
        p90 = float(np.percentile(changes, 90))
        history.append(round(p90, 4))
        log(f"  iteration {it} ({'reference' if it <= np_.reference_iterations else 'all frames'}): "
            f"update median {np.median(changes):.2f} px, p90 {p90:.2f} px, max {changes.max():.2f} px, "
            f"patch confidence {np.median(confs):.3f} ({time.time() - t0:.0f}s)")
        # converge on the frames still MOVING: a median test stopped after one
        # round while the frames that caused doubling needed 5+ px more
        if it > np_.reference_iterations and p90 < np_.converge_p90_px:
            break

    # ---- reject frames that still disagree with the template -------------------
    acc = np.zeros((h, w), np.float32)
    cov = np.zeros((h, w), np.float32)
    for k in range(m):
        fx, fy = ev.dense(A[k], L[k])
        acc += ev.warp(V[idx[k]], fx, fy)
        cov += ev.warp(O[idx[k]], fx, fy) >= 0.5
    T = np.where(cov > 0, acc / np.maximum(cov, 1), 0)
    region = cov >= 0.5 * m
    ncc = np.array([np.corrcoef(aligned(k, V[idx[k]])[region], T[region])[0, 1]
                    for k in range(m)])
    ncc = np.nan_to_num(ncc)
    z = robust_z(ncc)
    keep = ~((z < np_.reject_z) & (ncc < np.median(ncc) - np_.reject_ncc_drop))
    used = idx[keep]
    log(f"  template agreement (NCC): median {np.median(ncc):.3f}, min {ncc.min():.3f}; "
        f"rejected {int((~keep).sum())} of {m}")
    if used.size < params.min_frames:
        return _write_skip(out_dir, rec_t, f"only {used.size} frames agreed with the "
                           f"non-rigid template", params, np_, started)

    # ---- metrics on the vessel masks, same as translation ----------------------
    t0 = time.time()
    row_of = {int(i): k for k, i in enumerate(idx)}
    cache = {}

    def warp_ws(img, i):
        k = row_of[int(i)]
        if cache.get("k") != k:
            cache["k"], cache["f"] = k, ev.dense(A[k], L[k])
        return ev.warp(img, *cache["f"])

    before, _ = mask_consensus(M, traj_ws, used, params, aligned=False, O=O)
    tr_after, _ = mask_consensus(M, traj_ws, used, params, aligned=True, O=O)
    after, consensus = mask_consensus(M, traj_ws, used, params, aligned=True, O=O, warp=warp_ws)
    residual = residual_motion(V, traj_ws, used, S, params, warp=warp_ws)
    t_acc = np.zeros((h, w), np.float32)
    t_cov = np.zeros((h, w), np.float32)
    for i in used:
        t_acc += warp_ws(V[i], i)
        t_cov += warp_ws(O[i], i)
    template_v = np.where(t_cov > 0, t_acc / np.maximum(t_cov, 1e-6), 0).astype(np.float32)
    rotation = rotation_diagnostic(V, traj_ws, template_v, used, S, params, warp=warp_ws)
    spread_tr = tile_spread(V, used, lambda img, i: shift(img, traj_ws[i]), np_.tile_grid, S)
    spread_nr = tile_spread(V, used, warp_ws, np_.tile_grid, S)
    log(f"  metrics: vessel overlap {before['overlap']:.3f} -> translation {tr_after['overlap']:.3f} "
        f"-> non-rigid {after['overlap']:.3f} (same {used.size} frames) ({time.time() - t0:.0f}s)")
    if spread_nr is not None:
        log(f"  tile residual spread, max over tiles: translation {spread_tr.max():.2f} px "
            f"-> non-rigid {spread_nr.max():.2f} px")

    # ---- full-resolution projections: ONE warp per original frame --------------
    t0 = time.time()
    A_full, L_full, grid_full = to_full_resolution(A, L, grid, S)
    H, W = burst.height, burst.width
    evf = FieldEvaluator((H, W), grid_full)
    fs = burst.full_scale
    raw_sum = np.zeros((H, W))
    st_sum = np.zeros((H, W))
    st_sq = np.zeros((H, W))
    cover = np.zeros((H, W))
    for k in np.flatnonzero(keep):
        img = read_frame(burst.files[idx[k]]).astype(np.float32)
        raw_sum += img
        fx, fy = evf.dense(A_full[k], L_full[k])
        wimg = evf.warp(img, fx, fy).astype(np.float64)
        seen = evf.warp((~glare_mask(img, fs, params, 1.0)).astype(np.float32), fx, fy) >= 0.5
        st_sum += wimg * seen
        st_sq += wimg * wimg * seen
        cover += seen
    nk = used.size
    valid = cover >= params.coverage_min_fraction * nk
    mean_raw = (raw_sum / nk).astype(np.float32)
    mean_st = np.full((H, W), np.nan, np.float32)
    std_st = np.full((H, W), np.nan, np.float32)
    mean_st[valid] = st_sum[valid] / cover[valid]
    std_st[valid] = np.sqrt(np.maximum(
        st_sq[valid] / cover[valid] - (st_sum[valid] / cover[valid]) ** 2, 0))
    log(f"  projections ({time.time() - t0:.0f}s)")

    # ---- outputs ---------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    tifffile.imwrite(os.path.join(out_dir, "mean_raw.tif"), mean_raw)
    tifffile.imwrite(os.path.join(out_dir, "mean_stabilized.tif"), mean_st)
    tifffile.imwrite(os.path.join(out_dir, "std_stabilized.tif"), std_st)
    mask_full = cv2.resize(consensus.astype(np.uint8) * 255, (W, H),
                           interpolation=cv2.INTER_NEAREST)
    tifffile.imwrite(os.path.join(out_dir, "consensus_mask.tif"), mask_full)
    save_fields(os.path.join(out_dir, "fields.npz"), idx, keep, A_full, L_full, grid_full,
                (H, W), model)
    ncc_of = {int(i): ncc[k] for k, i in enumerate(idx)}
    used_set = set(int(i) for i in used)
    _write_transforms(out_dir, rows, used_set, ncc_of)

    record = json.loads(json.dumps(rec_t))
    record.update({"method": METHOD, "experimental": True,
                   "stability_index": after["overlap"],
                   "usable_fraction": float(used.size / n)})
    record["frames"].update({"registered": int(used.size),
                             "translation_registered": int(m),
                             "rejected_by_template": int((~keep).sum())})
    tr_quality = rec_t["quality"]
    record["quality"] = {
        "before": before, "after": after,
        "translation_after_same_frames": tr_after, **residual,
        **{k: tr_quality[k] for k in ("closure_median_px", "closure_p90_px") if k in tr_quality},
    }
    record["diagnostics"] = {
        **rotation,
        "iterations": len(history),
        "convergence_p90_px": history,
        "translation_stage": rec_t["diagnostics"],
        "nonrigid": {
            "model": model, "grid_rows_cols": [R, C],
            "patch_px": round(patch / S, 1), "stride_px": round(stride / S, 1),
            "reference_frames": [int(idx[ref[0]]), int(idx[ref[-1]])],
            "reference_ncc": best_score,
            "template_ncc_median": float(np.median(ncc)),
            "template_ncc_min": float(ncc.min()),
            "tile_residual_spread_px": {
                "translation": None if spread_tr is None else np.round(spread_tr, 3).tolist(),
                "nonrigid": None if spread_nr is None else np.round(spread_nr, 3).tolist()},
            "tile_residual_spread_max_px": {
                "translation": None if spread_tr is None else float(spread_tr.max()),
                "nonrigid": None if spread_nr is None else float(spread_nr.max())},
        },
    }
    record["processing"].update(_processing(params, np_, started))
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    registered = np.zeros(n, bool)
    registered[used] = True
    write_qc_png(os.path.join(out_dir, "qc.png"), mean_raw, mean_st, traj_px,
                 burst.times_s, registered, good, record)
    log(f"  done: stability index {after['overlap']:.3f}, usable {100 * used.size / n:.0f}% "
        f"({time.time() - started:.0f}s total)")
    return record


def _write_transforms(out_dir, rows, used_set, ncc_of):
    """The translation stage's table, with 'registered' now meaning 'used by
    the non-rigid result'. The fields themselves are in fields.npz."""
    fields = list(rows[0].keys()) + ["translation_registered", "template_ncc"]
    with open(os.path.join(out_dir, "transforms.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for r in rows:
            i = int(r["index"])
            out = dict(r)
            out["translation_registered"] = r["registered"]
            out["registered"] = int(i in used_set)
            out["template_ncc"] = f"{ncc_of[i]:.4f}" if i in ncc_of else ""
            wr.writerow(out)


def _fallback(burst, rec_t, tr_dir, out_dir, idx, traj_px, params, np_, started, reason, log):
    """Frames too short for a patch grid: the result IS the translation result,
    stored in the non-rigid layout so every consumer reads it the same way."""
    log(f"  falling back to translation: {reason}")
    os.makedirs(out_dir, exist_ok=True)
    for name in OUTPUT_FILES:
        src = os.path.join(tr_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_dir, name))
    A = np.zeros((idx.size, 2, 3))
    A[:, :, 2] = traj_px[idx]
    save_fields(os.path.join(out_dir, "fields.npz"), idx, np.ones(idx.size, bool), A,
                np.zeros((idx.size, 0, 0, 2), np.float32), Grid(0, 0, 1, 0, 0),
                (burst.height, burst.width), "translation")
    record = json.loads(json.dumps(rec_t))
    record.update({"method": METHOD, "experimental": True})
    record["diagnostics"]["nonrigid"] = {"model": "translation (fallback)",
                                         "fallback_reason": reason}
    record["processing"].update(_processing(params, np_, started))
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    return record
