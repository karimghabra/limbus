"""Stability metrics (METHODS.md §9–§10).

Two families, kept separate so eye motion and correction quality are never
conflated:

  motion      — how unstable the burst was (from the recovered trajectory)
  quality     — how well stabilization worked, judged from the vessel masks
                themselves rather than the registration's own estimates, so
                the method never grades its own homework
"""
import numpy as np

from .register import PhaseCorrelator, shift


def _robust_threshold(x, k):
    med = np.median(x)
    mad = 1.4826 * np.median(np.abs(x - med))
    return med + k * max(mad, 1e-9)


def motion_metrics(traj_px, times_s, registered, params):
    """Motion of the eye relative to the camera, in full-resolution pixels.

    Jitter is reported as a speed (px/s), not px per frame: bursts here run
    from 32 to 149 fps, and the step between frames scales with the time
    between them, so only a rate is comparable across bursts (§9).
    """
    idx = np.flatnonzero(registered)
    out = {"frames_registered": int(idx.size)}
    if idx.size < 3:
        return out
    t = times_s[idx]
    pos = traj_px[idx]
    centre = np.median(pos, axis=0)
    disp = np.hypot(*(pos - centre).T)
    out.update({
        "displacement_median_px": float(np.median(disp)),
        "displacement_p95_px": float(np.percentile(disp, 95)),
        "displacement_max_px": float(disp.max()),
        "range_x_px": float(np.ptp(pos[:, 0])),
        "range_y_px": float(np.ptp(pos[:, 1])),
    })
    dt = np.diff(t)
    ok = dt > 0
    speed = np.hypot(*np.diff(pos, axis=0).T)[ok] / dt[ok]
    if speed.size:
        cutoff = _robust_threshold(speed, params.saccade_mad_k)
        saccades = speed > cutoff
        steady = speed[~saccades] if (~saccades).any() else speed
        out.update({
            "speed_median_px_s": float(np.median(speed)),
            "jitter_rms_px_s": float(np.sqrt(np.mean(steady ** 2))),
            "saccades": int(saccades.sum()),
            "saccade_threshold_px_s": float(cutoff),
        })
    if np.ptp(t) > 0:
        # drift: slope of a straight-line fit to position against time
        slope = [np.polyfit(t, pos[:, k], 1)[0] for k in (0, 1)]
        out["drift_px_s"] = float(np.hypot(*slope))
    return out


def mask_consensus(M, traj_ws, frames, params, aligned, O=None, warp=None):
    """Vessel overlap and Dice of the vessel masks across frames (§9).

    overlap = probability that a pixel labelled vessel in one frame is also
    vessel in another frame. Per pixel, with k frames calling it vessel out of
    the n frames covering it:

        overlap = sum k(k-1) / sum k(n-1)

    an exact pairwise probability that excludes self-pairs. A still vessel
    concentrates k at n and scores near 1; motion smears the same vessel
    thinly over many pixels, spreading k and driving overlap down.

    (An earlier 'persistence' = mean |2p-1| over a band was dropped: pixels a
    smeared vessel only occasionally covered scored as strong agreement, so
    large motion inflated it. Caught by the synthetic ground-truth test.)

    Red-cell plasma gaps keep overlap below 1 even when perfectly aligned, so
    it compares bursts processed with the same parameters, not an absolute.

    n counts only frames that actually OBSERVED a pixel: glare is fixed to the
    camera, so after alignment its hole lands on different tissue in every
    frame, and counting those hidden pixels as 'not vessel' would score them
    as disagreement (O is the per-frame observed mask; None = all observed).
    Only pixels observed by at least coverage_min_fraction of frames count, so
    borders shifted in from outside the field can't distort the score either.

    warp(img, i) aligns frame i; the default shifts by traj_ws[i]. The
    non-rigid method passes its own, so both are scored identically.
    """
    from .register import observed

    warp = warp or (lambda img, i: shift(img, traj_ws[i]))

    h, w = M.shape[1:]
    msum = np.zeros((h, w), np.float32)
    cover = np.zeros((h, w), np.float32)
    for i in frames:
        m = M[i].astype(np.float32)
        obs = observed(O, i, (h, w))
        if aligned:
            # Re-binarise after warping. A sub-pixel shift interpolates mask
            # edges to fractions like 0.3 and 0.7; summing those blurs the
            # masks and deflates overlap for any burst that moved — penalising
            # motion that was correctly removed. Thresholding at 0.5 keeps the
            # sub-pixel edge position without the blur.
            seen = warp(obs, i) >= 0.5
            msum += (warp(m, i) >= 0.5) & seen
            cover += seen
        else:
            seen = obs >= 0.5
            msum += (m >= 0.5) & seen
            cover += seen
    valid = cover >= params.coverage_min_fraction * len(frames)
    p = np.zeros_like(msum)
    p[valid] = msum[valid] / cover[valid]
    consensus = (p >= 0.5) & valid
    k = msum[valid].astype(np.float64)
    n = cover[valid].astype(np.float64)
    denom = float(np.sum(k * (n - 1)))
    overlap = float(np.sum(k * (k - 1)) / denom) if denom > 0 else 0.0

    dices = []
    for i in frames:
        m = M[i].astype(np.float32)
        obs = observed(O, i, (h, w))
        if aligned:
            m = warp(m, i)
            obs = warp(obs, i)
        # compare only where this frame could see
        region = valid & (obs >= 0.5)
        mb = (m >= 0.5) & region
        cons = consensus & region
        denom = np.count_nonzero(mb) + np.count_nonzero(cons)
        if denom:
            dices.append(2 * np.count_nonzero(mb & cons) / denom)
    return {
        "overlap": overlap,
        "dice": float(np.mean(dices)) if dices else 0.0,
        "vessel_fraction": float(consensus.sum() / max(valid.sum(), 1)),
    }, consensus


def residual_motion(V, traj_ws, frames, scale, params, warp=None):
    """Motion left over after stabilization (§9): phase-correlate consecutive
    stabilized frames, which should now differ by ~0 px. Reported in
    full-resolution pixels, on an evenly spread subsample."""
    if len(frames) < 2:
        return {}
    warp = warp or (lambda img, i: shift(img, traj_ws[i]))
    pairs = list(zip(frames[:-1], frames[1:]))
    take = np.linspace(0, len(pairs) - 1,
                       min(params.diagnostic_frames, len(pairs))).astype(int)
    correlator = PhaseCorrelator(V.shape[1:])
    steps = []
    for k in take:
        a, b = pairs[k]
        d, _ = correlator(warp(V[a], a), warp(V[b], b))
        steps.append(np.hypot(*d) / scale)
    steps = np.array(steps)
    return {"residual_step_median_px": float(np.median(steps)),
            "residual_step_p90_px": float(np.percentile(steps, 90))}


def closure_error(V, frames, scale, params):
    """Estimator self-consistency without ground truth (§9): shifts must be
    transitive, so d(i->i+1) + d(i+1->i+2) - d(i->i+2) is pure estimator
    error. A lower bound — correlated errors can cancel — which is why the
    synthetic tests check against known shifts too."""
    if len(frames) < 3:
        return {}
    triples = list(zip(frames[:-2], frames[1:-1], frames[2:]))
    take = np.linspace(0, len(triples) - 1,
                       min(params.diagnostic_frames, len(triples))).astype(int)
    correlator = PhaseCorrelator(V.shape[1:])
    errs = []
    for k in take:
        a, b, c = triples[k]
        ab, _ = correlator(V[a], V[b])
        bc, _ = correlator(V[b], V[c])
        ac, _ = correlator(V[a], V[c])
        errs.append(np.hypot(*(ab + bc - ac)) / scale)
    errs = np.array(errs)
    return {"closure_median_px": float(np.median(errs)),
            "closure_p90_px": float(np.percentile(errs, 90))}


def rotation_diagnostic(V, traj_ws, template_v, frames, scale, params, warp=None):
    """Is a translation model enough? (§10) After alignment, register each
    quadrant separately against the template's quadrant. Pure translation
    leaves every quadrant at ~0; rotation moves opposite quadrants in
    opposite directions. Reports the largest quadrant disagreement per
    frame, in full-resolution pixels."""
    h, w = V.shape[1:]
    if h < 64 or w < 64 or len(frames) == 0:
        return {}
    warp = warp or (lambda img, i: shift(img, traj_ws[i]))
    quads = [(0, h // 2, 0, w // 2), (0, h // 2, w // 2, w),
             (h // 2, h, 0, w // 2), (h // 2, h, w // 2, w)]
    correlators = [PhaseCorrelator((y1 - y0, x1 - x0)) for y0, y1, x0, x1 in quads]
    take = np.linspace(0, len(frames) - 1,
                       min(params.diagnostic_frames, len(frames))).astype(int)
    spreads = []
    for k in take:
        i = frames[k]
        aligned = warp(V[i], i)
        shifts = []
        for (y0, y1, x0, x1), corr in zip(quads, correlators):
            d, _ = corr(template_v[y0:y1, x0:x1], aligned[y0:y1, x0:x1])
            shifts.append(d)
        shifts = np.array(shifts)
        spreads.append(np.max(np.hypot(*(shifts - shifts.mean(axis=0)).T)) / scale)
    spreads = np.array(spreads)
    return {"quadrant_disagreement_median_px": float(np.median(spreads)),
            "quadrant_disagreement_p90_px": float(np.percentile(spreads, 90))}
