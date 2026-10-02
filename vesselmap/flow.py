"""Flow-informed consolidation: the one part of vesselmap that uses velocity.

The map itself (``build_map``, ``refine_map``, ``add_faint_tier``,
``consolidate_map``) is built from a single still image.  This module comes
after it: it measures red-cell velocity along every mapped segment in a
registered video and uses flow continuity as further evidence of which
segments are one vessel.  Nothing here feeds back into how vessels are
detected, so the structural map stays independent of the velocities it is
later used to measure.

Velocity (limbusflow)
---------------------
For every segment, a kymograph K(t, s) is read along the centreline (averaged
across the central half of the lumen) from every registered frame, and
``limbusflow.velocity.estimate`` gives the signed velocity along the segment
(positive = from its node u towards its node v), with a reliability flag
(clear correlation ridge, and two of three independent estimators agree;
slow flow must also persist across both halves of the recording).

When are two segments one vessel?
---------------------------------
Candidate continuations come from ``consolidate.candidates`` (two ends
meeting at a node, or facing each other across a gap), judged on shape:
direction, calibre and blur.  Flow adds three tests:

* *direction*: blood must run through the joint, into it along one segment
  and out of it along the other.  Two segments that both drain into the
  joint (or both leave it) are a confluence (or a fork), never one vessel
  (unless the transit test below shows the pattern carrying across: then
  the two measurements disagree and flow does not decide);
* *speed*: along one vessel the red-cell speed is continuous, so the two
  speeds must agree (ratio <= ``speed_ratio``);
* *transit*: the pattern of red-cell aggregates and plasma gaps leaving one
  segment must reappear in the other, delayed by distance / speed.  The
  temporal correlation between points on opposite sides of the joint is
  compared with the correlation between points the same distance apart
  within each segment.  One vessel carries the same cells across the joint
  (ratio ~ 1); a branch receives only some of them.

A continuation that flow *confirms* is accepted even where shape alone was
ambiguous (a fork whose through vessel is not clearly straighter than the
branch).  One that flow *contradicts* is dropped even where shape alone
would have joined it.  Where flow is not measurable (too slow, too short,
out of focus), the shape-only rule of ``consolidate`` applies, including its
test against the image.  Segments not joined stay attached where they meet
the vessel (``info["through"]``), exactly as in ``consolidate``.

Every joined vessel records the evidence of each of its links
(``info["links"]``), and every edge its measured flow (``info["flow"]``);
edges with a reliable measurement are oriented along the flow.
"""
from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np

from .consolidate import (ConsolidateConfig, _fit, _oriented, build, candidates, chains, end_infos,
                          match, split_switches, verify)
from .fit import MapConfig, _say, residual_image
from .image import Prepared, prepare
from .network import VesselNetwork


def _limbusflow():
    """Import limbusflow's velocity module (analysis/limbusflow in this repo)."""
    try:
        from limbusflow import velocity
    except ImportError:
        here = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.join(os.path.dirname(here), "analysis", "limbusflow"))
        from limbusflow import velocity
    return velocity


@dataclass
class Video:
    """A registered burst: frames (T, H, W), limbusflow Registration `reg`
    (frame -> reference affine per frame), the frame indices to use (a run
    of consecutive good frames), the frame rate, and the reference image
    the map was built from (for the static-anatomy regressors)."""
    frames: object
    reg: object
    idx: np.ndarray
    fps: float
    ref: np.ndarray


@dataclass
class FlowConfig:
    lumen_frac: float = 0.5          # sample the central half of the lumen
    n_w: int = 5                     # samples across it
    min_diam: float = 2.0            # px, clamp of the sampled lumen width
    max_diam: float = 40.0           # (large vessels are sampled across their lumen)
    min_length: float = 12.0         # px, shorter segments are not measured
    time_scales: tuple = (1, 3, 9)   # limbusflow time binning (slow flow)
    fast_flow: bool = True           # fall back to the fast-flow estimator (see fast_flow)
    fast_lags: tuple = (1, 2, 3)
    fast_min_disp: float = 3.0       # px/frame: the fast estimator covers flow at least this fast
    fast_null_ratio: float = 4.0     # score must beat time-shuffled copies by this factor
    fast_min_score: float = 0.015
    profile_null_ratio: float = 2.5  # bands search only near the vessel's velocity
    profile_min_diam: float = 10.0   # px: vessels at least this wide get a velocity profile
    profile_bands: int = 5
    speed_ratio: float = 1.6         # speeds of one vessel agree within this ratio
    reject_ratio: float = 3.0        # speeds this different are two vessels
    transit_len: float = 40.0        # px on each side of a joint for the transit test
    transit_skip: float = 4.0        # px around the joint left out (branches overlap there)
    transit_min: float = 0.6         # cross-joint / within-segment correlation for "one vessel"
    flow_bonus: float = 1.0          # cost bonus of a flow-confirmed link in the matching
    verify_shape_links: bool = True  # test shape-only links against the image (as consolidate)
    verbose: bool = True


# ------------------------------------------------------------------ sampling
def lumen_points(xy, tan, diam, frac=0.5, n_w=5):
    """(n_s, n_w, 2) grid across the central `frac` of the lumen."""
    t = tan / np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-9)
    N = np.stack([-t[:, 1], t[:, 0]], 1)
    w = np.linspace(-1, 1, n_w)[None, :] * (frac * np.asarray(diam, float).reshape(-1, 1) / 2)
    return xy[:, None, :] + w[..., None] * N[:, None, :]


def _diam(smp, fc):
    return float(np.clip(2.0 * np.median(smp["r"]), fc.min_diam, fc.max_diam))


def fast_flow(K, grads=None, lags=(1, 2, 3), min_disp=3.0, max_shift=170, n_null=4,
              null_ratio=4.0, min_score=0.015, seed=0, v_range=None):
    """Velocity of fast flow, which limbusflow's defaults (tuned for slow
    capillary flow) miss.

    With an exposure of ~95 % of the frame period, a pattern moving v
    px/frame is smeared over v px, so in fast flow only large-scale flicker
    survives.  It is kept here: no spatial high-pass, only light smoothing.
    The displacement search is wide (up to `max_shift` px), and the flow
    line is found in the antisymmetric part of the correlation map,
    A(lag, d) = (C(lag, d) - C(lag, -d)) / 2.  Static anatomy and jitter are
    symmetric in d, directed flow is not.  Every line d = v * lag through
    the origin is scored by the mean of A along it over `lags` (only where
    |v| >= min_disp).  Significance: the best score must beat the best
    score of `n_null` time-shuffled copies (motion destroyed, spatial
    statistics kept) by `null_ratio`.  `v_range` (lo, hi) restricts the
    search, e.g. to a band of a vessel whose overall velocity is known (the
    null is searched over the same range).  Returns dict(v px/frame,
    score, null, reliable)."""
    ve = _limbusflow()
    K2, _ = ve.preprocess(ve.remove_static(K, grads), 10.0, None, 2.0)
    S = K2.shape[1]
    ms = int(min(max_shift, S // 2))
    if ms < min_disp * max(lags) + 4 or len(K2) < 30:
        return dict(v=np.nan, score=np.nan, null=np.nan, reliable=False)

    def best_line(K2):
        Cm, sh = ve.correlation_map(K2, max(lags), max_shift=ms)
        A = 0.5 * (Cm - Cm[:, ::-1])
        vs = np.arange(-(ms - 3), ms - 3 + 1e-9, 0.25)
        best = (np.nan, -np.inf)
        for v in vs:
            if abs(v) < min_disp or (v_range is not None and not v_range[0] <= v <= v_range[1]):
                continue
            use = [l for l in lags if abs(v) * l <= ms - 3]
            if len(use) < 2:
                continue
            sc = float(np.mean([np.interp(v * l, sh, A[l - 1]) for l in use]))
            if sc > best[1]:
                best = (float(v), sc)
        return best

    v, sc = best_line(K2)
    rng = np.random.default_rng(seed)
    null = max(best_line(K2[rng.permutation(len(K2))])[1] for _ in range(n_null))
    ok = bool(np.isfinite(v) and sc >= min_score and sc >= null_ratio * max(null, 1e-6))
    return dict(v=v, score=sc, null=null, reliable=ok)


def _lanes(smp, fc):
    """Lateral offsets (px) sampled across the lumen: the central half is
    the main kymograph (as limbusflow), and wide vessels get bands across
    +-0.8 of the radius for a velocity profile."""
    d = _diam(smp, fc)
    n = fc.n_w if d < fc.profile_min_diam else max(fc.n_w, 2 * int(d / 4) + 1)
    return np.linspace(-0.8, 0.8, n) * d / 2, d


def _estimate(K, grads, fps, fc):
    ve = _limbusflow()
    est = ve.estimate(K, grads, fps, scales=fc.time_scales)
    out = dict(v=float(est.get("v", np.nan)), v_time=float(est.get("v_time", np.nan)),
               reliable=bool(est.get("reliable", False)), quality=float(est.get("quality", 0.0)),
               n_agree=int(est.get("n_agree", 0)), scale=int(est.get("scale", 1)), method="limbusflow")
    if not out["reliable"] and fc.fast_flow:
        f = fast_flow(K, grads, fc.fast_lags, fc.fast_min_disp, null_ratio=fc.fast_null_ratio,
                      min_score=fc.fast_min_score)
        if f["reliable"]:
            out.update(v=f["v"], v_time=f["v"], reliable=True, quality=f["score"], scale=1,
                       method="fast", null=f["null"])
    return out


def _profile(KL, P, offs, r, video, fc):
    """Velocity in bands across the lumen, searched near the vessel's own
    velocity r["v"] (a band has less signal than the whole lumen)."""
    ve = _limbusflow()
    v0 = r["v"]
    lo, hi = sorted((0.4 * v0, 1.6 * v0))
    prof = []
    for band in np.array_split(np.arange(len(offs)), fc.profile_bands):
        Kb = np.nanmean([KL[i] for i in band], 0)
        g = ve.static_regressors(video.ref, P[:, band])
        vb = None
        if r["method"] == "fast":
            f = fast_flow(Kb, g, fc.fast_lags, fc.fast_min_disp, null_ratio=fc.profile_null_ratio,
                          min_score=0.5 * fc.fast_min_score, v_range=(lo, hi))
            vb = f["v"] if f["reliable"] else None
        else:
            est = ve.estimate(Kb, g, video.fps, scales=fc.time_scales)
            if est.get("reliable") and lo <= est["v"] <= hi:
                vb = float(est["v"])
        prof.append(dict(offset_px=round(float(np.mean(offs[band])), 1),
                         v=None if vb is None else round(float(vb), 3)))
    return prof


def measure_velocity(net: VesselNetwork, video: Video, fc: FlowConfig | None = None, eids=None,
                     progress=False):
    """Signed velocity (px/frame, positive u -> v) of every edge, and for
    wide vessels a velocity profile across the lumen.  Returns
    {eid: dict(v, v_time, reliable, quality, n_agree, scale, method,
    length[, profile])}.  limbusflow's estimator is tried first; where it
    finds nothing, the fast-flow estimator (see fast_flow)."""
    fc = fc or FlowConfig()
    ve = _limbusflow()
    eids = [e for e in (eids if eids is not None else net.edges)
            if net.length(e) >= fc.min_length]
    lanes, meta = [], []
    for e in eids:
        smp = net.sample(e, 1.0)
        offs, d = _lanes(smp, fc)
        t = smp["tan"] / np.maximum(np.linalg.norm(smp["tan"], axis=1, keepdims=True), 1e-9)
        N = np.stack([-t[:, 1], t[:, 0]], 1)
        a = len(lanes)
        lanes += [(smp["xy"] + o * N)[:, None, :] for o in offs]
        meta.append((e, a, offs, d))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        Ks = ve.kymographs(video.frames, video.reg, video.idx, lanes, progress=progress) if lanes else []
        out = {}
        for e, a, offs, d in meta:
            KL = Ks[a:a + len(offs)]
            P = np.concatenate(lanes[a:a + len(offs)], 1)
            central = np.abs(offs) <= fc.lumen_frac * d / 2 + 1e-9
            K = np.nanmean([k for k, c in zip(KL, central) if c], 0)
            grads = ve.static_regressors(video.ref, P[:, central])
            r = _estimate(K, grads, video.fps, fc)
            r["length"] = float(net.length(e))
            r["diam"] = float(d)
            if r["reliable"] and d >= fc.profile_min_diam:
                r["profile"] = _profile(KL, P, offs, r, video, fc)
            out[e] = r
    return out


# ------------------------------------------------------------------ joints
def _end_path(smp, end, length):
    """Samples of the last `length` px of an edge, ordered towards `end`."""
    o = _oriented(smp, end)                  # from `end` inwards
    n = max(2, int(np.searchsorted(smp["s_arc"], length)))
    return {k: o[k][:n][::-1] for k in ("xy", "tan", "r")}


def joint_path(net, samples, a, b, length):
    """Centreline through a joint: the last `length` px of end a (towards
    the joint), a straight bridge, and the first `length` px of end b.
    Returns (xy, tan, r, i_join) with i_join the index where b starts."""
    A = _end_path(samples[a[0]], a[1], length)
    B = _end_path(samples[b[0]], b[1], length)
    B = {k: v[::-1] for k, v in B.items()}   # away from the joint
    # tangents: point along the path (a -> joint -> b)
    ta = -A["tan"] if a[1] == 0 else A["tan"]
    tb = B["tan"] if b[1] == 0 else -B["tan"]
    gap = B["xy"][0] - A["xy"][-1]
    d = float(np.linalg.norm(gap))
    k = int(max(0, math.floor(d) - 1))
    br = A["xy"][-1] + np.outer(np.arange(1, k + 1) / (k + 1), gap) if k else np.zeros((0, 2))
    tbr = np.repeat((gap / max(d, 1e-9))[None], k, 0)
    xy = np.concatenate([A["xy"], br, B["xy"]])
    tan = np.concatenate([ta, tbr, tb])
    r = np.concatenate([A["r"], np.full(k, 0.5 * (A["r"][-1] + B["r"][0])), B["r"]])
    return xy, tan, r, len(A["xy"]) + k


def _pair_corr(K2, s1, s2, tau, tol=0.5, min_half=1.0):
    """Peak normalised temporal correlation of K2[:, s1] and K2[:, s2] for
    delays within tau*(1 +- tol) (frames, sign = direction)."""
    T = len(K2)
    half = max(min_half, tol * abs(tau))
    lags = np.arange(int(math.floor(tau - half)), int(math.ceil(tau + half)) + 1)
    lags = lags[np.abs(lags) < T // 3]
    if not len(lags):
        return np.nan
    a = K2[:, s1] - K2[:, s1].mean()
    b = K2[:, s2] - K2[:, s2].mean()
    den = a.std() * b.std() + 1e-12
    best = -np.inf
    for L in lags:
        if L >= 0:
            c = float(np.mean(a[:T - L] * b[L:]))
        else:
            c = float(np.mean(a[-L:] * b[:T + L]))
        best = max(best, c / den)
    return best


def transit_ratio(K, i_join, v, scale=1, grads=None, skip=4, dmax=None, n=24):
    """Cross-joint / within-segment correlation of the moving pattern.

    K: kymograph along a joint path (a -> b), i_join: first index of b,
    v: expected signed velocity along the path (px/frame).  Pairs of points
    D px apart are correlated at the delay D / v; pairs straddling the joint
    are compared with pairs within the side whose pattern is clearer.  Points within `skip` px
    of the joint are left out: a branch leaving there overlaps the lumen.  Returns (ratio, c_cross,
    c_within) or NaNs if the path is too short or v unknown."""
    ve = _limbusflow()
    if not np.isfinite(v) or abs(v) < 1e-3:
        return np.nan, np.nan, np.nan
    Kr = ve.remove_static(K, grads)
    Kb = ve.bin_time(Kr, scale)
    # fast flow: only large-scale flicker survives the motion smear (see fast_flow)
    K2, _ = ve.preprocess(Kb, 10.0, None, 2.0) if abs(v) * scale >= 3.0 else ve.preprocess(Kb)
    S = K2.shape[1]
    skip = int(math.ceil(skip))
    dmin = 2 * skip + 4
    dmax = dmax or min(i_join, S - i_join) - 2
    if dmax < dmin:
        return np.nan, np.nan, np.nan
    vb = v * scale                                      # px per bin
    Ds = np.unique(np.linspace(dmin, dmax, 5).astype(int)) if dmax > dmin else np.array([dmin])
    cross, wa_c, wb_c = [], [], []
    for D in Ds:
        tau = D / vb
        # straddling pairs: s1 <= i_join - skip, s1 + D >= i_join + skip
        s1s = np.arange(max(0, i_join + skip - D), min(i_join - skip + 1, S - D))
        # within pairs: both in a, or both in b
        wa = np.arange(0, max(0, i_join - D))
        wb = np.arange(i_join, max(i_join, S - D))
        pick = lambda arr: arr[np.linspace(0, len(arr) - 1, min(n, len(arr))).astype(int)] if len(arr) else arr
        cross += [_pair_corr(K2, s, s + D, tau) for s in pick(s1s)]
        wa_c += [_pair_corr(K2, s, s + D, tau) for s in pick(wa)]
        wb_c += [_pair_corr(K2, s, s + D, tau) for s in pick(wb)]
    fin = lambda x: np.array([c for c in x if np.isfinite(c)])
    cross, wa_c, wb_c = fin(cross), fin(wa_c), fin(wb_c)
    sides = [float(np.median(w)) for w in (wa_c, wb_c) if len(w) >= 3]
    if len(cross) < 3 or not sides:
        return np.nan, np.nan, np.nan
    # compare with the side that carries the clearer pattern: one vessel
    # carries it across the joint; noise on the other side must not
    # inflate the ratio
    cc, cw = float(np.median(cross)), max(sides)
    return (cc / cw if cw > 0.05 else np.nan), cc, cw


def _into(end, v):
    """Does flow v (signed along u -> v) run into the joint at `end`?"""
    return (end == 1 and v > 0) or (end == 0 and v < 0)


def judge(link, flow, transit, fc: FlowConfig):
    """'confirms' / 'contradicts' / 'unknown' and a record of why."""
    (ea, enda), (eb, endb) = link["a"], link["b"]
    fa, fb = flow.get(ea), flow.get(eb)
    ra = fa is not None and fa["reliable"]
    rb = fb is not None and fb["reliable"]
    rec = dict(va=fa["v"] if ra else None, vb=fb["v"] if rb else None, transit=transit.get("ratio"))
    tr = transit.get("ratio")
    if ra and rb:
        through = _into(enda, fa["v"]) != _into(endb, fb["v"])
        ratio = max(abs(fa["v"]), abs(fb["v"])) / max(min(abs(fa["v"]), abs(fb["v"])), 1e-9)
        rec.update(speed_ratio=ratio, through=through)
        carries = tr is not None and np.isfinite(tr) and tr >= fc.transit_min
        if not through or ratio > fc.reject_ratio:
            # the pattern carrying straight across the joint contradicts the
            # conflict: the two measurements disagree, so flow cannot decide
            return ("unknown" if carries else "contradicts"), rec
        if ratio <= fc.speed_ratio and (tr is None or not np.isfinite(tr) or tr >= fc.transit_min):
            return "confirms", rec
        if tr is not None and np.isfinite(tr) and tr < 0.5 * fc.transit_min:
            return "contradicts", rec
        return "unknown", rec
    if (ra or rb) and tr is not None and np.isfinite(tr) and tr >= fc.transit_min:
        return "confirms", rec
    return "unknown", rec


# ------------------------------------------------------------------ driver
def flow_consolidate(intensity: np.ndarray, net: VesselNetwork, video: Video,
                     cfg: MapConfig | None = None, cc: ConsolidateConfig | None = None,
                     fc: FlowConfig | None = None, prepared: Prepared | None = None):
    """Join segments that are one vessel, using shape and flow.  `net` is a
    map of the reference image (not modified).  Returns (network, report)."""
    cfg = cfg or MapConfig()
    cc = cc or ConsolidateConfig()
    fc = fc or FlowConfig()
    ve = _limbusflow()
    P = prepared or prepare(intensity)
    t0 = time.time()
    log = (lambda *a: _say(cfg, f"[{time.time() - t0:6.0f}s]", *a)) if fc.verbose else (lambda *a: None)
    base = net.to_segments() if net.has_through() else net.copy()
    n0 = len(base.edges)
    n_cut = split_switches(base, cc)
    flow = measure_velocity(base, video, fc)
    n_rel = sum(f["reliable"] for f in flow.values())
    log(f"velocity: {len(flow)} of {n0 + n_cut} segments measured, {n_rel} reliable")

    samples = {eid: base.sample(eid, 0.7) for eid in base.edges}
    ends = end_infos(base, cc, samples)
    strict = {(c["a"], c["b"]): c for c in candidates(base, cc, samples, ends)}
    loose_cc = ConsolidateConfig(**{**cc.__dict__, "margin": -1e9})
    loose = {(c["a"], c["b"]): c for c in candidates(base, loose_cc, samples, ends)}
    loose.update(strict)

    # transit test: one kymograph pass over all joints
    smp1 = {eid: base.sample(eid, 1.0) for eid in base.edges}
    keys, paths = [], []
    for key, c in loose.items():
        if not (c["a"][0] in flow or c["b"][0] in flow):
            continue
        xy, tan, r, ij = joint_path(base, smp1, c["a"], c["b"], fc.transit_len)
        if ij < 12 or len(xy) - ij < 12:
            continue
        d = float(np.clip(2 * np.median(r), fc.min_diam, fc.max_diam))
        keys.append((key, ij, d))
        paths.append(lumen_points(xy, tan, d, fc.lumen_frac, fc.n_w))
    import warnings
    transit = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        Ks = ve.kymographs(video.frames, video.reg, video.idx, paths, progress=False) if paths else []
        for (key, ij, d), Pp, K in zip(keys, paths, Ks):
            c = loose[key]
            # expected velocity along the path a -> b, from the reliable side(s)
            vs, sc = [], 1
            for (eid, end), sign_into in ((c["a"], 1), (c["b"], -1)):
                f = flow.get(eid)
                if f and f["reliable"]:
                    into = _into(end, f["v"])
                    # along the path a -> b: into the joint on a is +, out of it on b is +
                    vs.append(abs(f["v"]) * (1 if into == (sign_into == 1) else -1))
                    sc = max(sc, f["scale"])
            if not vs:
                continue
            v_exp = float(np.median(vs))
            ratio, c_x, c_w = transit_ratio(K, ij, v_exp, sc, ve.static_regressors(video.ref, Pp),
                                            skip=max(fc.transit_skip, 0.75 * d))
            transit[key] = dict(ratio=ratio, cross=c_x, within=c_w)

    def joint_xy(c):
        return (0.5 * (ends[c["a"]]["xy"] + ends[c["b"]]["xy"])).round(1).tolist()

    verdicts, pool = {}, []
    for key, c in loose.items():
        verdict, rec = judge(c, flow, transit.get(key, {}), fc)
        rec["xy"] = joint_xy(c)
        verdicts[key] = (verdict, rec)
        if verdict == "contradicts":
            continue
        if verdict == "confirms":
            pool.append(dict(c, cost=c["cost"] - fc.flow_bonus, evidence="shape+flow"))
        elif key in strict:
            pool.append(dict(c, evidence="shape"))
    count = lambda v: sum(1 for x, _ in verdicts.values() if x == v)
    log(f"candidates: {len(strict)} by shape alone, {len(loose)} with forks; flow confirms "
        f"{count('confirms')}, contradicts {count('contradicts')}, unknown {count('unknown')}")

    links = match(pool)
    trial, records = build(base, chains(base, links), cc, samples, ends)
    failed = []
    if fc.verify_shape_links and any(L.get("evidence") == "shape" for L in links):
        shape_recs = [r for r in records
                      if next(L for L in links if (L["a"], L["b"]) == (r["a"], r["b"]))["evidence"] == "shape"]
        m0 = _fit(base, P, cfg, cc, cc.iters)
        R0 = residual_image(m0)
        model = _fit(trial, P, cfg, cc, cc.iters)
        _, failed = verify(trial, shape_recs, model, R0, P, cfg, cc)
        if failed:
            bad = {(r["a"], r["b"]) for r in failed}
            links = [L for L in links if (L["a"], L["b"]) not in bad]
            trial, records = build(base, chains(base, links), cc, samples, ends)
        log(f"image test of shape-only links: {len(shape_recs) - len(failed)} passed, {len(failed)} failed")

    # evidence and flow on the result
    by_key = {(L["a"], L["b"]): L for L in links}
    for r in records:
        L = by_key[(r["a"], r["b"])]
        v, rec = verdicts[(L["a"], L["b"])]
        e = trial.edges[r["vessel"]]
        e.info.setdefault("links", []).append(dict(
            kind=r["kind"], evidence=L["evidence"],
            **{k: (None if x is None or (isinstance(x, float) and not np.isfinite(x)) else
                   (round(float(x), 3) if isinstance(x, (float, np.floating)) else x))
               for k, x in rec.items()}))
    _assign_flow(trial, base, flow, video.fps)
    n_multi = sum(1 for e in trial.edges.values() if len(e.info.get("consolidated_from", [])) > 1)
    ev = lambda s: sum(1 for L in links if L["evidence"] == s)
    report = dict(seconds=round(time.time() - t0, 1), segments=n0, switch_cuts=n_cut,
                  measured=len(flow), reliable=n_rel, candidates_shape=len(strict),
                  candidates_forks=len(loose), confirms=count("confirms"),
                  contradicts=count("contradicts"), unknown=count("unknown"),
                  links_flow=ev("shape+flow"), links_shape=ev("shape"),
                  shape_links_failed_image=len(failed), edges_after=len(trial.edges),
                  vessels_from_several=n_multi, fps=video.fps, frames=len(video.idx))
    clean = lambda d: {k: (None if x is None or (isinstance(x, float) and not np.isfinite(x)) else
                           (round(float(x), 3) if isinstance(x, (float, np.floating)) else x))
                       for k, x in d.items()}
    report["rejected_by_flow"] = [dict(clean(rec), kind=loose[k]["kind"], shape_ok=k in strict)
                                  for k, (v, rec) in verdicts.items() if v == "contradicts"]
    trial.meta["flow_consolidation"] = report
    log(f"flow consolidation: {n0} -> {len(trial.edges)} edges; links by shape+flow "
        f"{ev('shape+flow')}, by shape {ev('shape')}")
    return trial, report


def _assign_flow(trial: VesselNetwork, base: VesselNetwork, flow, fps):
    """info["flow"] on every edge from its segments' measurements (signed
    along the edge), and orient edges with reliable flow along it."""
    for eid, e in trial.edges.items():
        src = e.info.get("consolidated_from", [eid])
        smp = trial.sample(eid, 2.0)
        vals, wts, methods, prof = [], [], set(), None
        for s in src:
            f = flow.get(s)
            if not f or not f["reliable"] or s not in base.edges:
                continue
            # direction of segment s relative to this edge: compare its u->v chord
            bs = base.sample(s, 2.0)["xy"]
            i0 = int(np.argmin(np.linalg.norm(smp["xy"] - bs[0], axis=1)))
            i1 = int(np.argmin(np.linalg.norm(smp["xy"] - bs[-1], axis=1)))
            same = i1 >= i0
            vals.append(f["v"] if same else -f["v"])
            wts.append(f["length"])
            methods.add(f.get("method", "limbusflow"))
            if f.get("profile") and len(src) == 1:
                prof = [dict(p, v=None if p["v"] is None else (p["v"] if same else -p["v"]),
                             offset_px=p["offset_px"] if same else -p["offset_px"])
                        for p in f["profile"]]
        if vals:
            vals, wts = np.array(vals), np.array(wts)
            sgn = np.sign(np.sum(np.sign(vals) * wts))
            ok = np.sign(vals) == sgn
            v = float(np.average(vals[ok], weights=wts[ok]))
            e.info["flow"] = dict(v_px_per_frame=round(v, 4), speed_px_per_s=round(abs(v) * fps, 2),
                                  reliable=True, n_segments=int(ok.sum()),
                                  consistent=bool(ok.all()), direction="measured",
                                  method="+".join(sorted(methods)))
            if prof:
                e.info["flow"]["profile"] = prof
            if v < 0:
                trial.reverse_edge(eid)
                e = trial.edges[eid]
                e.info["flow"]["v_px_per_frame"] = round(-v, 4)
                for p in e.info["flow"].get("profile", []):
                    p["v"] = None if p["v"] is None else -p["v"]
                    p["offset_px"] = -p["offset_px"]
            e.info["orientation"] = "flow"
        else:
            e.info["flow"] = dict(reliable=False, direction="unknown")
