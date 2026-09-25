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
  joint (or both leave it) are a confluence (or a fork), never one vessel;
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
    max_diam: float = 8.0
    min_length: float = 12.0         # px, shorter segments are not measured
    time_scales: tuple = (1, 3, 9)   # limbusflow time binning (slow flow)
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


def measure_velocity(net: VesselNetwork, video: Video, fc: FlowConfig | None = None, eids=None,
                     progress=False):
    """Signed velocity (px/frame, positive u -> v) of every edge.  Returns
    {eid: dict(v, v_time, reliable, quality, n_agree, scale, length)}."""
    fc = fc or FlowConfig()
    ve = _limbusflow()
    eids = [e for e in (eids if eids is not None else net.edges)
            if net.length(e) >= fc.min_length]
    pts = []
    for e in eids:
        smp = net.sample(e, 1.0)
        pts.append(lumen_points(smp["xy"], smp["tan"], _diam(smp, fc), fc.lumen_frac, fc.n_w))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        Ks = ve.kymographs(video.frames, video.reg, video.idx, pts, progress=progress) if pts else []
        out = {}
        for e, P, K in zip(eids, pts, Ks):
            est = ve.estimate(K, ve.static_regressors(video.ref, P), video.fps, scales=fc.time_scales)
            out[e] = dict(v=float(est.get("v", np.nan)), v_time=float(est.get("v_time", np.nan)),
                          reliable=bool(est.get("reliable", False)),
                          quality=float(est.get("quality", 0.0)), n_agree=int(est.get("n_agree", 0)),
                          scale=int(est.get("scale", 1)), length=float(net.length(e)))
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
    are compared with pairs on either side of it.  Points within `skip` px
    of the joint are left out: a branch leaving there overlaps the lumen.  Returns (ratio, c_cross,
    c_within) or NaNs if the path is too short or v unknown."""
    ve = _limbusflow()
    if not np.isfinite(v) or abs(v) < 1e-3:
        return np.nan, np.nan, np.nan
    Kr = ve.remove_static(K, grads)
    Kb = ve.bin_time(Kr, scale)
    K2, _ = ve.preprocess(Kb)
    S = K2.shape[1]
    skip = int(math.ceil(skip))
    dmin = 2 * skip + 4
    dmax = dmax or min(i_join, S - i_join) - 2
    if dmax < dmin:
        return np.nan, np.nan, np.nan
    vb = v * scale                                      # px per bin
    Ds = np.unique(np.linspace(dmin, dmax, 5).astype(int)) if dmax > dmin else np.array([dmin])
    cross, within = [], []
    for D in Ds:
        tau = D / vb
        # straddling pairs: s1 <= i_join - skip, s1 + D >= i_join + skip
        s1s = np.arange(max(0, i_join + skip - D), min(i_join - skip + 1, S - D))
        # within pairs: both in a, or both in b
        wa = np.arange(0, max(0, i_join - D))
        wb = np.arange(i_join, max(i_join, S - D))
        pick = lambda arr: arr[np.linspace(0, len(arr) - 1, min(n, len(arr))).astype(int)] if len(arr) else arr
        cross += [_pair_corr(K2, s, s + D, tau) for s in pick(s1s)]
        within += [_pair_corr(K2, s, s + D, tau) for s in np.r_[pick(wa), pick(wb)]]
    cross = np.array([c for c in cross if np.isfinite(c)])
    within = np.array([c for c in within if np.isfinite(c)])
    if len(cross) < 3 or len(within) < 3:
        return np.nan, np.nan, np.nan
    cc, cw = float(np.median(cross)), float(np.median(within))
    return (cc / cw if cw > 0.02 else np.nan), cc, cw


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
        if not through or ratio > fc.reject_ratio:
            return "contradicts", rec
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

    verdicts, pool = {}, []
    for key, c in loose.items():
        verdict, rec = judge(c, flow, transit.get(key, {}), fc)
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
        vals, wts = [], []
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
        if vals:
            vals, wts = np.array(vals), np.array(wts)
            sgn = np.sign(np.sum(np.sign(vals) * wts))
            ok = np.sign(vals) == sgn
            v = float(np.average(vals[ok], weights=wts[ok]))
            e.info["flow"] = dict(v_px_per_frame=round(v, 4), speed_px_per_s=round(abs(v) * fps, 2),
                                  reliable=True, n_segments=int(ok.sum()),
                                  consistent=bool(ok.all()), direction="measured")
            if v < 0:
                trial.reverse_edge(eid)
                e = trial.edges[eid]
                e.info["flow"]["v_px_per_frame"] = round(-v, 4)
            e.info["orientation"] = "flow"
        else:
            e.info["flow"] = dict(reliable=False, direction="unknown")
