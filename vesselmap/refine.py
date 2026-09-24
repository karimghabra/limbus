"""Refinement of an existing map: parallel-vessel splits and fine-scale rounds.

``refine_map(intensity, net)`` keeps everything the map already has and
adds detail:

1. **Split test for parallel vessels.**  A vessel that forks and whose two
   branches then run side by side is often fitted as one wide edge, or as
   one edge with the close neighbour missed (proposals parallel to an
   existing edge inside its footprint are suppressed).  For every edge the
   cross-section of what that edge explains (its own contribution plus the
   residual) is averaged in windows along it.  Where the averaged profile
   has two separate dark peaks, that stretch is replaced by two parallel
   edges that fork from the parent where they converge.  All candidate
   splits are fitted jointly, and each is kept only if it lowers the NLL in
   its own neighbourhood by more than the MDL cost of the extra edge;
   otherwise it is undone.
2. **Fine-scale rounds** with elongated oriented filters (ridges.py), which
   integrate along the vessel and lift faint capillaries above the noise,
   repeated until a round adds nothing.
3. A second split test (new fine vessels create new forks) and a final
   joint fit.

Single image, intensities only, like the rest of vesselmap.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter1d

from .fit import (MapConfig, _say, band_scales, clean_topology, connect_ends,
                  n_params, optimize, reparameterize, residual_image,
                  score_and_prune, seeds_to_edges, shadow_filter)
from .image import Prepared, prepare
from .network import S_MIN, R_MIN, VesselNetwork
from .render import NetworkModel, profile
from .ridges import detect


@dataclass
class RefineConfig:
    fine_bands: tuple = ((1.6, 2.6), (1.1, 1.6), (0.8, 1.1))
    scales_per_band: int = 3
    oriented_elong: float = 3.0
    z_hi: float = 4.0
    z_lo: float = 2.0
    max_reps: int = 4               # per fine band, stops earlier when nothing is added
    min_new_px: float = 60.0        # a round adding less centreline than this ends the band
    iters_round: int = 100
    iters_split: int = 100
    iters_final: int = 200
    split_passes: int = 2
    # split test
    win: float = 16.0               # window length along the edge (px)
    step: float = 6.0
    min_sep: float = 3.0            # smallest resolvable separation of two peaks (px)
    peak_k: float = 4.0             # peak height in units of the window profile noise
    valley_frac: float = 0.25       # valley at least this much below the lower peak
    min_run: float = 18.0           # shortest parallel stretch worth splitting (px)
    verbose: bool = True


# ------------------------------------------------------------------ helpers
def _edge_own_profile(e_smp, u, hw, hs):
    """Model contribution of an edge at normal offsets u (analytic)."""
    r = torch.as_tensor(e_smp["r"], dtype=torch.float32)[:, None]
    s = torch.as_tensor(e_smp["s"], dtype=torch.float32)[:, None]
    a = torch.as_tensor(e_smp["a"], dtype=torch.float32)[:, None]
    d = torch.as_tensor(np.abs(u), dtype=torch.float32)[None, :].expand(len(r), -1)
    rr, ss = r.expand_as(d), s.expand_as(d)
    sh = torch.sqrt(ss * ss + hs * hs)
    m = a * ((1 - hw) * profile(d, rr, ss) + hw * profile(d, rr, sh))
    return m.numpy()


def _peaks(p, u, min_sep):
    """Local maxima of p (index list), strongest first, separated by min_sep."""
    idx = [i for i in range(1, len(p) - 1) if p[i] >= p[i - 1] and p[i] > p[i + 1]]
    idx.sort(key=lambda i: -p[i])
    out = []
    for i in idx:
        if all(abs(u[i] - u[j]) >= min_sep for j in out):
            out.append(i)
    return out


def parallel_candidates(net: VesselNetwork, P: Prepared, R: np.ndarray, rc: RefineConfig,
                        optics=None):
    """Stretches of edges whose explained cross-section has two peaks.

    Returns a list of dict(eid, s0, s1, u1, u2) with arclength range [s0, s1]
    and per-sample offsets (arrays over the edge's 1-px samples within the
    range) of the two vessels along the edge normal."""
    optics = optics or net.meta.get("optics", {})
    hw = float(optics.get("halo_weight", 0.0))
    hs = float(optics.get("halo_sigma", 1.0))
    out = []
    for eid in list(net.edges):
        smp = net.sample(eid, 1.0)
        L = smp["s_arc"][-1]
        if L < rc.min_run:
            continue
        w = float(np.median(smp["r"]) + np.median(smp["s"]))
        U = max(7.0, 2.5 * w + 3.0)
        u = np.arange(-U, U + 0.25, 0.5)
        nrm = np.stack([-smp["tan"][:, 1], smp["tan"][:, 0]], 1)
        pts = smp["xy"][:, None, :] + nrm[:, None, :] * u[None, :, None]
        res = ndi.map_coordinates(R, [pts[..., 1], pts[..., 0]], order=1, mode="nearest")
        wgt = ndi.map_coordinates(P.valid.astype(np.float32), [pts[..., 1], pts[..., 0]],
                                  order=0, mode="nearest")
        own = _edge_own_profile(smp, u, hw, hs)
        t = own - res                          # what this edge explains + what is left over
        sig = ndi.map_coordinates(P.sigma, [smp["xy"][:, 1], smp["xy"][:, 0]], order=1)
        n = len(smp["xy"])
        hwin = int(rc.win / 2)
        rows = []
        for c in np.arange(hwin, n - hwin, max(1, int(rc.step))):
            sl = slice(c - hwin, c + hwin + 1)
            ok = wgt[sl].min(0) > 0
            if ok.mean() < 0.9:
                continue
            p = gaussian_filter1d(t[sl].mean(0), 1.0)
            noise = float(np.median(sig[sl])) / math.sqrt(rc.win) * 1.7   # 2 bins / px, smoothing
            pk = _peaks(p, u, max(rc.min_sep, 1.2 * float(np.median(smp["s"][sl]))))
            if len(pk) < 2:
                continue
            i1, i2 = sorted(pk[:2])
            lo = min(p[i1], p[i2])
            valley = p[i1:i2 + 1].min()
            if lo < rc.peak_k * noise or valley > lo - max(2 * noise, rc.valley_frac * lo):
                continue
            if abs(u[i1]) > U - 1 or abs(u[i2]) > U - 1:
                continue
            rows.append((c, u[i1], u[i2]))
        if not rows:
            continue
        # group consecutive windows with consistent offsets into runs
        runs, cur = [], [rows[0]]
        for r_ in rows[1:]:
            if r_[0] - cur[-1][0] <= 2 * rc.step and abs(r_[1] - cur[-1][1]) < 3 and \
                    abs(r_[2] - cur[-1][2]) < 3:
                cur.append(r_)
            else:
                runs.append(cur)
                cur = [r_]
        runs.append(cur)
        for run in runs:
            c0 = max(0, run[0][0] - hwin)
            c1 = min(n - 1, run[-1][0] + hwin)
            if smp["s_arc"][c1] - smp["s_arc"][c0] < rc.min_run:
                continue
            cs = np.array([r_[0] for r_ in run], float)
            idx = np.arange(c0, c1 + 1)
            u1 = np.interp(idx, cs, [r_[1] for r_ in run])
            u2 = np.interp(idx, cs, [r_[2] for r_ in run])
            out.append(dict(eid=eid, i0=int(c0), i1=int(c1), u1=u1, u2=u2,
                            s0=float(smp["s_arc"][c0]), s1=float(smp["s_arc"][c1])))
    return out


def apply_split(net: VesselNetwork, cand, ramp=6.0):
    """Replace the stretch of cand['eid'] by two parallel edges.  Returns
    the ids of the two new edges, or None if the edge no longer exists."""
    eid = cand["eid"]
    if eid not in net.edges:
        return None
    smp = net.sample(eid, 1.0)
    n = len(smp["xy"])
    i0, i1 = cand["i0"], cand["i1"]
    u1, u2 = cand["u1"], cand["u2"]
    e = net.edges[eid]
    L = smp["s_arc"][-1]
    near_start = smp["s_arc"][i0] < 8.0
    near_end = L - smp["s_arc"][i1] < 8.0
    # split points (forks) where the stretch starts / ends inside the edge
    xy0, xy1 = smp["xy"][i0], smp["xy"][i1]
    A = e.u if near_start else None
    B = e.v if near_end else None
    mid = eid
    if A is None:
        A = net.split_edge(mid, xy0)
        mid = _edge_between(net, A, xy1)
    if B is None and mid is not None:
        B = net.split_edge(mid, xy1)
        mid = _edge_between(net, A, None, B)
    if mid is None or mid not in net.edges:
        return None
    ms = net.sample(mid, 1.0)
    # re-sample the offsets onto the middle edge's samples by arclength
    s_cand = smp["s_arc"][i0:i1 + 1] - smp["s_arc"][i0]
    s_mid = ms["s_arc"]
    if net.edges[mid].u != A:            # oriented the other way
        s_mid = s_mid[-1] - s_mid
    a1 = np.interp(s_mid, s_cand, u1)
    a2 = np.interp(s_mid, s_cand, u2)
    Lm = ms["s_arc"][-1]
    nrm = np.stack([-ms["tan"][:, 1], ms["tan"][:, 0]], 1)
    if net.edges[mid].u != A:
        nrm = -nrm
    # the two branches converge on the fork nodes (ramps), unless the
    # stretch reaches a free end of the original edge
    deg = net.degrees()
    conv_start = (not near_start) or deg[A] > 1      # branches meet at the start node
    conv_end = (not near_end) or deg[B] > 1
    sv = ms["s_arc"] if net.edges[mid].u == A else ms["s_arc"][-1] - ms["s_arc"]
    wr = np.ones_like(sv)
    if conv_start:
        wr = np.minimum(wr, np.clip(sv / ramp, 0, 1))
    if conv_end:
        wr = np.minimum(wr, np.clip((Lm - sv) / ramp, 0, 1))
    P1 = ms["xy"] + nrm * (a1 * wr)[:, None]
    P2 = ms["xy"] + nrm * (a2 * wr)[:, None]
    if net.edges[mid].u != A:             # arrays follow mid's own orientation;
        conv_start, conv_end = conv_end, conv_start   # flags must follow it too
    sep = np.abs(a2 - a1)
    r_new = np.maximum(R_MIN, np.minimum(ms["r"], 0.45 * sep))
    s_new = np.maximum(S_MIN, np.minimum(ms["s"], 0.6 * sep))
    a_new = ms["a"]
    info = dict(net.edges[mid].info)
    info.pop("gain", None)
    info["split_from"] = int(eid)
    u_mid, v_mid = net.edges[mid].u, net.edges[mid].v
    net.remove_edge(mid, drop_orphans=False)
    # a free end: the first branch takes the node (moved onto it), the second
    # gets a node of its own
    if not conv_start:
        net.nodes[u_mid].x, net.nodes[u_mid].y = (float(v) for v in P1[0])
    if not conv_end:
        net.nodes[v_mid].x, net.nodes[v_mid].y = (float(v) for v in P1[-1])
    e1 = net.add_edge_dense(P1, r_new, s_new, a_new, u=u_mid, v=v_mid, info=dict(info))
    e2 = net.add_edge_dense(P2, r_new, s_new, a_new, u=u_mid if conv_start else None,
                            v=v_mid if conv_end else None, info=dict(info))
    return (e1, e2)


def _edge_between(net, a, xy=None, b=None):
    """An edge incident to node a that continues towards xy (or ends at b)."""
    best, bd = None, float("inf")
    for eid, end in net.incident(a):
        e = net.edges[eid]
        other = e.v if end == 0 else e.u
        if b is not None:
            if other == b:
                return eid
            continue
        d = np.linalg.norm(net.sample(eid, 1.0)["xy"] - xy, axis=1).min()
        if d < bd:
            best, bd = eid, d
    return best


def _region_mask(net, eids, shape, pad=3.0):
    import cv2
    m = np.zeros(shape, np.uint8)
    for eid in eids:
        if eid not in net.edges:
            continue
        smp = net.sample(eid, 1.0)
        R = float((smp["r"] + 2.5 * smp["s"]).max() + pad)
        cv2.polylines(m, [np.round(smp["xy"] * 4).astype(np.int32)], False, 1,
                      thickness=int(2 * math.ceil(R) + 1), shift=2)
    return m.astype(bool)


def split_test(net: VesselNetwork, P: Prepared, cfg: MapConfig, rc: RefineConfig, log):
    """One pass of the parallel-vessel split test (see module docstring)."""
    m0 = NetworkModel(net, P.logI, P.weight, stride=1, bg_spacing=cfg.bg_spacing, ref=net)
    optimize(m0, rc.iters_split, cfg.lr_pos, cfg.lr_prof, cfg.lr_bg, cfg.rebuild_every,
             priors=cfg.priors(), track=cfg.anchor())
    m0.write_back()
    R0 = residual_image(m0)
    cands = parallel_candidates(net, P, R0, rc)
    log(f"split test: {len(cands)} parallel stretches on "
        f"{len({c['eid'] for c in cands})} edges")
    if not cands:
        return 0
    trial = net.copy()
    applied = []
    # one candidate per original edge per pass (the longest stretch)
    best = {}
    for c in cands:
        if c["eid"] not in best or (c["s1"] - c["s0"]) > (best[c["eid"]]["s1"] - best[c["eid"]]["s0"]):
            best[c["eid"]] = c
    for c in best.values():
        pair = apply_split(trial, c)
        if pair:
            applied.append(pair)
    if not applied:
        return 0
    m1 = NetworkModel(trial, P.logI, P.weight, stride=1, bg_spacing=cfg.bg_spacing, ref=trial)
    optimize(m1, rc.iters_split, cfg.lr_pos, cfg.lr_prof, cfg.lr_bg, cfg.rebuild_every,
             priors=cfg.priors(), track=cfg.anchor())
    m1.write_back()
    R1 = residual_image(m1)
    gains, _ = m1.edge_gains()
    gidx = {eid: gains[k] for k, eid in enumerate(m1.eids)}
    W = P.weight
    kept = 0
    for e1, e2 in applied:
        if e1 not in trial.edges or e2 not in trial.edges:
            continue
        mask = _region_mask(trial, [e1, e2], P.shape)
        dnll = 0.5 * float((W[mask] * (R0[mask] ** 2 - R1[mask] ** 2)).sum())
        L2 = trial.length(e2)
        pen = cfg.penalty_scale * 0.5 * n_params(trial, e2, L2) * math.log(max(mask.sum(), 2))
        if dnll > pen:
            kept += 1
            for k_ in (e1, e2):
                trial.edges[k_].info["split_dnll"] = dnll
        else:
            weaker = e1 if gidx.get(e1, 0) < gidx.get(e2, 0) else e2
            trial.remove_edge(weaker)
    trial.merge_joints()
    # adopt the trial network
    net.nodes, net.edges = trial.nodes, trial.edges
    net._nid, net._eid, net._adj = trial._nid, trial._eid, None
    net.background, net.meta = trial.background, trial.meta
    log(f"split test: kept {kept} of {len(applied)} splits")
    return kept


# ------------------------------------------------------------------ driver
def refine_map(intensity: np.ndarray, net: VesselNetwork, cfg: MapConfig | None = None,
               rc: RefineConfig | None = None, prepared: Prepared | None = None) -> VesselNetwork:
    """Add fine detail to an existing map of the same image (in place and
    returned)."""
    cfg = cfg or MapConfig()
    rc = rc or RefineConfig()
    P = prepared or prepare(intensity)
    t0 = time.time()
    if net.has_through():
        # a consolidated map: refine works on segments (consolidate again after)
        seg = net.to_segments()
        net.nodes, net.edges = seg.nodes, seg.edges
        net._nid, net._eid, net._adj = seg._nid, seg._eid, None
    log = lambda *a: _say(cfg, f"[{time.time() - t0:6.0f}s]", *a) if rc.verbose else None
    L_start = net.summary()["total_length_px"]
    split_test(net, P, cfg, rc, log)
    for band in rc.fine_bands:
        for rep in range(rc.max_reps):
            L0 = net.summary()["total_length_px"]
            model = NetworkModel(net, P.logI, P.weight, stride=1, bg_spacing=cfg.bg_spacing)
            res = residual_image(model)
            ml = max(cfg.min_length, 2.5 * band[0])
            seeds = detect(-res, P.sigma, band_scales(band, rc.scales_per_band),
                           z_hi=rc.z_hi, z_lo=rc.z_lo, min_len=ml, valid=P.valid,
                           oriented=True, elong=rc.oriented_elong)
            n_raw = len(seeds)
            seeds = shadow_filter(seeds, net, ml)
            if not seeds:
                log(f"band {band} rep {rep}: no proposals")
                break
            seeds_to_edges(net, seeds, band)
            clean_topology(net, cfg.min_length)
            connect_ends(net)
            net.resolve_near_crossings()
            model = NetworkModel(net, P.logI, P.weight, stride=1, bg_spacing=cfg.bg_spacing, ref=net)
            optimize(model, rc.iters_round, cfg.lr_pos, cfg.lr_prof, cfg.lr_bg,
                     cfg.rebuild_every, priors=cfg.priors(), track=cfg.anchor())
            model.write_back()
            reparameterize(net)
            score_and_prune(net, model, cfg)
            added = net.summary()["total_length_px"] - L0
            ps = {k: (v[0], round(v[1])) for k, v in cfg.prune_stats.items()}
            cfg.prune_stats.clear()
            log(f"band {band} rep {rep}: {n_raw} ridges -> {len(seeds)} proposals; "
                f"net +{added:.0f} px (pruned {ps}); {net.summary()}")
            if added < rc.min_new_px:
                break
    for _ in range(rc.split_passes - 1):
        split_test(net, P, cfg, rc, log)
    clean_topology(net, cfg.min_length)
    connect_ends(net)
    net.resolve_near_crossings()
    model = NetworkModel(net, P.logI, P.weight, stride=1, bg_spacing=cfg.bg_spacing, ref=net)
    optimize(model, rc.iters_final, cfg.lr_pos * 0.5, cfg.lr_prof, cfg.lr_bg,
             cfg.rebuild_every, priors=cfg.priors(), track=cfg.anchor())
    model.write_back()
    reparameterize(net)
    score_and_prune(net, model, cfg)
    cfg.prune_stats.clear()
    model = NetworkModel(net, P.logI, P.weight, stride=1, bg_spacing=cfg.bg_spacing)
    optimize(model, 60, cfg.lr_pos * 0.3, cfg.lr_prof * 0.5, cfg.lr_bg * 0.5,
             cfg.rebuild_every, priors=cfg.priors())
    model.write_back()
    gains, _ = model.edge_gains()
    for k, eid in enumerate(model.eids):
        net.edges[eid].info["gain"] = float(gains[k])
    net.orient_structural()
    net.meta.setdefault("refinements", []).append(dict(
        seconds=round(time.time() - t0, 1), length_before=round(L_start, 1),
        length_after=round(net.summary()["total_length_px"], 1),
        final_nll=float(model.data_nll(model.predict()).detach())))
    net.meta["final_nll"] = net.meta["refinements"][-1]["final_nll"]
    log(f"refine done: {net.summary()}")
    return net
