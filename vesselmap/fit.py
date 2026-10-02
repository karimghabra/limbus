"""Fitting a whole vessel network to ONE image, and re-fitting it per frame.

``build_map``   discovers the network from scratch (coarse to fine):

    1. robust smooth background, empty network
    2. for each scale band, coarse -> fine, a few times:
         a. residual = image - current model
         b. propose centrelines = multi-scale ridges of the residual in the
            band (so vessels already explained are not proposed twice, and a
            thin vessel crossing a thick one is still found)
         c. turn proposals into spline edges, join them to the network
            (gap bridging, bifurcation snapping, crossing resolution)
         d. jointly optimise every spline and the background
         e. score every edge by how much of the image it explains
            (drop of the negative log-likelihood) and remove edges whose
            gain does not pay for their parameters (MDL / BIC)
    3. final topology clean-up and joint optimisation at full resolution.

``fit_frame``   takes an existing map and adjusts it to another frame:
    global affine pre-alignment, then a joint optimisation with a prior that
    keeps every parameter close to the reference map.  The topology is not
    changed; every edge reports how visible it is in that frame.

Only single-frame intensities are used.  Nothing in this module looks at
temporal changes, kymographs or flow.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
from scipy.spatial import cKDTree

from .image import Prepared, prepare
from .network import PROFILE_SPACING, S_MIN, R_MIN, VesselNetwork, pos_spacing_for
from .render import NetworkModel, profile_peak
from .ridges import detect

torch.set_num_threads(max(1, torch.get_num_threads()))


@dataclass
class MapConfig:
    bands: tuple = ((10.0, 20.0), (5.0, 10.0), (2.5, 5.0), (1.2, 2.5), (0.8, 1.2))
    scales_per_band: int = 3
    z_hi: float = 5.0
    z_lo: float = 2.5
    reps_per_band: int = 3
    iters_band: int = 120
    iters_final: int = 300
    rebuild_every: int = 25
    penalty_scale: float = 2.0      # multiplies the BIC parameter cost
    min_gain_per_px: float = 1.0    # minimum explained NLL per px of length
    min_length: float = 6.0
    bg_spacing: float = 64.0
    lam_bend: float = 600.0
    lam_prof: float = 20.0
    lam_bg: float = 2e4
    lr_pos: float = 0.15
    lr_prof: float = 0.04
    lr_bg: float = 0.004
    verbose: bool = True
    log: list = field(default_factory=list)
    callback: object = None         # callback(net, stage_name) for debugging
    prune_stats: dict = field(default_factory=dict)   # why edges were removed (count, px)
    keep_pruned: list = None        # debugging: set to [] to collect pruned edges

    bg_free_below: float = 5.0      # background frozen for bands starting at >= this
    anchor_px: float = 5.0          # positional prior per optimisation round
    min_elongation: float = 2.5     # length / (2r + 2s) below which an edge is a blob
    wide_test_w: float = 8.0        # r + s from which an edge must also beat the background

    def priors(self):
        return dict(lam_bend=self.lam_bend, lam_prof=self.lam_prof, lam_bg=self.lam_bg)

    def anchor(self):
        # positions stay within a few px of where the round started; the
        # profiles are free (huge sigmas)
        return dict(sigma_pos=self.anchor_px, sigma_logprof=1e3, sigma_amp=1e3)


def _snap(cfg, net, stage):
    if cfg.callback is not None:
        cfg.callback(net, stage)


def _say(cfg, *a):
    msg = " ".join(str(x) for x in a)
    cfg.log.append(msg)
    if cfg.verbose:
        print(msg, flush=True)


# --------------------------------------------------------------- optimiser
def optimize(model: NetworkModel, iters: int, lr_pos=0.15, lr_prof=0.04, lr_bg=0.004,
             rebuild_every=25, track=None, priors=None, fit_pos=True, fit_prof=True,
             fit_bg=True, affine=False, lr_aff=(1e-3, 0.3)):
    groups = []
    if fit_pos:
        groups.append(dict(params=[model.node_xy, model.inner], lr=lr_pos))
    if fit_prof:
        groups.append(dict(params=[model.raw_r, model.raw_s, model.raw_a], lr=lr_prof))
        groups.append(dict(params=[model.raw_hw, model.raw_hs], lr=lr_prof * 0.5))
    if fit_bg and model.bg.requires_grad:
        groups.append(dict(params=[model.bg], lr=lr_bg))
    if affine:
        model.aff_A.requires_grad_(True)
        model.aff_t.requires_grad_(True)
        groups.append(dict(params=[model.aff_A], lr=lr_aff[0]))
        groups.append(dict(params=[model.aff_t], lr=lr_aff[1]))
    if not groups or iters <= 0:
        return []
    opt = torch.optim.Adam(groups)
    base = [g["lr"] for g in opt.param_groups]
    hist = []
    for it in range(iters):
        f = 0.5 * (1 + math.cos(math.pi * it / iters)) * 0.9 + 0.1
        for g, b in zip(opt.param_groups, base):
            g["lr"] = b * f
        opt.zero_grad(set_to_none=True)
        loss, nll, _, _ = model.loss(track=track, priors=priors)
        loss.backward()
        # parameters that are not fitted must not move
        if not fit_pos:
            for p in (model.node_xy, model.inner):
                p.grad = None
        opt.step()
        model.project()
        hist.append((float(loss.detach()), float(nll.detach())))
        if rebuild_every and (it + 1) % rebuild_every == 0 and it + 1 < iters:
            model.rebuild()
    model.rebuild()
    return hist


# --------------------------------------------------------------- proposals
def seeds_to_edges(net: VesselNetwork, seeds, band):
    new = []
    for sd in seeds:
        sc = np.clip(sd.scale, band[0], band[1])
        w = np.median(sc) / math.sqrt(2.0)                  # Gaussian-equivalent std
        s = max(S_MIN, 0.6 * w)
        r = max(R_MIN, 2.0 * math.sqrt(max(w * w - s * s, 0.04)))
        amp = float(np.clip(np.median(sd.amp), 0.005, 3.0))
        a = amp / float(profile_peak(r, s))
        n = len(sd.xy)
        eid = net.add_edge_dense(sd.xy, np.full(n, r), np.full(n, s), np.full(n, a),
                                 info=dict(band=list(band), z=float(np.median(sd.z))))
        new.append(eid)
    return new


def _edge_index(net: VesselNetwork, spacing=1.0):
    """KD-tree over dense samples of all edges, with tangents and reach."""
    xs, ts, rs, ids = [], [], [], []
    for eid in net.edges:
        smp = net.sample(eid, spacing)
        xs.append(smp["xy"])
        ts.append(smp["tan"])
        rs.append(np.stack([smp["r"], smp["s"]], 1))
        ids.append(np.full(len(smp["xy"]), eid))
    if not xs:
        return None
    X = np.concatenate(xs)
    return dict(tree=cKDTree(X), xy=X, tan=np.concatenate(ts), rs=np.concatenate(rs),
                eid=np.concatenate(ids))


def shadow_filter(seeds, net: VesselNetwork, min_len, cos_par=0.85, slack=1.0,
                  sharp_ratio=0.35, lumen_ratio=2.5):
    """Drop the parts of proposals that run parallel inside an existing
    vessel's footprint: they are misfit of that vessel, not new vessels.
    Misfit residuals cannot be sharper than the vessel's own blur, so a
    proposal much finer than the vessel's blur (a sharp capillary running
    over a defocused deep vessel) is kept."""
    idx = _edge_index(net)
    if idx is None:
        return seeds
    out = []
    for sd in seeds:
        xy = sd.xy
        tan = np.gradient(xy, axis=0)
        tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-9
        rmax = float(idx["rs"][:, 0].max() + idx["rs"][:, 1].max() + slack + 2)
        shadow = np.zeros(len(xy), bool)
        nb = idx["tree"].query_ball_point(xy, rmax)
        for i, cand in enumerate(nb):
            if not cand:
                continue
            cand = np.asarray(cand)
            d = np.linalg.norm(idx["xy"][cand] - xy[i], axis=1)
            reach = idx["rs"][cand, 0] + 0.7 * idx["rs"][cand, 1] + slack
            par = np.abs((idx["tan"][cand] * tan[i]).sum(1)) > cos_par
            coarse = sd.scale[i] >= sharp_ratio * idx["rs"][cand, 1]
            # any direction: a much finer ridge inside the lumen of a wide
            # vessel is texture or profile misfit of that vessel
            lumen = (d < idx["rs"][cand, 0] - 0.5) & (idx["rs"][cand, 0] >= lumen_ratio * sd.scale[i])
            if np.any((d < reach) & par & coarse) or np.any(lumen):
                shadow[i] = True
        # keep runs of un-shadowed points
        keep = ~shadow
        if keep.all():
            out.append(sd)
            continue
        lab = np.cumsum(np.r_[1, np.diff(keep.astype(int)) != 0])
        for l in np.unique(lab[keep]):
            m = lab == l
            sub = type(sd)(sd.xy[m], sd.scale[m], sd.amp[m], sd.z[m], [None, None])
            if sub.length >= min_len:
                out.append(sub)
    return out


# --------------------------------------------------------------- topology
def connect_ends(net: VesselNetwork, gap_factor=5.0, max_gap=40.0, cone_deg=40.0):
    """Bridge gaps between collinear free ends and snap free ends that run
    into another vessel onto it (creating a bifurcation)."""
    h, w = net.shape
    for _ in range(3):
        changed = False
        deg = net.degrees()
        free = [n for n in net.nodes if deg[n] == 1 and net.node_kind(n, 1) == "endpoint"]
        if not free:
            break
        info = {}
        for n in free:
            (eid, end), = net.incident(n)
            info[n] = (eid, end, net.end_tangent(eid, end), net.end_width(eid, end))
        # (a) end-to-end continuation
        cands = []
        P = np.array([net.nodes[n].xy for n in free])
        tree = cKDTree(P)
        for i, n in enumerate(free):
            eid, end, t, wd = info[n]
            G = min(max_gap, max(10.0, gap_factor * wd))
            for j in tree.query_ball_point(P[i], G):
                m = free[j]
                if m == n or info[m][0] == eid:
                    continue
                v = P[j] - P[i]
                dist = np.linalg.norm(v)
                if dist < 1e-6:
                    cands.append((0.0, n, m))
                    continue
                c1 = np.dot(v / dist, t)
                c2 = np.dot(-v / dist, info[m][2])
                c3 = np.dot(t, info[m][2])
                if c1 > math.cos(math.radians(cone_deg)) and c2 > math.cos(math.radians(cone_deg)) and c3 < -0.5:
                    wr = max(wd, info[m][3]) / min(wd, info[m][3])
                    if wr < 3.0:
                        cands.append((dist * (2.5 - c1 - c2) * (1 + 0.3 * math.log(wr)), n, m))
        cands.sort()
        used = set()
        for _, n, m in cands:
            if n in used or m in used or n not in net.nodes or m not in net.nodes:
                continue
            (e1, end1), = net.incident(n)
            (e2, end2), = net.incident(m)
            far1 = net.edges[e1].v if end1 == 0 else net.edges[e1].u
            far2 = net.edges[e2].v if end2 == 0 else net.edges[e2].u
            if e1 == e2 or far1 == far2:
                continue
            net.join_edges(e1, end1, e2, end2)
            used |= {n, m}
            changed = True
        # (b) snapping into another vessel
        idx = _edge_index(net, 1.0)
        if idx is None:
            break
        rmax = float(idx["rs"][:, 0].max())
        deg = net.degrees()
        for n in list(net.nodes):
            if n not in net.nodes or deg.get(n, 0) != 1 or net.node_kind(n, 1) != "endpoint":
                continue
            inc = net.incident(n)
            if len(inc) != 1:
                continue
            (eid, end), = inc
            t = net.end_tangent(eid, end)
            wd = net.end_width(eid, end)
            p = net.nodes[n].xy
            reach = max(6.0, 2.5 * wd)
            cand = np.asarray(idx["tree"].query_ball_point(p, reach + rmax), dtype=int)
            if cand.size == 0:
                continue
            cand = cand[idx["eid"][cand] != eid]
            if cand.size == 0:
                continue
            v = idx["xy"][cand] - p
            dist = np.linalg.norm(v, axis=1)
            along = (v * t).sum(1)
            inside = dist < idx["rs"][cand, 0] + 1.0          # end already inside
            ahead = (dist < reach + idx["rs"][cand, 0]) & (along >= 0.6 * dist)
            ok = inside | ahead
            if not ok.any():
                continue
            k = cand[ok][np.argmin(dist[ok])]
            q = idx["xy"][k]
            target = _current_edge_at(net, int(idx["eid"][k]), q, exclude=eid)
            if target is None:
                continue
            nid = net.split_edge(target, q)
            if nid == n:
                continue
            # extend the free end to the junction node, then merge the nodes;
            # the part of the free end already inside the target vessel is
            # dropped first so the extension cannot double back
            smp = net.sample(eid, 0.7)
            qn = net.nodes[nid].xy
            inner_r = float(idx["rs"][k, 0])
            dq = np.linalg.norm(smp["xy"] - qn, axis=1)
            keep = np.ones(len(dq), bool)
            if end == 0:
                j = 0
                while j < len(dq) - 3 and dq[j] < inner_r:
                    keep[j] = False
                    j += 1
            else:
                j = len(dq) - 1
                while j > 2 and dq[j] < inner_r:
                    keep[j] = False
                    j -= 1
            smp = {k_: v_[keep] for k_, v_ in smp.items()}
            xy = smp["xy"]
            if end == 0:
                xy = np.vstack([qn, xy])
                r, s, a = (np.r_[smp[k_][0], smp[k_]] for k_ in ("r", "s", "a"))
            else:
                xy = np.vstack([xy, qn])
                r, s, a = (np.r_[smp[k_], smp[k_][-1]] for k_ in ("r", "s", "a"))
            net.merge_nodes(nid, n)
            if eid in net.edges:
                net.refit_edge(eid, xy, r, s, a)
            changed = True
            deg = net.degrees()
        net.merge_joints()
        if not changed:
            break


def _current_edge_at(net: VesselNetwork, eid_hint, q, exclude=None, tol=2.0):
    """The edge that now passes through q: eid_hint if it still exists,
    otherwise the nearest current edge (edges get split while snapping)."""
    if eid_hint in net.edges:
        return eid_hint
    best, bd = None, tol
    for k, e in net.edges.items():
        if k == exclude:
            continue
        c = e.ctrl
        if (q[0] < c[:, 0].min() - 30 or q[0] > c[:, 0].max() + 30 or
                q[1] < c[:, 1].min() - 30 or q[1] > c[:, 1].max() + 30):
            continue
        d = np.linalg.norm(net.sample(k, 1.0)["xy"] - q, axis=1).min()
        if d < bd:
            best, bd = k, d
    return best


def clean_topology(net: VesselNetwork, min_length=6.0):
    net.merge_close_nodes(1.5)
    net.remove_short_loops()
    net.resolve_crossings()
    net.resolve_near_crossings()
    net.merge_joints()
    for eid in list(net.edges):
        if eid in net.edges and net.length(eid) < min_length:
            deg = net.degrees()
            e = net.edges[eid]
            # short spurs from a junction to nowhere are skeleton artefacts
            if deg[e.u] == 1 or deg[e.v] == 1:
                net.remove_edge(eid)
    net.merge_joints()


def remove_duplicates(net: VesselNetwork, gains: dict, frac=0.6, cos_par=0.9, ratio=2.0):
    """Remove edges that mostly run inside and parallel to a stronger edge of
    similar calibre and blur (a sharp thin vessel over a blurred wide one is
    a different vessel, at another depth)."""
    order = sorted(net.edges, key=lambda e: gains.get(e, 0.0))
    removed = []
    for eid in order:
        if eid not in net.edges or len(net.edges) < 2:
            continue
        others = {k: v for k, v in net.edges.items() if k != eid}
        smp = net.sample(eid, 1.0)
        me = net.edges[eid]
        w_me = float(np.mean(me.r) + np.mean(me.s))
        s_me = float(np.mean(me.s))
        # build a light index of the other edges near this one
        bb0 = smp["xy"].min(0) - 40
        bb1 = smp["xy"].max(0) + 40
        xs, ts, rs = [], [], []
        for k in others:
            e = net.edges[k]
            c = e.ctrl
            if (c[:, 0].max() < bb0[0] or c[:, 0].min() > bb1[0] or
                    c[:, 1].max() < bb0[1] or c[:, 1].min() > bb1[1]):
                continue
            if gains.get(k, 0.0) < gains.get(eid, 0.0):
                continue
            w_k = float(np.mean(e.r) + np.mean(e.s))
            s_k = float(np.mean(e.s))
            if max(w_k, w_me) / min(w_k, w_me) > ratio or max(s_k, s_me) / min(s_k, s_me) > ratio:
                continue
            so = net.sample(k, 1.0)
            xs.append(so["xy"])
            ts.append(so["tan"])
            rs.append(so["r"] + 0.7 * so["s"] + 1.0)
        if not xs:
            continue
        X = np.concatenate(xs)
        T = np.concatenate(ts)
        Rr = np.concatenate(rs)
        d, j = cKDTree(X).query(smp["xy"])
        par = np.abs((T[j] * smp["tan"]).sum(1)) > cos_par
        inside = (d < Rr[j]) & par
        if inside.mean() > frac:
            net.remove_edge(eid)
            removed.append(eid)
    removed += remove_inside_lumen(net)
    if removed:
        net.merge_joints()
    return removed


def remove_inside_lumen(net: VesselNetwork, ratio=3.0, frac=0.6):
    """Remove edges that run mostly inside the lumen of a vessel at least
    `ratio` times wider (texture or profile misfit of that vessel)."""
    big = [k for k, e in net.edges.items() if float(np.median(e.r)) >= 3.0]
    if not big:
        return []
    xs, rs, ids = [], [], []
    for k in big:
        so = net.sample(k, 1.0)
        xs.append(so["xy"])
        rs.append(so["r"])
        ids.append(np.full(len(so["xy"]), k))
    X, Rr, I = np.concatenate(xs), np.concatenate(rs), np.concatenate(ids)
    tree = cKDTree(X)
    removed = []
    for k in list(net.edges):
        e = net.edges[k]
        rk = float(np.median(e.r) + 0.5 * np.median(e.s))
        smp = net.sample(k, 1.0)
        d, j = tree.query(smp["xy"])
        inside = (d < Rr[j] - 0.5) & (Rr[j] >= ratio * rk) & (I[j] != k)
        if inside.mean() > frac:
            net.remove_edge(k)
            removed.append(k)
    return removed


def reparameterize(net: VesselNetwork, tol=0.25):
    """Refit edges whose length changed so much during optimisation that
    their control-point spacing no longer matches, and give edges that
    turned out thinner than proposed the finer spacing their calibre calls
    for (thin vessels can be tortuous)."""
    for eid in list(net.edges):
        e = net.edges[eid]
        L = net.length(eid)
        sp_ = e.info.get("spacing", 12.0)
        cal = float(np.median(e.r) + np.median(e.s))
        sp_cal = pos_spacing_for(cal)
        refine = sp_cal < 0.75 * sp_
        if refine:
            sp_ = sp_cal
        want = max(4, int(np.ceil(L / sp_)) + 1)
        if refine or abs(want - len(e.ctrl)) > max(1, tol * len(e.ctrl)):
            smp = net.sample(eid, 0.7)
            net.refit_edge(eid, smp["xy"], smp["r"], smp["s"], smp["a"], spacing=sp_)


def n_params(net: VesselNetwork, eid, L=None) -> float:
    """Effective number of free parameters of an edge for the MDL test.
    Control points are tied together by the bending/cusp priors, so the
    centreline counts as ~2 dof per 15 px of length (not 2 per control
    point, which would make finely parameterised thin vessels artificially
    expensive), plus its two ends and three profiles (one dof per 30 px
    each)."""
    L = net.length(eid) if L is None else L
    # profiles likewise by length (faithful re-fits of split / joined edges
    # carry denser profile knots, which must not make them look costlier)
    return 4.0 + 2.0 * L / 15.0 + 3.0 * max(2.0, L / PROFILE_SPACING + 1.0)


def score_and_prune(net, model: NetworkModel, cfg: MapConfig, protect=()):
    gains, counts = model.edge_gains()
    # wide edges must also beat the background: their gain is re-evaluated
    # with the low-frequency part of their contribution given to it
    wide = [k for k, eid in enumerate(model.eids) if eid in net.edges and
            float(np.mean(net.edges[eid].r) + np.mean(net.edges[eid].s)) >= cfg.wide_test_w]
    if wide:
        gperp = model.edge_gains_bg_orthogonal(wide)
        for k in wide:
            gains[k] = min(gains[k], gperp[k])
    gdict = {}
    removed = []
    for k, eid in enumerate(model.eids):
        if eid not in net.edges:
            continue
        L = max(net.length(eid), 1.0)
        pen = cfg.penalty_scale * 0.5 * n_params(net, eid, L) * math.log(max(counts[k], 2.0))
        net.edges[eid].info.update(gain=float(gains[k]), penalty=float(pen),
                                   gain_per_px=float(gains[k] / L))
        gdict[eid] = float(gains[k])
        # protected edges (e.g. the map a refinement started from) are only
        # removed when they make the fit worse
        if eid in protect or (net.edges[eid].info.get("protected") and gains[k] > 0):
            continue
        st_w = float(np.mean(net.edges[eid].r) + np.mean(net.edges[eid].s))
        dg = net.degrees()
        blob = L < cfg.min_elongation * 2.0 * st_w and \
            max(dg[net.edges[eid].u], dg[net.edges[eid].v]) <= 1
        why = ("mdl" if gains[k] < pen else "density" if gains[k] / L < cfg.min_gain_per_px
               else "blob" if blob else None)
        if why:
            removed.append(eid)
            r_ = cfg.prune_stats.setdefault(why, [0, 0.0])
            r_[0] += 1
            r_[1] += L
            if cfg.keep_pruned is not None:
                cfg.keep_pruned.append(dict(xy=net.sample(eid, 1.0)["xy"], why=why,
                                            gain=float(gains[k]), pen=float(pen), L=L,
                                            w=st_w, a=float(np.mean(net.edges[eid].a))))
    for eid in removed:
        if eid in net.edges:
            net.remove_edge(eid)
    net.merge_joints()
    dup = remove_duplicates(net, gdict)
    r_ = cfg.prune_stats.setdefault("duplicate", [0, 0.0])
    r_[0] += len(dup)
    return removed + dup, gdict


# --------------------------------------------------------------- the map
def band_scales(band, n):
    return np.geomspace(band[0], band[1], n)


def residual_image(model: NetworkModel) -> np.ndarray:
    with torch.no_grad():
        return (model.logI - model.predict()).numpy()


def build_map(intensity: np.ndarray, cfg: MapConfig | None = None,
              prepared: Prepared | None = None) -> VesselNetwork:
    """Discover the full vessel network of a single image."""
    cfg = cfg or MapConfig()
    P = prepared or prepare(intensity)
    net = VesselNetwork(P.shape)
    t0 = time.time()
    # the background starts as a robust upper envelope that ignores dark
    # (vessel) pixels, and stays frozen while the coarse vessels are found:
    # a least-squares background fitted first would absorb wide vessels
    model = NetworkModel(net, P.logI, P.weight, stride=2, bg_spacing=cfg.bg_spacing)
    model.write_back()
    for band in cfg.bands:
        lr_bg = 0.0 if band[0] >= cfg.bg_free_below else cfg.lr_bg
        stride = 2 if band[0] >= 5.0 else 1
        for rep in range(cfg.reps_per_band):
            model = NetworkModel(net, P.logI, P.weight, stride=stride, bg_spacing=cfg.bg_spacing)
            res = residual_image(model)
            seeds = detect(-res, P.sigma, band_scales(band, cfg.scales_per_band),
                           z_hi=cfg.z_hi, z_lo=cfg.z_lo,
                           min_len=max(cfg.min_length, 2.5 * band[0]), valid=P.valid)
            n_raw = len(seeds)
            seeds = shadow_filter(seeds, net, max(cfg.min_length, 2.5 * band[0]))
            if not seeds:
                _say(cfg, f"[{time.time()-t0:6.0f}s] band {band} rep {rep}: no proposals")
                break
            new = seeds_to_edges(net, seeds, band)
            _snap(cfg, net, f"b{band[0]}_r{rep}_1seeds")
            clean_topology(net, cfg.min_length)
            connect_ends(net)
            net.resolve_near_crossings()
            _snap(cfg, net, f"b{band[0]}_r{rep}_2topo")
            model = NetworkModel(net, P.logI, P.weight, stride=stride, bg_spacing=cfg.bg_spacing,
                                 ref=net)
            optimize(model, cfg.iters_band, cfg.lr_pos, cfg.lr_prof, lr_bg,
                     cfg.rebuild_every, priors=cfg.priors(), track=cfg.anchor(),
                     fit_bg=lr_bg > 0)
            model.write_back()
            reparameterize(net)
            _snap(cfg, net, f"b{band[0]}_r{rep}_3fit")
            n_before = len(net.edges)
            removed, _ = score_and_prune(net, model, cfg)
            new_alive = len(set(new) & set(net.edges))
            _snap(cfg, net, f"b{band[0]}_r{rep}_4pruned")
            ps = {k: (v[0], round(v[1])) for k, v in cfg.prune_stats.items()}
            cfg.prune_stats.clear()
            _say(cfg, f"   pruned (count, px): {ps}")
            _say(cfg, f"[{time.time()-t0:6.0f}s] band {band} rep {rep}: {n_raw} ridges -> "
                      f"{len(seeds)} proposals ({len(new)} edges), {new_alive} new edges kept; "
                      f"pruned {len(removed)} of {n_before}; network {net.summary()}")
            if len(new) and new_alive == 0 and len(removed) >= len(new):
                break
    # final joint refinement at full resolution
    clean_topology(net, cfg.min_length)
    connect_ends(net)
    net.resolve_near_crossings()
    model = NetworkModel(net, P.logI, P.weight, stride=1, bg_spacing=cfg.bg_spacing, ref=net)
    optimize(model, cfg.iters_final, cfg.lr_pos * 0.7, cfg.lr_prof, cfg.lr_bg,
             cfg.rebuild_every, priors=cfg.priors(), track=cfg.anchor())
    model.write_back()
    reparameterize(net)
    score_and_prune(net, model, cfg)
    model = NetworkModel(net, P.logI, P.weight, stride=1, bg_spacing=cfg.bg_spacing)
    optimize(model, 60, cfg.lr_pos * 0.3, cfg.lr_prof * 0.5, cfg.lr_bg * 0.5,
             cfg.rebuild_every, priors=cfg.priors())
    model.write_back()
    gains, _ = model.edge_gains()
    for k, eid in enumerate(model.eids):
        net.edges[eid].info["gain"] = float(gains[k])
    net.orient_structural()
    net.meta.update(builder="vesselmap.build_map", seconds=round(time.time() - t0, 1),
                    config={k: v for k, v in asdict(cfg).items()
                            if k not in ("log", "callback", "prune_stats", "keep_pruned")},
                    final_nll=float(model.data_nll(model.predict()).detach()))
    _say(cfg, f"[{time.time()-t0:6.0f}s] done: {net.summary()}")
    return net


# --------------------------------------------------------------- per frame
@dataclass
class FrameFitConfig:
    iters_align: int = 30           # global affine pre-alignment (stride 2)
    iters: int = 40                 # joint refinement under the map prior
    stride: int = 1
    rebuild_every: int = 20
    sigma_pos: float = 3.0          # px: how far control points may drift
    sigma_logprof: float = 0.35     # relative change of width / blur
    sigma_amp: float = 0.6          # relative change of contrast
    lr_pos: float = 0.1
    lr_prof: float = 0.03
    lr_bg: float = 0.004
    visible_gain_per_px: float = 1.0


def fit_frame(intensity: np.ndarray, ref: VesselNetwork, cfg: FrameFitConfig | None = None,
              prepared: Prepared | None = None, priors=None, init: VesselNetwork | None = None):
    """Adjust a reference map to one (stabilised) frame.

    ref   the map (from build_map), never modified.
    init  optional starting point, e.g. the result for the previous frame
          (faster convergence); the prior still pulls towards `ref`.

    Only this frame's intensities are used.  Returns (network, report).  The
    network has the same node and edge ids as `ref`; only positions and
    profiles move, within the prior set by `cfg`.  Each edge gets
    info["visible"], info["frame_gain"] and info["gain_per_px"].
    """
    cfg = cfg or FrameFitConfig()
    P = prepared or prepare(intensity)
    t0 = time.time()
    priors = priors or MapConfig().priors()
    base = ref.copy()
    align = None
    if cfg.iters_align > 0 and base.edges:
        m = NetworkModel(base, P.logI, P.weight, stride=2)
        # large capture range: phase correlation of the rendered vessels with
        # the frame's high-passed image gives the starting translation
        shift = phase_shift(m, P)
        with torch.no_grad():
            m.aff_t.copy_(torch.tensor(shift, dtype=torch.float32))
        optimize(m, cfg.iters_align, lr_bg=cfg.lr_bg, priors=priors, fit_pos=False,
                 fit_prof=False, affine=True)
        align = dict(A=(np.eye(2) + m.aff_A.detach().numpy()).tolist(),
                     t=m.aff_t.detach().numpy().tolist())
        m.write_back()
    if init is not None:
        net = _carry_deformation(ref, base, init)
    else:
        net = base.copy()
    model = NetworkModel(net, P.logI, P.weight, stride=cfg.stride, ref=base)
    track = dict(sigma_pos=cfg.sigma_pos, sigma_logprof=cfg.sigma_logprof, sigma_amp=cfg.sigma_amp)
    hist = optimize(model, cfg.iters, cfg.lr_pos, cfg.lr_prof, cfg.lr_bg, cfg.rebuild_every,
                    track=track, priors=priors)
    model.write_back()
    gains, _ = model.edge_gains()
    vis = {}
    for k, eid in enumerate(model.eids):
        L = max(net.length(eid), 1.0)
        e = net.edges[eid]
        e.info["frame_gain"] = float(gains[k])
        e.info["gain_per_px"] = float(gains[k] / L)
        e.info["visible"] = bool(gains[k] / L >= cfg.visible_gain_per_px)
        vis[eid] = e.info["visible"]
    disp = [np.linalg.norm(net.nodes[n].xy - base.nodes[n].xy) for n in net.nodes if n in base.nodes]
    report = dict(seconds=round(time.time() - t0, 2), affine=align,
                  final_nll=hist[-1][1] if hist else None,
                  node_shift_median=float(np.median(disp)) if disp else 0.0,
                  node_shift_p95=float(np.percentile(disp, 95)) if disp else 0.0,
                  visible_fraction=float(np.mean(list(vis.values()))) if vis else 0.0)
    net.meta["frame_fit"] = report
    return net, report


def apply_affine(net: VesselNetwork, A, t):
    """Apply p -> (p - c) A^T + c + t (c = image centre, as in the renderer's
    affine) to every node and control point of net, in place."""
    A = np.asarray(A, float)
    t = np.asarray(t, float)
    H, W = net.shape
    c = np.array([W / 2.0, H / 2.0])
    f = lambda p: (p - c) @ A.T + c + t
    for n in net.nodes.values():
        n.x, n.y = (float(v) for v in f(n.xy))
    for e in net.edges.values():
        e.ctrl = f(e.ctrl)


def _carry_deformation(ref: VesselNetwork, base: VesselNetwork, init: VesselNetwork):
    """Start for this frame: its globally aligned map (base) plus the local
    deformation the previous frame's fit (init) had relative to its own
    aligned map; profiles are taken from init."""
    net = base.copy()
    prev = ref.copy()
    aff = (init.meta.get("frame_fit") or {}).get("affine")
    if aff:
        apply_affine(prev, aff["A"], aff["t"])
    for k, n in net.nodes.items():
        if k in init.nodes and k in prev.nodes:
            d = init.nodes[k].xy - prev.nodes[k].xy
            n.x, n.y = (float(v) for v in n.xy + d)
    for k, e in net.edges.items():
        if k in init.edges and k in prev.edges and init.edges[k].ctrl.shape == e.ctrl.shape:
            e.ctrl = e.ctrl + (init.edges[k].ctrl - prev.edges[k].ctrl)
            ie = init.edges[k]
            if ie.r.shape == e.r.shape:
                e.r, e.s, e.a = ie.r.copy(), ie.s.copy(), ie.a.copy()
        net.sync_ends(k)
    return net


def phase_shift(model: NetworkModel, P: Prepared, hp_sigma=15.0):
    """Translation (dx, dy) that moves the model's vessels onto the frame,
    by phase correlation of the rendered vessel density with the frame's
    high-passed negative log image."""
    import cv2
    with torch.no_grad():
        full = model.stride == 1
        V = model.optical_density().numpy() if full else None
    if V is None:
        m1 = NetworkModel(model.net, P.logI, P.weight, stride=1)
        with torch.no_grad():
            V = m1.optical_density().numpy()
    L = np.where(P.valid, P.logI, np.median(P.logI)).astype(np.float32)
    hp = -(L - cv2.GaussianBlur(L, (0, 0), hp_sigma))
    hp = np.clip(hp, 0, None)
    win = cv2.createHanningWindow(hp.shape[::-1], cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(V.astype(np.float32), hp, win)
    if resp < 0.02:
        return (0.0, 0.0)
    return (float(dx), float(dy))


def fit_frames(frames, ref: VesselNetwork, cfg: FrameFitConfig | None = None, chain=True,
               callback=None):
    """fit_frame over an iterable of images (arrays or paths).  With chain,
    each frame starts from the previous frame's result.  Yields
    (index, network, report)."""
    from .image import load_image
    prev = None
    for i, f in enumerate(frames):
        I = load_image(f) if isinstance(f, (str, bytes)) or hasattr(f, "__fspath__") else f
        net, rep = fit_frame(I, ref, cfg, init=prev if chain else None)
        prev = net
        if callback:
            callback(i, net, rep)
        yield i, net, rep
