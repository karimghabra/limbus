"""Energy search: the fewest vessels that still render the image.

``consolidate_map`` (consolidate.py) joins segments that pass fixed
geometric tests (turn <= 40 deg, calibre and blur within bounds, an
unambiguous pair at a fork) and a per-link image test.  What it cannot join
stays in pieces, and nothing removes a vessel that is too weak to be one.
``search_map`` goes further by minimising one explicit objective over the
structure of the map:

    E(M) = NLL(image | M) + prior(M) + sum_v (tau * lam_vessel + price * L_v)

* NLL and prior are exactly the renderer's score (render.py), with the
  windowed calibre prior of consolidation (a long vessel may taper).
* The last term is new.  Every vessel pays a fixed ``tau * lam_vessel``, so
  two pieces cost more than the one vessel they belong to.  ``tau`` is the
  data temperature, the reduced chi-square of the map's residual:
  structured misfit inflates NLL differences by about that factor.
* Every pixel of centreline pays ``price``, so a vessel must explain enough
  of the image per pixel to be kept.  The price is set by the *texture
  null*: tissue texture makes bright ridges as often as dark ones, and
  vessels are only dark, so 'anti-vessels' fitted to the bright ridges of
  the residual show how much a dark trace of texture explains per pixel.
  A vessel that explains no more than that (``null_quantile``) is not told
  apart from texture.  (At least ``tau * lam_length``.)

Moves.  Each move is followed by a short gradient fit of what it changes:

    join     two vessel ends into one spline (ends that met at a node, a
             bridged gap, or overlapping ends of a vessel found twice)
    delete   a vessel (vessels duplicating it are re-fitted to take over)
    split    a vessel at a sharp bend, where the trace may switch vessels
    reroute  at a branch point, which arm continues the parent: an end
             touching a vessel takes over the vessel's far part, and the
             near part ends there
    revive   a vessel deleted earlier (so every move can be undone)

Exact local energy.  A move changes the image only near the vessels it
touches, so its energy change is computed on a window, rendering only
those vessels over the fixed rest of the model (the other vessels' optical
density, halo included, and the background).  A join, split or reroute
changes a vessel only near the junction: its fit frees the parameters
within ``focus_radius`` of it and scores the pixels around it, whatever
the vessel's length.

Search.  Rejection-free annealing: at every step every candidate move is
scored and one is drawn with probability proportional to exp(-dE / T),
staying put (dE = 0) included.  A score is cached until the optical
density of the other vessels in its window changes, and stale scores are
re-fitted in parallel worker processes.  At a high temperature the search
can leave a local minimum, e.g. join two pieces whose join only pays once
a duplicate between them is gone.  As T falls it becomes steepest descent,
and ends when no move lowers E.  A joint fit of every vessel and the
background follows, then a last greedy pass.

The result uses the map's usual representation (network.py): one edge per
vessel, and a branch ends at a node its parent passes through
(``info["through"]``), so ``to_segments`` / ``to_digraph`` still give the
segment graph.  Like the rest of vesselmap, only the intensities of one
image are used.
"""
from __future__ import annotations

import math
import multiprocessing
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from scipy.spatial import cKDTree

from . import spline as sp
from .fit import MapConfig, optimize
from .image import Prepared, prepare
from .network import (A_MIN, PROFILE_SPACING, R_MIN, S_MIN, Edge, VesselNetwork,
                      _fit_edge_params, despike)
from .render import NetworkModel, _cv_blur

BG_SPACING = 64.0


@dataclass
class SearchConfig:
    lam_vessel: float = 150.0       # cost of one vessel (x tau)
    lam_length: float = 1.0         # cost per px of centreline (x tau), at least ...
    texture_null: bool = True       # ... what background texture alone explains per px:
    null_quantile: float = 0.9      # this quantile of the evidence per px of 'anti-vessels'
    null_bands: tuple = ((1.2, 2.5), (2.5, 5.0), (5.0, 10.0))   # fitted to bright ridges
    tau: float = 0.0                # data temperature; <= 0: reduced chi-square of the map
    t_start: float = 0.1            # first temperature, as a fraction of tau * lam_vessel
    hot_steps: float = 0.5          # annealing steps (x number of input edges) before T = 0
    hopeless: float = 10.0          # a move scored worse than this (x tau * lam_vessel) is
                                    # not re-scored while its vessels exist
    local_iters: int = 40           # gradient steps after a move
    focus_radius: float = 30.0      # a join / split / reroute re-fits the vessel within this
                                    # distance (px) of the junction; the rest stays as it was
    lr_pos: float = 0.1
    lr_prof: float = 0.03
    join_gap: float = 40.0          # largest gap bridged (px) ...
    join_gap_factor: float = 6.0    # ... and at most this many vessel widths
    join_cone_deg: float = 60.0     # a gap is bridged only roughly straight ahead
    max_turn_deg: float = 120.0     # joins turning more sharply are not tried
    trim_max: float = 25.0          # overlap of two ends that a join may trim
    split_turn_deg: float = 35.0    # bends sharper than this are split candidates
    init_iters: int = 0             # joint fit before the search (unfitted input)
    global_iters: int = 150         # joint fit of everything after the search
    attach_tol: float = 1.5         # px beyond a vessel's r + s where a branch end attaches
    through_iters: int = 40         # joint fit once branch ends sit on their parents
    calibre_window: int = 4         # profile knots the calibre prior averages over (as in
                                    # consolidation: a long vessel may taper)
    workers: int = 0                # processes scoring moves in parallel; 0: one per CPU
    seed: int = 0
    verbose: bool = True


# ------------------------------------------------------------------ helpers
def edge_samples(e: Edge, spacing: float = 1.0) -> dict:
    """Samples of an edge evenly spaced in arclength (like
    VesselNetwork.sample, for edges that are not in a network)."""
    n = len(e.ctrl)
    m0 = max(128, 16 * n)
    xy_d = sp.design(n, m0) @ e.ctrl
    prof_d = sp.design(len(e.r), m0) @ np.stack([e.r, e.s, e.a], 1)
    s_d = sp.arclength(xy_d)
    L = float(s_d[-1])
    m = sp.n_samples_for_length(L, spacing)
    if L <= 0:
        xy = np.repeat(xy_d[:1], m, 0)
        pr = np.repeat(prof_d[:1], m, 0)
    else:
        q = np.linspace(0.0, L, m)
        f = lambda a: np.stack([np.interp(q, s_d, a[:, j]) for j in range(a.shape[1])], 1)
        xy, pr = f(xy_d), f(prof_d)
    tan = np.gradient(xy, axis=0)
    tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-9
    return dict(xy=xy, tan=tan, s_arc=sp.arclength(xy), r=np.maximum(pr[:, 0], 0.3),
                s=np.maximum(pr[:, 1], 0.6), a=np.maximum(pr[:, 2], 1e-3), L=L)


def _flip(smp):
    out = {k: (v[::-1].copy() if isinstance(v, np.ndarray) else v) for k, v in smp.items()}
    out["tan"] = -out["tan"]
    out["s_arc"] = out["L"] - out["s_arc"]
    return out


def _direction(xy, i, forward=True, look=4):
    """Unit direction of a polyline at index i, averaged over a few points."""
    j = min(len(xy) - 1, i + look) if forward else max(0, i - look)
    v = xy[j] - xy[i] if forward else xy[i] - xy[j]
    n = np.linalg.norm(v)
    if n < 1e-6:
        v = xy[-1] - xy[0]
        n = np.linalg.norm(v) + 1e-9
    return v / n


def join_path(A: dict, B: dict, trim_max: float, max_turn_deg: float, spacing=0.7):
    """Dense path of the vessel formed by A (joining end last) and B
    (joining end first): A's tail and B's head are cut where they come
    closest (so overlapping ends of a duplicate are trimmed), and a gap is
    bridged by a cubic Hermite curve that leaves A and enters B along their
    own directions.  Returns (xy, r, s, a) or None when the join would turn
    back on itself."""
    sA, sB = A["s_arc"], B["s_arc"]
    tail = np.flatnonzero(sA >= sA[-1] - trim_max)
    head = np.flatnonzero(sB <= trim_max)
    D = np.linalg.norm(A["xy"][tail][:, None] - B["xy"][head][None], axis=2)
    cost = D + 0.15 * ((sA[-1] - sA[tail])[:, None] + sB[head][None])
    i, j = np.unravel_index(int(np.argmin(cost)), cost.shape)
    ia, jb = int(tail[i]), int(head[j])
    if ia < 1 or jb > len(B["xy"]) - 2:
        return None
    pA, pB = A["xy"][ia], B["xy"][jb]
    tA = _direction(A["xy"], ia, forward=False)
    tB = _direction(B["xy"], jb, forward=True)
    if np.dot(tA, tB) < math.cos(math.radians(max_turn_deg)):
        return None
    h = float(np.linalg.norm(pB - pA))
    if h > 1.5:                          # a real gap (not two samples of one point)
        c = (pB - pA) / h
        if np.dot(tA, c) < -0.2 or np.dot(tB, c) < -0.2:     # the bridge would double back
            return None
    nb = int(math.ceil(h / spacing))
    a1, a2 = A["xy"][:ia + 1], B["xy"][jb:]
    if nb > 1:
        t = np.linspace(0, 1, nb + 1)[1:-1]
        h00, h10 = 2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t
        h01, h11 = -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2
        br = (h00[:, None] * pA + h10[:, None] * h * tA +
              h01[:, None] * pB + h11[:, None] * h * tB)
        xy = np.concatenate([a1, br, a2], 0)
    else:
        xy = np.concatenate([a1, a2], 0)
    r, s, a = _join_profiles(A, ia, B, jb, nb)
    xy, r, s, a = despike(xy, r, s, a)
    return xy, r, s, a


def fit_vessel(xy, r, s, a, spacing):
    """Spline parameters of a vessel assembled from pieces: the centreline
    follows the pieces faithfully, the profiles are fitted with the map's
    usual knot spacing and smoothing, in the log domain, after dropping the
    values within one vessel width of each piece's free end, where the end
    cap makes them unreliable (see `_join_profiles`)."""
    ctrl, _, _, _ = _fit_edge_params(xy, r, s, a, spacing, faithful=True)
    arc = sp.arclength(np.asarray(xy, float))
    L = float(arc[-1])
    n_p = sp.n_ctrl_for_length(L, PROFILE_SPACING, minimum=2)
    m = max(4 * n_p, int(np.ceil(L / 0.5)) + 1)
    q = np.linspace(0, L, m) if L > 0 else np.zeros(m)
    out = []
    for vals, lo in ((r, R_MIN), (s, S_MIN), (a, A_MIN)):
        lv = np.log(np.maximum(np.asarray(vals, float), lo))
        vu = np.interp(q, arc, lv) if L > 0 else np.full(m, lv.mean())
        c = sp.fit_ctrl(vu, n_p, smooth=1e-2, pin_ends=False)[:, 0]
        out.append(np.maximum(np.exp(c), lo))
    return ctrl, out[0], out[1], out[2]


def _join_profiles(A, ia, B, jb, nb):
    """Profile samples (r, s, a) along a joined path: A up to ia, a bridge of
    nb - 1 points, B from jb.  Within one vessel width of the joined ends
    the fitted profiles are distorted by the end caps, so there they are
    replaced by values from further inside, interpolated across the bridge."""
    out = []
    for k in ("r", "s", "a"):
        a1 = A[k][:ia + 1].copy()
        a2 = B[k][jb:].copy()
        wa = float(A["r"][ia] + 2 * A["s"][ia])
        wb = float(B["r"][jb] + 2 * B["s"][jb])
        sa = A["s_arc"][:ia + 1]
        sb = B["s_arc"][jb:] - B["s_arc"][jb]
        ka = np.flatnonzero(sa <= sa[-1] - wa)
        kb = np.flatnonzero(sb >= wb)
        va = a1[ka[-1]] if len(ka) else np.median(a1)
        vb = a2[kb[0]] if len(kb) else np.median(a2)
        if len(ka):
            a1[ka[-1] + 1:] = va
        if len(kb):
            a2[:kb[0]] = vb
        t = np.linspace(0, 1, nb + 1)[1:-1] if nb > 1 else np.zeros(0)
        out.append(np.concatenate([a1, va * (1 - t) + vb * t, a2]))
    return out


def _merge_info(infos):
    info = {}
    bands = [i["band"] for i in infos if i.get("band")]
    if bands:
        info["band"] = [min(b[0] for b in bands), max(b[1] for b in bands)]
    info["consolidated_from"] = sorted({f for i in infos for f in i.get("consolidated_from", [])})
    info["spacing"] = min(i.get("spacing", 12.0) for i in infos)
    return info


def explode(net: VesselNetwork) -> VesselNetwork:
    """Copy of net in which every edge is a free vessel with its own two end
    nodes and no through nodes.  The search works on vessels alone; which
    vessel a branch leaves is worked out again at the end (`to_through`)."""
    out = net.copy()
    for e in out.edges.values():
        e.info.pop("through", None)
    out._touch()
    for nid in [n for n in out.nodes if not out.incident(n)]:
        del out.nodes[nid]
    for nid in list(out.nodes):
        inc = out.incident(nid)
        for eid, end in inc[1:]:
            new = out.add_node(*out.nodes[nid].xy)
            e = out.edges[eid]
            if end == 0:
                e.u = new
            else:
                e.v = new
        out._touch()
    for eid, e in out.edges.items():
        e.info.setdefault("consolidated_from", [int(eid)])
        e.info.pop("gain", None)
    return out


# ------------------------------------------------------------------ search
class VesselSearch:
    """State of the search: the network (one edge per vessel, private end
    nodes), the fixed background image and the running sharp-core vessel
    image V = sum of every vessel's rendered patch."""

    def __init__(self, net: VesselNetwork, P: Prepared, cfg: SearchConfig):
        self.cfg = cfg
        self.P = P
        self.H, self.W = P.shape
        self.net = explode(net)
        self.priors = dict(MapConfig().priors(), calibre_window=cfg.calibre_window)
        self.rng = np.random.default_rng(cfg.seed)
        self.t0 = time.time()
        self._smp = {}
        self._cache = {}
        self.graveyard = {}             # deleted vessels, which a revive move may restore
        self._gsmp = {}
        self._gid = 0
        self._index = None
        self.n_evaluated = 0
        if cfg.init_iters > 0:
            self.global_fit(cfg.init_iters)
        else:
            self._render_all()
        self.tau = cfg.tau if cfg.tau > 0 else self.chi2()
        self.null = self.texture_null() if cfg.texture_null else dict(per_px=0.0)
        self.price = max(self.tau * cfg.lam_length, self.null["per_px"])

    # ------------------------------------------------------------ bookkeeping
    def log(self, *a):
        if self.cfg.verbose:
            print(f"[{time.time() - self.t0:6.0f}s]", *a, flush=True)

    def samples(self, eid):
        if eid not in self._smp:
            self._smp[eid] = edge_samples(self.net.edges[eid], 1.0)
        return self._smp[eid]

    def _global_model(self):
        net = self.net
        return NetworkModel(net, self.P.logI, self.P.weight, stride=1,
                            bg_spacing=net.bg_spacing or BG_SPACING, ref=net)

    @torch.no_grad()
    def _render_all(self, model=None):
        """Background image, halo and per-vessel patches of the current
        network (from a fitted global model when given)."""
        model = model or self._global_model()
        if self.net.background is None:
            self.net.background = model.bg.detach().numpy().copy()
            self.net.bg_spacing = model.bg_spacing
        self.B = model.background().numpy()
        hw, hs = model.halo()
        self.hw, self.hs = float(hw), float(hs)
        self.net.meta["optics"] = dict(halo_weight=self.hw, halo_sigma=self.hs)
        ent = model.vessel_entries().numpy()
        pix, eidx = model.e_pix.numpy(), model.e_edge.numpy()
        order = np.argsort(eidx, kind="stable")
        bounds = np.searchsorted(eidx[order], np.arange(len(model.eids) + 1))
        self.patch = {}
        for k, eid in enumerate(model.eids):
            sel = order[bounds[k]:bounds[k + 1]]
            self.patch[eid] = (pix[sel].copy(), ent[sel].astype(np.float32))
        self.V = np.zeros(self.H * self.W, np.float32)
        for p, v in self.patch.values():
            self.V[p] += v
        self._smp.clear()
        self._cache.clear()
        self._index = None

    def optical_density(self):
        V = self.V.reshape(self.H, self.W)
        return (1 - self.hw) * V + self.hw * _cv_blur(V, self.hs)

    def chi2(self):
        """Reduced chi-square of the current model's residual."""
        res = self.P.logI - (self.B - self.optical_density())
        w = self.P.weight
        m = w > 0
        return float(max(1.0, (w[m] * res[m] ** 2).mean()))

    def cost(self, e: Edge, L=None) -> float:
        L = edge_samples(e, 1.0)["L"] if L is None else L
        return self.tau * self.cfg.lam_vessel + self.price * L

    def texture_null(self):
        """What background texture alone buys a 'vessel', per px of
        centreline.  Texture makes bright ridges as often as dark ones, and
        vessels are only dark, so 'anti-vessels' fitted by the same renderer
        to the bright ridges of the map's residual (away from mapped
        vessels, whose misfit rims would bias it) show how much evidence a
        dark trace of texture gets.  A vessel explaining less per px than
        the null_quantile of the anti-vessels (length-weighted) is not told
        apart from texture: that is the price of a px of centreline."""
        import cv2
        from .fit import band_scales, seeds_to_edges
        from .ridges import detect
        cfg = self.cfg
        R = self.P.logI - (self.B - self.optical_density())
        foot = np.zeros(self.P.shape, np.uint8)
        for k in self.net.edges:
            s = self.samples(k)
            w = int(math.ceil(float(np.median(s["r"] + 2 * s["s"])) + 2))
            cv2.polylines(foot, [np.round(s["xy"] * 4).astype(np.int32)], False, 1,
                          thickness=2 * w + 1, shift=2)
        null = VesselNetwork(self.P.shape)
        null.meta = {"optics": dict(halo_weight=self.hw, halo_sigma=self.hs)}
        for band in cfg.null_bands:
            ml = max(6.0, 2.5 * band[0])
            seeds = detect(R, self.P.sigma, band_scales(band, 3), min_len=ml, valid=self.P.valid,
                           exclude=foot.astype(bool))
            seeds_to_edges(null, seeds, band)
        if not null.edges:
            return dict(n=0, per_px=0.0)
        gh = int(math.ceil((self.H - 1) / BG_SPACING)) + 1
        gw = int(math.ceil((self.W - 1) / BG_SPACING)) + 1
        null.background = np.zeros((gh, gw), np.float32)
        null.bg_spacing = BG_SPACING
        m = NetworkModel(null, -R, self.P.weight, stride=1, bg_spacing=BG_SPACING,
                         fit_background=False)
        m.raw_hw.requires_grad_(False)
        m.raw_hs.requires_grad_(False)
        self._fit(m, 60)
        gains, _ = m.edge_gains()
        m.write_back()
        L = np.array([max(null.length(k), 1.0) for k in m.eids])
        g = gains / L
        ok = gains > 0
        if not ok.any():
            return dict(n=0, per_px=0.0)
        o = np.argsort(g[ok])
        cw = np.cumsum(L[ok][o]) / L[ok].sum()
        q = float(g[ok][o][min(len(o) - 1, int(np.searchsorted(cw, cfg.null_quantile)))])
        out = dict(n=int(ok.sum()), length=round(float(L[ok].sum()), 1), per_px=q,
                   median_per_px=float(np.median(g[ok])))
        self.log(f"texture null: {out['n']} anti-vessels, {out['length']:.0f} px; evidence per px "
                 f"median {out['median_per_px']:.1f}, {cfg.null_quantile:.0%} {q:.1f}")
        return out

    def energy_total(self):
        """E of the whole network (data NLL + prior + structural cost)."""
        res = self.P.logI - (self.B - self.optical_density())
        nll = 0.5 * float((self.P.weight * res ** 2).sum())
        m = NetworkModel(self.net, self.P.logI, self.P.weight, stride=1,
                         bg_spacing=self.net.bg_spacing or BG_SPACING)
        with torch.no_grad():
            pr = float(m.prior(**self.priors))
        cst = sum(self.cost(e, self.samples(k)["L"]) for k, e in self.net.edges.items())
        return dict(total=nll + pr + cst, nll=nll, prior=pr, cost=cst)

    # ------------------------------------------------------------ local model
    def _window(self, smps, extra=4.0):
        xy = np.concatenate([s["xy"] for s in smps])
        reach = max(float((s["r"] + 3 * s["s"]).max()) for s in smps) + 1.5
        pad = reach + 3 * self.hs + extra
        x0 = int(max(0, math.floor(xy[:, 0].min() - pad)))
        y0 = int(max(0, math.floor(xy[:, 1].min() - pad)))
        x1 = int(min(self.W, math.ceil(xy[:, 0].max() + pad) + 1))
        y1 = int(min(self.H, math.ceil(xy[:, 1].max() + pad) + 1))
        return x0, y0, x1, y1

    def _local(self, edges, win, removed):
        """NetworkModel of `edges` alone on the window, fitted against the
        image minus the background and every vessel except `removed`."""
        x0, y0, x1, y1 = win
        pad = int(math.ceil(3 * self.hs)) + 2
        X0, Y0 = max(0, x0 - pad), max(0, y0 - pad)
        X1, Y1 = min(self.W, x1 + pad), min(self.H, y1 + pad)
        Vw = self.V.reshape(self.H, self.W)[Y0:Y1, X0:X1].copy()
        for eid in removed:
            p, v = self.patch[eid]
            ys, xs = np.divmod(p, self.W)
            m = (ys >= Y0) & (ys < Y1) & (xs >= X0) & (xs < X1)
            Vw[ys[m] - Y0, xs[m] - X0] -= v[m]
        OD = (1 - self.hw) * Vw + self.hw * _cv_blur(Vw, self.hs)
        OD = OD[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0]
        target = self.P.logI[y0:y1, x0:x1] - self.B[y0:y1, x0:x1] + OD
        sub = VesselNetwork((y1 - y0, x1 - x0))
        sub.meta = {"optics": dict(halo_weight=self.hw, halo_sigma=self.hs)}
        off = np.array([x0, y0], float)
        for e in edges:
            c = e.ctrl - off
            u, v = sub.add_node(*c[0]), sub.add_node(*c[-1])
            k = sub._eid
            sub._eid += 1
            sub.edges[k] = Edge(u, v, c.copy(), e.r.copy(), e.s.copy(), e.a.copy(), dict(e.info))
        gh = int(math.ceil((y1 - y0 - 1) / BG_SPACING)) + 1
        gw = int(math.ceil((x1 - x0 - 1) / BG_SPACING)) + 1
        sub.background = np.zeros((gh, gw), np.float32)
        sub.bg_spacing = BG_SPACING
        m = NetworkModel(sub, target, self.P.weight[y0:y1, x0:x1], stride=1,
                         bg_spacing=BG_SPACING, fit_background=False)
        m.raw_hw.requires_grad_(False)
        m.raw_hs.requires_grad_(False)
        return m

    def _local_energy(self, m):
        """(data NLL, prior) of a local model."""
        with torch.no_grad():
            loss, nll, _, _ = m.loss(priors=self.priors)
        return float(nll), float(loss - nll)

    def _fitted_edges(self, m, win):
        """The local model's edges back in image coordinates."""
        m.write_back()
        off = np.array([win[0], win[1]], float)
        out = []
        for eid in m.eids:
            e = m.net.edges[eid]
            out.append(Edge(-1, -1, e.ctrl + off, e.r.copy(), e.s.copy(), e.a.copy(),
                            dict(e.info)))
        return out

    def _patches(self, m, win):
        with torch.no_grad():
            ent = m.vessel_entries().numpy()
        py, px = np.divmod(m.e_pix.numpy(), m.W)
        gpix = (py + win[1]) * self.W + px + win[0]
        eidx = m.e_edge.numpy()
        return [(gpix[eidx == k], ent[eidx == k].astype(np.float32)) for k in range(len(m.eids))]

    # ------------------------------------------------------------ spatial index
    def _idx(self):
        """Per-state index of every vessel: bounding boxes, and a KD-tree of
        all centreline samples.  Rebuilt after every change."""
        if self._index is None:
            ids = list(self.net.edges)
            smp = [self.samples(k) for k in ids]
            bb = np.array([[s["xy"][:, 0].min(), s["xy"][:, 1].min(),
                            s["xy"][:, 0].max(), s["xy"][:, 1].max()] for s in smp]).reshape(-1, 4)
            X = np.concatenate([s["xy"] for s in smp]) if smp else np.zeros((0, 2))
            lab = np.concatenate([np.full(len(s["xy"]), k) for k, s in zip(ids, smp)]) \
                if smp else np.zeros(0, int)
            idx = np.concatenate([np.arange(len(s["xy"])) for s in smp]) if smp else np.zeros(0, int)
            self._index = dict(ids=np.array(ids, int), bb=bb, X=X, lab=lab, idx=idx,
                               tree=cKDTree(X) if len(X) else None)
        return self._index

    def _near(self, lo, hi, margin):
        """Vessels whose bounding box comes within margin of [lo, hi]."""
        I = self._idx()
        bb = I["bb"]
        sel = (bb[:, 2] >= lo[0] - margin) & (bb[:, 0] <= hi[0] + margin) & \
              (bb[:, 3] >= lo[1] - margin) & (bb[:, 1] <= hi[1] + margin)
        return I["ids"][sel]

    def _move_window(self, move):
        """The window a move is scored on (known before it is built)."""
        if move.get("focus") is not None:
            return self._focus_window(np.asarray(move["focus"], float), self._focus_reach(move))
        if move["anchor"]:
            return self._window([self.samples(k) for k in move["anchor"]])
        return self._window([move["smp"]])

    def _signature(self, win):
        """Fingerprint of the fixed optical density a move sees in its
        window: total, and its centroid."""
        x0, y0, x1, y1 = win
        V = self.V.reshape(self.H, self.W)[y0:y1, x0:x1].astype(np.float64)
        m = V.sum()
        if m <= 0:
            return (0.0, 0.0, 0.0)
        return (m, float(V.sum(0) @ np.arange(x0, x1)) / m, float(V.sum(1) @ np.arange(y0, y1)) / m)

    def _cached(self, move):
        """The cached score of a move if its context has not changed: the
        vessels it involves all exist and the fixed optical density in its
        window is the same (total within 0.2 %, centroid within 0.05 px).
        Joins far away replace vessels by new ids without changing what a
        move here sees, so their scores stay valid.  A hopeless move (e.g.
        deleting a strong vessel) keeps its score while its vessels exist:
        nothing nearby can bring it within reach."""
        c = self._cache.get(move["key"])
        if c is None:
            return None
        sig, win, r = c
        if r is not None and any(k not in self.net.edges for k in r[1][0]):
            return None
        if r is not None and r[0] > self.cfg.hopeless * self.tau * self.cfg.lam_vessel:
            return c            # far out of reach at any temperature the search uses
        now = self._signature(win)
        if abs(now[0] - sig[0]) > 2e-3 * max(sig[0], 1e-9) + 1e-6 or \
                abs(now[1] - sig[1]) > 0.05 or abs(now[2] - sig[2]) > 0.05:
            return None
        return c

    def _focus_reach(self, move):
        """Radius of a move's focus window: the re-fitted stretch plus the
        widest footprint among the vessels it touches."""
        rch = max(float((self.samples(k)["r"] + 3 * self.samples(k)["s"]).max())
                  for k in move["anchor"])
        return self.cfg.focus_radius + rch + 1.5

    # ------------------------------------------------------------ moves
    def evaluate(self, move):
        """dE of a move, with the new vessels fitted locally.  Returns
        (dE, proposal) with proposal = (old ids, new edges, patches of the
        new edges or None, window), or None when the move cannot be built.
        Cached (see _cached)."""
        c = self._cached(move)
        if c is not None:
            return c[2]
        win = self._move_window(move)
        out = self._compute(move)
        self._cache[move["key"]] = (self._signature(win), win, out)
        return out

    def _compute(self, move):
        built = move["build"]()
        if built is None:
            return None
        old, new0 = built
        focus = move.get("focus")
        if focus is None:
            smps = [self.samples(k) for k in old] + [edge_samples(e) for e in new0]
            win = self._window(smps)
        else:
            win = self._focus_window(focus, self._focus_reach(move))
        m_old = self._local([self.net.edges[k] for k in old], win, old)
        n0, p0 = self._local_energy(m_old)
        c0 = sum(self.cost(self.net.edges[k], self.samples(k)["L"]) for k in old)
        m_new = self._local(new0, win, old)
        if new0:
            free = None if focus is None else \
                self._free(m_new, np.asarray(focus, float) - [win[0], win[1]])
            self._fit(m_new, self.cfg.local_iters, free)
        new = self._fitted_edges(m_new, win) if new0 else []
        n1, p1 = self._local_energy(m_new)
        c1 = sum(self.cost(e) for e in new)
        self.last_parts = dict(nll=n1 - n0, prior=p1 - p0, cost=c1 - c0)
        patches = self._patches(m_new, win) if focus is None and new else None
        self.n_evaluated += 1
        return (n1 + p1 + c1 - n0 - p0 - c0, (old, new, patches, win))

    def _focus_window(self, focus, R):
        R = R + 3 * self.hs + 4.0
        x0 = int(max(0, math.floor(focus[0] - R)))
        y0 = int(max(0, math.floor(focus[1] - R)))
        x1 = int(min(self.W, math.ceil(focus[0] + R) + 1))
        y1 = int(min(self.H, math.ceil(focus[1] + R) + 1))
        return x0, y0, x1, y1

    @torch.no_grad()
    def _free(self, m, f):
        """Masks of the parameters of a local model within focus_radius of
        the point f (window coordinates): nodes, inner control points and
        profile knots.  The others are held fixed."""
        rad2 = self.cfg.focus_radius ** 2
        near = lambda P: ((P - torch.as_tensor(f, dtype=P.dtype)) ** 2).sum(1) <= rad2
        ctrl = m.ctrl_all().numpy()
        prof = []
        for k in range(len(m.eids)):
            n, mp = m.edge_nctrl[k], m.edge_nprof[k]
            c = ctrl[m.edge_ctrl_off[k]:m.edge_ctrl_off[k] + n]
            prof.append(sp.design(n, max(mp, 2))[:mp] @ c if mp > 1 else c[:1])
        prof = torch.as_tensor(np.concatenate(prof), dtype=torch.float32)
        return dict(node=near(m.node_xy), inner=near(m.inner), prof=near(prof))

    def _fit(self, m, iters, free=None):
        """Adam on a local model's vessels (positions and profiles; the
        halo and background stay fixed), only on `free` parameters."""
        cfg = self.cfg
        opt = torch.optim.Adam([dict(params=[m.node_xy, m.inner], lr=cfg.lr_pos),
                                dict(params=[m.raw_r, m.raw_s, m.raw_a], lr=cfg.lr_prof)])
        base = [g["lr"] for g in opt.param_groups]
        for it in range(iters):
            f = 0.5 * (1 + math.cos(math.pi * it / iters)) * 0.9 + 0.1
            for g, b in zip(opt.param_groups, base):
                g["lr"] = b * f
            opt.zero_grad(set_to_none=True)
            loss, _, _, _ = m.loss(priors=self.priors)
            loss.backward()
            if free is not None:
                m.node_xy.grad[~free["node"]] = 0
                m.inner.grad[~free["inner"]] = 0
                for q in (m.raw_r, m.raw_s, m.raw_a):
                    q.grad[~free["prof"]] = 0
            opt.step()
            m.project()
            if (it + 1) % 20 == 0 and it + 1 < iters:
                m.rebuild()
        m.rebuild()

    def apply(self, proposal):
        old, new, patches, win = proposal
        if not new:
            patches = []
        elif patches is None:   # fitted on a focus window: render the whole vessels
            smps = [edge_samples(e) for e in new]
            wfull = self._window(smps)
            patches = self._patches(self._local(new, wfull, old), wfull)
        for k in old:
            p, v = self.patch.pop(k)
            self.V[p] -= v
            self.net.remove_edge(k)
            self._smp.pop(k, None)
        ids = []
        for e, (p, v) in zip(new, patches):
            e = e.copy()
            u = self.net.add_node(*e.ctrl[0])
            w = self.net.add_node(*e.ctrl[-1])
            k = self.net._eid
            self.net._eid += 1
            e.u, e.v = u, w
            self.net.edges[k] = e
            self.net._touch()
            self.patch[k] = (p, v)
            self.V[p] += v
            ids.append(k)
        self._index = None
        return ids

    def _ends(self):
        """Every vessel end: (eid, end, position, outward tangent, width)."""
        out = []
        for k in self.net.edges:
            s = self.samples(k)
            for end in (0, 1):
                i = 0 if end == 0 else -1
                t = -_direction(s["xy"], 0, True) if end == 0 else \
                    _direction(s["xy"], len(s["xy"]) - 1, False)
                out.append((k, end, s["xy"][i], t, float(s["r"][i] + s["s"][i])))
        return out

    def _border(self, p, margin=3.0):
        return p[0] < margin or p[1] < margin or p[0] > self.W - 1 - margin or \
            p[1] > self.H - 1 - margin

    def join_moves(self):
        cfg = self.cfg
        ends = [e for e in self._ends() if not self._border(e[2])]
        if len(ends) < 2:
            return []
        P = np.array([e[2] for e in ends])
        tree = cKDTree(P)
        cone = math.cos(math.radians(cfg.join_cone_deg))
        moves, seen = [], set()
        for i, (k1, end1, p1, t1, w1) in enumerate(ends):
            G = min(cfg.join_gap, max(8.0, cfg.join_gap_factor * w1))
            for j in tree.query_ball_point(p1, G):
                k2, end2, p2, t2, w2 = ends[j]
                if k2 == k1:
                    continue
                pair = tuple(sorted([(k1, end1), (k2, end2)]))
                if pair in seen:
                    continue
                d = float(np.linalg.norm(p2 - p1))
                if np.dot(t1, t2) > 0.5:                  # both ends point the same way
                    continue
                close = d <= max(2.0, 0.5 * (w1 + w2))
                if not close:
                    v = p2 - p1
                    u = v / d
                    ahead = np.dot(t1, u) > cone and np.dot(t2, -u) > cone
                    overlap = np.dot(t1, t2) < -0.7 and d <= cfg.trim_max and \
                        abs(t1[0] * v[1] - t1[1] * v[0]) <= max(w1, w2) + 2.0
                    if not (ahead or overlap):
                        continue
                seen.add(pair)
                moves.append(dict(kind="join", anchor=[k1, k2], key=("join",) + pair,
                                  focus=0.5 * (p1 + p2),
                                  build=self._join_builder(k1, end1, k2, end2)))
        return moves

    def _join_builder(self, k1, end1, k2, end2):
        def build():
            e1, e2 = self.net.edges[k1], self.net.edges[k2]
            A = edge_samples(e1, 0.7)
            B = edge_samples(e2, 0.7)
            A = A if end1 == 1 else _flip(A)
            B = B if end2 == 0 else _flip(B)
            path = join_path(A, B, self.cfg.trim_max, self.cfg.max_turn_deg)
            if path is None:
                return None
            info = _merge_info([e1.info, e2.info])
            ctrl, r, s, a = fit_vessel(*path, info["spacing"])
            return [k1, k2], [Edge(-1, -1, ctrl, r, s, a, info)]
        return build

    def delete_moves(self):
        return [dict(kind="delete", anchor=[k], key=("delete", k),
                     build=self._delete_builder(k)) for k in self.net.edges]

    def _duplicates(self, k, frac=0.3, cos_par=0.8):
        """Vessels running inside and parallel to k along >= frac of k."""
        s = self.samples(k)
        xy, tan = s["xy"], s["tan"]
        out = []
        for j in self._near(xy.min(0), xy.max(0), 30.0):
            j = int(j)
            if j == k:
                continue
            o = self.samples(j)
            if "tree" not in o:
                o["tree"] = cKDTree(o["xy"])
            d, i = o["tree"].query(xy)
            inside = (d < o["r"][i] + 0.7 * o["s"][i] + 1.0) & \
                (np.abs((o["tan"][i] * tan).sum(1)) > cos_par)
            if inside.mean() >= frac:
                out.append(j)
        return out

    def _delete_builder(self, k):
        """Delete k; vessels duplicating it are re-fitted to take over."""
        def build():
            dup = self._duplicates(k)
            return [k] + dup, [self.net.edges[j].copy() for j in dup]
        return build

    def revive_moves(self):
        """Put back a vessel deleted earlier.  Deletions accepted at a high
        temperature would otherwise be final; with this move every move
        of the search can be undone."""
        out = []
        for g, e in self.graveyard.items():
            if g not in self._gsmp:
                self._gsmp[g] = edge_samples(e, 1.0)
            out.append(dict(kind="revive", anchor=[], key=("revive", g), grave=g,
                            smp=self._gsmp[g], build=(lambda e=e: ([], [e.copy()]))))
        return out

    def split_moves(self):
        """Split a vessel at a sharp bend (the path may switch vessels there)."""
        cfg = self.cfg
        moves = []
        cmax = math.cos(math.radians(cfg.split_turn_deg))
        look = 5
        for k in self.net.edges:
            s = self.samples(k)
            xy = s["xy"]
            n = len(xy)
            if n < 4 * look:
                continue
            a = xy[look:n - look] - xy[:n - 2 * look]
            b = xy[2 * look:] - xy[look:n - look]
            c = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
            idx = np.arange(look, n - look)
            cand = idx[c < cmax]
            if not len(cand):
                continue
            # one candidate per bend: the sharpest point of each run
            for run in np.split(cand, np.flatnonzero(np.diff(cand) > 1) + 1):
                i = int(run[np.argmin(c[run - look])])
                if s["s_arc"][i] < 6 or s["L"] - s["s_arc"][i] < 6:
                    continue
                moves.append(dict(kind="split", anchor=[k], key=("split", k, i),
                                  focus=xy[i], build=self._split_builder(k, s["s_arc"][i])))
        return moves

    def _split_builder(self, k, s_cut):
        def build():
            e = self.net.edges[k]
            s = edge_samples(e, 0.7)
            i = int(np.searchsorted(s["s_arc"], s_cut))
            out = []
            for sl in (slice(0, i + 1), slice(i, None)):
                if len(s["xy"][sl]) < 4:
                    return None
                ctrl, r, s_, a = _fit_edge_params(s["xy"][sl], s["r"][sl], s["s"][sl],
                                                  s["a"][sl], e.info.get("spacing", 12.0),
                                                  faithful=True)
                out.append(Edge(-1, -1, ctrl, r, s_, a, dict(e.info)))
            return [k], out
        return build

    def reroute_moves(self):
        """A vessel end E that touches vessel V in its interior: E continues
        into one part of V, the other part of V ends at the junction.  This
        changes which arm of a branch point continues the parent."""
        moves = []
        I = self._idx()
        if I["tree"] is None or len(I["ids"]) < 2:
            return moves
        X, lab, idx = I["X"], I["lab"], I["idx"]
        for (k, end, p, t, w) in self._ends():
            if self._border(p):
                continue
            near = {}
            for c in I["tree"].query_ball_point(p, 25.0):
                j = int(lab[c])
                if j == k:
                    continue
                d = float(np.linalg.norm(X[c] - p))
                if j not in near or d < near[j][0]:
                    near[j] = (d, int(idx[c]))
            for j, (d, i) in near.items():
                o = self.samples(j)
                if d > o["r"][i] + o["s"][i] + 2.0:
                    continue
                if o["s_arc"][i] < 8 or o["L"] - o["s_arc"][i] < 8:
                    continue                            # near V's end: a join, not a reroute
                for part in (0, 1):                     # E continues into V's part 0 or 1
                    moves.append(dict(kind="reroute", anchor=[k, j],
                                      key=("reroute", k, end, j, i, part), focus=p,
                                      build=self._reroute_builder(k, end, j, o["s_arc"][i],
                                                                  part)))
        return moves

    def _reroute_builder(self, k, end, j, s_cut, part):
        def build():
            e, v = self.net.edges[k], self.net.edges[j]
            s = edge_samples(v, 0.7)
            i = int(np.searchsorted(s["s_arc"], s_cut))
            if i < 4 or i > len(s["xy"]) - 5:
                return None
            cut = lambda sl: {kk: (vv[sl] if isinstance(vv, np.ndarray) else vv)
                              for kk, vv in s.items()}
            pieces = {0: cut(slice(0, i + 1)), 1: cut(slice(i, None))}
            pieces[1]["s_arc"] = pieces[1]["s_arc"] - pieces[1]["s_arc"][0]
            cont, stay = pieces[part], pieces[1 - part]
            if part == 0:                                # continue into V backwards
                cont = _flip(dict(cont, L=float(cont["s_arc"][-1])))
            A = edge_samples(e, 0.7)
            A = A if end == 1 else _flip(A)
            path = join_path(A, cont, self.cfg.trim_max, self.cfg.max_turn_deg)
            if path is None:
                return None
            info = _merge_info([e.info, v.info])
            ctrl, r, s_, a = fit_vessel(*path, info["spacing"])
            c2, r2, s2, a2 = _fit_edge_params(stay["xy"], stay["r"], stay["s"], stay["a"],
                                              v.info.get("spacing", 12.0), faithful=True)
            return [k, j], [Edge(-1, -1, ctrl, r, s_, a, info),
                            Edge(-1, -1, c2, r2, s2, a2, dict(v.info))]
        return build

    def all_moves(self, kinds=("join", "delete", "split", "reroute", "revive")):
        gen = dict(join=self.join_moves, delete=self.delete_moves, split=self.split_moves,
                   reroute=self.reroute_moves, revive=self.revive_moves)
        return [m for k in kinds for m in gen[k]()]

    # ------------------------------------------------------------ driver
    def global_fit(self, iters):
        if not self.net.edges:
            self._render_all()
            return
        m = self._global_model()
        optimize(m, iters, self.cfg.lr_pos * 0.5, self.cfg.lr_prof, MapConfig().lr_bg,
                 rebuild_every=25, priors=self.priors, track=MapConfig().anchor())
        m.write_back()
        self._render_all(m)

    def _evaluate_all(self, moves):
        """Score every move whose cached score is stale, in worker processes
        when there are enough of them.  The workers are forked from the
        current state, so they see it without copying; only the results
        come back."""
        todo = [i for i, m in enumerate(moves) if self._cached(m) is None]
        nw = self.cfg.workers or os.cpu_count() or 1
        if nw <= 1 or len(todo) < 2 * nw:
            for i in todo:
                self.evaluate(moves[i])
            return
        global _SHARED
        _SHARED = (self, moves)
        chunks = [todo[j::nw] for j in range(nw)]
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(nw) as pool:
            for part in pool.map(_evaluate_chunk, chunks):
                for i, r in part:
                    win = self._move_window(moves[i])
                    self._cache[moves[i]["key"]] = (self._signature(win), win, r)
        _SHARED = None
        self.n_evaluated += len(todo)

    def anneal(self, T0, steps_hot, kinds=("join", "delete", "split", "reroute", "revive"),
               max_steps=None):
        """Rejection-free annealing.  At every step all candidate moves are
        scored (cached: only moves near the last change are re-fitted) and
        one is drawn with probability proportional to exp(-dE / T), the
        current state ("stay", dE = 0) included.  T falls geometrically from
        T0 to T0 / 100 over steps_hot steps and is then 0: from there on the
        best improving move is taken until none is left."""
        decay = 0.01 ** (1.0 / max(1, steps_hot))
        T = T0
        max_steps = max_steps or (3 * steps_hot + 100)
        acc, stays = {}, 0
        for step in range(max_steps):
            moves = self.all_moves(kinds)
            self._evaluate_all(moves)
            res = []
            for m in moves:
                r = self._cache[m["key"]][2]
                if r is not None:
                    res.append((r[0], m, r[1]))
            live = {m["key"] for m in moves}
            self._cache = {k: v for k, v in self._cache.items() if k in live}
            if not res:
                break
            dE = np.array([r[0] for r in res])
            if T <= 0:
                i = int(np.argmin(dE))
                if dE[i] >= -1e-3:
                    break
            else:
                logit = np.r_[-dE, 0.0] / T
                p = np.exp(logit - logit.max())
                i = int(self.rng.choice(len(p), p=p / p.sum()))
                T = T * decay if step < steps_hot else 0.0
                if i == len(res):
                    stays += 1
                    continue
            d, m, prop = res[i]
            if m["kind"] == "delete":
                self.graveyard[self._gid] = self.net.edges[m["anchor"][0]].copy()
                self._gid += 1
            elif m["kind"] == "revive":
                del self.graveyard[m["grave"]]
            ids = self.apply(prop)
            if m["kind"] in ("join", "reroute") and ids:
                e = self.net.edges[ids[0]]
                e.info["links"] = list(e.info.get("links", [])) + [dict(
                    kind=m["kind"], evidence="energy", dE=round(float(d), 1),
                    xy=[round(float(v), 1) for v in m["focus"]])]
            acc[m["kind"]] = acc.get(m["kind"], 0) + 1
            if self.cfg.verbose and (step % 25 == 0):
                self.log(f"step {step}: T = {T:.1f}, {len(res)} moves, took {m['kind']} "
                         f"(dE = {d:.0f}); {len(self.net.edges)} vessels; "
                         f"{self.n_evaluated} local fits")
        return dict(steps=step + 1, accepted=acc, stays=stays)

    def run(self):
        cfg = self.cfg
        n0 = len(self.net.edges)
        E0 = self.energy_total()
        self.log(f"search: {n0} vessels, tau = {self.tau:.2f}, price {self.price:.1f} per px, "
                 f"E = {E0['total']:.0f} "
                 f"(nll {E0['nll']:.0f}, prior {E0['prior']:.0f}, cost {E0['cost']:.0f})")
        T0 = cfg.t_start * self.tau * cfg.lam_vessel
        hist = [self.anneal(T0, max(10, int(cfg.hot_steps * n0)))]
        self.log(f"annealed: {hist[-1]}; {len(self.net.edges)} vessels")
        if cfg.global_iters > 0:
            self.global_fit(cfg.global_iters)
            hist.append(self.anneal(0.0, 0))
            self.log(f"after the joint fit: {hist[-1]}; {len(self.net.edges)} vessels")
        to_through(self.net, cfg.attach_tol)
        self.net.snap_through_nodes()
        self.global_fit(cfg.through_iters)
        self.net.orient_structural()
        E1 = self.energy_total()
        m = NetworkModel(self.net, self.P.logI, self.P.weight, stride=1,
                         bg_spacing=self.net.bg_spacing or BG_SPACING)
        gains, _ = m.edge_gains()
        for k, eid in enumerate(m.eids):
            L = max(self.samples(eid)["L"], 1.0)
            self.net.edges[eid].info.update(gain=float(gains[k]), gain_per_px=float(gains[k] / L))
        self.net.meta["search"] = dict(
            config={k: v for k, v in asdict(cfg).items()}, tau=self.tau,
            texture_null=self.null, price_per_px=self.price,
            vessels_before=n0, vessels_after=len(self.net.edges),
            energy_before=E0, energy_after=E1, steps=hist, local_fits=self.n_evaluated,
            seconds=round(time.time() - self.t0, 1))
        self.net.meta["final_nll"] = E1["nll"]
        self.log(f"done: {n0} -> {len(self.net.edges)} vessels; E {E0['total']:.0f} -> "
                 f"{E1['total']:.0f} (nll {E0['nll']:.0f} -> {E1['nll']:.0f}); "
                 f"{self.n_evaluated} local fits")
        return self.net


_SHARED = None


def _evaluate_chunk(idx):
    """Worker: score some of the moves of the forked search state."""
    torch.set_num_threads(1)
    C, moves = _SHARED
    return [(i, C._compute(moves[i])) for i in idx]


def to_through(net: VesselNetwork, tol=1.5, end_tol=3.0):
    """Branch points in the map's representation (network.py).  A vessel end
    lying on another vessel (within that vessel's r + s + tol) becomes a
    node the other vessel passes through (``info["through"]``), moved onto
    its centreline; near the other vessel's own end, the two ends share one
    node instead.  Closest contacts are settled first."""
    smp = {k: net.sample(k, 1.0) for k in net.edges}
    if len(smp) < 2:
        return
    ids = list(smp)
    X = np.concatenate([smp[k]["xy"] for k in ids])
    lab = np.concatenate([np.full(len(smp[k]["xy"]), k) for k in ids])
    idx = np.concatenate([np.arange(len(smp[k]["xy"])) for k in ids])
    tree = cKDTree(X)
    contacts = []
    for k, e in net.edges.items():
        for nid in (e.u, e.v):
            p = net.nodes[nid].xy
            if net.node_kind(nid, 1) == "border":
                continue
            best = None
            for c in tree.query_ball_point(p, 30.0):
                j, i = int(lab[c]), int(idx[c])
                if j == k:
                    continue
                d = float(np.linalg.norm(X[c] - p))
                if d <= smp[j]["r"][i] + smp[j]["s"][i] + tol and (best is None or d < best[0]):
                    best = (d, j, i)
            if best is not None:
                contacts.append((best[0], k, nid, best[1], best[2]))
    for _, k, nid, j, i in sorted(contacts):
        if k not in net.edges or j not in net.edges or nid not in net.nodes:
            continue
        e, ej = net.edges[k], net.edges[j]
        if nid not in (e.u, e.v):
            continue
        s = smp[j]
        L, sa = float(s["s_arc"][-1]), float(s["s_arc"][i])
        if min(sa, L - sa) < end_tol:                  # end meets end: one shared node
            keep = ej.u if sa < 0.5 * L else ej.v
            if keep != nid:
                net.merge_nodes(keep, nid)
            continue
        p = s["xy"][i]
        near = [t for t in ej.through if t in net.nodes and
                np.linalg.norm(net.nodes[t].xy - p) < 3.0]
        if near:                                        # two branches leave at one point
            if near[0] != nid:
                net.merge_nodes(near[0], nid)
            continue
        if nid in ej.through or nid in (ej.u, ej.v):
            continue
        net.nodes[nid].x, net.nodes[nid].y = float(p[0]), float(p[1])
        for kk, _ in net.incident(nid):
            net.sync_ends(kk)
        ej.info["through"] = list(ej.through) + [nid]
        net._touch()


def search_map(intensity: np.ndarray, net: VesselNetwork, cfg: SearchConfig | None = None,
               prepared: Prepared | None = None) -> VesselNetwork:
    """The fewest vessels (one spline each) that render the image as well as
    `net` does, by annealing over joins, deletions, splits and reroutes
    under the vessel-count energy (see the module docstring).  `net` is any
    map of the same image: segments from build_map / refine_map, or better
    the output of consolidate_map, which the search then continues.  It is
    not modified."""
    cfg = cfg or SearchConfig()
    P = prepared or prepare(intensity)
    return VesselSearch(net, P, cfg).run()
