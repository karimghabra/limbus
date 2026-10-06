"""vesselmap glue for vesselnet: images in vesselmap's domain, the truth oracle, E with a fixed scale.

E is vesselmap's search energy (search.py):

    E(G) = NLL(I | render(G)) + prior(G) + sum_v (tau * lam_vessel + price * L_v) + Phi(G)

tau (the data temperature) and price (per px of centreline, from the texture null) are properties of the map
the search starts from.  To compare two graphs of one image, both must be scored with the same tau and price:
`energy` takes them as arguments, and `oracle` provides them (the truth's).

    I = to_unit(still)                       # DN (NaN = no data) -> [0, 1], as vesselmap.image.load_image
    P = vesselmap.image.prepare(I)
    net, info = oracle(truth_net, P)         # profiles, halo and background fitted; centrelines held fixed
    E = energy(other_net, P, info["tau"], info["price"])
"""
from __future__ import annotations

import time

import numpy as np
import torch

from vesselmap.fit import MapConfig, optimize
from vesselmap.image import FULL_SCALE_12BIT, Prepared, prepare
from vesselmap.network import VesselNetwork
from vesselmap.render import NetworkModel
from vesselmap.search import BG_SPACING, SearchConfig, VesselSearch


def energy_config(**kw) -> SearchConfig:
    """vesselnet's energy (decided in iteration 0 from the energy check): vesselscene's strict convention
    (strict_forks: three ends at a branch point are a fork, an end on another vessel's interior pays a
    vessel) and the floor price tau * lam_length per px (no texture null: the null's price deleted 16 of 20
    faint observable vessels; texture is told from vessels by a learned prior instead, PLAN §4D)."""
    return SearchConfig(**{**dict(strict_forks=True, texture_null=False), **kw})


def to_unit(img: np.ndarray) -> np.ndarray:
    """A still in sensor DN (float, NaN = no data) as vesselmap's [0, 1] intensities (load_image's rule)."""
    a = np.asarray(img, np.float32)
    mx = float(np.nanmax(a))
    scale = (FULL_SCALE_12BIT if mx <= FULL_SCALE_12BIT else mx) if mx > 1.0 else 1.0
    return np.clip(a / scale, 0.0, 1.0)          # NaN stays NaN: prepare() marks it invalid


def search_priors(cfg: SearchConfig | None = None) -> dict:
    cfg = cfg or SearchConfig()
    return dict(MapConfig().priors(), calibre_window=cfg.calibre_window)


def model_of(net: VesselNetwork, P: Prepared, fit_background=True) -> NetworkModel:
    return NetworkModel(net, P.logI, P.weight, stride=1, bg_spacing=net.bg_spacing or BG_SPACING, inside=True,
                        fit_background=fit_background)


def fit(net: VesselNetwork, P: Prepared, iters: int, fit_pos=True, fit_prof=True, fit_bg=True,
        cfg: SearchConfig | None = None, lr_pos=None, lr_prof=None, lr_bg=None) -> dict:
    """Joint gradient fit of `net` (in place) to P; returns timing and the last loss / NLL."""
    mc = MapConfig()
    t0 = time.perf_counter()
    m = model_of(net, P)
    hist = optimize(m, iters, lr_pos or mc.lr_pos, lr_prof or mc.lr_prof, lr_bg or mc.lr_bg,
                    rebuild_every=25, priors=search_priors(cfg), fit_pos=fit_pos, fit_prof=fit_prof,
                    fit_bg=fit_bg)
    m.write_back()
    return dict(seconds=round(time.perf_counter() - t0, 2), iters=iters,
                loss=hist[-1][0] if hist else None, nll=hist[-1][1] if hist else None)


def fit_local(net: VesselNetwork, P: Prepared, pts, radius: float, iters: int, S: VesselSearch) -> dict:
    """Fit only what lies within `radius` px of any of `pts` (control points and profile knots), with the halo,
    the background and everything else fixed, as the search fits a move (VesselSearch._free_near / _fit;
    S supplies its priors and learning rates).  In place; returns the time.  A global refit would let a
    change at one point move the whole map: its drift elsewhere (up to hundreds of thousands of nats on a
    512 px window) swamps the change being scored."""
    t0 = time.perf_counter()
    m = model_of(net, P, fit_background=False)
    m.raw_hw.requires_grad_(False)
    m.raw_hs.requires_grad_(False)
    free = S._free_near(m, np.asarray(pts, float).reshape(-1, 2), radius)
    S._fit(m, iters, free)
    m.write_back()
    return dict(seconds=round(time.perf_counter() - t0, 2), iters=iters,
                free=int(sum(int(v.sum()) for v in free.values())))


def _search(net: VesselNetwork, P: Prepared, tau: float | None, price: float | None,
            cfg: SearchConfig | None = None) -> VesselSearch:
    """A VesselSearch on `net` with tau and price fixed (None: computed from this map, as search_map does)."""
    cfg = SearchConfig(**{**(cfg.__dict__ if cfg else {}), "verbose": False})
    if tau is not None:
        cfg.tau = float(tau)
    if price is not None:
        cfg.texture_null = False                 # the null is replaced by the given price below
    S = VesselSearch(net, P, cfg)                # (the texture null fits anti-vessels: gradients needed)
    if price is not None:
        S.null, S.price = dict(per_px=float(price), fixed=True), float(price)
    return S


def energy(net: VesselNetwork, P: Prepared, tau: float | None = None, price: float | None = None,
           cfg: SearchConfig | None = None) -> dict:
    """E of `net` as the search scores it (its current parameters, no fitting), with tau and price fixed."""
    t0 = time.perf_counter()
    S = _search(net, P, tau, price, cfg)
    E = S.energy_total()
    w = P.weight
    res = P.logI - (S.B - S.optical_density())
    m = w > 0
    E.update(tau=S.tau, price=S.price, n_vessels=len(S.net.edges),
             length=float(sum(S.samples(k)["L"] for k in S.net.edges)),
             chi2_reduced=float((w[m] * res[m] ** 2).mean()), seconds=round(time.perf_counter() - t0, 2))
    return E


def clip_to_frame(net: VesselNetwork, spacing: float = 0.5, min_len: float = 4.0) -> VesselNetwork:
    """The part of every edge inside the image box [0, W-1] x [0, H-1], as vesselmap models an image.

    vesselmap's full-image models keep control points inside the box (NetworkModel inside=True), so a truth
    vessel running on beyond the frame (vesselscene's map.json keeps it whole) would be squashed onto the
    border.  Edges wholly inside are kept as they are; an edge that leaves the frame is replaced by its inside
    runs (at least min_len px), refitted faithfully from samples; an end inside the frame keeps its node, so
    forks stay connected.  A vessel leaving and re-entering becomes two edges with the same info.  Each
    replaced edge's info gets clipped_from (its old id) and clip_err_px (largest distance of the refit
    centreline from the samples)."""
    H, W = net.shape
    out = net.copy()
    for eid in list(out.edges):
        smp = out.sample(eid, spacing)
        xy = smp["xy"]
        ins = (xy[:, 0] >= 0) & (xy[:, 0] <= W - 1) & (xy[:, 1] >= 0) & (xy[:, 1] <= H - 1)
        if ins.all():
            continue
        e = out.edges[eid]
        out.remove_edge(eid, drop_orphans=False)
        edges = np.flatnonzero(np.diff(np.r_[0, ins.astype(np.int8), 0]))
        for a, b in zip(edges[::2], edges[1::2]):              # inside runs [a, b)
            if (b - a - 1) * spacing < min_len:
                continue
            sl = slice(a, b)
            u = e.u if a == 0 else None
            v = e.v if b == len(xy) else None
            k = out.add_edge_dense(xy[sl], smp["r"][sl], smp["s"][sl], smp["a"][sl], u=u, v=v,
                                   info=dict(e.info, clipped_from=int(eid)), faithful=True)
            q = out.sample(k, spacing)["xy"]
            from scipy.spatial import cKDTree
            out.edges[k].info["clip_err_px"] = float(cKDTree(q).query(xy[sl])[0].max())
    for nid in [n for n in out.nodes if not out.incident(n) and not out.passing(n)]:
        del out.nodes[nid]
    out._touch()
    return out


def oracle(truth: VesselNetwork, P: Prepared, iters: int = 300, cfg: SearchConfig | None = None):
    """The truth oracle (PLAN.md §5, iteration 0): the truth clipped to the frame (clip_to_frame), its
    centrelines held fixed, its profiles, halo and background fitted by vesselmap.  Returns (fitted copy,
    info) with E, tau and price (the texture null of the oracle's own residual), the scale every other graph
    of this image is scored on."""
    net = clip_to_frame(truth)
    f = fit(net, P, iters, fit_pos=False, cfg=cfg)
    S = _search(net, P, None, None, cfg)
    info = dict(fit=f, tau=S.tau, price=S.price, texture_null=S.null)
    info["E"] = energy(net, P, S.tau, S.price, cfg)
    return net, info
