"""Two or more splines that meet: contact, crossings at different depths,
bifurcations, a branch leaving a through vessel, and a sharp joint.

Physics.  Separate vessels (touching, or crossing at another depth) add in
optical density.  Where lumens join (a node), the blood is the union of the
tubes; for tubes whose axes lie at the same depth the vertical chord through
the union is the MAX of the individual chords, so the sharp density is
max_e f_e, not sum_e f_e.

Renderers compared against the exact image:
  current    vesselmap's NetworkModel (sum over edges, flat blurred end caps)
  sum        the closed-form line integral per edge, summed over edges
  sum+nodes  the same, but within a small radius of every node the incident
             tubes are replaced by the union of their lumens, drawn as
             polygons per density level and blurred in closed form:
             the 2-D Gaussian mass of a polygon is a sum over its edges of
             atan(t)/2pi - T(h, t)  (T = Owen's T function).
"""
import os
import sys

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from scipy.special import owens_t
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import closed_form_check as cf          # noqa: E402
import render_figures as rf             # noqa: E402
from vesselmap import spline as sp      # noqa: E402
from vesselmap.network import VesselNetwork   # noqa: E402
from vesselmap.render import NetworkModel, _C, _W   # noqa: E402

A0 = 0.12                                        # OD per px of radius: same blood everywhere
S_CUM = np.r_[np.cumsum(_W[::-1])[::-1], 0.0]    # S_k = sum_{j>=k} W_j, k = 0..6


# ---- the 2-D Gaussian mass of a polygon, in closed form -------------------
def _g(h, t):
    return np.arctan(t) / (2 * np.pi) - owens_t(h, t)


def ring_mass(P, ring, s):
    """Signed Gaussian mass (std s) of the polygon `ring` (m x 2, not closed)
    around each point of P: positive for a counter-clockwise ring.  Sum over
    edges AB of sign(A x B) * [g(h, t_B) - g(h, t_A)], with h the distance
    from the point to the line AB and t the positions of A, B along it."""
    A = (ring[None] - P[:, None]) / s
    B = np.roll(A, -1, axis=1)
    D = B - A
    u = D / np.maximum(np.linalg.norm(D, axis=-1), 1e-12)[..., None]
    cross = A[..., 0] * B[..., 1] - A[..., 1] * B[..., 0]
    h = np.maximum(np.abs(A[..., 0] * u[..., 1] - A[..., 1] * u[..., 0]), 1e-12)
    tA, tB = (A * u).sum(-1) / h, (B * u).sum(-1) / h
    return (np.sign(cross) * (_g(h, tB) - _g(h, tA))).sum(1)


def geom_mass(P, geom, s):
    out = np.zeros(len(P))
    for pg in getattr(geom, "geoms", [geom]):
        if pg.is_empty:
            continue
        ring = np.asarray(pg.exterior.coords)[:-1]
        out += (1 if pg.exterior.is_ccw else -1) * ring_mass(P, ring, s)
        for hole in pg.interiors:
            hr = np.asarray(hole.coords)[:-1]
            out -= (1 if hole.is_ccw else -1) * ring_mass(P, hr, s)
    return out


# ---- scenes -------------------------------------------------------------
def ray(p0, deg, length, curv=0.0, step=0.05):
    n = int(length / step)
    h = np.deg2rad(deg) + curv * np.arange(n) * step
    d = step * np.stack([np.cos(h), np.sin(h)], 1)
    return np.vstack([p0, np.asarray(p0, float) + np.cumsum(d, 0)])


def E(xy, r, s, u=None, v=None):
    return dict(xy=np.asarray(xy, float), r=r, s=s, a=A0 * r, u=u, v=v)


def scenes():
    out = []
    # 1. two separate vessels touching side by side
    phi = np.linspace(-1.5, 1.5, 6000)
    arc = np.stack([32 + 40 * np.sin(phi), -9.5 + 40 * np.cos(phi)], 1)
    out.append(dict(name="Two vessels touching", mode="add",
                    edges=[E(ray((-40, 34), 0, 150), 2.0, 1.5), E(arc, 1.5, 1.5)],
                    prof=(32, 18, 46)))
    # 2. crossing at different depths (deep and blurred / shallow and sharp)
    d60 = np.array([np.cos(np.deg2rad(60)), np.sin(np.deg2rad(60))])
    out.append(dict(name="Crossing at another depth", mode="add",
                    edges=[E(ray((-40, 30), 3, 150), 3.0, 4.0),
                           E(ray(np.array([32, 34]) - 80 * d60, 60, 160), 1.0, 1.0)],
                    prof=(26, 12, 50)))
    # 3. bifurcation: three edges end at one node
    N = np.array([32.0, 30.0])
    out.append(dict(name="Bifurcation", mode="union", nodes=dict(N=N),
                    edges=[E(ray(N, 90, 90, 0.002), 2.0, 1.5, u="N"),
                           E(ray(N, 235, 90, -0.004), 1.6, 1.5, u="N"),
                           E(ray(N, 305, 90, 0.004), 1.4, 1.5, u="N")],
                    junctions=[dict(node="N", inc=[(0, "start"), (1, "start"), (2, "start")])],
                    prof=(32, 12, 46)))
    # 4. a branch leaving a vessel that runs through the node (consolidated map)
    x = np.arange(-40, 104, 0.05)
    parent = np.stack([x, 36 + 0.003 * (x - 32) ** 2], 1)
    N = np.array([28.0, 36 + 0.003 * 16])
    out.append(dict(name="Branch off a through vessel", mode="union", nodes=dict(N=N),
                    edges=[E(parent, 2.5, 1.5), E(ray(N, 305, 80), 1.5, 1.5, u="N")],
                    junctions=[dict(node="N", inc=[(0, "through"), (1, "start")])],
                    prof=(31, 22, 48)))
    # 5. a sharp joint: two edges meet at 70 degrees (a 'joint' node)
    N = np.array([34.0, 34.0])
    out.append(dict(name="Sharp joint (70° turn)", mode="union", nodes=dict(N=N),
                    edges=[E(ray(N, 180, 90), 1.8, 1.5, u="N"), E(ray(N, 290, 90), 1.8, 1.5, u="N")],
                    junctions=[dict(node="N", inc=[(0, "start"), (1, "start")])],
                    prof=(34, 20, 48)))
    return out


# ---- build with vesselmap -------------------------------------------------
def build(sc, H=64, W=64):
    smax = max(e["s"] for e in sc["edges"])
    M = int(np.ceil(4 * np.hypot(smax, 5.0))) + 2
    Hc, Wc = H + 2 * M, W + 2 * M
    net = VesselNetwork((Hc, Wc))
    nid = {k: net.add_node(*(np.asarray(p) + M)) for k, p in sc.get("nodes", {}).items()}
    eids = []
    for e in sc["edges"]:
        n = len(e["xy"])
        eids.append(net.add_edge_dense(e["xy"] + M, np.full(n, e["r"]), np.full(n, e["s"]),
                                       np.full(n, e["a"]), u=nid.get(e["u"]), v=nid.get(e["v"])))
    model = NetworkModel(net, np.zeros((Hc, Wc), np.float32), np.ones((Hc, Wc), np.float32))
    with torch.no_grad():
        od = model.optical_density().numpy()
        hw, hs = (float(v) for v in model.halo())
    prm = []
    for eid in eids:
        e = net.edges[eid]
        prm.append(dict(ctrl=e.ctrl, r=float(np.median(e.r)), s=float(np.median(e.s)),
                        a=float(np.median(e.a)), L=net.length(eid)))
    return dict(net=net, eids=eids, M=M, Hc=Hc, Wc=Wc, od_current=od, hw=hw, hs=hs, prm=prm,
                node_xy={k: np.asarray(p) + M for k, p in sc.get("nodes", {}).items()})


# ---- the exact image -----------------------------------------------------
def exact(sc, b, ss=8):
    Hc, Wc = b["Hc"], b["Wc"]
    gx, gy = np.meshgrid(np.arange(Wc * ss) / ss, np.arange(Hc * ss) / ss)
    P = np.stack([gx.ravel(), gy.ravel()], 1)
    fields = []
    for e, p in zip(sc["edges"], b["prm"]):
        cl = sp.design(len(p["ctrl"]), int(p["L"] / 0.02) + 2) @ p["ctrl"]
        d, j = cKDTree(cl).query(P, distance_upper_bound=p["r"] + 1)
        ok = np.isfinite(d)
        if e["u"] is None:              # free ends are cut flat; ends at a node are round
            ok &= j != 0
        if e["v"] is None:
            ok &= j != len(cl) - 1
        f = np.zeros(len(P))
        f[ok] = p["a"] * rf.box_sharp(d[ok], p["r"])
        fields.append(f.reshape(gx.shape))
    hw, hs = b["hw"], b["hs"]
    if sc["mode"] == "union":
        s = b["prm"][0]["s"]
        sharp = np.max(fields, 0)
        od = (1 - hw) * gaussian_filter(sharp, s * ss, mode="constant", truncate=4) + \
            hw * gaussian_filter(sharp, np.hypot(s, hs) * ss, mode="constant", truncate=4)
    else:
        od = sum((1 - hw) * gaussian_filter(f, p["s"] * ss, mode="constant", truncate=4) +
                 hw * gaussian_filter(f, np.hypot(p["s"], hs) * ss, mode="constant", truncate=4)
                 for f, p in zip(fields, b["prm"]))
    return od[::ss, ::ss], od


# ---- closed forms --------------------------------------------------------
def edge_frame(ctrl):
    n = len(ctrl)
    ud = np.linspace(0, 1, max(4000, 40 * n))
    sd = sp.arclength(sp.design_at(n, ud) @ ctrl)

    def at(s):
        u = np.interp(np.atleast_1d(s), sd, ud)
        C = sp.design_at(n, u) @ ctrl
        D = sp.design_at(n, u, 1) @ ctrl
        T = D / np.linalg.norm(D, axis=1, keepdims=True)
        return C, np.stack([-T[:, 1], T[:, 0]], 1), T
    return at, sd[-1]


def piece_bounds(pc):
    w = pc["hi"] - pc["lo"]
    hi = np.cumsum(w)
    return hi - w, hi, hi - w / 2                    # absolute lo, hi, mid


def closed_form(sc, b, node_union=True):
    Hc, Wc = b["Hc"], b["Wc"]
    gy, gx = np.mgrid[0:Hc, 0:Wc]
    P = np.stack([gx.ravel(), gy.ravel()], 1).astype(float)
    pcs = [cf.pieces(p["ctrl"]) for p in b["prm"]]
    keep = [np.ones(len(pc["C"]), bool) for pc in pcs]
    V = np.zeros(len(P))
    if node_union:
        for jn in sc.get("junctions", []):
            V += node_region(jn, sc, b, pcs, keep, P)
    for pc, p, k in zip(pcs, b["prm"], keep):
        sub = {key: (val[k] if isinstance(val, np.ndarray) else val) for key, val in pc.items()}
        V += cf.render_sum(P, sub, p["r"], p["s"], p["a"])[0]
    return cf.with_halo(V.reshape(Hc, Wc), b["hw"], b["hs"])


def node_region(jn, sc, b, pcs, keep, P):
    """Union of the incident tubes within R of the node, as polygons per
    density level, each blurred in closed form (ring_mass)."""
    node = b["node_xy"][jn["node"]]
    items, dirs = [], []
    for ei, kind in jn["inc"]:
        p = b["prm"][ei]
        at, L = edge_frame(p["ctrl"])
        if kind == "through":
            dd = sp.design(len(p["ctrl"]), 20000) @ p["ctrl"]
            s0 = float(sp.arclength(dd)[np.argmin(np.linalg.norm(dd - node, axis=1))])
            T0 = at(s0)[2][0]
            dirs += [(T0, p["r"]), (-T0, p["r"])]
        else:
            s0 = 0.0 if kind == "start" else L
            T0 = at(s0)[2][0] * (1 if kind == "start" else -1)
            dirs.append((T0, p["r"]))
        items.append((ei, kind, s0, at))
    # radius beyond which the incident tubes no longer overlap
    R = 0.0
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            (ti, wi), (tj, wj) = dirs[i], dirs[j]
            ang = np.arccos(np.clip(np.dot(ti, tj), -1, 1))
            R = max(R, (wi + wj) / max(np.sin(ang), 0.25) if ang < np.deg2rad(150) else wi + wj)
    R = min(1.25 * R + 1.0, 20.0)
    stubs = []
    for ei, kind, s0, at in items:
        lo, hi, mid = piece_bounds(pcs[ei])
        sel = (mid <= s0 + R) & (mid >= s0 - R) if kind == "through" else \
            (mid <= R if kind == "start" else mid >= s0 - R)
        keep[ei] &= ~sel
        bnd = np.unique(np.r_[lo[sel], hi[sel]])
        C, N, _ = at(bnd)
        stubs.append((ei, kind, C, N))
    # density levels: thresholds of every incident edge's box staircase
    th = np.unique(np.concatenate([b["prm"][ei]["a"] * S_CUM for ei, *_ in stubs]))
    s_node = float(np.mean([b["prm"][ei]["s"] for ei, *_ in stubs]))
    allpts = np.vstack([C for _, _, C, _ in stubs])
    pad = max(b["prm"][ei]["r"] for ei, *_ in stubs) + 5 * s_node + 1
    x0, y0 = allpts.min(0) - pad
    x1, y1 = allpts.max(0) + pad
    sel_px = (P[:, 0] >= x0) & (P[:, 0] <= x1) & (P[:, 1] >= y0) & (P[:, 1] <= y1)
    Pn = P[sel_px]
    out = np.zeros(len(P))
    acc = np.zeros(len(Pn))
    for t0, t1 in zip(th[:-1], th[1:]):
        polys = []
        for ei, kind, C, N in stubs:
            p = b["prm"][ei]
            if t0 >= p["a"] * S_CUM[0]:
                continue
            k = int((p["a"] * S_CUM[1:7] > t0).sum())
            w = p["r"] * _C[k]
            polys.append(Polygon(np.vstack([C + w * N, (C - w * N)[::-1]])).buffer(0))
            if kind != "through":
                polys.append(Point(node).buffer(w, quad_segs=32))
        if polys:
            acc += (t1 - t0) * geom_mass(Pn, unary_union(polys), s_node)
    out[sel_px] = acc
    return out


def run():
    rows = []
    for sc in scenes():
        b = build(sc)
        ex_c, ex_fine = exact(sc, b)
        M = b["M"]
        crop = (slice(M, M + 64), slice(M, M + 64))
        res = dict(sc=sc, b=b, exact=ex_c[crop], exact_fine=ex_fine[M * 8:(M + 64) * 8, M * 8:(M + 64) * 8],
                   current=b["od_current"][crop],
                   sum=closed_form(sc, b, node_union=False)[crop],
                   nodes=closed_form(sc, b, node_union=True)[crop])
        hw, hs = b["hw"], b["hs"]
        res["peak"] = max(p["a"] * ((1 - hw) * rf.box_blur(np.zeros(1), p["r"], p["s"])[0] +
                                    hw * rf.box_blur(np.zeros(1), p["r"], np.hypot(p["s"], hs))[0])
                          for p in b["prm"])
        e = {k: 100 * (res[k] - res["exact"]) / res["peak"] for k in ("current", "sum", "nodes")}
        res["err"] = e
        print(f"{sc['name']:30s} max|err| % of peak: current {np.abs(e['current']).max():5.1f}   "
              f"closed form per edge {np.abs(e['sum']).max():5.1f}   "
              f"+ union at nodes {np.abs(e['nodes']).max():5.1f}", flush=True)
        rows.append(res)
    return rows


if __name__ == "__main__":
    # sanity: the polygon formula on a square equals the separable erf product
    from scipy.special import ndtr
    P = np.array([[0.3, -0.2], [2.5, 1.0], [-3.0, 0.4]])
    sq = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float)
    ref = (ndtr((1 - P[:, 0]) / 0.7) - ndtr((-1 - P[:, 0]) / 0.7)) * \
          (ndtr((1 - P[:, 1]) / 0.7) - ndtr((-1 - P[:, 1]) / 0.7))
    print("polygon formula check:", np.round(ring_mass(P, sq, 0.7), 6), "vs", np.round(ref, 6))
    run()
