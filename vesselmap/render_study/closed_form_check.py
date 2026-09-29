"""Test closed-form renderings of a curved tube against the exact image.

Exact model: the sharp tube (6-box chord cross-section around the spline
centreline) convolved with a 2-D Gaussian, plus the halo.  Candidates:

  current   render.py: nearest sample only, straight tube in its tangent frame
  sum       every centreline piece adds a separable 2-D blurred box
            A_j(u) * B_k(d)      (no curvature term)
  sum+jac   the same with the tube Jacobian (1 - kappa d'):
            A_j(u) * [B_k(d) - kappa_j * D_k(d)]
  laplace   nearest sample only, osculating circle expanded to 2nd order
"""
import os
import sys
import time

import numpy as np
import torch
from scipy.special import erf, ndtr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_figures as rf                     # noqa: E402
from vesselmap import spline as sp              # noqa: E402
from vesselmap.render import gaussian_blur, _C, _W   # noqa: E402

S2 = np.sqrt(2.0)
MASK_ENDS = False
SQ2PI = np.sqrt(2 * np.pi)


def gauss(t, s):
    return np.exp(-0.5 * (t / s) ** 2) / (SQ2PI * s)


def B_(d, w, s):
    """Across factor of one box: integral over |d'| < w of phi_s(d - d')."""
    return 0.5 * (erf((w - d) / (S2 * s)) + erf((w + d) / (S2 * s)))


def D_(d, w, s):
    """First moment: integral over |d'| < w of d' phi_s(d - d')."""
    return d * B_(d, w, s) + s * s * (gauss(d + w, s) - gauss(d - w, s))


def F_(d, kap, r, s):
    """Across factor of the whole cross-section with the tube Jacobian."""
    w = r * _C
    dd = d[..., None]
    return ((B_(dd, w, s) - kap[..., None] * D_(dd, w, s)) * _W).sum(-1)


# ---- centreline pieces ---------------------------------------------------
def pieces(ctrl, spacing=0.75, tol=None, lmax=8.0):
    """Centreline pieces [lo, hi] (arclength) with the midpoint, unit tangent
    and signed curvature (dT/ds = kappa N, N = T rotated +90 deg).
    tol=None: uniform pieces of about `spacing` px.  tol=t: adaptive pieces
    whose sagitta kappa l^2 / 8 stays below t px (straight stretches get
    long pieces, which is exact for a straight tube)."""
    n = len(ctrl)
    ud = np.linspace(0, 1, max(4000, 40 * n))
    xy = sp.design_at(n, ud) @ ctrl
    sd = sp.arclength(xy)
    L = sd[-1]
    d1 = sp.design_at(n, ud, 1) @ ctrl
    d2 = sp.design_at(n, ud, 2) @ ctrl
    kd = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.linalg.norm(d1, axis=1) ** 3
    if tol is None:
        m = int(np.ceil(L / spacing))
        br = np.linspace(0, L, m + 1)
    else:
        br = [0.0]
        while br[-1] < L:
            s0 = br[-1]
            win = (sd >= s0) & (sd <= s0 + lmax)
            kmax = np.abs(kd[win]).max() if win.any() else 0.0
            l = np.clip(np.sqrt(8 * tol / max(kmax, 1e-9)), spacing, lmax)
            br.append(min(L, s0 + l))
        br = np.array(br)
    lo, hi = br[:-1], br[1:]
    mid = 0.5 * (lo + hi)
    um = np.interp(mid, sd, ud)
    C = sp.design_at(n, um) @ ctrl
    D1 = sp.design_at(n, um, 1) @ ctrl
    D2 = sp.design_at(n, um, 2) @ ctrl
    sp1 = np.linalg.norm(D1, axis=1)
    T = D1 / sp1[:, None]
    kap = (D1[:, 0] * D2[:, 1] - D1[:, 1] * D2[:, 0]) / sp1 ** 3
    return dict(C=C, T=T, kap=kap, lo=lo - mid, hi=hi - mid, L=L)


def render_sum(P, pc, r, s, a, jac=True, chunk=1500):
    """a * sum_j A_j(u) * F(d; kappa_j): the line integral of the 2-D
    Gaussian along the tube, one closed-form term per piece."""
    C, T = pc["C"], pc["T"]
    N = np.stack([-T[:, 1], T[:, 0]], 1)
    kap = pc["kap"] if jac else np.zeros_like(pc["kap"])
    out = np.zeros(len(P))
    pairs = 0
    for i0 in range(0, len(P), chunk):
        v = P[i0:i0 + chunk, None, :] - C[None]
        u = (v * T[None]).sum(-1)
        d = (v * N[None]).sum(-1)
        A = ndtr((u - pc["lo"][None]) / s) - ndtr((u - pc["hi"][None]) / s)
        near = (np.abs(d) < r + 4 * s) & (A > 1e-7)
        pairs += int(near.sum())
        F = F_(d, np.broadcast_to(kap, d.shape), r, s)
        out[i0:i0 + chunk] = a * (A * F * near).sum(1)
    return out, pairs


def render_laplace(P, pc, r, s, a):
    """Nearest piece only; osculating circle, Laplace-expanded to 2nd order:
    a / (1 - k d) * [F(d) + F'(d) s^2 k / (2 (1 - k d))] * taper."""
    C, T, kap = pc["C"], pc["T"], pc["kap"]
    N = np.stack([-T[:, 1], T[:, 0]], 1)
    from scipy.spatial import cKDTree
    _, j = cKDTree(C).query(P)
    v = P - C[j]
    u = (v * T[j]).sum(1)
    d = (v * N[j]).sum(1)
    k = kap[j]
    J = np.maximum(1 - k * d, 0.25)
    h = 1e-3
    F0 = F_(d, k, r, s)
    F1 = (F_(d + h, k, r, s) - F_(d - h, k, r, s)) / (2 * h)
    arc = np.cumsum(pc["hi"] - pc["lo"]) - pc["hi"]            # piece midpoints in arclength
    along = arc[j] + u
    taper = ndtr(along / s) - ndtr((along - pc["L"]) / s)
    return a / J * (F0 + F1 * s * s * k / (2 * J)) * taper


def with_halo(V, hw, hs):
    t = torch.tensor(V, dtype=torch.float32)
    return ((1 - hw) * t + hw * gaussian_blur(t, hs)).numpy()


def straight():
    ang = np.deg2rad(28)
    t = np.arange(-60, 60, 0.05)
    xy = np.stack([32 + t * np.cos(ang), 32 + t * np.sin(ang)], 1)
    return xy, dict(r=1.5, s=2.0, a=0.35), ("col", 32, 18, 46), "Straight (reference check)"


def run(H=64, W=64, ss=8):
    rows = []
    for f in [straight] + rf.SCENES:
        xy, prm, prof, title = f()
        res = rf.scene(xy, prm["r"], prm["s"], prm["a"], H, W, ss=ss)
        M, net, eid = res["M"], res["net"], res["eid"]
        Hc, Wc = H + 2 * M, W + 2 * M
        ctrl = net.edges[eid].ctrl
        gy, gx = np.mgrid[0:Hc, 0:Wc]
        P = np.stack([gx.ravel(), gy.ravel()], 1).astype(float)
        r, s, a = res["r"], res["s"], res["a"]
        hw, hs = (float(v) for v in res["model"].halo())
        crop = (slice(M, M + H), slice(M, M + W))
        out = dict(title=title, prof=prof, res=res, current=res["od_m"])
        for name, kw in [("sum", dict(jac=False)), ("sum+jac", dict(jac=True))]:
            t0 = time.time()
            V, pairs = render_sum(P, pieces(ctrl), r, s, a, **kw)
            out[name] = with_halo(V.reshape(Hc, Wc), hw, hs)[crop]
            out[name + "_pairs"] = pairs
        V, pairs = render_sum(P, pieces(ctrl, tol=0.02), r, s, a, jac=True)
        out["adaptive"] = with_halo(V.reshape(Hc, Wc), hw, hs)[crop]
        out["adaptive_pairs"] = pairs
        out["adaptive_n"] = len(pieces(ctrl, tol=0.02)["C"])
        out["uniform_n"] = len(pieces(ctrl)["C"])
        out["laplace"] = with_halo(render_laplace(P, pieces(ctrl), r, s, a).reshape(Hc, Wc), hw, hs)[crop]
        # current renderer's pairs: one per (pixel, edge) within reach
        out["current_pairs"] = len(res["model"].e_pix)
        rows.append(out)
        # the reference has round end caps, the models flat blurred ones: skip the free ends
        ends = ctrl[[0, -1]] - M
        yy, xx = np.mgrid[0:H, 0:W]
        dend = np.min([np.hypot(xx - p[0], yy - p[1]) for p in ends], 0)
        keep = dend > r + 4 * s if MASK_ENDS else np.ones_like(dend, bool)
        out["keep"] = keep
        e = lambda k: np.abs(100 * (out[k] - res["od_e"]) / res["peak"])[keep]
        print(f"{f.__name__:9s} max|err| % of peak:  current {e('current').max():5.1f}   "
              f"laplace {e('laplace').max():5.1f}   sum {e('sum').max():5.1f}   "
              f"sum+jac {e('sum+jac').max():5.1f}   adaptive {e('adaptive').max():5.1f}   | "
              f"pairs/px-in-reach: uniform {out['sum+jac_pairs'] / out['current_pairs']:.1f}  "
              f"adaptive {out['adaptive_pairs'] / out['current_pairs']:.1f}  "
              f"(pieces {out['uniform_n']} -> {out['adaptive_n']})")
    return rows


if __name__ == "__main__":
    ss = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    run(ss=ss)
