"""Synthetic vessel images with known ground truth, and centreline metrics.

The generator is deliberately independent of render.py: every vessel is a
cylinder whose optical path is computed per pixel on a 2x super-sampled
grid, blurred with its own depth-dependent Gaussian, and the scene has a
lumpy textured background, illumination falloff and shot + read noise.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


def _smooth_path(rng, start, heading, length, step=1.0, wiggle=0.08, corr=0.9):
    pts = [np.array(start, float)]
    h = heading
    dh = 0.0
    for _ in range(int(length / step)):
        dh = corr * dh + wiggle * rng.standard_normal()
        h += dh
        pts.append(pts[-1] + step * np.array([math.cos(h), math.sin(h)]))
    return np.array(pts)


def make_network(rng, shape, n_trees=4, n_cross=4, n_capillary=25):
    """Ground-truth vessels: list of dict(xy, r (per point), blur, amp)."""
    H, W = shape
    vessels = []
    bif = []

    def add_tree(start, heading, r0, blur, depth, length):
        xy = _smooth_path(rng, start, heading, length, wiggle=0.015)
        r = np.linspace(r0, max(0.6, 0.8 * r0), len(xy))
        vessels.append(dict(xy=xy, r=r, blur=blur, amp=0.35 + 0.1 * rng.random()))
        if depth <= 0 or r0 < 1.0:
            return
        for _ in range(rng.integers(1, 3)):
            i = int(rng.integers(len(xy) // 5, 4 * len(xy) // 5))
            tan = xy[min(i + 3, len(xy) - 1)] - xy[max(i - 3, 0)]
            ang = math.atan2(tan[1], tan[0]) + rng.choice([-1, 1]) * rng.uniform(0.5, 1.3)
            bif.append(xy[i].copy())
            add_tree(xy[i], ang, max(0.7, r[i] * rng.uniform(0.45, 0.75)), blur, depth - 1,
                     length * rng.uniform(0.4, 0.7))

    for _ in range(n_trees):
        start = (rng.uniform(0, W), rng.choice([0, H - 1]))
        heading = -math.pi / 2 if start[1] > 0 else math.pi / 2
        heading += rng.uniform(-0.6, 0.6)
        add_tree(start, heading, rng.uniform(3, 9), rng.uniform(0.7, 1.5), 3, rng.uniform(0.6, 1.0) * H)
    # out-of-focus vessels that cross everything at another depth
    for _ in range(n_cross):
        start = (0, rng.uniform(0, H))
        xy = _smooth_path(rng, start, rng.uniform(-0.4, 0.4), W * 1.1, wiggle=0.008)
        r0 = rng.uniform(2, 10)
        vessels.append(dict(xy=xy, r=np.full(len(xy), r0), blur=rng.uniform(2.5, 9.0),
                            amp=0.25 + 0.25 * rng.random()))
    # small capillaries, tortuous, faint and sharp
    for _ in range(n_capillary):
        start = (rng.uniform(0, W), rng.uniform(0, H))
        xy = _smooth_path(rng, start, rng.uniform(0, 2 * math.pi), rng.uniform(40, 200),
                          wiggle=0.07, corr=0.85)
        vessels.append(dict(xy=xy, r=np.full(len(xy), rng.uniform(0.6, 1.4)),
                            blur=rng.uniform(0.6, 1.5), amp=0.12 + 0.2 * rng.random()))
    for v in vessels:
        inside = (v["xy"][:, 0] >= -20) & (v["xy"][:, 0] < W + 20) & \
                 (v["xy"][:, 1] >= -20) & (v["xy"][:, 1] < H + 20)
        v["xy"], v["r"] = v["xy"][inside], v["r"][inside]
    vessels = [v for v in vessels if len(v["xy"]) > 10]
    return vessels, np.array(bif).reshape(-1, 2)


def render(vessels, shape, rng, ss=2, noise=True, bright=0.8):
    H, W = shape
    Hs, Ws = H * ss, W * ss
    od = np.zeros((Hs, Ws), np.float32)
    for v in vessels:
        xy = v["xy"] * ss + (ss - 1) / 2.0
        r = v["r"] * ss
        R = r.max() + 1
        x0, x1 = int(max(0, xy[:, 0].min() - R - 2)), int(min(Ws, xy[:, 0].max() + R + 3))
        y0, y1 = int(max(0, xy[:, 1].min() - R - 2)), int(min(Hs, xy[:, 1].max() + R + 3))
        if x1 <= x0 or y1 <= y0:
            continue
        # dense resample so the nearest-point distance is accurate
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        s = np.r_[0, np.cumsum(seg)]
        q = np.linspace(0, s[-1], int(s[-1] / 0.3) + 2)
        dx = np.stack([np.interp(q, s, xy[:, 0]), np.interp(q, s, xy[:, 1])], 1)
        dr = np.interp(q, s, r)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        P = np.stack([xx.ravel(), yy.ravel()], 1).astype(float)
        d, j = cKDTree(dx).query(P, distance_upper_bound=R + 1)
        ok = np.isfinite(d)
        layer = np.zeros(len(P), np.float32)
        rr = dr[j[ok]]
        layer[ok] = np.sqrt(np.clip(1 - (d[ok] / rr) ** 2, 0, 1))
        layer = layer.reshape(y1 - y0, x1 - x0)
        full = np.zeros((Hs, Ws), np.float32)
        full[y0:y1, x0:x1] = layer * v["amp"]
        od += ndi.gaussian_filter(full, v["blur"] * ss, truncate=3.5)
    od = cv2.resize(od, (W, H), interpolation=cv2.INTER_AREA)
    # background: lumpy scleral texture + illumination falloff
    yy, xx = np.mgrid[0:H, 0:W]
    illum = bright * np.exp(-(((xx - W * rng.uniform(0.3, 0.7)) / (0.9 * W)) ** 2 +
                              ((yy - H * rng.uniform(0.3, 0.7)) / (0.9 * H)) ** 2))
    tex = ndi.gaussian_filter(rng.standard_normal((H, W)), 45)
    tex = 0.12 * tex / tex.std()
    fine = ndi.gaussian_filter(rng.standard_normal((H, W)), 4)
    fine = 0.006 * fine / fine.std()
    I = illum * np.exp(tex + fine - od)
    if noise:
        e = 4095.0 * I / 1.2
        e = rng.poisson(np.maximum(e, 0) * 1.0) + rng.normal(0, 3.0, e.shape)
        I = np.clip(e / 4095.0 * 1.2, 0, 1)
    return I.astype(np.float32), od


def make_scene(seed=0, shape=(512, 768), **kw):
    rng = np.random.default_rng(seed)
    vessels, bif = make_network(rng, shape, **kw)
    I, od = render(vessels, shape, rng)
    return I, vessels, bif


def centreline_metrics(net, vessels, shape, tol_min=2.0, tol_frac=0.5, spacing=1.0,
                       min_amp_visible=0.0):
    """Centreline recall (per true vessel point) and precision (per detected
    point), with tolerance max(tol_min, tol_frac * true diameter)."""
    H, W = shape
    tx, tr, tb, ta = [], [], [], []
    for v in vessels:
        xy = v["xy"]
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        s = np.r_[0, np.cumsum(seg)]
        q = np.arange(0, s[-1], spacing)
        p = np.stack([np.interp(q, s, xy[:, 0]), np.interp(q, s, xy[:, 1])], 1)
        ok = (p[:, 0] >= 2) & (p[:, 0] < W - 2) & (p[:, 1] >= 2) & (p[:, 1] < H - 2)
        tx.append(p[ok])
        tr.append(np.interp(q, s, v["r"])[ok])
        tb.append(np.full(ok.sum(), v["blur"]))
        ta.append(np.full(ok.sum(), v["amp"]))
    tx, tr, tb, ta = map(np.concatenate, (tx, tr, tb, ta))
    dx = [net.sample(e, spacing)["xy"] for e in net.edges]
    dx = np.concatenate(dx) if dx else np.zeros((0, 2))
    tol_t = np.maximum(tol_min, tol_frac * 2 * tr)
    if len(dx):
        d_t, _ = cKDTree(dx).query(tx)
    else:
        d_t = np.full(len(tx), np.inf)
    hit = d_t <= tol_t
    if len(dx):
        d_d, j = cKDTree(tx).query(dx)
        prec_hit = d_d <= tol_t[j]
    else:
        prec_hit = np.zeros(0, bool)
    out = dict(recall=float(hit.mean()), precision=float(prec_hit.mean()) if len(prec_hit) else 0.0,
               true_len=int(len(tx)), detected_len=int(len(dx)))
    bins = [(0, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 20)]
    out["recall_by_radius"] = {f"{a}-{b}": float(hit[(tr >= a) & (tr < b)].mean())
                               for a, b in bins if ((tr >= a) & (tr < b)).any()}
    bb = [(0, 1.5), (1.5, 4.0), (4.0, 20)]
    out["recall_by_blur"] = {f"{a}-{b}": float(hit[(tb >= a) & (tb < b)].mean())
                             for a, b in bb if ((tb >= a) & (tb < b)).any()}
    return out
