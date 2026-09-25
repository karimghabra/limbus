"""Smoothing-spline representation of a vessel centreline.

The skeleton gives an ordered chain of integer pixel centres p_0 ... p_{N-1};
it is jagged (staircase) and cannot be differentiated. We replace it by a
parametric cubic B-spline

    r(u) = (x(u), y(u)) = sum_i  c_i  B_{i,3}(u),     u in [0, 1]

where B_{i,3} are cubic B-spline basis functions on a knot vector and c_i are
2-D control points. scipy's `splprep` chooses knots and control points to
minimise roughness subject to

    sum_j  w_j^2 |r(u_j) - p_j|^2  <=  s          (smoothing condition)

so `s` trades fidelity for smoothness; we use s = N * sigma^2 with sigma the
expected positional noise of skeleton pixels (~0.5-1 px).

Arc-length re-parameterisation
------------------------------
u is not proportional to distance travelled. We integrate the speed

    s(u) = int_0^u |r'(v)| dv

numerically, invert it, and resample so that consecutive samples are exactly
`ds` pixels apart along the curve. All per-vessel profiles (diameter,
curvature, kymographs) are functions of this arc length s.

Frenet frame and curvature
--------------------------
    T(s) = r'/|r'|                      unit tangent (direction of travel)
    N(s) = (-T_y, T_x)                  unit normal (T rotated +90 degrees)
    kappa(s) = (x' y'' - y' x'') / |r'|^3   signed curvature, 1/px
Radius of curvature is 1/|kappa|.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import interpolate


@dataclass
class VesselSpline:
    tck: tuple            # scipy (knots, [cx, cy], degree)
    s: np.ndarray         # arc length samples (px), uniform spacing ds
    u: np.ndarray         # spline parameter at those samples
    xy: np.ndarray        # (n, 2) centreline points
    T: np.ndarray         # (n, 2) unit tangents
    N: np.ndarray         # (n, 2) unit normals
    kappa: np.ndarray     # (n,) signed curvature 1/px
    raw: np.ndarray       # the input pixel chain

    @property
    def length(self) -> float:
        return float(self.s[-1])

    @property
    def control_points(self) -> np.ndarray:
        return np.stack(self.tck[1], 1)

    def at(self, s_query):
        """Point(s) at arc length s_query."""
        u = np.interp(s_query, self.s, self.u)
        return np.stack(interpolate.splev(u, self.tck), -1)

    def reversed(self) -> "VesselSpline":
        """Same curve traversed in the opposite direction (used after flow orientation)."""
        L = self.length
        return VesselSpline(tck=self.tck, s=(L - self.s[::-1]), u=self.u[::-1], xy=self.xy[::-1],
                            T=-self.T[::-1], N=-self.N[::-1], kappa=-self.kappa[::-1], raw=self.raw[::-1])


def _dedupe(p):
    keep = np.ones(len(p), bool)
    keep[1:] = np.any(np.abs(np.diff(p, axis=0)) > 1e-9, axis=1)
    return p[keep]


def fit_spline(path, sigma=0.8, ds=1.0, end_weight=5.0, oversample=20) -> VesselSpline:
    p = _dedupe(np.asarray(path, float))
    n = len(p)
    k = int(min(3, n - 1))
    if k < 1:
        raise ValueError("path too short")
    # chord-length parameterisation of the data points
    d = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
    u0 = d / d[-1]
    w = np.ones(n); w[0] = w[-1] = end_weight          # pin the ends to the nodes
    tck, _ = interpolate.splprep([p[:, 0], p[:, 1]], u=u0, w=w, k=k, s=n * sigma**2)

    # numerically integrate |r'(u)| on a fine grid -> s(u), then invert
    uf = np.linspace(0, 1, max(200, n * oversample))
    dx, dy = interpolate.splev(uf, tck, der=1)
    speed = np.hypot(dx, dy)
    sf = np.concatenate([[0], np.cumsum(0.5 * (speed[1:] + speed[:-1]) * np.diff(uf))])
    L = sf[-1]
    m = max(int(np.floor(L / ds)) + 1, 2)
    s = np.linspace(0, L, m)
    u = np.interp(s, sf, uf)

    x, y = interpolate.splev(u, tck)
    x1, y1 = interpolate.splev(u, tck, der=1)
    if k >= 2:
        x2, y2 = interpolate.splev(u, tck, der=2)
    else:
        x2 = y2 = np.zeros_like(x1)
    sp = np.hypot(x1, y1) + 1e-12
    T = np.stack([x1 / sp, y1 / sp], 1)
    N = np.stack([-T[:, 1], T[:, 0]], 1)
    kappa = (x1 * y2 - y1 * x2) / sp**3
    return VesselSpline(tck=tck, s=s, u=u, xy=np.stack([x, y], 1), T=T, N=N, kappa=kappa, raw=p)
