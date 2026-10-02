"""Clamped uniform B-splines used for centrelines and profile parameters."""
from __future__ import annotations

import functools

import numpy as np
from scipy.interpolate import BSpline


def _knots(n_ctrl: int, degree: int) -> np.ndarray:
    n_inner = n_ctrl - degree - 1
    inner = np.linspace(0.0, 1.0, n_inner + 2)[1:-1] if n_inner > 0 else np.array([])
    return np.concatenate([np.zeros(degree + 1), inner, np.ones(degree + 1)])


def degree_for(n_ctrl: int) -> int:
    return int(min(3, n_ctrl - 1))


@functools.lru_cache(maxsize=4096)
def _design(n_ctrl: int, n_samp: int, deriv: int) -> np.ndarray:
    k = degree_for(n_ctrl)
    t = _knots(n_ctrl, k)
    u = np.linspace(0.0, 1.0, n_samp)
    eye = np.eye(n_ctrl)
    spl = BSpline(t, eye, k, extrapolate=False)
    if deriv:
        spl = spl.derivative(deriv)
    m = spl(u)
    m = np.nan_to_num(m)
    return m.astype(np.float32)


def design(n_ctrl: int, n_samp: int, deriv: int = 0) -> np.ndarray:
    """(n_samp x n_ctrl) matrix evaluating the spline (or a derivative wrt u)
    at n_samp uniformly spaced parameters in [0, 1]."""
    return _design(int(n_ctrl), int(n_samp), int(deriv))


def arclength(poly: np.ndarray) -> np.ndarray:
    d = np.sqrt((np.diff(poly, axis=0) ** 2).sum(1))
    return np.concatenate([[0.0], np.cumsum(d)])


def resample_polyline(poly: np.ndarray, n: int) -> np.ndarray:
    s = arclength(poly)
    if s[-1] <= 0:
        return np.repeat(poly[:1], n, axis=0)
    q = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(q, s, poly[:, i]) for i in range(poly.shape[1])], 1)


def fit_ctrl(values: np.ndarray, n_ctrl: int, smooth: float = 1e-3,
             pin_ends: bool = True) -> np.ndarray:
    """Least-squares control points so that the spline sampled uniformly at
    len(values) parameters matches `values` (shape (m, d)).  `values` should be
    sampled uniformly in arclength.  A small second-difference penalty keeps
    the problem well posed.  With pin_ends the first/last control points equal
    the first/last values (clamped splines interpolate their ends)."""
    values = np.asarray(values, np.float64)
    if values.ndim == 1:
        values = values[:, None]
    m = values.shape[0]
    if m < 2:
        values = np.repeat(values, 2, axis=0)
        m = 2
    B = design(n_ctrl, m).astype(np.float64)
    D = np.diff(np.eye(n_ctrl), 2, axis=0) if n_ctrl > 2 else np.zeros((0, n_ctrl))
    A = B.T @ B + smooth * m * (D.T @ D) + 1e-9 * np.eye(n_ctrl)
    rhs = B.T @ values
    if not pin_ends or n_ctrl < 3:
        return np.linalg.solve(A, rhs)
    ends = np.stack([values[0], values[-1]])
    inner = slice(1, n_ctrl - 1)
    Ai = A[inner, inner]
    rhs_i = rhs[inner] - A[inner][:, [0, n_ctrl - 1]] @ ends
    ci = np.linalg.solve(Ai, rhs_i)
    return np.concatenate([ends[:1], ci, ends[1:]], 0)


def n_ctrl_for_length(length: float, spacing: float, minimum: int = 4) -> int:
    return int(max(minimum, int(np.ceil(length / spacing)) + 1))


def n_samples_for_length(length: float, spacing: float = 0.7) -> int:
    return int(max(4, int(np.ceil(length / spacing)) + 1))


def design_at(n_ctrl: int, u: np.ndarray, deriv: int = 0) -> np.ndarray:
    """(len(u) x n_ctrl) design matrix at arbitrary parameters u in [0, 1]."""
    k = degree_for(n_ctrl)
    t = _knots(n_ctrl, k)
    spl = BSpline(t, np.eye(n_ctrl), k, extrapolate=False)
    if deriv:
        spl = spl.derivative(deriv)
    u = np.clip(np.asarray(u, float), 0.0, 1.0)
    return np.nan_to_num(spl(u)).astype(np.float32)


def arclength_params(ctrl: np.ndarray, spacing: float, oversample: int = 8):
    """Parameters u in [0, 1] of points spaced (about) `spacing` apart along
    the curve, and the curve length.  Rendering and sampling at these u keeps
    samples evenly spread even where the parameterisation is uneven (control
    points bunched up near a sharp turn)."""
    n = len(ctrl)
    m0 = max(64, oversample * n)
    xy = design(n, m0) @ ctrl
    s = arclength(xy)
    L = float(s[-1])
    m = n_samples_for_length(L, spacing)
    if L <= 0:
        return np.linspace(0.0, 1.0, m), 0.0
    u0 = np.linspace(0.0, 1.0, m0)
    return np.interp(np.linspace(0.0, L, m), s, u0), L
