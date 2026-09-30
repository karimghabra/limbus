"""A zoo of controlled vessel structures, and a completeness score.

The random scenes of synthetic.make_scene contain structures by accident. The
zoo lays out one small tile per case, each a structure with one parameter
swept along a row, so an annotator's failures can be counted per structure:

    bifurcations (branch angle; Murray's-law asymmetry), crossings at the
    same and at different depths (angle), parallel pairs (separation),
    kissing pairs (closest gap), a vessel ending on another (angle),
    hairpins (turn radius), forks whose branches run on side by side
    (separation), faint capillaries under a blurred vessel (contrast), and
    complex nodes (several junctions at one place).

Tiles lie on a background texture with the power spectrum of reference
frame 20's residual off its mapped vessels, flat illumination, and shot and
read noise. Radius may vary along a vessel, blur is constant along it, and
contrast varies along it without being smooth, as in single real frames.

completeness() scores an annotated vessel: a stretch whose cross-section,
within the annotated lumen, has two dark peaks with a brighter gap between
them is two vessels annotated as one.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import interpolate
from scipy import ndimage as ndi

from .synthetic import render

# power spectrum (variance per frequency sample of a Hann-windowed periodogram)
# of frame 20's log-intensity residual off the mapped vessels, less the white
# sensor-noise floor (sigma 0.020): 44 tiles of 48 px
F20_FREQ = (0.010, 0.031, 0.052, 0.073, 0.094, 0.115, 0.135, 0.156)
F20_PSD = (1.27e-2, 2.03e-2, 4.31e-3, 7.83e-4, 1.93e-4, 6.06e-5, 6.10e-5, 1.20e-5)


def texture(shape, rng, scale=1.0, freq=F20_FREQ, psd=F20_PSD):
    """Gaussian texture (log units) with the given radial power spectrum:
    white noise shaped in the Fourier domain, calibrated so that a
    Hann-windowed periodogram of 48 px tiles reads the spectrum back."""
    H, W = shape
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.fftfreq(W)[None, :]
    fr = np.hypot(fx, fy)
    lp = np.interp(fr, freq, np.log(np.asarray(psd)), left=np.log(psd[0]), right=-40.0)
    amp = np.sqrt(np.exp(lp))
    amp[0, 0] = 0.0
    t = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((H, W))) * amp))
    # calibration: white unit noise shaped by sqrt(P) has periodogram ~ P
    # (numpy's unnormalised FFT pair keeps the per-sample variance)
    return (scale * t).astype(np.float32)


def _path(ctrl, step=1.0):
    """Points 1 px apart along a smooth curve through the control points."""
    ctrl = np.asarray(ctrl, float)
    k = min(3, len(ctrl) - 1)
    tck, _ = interpolate.splprep([ctrl[:, 0], ctrl[:, 1]], s=0, k=k)
    u = np.linspace(0, 1, 2000)
    xy = np.stack(interpolate.splev(u, tck), 1)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    q = np.arange(0, s[-1], step)
    return np.stack([np.interp(q, s, xy[:, 0]), np.interp(q, s, xy[:, 1])], 1)


def _vessel(xy, r0, r1=None, blur=1.0, amp=0.35, rng=None, jitter=0.15):
    """A vessel dict: radius from r0 to r1 along it, constant blur, contrast
    varying along it (rough, 2 px correlation) by about jitter."""
    n = len(xy)
    r = np.linspace(r0, r0 if r1 is None else r1, n)
    a = np.full(n, float(amp))
    if rng is not None and jitter > 0:
        g = ndi.gaussian_filter1d(rng.standard_normal(n), 2.0)
        a = a * np.clip(1.0 + jitter * g / (g.std() + 1e-9), 0.4, 1.8)
    return dict(xy=np.asarray(xy, float), r=r, blur=float(blur), amp=a)


def _ray(p, ang, length):
    return np.asarray(p, float) + np.outer(np.arange(0, length, 1.0),
                                           [math.cos(ang), math.sin(ang)])


def _murray(r0, f):
    """Child radii of a parent r0 splitting its r^3 in shares f, 1 - f."""
    return r0 * f ** (1 / 3), r0 * (1 - f) ** (1 / 3)


# ---------------------------------------------------------------- structures
# Each builder takes the tile size T, the swept value and rng; it returns the
# vessels in tile coordinates and the locations that are ambiguous by
# construction (junctions, crossings, closest approaches).

def bifurcation(T, angle, rng, f=0.6, r0=2.6):
    c = np.array([T * 0.45, T * 0.55])
    parent = _path([c + [-60, 20], c + [-30, 8], c])
    r1, r2 = _murray(r0, f)
    a = math.radians(angle)
    # the thicker child deviates less (branching roughly balances momentum)
    a1, a2 = -a * (1 - f), a * f
    base = -0.3
    kids = [_path([c, c + 25 * np.array([math.cos(base + t), math.sin(base + t)]),
                   c + 70 * np.array([math.cos(base + t * 1.1), math.sin(base + t * 1.1)])])
            for t in (a1, a2)]
    return [_vessel(parent, r0, r0 * 0.97, 1.0, 0.4, rng),
            _vessel(kids[0], r1, r1 * 0.85, 1.0, 0.4, rng),
            _vessel(kids[1], r2, r2 * 0.85, 1.0, 0.4, rng)], [c]


def bifurcation_asym(T, f, rng):
    return bifurcation(T, 70.0, rng, f=f)


def crossing(T, angle, rng, depth=False, ra=1.6, rb=1.2):
    """Two vessels crossing at angle (deg), radii ra and rb (same depth)."""
    c = np.array([T / 2, T / 2])
    a = math.radians(angle)
    v1 = _path([c + [-80, 0], c, c + [80, 0]])
    d = np.array([math.cos(a), math.sin(a)])
    v2 = _path([c - 80 * d, c, c + 80 * d])
    if depth:        # a sharp capillary over a blurred vessel at another depth
        return [_vessel(v1, 4.0, 4.0, 4.0, 0.35, rng), _vessel(v2, 0.9, 0.9, 1.0, 0.3, rng)], [c]
    return [_vessel(v1, ra, ra * 0.95, 1.0, 0.4, rng), _vessel(v2, rb, rb * 0.95, 1.0, 0.35, rng)], [c]


def crossing_depth(T, angle, rng):
    return crossing(T, angle, rng, depth=True)


def parallel(T, sep, rng, r=1.2):
    """Two vessels whose centrelines run sep px apart along a gentle curve."""
    xs = np.linspace(-80, 80, 9)
    mid = np.stack([T / 2 + xs, T / 2 + 10 * np.sin(xs / 45.0)], 1)
    t = np.gradient(mid, axis=0)
    n = np.stack([-t[:, 1], t[:, 0]], 1) / np.linalg.norm(t, axis=1, keepdims=True)
    a, b = _path(mid + n * sep / 2), _path(mid - n * sep / 2)
    return [_vessel(a, r, r, 1.0, 0.35, rng), _vessel(b, r * 0.9, r * 0.9, 1.0, 0.35, rng)], \
        [mid[len(mid) // 2]]


def kissing(T, gap, rng, r=1.3):
    """Two vessels that approach until their lumens are gap px apart, then part."""
    c = np.array([T / 2, T / 2])
    h = r + r + gap
    a = _path([c + [-80, -25], c + [-20, -h / 2 - 2], c + [0, -h / 2], c + [20, -h / 2 - 2], c + [80, -25]])
    b = _path([c + [-80, 25], c + [-20, h / 2 + 2], c + [0, h / 2], c + [20, h / 2 + 2], c + [80, 25]])
    return [_vessel(a, r, r, 1.0, 0.35, rng), _vessel(b, r, r, 1.0, 0.35, rng)], [c]


def ends_on(T, angle, rng, r_h=2.2, r_a=1.0):
    """A vessel (radius r_a) whose end lies on the side of another (r_h), at
    angle (deg) to it."""
    c = np.array([T / 2, T * 0.6])
    host = _path([c + [-80, 0], c, c + [80, 0]])
    a = math.radians(180 + angle)
    tip = c + [0, -r_h * 0.5]
    arm = _path([tip + 80 * np.array([math.cos(a), -abs(math.sin(a))]), tip + 30 * np.array([math.cos(a), -abs(math.sin(a))]), tip])
    return [_vessel(host, r_h, r_h, 1.0, 0.4, rng), _vessel(arm, r_a, r_a, 1.0, 0.35, rng)], [tip]


def hairpin(T, radius, rng, r=0.8):
    """A capillary that turns back on itself: a straight leg, a half circle
    of the given radius, and a straight leg back."""
    c = np.array([T / 2 + 20, T / 2])
    L = 70.0
    leg1 = np.stack([np.arange(c[0] - L, c[0], 1.0), np.full(int(L), c[1] - radius)], 1)
    th = np.arange(-math.pi / 2, math.pi / 2, 1.0 / radius)
    turn = np.stack([c[0] + radius * np.cos(th), c[1] + radius * np.sin(th)], 1)
    leg2 = np.stack([np.arange(c[0], c[0] - L, -1.0), np.full(int(L), c[1] + radius)], 1)
    return [_vessel(np.vstack([leg1, turn, leg2]), r, r, 1.2, 0.3, rng)], [c]


def fork_parallel(T, sep, rng, r0=1.8):
    """A fork whose two branches then run on side by side sep px apart."""
    c = np.array([T * 0.3, T / 2])
    parent = _path([c + [-60, 0], c + [-25, 0], c])
    r1, r2 = _murray(r0, 0.5)
    k1 = _path([c, c + [15, -sep / 2], c + [40, -sep / 2], c + [100, -sep / 2]])
    k2 = _path([c, c + [15, sep / 2], c + [40, sep / 2], c + [100, sep / 2]])
    return [_vessel(parent, r0, r0, 1.0, 0.4, rng), _vessel(k1, r1, r1, 1.0, 0.4, rng),
            _vessel(k2, r2, r2, 1.0, 0.4, rng)], [c, c + [40, 0]]


def faint_under_blurred(T, amp, rng):
    c = np.array([T / 2, T / 2])
    wide = _path([c + [-15, -80], c, c + [15, 80]])
    cap = _path([c + [-80, -15], c + [-30, 5], c + [10, -5], c + [80, 10]])
    return [_vessel(wide, 5.0, 5.0, 5.0, 0.35, rng), _vessel(cap, 0.8, 0.8, 1.0, amp, rng)], [c]


def parallel_rel(T, k, rng, r=1.2):
    """A parallel pair of radii r and 0.9 r whose centres are k times the sum
    of their radii apart (k = 1: the lumens touch)."""
    return parallel(T, k * 1.9 * r, rng, r=r)


def complex_node(T, case, rng, k=1.0):
    """Several junctions at one place: 0 two bifurcations 4 px apart,
    1 a bifurcation with a crossing through it, 2 three vessels crossing at
    one point, 3 a trifurcation, 4 a crossing 6 px from a bifurcation."""
    c = np.array([T / 2, T / 2])
    _v = _vessel
    _vessel_k = lambda xy, r0, r1, blur, amp, rng: _v(xy, r0 * k, r1 * k, blur, amp, rng)
    ray = lambda ang, L=75, p=c: _path([p, p + L / 2 * np.array([math.cos(ang), math.sin(ang)]),
                                        p + L * np.array([math.cos(ang), math.sin(ang)])])
    thru = lambda ang, p=c: _path([p - 75 * np.array([math.cos(ang), math.sin(ang)]), p,
                                   p + 75 * np.array([math.cos(ang), math.sin(ang)])])
    case = int(case)
    if case == 0:
        p2 = c + [4, 0]
        vs = [_vessel_k(ray(math.pi), 2.4, 2.4, 1.0, 0.4, rng), _vessel_k(ray(-0.9), 1.6, 1.4, 1.0, 0.4, rng),
              _vessel_k(ray(0.1, 75, p2), 1.7, 1.5, 1.0, 0.4, rng), _vessel_k(ray(1.0, 75, p2), 1.2, 1.1, 1.0, 0.35, rng)]
    elif case == 1:
        vs = [_vessel_k(ray(math.pi), 2.2, 2.2, 1.0, 0.4, rng), _vessel_k(ray(-0.5), 1.6, 1.4, 1.0, 0.4, rng),
              _vessel_k(ray(0.6), 1.6, 1.4, 1.0, 0.4, rng), _vessel_k(thru(1.5), 1.0, 1.0, 1.0, 0.35, rng)]
    elif case == 2:
        vs = [_vessel_k(thru(a), 1.2, 1.2, 1.0, 0.35, rng) for a in (0.0, 1.05, 2.1)]
    elif case == 3:
        vs = [_vessel_k(ray(math.pi), 2.4, 2.4, 1.0, 0.4, rng)] + \
             [_vessel_k(ray(a), 1.5, 1.3, 1.0, 0.4, rng) for a in (-0.8, 0.0, 0.8)]
    else:
        p2 = c + [6, 0]
        vs = [_vessel_k(ray(math.pi), 2.2, 2.2, 1.0, 0.4, rng), _vessel_k(ray(-0.6), 1.6, 1.4, 1.0, 0.4, rng),
              _vessel_k(ray(0.6), 1.6, 1.4, 1.0, 0.4, rng), _vessel_k(thru(1.4, p2), 1.0, 1.0, 1.0, 0.35, rng)]
    return vs, [c]


def _crossings(vessels, tol=1.0):
    """Where the centrelines of two vessels cross or touch (within tol px):
    one point per contiguous contact."""
    from scipy.spatial import cKDTree
    out = []
    for i in range(len(vessels)):
        for j in range(i + 1, len(vessels)):
            a, b = vessels[i]["xy"], vessels[j]["xy"]
            d, _ = cKDTree(b).query(a)
            hit = np.flatnonzero(d <= tol)
            if not len(hit):
                continue
            for run in np.split(hit, np.flatnonzero(np.diff(hit) > 2) + 1):
                out.append(a[run[len(run) // 2]])
    return out


def twisted(T, n, rng, ra=2.0, rb=1.0):
    """Two vessels (radii ra, rb) winding round each other, crossing n times."""
    x = np.arange(4.0, T - 4.0, 1.0)
    L = x[-1] - x[0]
    y = 9.0 * np.cos(math.pi * (x - x[0]) * n / L)      # n zeros inside, ends apart
    a = np.stack([x, T / 2 + y], 1)
    b = np.stack([x, T / 2 - y], 1)
    vs = [_vessel(a, ra, ra, 1.0, 0.4, rng), _vessel(b, rb, rb, 1.0, 0.35, rng)]
    return vs, [np.array([x[0] + L * (q + 0.5) / n, T / 2]) for q in range(int(n))]


def weave(T, n, rng, r_host=3.0, r_cap=0.9):
    """A capillary weaving across a straight thick vessel, crossing it n times."""
    x = np.arange(4.0, T - 4.0, 1.0)
    L = x[-1] - x[0]
    host = np.stack([x, np.full_like(x, T / 2)], 1)
    cap = np.stack([x, T / 2 + 12.0 * np.cos(math.pi * (x - x[0]) * n / L)], 1)
    vs = [_vessel(host, r_host, r_host, 1.0, 0.4, rng), _vessel(cap, r_cap, r_cap, 1.0, 0.3, rng)]
    return vs, [np.array([x[0] + L * (q + 0.5) / n, T / 2]) for q in range(int(n))]


def ladder(T, k, rng, radii=(0.8, 3.5, 1.5, 4.5, 1.0, 2.5)):
    """One vessel crossing k parallel vessels of different thicknesses."""
    k = int(k)
    xs = np.linspace(T * 0.15, T * 0.85, k)
    vs = [_vessel(_path([[x0, 4], [x0 + 3, T / 2], [x0, T - 4]]), radii[q % len(radii)],
                  radii[q % len(radii)], 1.0, 0.4, rng) for q, x0 in enumerate(xs)]
    a = math.radians(20)
    cross = _path([[4, T / 2 - (T / 2 - 4) * math.tan(a)], [T / 2, T / 2],
                   [T - 4, T / 2 + (T / 2 - 4) * math.tan(a)]])
    vs.append(_vessel(cross, 1.4, 1.4, 1.0, 0.35, rng))
    return vs, _crossings(vs)


def mesh(T, case, rng, radii=(0.8, 1.2, 2.0, 3.0, 4.0)):
    """Straight vessels crossing each other: 0/1/2 a grid of 2/3/4 lines each
    way, 3/4 lines in three directions (2/3 each), radii drawn from radii."""
    case = int(case)
    c = np.array([T / 2, T / 2])
    if case <= 2:
        dirs, per = (0.15, 0.15 + math.pi / 2), case + 2
    else:
        dirs, per = (0.1, 0.1 + math.pi / 3, 0.1 + 2 * math.pi / 3), case - 1
    vs = []
    for a in dirs:
        d = np.array([math.cos(a), math.sin(a)])
        nrm = np.array([-d[1], d[0]])
        for off in np.linspace(-(per - 1) / 2, (per - 1) / 2, per) * (60.0 / max(per, 2)):
            p = c + nrm * off
            r = float(rng.choice(radii))
            vs.append(_vessel(_path([p - 90 * d, p, p + 90 * d]), r, r, 1.0, 0.35, rng))
    return vs, _crossings(vs)


from functools import partial as _p

ROWS = [
    ("bifurcation, branch angle (deg)", bifurcation, (40, 60, 90, 120, 150)),
    ("bifurcation, Murray share of the thicker child", bifurcation_asym, (0.5, 0.6, 0.7, 0.8, 0.9)),
    ("crossing, same depth, angle (deg)", crossing, (10, 20, 35, 60, 90)),
    ("crossing, capillary over blurred vessel, angle", crossing_depth, (10, 20, 35, 60, 90)),
    ("parallel pair, centre separation (px)", parallel, (2.4, 3.2, 4.2, 5.5, 7.5)),
    ("kissing pair, closest lumen gap (px)", kissing, (-0.5, 0.0, 0.5, 1.0, 2.0)),
    ("vessel ending on another, angle (deg)", ends_on, (30, 50, 70, 90, 110)),
    ("hairpin, turn radius (px)", hairpin, (2.0, 3.0, 4.0, 6.0, 9.0)),
    ("fork then side by side, separation (px)", fork_parallel, (3.0, 4.0, 5.0, 6.5, 8.0)),
    ("faint capillary under blurred vessel, contrast", faint_under_blurred, (0.06, 0.1, 0.15, 0.2, 0.3)),
    ("complex node (see complex_node)", complex_node, (0, 1, 2, 3, 4)),
]

# the same structures over the range of calibres, and thin with thick
ROWS_CALIBRE = [
    ("thin bifurcation (parent r 1.2), branch angle", _p(bifurcation, r0=1.2), (40, 60, 90, 120, 150)),
    ("thick bifurcation (parent r 5.5), branch angle", _p(bifurcation, r0=5.5), (40, 60, 90, 120, 150)),
    ("thin crossing (r 0.9 and 0.7), angle", _p(crossing, ra=0.9, rb=0.7), (10, 20, 35, 60, 90)),
    ("thick crossing (r 4.5 and 3.5), angle", _p(crossing, ra=4.5, rb=3.5), (10, 20, 35, 60, 90)),
    ("thin over thick, same depth (r 0.9 / 4.5), angle", _p(crossing, ra=4.5, rb=0.9), (10, 20, 35, 60, 90)),
    ("thin parallel pair (r 0.8), separation / (r1 + r2)", _p(parallel_rel, r=0.8), (1.0, 1.3, 1.7, 2.2, 3.0)),
    ("thick parallel pair (r 3), separation / (r1 + r2)", _p(parallel_rel, r=3.0), (1.0, 1.3, 1.7, 2.2, 3.0)),
    ("thick kissing pair (r 3.5), closest lumen gap", _p(kissing, r=3.5), (-1.0, 0.0, 1.0, 2.0, 4.0)),
    ("thick vessel ending on a thick one (r 2 / 4.5), angle", _p(ends_on, r_h=4.5, r_a=2.0), (30, 50, 70, 90, 110)),
    ("thick hairpin (r 1.6), turn radius", _p(hairpin, r=1.6), (3.0, 4.0, 5.0, 7.0, 10.0)),
    ("thick fork then side by side (parent r 4), separation", _p(fork_parallel, r0=4.0), (6.0, 8.0, 10.0, 13.0, 16.0)),
    ("thick complex node (radii x 2)", _p(complex_node, k=2.0), (0, 1, 2, 3, 4)),
]

# several crossings in one place
ROWS_CROSSINGS = [
    ("twisted pair (r 2 and 1), crossings", twisted, (1, 2, 3, 4, 6)),
    ("twisted thin pair (r 0.9 and 0.7), crossings", _p(twisted, ra=0.9, rb=0.7), (1, 2, 3, 4, 6)),
    ("capillary weaving over a thick vessel (r 0.9 / 3), crossings", weave, (1, 2, 3, 4, 5)),
    ("ladder: one vessel across k vessels of r 0.8-4.5, k", ladder, (2, 3, 4, 5, 6)),
    ("mesh: grids 2x2 / 3x3 / 4x4, three directions x2 / x3", mesh, (0, 1, 2, 3, 4)),
]

ALL_ROWS = ROWS + ROWS_CALIBRE + ROWS_CROSSINGS


def _clip(v, box):
    """The part of a vessel inside box (x0, y0, x1, y1): the longest run."""
    x0, y0, x1, y1 = box
    xy = v["xy"]
    ok = (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
    if not ok.any():
        return None
    idx = np.flatnonzero(ok)
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    run = max(runs, key=len)
    if len(run) < 6:
        return None
    return dict(v, xy=xy[run], r=v["r"][run],
                amp=v["amp"][run] if np.ndim(v["amp"]) else v["amp"])


def zoo_sheet(seed=0, tile=128, gap=8, rows=None, texture_scale=1.0, noise=True):
    """One tile per case, a row per structure with its parameter swept.
    Returns (image, vessels, tiles): tiles is a list of dict(row, name,
    value, box=(x0, y0, x1, y1), vessels=[indices], ambiguous=[points])."""
    rng = np.random.default_rng(seed)
    rows = ROWS if rows is None else rows
    ncol = max(len(r[2]) for r in rows)
    H = len(rows) * (tile + gap) + gap
    W = ncol * (tile + gap) + gap
    vessels, tiles = [], []
    for i, (name, fn, values) in enumerate(rows):
        for j, val in enumerate(values):
            x0, y0 = gap + j * (tile + gap), gap + i * (tile + gap)
            box = (x0, y0, x0 + tile - 1, y0 + tile - 1)
            vs, amb = fn(tile, val, rng)
            ids = []
            for v in vs:
                v = dict(v, xy=v["xy"] + [x0, y0])
                v = _clip(v, box)
                if v is not None:
                    ids.append(len(vessels))
                    vessels.append(v)
            amb = [np.asarray(p, float) + [x0, y0] for p in amb]
            amb = [p for p in amb if x0 <= p[0] <= box[2] and y0 <= p[1] <= box[3]]
            tiles.append(dict(row=i, name=name, value=val, box=box, vessels=ids, ambiguous=amb))
    tex = texture((H, W), rng, texture_scale)
    I, od = render(vessels, (H, W), rng, texture=tex, noise=noise)
    return I, vessels, tiles


# ---------------------------------------------------------------- completeness
def _bilinear(M, x, y):
    H, W = M.shape
    x = np.clip(x, 0, W - 1.001)
    y = np.clip(y, 0, H - 1.001)
    x0, y0 = x.astype(int), y.astype(int)
    fx, fy = x - x0, y - y0
    return ((1 - fx) * (1 - fy) * M[y0, x0] + fx * (1 - fy) * M[y0, x0 + 1] +
            (1 - fx) * fy * M[y0 + 1, x0] + fx * fy * M[y0 + 1, x0 + 1])


def completeness(logI, sigma, xy, r, stretch=12.0, z_gap=4.0, flank=4.0, du=0.5):
    """Whether an annotated vessel (centreline xy 1 px apart, radius r per
    point) is one vessel along its length.

    Along every stretch of `stretch` px the image's cross-section is
    averaged, darkness taken against a line through the two flanks just
    outside the annotated lumen. Within the lumen (|u| <= r), a stretch
    whose darkness has two peaks with a gap between them brighter than the
    weaker peak by more than z_gap noise levels (per-pixel sensor sigma over
    the root of the stretch length) is two vessels annotated as one. Where
    two vessels are too close to show a gap, no annotation could tell.

    Returns dict(score = fraction of stretches without a gap, gap_z per
    stretch, worst = the largest gap_z)."""
    xy = np.asarray(xy, float)
    n = len(xy)
    t = np.gradient(xy, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    nrm = np.stack([-t[:, 1], t[:, 0]], 1)
    gaps = []
    step = max(1, int(round(stretch)))
    for a in range(0, max(1, n - step // 2), step):
        sl = slice(a, min(n, a + step))
        R = float(np.median(r[sl]))
        span = R + 0.5
        u = np.arange(-(span + flank), span + flank + 1e-9, du)
        P = xy[sl, None, :] + u[None, :, None] * nrm[sl, None, :]
        prof = _bilinear(logI, P[..., 0], P[..., 1]).mean(0)
        sig = float(np.median(_bilinear(sigma, xy[sl, 0], xy[sl, 1]))) / math.sqrt(max(1, sl.stop - sl.start))
        left, right = u < -span, u > span
        ul, ur = u[left].mean(), u[right].mean()
        bl, br = prof[left].mean(), prof[right].mean()
        base = bl + (br - bl) * (u - ul) / (ur - ul)
        D = base - prof                                   # darkness
        inside = np.abs(u) <= span
        Di, ui = D[inside], u[inside]
        k = int(np.argmax(Di))
        best = 0.0
        # a second peak on either side of the darkest, with a brighter gap between
        for side in (range(k + 1, len(Di)), range(k - 1, -1, -1)):
            lo = Di[k]
            for m in side:
                lo = min(lo, Di[m])
                if abs(ui[m] - ui[k]) < 1.5:
                    continue
                best = max(best, (min(Di[k], Di[m]) - lo) / (sig + 1e-12))
        gaps.append(best)
    gaps = np.array(gaps)
    return dict(score=float((gaps < z_gap).mean()) if len(gaps) else 1.0,
                gap_z=gaps, worst=float(gaps.max()) if len(gaps) else 0.0)
