"""A benchmark for the two things the current detector is said to get wrong:
vessels running close together in parallel, and junctions crowded together.

The existing injection-recovery benchmark plants ISOLATED vessels in empty
sclera and asks whether each was found. It cannot see either failure, because
it never plants two vessels near each other and its only metric is coverage -
"is there a detection near this truth line" - which a merged pair, a swapped
crossing and a network shattered into fragments all pass.

So the primitives here are different:

  * scenarios with a controlled geometry - a parallel pair at a known
    separation, a crossing at a known angle, a bifurcation, and clusters of
    junctions within a known window - planted into the absorbance of a real
    averaged frame the same way the old harness plants its singles;

  * point-level NEAREST-TRUTH assignment. Every point of a detected centreline
    is attributed to the truth line it is closest to, so a single line drawn
    down the middle of a close pair is scored as covering BOTH - which is what
    a merge is - instead of scoring as a clean hit on each;

  * metrics that can see continuity and identity, not only position:
      span    the largest fraction of a truth vessel covered by ONE detection
              (1.0 = that vessel came back as a single object)
      frags   how many detections it takes to cover it
      merge   a detection that covers a large part of two different truth
              vessels - a fused pair, or a crossing whose identity swapped

`span` is the metric the walker experiment lacked: pixel overlap said 92 %
while the network was in 88 pieces.
"""
import cv2
import numpy as np


PSF = 1.8


def render_tube(shape, C, r, D, psf=PSF, ss=4):
    """Absorbance of a blurred cylinder along a dense centreline."""
    H, W = shape
    m = np.full((H * ss, W * ss), 255, np.uint8)
    cv2.polylines(m, [np.rint(np.asarray(C, float) * ss).astype(np.int32)], False, 0, 1)
    d = cv2.distanceTransform(m, cv2.DIST_L2, 5) / ss
    a = (D * np.sqrt(np.clip(1 - (d / r) ** 2, 0, None))).astype(np.float32)
    a = cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(a, (0, 0), psf)


# ----------------------------------------------------------------- geometry

def _arc(p0, ang, L, curv=0.0, ds=0.5):
    """A gently curving line from p0 in direction ang."""
    s = np.arange(0, L, ds)
    th = ang + curv * s
    x = p0[0] + np.cumsum(np.cos(th)) * ds
    y = p0[1] + np.cumsum(np.sin(th)) * ds
    return np.stack([x, y], 1)


def _offset(C, d):
    """C pushed sideways by d px (positive = to the left of travel)."""
    t = np.gradient(C, axis=0)
    t /= np.maximum(np.hypot(t[:, 0], t[:, 1])[:, None], 1e-9)
    n = np.stack([-t[:, 1], t[:, 0]], 1)
    return C + n * d


def sc_parallel(rng, W, H, sep, r=2.0, L=160.0, curv=0.0):
    """Two vessels of the same calibre running side by side, `sep` px apart."""
    ang = rng.uniform(0, 2 * np.pi)
    p0 = np.array([W / 2, H / 2])
    mid = _arc(p0, ang, L, curv)
    return [{"C": _offset(mid, +sep / 2), "r": r, "role": "A"},
            {"C": _offset(mid, -sep / 2), "r": r, "role": "B"}]


def sc_cross(rng, W, H, theta, r=2.0, r2=None, L=160.0):
    """Two vessels crossing at `theta` degrees, meeting at the middle."""
    r2 = r if r2 is None else r2
    ang = rng.uniform(0, 2 * np.pi)
    c = np.array([W / 2, H / 2])
    th = np.radians(theta)
    out = []
    for k, (a, rad) in enumerate(((ang, r), (ang + th, r2))):
        start = c - np.array([np.cos(a), np.sin(a)]) * L / 2
        out.append({"C": _arc(start, a, L), "r": rad, "role": "AB"[k]})
    return out


def sc_bifurcation(rng, W, H, theta, r=3.0, L=95.0):
    """A parent splitting into two children, radii by Murray's law."""
    ang = rng.uniform(0, 2 * np.pi)
    c = np.array([W / 2, H / 2])
    rc = r / 2 ** (1 / 3)                      # r_p^3 = r_1^3 + r_2^3, equal children
    par = _arc(c - np.array([np.cos(ang), np.sin(ang)]) * L, ang, L)
    out = [{"C": par, "r": r, "role": "parent"}]
    for s in (+1, -1):
        a = ang + s * np.radians(theta) / 2
        out.append({"C": _arc(c, a, L), "r": rc, "role": "child"})
    return out


def sc_ladder(rng, W, H, n_j, window, r=2.5, L=150.0):
    """`n_j` side branches leaving one trunk inside a `window` px stretch -
    the crowded-junction case, with the branches alternating sides."""
    ang = rng.uniform(0, 2 * np.pi)
    p0 = np.array([W / 2, H / 2]) - np.array([np.cos(ang), np.sin(ang)]) * L / 2
    trunk = _arc(p0, ang, L)
    out = [{"C": trunk, "r": r, "role": "trunk"}]
    at = np.linspace(L / 2 - window / 2, L / 2 + window / 2, n_j)
    for k, s in enumerate(at):
        i = int(s / 0.5)
        side = 1 if k % 2 == 0 else -1
        a = ang + side * np.radians(55)
        out.append({"C": _arc(trunk[i], a, 55.0), "r": r / 2 ** (1 / 3), "role": "branch"})
    return out


def sc_braid(rng, W, H, n, sep, r=2.0, L=140.0):
    """`n` vessels running roughly parallel `sep` apart, each wandering, so
    they converge and separate - the plexus case, without true junctions."""
    ang = rng.uniform(0, 2 * np.pi)
    p0 = np.array([W / 2, H / 2])
    base = _arc(p0, ang, L)
    out = []
    for k in range(n):
        d = (k - (n - 1) / 2) * sep
        C = _offset(base, d)
        s = np.arange(len(C)) * 0.5
        wob = 0.35 * sep * np.sin(s / rng.uniform(40, 70) + rng.uniform(0, 6))
        out.append({"C": _offset(C, wob[:, None]), "r": r, "role": f"v{k}"})
    return out


SCENARIOS = {
    "parallel": [dict(sep=s) for s in (3, 4, 5, 6, 8, 10, 14, 20)],
    "parallel_thick": [dict(sep=s, r=4.0) for s in (6, 8, 10, 12, 16, 24)],
    "cross": [dict(theta=t) for t in (15, 25, 40, 60, 90)],
    "cross_uneven": [dict(theta=t, r=2.0, r2=5.0) for t in (25, 40, 60, 90)],
    "bifurcation": [dict(theta=t) for t in (20, 35, 60, 90)],
    "ladder": [dict(n_j=n, window=w) for n, w in ((2, 60), (3, 60), (3, 30), (4, 40))],
    "braid": [dict(n=n, sep=s) for n, s in ((3, 6), (3, 10), (4, 8), (5, 6))],
}
MAKERS = {"parallel": sc_parallel, "parallel_thick": sc_parallel, "cross": sc_cross,
          "cross_uneven": sc_cross, "bifurcation": sc_bifurcation,
          "ladder": sc_ladder, "braid": sc_braid}


# ------------------------------------------------------------------ planting

def place(free, rng, lines, margin=14):
    """Move a scenario, built around the crop centre, to a free spot.

    Tries a random translation and rotation about the centre until every
    planted point sits on ground that is free of real vessels and inside the
    valid region."""
    H, W = free.shape
    pts = np.vstack([ln["C"] for ln in lines])
    c = np.array([W / 2, H / 2])
    rad = float(np.hypot(*(pts - c).T).max()) + margin
    if 2 * rad >= min(H, W):
        return None
    for _ in range(400):
        a = rng.uniform(0, 2 * np.pi)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        t = np.array([rng.uniform(rad, W - rad), rng.uniform(rad, H - rad)])
        moved = [{**ln, "C": (ln["C"] - c) @ R.T + t} for ln in lines]
        ok = True
        for ln in moved:
            xi = np.rint(ln["C"][:, 0]).astype(int)
            yi = np.rint(ln["C"][:, 1]).astype(int)
            if xi.min() < margin or yi.min() < margin or xi.max() >= W - margin or yi.max() >= H - margin:
                ok = False
                break
            if not free[yi, xi].all():
                ok = False
                break
        if ok:
            return moved
    return None


def paint(shape, lines, depth, psf=PSF):
    """Absorbance of every line in the scenario, each a blurred cylinder whose
    PEAK after blur is `depth` for the thinnest calibre present - thicker ones
    keep the same D, so a wide vessel is genuinely darker, as it is in life."""
    A = np.zeros(shape, np.float32)
    rmin = min(ln["r"] for ln in lines)
    probe = render_tube(shape, lines[0]["C"], rmin, 1.0, psf)
    D = depth / max(probe.max(), 1e-9)
    for ln in lines:
        A += render_tube(shape, ln["C"], ln["r"], D, psf)
    return A


