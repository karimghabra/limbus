"""An ensemble of walkers, propagated in expectation rather than sampled.

The transition probabilities are a deterministic function of the image, so the
ensemble's visitation density is an expectation that can be computed directly.
Sampling walks would only be a noisy estimate of the same quantity, and would
cost reproducibility for nothing.

The state is POSITION AND DIRECTION. A walker that only knows where it is
diffuses isotropically and a vessel is indistinguishable from a smudge; a
walker that also knows which way it is going can be made to prefer continuing,
which is what makes a vessel - a structure that persists in one direction -
accumulate mass while texture does not.

    mass_{t+1}(x', d') = sum over d of mass_t(x, d) * P(d -> d') * survive(x', d')

where x' = x + step * d', turns are limited to neighbouring directions, and
survive() is the local evidence that a vessel runs through x' along d'. Mass
that reaches unsupported ground dies rather than spreading.

Seeds inject mass at the ends of the fragments we are sure of, pointing the
way they were going. After enough steps the accumulated visitation is highest
along the vessels those seeds belong to, and its dominant ridges are the
network - including branches, because mass divides wherever the evidence
supports two continuations, which is what an extend-only tracker cannot do.
"""
import sys

import cv2
import numpy as np

sys.path.insert(0, r"E:\Conjunctiva Code\limbus\analysis")
sys.path[:0] = [".", ".."]
from vessels import evidence as ev  # noqa: E402


def direction_field(A, valid, sigmas, n_dir=16, sharpness=4.0):
    """support[d] = how strongly a vessel runs along direction d at each pixel.

    The ridge response says how vessel-like a pixel is and which way the ridge
    runs; this turns that into a per-direction field by weighting the strength
    with how well each direction aligns with the ridge.
    """
    z, ang, scale = ev.ridge_z(A, valid, sigmas, with_angle=True, with_scale=True)
    ridge_dir = ang + np.pi / 2            # the ridge runs across its normal
    out = np.empty((n_dir,) + A.shape, np.float32)
    dirs = np.arange(n_dir) * np.pi / n_dir
    for i, d in enumerate(dirs):
        align = np.abs(np.cos(d - ridge_dir))
        out[i] = np.clip(z, 0, None) * align ** sharpness
    return out, z, scale, dirs


def propagate(support, dirs, seeds, steps=160, step_px=2.0, turn=1, decay=0.995,
              z_floor=1.0, renorm=True):
    """Push the ensemble forward and accumulate where it goes.

    support: (n_dir, H, W) evidence per direction, dirs: their angles.
    seeds: list of (x, y, direction index, mass).
    Returns the visitation map.
    """
    n_dir, H, W = support.shape
    mass = np.zeros_like(support)
    for x, y, di, m in seeds:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < W and 0 <= yi < H:
            mass[di % n_dir, yi, xi] += m
            mass[(di + n_dir // 2) % n_dir, yi, xi] += m      # both senses of travel
    visit = np.zeros((H, W), np.float32)
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    # a walker in direction d moves along d; sample the source pixel it came from
    maps = []
    for d in np.concatenate([dirs, dirs + np.pi]):
        maps.append(((gx - np.cos(d) * step_px).astype(np.float32),
                     (gy - np.sin(d) * step_px).astype(np.float32)))
    full_dirs = np.concatenate([dirs, dirs + np.pi])
    sup2 = np.concatenate([support, support], 0)               # same support both senses
    mass2 = np.concatenate([mass, mass], 0) * 0.5
    total = float(mass2.sum())
    for t in range(steps):
        nxt = np.zeros_like(mass2)
        for i in range(len(full_dirs)):
            mx, my = maps[i]
            for dj in range(-turn, turn + 1):                   # turn by one direction bin
                j = (i + dj) % len(full_dirs)
                if abs(dj) == turn and turn > 0:
                    w = 0.25
                elif dj == 0:
                    w = 0.5
                else:
                    w = 0.25
                src = cv2.remap(mass2[j], mx, my, cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
                nxt[i] += w * src
        nxt *= np.clip(sup2 / max(z_floor, 1e-6), 0, 1.0)       # survive only where supported
        nxt *= decay
        mass2 = nxt
        visit += mass2.sum(0)
        s = float(mass2.sum())
        if s <= 1e-6 * total:
            break
        if renorm and s > 0:
            pass
    return visit


def propagate_bottleneck(support, dirs, seeds, steps=300, step_px=2.0, turn=1, z_scale=6.0):
    """The best path from a seed to every point, scored by its WEAKEST step.

    Summing visits makes distal parts of a vessel exponentially fainter than
    the parts nearest the seed, so one threshold cannot serve both: measured,
    a global cut on the visitation map kept 87-97% precision but reached only
    a third of the network. The bottleneck score does not decay. V(x, d) is
    the best over paths of the weakest support along the path, which asks the
    question that actually matters - is there a way here along which a vessel
    is supported at every step - and is independent of how far away the seed
    is.

    Iterating max-min is monotone and converges; each pass is the same cost as
    a step of the sum-product propagation.
    """
    n_dir, H, W = support.shape
    full = np.concatenate([dirs, dirs + np.pi])
    sup2 = np.concatenate([support, support], 0) / max(z_scale, 1e-6)
    np.clip(sup2, 0, 1.0, out=sup2)
    V = np.zeros_like(sup2)
    for x, y, di, m in seeds:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < W and 0 <= yi < H:
            for k in (di % n_dir, (di % n_dir) + n_dir):
                V[k, yi, xi] = 1.0
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    maps = [((gx - np.cos(d) * step_px).astype(np.float32),
             (gy - np.sin(d) * step_px).astype(np.float32)) for d in full]
    for t in range(steps):
        nxt = V.copy()
        for i in range(len(full)):
            mx, my = maps[i]
            best = None
            for dj in range(-turn, turn + 1):
                j = (i + dj) % len(full)
                src = cv2.remap(V[j], mx, my, cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
                best = src if best is None else np.maximum(best, src)
            np.maximum(nxt[i], np.minimum(best, sup2[i]), out=nxt[i])
        if np.allclose(nxt, V, atol=1e-4):
            V = nxt
            break
        V = nxt
    return V.max(0)


def seeds_from(segments, n_dir=16, mass=1.0, span=8.0):
    """Mass at both ends of every seed fragment, pointing the way it runs."""
    out = []
    for s in segments:
        P = np.asarray(s["points"], float)
        if len(P) < 3:
            continue
        for at_start in (True, False):
            p = P[0] if at_start else P[-1]
            q = P[min(len(P) - 1, 8)] if at_start else P[max(0, len(P) - 9)]
            d = p - q
            ang = np.arctan2(d[1], d[0]) % np.pi
            di = int(round(ang / (np.pi / n_dir))) % n_dir
            out.append((p[0], p[1], di, mass))
    return out
