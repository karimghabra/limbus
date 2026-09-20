"""Model fit: candidate centrelines -> measured vessels.

Every candidate is fitted with the physical model on the averaged frame's
absorbance, strongest first, each against the residual of what is already
accepted. Before fitting, any part of a candidate inside an accepted
vessel's lumen is cut away: that removes double counting where pieces meet
and the echo ridges a fine-scale detector finds along the flanks of a wide
vessel, without fitting them at all.

By default the centreline is held where the ridge detector put it
(max_move=0) and only radius, peak absorbance and the background plane are
fitted - a local fit moves a faint centreline onto texture more often than
it improves it. Acceptance is deliberately lenient (a vessel must be deeper
than min_depth and improve the fit): the detector's length and hysteresis
rules already did the rejecting.

Deterministic: fixed candidate order, fixed multi-start radii, no randomness.
"""
import cv2
import numpy as np

from . import image as img
from . import model as vmodel
from . import ridges


SPACING = 16.0
OFFS = np.arange(-30, 30.5, 0.5, dtype=np.float32)


def resample(C, spacing=SPACING):
    L = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(C, axis=0).T))])
    n = max(2, int(round(L[-1] / spacing)) + 1)
    s = np.linspace(0, L[-1], n)
    return np.stack([np.interp(s, L, C[:, 0]), np.interp(s, L, C[:, 1])], 1), float(L[-1])


def profile_radius(A, C):
    """Half-maximum half-width and peak of the absorbance profile across C,
    averaged along it (robust first guess for radius and depth)."""
    t = np.gradient(C, axis=0)
    t /= np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-9)[:, None]
    n = np.stack([-t[:, 1], t[:, 0]], 1)
    sel = slice(None, None, max(1, len(C) // 12))
    px = (C[sel, None, 0] + n[sel, None, 0] * OFFS[None, :]).astype(np.float32)
    py = (C[sel, None, 1] + n[sel, None, 1] * OFFS[None, :]).astype(np.float32)
    prof = np.median(cv2.remap(A, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), 0)
    c = len(OFFS) // 2
    bg = min(np.median(prof[:20]), np.median(prof[-20:]))
    pk = prof[c - 4:c + 5].max() - bg
    if pk <= 0:
        return 2.0, 0.0
    half = bg + pk / 2
    r = np.flatnonzero(prof[c:] < half)
    l_ = np.flatnonzero(prof[:c + 1][::-1] < half)
    hw = 0.25 * ((r[0] if r.size else 20) + (l_[0] if l_.size else 20))
    return float(max(hw, 1.0)), float(pk)


def split_outside(C, occupied, min_len):
    """Pieces of centreline C (dense) lying outside `occupied`, each >= min_len."""
    H, W = occupied.shape
    xi = np.clip(np.rint(C[:, 0]).astype(int), 0, W - 1)
    yi = np.clip(np.rint(C[:, 1]).astype(int), 0, H - 1)
    free = ~occupied[yi, xi]
    pieces, cur = [], []
    for i, f in enumerate(free):
        if f:
            cur.append(i)
        elif cur:
            pieces.append(cur)
            cur = []
    if cur:
        pieces.append(cur)
    out = []
    for p in pieces:
        seg = C[p[0]:p[-1] + 1]
        if len(seg) > 1 and np.hypot(*np.diff(seg, axis=0).T).sum() >= min_len:
            out.append(seg)
    return out


def densify(C, step=0.5):
    L = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(C, axis=0).T))])
    s = np.arange(0, L[-1], step)
    return np.stack([np.interp(s, L, C[:, 0]), np.interp(s, L, C[:, 1])], 1)


def run(A, valid, candidates, psf=1.8, kappa=10.0, min_depth=0.015, min_len=20.0, max_D=2.5,
        max_move=0.0, merged_occupancy="full", log=None):
    """candidates: list of (strength, centreline). Returns accepted vessels
    [(control_points, radii, D, record)], the model image and rejects."""
    H, W = A.shape
    noise = img.local_noise(np.exp(-A), valid, 25.0)
    weight = np.where(valid, 1.0 / noise ** 2, 0).astype(np.float32)
    model = np.zeros_like(A)
    occupied = np.zeros((H, W), bool)
    accepted, rejected = [], []
    order = sorted(range(len(candidates)), key=lambda i: (-candidates[i][0], float(candidates[i][1][0, 0]),
                                                           float(candidates[i][1][0, 1])))
    for ci in order:
        strength, C0 = candidates[ci]
        for pi, piece in enumerate(split_outside(densify(C0), occupied, min_len)):
            P, L = resample(piece)
            r_prof, pk = profile_radius(A - model, densify(P))
            resid_img = A - model
            best = None
            starts = sorted({round(min(r_prof, 25.0), 2), 1.5, 3.0, 6.0, 12.0})
            for r_start in starts:
                v = vmodel.Vessel(P, r_start, max(pk, 0.005))
                res = vmodel.fit_vessel_cached(v, resid_img, weight, psf, max_move=max_move)
                if best is None or res[1] < best[1][1]:
                    best = (v, res)
            v, (chi_null, chi, box, n_data) = best
            Pf, R, D, _ = v.unpack()
            depth = 1 - np.exp(-D)
            gain = chi_null - chi
            n_par = len(v.p) + 2          # + background slopes
            rec = {"gain": float(gain), "n_par": n_par, "n_data": int(n_data), "chi_null": float(chi_null), "depth": float(depth), "D": float(D),
                   "r_median": float(np.median(R)), "length": L, "strength": float(strength),
                   "cand": int(ci), "piece": int(pi),
                   "mid": [float(Pf[len(Pf) // 2, 0]), float(Pf[len(Pf) // 2, 1])]}
            why = [w for w, c in (("small gain", gain <= kappa * n_par), ("shallow", depth < min_depth),
                                  ("absorbance implausible", D > max_D),
                                  ("shorter than 2 radii", L < 2 * float(np.median(R)))) if c]
            if why:
                rec["why"] = why
                rejected.append((Pf, R, D, rec))
                continue
            x0, y0, x1, y1 = v.box((H, W), psf)
            xs, ys = np.arange(x0, x1, dtype=np.float32), np.arange(y0, y1, dtype=np.float32)
            model[y0:y1, x0:x1] += v.render(xs, ys, psf, with_offset=False)
            # The occupied zone is the lumen plus a margin of a couple of blur
            # widths: a wide vessel throws a ridge along each WALL, which sits
            # just outside the lumen itself, and without the margin those
            # echoes are fitted as vessels in their own right - the radius
            # then comes back as the wall's, not the vessel's.
            lumen = v.render(xs, ys, 0.0, with_offset=False) > 0.25 * D
            # The margin has to cover a wall echo, which sits at the vessel's
            # own radius, without reaching a genuine neighbour: a fixed margin
            # wide enough for a 12 px vessel swallows a thin vessel running
            # 8 px from a thin one. It therefore scales with the radius.
            pad = max(1, int(round(0.2 * float(np.median(R)) + psf)))
            # A fit whose cross-profile DIPS in the middle is describing two
            # vessels as one, and its radius is not a measurement of anything.
            # Two r=2 vessels 5 px apart fit at r=6, 8 px apart at r=7.75, and
            # the occupied zone that follows - lumen plus margin - is then far
            # wider than the separation, so the neighbour is cut away by
            # split_outside before it is ever fitted. The absorbance is still
            # subtracted into the model, so nothing is claimed twice; the
            # ground is simply not declared taken, and the two candidates
            # compete on the residual instead.
            merged = (merged_occupancy != "full"
                      and not ridges.single_peaked(A, vmodel.catmull(Pf)[0]))
            if merged and merged_occupancy == "off":
                rec["merged_profile"] = True
            else:
                if merged:                      # "core": keep the middle only
                    lumen = v.render(xs, ys, 0.0, with_offset=False) > 0.7 * D
                    pad = 1
                    rec["merged_profile"] = True
                occ = occupied[y0:y1, x0:x1]
                occ |= cv2.dilate(lumen.astype(np.uint8),
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1,) * 2)) > 0
            accepted.append((Pf, R, D, rec))
    if log:
        log(f"  {len(candidates)} candidates -> {len(accepted)} vessels accepted, {len(rejected)} pieces rejected")
    return accepted, model, rejected, occupied


def centrelines(accepted):
    return [vmodel.catmull(P)[0] for P, _, _, _ in accepted]
