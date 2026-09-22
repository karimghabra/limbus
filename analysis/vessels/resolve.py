"""Work out what happens inside one intersection, slowly and on the image.

`regions.py` says where vessels meet and what kind of meeting it is. This says
which arm continues which, for one region at a time, and it is allowed to be
expensive because it runs on the 5-10 places per frame where the cheap rules
gave up rather than on every pixel.

The decision is a matching on the arms: each arm either continues into exactly
one other arm, or continues into none (a branch off a trunk, or a vessel that
truly ends here). The score for pairing two arms has three parts, and they are
separable so their worth can be measured one at a time rather than asserted:

    straightness  how nearly the two arms continue each other. A vessel may
                  bend through a junction but it does not kink. On its own
                  this settles every crossing, bifurcation and three-way
                  meeting that was constructed to test it.
    calibre       a vessel does not change width crossing another vessel. This
                  earns its place: two vessels of 4 px and 2 px that touch and
                  separate are resolved by calibre and NOT by straightness,
                  because there both pairings are equally straight.
    evidence      the absorbance along the curve that would join them. It is
                  WEIGHTED ZERO by default. It carries real information - on
                  its own it resolves the same unequal-calibre case - but on
                  everything measured so far it only wins where calibre has
                  already won, so it is not yet earning its cost. Turn it on
                  with `w` if a case appears that needs it.

And a limit worth stating plainly: two vessels of the SAME calibre that touch
and separate are not resolved by any of the three. Straightness cannot see the
difference because both pairings are straight, calibre cannot because they are
identical, and the image cannot because the pair is unresolved in the frame to
begin with. That is the close-parallel case from the original brief, and it is
not solved here.

The joining curve is a Hermite spline honouring both arms' headings, so the
evidence is sampled along the path a vessel would actually take, not along the
straight line between two points, which at anything but a shallow angle leaves
the vessel entirely.

Nothing here changes a detection. `resolve` returns what it worked out and the
caller decides; `annotate` writes it onto the region.
"""
import itertools

import cv2
import numpy as np


def hermite(p0, t0, p1, t1, n=60, strength=0.5):
    """Cubic Hermite from p0 to p1 leaving along t0 and arriving along t1."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = float(np.hypot(*(p1 - p0))) * strength * 2
    m0, m1 = np.asarray(t0, float) * L, np.asarray(t1, float) * L
    s = np.linspace(0, 1, n)[:, None]
    h00 = 2 * s ** 3 - 3 * s ** 2 + 1
    h10 = s ** 3 - 2 * s ** 2 + s
    h01 = -2 * s ** 3 + 3 * s ** 2
    h11 = s ** 3 - s ** 2
    return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1


def _sample(A, C):
    x = np.clip(C[:, 0], 0, A.shape[1] - 1).astype(np.float32)
    y = np.clip(C[:, 1], 0, A.shape[0] - 1).astype(np.float32)
    return cv2.remap(A, x.reshape(1, -1), y.reshape(1, -1), cv2.INTER_LINEAR).ravel()


W_DEFAULT = (1.0, 1.0, 0.0)     # straightness, calibre, image evidence


def pair_score(A, a, b, background=0.0, w=W_DEFAULT):
    """How well arm `a` continues into arm `b`. Returns (total, parts)."""
    # arms point OUT of the region, so a vessel running a -> b enters against
    # a's heading and leaves along b's
    path = hermite(a["p"], -np.asarray(a["heading"], float),
                   b["p"], np.asarray(b["heading"], float))
    turn = float(np.degrees(np.arccos(np.clip(
        np.dot(-np.asarray(a["heading"], float), np.asarray(b["heading"], float)), -1, 1))))
    straight = float(np.exp(-(turn / 45.0) ** 2))
    ra, rb = float(a["radius"]), float(b["radius"])
    calibre = float(np.exp(-((np.log(max(ra, 1e-3) / max(rb, 1e-3))) / np.log(2.0)) ** 2))
    prof = _sample(A, path) - background
    # Evidence is "does the weakest point on this path stay as dark as the
    # VESSEL", referenced to the vessel's own depth at the two arms. An earlier
    # version referenced it to the path's own maximum, which rewarded a uniform
    # path and so scored the WRONG pairings higher: a short curved path that
    # stays inside the junction blob is more uniform than a straight path
    # running the length of a vessel and through the darker crossing.
    ends = np.r_[_sample(A, np.asarray(a["p"], float)[None, :]),
                 _sample(A, np.asarray(b["p"], float)[None, :])] - background
    depth = float(np.median(ends))
    scale = max(0.5 * (ra + rb), 1.0)
    ev = float(np.clip(np.percentile(prof, 10) / max(depth, 1e-6), 0, 1))
    parts = {"straight": straight, "calibre": calibre, "evidence": ev,
             "turn_deg": turn, "length": float(np.hypot(*(np.asarray(b["p"]) - np.asarray(a["p"])))),
             "scale": scale}
    total = (w[0] * straight + w[1] * calibre + w[2] * ev) / max(sum(w), 1e-9)
    return total, parts


def _matchings(n):
    """Every way to pair up n arms, leaving any number unpaired."""
    items = list(range(n))

    def rec(rest):
        if not rest:
            yield []
            return
        a = rest[0]
        yield from ([*m] for m in rec(rest[1:]))            # a unpaired
        for i in range(1, len(rest)):
            b = rest[i]
            tail = rest[1:i] + rest[i + 1:]
            for m in rec(tail):
                yield [(a, b), *m]
    return rec(items)


def resolve(A, region, background=None, w=W_DEFAULT, min_pair=0.65,
            max_arms=7, log=None):
    """Decide which arms of one region continue each other.

    `min_pair` is the score a pairing must reach to be worth making at all.
    Below it the arms are left loose, which is the right answer at a
    bifurcation and at a vessel that genuinely ends here.
    """
    arms = region.arms
    n = len(arms)
    out = {"pairs": [], "scores": [], "unpaired": list(range(n)), "n_through": 0}
    if n < 2 or n > max_arms:
        return out
    if background is None:
        background = float(np.median(A))
    S = {}
    for i, j in itertools.combinations(range(n), 2):
        S[(i, j)] = pair_score(A, arms[i], arms[j], background, w)
    best, best_m = None, []
    for m in _matchings(n):
        keys = [tuple(sorted(p)) for p in m]
        if any(S[k][0] < min_pair for k in keys):
            continue
        tot = sum(S[k][0] for k in keys)
        if best is None or tot > best:
            best, best_m = tot, m
    out["pairs"] = [tuple(sorted(p)) for p in best_m]
    out["scores"] = [S[p][1] | {"score": round(S[p][0], 3)} for p in out["pairs"]]
    out["unpaired"] = [i for i in range(n) if not any(i in p for p in out["pairs"])]
    out["n_through"] = len(out["pairs"])
    if log:
        log(f"[resolve] {n} arms -> {len(out['pairs'])} through-vessels, "
            f"{len(out['unpaired'])} loose: "
            + ", ".join(f"{p[0]}-{p[1]} ({s['score']:.2f}, turn {s['turn_deg']:.0f} deg)"
                        for p, s in zip(out["pairs"], out["scores"])))
    return out


def annotate(A, regions, kinds=("unresolved", "crossing", "overlap"), log=None, **kw):
    """Run `resolve` on the regions worth the time and write it onto them."""
    done = 0
    for r in regions:
        if r.kind not in kinds:
            continue
        r.resolved = resolve(A, r, log=log, **kw)
        r.pairs = r.resolved["pairs"]
        done += 1
    if log:
        log(f"[resolve] {done} regions resolved")
    return regions
