"""Where vessels meet: annotate the intersections, and hand them on.

Everything upstream measures a vessel ALONG its length, and everything
upstream is wrong where two vessels meet. The width there is not measurable -
the cross-profile of a thin vessel crossing a thick one runs along the thick
one and measures it - and 34 % of detected centreline on a sharp crop runs
through or beside another vessel. Rather than patch each measurement to cope,
this marks those places out as regions, says what kind each one is, and leaves
them for an analysis that can afford to be slower and cleverer.

A region is found geometrically, from centrelines and their measured radii
alone, so it works on any vessel set - the staged network, `large.detect`, or
a hand-traced one. Nothing here depends on how the vessels were found.

The classification counts ARMS: the directions in which vessel runs leave the
region. Counting arms rather than trusting which vessel object a piece belongs
to is what makes this independent of the detector's own linking decisions,
which at a junction are exactly what is in doubt.

    extended  overlap     two vessels running side by side, lumens touching,
                          over a stretch much longer than they are wide
    2 arms    touch        two vessels graze, or one passes close by
    3 arms    bifurcation  checked against Murray's law, r_p^3 = r_1^3 + r_2^3
    4 arms    crossing     checked by whether the arms pair off back-to-back
    other     unresolved   too tangled to call from geometry
"""
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Region:
    p: np.ndarray                  # centre (x, y)
    extent: float                  # radius of the disc that covers it
    kind: str = "unresolved"
    arms: list = field(default_factory=list)      # dicts: vessel, heading, radius
    members: list = field(default_factory=list)   # vessel indices involved
    angle_deg: float = float("nan")               # crossing angle, or branch angle
    murray_residual: float = float("nan")
    pairs: list = field(default_factory=list)     # for a crossing, which arms continue which

    @property
    def xy(self):
        return float(self.p[0]), float(self.p[1])


def _as_pairs(vessels):
    """Accept (C, r), (C, r, meta) or a staged vessel dict."""
    out = []
    for v in vessels:
        if isinstance(v, dict):
            C = np.asarray(v["centreline"], float)
            r = v.get("radius_profile")
            r = (np.full(len(C), float(v.get("radius_px", 2.0))) if r is None
                 else np.asarray(r, float))
            if len(r) != len(C):
                r = np.interp(np.linspace(0, 1, len(C)), np.linspace(0, 1, len(r)), r)
        else:
            C = np.asarray(v[0], float)
            r = np.asarray(v[1], float)
        if len(C) >= 2:
            out.append((C, r))
    return out


def find(vessels, near=1.0, min_sep=8.0, log=None):
    """Regions where vessels meet. `near` scales the contact distance, which is
    the sum of the two local radii - i.e. the lumens touch."""
    V = _as_pairs(vessels)
    hits = []
    for i in range(len(V)):
        Ci, ri = V[i]
        for j in range(i + 1, len(V)):
            Cj, rj = V[j]
            # coarse reject before the full distance matrix
            if (Ci[:, 0].min() > Cj[:, 0].max() + 60 or Cj[:, 0].min() > Ci[:, 0].max() + 60
                    or Ci[:, 1].min() > Cj[:, 1].max() + 60 or Cj[:, 1].min() > Ci[:, 1].max() + 60):
                continue
            D = np.linalg.norm(Ci[:, None, :] - Cj[None, :, :], axis=2)
            thr = near * (ri[:, None] + rj[None, :])
            a, b = np.nonzero(D <= thr)
            for k in range(len(a)):
                hits.append((0.5 * (Ci[a[k]] + Cj[b[k]]),
                             float(ri[a[k]] + rj[b[k]]), i, j))
    if log:
        log(f"[regions] {len(hits)} contact points between {len(V)} vessels")
    if not hits:
        return []

    # cluster contacts into regions: single-linkage on distance, which is right
    # here because one junction really is one connected blob of contact
    P = np.array([h[0] for h in hits])
    lab = -np.ones(len(P), int)
    nxt = 0
    for i in range(len(P)):
        if lab[i] >= 0:
            continue
        lab[i] = nxt
        stack = [i]
        while stack:
            k = stack.pop()
            d = np.linalg.norm(P - P[k], axis=1)
            for q in np.flatnonzero((d <= min_sep) & (lab < 0)):
                lab[q] = nxt
                stack.append(int(q))
        nxt += 1

    regions = []
    for g in range(nxt):
        sel = np.flatnonzero(lab == g)
        centre = P[sel].mean(0)
        spread = float(np.linalg.norm(P[sel] - centre, axis=1).max()) if len(sel) > 1 else 0.0
        extent = max(spread, max(hits[k][1] for k in sel) * 0.5)
        members = sorted({hits[k][2] for k in sel} | {hits[k][3] for k in sel})
        regions.append(_describe(Region(p=centre, extent=extent, members=members),
                                 V, cloud=P[sel],
                                 width=float(np.mean([hits[k][1] for k in sel]))))
    if log:
        kinds = {}
        for r in regions:
            kinds[r.kind] = kinds.get(r.kind, 0) + 1
        log(f"[regions] {len(regions)} regions: "
            + ", ".join(f"{v} {k}" for k, v in sorted(kinds.items(), key=lambda t: -t[1])))
    return regions


def _describe(reg, V, cloud=None, width=0.0, out=1.6, elong=2.5):
    """Count the arms leaving the region and decide what it is.

    Elongation alone cannot tell a shallow crossing from a parallel pair: at
    20 degrees two vessels stay in contact for a long stretch, so the contact
    cloud is just as elongated as two vessels running side by side, and a rule
    based on shape called four of these wrong. What separates them is
    topology, not shape. Going round the region, a crossing's arms ALTERNATE
    between the two vessels - A, B, A, B - because each vessel passes through
    and comes out the far side. A parallel pair gives A, B, B, A: both leave
    together at each end. That holds at any angle.
    """
    major = 0.0
    if cloud is not None and len(cloud) >= 4:
        c = cloud - cloud.mean(0)
        axis = _unit(np.linalg.svd(c, full_matrices=False)[2][0])
        proj = c @ axis
        major = float(proj.max() - proj.min())
    stretched = major > elong * max(width, 1.0)
    # look for arms beyond the whole extent of the contact, not just beyond a
    # disc round its centre, or an elongated region swallows both vessels whole
    R = max(reg.extent, major / 2) * out
    arms = []
    for i in reg.members:
        C, r = V[i]
        d = np.linalg.norm(C - reg.p, axis=1)
        inside = d <= R
        if not inside.any():
            continue
        idx = np.flatnonzero(inside)
        for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            for end, step in ((run[0], -1), (run[-1], +1)):
                if end + step < 0 or end + step >= len(C):
                    continue          # the vessel stops inside: not an arm
                far = min(max(end + step * 12, 0), len(C) - 1)
                h = C[far] - C[end]
                n = np.hypot(*h)
                if n > 1e-6:
                    arms.append({"vessel": i, "heading": h / n,
                                 "radius": float(r[end]), "p": C[end].copy()})
    keep = []
    for a in arms:
        if not any(b["vessel"] == a["vessel"]
                   and float(np.dot(a["heading"], b["heading"])) > 0.94
                   and np.linalg.norm(a["p"] - b["p"]) < R for b in keep):
            keep.append(a)
    reg.arms = keep
    k = len(keep)
    if k == 3:
        reg.kind = "bifurcation"
        rr = sorted((a["radius"] for a in keep), reverse=True)
        murray = (rr[1] ** 3 + rr[2] ** 3) ** (1 / 3)
        reg.murray_residual = float((murray - rr[0]) / max(rr[0], 1e-9))
        reg.angle_deg = float(min(_angle(keep[a]["heading"], keep[b]["heading"])
                                  for a, b in ((0, 1), (0, 2), (1, 2))))
    elif k == 4 and len({a["vessel"] for a in keep}) == 2:
        order = sorted(range(4), key=lambda q: np.arctan2(
            keep[q]["p"][1] - reg.p[1], keep[q]["p"][0] - reg.p[0]))
        lab = [keep[q]["vessel"] for q in order]
        alternates = lab[0] != lab[1] and lab[1] != lab[2] and lab[2] != lab[3]
        if alternates:
            reg.kind = "crossing"
            reg.pairs = [(order[0], order[2]), (order[1], order[3])]
            a = _angle(keep[order[0]]["heading"], keep[order[1]]["heading"])
            reg.angle_deg = min(a, 180 - a)
        else:
            reg.kind = "overlap" if stretched else "touch"
    elif k == 2:
        reg.kind = "overlap" if stretched else "touch"
        reg.angle_deg = _angle(keep[0]["heading"], keep[1]["heading"])
    else:
        reg.kind = "overlap" if stretched else "unresolved"
    if reg.kind == "overlap":
        reg.extent = max(reg.extent, major / 2)
    return reg


def _unit(v):
    n = float(np.hypot(*v[:2]))
    return v[:2] / (n if n > 1e-9 else 1.0)


def _angle(u, v):
    c = float(np.clip(np.dot(u, v), -1, 1))
    return float(np.degrees(np.arccos(c)))


def mask(regions, shape, grow=1.0):
    """int32 image, each region's disc carrying its 1-based index, so a later
    analysis can pull out the pixels belonging to one junction."""
    lab = np.zeros(shape, np.int32)
    for i, r in enumerate(sorted(regions, key=lambda q: -q.extent), 1):
        cv2.circle(lab, (int(round(r.p[0])), int(round(r.p[1]))),
                   max(2, int(round(r.extent * grow))), i, -1)
    return lab


def clean(vessels, regions, shape, grow=1.0):
    """Boolean mask of centreline that does NOT run through an intersection -
    the stretches on which a measurement can be trusted."""
    lab = mask(regions, shape, grow)
    out = []
    for C, _ in _as_pairs(vessels):
        xi = np.clip(np.rint(C[:, 0]).astype(int), 0, shape[1] - 1)
        yi = np.clip(np.rint(C[:, 1]).astype(int), 0, shape[0] - 1)
        out.append(lab[yi, xi] == 0)
    return out
