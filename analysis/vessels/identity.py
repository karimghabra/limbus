"""Making a vessel's identifier mean the same thing twice.

Ids are assigned in network._thread by walking the hierarchy depth-first with
roots sorted by (-calibre, -length), so every id is a GLOBAL RANK. Insert one
vessel, or nudge one calibre past another, and every id after it moves.
Measured on a real crop, against perturbations smaller than the difference
between two halves of one burst:

    identical rerun               31/31 matched   100 % keep their id
    half-pixel shift              15/31 matched     0 %
    noise at one frame's level    15/31 matched     0 %

Not one vessel keeps its id. Checked by hand on the longest vessels: 1 -> 19,
28 -> 30, 22 -> 15, 3 -> 10, 23 -> 31, with centroid movements of 0.0-2.5 px
and lengths unchanged - 399 px -> 399 px in one case. The vessel count moves
too, 31 -> 33 under noise alone.

The consequence is not cosmetic. Every per-vessel measurement downstream is
indexed by the id, so comparing a vessel between two bursts, or before and
after a change to the detector, silently compares two different vessels. It
also means an improvement in resolving power cannot be banked: a change that
splits one fused pair renumbers the network and invalidates the comparison
that would have shown it was an improvement.

The fix is not a better sort. A rank cannot be stable under insertion, and no
ordering rule avoids that. Identity has to come from the vessel itself, and
between two runs it has to be established by MATCHING rather than assumed from
position in a list:

    match(a, b)     pair the vessels of two runs by geometry, one to one
    inherit(a, b)   give the vessels of run b the ids of the vessels of run a
                    they were matched to, minting new ids for the unmatched and
                    recording where a vessel was split or two were merged

so that a burst reprocessed, or a burst compared with its neighbour, carries
the identifiers of a named reference rather than of whatever order this run
happened to produce.

What matching cannot do is invent stability where the detector has none: if a
vessel is found in one run and not the other, no scheme can carry its id
across. The honest output is that it is missing, which `inherit` records.
"""
import numpy as np


def _summary(v):
    C = np.asarray(v["centreline"] if "centreline" in v else v["C"], float)
    L = float(np.hypot(*np.diff(C, axis=0).T).sum()) if len(C) > 1 else 0.0
    return {"C": C, "L": L, "c": C.mean(0) if len(C) else np.zeros(2),
            "r": float(v.get("radius_px") or 0.0)}


def overlap(a, b, tol=3.0):
    """Fraction of a's centreline lying within `tol` of b's.

    Asymmetric on purpose: a vessel split in two overlaps each half almost
    completely one way and about half the other, which is how `inherit` tells
    a split from a match.
    """
    A, B = np.asarray(a, float), np.asarray(b, float)
    if len(A) < 2 or len(B) < 2:
        return 0.0
    step = max(1, len(A) // 400)
    d = np.hypot(*(A[::step, None, :] - B[None, ::max(1, len(B) // 400), :]).transpose(2, 0, 1))
    return float((d.min(1) <= tol).mean())


def match(va, vb, tol=3.0, min_overlap=0.5):
    """One-to-one correspondence between two runs, by geometry alone.

    Greedy on the symmetric overlap, which is what makes it independent of the
    ids it exists to replace. Returns [(i, j, score)], and the unmatched of
    each side.
    """
    A = [_summary(v) for v in va]
    B = [_summary(v) for v in vb]
    cand = []
    for i, x in enumerate(A):
        for j, y in enumerate(B):
            if abs(x["c"][0] - y["c"][0]) > 200 or abs(x["c"][1] - y["c"][1]) > 200:
                continue
            f, g = overlap(x["C"], y["C"], tol), overlap(y["C"], x["C"], tol)
            s = min(f, g)
            if s >= min_overlap:
                cand.append((s, i, j, f, g))
    cand.sort(reverse=True)
    used_a, used_b, out = set(), set(), []
    for s, i, j, f, g in cand:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        out.append((i, j, float(s)))
    return out, [i for i in range(len(A)) if i not in used_a], \
                [j for j in range(len(B)) if j not in used_b]


def inherit(reference, current, tol=3.0, min_overlap=0.5, part_overlap=0.7):
    """Give `current` the identifiers of `reference`.

    Each vessel of `current` that matches one of `reference` takes its id. An
    unmatched vessel that lies mostly along a reference vessel which already
    gave its id away is recorded as a PART of it - `12.a`, `12.b` - rather than
    renumbering anything, because a split is new information about an old
    vessel and not a new vessel. Everything else is minted a fresh id above the
    highest in the reference, so an id is never reused for a different object.

    Returns (ids, report), ids parallel to `current`.
    """
    pairs, only_ref, only_cur = match(reference, current, tol, min_overlap)
    ids = [None] * len(current)
    taken = {}
    for i, j, s in pairs:
        ids[j] = reference[i].get("id", i + 1)
        taken[i] = ids[j]
    ref_C = [_summary(v)["C"] for v in reference]
    parts = {}
    for j in only_cur:
        Cj = _summary(current[j])["C"]
        best = None
        for i, Ci in enumerate(ref_C):
            f = overlap(Cj, Ci, tol)
            if f >= part_overlap and (best is None or f > best[0]):
                best = (f, i)
        if best is not None:
            i = best[1]
            base = reference[i].get("id", i + 1)
            k = parts.setdefault(base, 0)
            parts[base] += 1
            ids[j] = f"{base}.{chr(ord('a') + k)}"
    nxt = max([v.get("id", 0) for v in reference] or [0])
    for j in range(len(current)):
        if ids[j] is None:
            nxt += 1
            ids[j] = nxt
    report = {"matched": len(pairs),
              "split_parts": sum(parts.values()),
              "new": sum(1 for j in range(len(current))
                         if isinstance(ids[j], int) and ids[j] > max(
                             [v.get("id", 0) for v in reference] or [0])),
              "missing": [reference[i].get("id", i + 1) for i in only_ref if i not in taken]}
    return ids, report
