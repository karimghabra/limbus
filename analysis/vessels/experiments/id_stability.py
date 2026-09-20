"""Do vessel identifiers mean the same thing twice?

The stage exists to give each vessel an id that means the same thing in every
frame of the burst and on every rerun. The test suite checks that two runs on
the SAME array agree, which is reproducibility, not stability. What the
downstream work needs is that a vessel which is geometrically the same object
still carries the same id after a perturbation no larger than the difference
between two halves of one burst.

It does not. Ids are assigned in network._thread by enumerating a depth-first
order over the hierarchy, with roots sorted by (-calibre, -length), so every
id is a GLOBAL RANK: insert one vessel, or nudge one calibre past another,
and every id after it moves. Measured on a sharp crop:

    perturbation                 matched   same id   same label
    identical rerun                          93 %       93 %
    half-pixel shift              27/31        0 %        4 %
    noise at one frame's level    25/31        0 %        0 %

Checked again with a stricter matcher on the longest vessels: 1 -> 19,
28 -> 30, 22 -> 15, 3 -> 10, 23 -> 31, with centroid movements of 0.0-2.5 px
and lengths unchanged (399 -> 399 px in one case). The vessel count is not
stable either - 31 became 33 under noise alone.

The consequence is that any change which splits or adds a vessel renumbers the
network, so an improvement in resolving power cannot be banked until ids are
content-addressed: a geometric fingerprint in a burst-independent frame, with
an explicit matching stage between runs, and a split reported as provenance
(id -> id.a, id.b) rather than as a renumbering.

usage: python -m vessels.experiments.id_stability <mean_stabilized.tif>
"""
import sys

import cv2
import numpy as np
import tifffile

from .. import image as vimg
from .. import network as net

NOISE = 0.0015           # per-pixel noise of an averaged frame, measured


def detect(A, valid):
    ves, _, _ = net.detect(A, valid, net.NetConfig())
    out = []
    for x in ves:
        C = np.asarray(x["centreline"], float)
        out.append({"id": x["id"], "label": x.get("label"), "C": C,
                    "L": float(np.hypot(*np.diff(C, axis=0).T).sum()),
                    "c": C.mean(0)})
    return out


def correspond(a, b, max_move=25.0, len_ratio=0.5):
    """Which vessel of b is the same object as each vessel of a.

    Greedy on centroid distance, and only where the two also agree in length -
    so the correspondence does not depend on the ids it is there to test.
    """
    used, out = set(), []
    for x in sorted(a, key=lambda z: -z["L"]):
        best = None
        for j, y in enumerate(b):
            if j in used:
                continue
            d = float(np.hypot(*(x["c"] - y["c"])))
            if d <= max_move and abs(x["L"] - y["L"]) <= len_ratio * max(x["L"], y["L"]):
                if best is None or d < best[0]:
                    best = (d, j, y)
        if best is None:
            out.append((x, None, None))
        else:
            used.add(best[1])
            out.append((x, best[2], best[0]))
    return out


def report(A, valid, log=print):
    base = detect(A, valid)
    rng = np.random.default_rng(0)
    H, W = A.shape
    M = np.float32([[1, 0, 0.5], [0, 1, 0.5]])
    cases = {
        "identical rerun": A,
        "half-pixel shift": cv2.warpAffine(A, M, (W, H), flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_REFLECT),
        "noise at one frame's level": A + rng.normal(0, NOISE, A.shape).astype(np.float32),
    }
    log(f"{len(base)} vessels in the reference run")
    log(f"{'perturbation':>28} {'vessels':>8} {'matched':>8} {'same id':>8} {'same label':>11}")
    rows = {}
    for name, A2 in cases.items():
        other = detect(A2, valid)
        pairs = [(x, y) for x, y, _ in correspond(base, other) if y is not None]
        same = sum(1 for x, y in pairs if x["id"] == y["id"])
        samel = sum(1 for x, y in pairs if x["label"] and x["label"] == y["label"])
        n = max(len(pairs), 1)
        rows[name] = (len(other), len(pairs), same / n, samel / n)
        log(f"{name:>28} {len(other):8d} {len(pairs):3d}/{len(base):<4d} "
            f"{100 * same / n:7.0f}% {100 * samel / n:10.0f}%")
    return rows


if __name__ == "__main__":
    src = sys.argv[1]
    mean = tifffile.imread(src).astype(np.float32)
    if len(sys.argv) > 5:
        x0, y0, x1, y1 = (int(v) for v in sys.argv[2:6])
        mean = mean[y0:y1, x0:x1]
    A, valid = vimg.prepare(mean)
    report(A, valid)
