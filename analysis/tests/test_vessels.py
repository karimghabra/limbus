"""Tests for vessel identification. No data files needed: every image here is
rendered from the physical model, so the tests state what the code should do
rather than what it happens to do today.

usage: python analysis/tests/test_vessels.py
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from vessels import evidence as ev  # noqa: E402
from vessels import image as vimg  # noqa: E402
from vessels import join as vjoin  # noqa: E402
from vessels import network as net  # noqa: E402
from vessels import pathopen  # noqa: E402
from vessels import ridges as rg  # noqa: E402

fail = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fail.append(msg)


def tube(shape, C, r, D, psf=1.8):
    """Absorbance of a blurred cylinder of peak absorbance D along C."""
    H, W = shape
    ss = 4
    line = np.full((H * ss, W * ss), 255, np.uint8)
    cv2.polylines(line, [np.rint(C * ss).astype(np.int32)], False, 0, 1)
    d = cv2.distanceTransform(line, cv2.DIST_L2, 5) / ss
    a = D * np.sqrt(np.clip(1 - (d / r) ** 2, 0, None))
    a = cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(a.astype(np.float32), (0, 0), psf)


def texture(shape, seed, amp=0.02, scale=6.0):
    """Blotchy scleral-like texture: correlated, not white noise."""
    rng = np.random.default_rng(seed)
    t = cv2.GaussianBlur(rng.normal(0, 1, shape).astype(np.float32), (0, 0), scale)
    return t / t.std() * amp


# ---- path opening: length in every direction, blobs stay short --------------
B = np.zeros((140, 140), bool)
B[10, 10:60] = True
B[20:90, 120] = True
for i in range(40):
    B[30 + i, 20 + i] = True
B[110:116, 20:26] = True
L = pathopen.path_length(B, gap=0)
check(L[10, 30] == 50, f"horizontal line length 50 (got {L[10, 30]})")
check(L[50, 120] == 70, f"vertical line length 70 (got {L[50, 120]})")
check(L[50, 40] == 40, f"diagonal line length 40 (got {L[50, 40]})")
check(L[113, 23] <= 12, f"a 6x6 blob stays short (got {L[113, 23]})")
B2 = B.copy()
B2[10, 35] = False
check(pathopen.path_length(B2, 0)[10, 20] == 25, "a 1-px gap cuts a line without gap tolerance")
check(pathopen.path_length(B2, 1)[10, 20] == 50, "gap tolerance 1 bridges it")
check(np.array_equal(pathopen.path_length(B.T, 0), L.T), "path length is transpose-symmetric")

# ---- absorbance: a known vessel comes back with its own depth ---------------
H, W = 200, 300
C = np.stack([np.linspace(20, 280, 800), np.full(800, 100.0)], 1)
A_true = tube((H, W), C, 2.0, 0.08)
mean = (1000 * np.exp(-A_true)).astype(np.float32)
A, valid = vimg.prepare(mean)
peak = float(A[98:103, 140:160].max())
true_peak = float(A_true.max())          # the blur lowers a thin vessel's peak
check(abs(peak - true_peak) < 0.005,
      f"flat-fielded absorbance recovers the planted depth ({peak:.3f} vs {true_peak:.3f})")

# ---- evidence and ridges ---------------------------------------------------
scene = A_true + tube((H, W), np.stack([np.full(600, 150.0), np.linspace(20, 180, 600)], 1), 1.5, 0.05)
scene = scene + texture((H, W), 0)
valid = np.ones((H, W), bool)
z, ang = ev.ridge_z(scene, valid, with_angle=True)
check(z[100, 150] > 5 and z[40, 60] < 3, "evidence is high on a vessel and low on texture")
two = tube((H, W), np.stack([np.linspace(20, 280, 800), np.full(800, 80.0)], 1), 1.5, 0.06) + \
      tube((H, W), np.stack([np.linspace(20, 280, 800), np.full(800, 88.0)], 1), 1.5, 0.06) + texture((H, W), 1)
z2, ang2 = ev.ridge_z(two, valid, with_angle=True)
col = ev.nms(z2, ang2)[:, 150] & (z2[:, 150] > 2)
rows = np.flatnonzero(col)
groups = np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1) if rows.size else []
check(len([g for g in groups if 70 <= g.mean() <= 100]) == 2,
      f"two vessels 8 px apart keep separate ridges (got {[g.tolist() for g in groups]})")
cl = rg.detect(z, ang)
check(len(cl) >= 2 and sum(np.hypot(*np.diff(c, axis=0).T).sum() for c in cl) > 350,
      f"both vessels are traced ({len(cl)} pieces)")

# ---- joining: one vessel in two pieces gets one identity -------------------
broken = texture((H, W), 2, amp=0.012)
for seg in (C[:330], C[470:]):
    broken = broken + tube((H, W), seg, 2.0, 0.08)
zb, angb = ev.ridge_z(broken, valid, with_angle=True)
clb = rg.detect(zb, angb)
halves = [i for i, c in enumerate(clb) if np.hypot(*np.diff(c, axis=0).T).sum() > 60]
groups, joins = vjoin.join_ends(clb, zb)
check(len(halves) == 2 and any(all(h in g for h in halves) for g in groups),
      f"the two pieces of one vessel become one vessel ({len(clb)} pieces -> {len(groups)})")
apart = [np.stack([np.linspace(10, 100, 100), np.full(100, 40.0)], 1),
         np.stack([np.linspace(200, 290, 100), np.full(100, 160.0)], 1)]
g2, j2 = vjoin.join_ends(apart, np.zeros_like(zb))
check(len(g2) == 2 and not j2, "distant, misaligned pieces are not joined")

# ---- the whole pipeline on a known scene -----------------------------------
ves, model, _ = net.detect(scene, valid, net.NetConfig())
found = max((v for v in ves), key=lambda v: sum(len(c) for c in v["centrelines"]))
rad = np.median([np.median(R) for (_, R, _, _) in found["pieces"] if R is not None])
depth = max(float(D) for (_, _, D, _) in found["pieces"] if D is not None)
# the optical blur trades radius against depth for thin vessels: the radius
# comes out a few tenths high and the depth low, while their product (the
# absorbance integrated across the vessel) stays accurate
check(abs(rad - 2.0) < 1.2, f"the fitted radius is near the true 2.0 px (got {rad:.2f})")
check(abs(rad * depth - 2.0 * 0.08) < 0.25 * 2.0 * 0.08,
      f"radius x depth is within 25% of the truth (got {rad * depth:.3f} vs {2.0 * 0.08:.3f})")
check(0.4 * 0.08 < depth < 1.6 * 0.08,
      f"the fitted peak absorbance is within a factor 1.6 of the true 0.08 (got {depth:.3f})")
lab = net.labels(ves, scene.shape)
check(lab[100, 150] > 0 and lab[40, 60] == 0, "labels cover the vessel and not the background")

# ---- determinism and empty input -------------------------------------------
v1 = net.detect(scene, valid, net.NetConfig(fit=False))[0]
v2 = net.detect(scene, valid, net.NetConfig(fit=False))[0]
check([v["id"] for v in v1] == [v["id"] for v in v2] and
      all(np.allclose(a["anchor"], b["anchor"]) for a, b in zip(v1, v2)), "ids are deterministic")
anch = [v["anchor"] for v in v1]
check(anch == sorted(anch, key=lambda p: (int(p[1] // 40), p[0])), "ids follow position order")
flat = texture((H, W), 5, amp=0.004)
check(len(net.detect(flat, valid, net.NetConfig())[0]) == 0, "a vessel-free image gives no vessels")

print("\nRESULT:", "FAIL " + "; ".join(fail) if fail else "PASS")
sys.exit(1 if fail else 0)
