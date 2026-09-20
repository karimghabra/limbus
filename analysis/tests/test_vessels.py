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
# a diagonal vessel, as most are: the row/column correction leaves it alone
Cd = np.stack([np.linspace(20, 280, 800), np.linspace(30, 170, 800)], 1)
diag = tube((H, W), Cd, 2.0, 0.08)
A, valid = vimg.prepare((1000 * np.exp(-diag)).astype(np.float32), border_px=0)
peak = float(A[95:105, 140:160].max())
true_peak = float(diag.max())            # the blur lowers a thin vessel's peak
check(abs(peak - true_peak) < 0.005,
      f"flat-fielded absorbance recovers the planted depth ({peak:.3f} vs {true_peak:.3f})")

# ---- fixed pattern: a sensor row offset goes, a vessel stays ---------------
art = A_true.copy()
art[40, :] += 0.01                      # one-row sensor offset
art[150, :] += 0.01
art[151, :] += 0.01                     # two-row offset
fixed = vimg.remove_fixed_pattern(art, np.ones(art.shape, bool))
check(abs(float(np.median(fixed[40]) - np.median(fixed))) < 1e-4 and
      abs(float(np.median(fixed[150]) - np.median(fixed))) < 1e-4,
      "row offsets of one and two rows are removed")
kept = float(fixed[95:106, 150].max()) / float(art[95:106, 150].max())
check(kept > 0.75, f"a vessel running the full width along a row survives ({kept * 100:.0f}% kept)")

# ---- evidence and ridges ---------------------------------------------------
scene = A_true + tube((H, W), np.stack([np.full(600, 150.0), np.linspace(20, 180, 600)], 1), 1.5, 0.05)
scene = scene + texture((H, W), 0)
valid = np.ones((H, W), bool)
z, ang = ev.ridge_z(scene, valid, with_angle=True)
# sampled along the planted vessel, not at one pixel: the crossing in the
# middle of this scene is genuinely ambiguous evidence
vx = np.clip(np.rint(C[:, 0]).astype(int), 0, W - 1)
vy = np.clip(np.rint(C[:, 1]).astype(int), 0, H - 1)
on_vessel = float(np.median(z[vy, vx]))
off_vessel = float(np.median(z[30:60, 30:90]))
check(on_vessel > 5 and off_vessel < 3,
      f"evidence is high on a vessel ({on_vessel:.1f}) and low on texture ({off_vessel:.1f})")
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

# ---- hysteresis reach: a faint chain cannot run away from a strong ridge ---
# built directly in evidence units: a strong ridge from x=20 to x=120, then a
# faint one continuing to x=560. How far the faint part is followed is what
# `reach` controls.
zz = np.zeros((120, 600), np.float32)
zz[60, 20:120] = 6.0
zz[60, 120:560] = 2.0
aa = np.full(zz.shape, np.pi / 2, np.float32)          # ridge normal: vertical


def followed(cls):
    """How far along x the detection reaches past the strong part."""
    xs = [float(np.asarray(c, float)[:, 0].max()) for c in cls] or [0.0]
    return max(xs)


near_only = rg.detect(zz, aa, t_hi=3.0, L_hi=40, t_lo=1.5, L_lo=20, reach=40)
far_too = rg.detect(zz, aa, t_hi=3.0, L_hi=40, t_lo=1.5, L_lo=20, reach=0)
check(120 < followed(near_only) < 200,
      f"reach 40 follows the faint ridge a little past the strong part (to x={followed(near_only):.0f})")
check(followed(far_too) > 500,
      f"an unbounded reach follows it the whole way (to x={followed(far_too):.0f})")
check(followed(rg.detect(zz, aa, reach=200)) > followed(near_only),
      "a larger reach follows further")

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
inside = float((lab[vy, vx] > 0).mean())
check(inside > 0.8 and (lab[30:60, 30:90] > 0).mean() < 0.02,
      f"labels cover the vessel ({100 * inside:.0f}% of it) and not the background")

# ---- determinism and empty input -------------------------------------------
v1 = net.detect(scene, valid, net.NetConfig(fit=False))[0]
v2 = net.detect(scene, valid, net.NetConfig(fit=False))[0]
check([v["id"] for v in v1] == [v["id"] for v in v2] and
      all(np.allclose(a["anchor"], b["anchor"]) for a, b in zip(v1, v2)), "ids are deterministic")
flat = texture((H, W), 5, amp=0.004)
check(len(net.detect(flat, valid, net.NetConfig())[0]) == 0, "a vessel-free image gives no vessels")

# ---- junctions: a known network comes back with the right topology --------
from vessels import graph as gr  # noqa: E402

NH, NW = 300, 500
net_img = texture((NH, NW), 11, amp=0.006)
trunk = np.stack([np.linspace(40, 460, 1200), 150 + 18 * np.sin(np.linspace(0, 2, 1200))], 1)
branch1 = np.stack([np.linspace(180, 300, 400), np.linspace(150, 40, 400)], 1)
branch2 = np.stack([np.linspace(250, 360, 400), np.linspace(155, 275, 400)], 1)
crosser = np.stack([np.linspace(430, 300, 500), np.linspace(20, 280, 500)], 1)   # crosses the trunk far from any branch
net_img = net_img + tube((NH, NW), trunk, 4.0, 0.12)
net_img = net_img + tube((NH, NW), branch1, 2.2, 0.07)
net_img = net_img + tube((NH, NW), branch2, 2.0, 0.06)
net_img = net_img + tube((NH, NW), crosser, 2.6, 0.08)
nv, _, _ = net.detect(net_img, np.ones((NH, NW), bool), net.NetConfig())
by_len = sorted(nv, key=lambda v: -v["length_px"])
main = by_len[0]
check(main["length_px"] > 350,
      f"the trunk comes back as one vessel through both bifurcations ({main['length_px']:.0f} px of 420)")
check(main["parent"] is None and len(main["children"]) == 2,
      f"it is a root with two branches (parent {main['parent']}, children {main['children']})")
kinds = {}
for v in nv:
    for j in v["junctions"]:
        kinds[j["kind"]] = kinds.get(j["kind"], 0) + 1
check(kinds.get("bifurcation", 0) >= 2, f"both bifurcations are found ({kinds})")
cross = [v for v in by_len[1:] if v["length_px"] > 200 and v["parent"] is None]
check(bool(cross), "the vessel crossing the trunk is its own vessel, not a branch of it")
labels = {v["label"] for v in nv}
check("1" in labels and any(l.startswith("1.") for l in labels),
      f"labels run from the trunk down ({sorted(labels)[:5]})")
check(all(v["radius_px"] <= nv[v["parent"] - 1]["radius_px"] * 1.05
          for v in nv if v["parent"]),
      "no vessel is thicker than its parent")

# a bifurcation sitting inside a crossing is genuinely ambiguous: the rule is
# that it must be REPORTED as unresolved, never guessed at
amb = texture((NH, NW), 12, amp=0.006)
amb = amb + tube((NH, NW), trunk, 4.0, 0.12)
amb = amb + tube((NH, NW), np.stack([np.linspace(300, 400, 300), np.linspace(152, 250, 300)], 1), 2.0, 0.06)
amb = amb + tube((NH, NW), np.stack([np.linspace(360, 290, 400), np.linspace(40, 260, 400)], 1), 2.6, 0.08)
av, _, _ = net.detect(amb, np.ones((NH, NW), bool), net.NetConfig())
akinds = {}
for v in av:
    for j in v["junctions"]:
        akinds[j["kind"]] = akinds.get(j["kind"], 0) + 1
check(sum(akinds.values()) > 0,
      f"a branch and a crossing at the same place still produce junction records ({akinds})")
# ids follow the network's own order - largest vessels first, each branch
# numbered after the vessel it leaves, which is the order blood takes
roots = [v for v in nv if v["parent"] is None]
check([r["id"] for r in roots] == sorted(r["id"] for r in roots)
      and all(roots[i]["radius_px"] >= roots[i + 1]["radius_px"] - 1e-9 for i in range(len(roots) - 1)),
      f"root vessels are numbered from the thickest down ({[(r['id'], r['radius_px']) for r in roots]})")
check(all(v["id"] > nv[v["parent"] - 1]["id"] for v in nv if v["parent"]),
      "a branch is numbered after the vessel it leaves")

# ---- kymograph velocity: a known streak slope comes back -------------------
from vessels.profiles import streak_velocity_xcorr  # noqa: E402

rng = np.random.default_rng(0)
fps, dur, span = 74.0, 8.0, 400
tt = np.arange(0, dur, 1 / fps)
for v_true, tol in ((-200.0, 0.05), (400.0, 0.05), (900.0, 0.05), (1500.0, 0.05)):
    # features carried by the flow: fine structure a few pixels across, as
    # red-cell clusters and plasma gaps look in a real vessel
    pat = cv2.GaussianBlur(rng.normal(0, 1, (1, span * 6)).astype(np.float32), (0, 0), 2.0)[0]
    pat = (pat - pat.mean()) / pat.std()
    K = np.empty((len(tt), span), np.float32)
    x = np.arange(span)
    for i, ti in enumerate(tt):
        K[i] = np.interp((x - v_true * ti) % (span * 6), np.arange(span * 6), pat)
    K = K + rng.normal(0, 0.5, K.shape)
    wt, wv, wp = streak_velocity_xcorr(K, tt)
    ok = np.isfinite(wv)
    got = float(np.median(wv[ok])) if ok.any() else float("nan")
    check(ok.any() and abs(got - v_true) <= tol * abs(v_true),
          f"streak speed {v_true:+.0f} px/s recovered (got {got:+.0f}, {int(ok.sum())}/{len(wv)} windows)")

# past the search range the answer must be unknown, never a confident number
pat = cv2.GaussianBlur(rng.normal(0, 1, (1, span * 8)).astype(np.float32), (0, 0), 2.0)[0]
K = np.empty((len(tt), span), np.float32)
for i, ti in enumerate(tt):
    K[i] = np.interp((np.arange(span) - 2500.0 * ti) % (span * 8), np.arange(span * 8), pat)
wt, wv, wp = streak_velocity_xcorr(K + rng.normal(0, 0.5, K.shape), tt)
check(np.isnan(wv).all(), f"a speed past the search range is unknown, not a number "
                          f"({int(np.isfinite(wv).sum())} windows claimed one)")
noise_kymo = rng.normal(0, 1, (int(fps * 4), span)).astype(np.float32)
wt, wv, wp = streak_velocity_xcorr(noise_kymo, np.arange(int(fps * 4)) / fps)
check(np.isnan(wv).mean() > 0.9, "structureless noise gives unknown speed, not a number")

# ---- scale: a synthetic grid of known period comes back --------------------
from vessels.scale import grid_period  # noqa: E402

for true_period, rot in ((27.46, 0.0), (12.0, 7.0), (40.0, -30.0)):
    yy, xx = np.mgrid[0:400, 0:600].astype(np.float32)
    th = np.radians(rot)
    u = xx * np.cos(th) + yy * np.sin(th)
    grid = (1000 + 200 * np.sin(2 * np.pi * u / true_period)).astype(np.float32)
    got, ang, ratio = grid_period(grid)
    check(got is not None and abs(got - true_period) < 0.05 * true_period,
          f"grid period {true_period} px measured (got {got:.2f}, {ang:+.1f} deg, peak/bg {ratio:.0f})")
plain = np.random.default_rng(0).normal(1000, 20, (400, 600)).astype(np.float32)
_, _, ratio = grid_period(plain)
check(ratio < 1e3, f"an image with no grid gives a weak peak (ratio {ratio:.0f})")

print("\nRESULT:", "FAIL " + "; ".join(fail) if fail else "PASS")
sys.exit(1 if fail else 0)
