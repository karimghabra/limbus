"""Tests for the intersection detector (intersections.py) on zoo tiles."""
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from vesselmap import intersections as X
from vesselmap.image import prepare
from vesselmap.zoo import ROWS, ROWS_CROSSINGS, zoo_sheet


def _score(rows, seed=0):
    I, V, tiles = zoo_sheet(seed, rows=rows)
    P = prepare(I)
    dets = X.detect(P.logI, P.sigma, P.valid)
    found = marks = 0
    for t in tiles:
        for p in t["ambiguous"]:
            marks += 1
            found += any(np.linalg.norm(d["xy"] - p) <= 7.0 for d in dets)
    return found, marks, dets, tiles


def test_finds_bifurcations_and_crossings_of_a_ladder():
    found, marks, dets, _ = _score([ROWS[0], ROWS_CROSSINGS[3]])
    assert found >= 0.8 * marks, (found, marks)
    ladder = [d for d in dets if d["kind"] == "crossing"]
    assert len(ladder) >= 8                            # four arms in two straight pairs


def test_few_intersections_on_parallel_pairs_and_hairpins():
    """Ten tiles without an intersection (a hairpin's tight turn may pass
    for one: two legs and the turn)."""
    rows = [r for r in ROWS if r[0].startswith(("parallel", "hairpin"))]
    _, _, dets, _ = _score(rows)
    assert len(dets) <= 3, [(d["xy"], d["kind"]) for d in dets]


def test_arms_at_bifurcations_are_the_three_vessels():
    """At every bifurcation found, the arms are the parent and both
    children, within 20 degrees."""
    from vesselmap.zoo import match_arms, true_arms
    I, V, tiles = zoo_sheet(0, rows=[ROWS[0], ROWS[1]])
    P = prepare(I)
    dets = X.detect(P.logI, P.sigma, P.valid)
    right = 0
    for t in tiles:
        p = t["ambiguous"][0]
        near = [d for d in dets if np.linalg.norm(d["xy"] - p) <= 7.0]
        if near:
            d = min(near, key=lambda d: np.linalg.norm(d["xy"] - p))
            ta = true_arms([V[i] for i in t["vessels"]], p)
            right += match_arms(d["arms"], ta)[0] == len(ta) == len(d["arms"])
    assert right >= 8, right


def test_a_neighbour_alongside_is_not_an_arm():
    """On one vessel of a thick parallel pair the other one, seen across
    the gap, peaks in the ray profile like an arm parting at a small angle;
    traced back it runs past (approach), while the vessel itself arrives."""
    from scipy import ndimage as ndi
    from vesselmap.zoo import ROWS_CALIBRE
    row = [r for r in ROWS_CALIBRE if r[0].startswith("thick parallel")]
    I, V, tiles = zoo_sheet(0, rows=row)
    P = prepare(I)
    D = ndi.gaussian_filter(X.darkness(P.logI), 1.0)
    J, Js, R, Th, Ts, Q = X.maps(P.logI, P.valid)
    t = [t for t in tiles if t["value"] == 1.7][0]
    a, b = (V[i] for i in t["vessels"])
    k = len(a["xy"]) // 2
    p = a["xy"][k]
    along = np.arctan2(*(a["xy"][k + 10] - p)[::-1])
    j = np.argmin(np.linalg.norm(b["xy"] - (p + 18 * np.array([np.cos(along), np.sin(along)])), axis=1))
    across = np.arctan2(*(b["xy"][j] - p)[::-1])      # towards the neighbour, 18 px on
    assert X.approach(D, p, along, 24.0, [across], Th, R) <= 2.0
    assert X.approach(D, p, across, 24.0, [along], Th, R) >= 5.0


def test_shallow_thick_crossings_are_found_as_one_crossing():
    """Thick vessels crossing at 20 degrees overlap over some 40 px; the two
    ends of the overlap, each like a fork, pair up into one crossing near
    the true one, with four arms (at 10 degrees the ends are too far apart
    for a tile)."""
    from vesselmap.zoo import ROWS_CALIBRE
    row = [r for r in ROWS_CALIBRE if r[0].startswith("thick crossing")]
    I, V, tiles = zoo_sheet(0, rows=row)
    P = prepare(I)
    dets = X.detect(P.logI, P.sigma, P.valid)
    t = [t for t in tiles if t["value"] == 20][0]
    p = t["ambiguous"][0]
    near = [d for d in dets if np.linalg.norm(d["xy"] - p) <= 8.5]
    assert any("parts" in d and d["kind"] == "crossing" for d in near), \
        [(d["xy"] - p, d["arms"], d.get("parts")) for d in near]


def test_texture_raises_the_noise_that_arms_must_beat():
    """Correlated texture makes ray averages vary from direction to
    direction more than sensor noise alone would (texture_noise)."""
    I, V, tiles = zoo_sheet(0, rows=ROWS[:2])
    P = prepare(I)
    D = X.darkness(P.logI)
    sensor = float(np.median(P.sigma)) / np.sqrt(20.0)
    assert float(np.median(X.texture_noise(D))) > 1.3 * sensor
    I, V, tiles = zoo_sheet(0, rows=ROWS[:2], texture_scale=0.0)
    P = prepare(I)
    assert float(np.median(X.texture_noise(X.darkness(P.logI)))) < 1.3 * sensor
