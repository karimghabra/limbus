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
