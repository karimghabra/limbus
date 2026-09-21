"""Scenarios with a known geometry, and a metric that can see what happened.

The injection benchmark plants ISOLATED vessels and asks whether each was
found. That cannot see two vessels merged into one, a crossing whose identity
swapped, or a network in fragments - and all three are what goes wrong when
vessels run close together or junctions crowd.

    geometry   parallel pairs, crossings, bifurcations, ladders and braids,
               and the planting that puts them on real sclera
    score      span / cover / parts / merges, with a tolerance that scales
               with how close the truth lines actually are

Comparing two variants needs `paired`, not two runs of the sweep: scenarios
are planted on ground the detector finds empty, so the free mask depends on
the variant and the scenes move when the variant changes. Two separate sweeps
once read 0.21 and 0.08 on shallow bifurcations and looked like a clear
regression; the same variants on the same scenes give 0.183 and 0.183, and
disagree on none of 60.

The scorer is validated against known configurations in analysis/tests - a
resolved pair, a merged midline, one vessel of two found, a shattered vessel -
and that validation is not optional. The first version of this metric gave
each detected point to the nearest truth line by a Voronoi partition, so a
line down the middle of a close pair scored as a clean hit on one vessel and
a total miss on the other, and it reported a detector failure that was an
artefact of where the tolerance fell. It was blind in the same way as the
metric it was built to replace.
"""
from .geometry import MAKERS, SCENARIOS, paint, place, render_tube  # noqa: F401
from .paired import build_scenes, compare, free_ground, resolved  # noqa: F401
from .score import exempt_pairs, score, transverse  # noqa: F401
