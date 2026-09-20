# Experiments: seed-and-grow, and an ensemble of walkers

Two attempts to replace the staged search with something that starts from what
we are certain of and works outwards, instead of thresholding the whole frame.
Neither is in the pipeline. Both are kept because the measurements are worth
more than the code, and because the reasons they fell short are specific.

## The idea

The staged search decides the network by thresholding evidence computed the
same way everywhere. Everything after it — junctions, consolidation, growth —
can only rearrange that decision. A vessel that has already been found, on the
other hand, tells us its own diameter, depth and direction, which is a far more
specific thing to look for than "a ridge".

So: take only the fragments we are sure of (`seeded.py`: z > 5 over at least
60 px, then fitted), and grow the network out of them.

## What happened

**Growth is worth 200× more when it starts early.** The same growth code, the
same thresholds, moved from the end of the pipeline to the start:

| growth runs… | real centreline gained | texture surrogate gained |
|---|---|---|
| after the full pipeline | +0.2 % | +3.0 % |
| from confident seeds only | **+44 %** (1522L), **+64 %** (1522R) | **0 %** |

Run last it has nothing to do, because the detector has already found what it
is going to find. That is the whole of the difference.

**The seeded network is clean but incomplete.** 30 seeds and growth give
16 024 px/MP on 1522L against the staged pipeline's 20 541, and **zero** on a
vessel-free texture surrogate against the staged pipeline's 1 939 px/MP — a
strictly better precision, at about 78 % of the coverage. Growth can only
extend, never branch, so it cannot reach the side vessels that make up the
missing fifth.

**Walkers, propagated in expectation** (`walkers.py`). The transition
probabilities are a deterministic function of the image, so the ensemble's
density can be computed directly rather than sampled; there is no RNG and the
result is reproducible. The state carries position **and direction**, because a
position-only walker diffuses isotropically and a vessel stops being
distinguishable from a smudge.

Summing visits turned out to be the wrong statistic: mass decays with distance,
so the far end of a vessel is exponentially fainter than the end by its seed,
and no single threshold serves both (87–97 % precision, but reaching only a
third of the network). Scoring each point by the **weakest step on the best
path to it** — a max-min, or bottleneck, propagation — does not decay with
distance, and at a threshold of 0.25 the result overlapped the staged network
92 % one way and 91 % the other.

**And it is still worse**, which the overlap numbers hid. The walker network
breaks long vessels: 88 fragments where the staged pipeline gives 52 vessels.
Pixel overlap measures *where* the vessels are and says nothing about
continuity, which is the property the whole identification exists to provide.
A metric that cannot see the defect the eye sees first is the wrong metric.

It is also slow — 167 s per crop against about 30 s — because the max-min
iteration runs to convergence over 32 direction planes.

## What would have to change for this to win

- Extraction that preserves continuity. The fragmentation is in the step from
  the bottleneck map to centrelines (threshold, then skeletonise), not
  necessarily in the map. Tracing the map by following its maxima, or feeding
  it to the existing junction machinery instead of skeletonising it, would test
  that.
- Branching for the seeded grower: sprouts off the flanks of a growing vessel,
  which is what the missing fifth of the network consists of.
- The isolated-vessel case, still unmeasured. Planted benchmark vessels sit
  where nothing else is, so if one does not seed, walkers can never reach it.
  `bench_walk.py` runs that test; an earlier attempt ran with a sign error and
  its numbers were discarded.

## Running them

```bash
python analysis/vessels/experiments/seeded.py 1522L 5.0 60
python analysis/vessels/experiments/bench_walk.py 1522L
```

Both expect the benchmark harness from the vessel-detection work, which lives
outside this repository; they are kept here as a record of method and result
rather than as something that runs unattended.
