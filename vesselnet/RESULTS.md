# vesselnet results log

One entry per iteration of `PLAN.md`: what changed, the data and model, validation metrics, test metrics at the
end of the iteration, timings, and the commit. Newest last.

---

## Entry 0: CPU pilot (before iteration 0)

**Setup.**
- Data: vesselscene 512 × 512 single frames (`kind = frame`, not averages), presets cycling healthy / random /
  healthy / random / pathologic by seed. 60 scenes were generated; training used the 43 present at the start
  (seeds not divisible by 10).
- Model: `pilot/net.py` U-Net, 1.6M parameters. Heads: junction heat (σ 3 px), type (crossing / branch /
  compound), lumen, centreline.
- Training: 3000 iterations, batch 8 crops of 256², AdamW with one-cycle learning rate 2e-3, contrast gain
  0.6–2.6. 2 CPU threads, 2.2–2.5 s per iteration.

**Held-out vesselscene frames** (6 frames, 407 observable junctions). A detection matches within
max(radius, 11.9 px).

| threshold | recall | precision | F1 | type accuracy |
|---|---|---|---|---|
| 0.10 | 0.909 | 0.399 | 0.554 | 0.55 |
| 0.15 | 0.789 | 0.638 | 0.705 | 0.54 |
| 0.20 | 0.627 | 0.812 | 0.707 | 0.55 |
| 0.25 | 0.482 | 0.907 | 0.629 | 0.54 |
| 0.30 | 0.361 | 0.919 | 0.519 | 0.50 |

The first run (contrast gain 0.7–1.4, 1000 iterations, 37 scenes) reached F1 0.69 at threshold 0.15 on its 4
held-out frames.

**Hand-built detector on the same frames.** `vesselmap.intersections.detect`, tuned on the zoo, found about
nothing: 27 candidates on a frame with 78 observable junctions, 1 of them with three arms.

**Zoo, seeds 0/10/20** (609 marked junctions; a hit must lie within 4 px + radius).

| threshold | found | false |
|---|---|---|
| 0.20 | 524 | 170 |
| 0.30 | 483 | 57 |
| 0.40 | 454 | 26 |
| 0.50 | 424 | 14 |
| hand-built detector | 569 | 18 |

The first run (gain 0.7–1.4) had 528 found / 500 false at threshold 0.2. The zoo misses at threshold 0.4 are
mostly:
- closely repeated crossings, which vesselscene clusters into one junction: weaving capillary 31, twisted pairs
  34;
- side-by-side forks: 22;
- thin crossings at radius 0.7–0.9 px.

A looser match radius adds only about 10.

**Real frame 20** (burst 1, a single frame): 549 junctions at threshold 0.2. Many are where vessels meet, some
on single vessels or faint lines.
