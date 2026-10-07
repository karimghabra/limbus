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

---

## Iteration 0: environment, data, energy check, baselines (in progress, 2026-10-05/06)

Machine: Windows 11, RTX 3080 (10 GB), 12 logical cores, 32 GB RAM (commit limit 37–54 GB), shared with
other jobs that held up to ~30 GB of commit. Python 3.12, torch 2.11.0+cu128.

**Decisions taken with the user (PLAN §10 and since).**
- LIMBUS is the single home of `vesselnet/`; vesselscene keeps a pointer. Main merged in at iteration 0.
- Targets: averaged frames; the single frame is only a robustness set from S2 on.
- **Strict fork convention** for R2 and for E: a vessel ends at every fork (vesselscene's truth); a
  vesselmap through-edge at a fork joins two vessels.
- **Energy**: `vesselnet.optimize.energy_config` = vesselmap's E with `strict_forks` (below) and the floor
  price τ·λ_length per px (no texture null); vessel versus texture is left to a learned prior (§4D).

**Setup fixes.** vesselmap's search used `fork` (absent on Windows): moves are now scored in-process there
(de5d6b7), serially, which makes the search slow (below). The flow tests need pandas. vesselmap's renderer
and fit are CPU-only, so "fit time on a GPU" does not apply.

**Target averages.** `stabilize --method nonrigid` beats `translation` on every burst it processes
(registration Dice after: burst 1 0.940 vs 0.933, burst 2 0.908 vs 0.791, burst 4 0.962 vs 0.955); it skips
burst 3, which translation registers poorly (usable 22 %, Dice 0.503). Targets: nonrigid averages of
bursts 1, 2, 4; burst 3 (translation) flagged poor; burst 5 too thin.

**Data (S0).** `vesselnet/gen.py`, compact export (scene.npz ~10 MB + JSON ~10 MB; test scenes also keep the
full `save_scene` folder). Full 1200 × 1920 averages; times per scene with 2–3 workers on the shared machine:

| split | scenes | healthy (n, median s, range) | random | pathologic | peak VRAM (median / max) | disk |
|---|---|---|---|---|---|---|
| val | 50 / 50 | 28, 164 s, 118–655 | 13, 137 s, 40–1129 | 9, 252 s, 163–428 | 1652 / 1892 MB | 0.95 GB |
| test | 98 / 100 | 54, 224 s, 121–606 | 32, 244 s, 76–1402 | 12, 289 s, 165–456 | 1635 / 2121 MB | 8.3 GB |

A healthy scene alone takes 120–160 s. With 3 workers: ~45 scenes / h once native thread pools are capped
(OMP/MKL/OpenBLAS/OpenCV at 4 per worker; uncapped, a 46 s scene ran over an hour). Test seeds 2,000,057 and
2,000,081 (random preset) need single 1.9 / 3.9 GB arrays and failed three times at the commit limit; the
test set is 98 scenes until they can be made (no seed is replaced).

**Backbone speed** (`vesselnet/bench.py`, bf16, 512² crops, 16 outputs): U-Net ResNet-34 (24.4M) batch 12:
0.142 s / step, 84 crops / s, 3.5 GB; ConvNeXt-T (31.9M) batch 12: 0.173 s, 69 crops / s, 6.1 GB.

**vesselmap timing (CPU, 4 threads, shared machine).** Oracle fit (300 steps, centrelines fixed) of a 512²
window: median 112–144 s (79–397 s). Pipeline on one 512² test window (s2,000,000): build_map 1040 s,
refine_map 846 s, consolidate_map 372 s, search_map more than 3.5 h (serial move scoring on Windows).

### Energy check (PLAN §1)

`vesselnet/energy_check.py`: 10 windows of 512² from validation seeds 1,000,000–004; for each, the truth
oracle fixes τ and the price; the truth and each perturbation get the same **local** fit (40 px around the
event, 20 px around a deleted / added vessel; halo and background fixed, as the search fits a move). A first
version refitted the whole window and was discarded: the refit drifted elsewhere by up to ~360k nats (at one
fork 369k over the window, 8.4k within 30 px of it). τ varies 3.6–307 and the texture-null price 3.8–11,980
per px across windows.

E prefers the truth (count / n):

| perturbation | n | median ΔNLL | vesselmap's E | + strict forks | + floor price (chosen) |
|---|---|---|---|---|---|
| swap pairing at a crossing | 20 | +48k | 19 | 19 | **20** |
| crossing made a branch point | 20 | +1.4k | 19 | 20 | **20** |
| split a vessel | 20 | +4.0k | 20 | 20 | **20** |
| delete the faintest observable vessel | 20 | +70k | 4 | 5 | **19** |
| fork joined into a through vessel | 20 | +0.07k | 1 | 7 | **10** (median ΔE +45) |
| add a vessel along texture | 5 | −72k | 3 | 3 | **0** |
| delete a non-observable vessel (should fall) | 7 | +56k | 5 | 3 | 1 |

- **Strict forks** (vesselmap 7c06038 and earlier): Φ does not charge the three ends of a fork; an end on
  another vessel's interior pays τ·λ_vessel (a through vessel with a branch is three strict vessels); fork
  ends are never charged as attachments. All exact in the search's ΔE (tests). With them, 18 of 20 forks
  are structurally neutral (attachment +τλ, one vessel fewer −τλ), so ΔE ≈ ΔNLL: the data itself is near
  neutral at a typical fork, and fork-versus-through is left to the learned prior (§4D). A few forks fail in
  the data (a wide vein continuing straight: ΔNLL −58k).
- **Price**: the texture null deleted 16 of 20 faint observable vessels; the floor price keeps 19, but then
  every texture trace is accepted (they explain 40k–1.2M nats, the faint real vessels 7k–131k). No price
  separates them: vessel versus texture is the learned prior's job, and the network does not propose texture.
- "Non-observable" vessels (below vesselscene's CNR rule) explain a median 56k nats of the average; keeping
  them is not an error of E.

### Baselines on test windows (seeds 2,000,000–009, two 512² windows each)

- **Truth oracle** (20 windows): τ 5–252, price 144–19,463 per px; centreline recall 1.000; precision
  0.40–0.88 without the complete truth's profiles (the truth also holds non-observable vessels), e.g. 0.54 /
  0.68 on seed 2,000,000 with them.
- **Hand-built detector**, retuned on 10 validation windows (zoo-tuned defaults: junction F1 0.095, recall
  0.05): best texture off, no traced-arm requirement, darkness width 31, z_junction 2, z_arm 4: junction
  recall 0.406, precision 0.650, **F1 0.500**, event recall 0.334. On the 20 test windows: recall 0.429,
  precision 0.660, **F1 0.520**, event recall 0.306; 6 s per window.
- **vesselmap pipeline**: running (first window: ΔE to the oracle +26.8M after build_map, +13.7M after
  refine, +16.4M after consolidate; search pending).

**Pilot reproduced on averages** (`pilot/`, 60 scenes of 512² averages, 54 train / 6 held out, 3000
iterations, contrast gain 0.6–2.6; GPU 0.07 s / iteration against 2.2–2.5 s on 2 CPU threads):

| threshold | recall | precision | F1 | type accuracy |
|---|---|---|---|---|
| 0.10 | 0.962 | 0.476 | 0.637 | 0.58 |
| 0.15 | 0.817 | 0.754 | **0.784** | 0.57 |
| 0.20 | 0.711 | 0.837 | 0.769 | 0.57 |
| 0.30 | 0.530 | 0.922 | 0.673 | 0.55 |

470 held-out observable junctions (entry 0, single frames: F1 0.705 at 0.15 on 407). Zoo, seeds 0/10/20 (609
marked): 483 found / 52 false at 0.3, 440 / 13 at 0.4 (entry 0: 483 / 57, 454 / 26).

**Tests.** vesselmap's search tests and vesselnet's tests after the energy changes: 56 passed (12 min).

Pending for this iteration: the vesselmap pipeline on the 20 test windows (about 3 days, running beside
iteration 1, at the user's choice); the two missing test seeds (retried when memory allows); numeric targets
for iterations 1–3.
