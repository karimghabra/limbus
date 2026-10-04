# vesselnet: learning the vessel graph of real averaged frames from synthetic scenes

This plan is for a Claude Code session on a machine with an RTX 3080 (10 GB). It continues work done on CPU in
this repository and in [vesselscene](https://github.com/karimghabra/vesselscene). Branch `claude/vesselnet-plan`
starts from `claude/consolidation-energy-search`, which holds the latest `vesselmap`: renderer, energy search,
structure zoo, intersection detector and tracer. It does not include main's newer app and installer commits.
Merge main when convenient; `vesselnet/` will not conflict.

The pilot that motivates the plan is in `vesselnet/pilot/`. Its results are in `vesselnet/RESULTS.md` (entry 0)
and summarised in §9.

---

## 0. Start here

1. **Code.**
   - This repository, branch `claude/vesselnet-plan`.
   - vesselscene beside it: `git clone https://github.com/karimghabra/vesselscene ../vesselscene` and
     `pip install -e "../vesselscene[test]"`.
   - `pip install -r requirements-vesselmap.txt`.
   - Put the repository root on `PYTHONPATH`. `vesselmap` lives there, and vesselscene's `to_vesselnetwork`
     imports it.
2. **GPU.** `python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.mem_get_info())"`.
3. **Real averaged frames.**
   - Fetch the bursts: `python tools/fetch_reference_data.py`. That is 1.2 GB; `reference_data/README.md`
     describes the five bursts.
   - Stabilize them: `cd analysis && python -m stabilize ../reference_data`.
   - Each burst's average is `mean_stabilized.tif` under `stabilization/<method>/<burst>/`. These are the target
     images. Bursts 1–3 are 1920 × 1200, burst 4 is 1920 × 500, and burst 5 (1920 × 100) is too thin to be useful.
4. **Tests.**
   - `python -m pytest -m "not slow" vesselmap/tests`: 73 tests, about 9 minutes on 4 CPU cores.
   - vesselscene's suite: `python -m pytest ../vesselscene` takes 11–15 minutes on a GPU. Run at least
     `test_scene.py` and `test_junctions.py`.
5. **Smoke test of the pilot on the GPU.**
   ```bash
   python vesselnet/pilot/gen.py --start 0 --stop 12 --kind average --size 512 --device cuda --out data/pilot
   python vesselnet/pilot/train.py --data data/pilot --iters 300 --run runs/pilot_smoke --gain 0.6,2.6
   python vesselnet/pilot/evaluate.py val --run runs/pilot_smoke --data data/pilot
   ```
6. Read §1–§4, then start iteration 0 (§5).

---

## 1. The goal, stated as an optimisation problem

**Output.** For an image I, a real averaged frame, we want a vessel graph G in `vesselmap`'s `VesselNetwork`
format (`vesselmap/network.py`, saved as `map.json`):
- **Edges.** Each edge is a vessel centreline, a clamped cubic B-spline. Along it run radius r(t), blur s(t)
  (defocus, a depth cue) and contrast a(t) (optical density) profiles.
- **Nodes.** Nodes are where vessels branch or join (bifurcation, junction), end, or leave the image.
- **Crossings.** Vessels crossing at different depths are not nodes. They overlap, their densities add, and
  they are listed in `G.graph["crossings"]`.

**Objective.** Minimise the energy that `vesselmap/search.py` already defines:

    E(G, B) = NLL(I | render(G, B)) + prior(G) + Σ_v (τ·λ_vessel + price·L_v) + Φ(G)

| term | meaning | where |
|---|---|---|
| NLL | weighted squared residual between log I and the render B − Σ a·P·T, with per-pixel noise | `render.py` (`NetworkModel`) |
| prior | bending energy, smoothness of the profiles along each vessel, smoothness of the background B | `render.py`, `fit.py` |
| τ·λ_vessel + price·L_v | existence cost: a vessel must explain more per pixel than a dark trace of texture does (the texture null) | `search.py` |
| Φ | fragmentation: pairs of vessel ends that face each other and would join | `search.py` |

**Two requirements, both needed.**
- **R1, residual.** The rendered graph explains the image: E as low as possible.
- **R2, architecture.** The graph is the true one:
  - every true vessel is one edge, or a chain through its own branch points;
  - crossings are not made into branch points;
  - at every junction the arms are paired as in reality (which arm continues which);
  - nothing is invented in texture, and nothing visible is left out.

R1 alone is ambiguous. Many graphs render almost the same image:
- a crossing versus two vessels that touch and part (a kiss);
- a fork versus a crossing seen in 2-D;
- one vessel versus two pieces of it;
- a blurred vessel versus a dark band of texture.

Those alternatives differ only slightly in E. R2 can be measured only where the truth is known, on synthetic
scenes. So:
- Develop and validate on vesselscene, where both R1 and R2 can be measured.
- Apply to real averaged frames, where R1 and plausibility checks are all we have.
- The synthetic validation is what allows trusting E's minimum on real frames.

**The energy check, the question to keep asking.** On synthetic scenes, is the true graph the minimum of E?
- Take the truth, exported to `map.json` by vesselscene's `to_vesselnetwork`, with profiles fitted by
  `vesselmap`.
- Perturb it: swap the arm pairing at a crossing, split a vessel, merge two, add a vessel along texture, delete
  a faint vessel, turn a crossing into a branch point.
- Check that E rises each time. Where it does not, no optimiser can recover the truth, and the energy must
  change first (priors, §4D).

Learning enters in two places:
- **Proposals:** where vessels and junctions probably are, and how arms pair.
- **Priors:** which of two near-equal explanations is the real one.

The optimiser (gradient fitting plus discrete moves) is what makes the graph explain the image.

**The iterative approach is itself an outer optimisation.** Each iteration changes one of four things: the
data, the network, the extraction, or the energy. It is measured on fixed metrics (§6) and kept only if it
improves them. Results go to `vesselnet/RESULTS.md`.

---

## 2. What already exists

**vesselmap (this repository).**
- **Model and fitting.** The renderer and likelihood (`render.py`). Discovery by Hessian ridges, band by band
  (`fit.build_map`). Fine detail (`refine.refine_map`). One spline per vessel (`consolidate.consolidate_map`).
  The energy search with join / split / delete / reroute / revive moves (`search.search_map`).
- **Synthetic evaluation.** `synth-eval` and `consolidate-eval`.
- **Real-frame result (vesselmap README).** On frame 20 of burst 1, map + refine gives 578 edges, 68.9k px of
  centreline and a data NLL of 1.474M. It took 41 + 64 min on 4 CPU cores.
- **A warning from the README:** the registered mean of a burst is the wrong input for the current mapper. It
  has almost no noise, so tissue texture passes the mapper's noise-relative tests and is traced as vessels. A
  method meant for averaged frames must tell vessels from texture by their structure (learned from realistic
  averages) and by the texture-null price in E, not by noise-relative thresholds.
- **Structure zoo and junction detection** (`zoo.py`, `intersections.py`, `tracing.py`). The zoo has controlled
  structures in tiles with marked junctions. The hand-built detector with tracing finds 569 of 609 marked
  junctions with 18 false detections on seeds 0/10/20, and 571 / 19 on seeds 30/40/50. Arms are exactly right
  at 95–97 % of found junctions.

**vesselscene (separate repository, MIT).**
- **Images.** Physiological synthetic stills of conjunctival and episcleral vessels: the average of a burst and
  single frames. Image formation is calibrated on LIMBUS's real stills.
- **Scale.** k = 4 px per capillary diameter by default. The `random` preset draws k from 1.5 to 8.
- **Presets.** `healthy`, `pathologic` (dilated, tortuous) and `random` (wide ranges, meant for training).
- **Truth.**
  - The complete graph: splines on a flow digraph, with radius, depth and haematocrit along each vessel.
  - An observable truth: what a careful annotator could see, with runs, junctions, an image graph and a
    don't-care mask.
  - An image-junction truth: every place vessels meet or overlap, typed, with arms, which arms belong to the
    same vessel (`partition`), lineage and difficulty.
  - Label rasters (`labels.npz`): lumen, centreline, junction heatmaps by type, arm direction, group and
    visibility, radius, depth.
- **Export.** `scene.to_vesselnetwork` writes the truth as a `vesselmap` `VesselNetwork` (`map.json`). The truth
  can therefore be rendered and scored by `vesselmap`'s own energy.
- **Scoring helpers.** `truth.score_tracing` scores centrelines and junctions against the observable truth. A
  junction matches within max(radius, λ = 11.9 px). `truth.score_fit` scores a fit against the complete truth
  per visibility level.
- **Realism scorecard.** `python -m vesselscene score`, against aggregate statistics of the real stills that
  ship in vesselscene's `data/`.
- **Known realism gaps (vesselscene v2):**
  - hard-edged dark shapes where a tributary runs along the vein it joins;
  - somewhat too few thin, dark, sharp lines;
  - the widest veins with too flat a floor;
  - too few arteriole–venule pairs;
  - capillary dotting in frames more regular than real.

---

## 3. Data: synthetic scenes at a realistic scale on one RTX 3080

### What to generate
- **Kind.** `average`, because the target is real averaged frames. Optionally keep the single `frame` of 10–20 %
  of scenes from stage S2 on, as a robustness set.
- **Size.** Full 1200 × 1920, the real frame size. vesselscene lays out the anatomy for the whole frame:
  collecting veins through its central 60 %, the deep plexus, arcades. Smaller scenes lose that structure.
  Train on crops; infer on whole images.
- **Presets.** healthy 50 %, random 35 %, pathologic 15 %. Revisit after iteration 1 using per-preset metrics.
- **Splits.** Fixed by seed range, never mixed:

| split | seeds | count | outputs kept |
|---|---|---|---|
| train | 0 – 99,999 | as generated | compact |
| validation | 1,000,000 – 1,000,049 | 50 | compact + `map.json` |
| test | 2,000,000 – 2,000,099 | 100, frozen | full `save_scene` (complete truth for `score_fit`) |

The test set is reported only at the end of an iteration.

### What to keep per scene
A full `save_scene` folder is about 75 MB, plus about 25 MB per image for the observable and complete truth.
That is too much for thousands of scenes. Extend `pilot/gen.py` to a compact exporter of about 10–15 MB:
- the average still (float32) and its valid mask;
- **for learning**, from `labels.npz` (float16 or bit-packed):
  - observable lumen, centreline with vessel id, don't-care;
  - junction heatmaps by type and the junction don't-care;
  - junction arm direction, group and visibility;
  - radius, depth and blur of the dominant vessel;
  - the second vessel where two overlap;
  - derived from the truth polylines: centreline orientation and a vessel-id map, for pairing and instance
    losses;
- **for the graph:** `observable_average.json`, `junctions_average.json`, `map.json` and `truth.json`;
- `scene.json`: parameters, seeds, code versions, timings.

Write one npz plus the JSON files per scene, in folders of 100. Make generation resumable (skip existing
outputs) and log failures.

### Timing and scale (estimates; measure them in iteration 0)
vesselscene's reference timings on an RTX 3080, two scenes at a time, for a healthy full scene with both kinds:
- `make_scene` + `save_scene` about 160 s;
- anatomy, on the CPU: 51 s (32–87);
- render of the average: 13.5 s;
- junction truth: 11 s; observable truth: 6 s;
- pathologic scenes 250–265 s; random 85–470 s.

Average only with the compact save should take about 110–150 s per scene in one process. Anatomy is CPU-bound and
overlaps the GPU, so 2–3 processes in parallel should give about 40–60 scenes an hour. Check that VRAM allows it.

| stage | scenes | wall time | disk | when |
|---|---|---|---|---|
| S0 | 20, plus validation 50 and test 100 | about 4 h | 3 GB + 10 GB (full test folders) | iteration 0 |
| S1 | 500 | about 10 h, overnight | 7 GB | iteration 1 |
| S2 | 2,000 | about 2 days, in the background while training | 25 GB | iterations 3–4, only if S1's learning curves still rise |
| S3 | 5,000 or more | about 5 days | 60 GB | only if S2's curves still rise |

A full healthy frame has 550–730 vessels and about 700 visible junctions. S1 already holds about 350,000
junctions. Decide each stage's size from learning curves: validation metrics against 50 / 150 / 500 / 2000
scenes. Do not generate more by default.

After each batch of scenes, run the realism scorecard on 20 of them and keep its output with the batch.

---

## 4. Approach: learned perception plus explicit optimisation

For one image the pipeline has four stages:

    image ──A──► dense maps ──B──► initial graph ──C──► optimised graph (min E)
                     ▲                                        │
                     └──────────────── D ◄────────────────────┘

**A. Dense prediction network.** Start simple and grow only when the metrics say so.
- **Input.** Log intensity, flattened and normalised exactly as in the pilot: log I less its Gaussian of
  σ 24 px, divided by the median absolute deviation. Real frames get the same preprocessing.
- **Backbone.** A U-Net with a ResNet-34 or ConvNeXt-T encoder (20–30M parameters), mixed precision, 512 × 512
  crops, batch 8–12. Check that it fits in 10 GB.
- **Heads.** vesselscene provides truth for all of them.
  1. Vessel lumen: observable, don't-care masked.
  2. Centreline, orientation (cos 2θ, sin 2θ), radius, blur and contrast.
  3. Overlaps: a second-vessel channel where two lumens cover a pixel, and which one is on top.
  4. Junction heatmaps at two granularities:
     - vesselscene's clustered junctions by type (bifurcation / confluence, crossing, compound, pseudo-T): the
       annotation target;
     - each event point: every crossing and every branch point, which vesselscene's junctions list as
       `members`. This matches the zoo's and `vesselmap`'s granularity. The pilot showed this choice matters
       (§9).
  5. Arm fields near junctions: each arm's direction, and an embedding in which arms of the same vessel lie
     close together. This is the information tracing needs to go straight through a crossing.
- **Losses.**
  - focal loss on heatmaps (CenterNet-style);
  - BCE + Dice on masks;
  - an angular loss for orientation;
  - L1 on log radius, blur and contrast at centreline pixels;
  - a discriminative loss on the arm embeddings;
  - don't-care masks everywhere.
- **Augmentation.**
  - flips and rotations;
  - contrast gain 0.6–2.6, log-uniform (a pilot finding);
  - noise, blur and illumination ramps;
  - scale jitter 0.7–1.4×, because k is not known exactly on real frames;
  - residual registration blur.

**B. Graph extraction.**
- Trace centrelines on the predicted centreline and orientation fields. Reuse `vesselmap.tracing`'s tracer,
  with the network's orientation field in place of its hand-made orientation score.
- Go straight through junctions using the predicted arm pairing; stop at vessel ends.
- Place branch nodes from the junction heatmaps, where traced centrelines meet. Record crossings without making
  them nodes, as `vesselmap` does. Make one edge per traced vessel.
- Take initial profiles r, s and a from the predicted maps.
- Export a `VesselNetwork`, the same object `build_map` produces, so everything downstream works unchanged.

**C. Optimisation, with `vesselmap`.**
- `fit.optimize`: a joint gradient fit of all splines, profiles and the background, starting from the
  extracted graph.
- Then `search.search_map`: discrete moves with the Φ term, lowering E further.
- This replaces `build_map`'s ridge proposals with the network's graph. Compare it on the same images against
  `build_map` → `refine_map` → `consolidate_map` → `search_map`.
- The fitting code was written for CPU threads. Profile it on the GPU in iteration 0. If a full frame takes
  too long, fit and search in overlapping windows; `search.py` already scores moves on local footprints.

**D. Learning from the optimiser** (iterations 4–6).
- **Hard examples.** Crops where the optimiser changed the network's graph (deleted, joined, rerouted) or where
  the residual stays structured. Train on them more often, so the network learns to anticipate the optimiser.
- **Learned priors.** A small model scoring local alternatives: pairing at a junction, crossing versus touch,
  one vessel versus two, vessel versus texture. Train it on synthetic truth. Use it as an extra term in E, or to
  rank `search_map`'s moves. It is the remedy wherever the energy check (§1) fails.
- **Self-training on real averaged frames, guarded.** Take pseudo-labels only where the optimised graph
  lowers E clearly and agrees with the learned priors. Never use a real frame's own output as its truth
  without that test, which keeps the pipeline from becoming circular.

---

## 5. Iterations

Every iteration follows the same steps:
- state a hypothesis;
- make one change;
- measure on the validation set with the fixed metrics of §6;
- decide whether to keep it.

At the end of each iteration, score the test set once, commit, add a results table to `vesselnet/RESULTS.md`,
and write the next decision. Time boxes are guides.

### Iteration 0: environment, baselines, energy check (1–2 days)
- Install, run the tests, check the GPU. Measure:
  - time per scene, per preset;
  - VRAM per generation process;
  - training speed of the backbone;
  - `vesselmap` fit and search time on a GPU, for a 512² window and a full frame.
- Generate the validation (50) and test (100) scenes.
- Run three baselines on 10 test scenes:
  1. **Truth oracle.** The truth's `map.json` with profiles and background fitted by `vesselmap`, centrelines
     held fixed. This gives E_truth and the residual floor of `vesselmap`'s model on vesselscene images. The two
     renderers differ: vesselscene integrates tubes in closed form and blurs junction unions, `vesselmap` uses
     nested boxes. So E_truth is the floor to measure against, not zero.
  2. **vesselmap's pipeline.** `build_map` → `refine_map` → `consolidate_map` → `search_map`: E, the
     architecture metrics and the time.
  3. **The hand-built junction detector,** `intersections.detect(..., trace=True)`.
     - Give it intensities in 0–1 (`vesselmap.image.load_image` / `prepare`).
     - It was tuned on the zoo. On the pilot's vesselscene frames it found almost nothing: 27 candidates above
       threshold on a 512² frame with 78 observable junctions, and 1 with three arms.
     - Retune its thresholds and its darkness width (31 px, narrower than the widest veins) on 5 validation
       scenes before using it as a bar.
- **Energy check** (§1) on 5 validation scenes: a table of perturbation type against ΔE.
- Port the pilot to the averaged images on the GPU and reproduce its numbers at small scale.
- **Exit:** a baseline table, timings, energy-check results. Set the numeric targets for iterations 1–3 from
  these baselines.

### Iteration 1: dense network v1 on S1 (3–5 days)
- Generate S1 (500 scenes) overnight.
- Train the backbone with heads 1–4.
- Learning curves for 50 / 150 / 500 scenes.
- **Metrics:**
  - centreline precision and recall (observable, 3 px);
  - junction F1 at both granularities, and type accuracy;
  - orientation error; radius error;
  - all per preset and per visibility level.
- **Decide:** if the curves still rise at 500 scenes, schedule S2. If they are flat, the model or targets are
  the limit, not the data.

### Iteration 2: graph extraction and fit (3–5 days)
- Extraction v1 (§4B), then `fit.optimize`.
- **Metrics:** the architecture metrics and ΔE to the oracle, against `vesselmap`'s pipeline, on the validation
  set.
- **Exit:** ΔE to the oracle smaller than the pipeline's; fewer edges per vessel; arm-pairing accuracy reported.

### Iteration 3: search from the network's graph (1 week)
- `search_map` starting from the extracted graph.
- **Ablation:** network start against `build_map` start against a perturbed truth start.
- Count accepted moves by type.
- Wherever the energy check failed, try prior changes.
- **Exit:** the network start reaches a lower E and better architecture than the `build_map` start, in less
  time.

### Iteration 4: scale and pairing (1–2 weeks)
- S2 data, if iteration 1's curves justified it.
- The arm-pairing head (5) and pairing-aware extraction.
- A larger model if it underfits.
- **Active data generation:** generate more of the structures with the worst metrics, by adjusting vesselscene's
  preset parameters. Examples: dense meshes, tortuous capillaries, deep blurred vessels under sharp ones.

### Iteration 5: real averaged frames (1 week)
- Run on `mean_stabilized.tif` of bursts 1–4.
- Rerun `vesselmap`'s baseline on the same averages: its README numbers are for a single frame, and averages
  behave differently (§2).
- Compare E, data NLL, unexplained ridges in the residual, and edges and centreline length.
- **Review sheets for the user:** crops with the graph overlaid, the render and the residual. Real frames have
  no truth, so a person judges the plausibility.
- **Consistency without truth:** the same eye mapped from two bursts, or from the two poses of burst 1, should
  give overlapping graphs after registration.
- **Realism gap:** vesselscene's scorecard statistics on the real averages against the synthetic set.
- Feed systematic failures back:
  - realism failures into vesselscene (generator);
  - otherwise into augmentation or priors.

### Iteration 6 onward: closing the loop
- Learned priors and move proposals (§4D).
- Guarded self-training on real averages.
- Repeat iterations 1–5 with the improved components while the metrics keep improving.

---

## 6. Metrics and evaluation protocol

**Synthetic (validation and test; vesselscene truth).**

| what | metric | tool |
|---|---|---|
| vessels found / invented | centreline recall and precision within 3 px against the observable runs; per visibility level against the complete truth | `vesselscene.truth.score_tracing`, `score_fit` |
| one spline per vessel | cover, purity, edges per vessel (length-weighted; 1 is ideal) | port `vesselmap.synthetic.vessel_metrics` to vesselscene's truth (`true_vessel` in each `map.json` edge's info) |
| junctions | recall and precision within max(radius, 11.9 px), type accuracy; per-event recall within 4 px + r (the zoo's rule) | `score_tracing`, `pilot/evaluate.py` |
| topology | crossings made into nodes (should be 0); forks missed; pairing accuracy (share of junctions where every pair of arms is joined as in the truth's `partition`) | new, against `junctions_average.json` |
| residual | ΔE = E(G) − E(oracle); data NLL; reduced χ²; unexplained ridge pixels in the finest band | `vesselmap` render and fit |
| cost | seconds per frame for network, extraction and optimisation | |

**Real averaged frames.**
- E, data NLL, unexplained ridge pixels, edges and centreline length, centreline of vessels thinner than 6 px;
- consistency across bursts or poses;
- review sheets.

**The zoo** (`vesselmap.zoo`, a different generator): per-event junction found / false with
`pilot/evaluate.py zoo`. Report it, but do not optimise for it:
- zoo vessels are about twice as dark relative to their background as vesselscene's or real frames';
- the zoo counts each crossing of a repeated-crossing structure separately.

**Reporting.** One table per iteration in `vesselnet/RESULTS.md`. It lists validation metrics, test metrics at
the iteration's end, timings, the data used, the model and the commit hash.

---

## 7. Engineering

```
vesselnet/
  PLAN.md          this plan
  RESULTS.md       results log, one entry per iteration
  data.py          compact scene export, loading, sharding, target building
  gen.py           generation command (resumable, seed ranges, presets, parallel processes)
  model.py         backbone and heads
  train.py         training (mixed precision, checkpoints, validation every N iterations)
  extract.py       graph extraction (§4B) into a VesselNetwork
  optimize.py      vesselmap fit / search glue, windowed for full frames
  metrics.py       §6 metrics
  evaluate.py      validation, test, zoo, real frames, review sheets
  configs/         one YAML per experiment
  tests/           fast unit tests: preprocessing, targets, extraction on hand-made fields, metrics on hand-made graphs
  pilot/           the CPU pilot, kept as a reference
```

- **Data and runs** live outside git: `~/vesselnet_data` and `~/vesselnet_runs`, or `data/` and `runs/` at the
  repository root (ignored). `.gitignore` already blocks `*.png` and `*.tif`.
- **Long jobs** run under `tmux` or `nohup` and must be resumable: skip existing outputs, checkpoint often. Log
  timings. Record seeds and the git hash with every output.
- Keep `vesselmap`'s fast tests green. Add fast tests for every new module.
- **Commits** follow the repository's style: `vesselnet <part>: what changed`, with the numbers in the body.

---

## 8. Risks and mitigations

| risk | mitigation |
|---|---|
| Synthetic-to-real gap (vesselscene's known differences, §2) | Realism scorecard per batch; augmentation; iteration 5 analysis; feed generator fixes back to vesselscene |
| E's minimum is not the true architecture | The energy check from iteration 0 on; learned priors (§4D) |
| Averages have little noise, so texture is the main confusion | Train on realistic averages; texture-null price in E; no noise-relative thresholds |
| Junction granularity (clustered or per event) | Both heads, metrics for both |
| Scale k unknown on real frames (about 4, calibrated) | Scale augmentation; compare predicted widths with the real width distribution (median FWHM 11.9 px) |
| Tiling seams on full frames | Overlap tiles by at least the receptive field (the pilot pads 32 px) |
| Generation slower than estimated | Use 768 × 768 scenes for S1 and keep full frames for validation and test |
| Overfitting vesselscene's renderer | Also report the zoo, a different generator, and E on real frames |
| `vesselmap` fit or search too slow on full frames | Windowed optimisation; profile in iteration 0 |

---

## 9. The pilot (CPU, this branch): results and lessons

Pilot data: 512 × 512 single frames (not averages) from vesselscene, mixed presets. The model is a 1.6M-parameter
U-Net (`pilot/net.py`) with heads for junction heat, type (crossing / branch / compound), lumen and centreline.
It was trained for 3000 iterations of 8 crops of 256 × 256 on 43 scenes, on 2 CPU threads (about 2 hours).

| check | result |
|---|---|
| held-out vesselscene frames (6 frames, 407 junctions) | F1 0.71: 63 % found at 81 % precision (threshold 0.2), or 79 % at 64 % (0.15); type accuracy 0.54 |
| hand-built detector on the same frames | about nothing (it was tuned on the zoo) |
| zoo, seeds 0/10/20 (609 junctions) | 454 found / 26 false (threshold 0.4), 424 / 14 (0.5); the hand-built detector: 569 / 18 |
| real frame 20 (a single frame) | 549 junctions at threshold 0.2, many where vessels meet, some on single vessels |

Lessons:
1. **The network learns junctions from vesselscene quickly.** It was still improving when the run stopped. Data
   and compute were the limit.
2. **Most zoo misses were definitional.** 65 of 155 were crossings 20–25 px apart (a weaving capillary, twisted
   pairs) that vesselscene clusters into one junction. 22 more were forks whose branches run side by side.
   Loosening the match radius changes almost nothing, so it is not localisation. Hence the two granularities in
   §4A.
3. **Contrast domain.** Every false detection on the zoo lay on a thick, sharp, dark vessel; none were on
   background. Zoo vessels are about twice as dark relative to their background as vesselscene's or the real
   frame's. Real frame 20 resembles vesselscene, not the zoo. Training over a contrast range of 0.6–2.6 instead
   of 0.7–1.4 cut zoo false detections at threshold 0.2 from 500 to 170.
4. **Input scaling is not the problem.** After the pilot's preprocessing, background texture comes out at the
   same level in all three domains: a bright-half spread of 0.86–0.97. Scaling by the background alone would
   barely change the input, so it was not tried.
5. **Practicalities.**
   - `vesselmap`'s `prepare()` expects intensities in 0–1; raw 12-bit counts make every pixel saturated.
   - On the CPU a 512² frame takes about 150 s with 4 threads and 4–20 minutes with 1.
   - Training was 2.2–2.5 s per batch of 8 on 2 threads.
   - Validation inside training must detect peaks at the lowest threshold it reports (fixed in `pilot/train.py`).

---

## 10. Decisions to confirm with the user

- **Target images:** averaged frames (`mean_stabilized.tif`). Are single frames a secondary target?
- **The final graph's convention:** `vesselmap`'s branch nodes plus a crossing list. Junction clusters are a
  learning target, not an output.
- **Numeric acceptance targets** for iterations 1–3, set from iteration 0's baselines.
- **Disk and time budget** for stages S2 and S3.
- **When to merge main** into this branch.
