# vesselmap — a spline-network model of every vessel in one still image

`vesselmap` finds the vessels in a single greyscale image: large and small, in
focus and out of focus, crossing and branching. It returns them as a **directed
graph** whose edges are parametric vessel splines. The same model is then
re-fitted to each (stabilised) frame of a burst. The result is one map with
fixed node and edge ids, adjusted slightly per frame.

It uses **structure only**: the intensities of one image. No kymographs, frame
differences, temporal variance or velocities are used anywhere. That keeps the
map independent of the velocity measurements it is meant to support. Because
flow is not observed, edge direction is a structural convention (see
*Direction* below).

## Quick start

```bash
pip install -r requirements-vesselmap.txt

# 1. discover the network in one frame (about 40 min for 1920x1200 on 4 CPU cores;
#    --set reps_per_band=2 penalty_scale=3.0 is about 25 min at slightly lower recall)
python -m vesselmap map reference_data/burst_2026-09-16_15-50-52/frame_000020.tif -o out/map

# 2. add fine detail: small vessels and forks whose branches run side by side
#    (about 40 min; can be repeated on its own output)
python -m vesselmap refine out/map/map.json reference_data/burst_2026-09-16_15-50-52/frame_000020.tif -o out/refined

# 2b. recall tier for the search mask: faint vessels traced along their length
#    (about 3 min; writes map_search_mask.png)
python -m vesselmap faint out/map/map.json reference_data/burst_2026-09-16_15-50-52/frame_000020.tif -o out/faint

# 2c. measure red-cell velocity along every segment (limbusflow) and join the
#     segments that flow shows to be one vessel; the map comes from one image
python -m vesselmap flow out/faint/map.json --burst BURST_DIR --reference IMAGE -o out/flow

# 3. optional: one spline per vessel instead of one per segment
python -m vesselmap consolidate out/refined/map.json reference_data/burst_2026-09-16_15-50-52/frame_000020.tif -o out/vessels

# 4. adjust it to other (stabilised) frames: same ids, small moves
python -m vesselmap fit-frames out/vessels/map.json stabilized/frame_*.tif -o out/frames --chain --overlays

# 5. one HTML page of a run's outputs (images, graph files, per-frame fits)
python -m vesselmap report out/faint --image reference_data/burst_2026-09-16_15-50-52/frame_000020.tif --frames out/frames -o out/report

# score the mapper on synthetic images with known ground truth
python -m vesselmap synth-eval --seeds 0 1 2 -o out/synth
```

```python
from vesselmap import load_image, build_map, fit_frame, VesselNetwork

ref = build_map(load_image("frame_000020.tif"))
ref.save("map.json")
G = ref.to_digraph()                 # networkx.DiGraph, attributes below

net, report = fit_frame(load_image("stab_000021.tif"), ref)
```

`map` writes `map.json` (the full model), `map.graphml` and `map_edges.csv`
(the graph), `map_digraph.html` (interactive viewer: hover for attributes,
colour by diameter/blur/contrast), `map_digraph.png`, `map_overlay*.png` and
`map_model_residual.png` (image | rendered model | residual), and
`map_search_mask.png`: 0 = background, 1 = within r + 3 px of a mapped
vessel, 2 = within 5 px of a faint-tier path.

## The model

Vessels absorb light, so in the log domain overlapping vessels add:

    log I(x) = B(x) − Σ_e a_e(t) · P(d_e(x); r_e(t), s_e(t)) · T_e(x)

| symbol | meaning | representation |
|---|---|---|
| `B` | background (illumination, tissue) | bicubic grid, 64 px spacing, curvature penalty |
| `C_e(t)` | centreline of edge *e* | clamped cubic B-spline; control spacing adapts to calibre and tortuosity (5–20 px) |
| `r_e(t)` | lumen half-width | B-spline along the edge (30 px spacing) |
| `s_e(t)` | blur: defocus plus optics, so a depth cue | B-spline along the edge |
| `a_e(t)` | peak optical density (contrast) | B-spline along the edge |
| `P` | cross-section: chord length of a cylinder (the optical path through blood) convolved with the point-spread function | nested-box approximation, closed form with erf |
| PSF | `(1−h)·G(s) + h·G(√(s²+s_h²))`: a sharp core plus a tissue-scattering halo | `h`, `s_h` fitted once per image |
| `T_e` | end caps that fade along the axis with the same blur | two collinear caps sum to exactly 1, so joins are seamless |

**Score.** The score is the negative log-likelihood of the whole network:
weighted squared residuals, with per-pixel noise estimated from the image,
plus bending energy, smoothness of the profiles along each edge and smoothness
of the background. Every edge, and every overlap and junction, is scored
together. The renderer is written in PyTorch, so all splines are optimised
jointly with Adam. Each centreline is rendered from samples spaced evenly in
arclength, not in the spline parameter. Otherwise the fit can exploit gaps
where control points bunch up. On CPU the per-pixel kernel is fused with `torch.compile`
when a C++ compiler is available.

**Discovery** (`build_map`) runs coarse to fine over ridge-scale bands 10–20,
5–10, 2.5–5, 1.2–2.5 and 0.8–1.2 px. For each band:

1. Residual = image − current model. Vessels already explained are not
   proposed again. A thin vessel crossing a thick one still shows up.
2. Propose centrelines from multi-scale Hessian ridges of the residual. The
   ridge measure is anisotropic, so blobs are rejected. Each ridge is scored
   against the larger of sensor noise and the local texture level. The local
   texture level is the RMS of valleys of the opposite polarity: background
   texture makes valleys as often as ridges, but vessels make only ridges.
   Long, continuous ridges are kept even when no single pixel is strong,
   which catches faint defocused vessels.
3. Turn proposals into spline edges and join them to the network. Collinear
   gaps are bridged. A free end that runs into a vessel is snapped onto it as
   a bifurcation. At junctions, straight-through pairs become one continuous
   vessel, so a crossing is not mistaken for two bifurcations.
4. Jointly optimise every spline and the background. The background is frozen
   while the coarse bands are found; otherwise it absorbs wide vessels. It
   starts as a morphological upper envelope. Each edge's width and blur are
   capped relative to the band it came from, so a thin proposal cannot
   inflate into a blob that models background texture.
5. Score each edge by the drop in negative log-likelihood it is responsible
   for, and remove edges whose gain does not pay for their parameters (a BIC
   / MDL test). Also remove blob-shaped edges and edges that duplicate a
   stronger parallel edge.

**Refinement** (`refine_map`, `python -m vesselmap refine`) keeps the map
and adds fine detail:

* *Split test for parallel vessels.* A fork whose two branches then run side
  by side is often fitted as one wide vessel, or as one vessel with its
  neighbour missed. For every edge, the cross-section of what it explains
  (its own contribution plus the residual) is averaged in 16 px windows. A
  stretch where that profile has two separate dark peaks is replaced by two
  parallel edges that fork where they converge. All candidate splits are
  fitted jointly, and each one is kept only if it lowers the NLL in its own
  neighbourhood by more than the MDL cost of the extra edge.
* *Fine-scale rounds* with elongated oriented ridge filters, 12
  orientations and about 3× longer along the vessel than across it. They
  integrate along a vessel, so faint capillaries rise above the noise while
  speckle does not. Bands 1.6–2.6, 1.1–1.6 and 0.8–1.1 px each repeat
  until a round adds less than 60 px of centreline.
* A second split test, then a joint fit and pruning.

**Consolidation** (`consolidate_map`, `python -m vesselmap consolidate`)
turns the segment map into one spline per vessel. Discovery gives one edge
per *segment*: every edge ends where it meets another edge. A vessel that
gives off five branches is six edges. A vessel the detector lost for a
stretch, or found twice, is several edges with free ends. Consolidation
finds the edges that are one vessel and refits each group as a single edge:

* *Switch cuts.* An edge whose blur or calibre *steps* by more than 1.6×
  along it usually runs along one vessel and then switches to another at a
  different depth, where two vessels cross or touch. It is cut at the step
  (a two-segment change point in log blur and log width). Each piece can
  then continue into its own vessel. A gradual drift is not cut: blur
  changes smoothly along a vessel that changes depth. The step must fit the
  profile much better than a linear trend. On reference frame 20 this cuts
  36 of 368 edges.
* *Candidate continuations* between edge ends. Each end is described by the
  direction of the edge's body, its calibre, blur and contrast. The last
  few px next to a node are skipped, because fitted edges often hook into
  the node. A *node* link continues straight through a node (turn ≤ 40°,
  width within 1.8×, blur within 2.2×; blur is a depth cue). At a branch
  point it must beat every other pairing by a margin, so a symmetric fork
  is left alone. A *gap* link bridges a free end to an end ahead of it,
  within a 30° cone and up to 30–60 px. It also joins two nodes a few px
  apart (a junction found twice). An *overlap* link blends two free ends
  that run past each other on the same course.
* *Matching.* Each end continues into at most one other end, chosen by a
  maximum-weight matching over all candidates. The matched ends form chains
  of edges. Each chain is rebuilt as one path: junction hooks are replaced
  by a cubic Hermite bridge between the edge bodies, and overlaps are
  blended. The path is then refitted as one spline.
* *Branches stay attached.* A node where a branch leaves a consolidated
  vessel stays in the graph. The vessel passes *through* it
  (`info["through"]`), the branch still ends there, and a penalty in the
  renderer keeps the node on the vessel's centreline. Such a node counts as
  a bifurcation. Nodes left with nothing attached disappear.
* *Verification.* Segments and vessels are fitted jointly under the same
  priors, and every link is tested in its own neighbourhood. At a node the
  pieces already explain the image, so the NLL can barely tell one vessel
  from two there. The test only rejects a merge that clearly hurts the fit:
  the single spline must keep 95 % of the evidence (NLL drop) in the link's
  zone. The pieces' geometry and profiles, used when proposing the link,
  are the real evidence of continuity. Across a gap nothing explained the
  image before, so the data decide strictly. The fit there may not get
  worse by more than the MDL cost of the removed pair of ends. The bridge
  must also pay for its own length from the NLL drop of the bridged stretch
  alone, so two different vessels that happen to line up are not joined
  across empty tissue. Failed links are dropped and the matching is re-run,
  up to three rounds.

The calibre prior of these fits compares each width with the mean over
±4 profile knots (±120 px), not over the whole edge, so a long vessel
may taper. `to_segments()` and `to_digraph()` still give the segment
graph: every segment carries `vessel`, the id of the spline it belongs to.
`map_overlay_vessels.png` colours each vessel differently. `refine` accepts
a consolidated map and works on its segments; run `consolidate` again after
it.

**Per-frame fitting** (`fit_frame`) first aligns the map to the frame
globally. Phase correlation between the rendered vessel image and the frame's
high-passed image gives a translation with a large capture range: raw frames
of reference burst 1 drift up to about 60 px. A gradient-refined affine
follows. It then optimises all parameters with a
Gaussian prior that ties them to the map (default σ = 3 px for control
points, ±35 % for width and blur, ±60 % for contrast). Topology and ids are
unchanged. Each edge reports `visible` and `gain_per_px` for that frame. For
example, a capillary whose red cells have moved on may be only faintly
visible. `fit_frames` / `--chain` starts each frame from the previous frame's
result, while the prior stays on the map, so the fits cannot drift.

**The faint tier** (`faint.py`, `vesselmap faint`) serves the map's purpose
as a *search mask* for later velocity analysis. Recall matters more there
than a strict per-segment evidence test. Many thin, defocused vessels are
at the texture floor pixel by pixel. Along 42 hand-marked missed segments
of frame 20, the median ridge z-score lies between the 75th and 95th
percentiles of the background. No per-pixel threshold and no MDL test can
accept them, but their evidence adds up along their length. The tier works
on the residual of the map (what the map does not explain yet):

1. Oriented line responses Z(θ, x). The filters are elongated
   second-derivative filters (σ = 1.2, 2, 3 px; 6× longer than wide;
   16 orientations), standardised by a robust (MAD) local scale.
2. Seeds are local maxima with Z > 4 outside a zone of r + 2 px around
   mapped centrelines.
3. A tracker steps 2 px at a time. It turns by at most one orientation bin
   and may shift ±1 px sideways. It stops when the mean Z of the last 16 px
   drops below 1.2, when it reaches the map zone (a join) or when it reaches
   an accepted path.
4. The path is trimmed to its maximal-evidence stretch (maximum of
   Σ(Z − 2)), smoothed, and re-scored along its own tangent. This removes
   most of the bias of the greedy steps.
5. A path is kept if its mean Z ≥ max(2.5, 2.0 + 12/√L). The bound is
   calibrated on pure white and correlated noise: there it admits about 4
   paths per 1920×1200 image (`sig_a=2.2`: fewer than 1).
6. Paths that run parallel to a mapped vessel just outside its zone for
   most of their length are misfit edges and are dropped.
7. Ends that stopped at the map are bridged straight onto the nearest
   centreline, if the bridge continues the path's direction. The mapped
   edge is split there, so the tier is part of the graph.
8. Width, blur and contrast are fitted with positions frozen.

Faint edges carry `info["tier"] = "faint"` and the graph attribute
`tier` ("mapped" or "faint"). They show orange in `map_overlay_tier.png`
and under colour "tier" in the HTML viewer. `fit_frame` adjusts them like
any other edge. They are candidates for the velocity search, not confirmed
vessels: the velocity analysis decides.

**Flow-informed consolidation** (`flow.py`, `vesselmap flow`) is the one
part of vesselmap that uses velocity. It comes after the map and never
changes how vessels are detected, so the structural map stays independent
of the velocities it is later used to measure.

1. *Velocity per segment.* limbusflow registers the burst to the
   reference image. For every segment a kymograph is read along the
   centreline, averaged across the central half of the lumen.
   `limbusflow.velocity.estimate` gives the signed red-cell velocity:
   positive runs from the segment's node u to its node v. A result is
   reliable only when:
   * there is a clear correlation ridge;
   * two of three estimators (LSPIV ridge, structure tensor, time of
     flight) agree;
   * for slow flow, the result also persists across both halves of the
     recording.
2. *Candidates.* `consolidate.candidates` supplies the candidate joins:
   continuations through a node and across gaps, judged on direction,
   calibre and blur. At a fork, every feasible pairing is kept here, not
   only the unambiguous one.
3. *Flow tests at each joint*:
   * **direction**: blood runs into the joint along one segment and out
     along the other. Both in (a confluence) or both out (a fork) is
     never one vessel;
   * **speed**: the speeds agree within ×1.6. Beyond ×3 they are two
     vessels;
   * **transit**: the red-cell pattern leaving one segment reappears in
     the other, at a delay of distance / speed. The temporal correlation
     across the joint is compared with the correlation within the side
     whose pattern is clearer, over the same distances. Points within a
     few px of the joint are left out, because a branch leaving there
     overlaps the lumen. On synthetic flow the ratio is 1.07 for a
     continuation and 0.22 for a branch; the threshold is 0.6.
4. *Decision.*
   * A join that flow **confirms** is made even at an ambiguous fork.
   * One it **contradicts** is not made, even when shape agrees.
   * Where flow is unknown (too slow, too short, out of focus), the shape
     rule and image test of `consolidate` apply.
   * Segments not joined stay attached where they meet the vessel.

Each joined vessel lists the evidence for each link in `info["links"]`.
Each edge carries its measured flow in `info["flow"]`, and edges with a
reliable measurement point along the flow (`orientation = "flow"`).
Outputs: `map_overlay_flow.png` (speed on a log scale, grey = unknown) and
`map_flow.csv`. In the HTML viewer, "flow speed" is a colour option.

## The graph

Node kinds: `bifurcation` (degree 3), `junction` (degree ≥ 4), `endpoint`
(a vessel fades or leaves the focal volume), `border` (a vessel leaves the
image) and `joint`. A `joint` is degree 2: two segments meet at a sharp
angle, usually where a branch was too faint to keep. They stay as two
splines, because fusing them would force a cusp into one. Crossings of vessels at different depths are not nodes. The two
vessels overlap and their densities add, and the crossings are listed in
`G.graph["crossings"]`.

Edge attributes: `length`, `chord`, `tortuosity`, `diameter` (mean, min,
max), `blur`, `contrast`, `mean_curvature`, `max_curvature`, `gain`
(evidence), `orientation`, and `vessel` (the spline the segment belongs to;
after consolidation several segments share it). The full spline parameters
are in `map.json`.

**Direction.** Flow cannot be seen in a still image. Edges are oriented by a
structural convention: away from the widest vessel of each connected
component, from wide to narrow, with BFS depth breaking ties. Every edge
carries `orientation="structural"`. `VesselNetwork.reverse_edge` and
`set_direction` let an independent measurement override it later.

## Results

**Synthetic scenes with ground truth.** `python -m vesselmap synth-eval`
renders 768x512 scenes independently of the fitting model: super-sampled
cylinders, per-vessel depth blur of 0.6–9 px, bifurcating trees, defocused
vessels crossing everything, tortuous capillaries of radius 0.6–1.4 px, lumpy
background texture, illumination fall-off, and shot and read noise. A true
centreline point counts as found within max(2 px, its radius):

| scene | recall | precision | recall by radius <1 / 1–2 / 2–4 / ≥4 px | recall on blur ≥4 px | time |
|---|---|---|---|---|---|
| seed 0 | 0.93 | 0.81 | 0.92 / 0.90 / 0.93 / 0.97 | 0.92 | 4.0 min |
| seed 1 | 0.94 | 0.71 | 0.84 / 0.95 / 0.95 / 0.98 | 0.96 | 4.4 min |
| seed 2 | 0.95 | 0.85 | 0.88 / 0.91 / 0.99 / 1.00 | 1.00 | 3.2 min |

The misses are mostly sub-pixel capillaries along tight curls. The false
positives are mostly very blurred "vessels" fitted to dark lumps of the
synthetic background texture. This is the one ambiguity that intensity alone
cannot resolve completely. A wide edge must therefore also beat a smooth
background explanation (see step 5), which removes most of them.

**Refinement on the same scenes** (current code, `build_map` then
`refine_map` with `max_reps=6`):

| scene | recall, map → refined | recall at radius < 1 px | recall at radius 1–2 px | precision |
|---|---|---|---|---|
| seed 0 | 0.875 → 0.904 | 0.80 → 0.85 | 0.87 → 0.88 | 0.79 → 0.73 |
| seed 1 | 0.862 → 0.890 | 0.63 → 0.73 | 0.88 → 0.89 | 0.73 → 0.69 |
| seed 2 | 0.871 → 0.914 | 0.73 → 0.81 | 0.74 → 0.82 | 0.83 → 0.79 |

Refinement mostly adds the thinnest vessels. It costs 4–6 points of
precision, because some faint proposals are texture.

These `map` numbers are lower than the table above, which was measured
before the arclength renderer. On seed 0, repeated runs give recall 0.85–0.89
now against 0.93–0.95 then, with precision about 0.80 now against 0.76 then.
The recall now lost is mostly short stretches of wide, blurred vessels where
they cross others. Single runs vary by a few points: builds are not
bit-for-bit deterministic, because multithreaded sums change the order of
discrete decisions.

**Consolidation on the same scenes.** `synthetic.vessel_metrics` measures
how completely single splines annotate single vessels. In the ground truth a
vessel keeps its identity through a fork: the parent runs on and the branch
is a separate vessel. *Cover* is the fraction of each true vessel covered by
its single best spline, length-weighted; recall is its upper bound.
*Purity* is the fraction of each spline's length on its main vessel.
Scenes are mapped with `build_map`, then `consolidate_map` is run with
defaults:

| scene | splines | edges per vessel | cover | purity | recall | precision | links (node / gap) | time |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 121 → 106 | 4.53 → 3.50 | 0.41 → 0.47 | 0.86 → 0.86 | 0.87 → 0.86 | 0.77 → 0.76 | 22 / 1 | 82 s |
| seed 1 | 131 → 122 | 3.06 → 2.54 | 0.57 → 0.63 | 0.86 → 0.85 | 0.93 → 0.93 | 0.75 → 0.74 | 24 / 1 | 83 s |
| seed 2 | 103 → 102 | 3.53 → 3.02 | 0.52 → 0.59 | 0.83 → 0.88 | 0.90 → 0.90 | 0.81 → 0.80 | 11 / 0 | 63 s |

*Edges per vessel* is the number of splines that annotate each true vessel,
length-weighted (1 is ideal). Most of the remaining fragmentation in these
scenes comes from tangled traces of tortuous capillaries in the segment
map. Their pieces turn by more than 40° or change calibre where they
meet, so no link joins them. Consolidation cannot repair a trace that
follows the wrong path.

**LIMBUS reference burst 1, frame 20** (1920x1200, 12-bit). NLLs are
from the arclength renderer, so they are comparable with each other, not
with numbers from before that change.

| | `map` | `map` + `refine` |
|---|---|---|
| edges / centreline | 443 / 57,800 px | 578 / 68,900 px |
| centreline of vessels < 6 px in diameter | 3,900 px | 12,000 px |
| parallel-vessel splits kept | – | 31 |
| crossings of vessels at different depths | 158 | 283 |
| data NLL | 1.566M | 1.474M |
| unexplained ridges in the residual, finest band | 10,000 px | 7,600 px |
| time on 4 CPU cores | 41 min | +64 min |

The earlier map made before the arclength renderer (65,300 px) scores
1.694M under it. It carried centreline that exploited sampling gaps
rather than explaining the image.

* Fitted calibre after refinement: diameter 3.8–30 px (5th–95th
  percentile, median 9.8 px). Blur σ 1.7–16 px, spanning sharp capillaries
  to deep, strongly defocused vessels. Contrast 0.05–0.42 OD.
* The rendered model reproduces the frame closely, and the residual has no
  wide vessels left in it (`map_model_residual.png`).

*Consolidation* of a fresh `map` of this frame (without `refine`) took
14.5 min. It merged 368 segments into 324 splines: 58 vessels from several
pieces, with 77 node links and 3 bridged gaps. 16 of 97 matched links failed
verification in the first round. Half of all centreline now lies in splines
of at least 256 px (202 px before), and 51 splines are longer than 300 px
(35 before). The data NLL fell from 1.627M to 1.620M, so the vessels fit
the frame slightly better than their pieces did.

**Faint tier on frame 20.** Evaluated against 42 strokes drawn by hand
along segments the map missed (5021 px). The strokes are rough guides, not
centrelines, so coverage is measured with a tolerance:

| map | centreline length | strokes within 6 px | within 8 px | inside the search mask | strokes > 80 % / < 20 % covered (6 px) |
|---|---|---|---|---|---|
| base `map` (the liked map) | 65.3k px | 21 % | 28 % | 36 % | 1 / 19 |
| `refine` of a later base map | 68.9k px | 44 % | 50 % | 54 % | 10 / 10 |
| base + `faint` | 74.7k px (+9.4k faint) | 72 % | 79 % | 83 % | 17 / 1 |

`faint` took 2.5 min. A second `faint` pass on its own output adds only
1.2k px (73 %). The segments still missed lie mostly within a few px of
a mapped vessel, inside its zone. For the mask they are covered: 83 % of
stroke length lies inside it.

**Flow-informed consolidation on burst 15-22-26** (1105 frames at 74 fps,
1920×500). limbusflow registered the burst in two passes: 942 frames are
good, and the longest unbroken run is frames 503–880 (378 frames, 5.1 s).
The map was built from one registered frame, 611, chosen as the sharpest
in that run. The registered mean of all frames is the wrong input: it has
almost no noise, so tissue texture passes the mapper's noise-relative
tests and is traced as vessels. The flow step took 4.8 min:

| | count |
|---|---|
| segments measured / with a reliable velocity | 292 / 69 (median 176 px/s) |
| candidate joins by shape / including ambiguous forks | 84 / 96 |
| flow confirms / contradicts / cannot tell | 26 / 1 / 69 |
| joins made: by shape and flow / by shape alone | 24 / 53 (5 shape-only joins failed the image test) |
| edges before → after | 278 → 225 |

Two rule refinements came from this burst:
* **Transit is compared with the clearer side.** A side with no
  measurable pattern had inflated the ratio when the two sides were
  pooled.
* **Conflicting measurements don't rule out a join.** A large vessel
  whose two segments got opposite signs, while the pattern carried
  straight across, is left to the shape rule.

The one rejection is a crossing where two segments flow in from opposite
sides.

**Per-frame fitting on the same burst** (raw, *unstabilised* frames, so
harder than the intended use; `fit-frames --chain`; this table was measured
with the earlier map and renderer, so its NLLs are on that scale):

| frame | global shift (px) | NLL | node adjustment after alignment, median / p95 | visible edges | time |
|---|---|---|---|---|---|
| 20 (map) | – | 1.585M | – | – | – |
| 21 | (1.6, −2.3) | 1.60M | 0.8 / 2.0 px | 99 % | 2.5 min |
| 40 | (−11.1, −12.4) | 1.73M | 1.1 / 3.2 px | 98 % | 2.0 min |
| 60 | (7.5, −31.9) | 1.76M | 1.4 / 4.2 px | 98 % | 2.0 min |
| 80 | (14.3, −33.0) | 1.77M | 1.6 / 5.1 px | 98 % | 2.0 min |
| 100 | (40.8, −27.3) | 1.88M | 2.0 / 5.7 px | 94 % | 2.0 min |
| 120 | (51.0, −18.4) | 1.67M | 2.3 / 6.2 px | 81 % (part of the map has drifted out of the frame) | 2.0 min |

The NLL of the map on its own frame is the scale to compare against: every
frame fits about as well as the frame the map was built from.

## Limitations

* Since the switch to the arclength renderer, the base `map` finds fewer
  short stretches of wide, blurred vessels where they cross others on
  synthetic scenes (see *Refinement on the same scenes*). `refine` adds fine
  vessels but does not re-examine coarse bands.

* The densest region, with several vessels overlapping at different depths,
  still has a few traced paths that switch between neighbouring vessels.
  Use `map_model_residual.png` and the HTML viewer to review them.
* Heavy tissue texture can be mistaken for very blurred deep vessels.
  Raising `penalty_scale` or `wide_test_w` trades recall for precision.
* Widths below about 1.5 px are degenerate with blur: the product of
  contrast and width is well determined, but the split between them is not.
* Direction is a structural convention, not a flow measurement.
* Flow evidence helps mostly with small vessels and capillaries. Large
  vessels carry a dense red-cell column, and at 74 fps the 12.8 ms
  exposure smears their fast flow, so they rarely show a trackable
  pattern. There the shape rule decides.
* The faint tier trades precision for recall by design. Its threshold
  admits a few paths per image from noise alone, and bright or dark
  banding next to badly fitted wide vessels can still produce a path.
  A faint vessel that runs within r + 2 px of a mapped vessel is not
  traced separately; it is only covered by that vessel's mask.
* Consolidation joins pieces only where they continue each other smoothly
  (turn ≤ 40°) with similar calibre and blur. A vessel whose segments were
  traced with kinks or along the wrong neighbour stays in pieces, and an
  unambiguous but wrong continuation at a fork is still possible where a
  branch leaves nearly straight.

## Files

`image.py` loading and the log domain · `ridges.py` proposals · `spline.py`
B-splines · `network.py` graph and topology · `render.py` differentiable
renderer and score · `fit.py` discovery and per-frame fitting ·
`refine.py` fine detail · `faint.py` faint tier and search mask · `flow.py` velocity and flow-informed consolidation · `report.py` HTML report of a run ·
`consolidate.py` one spline per vessel · `draw.py`
figures and HTML · `synthetic.py` ground-truth scenes and metrics ·
`tests/` (`python -m pytest vesselmap/tests`; add `-m "not slow"` for the
fast ones).
