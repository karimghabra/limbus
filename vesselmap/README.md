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

# 2. adjust it to other (stabilised) frames: same ids, small moves
python -m vesselmap fit-frames out/map/map.json stabilized/frame_*.tif -o out/frames --chain --overlays

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
`map_model_residual.png` (image | rendered model | residual).

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
jointly with Adam. On CPU the per-pixel kernel is fused with `torch.compile`
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
(evidence), `orientation`. The full spline parameters are in `map.json`.

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

**LIMBUS reference burst 1, frame 20** (1920x1200, 12-bit):

* 524 vessel edges and 65,300 px of centreline. Nodes: 201 bifurcations,
  56 endpoints, 50 border exits and 170 joints. 214 crossings of vessels at
  different depths. Built in 34 min on 4 CPU cores.
* Fitted calibre: diameter 4.7–35 px (5th–95th percentile, median 11 px).
  Blur σ 1.3–11.5 px, spanning sharp capillaries to deep, strongly
  defocused vessels. Contrast 0.08–0.42 OD. Global scattering halo: weight
  0.55, σ 3.9 px.
* The rendered model reproduces the frame closely, and the residual has no
  wide vessels left in it (`map_model_residual.png`).
* Development history on this frame, measured by the final data NLL
  (lower is better): 1.66M in an early version, with 32 self-loops and 236
  kinked edges. Then 1.81M after the anti-cusp fixes, before recall was
  restored. Then 1.71M, then 1.65M, and finally 1.585M, with no self-loops.

**Per-frame fitting on the same burst** (raw, *unstabilised* frames, so
harder than the intended use; `fit-frames --chain`):

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

* The densest region, with several vessels overlapping at different depths,
  still has a few traced paths that switch between neighbouring vessels.
  Use `map_model_residual.png` and the HTML viewer to review them.
* Heavy tissue texture can be mistaken for very blurred deep vessels.
  Raising `penalty_scale` or `wide_test_w` trades recall for precision.
* Widths below about 1.5 px are degenerate with blur: the product of
  contrast and width is well determined, but the split between them is not.
* Direction is a structural convention, not a flow measurement.

## Files

`image.py` loading and the log domain · `ridges.py` proposals · `spline.py`
B-splines · `network.py` graph and topology · `render.py` differentiable
renderer and score · `fit.py` discovery and per-frame fitting · `draw.py`
figures and HTML · `synthetic.py` ground-truth scenes and metrics ·
`tests/` (`python -m pytest vesselmap/tests`; add `-m "not slow"` for the
fast ones).
