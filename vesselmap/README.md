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

# 1. discover the network in one frame (about 15–30 min for 1920x1200 on 4 CPU cores)
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

## Validation

See *Results* below. Validation uses synthetic scenes with ground truth
(`synthetic.py` renders vessels independently of the fitting model:
super-sampled cylinders, per-vessel depth blur, lumpy background texture, and
shot and read noise), plus the LIMBUS reference bursts.

## Files

`image.py` loading and the log domain · `ridges.py` proposals · `spline.py`
B-splines · `network.py` graph and topology · `render.py` differentiable
renderer and score · `fit.py` discovery and per-frame fitting · `draw.py`
figures and HTML · `synthetic.py` ground-truth scenes and metrics ·
`tests/` (`python -m pytest vesselmap/tests`; add `-m "not slow"` for the
fast ones).
