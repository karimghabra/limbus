# limbusflow

Spline-based vessel annotation, morphometry and red-blood-cell velocimetry for limbal / scleral video,
with the network stored as a directed flow graph.

* `limbus_vessel_workflow.ipynb`: the step-by-step educational walkthrough, with the maths for every stage (start here)
* `limbusflow/`: the library the notebook calls, one module per stage (see `limbusflow/__init__.py`)
* `build_notebook.py`: regenerates the notebook from source

The notebook is committed **without outputs**. Its figures and videos show the subject's eye, and captured
images stay off the repository (see the root `.gitignore`). Run it to regenerate everything.

## Pipeline

| stage | method |
|---|---|
| registration | SIFT + RANSAC similarity, ECC affine refinement, per-frame quality flags |
| detection | structural Frangi vesselness (mean image) + functional vesselness from the jitter-corrected temporal flicker of the registered video (finds thin perfused vessels) |
| graph | skeleton → nodes/segments, spur pruning, junction merging, crossing detection |
| centre-lines | smoothing cubic B-splines, arc-length parameterisation, Frenet frame, curvature |
| diameter | FWHM of normal intensity profiles; blurred-box model fit as a shape check |
| tortuosity | DM, SOAM, mean κ², inflection count |
| velocity | kymographs → spatio-temporal correlation (LSPIV), structure tensor and time of flight; time-binning pyramid for slow flow, with a split-half persistence test |
| flow graph | `networkx.MultiDiGraph` oriented by measured flow; Kirchhoff mass conservation to infer missing directions and check consistency |
| validation | synthetic phantoms (diameter, curvature, flow 0.05–11 px/frame), negative control in static tissue, co-moving kymographs, tracer video |

## Data

The notebook expects, next to itself or in the repository's `reference_data/` folder:

* `mean_stabilized.tif`: the registered mean image of the burst (the reference)
* `burst_2026-09-16_15-22-26/`: the raw TIFF burst with `frames.csv` and `manifest.json`

Change `REFERENCE` / `BURST` in the first code cell to use another recording.

## Setup

From this folder:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.venv/Scripts/python.exe -m jupyter lab limbus_vessel_workflow.ipynb
```

A full run takes about 10 minutes. The first run also registers all frames (about 5 minutes) and caches the result in `cache/`.
Exports (`.json`, `.graphml`, `.csv`) and validation videos go to `outputs/`. Both folders are git-ignored.

## Minimal use from code

```python
from limbusflow import io, register as rg, network as nw
ref, valid = io.load_reference("mean_stabilized.tif")
burst = io.load_burst("burst_2026-09-16_15-22-26")
reg = rg.register_cached(ref, valid, burst.frames, "cache/registration.npz")
a, b = reg.good_runs()[0]
net = nw.run_all(ref, valid, burst.frames, reg, range(a, b + 1), burst.fps, um_per_px=None)
net.table()                     # one row per vessel
net.dg                          # networkx.MultiDiGraph oriented along the flow
net.export("outputs/limbus_network")
```

Units are pixels, px/s and px³/s unless `um_per_px` is given (sensor pitch 5.86 µm divided by the optical magnification).
