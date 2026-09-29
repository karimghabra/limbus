# Segmenters: the vessel overlay's swappable part

The Review tab draws a vessel overlay over playback — centrelines, widths,
junctions and mask layers — on the stabilized views, and on raw frames
carried through each frame's stabilization. *What* it draws comes from a
**segmenter**: one Python file in this folder's `plugins/`. The app knows
nothing about how any segmenter works. It lists the files it finds, runs the
chosen one in a separate process, and reads back one file, `overlay.json`.
So a segmenter can be rewritten, replaced or added without touching the app.

## Using one

In the Review tab, select a burst and a stabilization method (stabilize it
first), choose a **Segmenter** and press **Find vessels**. Progress appears in
the log below; **Cancel** stops it. Each segmenter keeps its own result for
each stabilization result, so switching the dropdown switches overlays
instantly — handy for comparing two. The status line warns when an overlay was
made on an earlier stabilization, or by an earlier version of the segmenter's
file: press **Re-run** to update it.

From the command line (in `analysis/`):

```bash
python -m segmentation list
python -m segmentation run vesselmap ../stabilization/nonrigid/burst_2026-09-16_15-50-52
```

Results go to `<stabilization result>/segmentation/<segmenter>/`: the
`overlay.json` the app reads, its mask PNGs, and whatever else the segmenter
writes there. A run works in `<segmenter>.running/` and replaces the previous
result only when it succeeds, so a cancelled or failed run keeps the last
overlay (a failed run's partial files stay in `.running/` to look at).

## Writing one

A segmenter is a `.py` file with a `segment(inputs)` function that returns an
overlay. Minimal example:

```python
"""One line on what this finds (the first docstring line is the fallback description)."""
LABEL = "My segmentation"               # shown in the Review tab's dropdown
DESCRIPTION = "What it does, how long it takes, what it needs."   # its tooltip
VERSION = "1"                            # recorded in each overlay


def segment(inputs):
    import numpy as np                   # import heavy packages here, not at the top
    mean = inputs.mean()                 # float32 (H, W), sensor DN, NaN = no data
    ov = inputs.overlay()                # empty, the size of the stabilized frame
    # ... find vessels ...
    ov.add_vessel([[x0, y0], [x1, y1], ...], radius=[r0, r1, ...])
    ov.add_junction(x, y, "bifurcation")
    ov.add_mask("vessels", mask)         # optional raster layer, bool (H, W)
    inputs.log("found N vessels")        # progress, shown in the app's log
    return ov
```

Save it in `plugins/` and press **⟳ Refresh** in the Review tab: it appears in
the dropdown. The file's id — its name without `.py` — names its results
folder. Files starting with `_` are ignored, so helpers can live beside the
segmenters.

**What `inputs` gives you** (`inputs.py`):

| | |
|---|---|
| `mean()` | the stabilized mean, float32, sensor DN, NaN where too few frames saw the tissue |
| `valid()` | where the mean has data |
| `vessel_mask()` | the stabilization's own vessel mask (what it registered on) |
| `frames(used_only=True)` | the burst's frames as stabilized, `(index, frame)`, for segmenters that want time (red-cell flicker, say); needs the raw burst |
| `shape`, `method`, `metrics` | frame size, stabilization method, the result's `metrics.json` |
| `result_dir`, `burst_dir` | the stabilization result and the raw burst (None if not found) |
| `out_dir` | a folder for your own files |
| `log(text)` | progress for the app's log |

**What the overlay holds** (`overlay.py`, which documents the file format):
coordinates are full-resolution pixels of the stabilized frame — the frame of
`mean_stabilized.tif` — with pixel centres on whole numbers.

- `add_vessel(points, radius=None, id=None, group=None, **props)`: a
  centreline of ≥ 2 points; `radius` per point (the half-width; draws the
  walls); `group` gives segments of one vessel one colour; anything else is
  kept in `props`.
- `add_junction(x, y, kind, vessels=())`: the app has markers for
  `bifurcation`, `junction`, `crossing`, `crossing under`, `overlap`,
  `unresolved` and `endpoint`; any other kind gets a white diamond, labelled
  with its name in the legend.
- `add_mask(name, mask)`: drawn as a tint on the stabilized views only (a
  mask can't follow a raw frame's correction the way a line can).
- `ov.summary = "..."`: the status line (a count of vessels and junctions if
  unset).

**Developing outside the repository.** Put segmenters in any folder and list
it in the `LIMBUS_SEGMENTERS` environment variable (several separated by `;`
on Windows, `:` elsewhere). Those folders come first, so a copy there of a
built-in segmenter, edited, stands in for it without changing the repository.

**Other languages or environments.** `segment()` can run anything — another
Python environment, a compiled program — with `subprocess`, as long as it
returns an overlay built from the result.

The contract is tested by `tests/test_segmentation.py`, and the Review tab's
use of it by `benchmarks/test_review_segmentation.py`.

## The built-in segmenters

- **`stabilization_mask`** — *Stabilization mask (quick)*: the vessel mask the
  stabilization registered on, thinned to centrelines, widths from the
  distance to its edge. About a second. A quick look and a check of the
  overlay, not a measurement: the mask merges close vessels and misses faint
  ones.
- **`vesselmap`** — *vesselmap (spline network)*: vesselmap's discovery
  (`vesselmap/README.md`) on the stabilized mean; each vessel a fitted spline
  with its own width, blur and contrast, plus branch points and crossings.
  About an hour for a full 1920×1200 frame on the CPU (65 min for
  `15-50-52` on a 6-core Ryzen 5 5600; about 2 min for a 1920×100 strip);
  needs `requirements-vesselmap.txt`. vesselmap's own outputs (`map.json`, the
  interactive `map_digraph.html`, overlays, `map_edges.csv`) are written
  beside the overlay. Its settings are at the top of the file: `MAP_SETTINGS`
  (vesselmap `MapConfig` values), `STEPS` (add `"refine"`, `"faint"`,
  `"consolidate"`), `WRITE_VESSELMAP_OUTPUTS`.
