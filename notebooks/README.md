# Analysis notebooks

`foundation.py` is the foundation of the quantitative analysis: one burst, from its TIFF frames to the
vessel graph and the flow along each vessel. It is a [marimo](https://marimo.io) notebook, a plain
Python file that diffs like code, runs as a script or an app, and is edited in the browser. To use it
from another computer, live, see [LIVE.md](LIVE.md).

## What is in it

Sections 0–25, as the notebook's own table of contents and section texts describe them:

| § | | |
|---|---|---|
| 0–9 | setup, loading, frame gating, motion, fixations, alignment, dust and glare, stabilization, mean frames | |
| 10–13 | **the background**: fit and optical density, quiet pixels, the refit on the quiet pixels, the shading from frame to frame | the newer version, see below |
| 14–16 | the smear of the temporal burst, its background, the two vessel pictures combined | |
| 17–25 | orientation field, hyperspline, tracing, the field as one system with its junctions, continuations, refinement, coarse to fine, flow, the vessel graph | |

The notebook was put together from two recovered analysis notebooks, which are not in this repository:
the notebook of 2026-10-26 with sections 0–25, and the preprocessing notebook of 2026-10-30, a later
copy of its sections 0–13. Sections 0–9 are the same in both, line for line. In sections 10–13 the
later notebook fits the background differently, and those four sections are taken from it, text and
code; everything else is the 2026-10-26 notebook's:

| § | Cells taken from the 2026-10-30 notebook | What changed |
|---|---|---|
| 10 | the section text, the `fit_background` cell, the per-fixation fit cell and its text | every background that is used is a thin plate (`thin_plate_background`, the surface of least bending through the background pixels, `BACKGROUND_WAVELENGTH = 30` px) instead of a local mean, which breaks into a mosaic where background pixels are sparse; the local mean (`BACKGROUND_SIGMA`) still builds the vessel mask |
| 11 | the section text and cell | the slow movement, each pixel's running mean over `SLOW_WINDOW = 9` frames, is removed before the fluctuation is measured (`fast_changes`), so slowly drifting patches between the vessels are no longer taken for flow |
| 12 | the section text and cell | the refit on the quiet pixels is a thin plate as well, and a quiet pixel that stands out as a vessel (`VESSEL_GUARD = 3` times the noise) is kept out of the background |
| 13 | the section text, the cell and the preview cell | `slow_movement` and `fast_frame`; the preview compares the per-frame background with and without the slow movement |

Outputs are not kept in the file: run the notebook to see them.

### As a marimo notebook

marimo runs a cell whenever what it reads has changed, so a name may be defined by one cell only. The
code is otherwise the notebooks' own, with these changes:

- Loop variables, figures and other names that several cells assign are private to each cell, with a
  leading underscore (`_number`, `_figure`, `_axes`, …). A cell that later cells read a name from keeps
  it as it was.
- The `import scipy.…` lines of §17 and §23 are in §0, with the other imports.
- §23's own `blur_valid` (a blur over the pixels with data, unlike §11's) is called `fine_blur_valid`.
- §24 updates `haze` in place (`haze[...] = …`), where it assigned it anew.
- §1 loads the burst named by the environment variable `LIMBUS_BURST`, when it is set; otherwise it
  searches as before.

Nothing is lost against the Jupyter version: `marimo export ipynb notebooks/foundation.py -o
foundation.ipynb` writes the same cells, in the same order, as a Jupyter notebook (for Colab, for
example).

## Running it

```bash
notebooks/live.sh install                     # the environment, in notebooks/.venv (see LIVE.md)
source notebooks/.venv/bin/activate
export LIMBUS_BURST="/mnt/d/Conjunctiva Code/limbus/recordings/burst_2026-09-16_15-50-52"
cd ~/limbus-work                              # the notebook writes previews/, vessel_pictures/, neural_cache/ here
marimo edit /path/to/limbus/notebooks/foundation.py      # in the browser
marimo export html /path/to/limbus/notebooks/foundation.py -o foundation.html   # run it all, keep the outputs
```

Without `LIMBUS_BURST`, the first `burst_*` folder in the working directory or one of its parents is
loaded. `notebooks/pyproject.toml` sets marimo up for a notebook this heavy: opening it runs nothing,
and a changed cell marks the cells after it stale instead of running them.

### Memory and time

Measured on the lab PC with 6 threads, burst `2026-09-16_15-50-52` (140 frames of 1920 × 1200):

- The whole frame: §0–§16 run in about 2½ minutes. §17 computes and saves the orientation field and
  then needs more than 6 GB; from the run on the crop below, the whole notebook needs roughly 8–11 GB.
- A 720 × 1152 crop of the same burst (all 140 frames): every section, §0–§25, runs without an error
  in 3 minutes, at a peak of 3.9 GB.

`notebooks/live.sh` gives the notebook server at most 12 GB (`LIMBUS_LIVE_MEMORY_MAX`) and 6 CPUs
(`LIMBUS_LIVE_THREADS`), so that a large run cannot starve the other work on the PC.
