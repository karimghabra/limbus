# Analysis notebooks

| Notebook | |
|---|---|
| `foundation.py` | **The preprocessing**: one burst, from its TIFF frames to the cleaned, stabilized pictures that vessel annotation starts from, saved to one file at the end |
| `flow.py` | **Flow without annotation**: why subtracting each pixel's temporal mean makes flow visible, the perfusion map, the centrelines of the perfused vessels, and the red-cell velocity along each from kymographs, cross-checked by 2D cross-correlation |
| `archive_vessel_annotation.py` | The vessel annotation that used to follow the preprocessing (sections 17–25), kept unchanged for reference; a different annotation algorithm is planned |

They are [marimo](https://marimo.io) notebooks: plain Python files that diff like code, run as scripts
or apps, and are edited in the browser. To use them from another computer, live, see [LIVE.md](LIVE.md).

## The foundation: preprocessing

Sections 0–17, one per step, each opening with what the step does, its settings and what it hands on:

| § | | |
|---|---|---|
| 0–9 | setup, loading, frame gating, motion, fixations, alignment, dust and glare, stabilization, mean frames | |
| 10–13 | **the background**: fit and optical density, quiet pixels, the refit on the quiet pixels, the shading from frame to frame | the newer version, see below |
| 14–16 | the smear of the temporal burst, its background, the two vessel pictures combined | |
| 17 | **the result**, saved to `preprocessed/<burst folder>.npz` | 10 arrays |

§17 saves what vessel annotation needs to `preprocessed/<burst folder>.npz` in the working directory: per fixation the still picture (`densities`, filled where there is no data, so mask it with `usable_pixels`), the cleaned smear, the two combined (`combined_sums`, which needs `usable_pixels` as well as its `NaN`) and the usable, background and quiet masks, with the maps between the fixations; the notebook's §17 text lists the keys.

### Where it comes from

The notebook was put together from two recovered analysis notebooks, which are not in this repository:
the notebook of 2026-10-26 with sections 0–25, and the preprocessing notebook of 2026-10-30, a later
copy of its sections 0–13. Sections 0–9 are the same in both, line for line. In sections 10–13 the
later notebook fits the background differently, and those four sections are taken from it, text and
code; sections 14–16 are the 2026-10-26 notebook's:

| § | Cells taken from the 2026-10-30 notebook | What changed |
|---|---|---|
| 10 | the section text, the `fit_background` cell, the per-fixation fit cell and its text | every background that is used is a thin plate (`thin_plate_background`, the surface of least bending through the background pixels, `BACKGROUND_WAVELENGTH = 30` px) instead of a local mean, which breaks into a mosaic where background pixels are sparse; the local mean (`BACKGROUND_SIGMA`) still builds the vessel mask |
| 11 | the section text and cell | the slow movement, each pixel's running mean over `SLOW_WINDOW = 9` frames, is removed before the fluctuation is measured (`fast_changes`), so slowly drifting patches between the vessels are no longer taken for flow |
| 12 | the section text and cell | the refit on the quiet pixels is a thin plate as well, and a quiet pixel that stands out as a vessel (`VESSEL_GUARD = 3` times the noise) is kept out of the background |
| 13 | the section text, the cell and the preview cell | `slow_movement` and `fast_frame`; the preview compares the per-frame background with and without the slow movement |

§15 calls `fit_background` with `wavelength=None`, which keeps the smear's background the 64 px local
mean of the 2026-10-26 notebook; with §10's newer `fit_background` it would otherwise have become a
30 px thin plate.

The computations are the notebooks' own. The texts no longer point to notebooks and files that are not
here, the Google Colab set-up is gone (marimo runs locally), and `frame_density` and `burst_density`
keep only the path the notebook uses (frames already stabilized). As a marimo notebook, a name may be
defined by one cell only, so:

- Loop variables, figures and other names that several cells assign are private to each cell, with a
  leading underscore (`_number`, `_figure`, `_axes`, …).
- §12 refits §10's `background_fits`, `density_fixations`, `density_offsets` and `density_means` in
  place (the two measurements never sit in memory together) and returns them as
  `final_background_fits`, `final_density_fixations`, `final_density_offsets` and
  `final_density_means`; the later cells read those names, so marimo knows what depends on §12.
- §1 loads the burst named by the environment variable `LIMBUS_BURST`, when it is set; otherwise the
  first `burst_*` folder in the working directory or one of its parents.

`marimo export ipynb notebooks/foundation.py -o foundation.ipynb` writes the same cells, in the same
order, as a Jupyter notebook.

## Flow: where blood moves, and how fast

`flow.py` runs `foundation.py` inside it (`app.embed()`, with the preprocessing's figures and previews left out)
and works on one fixation's stabilized density frames. It needs no vessel annotation:

| § | |
|---|---|
| 1 | Subtracting the temporal mean, measured: signal, noise and clutter of the moving and static pixels before and after, against a rolling mean and median, and the temporal standard-deviation map. It raises the contrast of the moving signal against the static clutter, not the signal-to-noise ratio of a pixel; the real gains come from pooling pixels and frames |
| 2 | The perfusion map: the temporal variance of the video high-passed in time (`DRIFT_WINDOW`), levelled for shot noise and thresholded into perfused regions |
| 3 | Centrelines: the skeleton of the perfused regions, cut at junctions into segments |
| 4 | Kymographs along every segment and the red-cell velocity from the slope of their streaks (a multi-lag slope search, the Radon transform without its noise term), with block-of-time uncertainties and a structure-tensor estimate |
| 5 | A cross-check by ensemble 2D cross-correlation of the frames around each segment |
| 6 | The velocity map, the per-segment table, and what the frame rate and exposure allow |
| 7 | Videos, figures and tables saved to `flow_outputs/<burst folder>/` in the working directory |

Velocities are in px/frame and px/s; set `UM_PER_PX` (5.86 µm divided by the optical magnification) for mm/s.
The previews it shows in the notebook are kept under 2.5 MB each; the full side-by-side video is saved as WebM.

## The archive: vessel annotation

`archive_vessel_annotation.py` holds sections 17–25 of the combined notebook, unchanged, and the notes
that led into them: the orientation field, the hyperspline, tracing, the field as one system with its
junctions, continuations, refinement, coarse to fine, flow, and the vessel graph. A different vessel
annotation algorithm is planned, so they are kept for reference rather than developed.

Its first cells run `foundation.py` inside it (`app.embed()`) and pass on the 10 names that sections
17–25 read from it; nothing else was added. *Run all* in the archive therefore runs the preprocessing first. In
its marimo form, §23's own `blur_valid` (a blur over the pixels with data, unlike §11's) is called
`fine_blur_valid`, and §24's haze after the joint levels is `joint_haze`.

## Running them

```bash
notebooks/live.sh install                     # the environment, in notebooks/.venv (see LIVE.md)
source notebooks/.venv/bin/activate
export LIMBUS_BURST="/mnt/d/Conjunctiva Code/limbus/recordings/burst_2026-09-16_15-50-52"
cd ~/limbus-work                              # the notebooks write previews/, vessel_pictures/, preprocessed/, neural_cache/, flow_outputs/ here
marimo edit /path/to/limbus/notebooks         # the folder: a list of the notebooks, in the browser
marimo export html /path/to/limbus/notebooks/foundation.py -o foundation.html   # run it all, keep the outputs
```

`notebooks/pyproject.toml` sets marimo up for notebooks this heavy: opening one runs nothing, and a
changed cell marks the cells after it stale instead of running them.

### Memory and time

Measured on the lab PC with 6 threads, burst `2026-09-16_15-50-52` (140 frames of 1920 × 1200):

- `foundation.py` on the whole frame runs in about 2½ minutes.
- On a 720 × 1152 crop of the same burst (all 140 frames), `foundation.py` runs in 42 seconds at a peak
  of 2.7 GB and saves a 17 MB result.
- `flow.py` on the crop, preprocessing included, runs in about 1¾ minutes at a peak of 4.5 GB (the
  preprocessing about 30 s of it, without its figures). Its own memory is mostly two `float32` copies of the
  fixation's frames, 0.8 GB on the crop; on the whole frame they take about 3.4 GB on top of the preprocessing.
- The archive's sections 17–25 need much more: on the crop the archive, preprocessing included, runs
  in 2¼ minutes at a peak of 4.6 GB; on the whole frame it needs roughly 8–11 GB.

`notebooks/live.sh` gives the notebook server at most 12 GB (`LIMBUS_LIVE_MEMORY_MAX`) and 6 CPUs
(`LIMBUS_LIVE_THREADS`), shared by every notebook open on it, so that a large run cannot starve the
other work on the PC.
