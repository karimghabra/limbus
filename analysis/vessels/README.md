# Vessel identification

Finds the vessels in a burst's **averaged stabilized frame** and gives each one
an id that is stable across runs — and, through the stabilization fields,
across every raw frame of the burst.

Averaging over a burst is what makes this possible: single frames are noisy and
flicker, while the average shows the network cleanly. No velocity information is
used, so vessels are found whether or not blood was moving visibly in them.

## Running it

```bash
python -m vessels burst_2026-09-16_15-22-26            # from the analysis/ folder
python -m vessels <burst> --method translation --out /tmp/v
python -m vessels /path/to/mean_stabilized.tif         # any averaged frame
```

Outputs, in `<stabilization>/<method>/<burst>/vessels/`:

| file | what it holds |
|---|---|
| `vessel_labels.tif` | int32 image, each vessel's lumen carrying its id |
| `vessels.json` | per vessel: id, length, radius, peak absorbance, centreline points, and every setting used |
| `overlay.png` | the averaged frame with each vessel drawn and numbered |

Every stage can be switched off: `--no-join` (each ridge piece its own vessel),
`--no-fit` (centrelines only, no radii and no echo trimming), `--move-fit 6`
(let the fit move the centreline instead of keeping the ridge), and the
thresholds `--t-hi --t-lo --L-hi --L-lo --min-len --min-depth --psf`.

From Python:

```python
from vessels import NetConfig, detect, labels
from vessels import image as vimg
A, valid = vimg.prepare(mean)                 # absorbance of the averaged frame
vessels, model, z = detect(A, valid, NetConfig())
lab = labels(vessels, A.shape)
```

## How it works

1. **Absorbance** (`image.py`). The frame is divided by its own illumination
   (a morphological closing wider than any vessel) and turned into
   `A = -log(T)`: a vessel is then a positive ridge whose height is its peak
   absorbance, which is what the physical model fits.
   A sensor's row and column offsets survive averaging and look exactly like
   long, perfectly straight vessels - they were being detected as such, with
   zero deviation from a single image row. They are removed first, as the
   narrow part of each row's and column's median offset: an offset one or two
   rows wide goes completely, while a vessel running the full width along a
   row - the worst case - keeps over 80 % of its depth (`--no-fixed-pattern`
   to switch this off). A margin is also kept away from the edge of the
   covered region, where only a few frames contributed to the average.
2. **Evidence** (`evidence.py`). Hessian ridge strength of the blurred
   transmission at σ = 1–3 px, each scale put in units of **its own** robust
   spread before the scales are combined. Coarse scales respond strongly to the
   sclera's texture blotches; without the per-scale normalisation they drown the
   thin vessels the fine scales see.
3. **Ridges** (`ridges.py`). Non-maximum suppression across the ridge keeps
   vessels that run side by side separate instead of merging them into one
   blob. A **path opening** then keeps only pixels lying on a long, directed
   path — texture blotches are round and short, so *length*, not contrast,
   does most of the rejecting — and **hysteresis** keeps a faint stretch when
   it connects to a strong one, which is how a vessel survives the dip where
   another vessel crosses it.
4. **Joining** (`join.py`). Two pieces whose free ends point at each other
   across a short gap become one vessel when the cheapest path between them
   through the evidence stays on vessel-like pixels. This carries **identity
   only**: no centreline is invented across the gap, so nothing is reported as
   detected where no ridge was found.
5. **Fit** (`fit.py`, `model.py`). Each piece is fitted with the physical model
   — a spline with a radius at every control point, absorbance
   `D·sqrt(1-(d/r)²)` blurred by the optics, over a plane background — against
   the residual of the vessels already accepted. Parts of a candidate lying
   inside an accepted vessel's lumen are cut away first, which removes the echo
   ridges a fine-scale detector finds along the flanks of a wide vessel. By
   default the centreline stays where the ridge detector put it.
6. **Ids** (`network.py`). Vessels are numbered by position (row band, then x),
   so the same network always gives the same labels.

## What it measures, and how well

Injection–recovery on real averaged frames (small vessels of known radius and
peak absorbance planted in the absorbance of two crops, one sharp and one
defocused; "found" means half the planted centreline is within 2.5 px of a
detection):

| planted vessel | 1522L (sharp) | 1522R | 1550C (defocused) |
|---|---|---|---|
| r 1.0 px, 3 % deep | 95 % | 100 % | 100 % |
| r 1.5 px, 3 % deep | 95 % | 100 % | 98 % |
| r 2.5 px, 3 % deep | 85 % | 100 % | 100 % |
| any, ≥ 5 % deep | 98–100 % | 100 % | 100 % |

Detected centreline on the untouched crops is 18 800–28 000 px per megapixel.
Turning the hysteresis off (`--t-lo 3 --L-lo 40`) costs 7 points of thin-vessel
recall on the sharp crop and 30 on the wide faint ones, which is what the
faint-stretch rule buys. The stricter setting (`--t-hi 4 --t-lo 2`) halves the
false alarms and keeps thin-vessel recall, losing the wide faint vessels
(48 % at r 2.5).

False alarms are measured two ways, because both are biased and the truth lies
between them. On the **inverted frame** — pessimistic, since every
centre-surround filter echoes along the flanks of the now-bright vessels —
6 100 to 11 100 px/MP. On a **phase-randomised texture surrogate**, which has
the texture's power spectrum and no vessels at all, far less (measured
separately below). Neither control can be taken for precision on real data,
which is why detections were also reviewed by eye, tile by tile, on both crops.

Radius and depth: the optical blur trades them off for thin vessels. Against
known planted vessels the fitted radius comes out a few tenths of a pixel high
and the peak absorbance correspondingly low, while their **product** — the
absorbance integrated across the vessel, which is what blood volume per unit
length depends on — is recovered within about 10 %. Radii of vessels wider than
about 3 px are accurate to ~0.2 px. Treat a small vessel's radius as an upper
bound and use `radius × peak_absorbance` when a quantity has to be compared
between vessels.

A piece whose fitted radius sits on the model's 0.8 px floor carries
`"radius_at_limit": true` in the JSON. The optics cannot produce a vessel that
narrow, so such a piece is more likely a sensor line or a texture ridge — but
about one planted vessel in ten lands there too, so they are flagged rather
than dropped, and filtering them is left to whoever is counting.

## Per-vessel profiles, spectra and speed

Once vessels have identities, `vessels.profiles` samples every raw frame of the
burst along each vessel's own centreline:

```bash
python -m vessels.profiles burst_2026-09-16_15-22-26 --max-vessels 20
```

Stabilization maps `stabilized(x) = raw(x + d(x))`, so a centreline point `p`
sits at `p + d(p)` in raw frame `i`. Sampling the **raw** frames keeps every
measurement on the pixels the camera recorded, with one interpolation instead
of two. Frames the stabilization rejected are skipped, and the camera's own
timestamps give the time axis, so dropped frames leave a real gap instead of
shifting everything after them.

For each vessel this produces

- `kymo_<id>.png` and `kymographs.npz` — the **kymograph** `K[frame, position]`:
  absorbance along the vessel, averaged across the lumen, background beside the
  vessel subtracted. Moving blood draws diagonal streaks in it.
- an **amplitude spectrum**, computed by least squares at each frequency (a
  Lomb-Scargle periodogram, correct for uneven sampling) after the **common
  mode** — the median of the detrended series across vessels, which is the
  focus and illumination drift every vessel shares — has been regressed out of
  each vessel.
- a **speed** from the slope of the streaks, measured with the structure tensor
  over 0.3 s windows, with a coherence value; windows below the coherence
  threshold are reported as unknown rather than guessed. Against synthetic
  kymographs of known slope the speed is recovered within 10 % between 200 and
  600 px/s. Past roughly 8 px per frame, features move further between frames
  than they are long, the slope is under-read, and those windows are **flagged**
  (`speed_windows_beyond_frame_rate`) rather than silently reported — at 74 fps
  that limit is around 600 px/s.

Every vessel is measured twice: along its centreline, and along a **control
line pushed 12 px sideways into the background**, where nothing flows. If the
measurement were picking up residual motion or a sampling artefact, both would
read the same. On burst 15-22-26 the vessels read between -243 and +228 px/s
while their controls sit within a few px/s of zero:

| vessel | length | speed | control |
|---|---|---|---|
| 14 | 630 px | -243 px/s | -3.6 px/s |
| 56 | 543 px | +228 px/s | +4.9 px/s |
| 6 | 458 px | -219 px/s | -4.3 px/s |
| 3 | 469 px | +119 px/s | +1.7 px/s |
| 2 | 648 px | -2.7 px/s | -2.0 px/s |

Each vessel carries `speed_trusted`: the speed beat its control by more than a
factor of three, the frames matched (peak ≥ 0.3), and at least half the windows
returned a value. On this burst 9 of 20 vessels pass; the ones that fail
include a vessel whose background control moved faster than the vessel itself,
which is exactly the case the control exists to catch.

The signs differ between vessels, as they should when some carry blood towards
the limbus and others away. A vessel reading near zero with a low match peak
(vessel 2 here) means the flow left no trackable pattern, not that it stopped.

## Image scale

```bash
python -m vessels.scale burst_2026-09-16_15-05-41 --pitch-um <grid pitch>
```

`vessels.scale` measures the grid period of a ruled-target burst from the peak
of its power spectrum, refined to sub-pixel and reported with the grid's
rotation and the peak-to-background ratio, so a frame with no grid is obvious
rather than silently fitted. The two calibration bursts recorded on 16 Sep give
**27.46 px** (15-05-41) and **24.99 px** (12-57-20) per grid period, each stable
to better than 0.01 px across frames — they differ by 10 %, so the working
distance was not the same and each session needs its own calibration.

With the target's pitch that becomes µm/px, and speeds in px/s become mm/s.
Until the pitch is supplied everything downstream stays in pixels.

## Tests

```bash
python analysis/tests/test_vessels.py
```

Everything is rendered from the physical model, so no data files are needed:
path-opening lengths in every direction, flat-fielding recovering a planted
depth, two vessels 8 px apart staying separate, a broken vessel becoming one
identity, radius and depth against the truth, determinism of the ids, and a
vessel-free image giving no vessels.
