# Reference averaged frames

Three averaged, stabilized frames of the bulbar conjunctiva, for developing an
analysis pipeline against real data without needing the camera or the raw
bursts. Recorded 16 Sep 2026; the imaged participant has consented to
publication.

| file | size | no-data | vessel wall rise (10-90 %) |
|---|---|---|---|
| `mean_2026-09-16_15-31-50.tif` | 1920 x 500 | 0.8 % | **12.00 px** — sharpest |
| `mean_2026-09-16_15-31-37.tif` | 1920 x 500 | ~0 | 12.75 px |
| `mean_2026-09-16_15-22-26.tif` | 1920 x 500 | 1.8 % | 13.88 px — most vessels |

The last column is how sharp the frames actually are, measured rather than
judged: the 10-90 % rise distance across the wall of a large vessel, median
over every vessel of radius >= 5 px in the frame. It is a like-for-like
ranking, not an absolute PSF - a vessel wall is a real edge of unknown
steepness, so the number includes the vessel as well as the optics. Two other
frames from the same day are deliberately not here: `15-50-52` (15.12 px) is
the softest, and `12-57-20` is a ruled calibration target, not an eye.

## What is in the file

- **float32 TIFF**, single plane, no compression. Values are camera counts
  (12-bit sensor, so roughly 0-4095); the frames here sit around 2100-2500.
- **`NaN` marks no-data** - pixels no frame covered after stabilization, at
  the edges of the covered region. Mask them before doing anything; they are
  not zeros.
- Each is the **mean of a burst** of ~74 fps frames after non-rigid
  stabilization. Averaging is what makes the vessels clean: a single frame is
  noisy and flickers.

## Getting to absorbance

Vessels absorb, so the physically meaningful quantity is `A = -log(T)`, where
`T` is the frame divided by its own illumination. In `A` a vessel is a
positive ridge whose height is its peak absorbance.

```python
import numpy as np, tifffile, cv2
mean = tifffile.imread("mean_2026-09-16_15-31-50.tif").astype(np.float32)
valid = cv2.erode(np.isfinite(mean).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
f = np.where(valid, mean, np.nanmedian(mean[valid])).astype(np.float32)
illum = cv2.GaussianBlur(cv2.morphologyEx(
    f, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (101, 101))), (0, 0), 34)
A = -np.log(np.clip(f / np.maximum(illum, 1e-6), 1e-3, None))
```

The closing has to be wider than any vessel (101 px here) or it will eat the
thick ones. `analysis/vessels/image.py` is the maintained version of this and
also removes the sensor's row and column offsets, which survive averaging and
look exactly like long, perfectly straight vessels to any line detector.

Typical numbers once you are in absorbance: background sits near 0.02-0.03,
a mid-calibre vessel around 0.10-0.20, a trunk 0.30-0.50.

## Scale

Not calibrated for these three. The two ruled-target bursts from the same day
give 27.46 px and 24.99 px per grid period and differ by 10 %, so the working
distance was not the same between sessions and neither can be applied to these
frames. Everything stays in pixels until a target is recorded in the same
session.

## Known caveats

- Stabilization is non-rigid and **experimental**. Residual parallel "doubles"
  are present in some raw frames and can survive averaging as a faint second
  line beside a vessel.
- The frames are 1920 x 500 crops of the sensor, not full frames.
- Vessel radii in these images run from about 1 px to 19 px, and the optics
  trade radius against depth for the thin ones: treat a small vessel's fitted
  radius as an upper bound and use `radius x peak absorbance` when comparing
  between vessels.
