# Lucky fusion and vessel annotation: methods

How and why each stage of `lucky` works. Section numbers (§L) are cited by
the code and by `config.py`. It builds on the stabilization stage; its own
methods are in `../stabilize/METHODS.md`, cited here as "stabilize §n".
§L5 records how the stage was validated, including the approaches that
failed, because those are what justify the final design.

---

## §L1 Inputs: frames on the stabilized grid

Nothing is re-registered. Each raw frame is warped once with the transform
the stabilization saved for it: the non-rigid field (`fields.npz`) when that
result exists and succeeded, otherwise the translation (`transforms.csv`).
Only frames the chosen result actually used are fused. The fused images
therefore land on the same pixel grid as that result's `mean_stabilized.tif`
and `consensus_mask.tif`, so annotations from either overlay directly.

Each frame carries an **observed** map: inside the field of view after
warping, and not hidden by glare (stabilize §4). Unobserved pixels are
filled by normalised convolution before any filtering, so that band-pass
filters don't ring at hole edges, and they get zero weight.

Brightness changes from frame to frame (illumination fixed to the camera
while the tissue moves, auto-exposure), so every frame is scaled to a
common median level before it is fused.

## §L2 Lucky fusion

**Idea.** Every frame is blurred by the optics, by motion during the
exposure, and, where the eye's curved surface leaves the shallow depth of
field, by defocus. The last two vary from frame to frame and from place to
place. Where a frame was sharp, its fine detail should count for more.
Averaging everything (the stabilized mean) mixes sharp and soft frames.
Taking only the sharpest frame throws away the noise averaging.

**Bands.** Each frame is split into spatial-frequency bands by differences
of Gaussians at σ = 0, 1, 2, 4, 8, 16 px. The bands plus the σ = 16 lowpass
add back up to the frame exactly. Blur removes fine bands first and hardly
touches coarse ones. Each band is therefore fused separately, and the
lowpass is always a plain mean.

**Weights.** Per band k, frame i and place x:

    fused_k(x) = Σ_i w_ik(x) band_ik(x) / Σ_i w_ik(x),    w_ik = a_ik^p

    a_ik(x) = max(0, G_r * (band_ik · band_ref,k)) / G_r * (band_ref,k²)  +  floor

`a_ik` is the frame's **relative amplitude**. It is the frame's band
projected onto the same band of a reference (the plain mean, from a first
pass), over the reference's own energy, both averaged over a Gaussian window
of radius r_k = max(12, 4 σ_k+1) px. A frame that carries the reference's
detail at full strength scores ≈ 1, and a blurred or locally misregistered
one scores lower. The floor (0.25) keeps weights from being driven by noise
where the reference has no detail at all.

**Why a projection and not the frame's own energy.** The classic lucky-
imaging and Fourier-burst-accumulation weight is the frame's own band
energy. Noise adds energy, so that weight rewards the *noisiest* frames.
This was measured on the reference bursts (§L5): gain and exposure are
constant within each burst, but local brightness is not, and the shot noise
at a place rises as the tissue there moves into dimmer illumination. In one
region the 1–2 px band energy drifted by ±25% over the burst in step with
brightness, not with sharpness. Noise in the frame is independent of the
reference, so it averages out of the projection. On a constant-focus burst
the projection weights stayed flat (~80 of 80 effective frames), while
energy weights fell to 62.

**Why p = 1.** If band_i = H_i·X + noise, with H_i the frame's amplitude
transfer and equal noise in every frame, the linear combination with the
best signal-to-noise ratio weights each frame by H_i. That is the matched
filter, and p = 1 here. Larger p gets closer to picking the sharpest frame.
It sharpens further but averages fewer frames. On the benchmark p = 2 and 4
raised recall but lost more in precision to noise and resolved texture, and
p = 1 was the only power never worse than the mean (§L5). `powers` builds
p = 0, 1, 2 side by side; `output` (1) is the image kept as `lucky.tif`.

**Effective frames.** Per band, n_eff = (Σw)²/Σw² counts how many equally
weighted frames would average noise as well. It is recorded per image: a
burst whose n_eff stays near the frame count had little sharpness diversity
to exploit.

## §L3 Annotation

    image → contrast → vesselness with orientation → ridges (non-maximum suppression)
          → hysteresis along ridges → thinning → wide vessels re-traced by line points
          → spur pruning → half-maximum mask and widths

**Contrast.** A grey-level closing (40 px radius, on a 4× reduced image)
removes dark structures narrower than about 80 px. After a blur it is the
local background, and contrast = image / background − 1: 0 on bare tissue,
negative on vessels. The closing, unlike a plain blur, doesn't let vessels
pull their own background down.

**Vesselness.** Frangi's measure for dark tubes (stabilize §5), computed on
the contrast image at σ = 1.5, 3, 6, 12 and 24 px, maximum over scales.
Two changes from stabilize §5. The Hessian is normalised by σ^1.5
(Lindeberg's ridge normalisation) rather than σ², because with σ² the wide
scales won on the halo beside thin vessels. The contrast constant c is half
the **95th** percentile of the Hessian norm (not the 99th), so capillaries
are not measured only against the largest vessels. Each pixel also keeps
its winning scale and the across-vessel direction (the eigenvector of the
larger eigenvalue).

**Centrelines are ridges, not a skeleton of a mask.** The first version
thresholded vesselness into a mask and skeletonized it. A multi-scale mask
is fattened by its large scales: it merged vessels running close together,
and the skeleton then cut through the merged blob. Centrelines are now
found the way Canny finds edges. A pixel is kept only if its vesselness is
at least that of both neighbours one pixel away across the vessel. Ridge
pixels are then linked by hysteresis (low 0.06, high 0.25), so a faint
stretch survives when it connects to a strong one. On the ground truth
(before wide vessels were added to it) this change alone took centreline
F1 of the plain mean from 0.51 to 0.66, and threshold tuning took it to
0.82.

**Wide vessels.** On the reference bursts the ridges of the widest vessels
(30–60 px, the dark flat-bottomed ones) ran along their *walls* and not down
their middle. Across a flat bottom, the small scales respond most strongly
at the two inner corners, and the maximum over scales peaks there. The
benchmark had not caught this: its widest vessels were 16 px, so wide ones
were added. The fix uses Steger's line points (IEEE PAMI 1998): pixels where
the first derivative across the vessel crosses zero inside the pixel, which
is the centre of a dark profile at any width and never a corner, where the
wall's slope is nonzero. Line points alone were worse overall (F1 0.77 vs
0.81, lower precision on small vessels), so they are used only where they
are needed. Wide vessels are found geometrically: the part of the "dark core"
that survives an opening by a disk of radius 10 px, i.e. at least 20 px wide.
The dark core is the contrast (smoothed at σ 2) below 0.6 × the burst's
0.5th-percentile contrast. Inside those bodies, ridge pixels are replaced by
line points at σ ≥ 12, and by the body's medial axis where those are
missing. On the real bursts this raised the share of the consensus
skeleton (mostly wide vessels) recovered by the annotation from 0.76 to
0.80 (`15-50-52`) and from 0.77 to 0.78 (`15-45-12`); `15-31-37` changed
by −0.007. On the benchmark it cost 0.004 F1. A looser body (radius 8,
fraction 0.5) recovered more on real data (0.82) but, at dense crossings,
swallowed small vessels and cost 0.04 F1 on the benchmark.

**Cleanup.** Holes under 40 px in the traced lines are filled (they would
thin into small loops), and the lines are thinned to 1 px. Terminal branches shorter than
12 px (hairs on a bumpy ridge) are pruned, and pieces shorter than 25 px are
dropped.

**Mask and width.** A pixel is vessel if it is darker than half the contrast
depth of its nearest centreline point, within 3× that point's scale: the
full width at half maximum, the usual diameter of an intensity profile.
`vessel_width.tif` holds that width at each centreline pixel.

**Segments.** Removing junction pixels splits the centreline into
segments. `segments.csv` gives each one's length along the centreline
(diagonal steps √2), the chord between its ends, tortuosity (length/chord),
and the median and 10th–90th percentile half-maximum width.

**Same constants for every image of a burst.** c is computed once, from the
plain mean, and every image of the burst is annotated with it and with the
same thresholds. The thresholds were tuned for the best F1 of the **mean**,
not of the lucky image. Any difference between the two annotations
therefore comes from the images.

## §L4 Scoring

**Centreline agreement.** A detected centreline pixel counts as correct if
a reference centreline lies within a tolerance (2 px against ground truth).
A reference pixel counts as recovered if a detected one does. Precision,
recall and F1 follow. Scoring skips a border of 2× the largest filter scale,
where filters run off the field.

**Real data has no ground truth**, so each result records:

- *split-half F1*: alternate frames are fused into two independent images,
  each annotated, and their centrelines are compared. This measures
  reproducibility: how much of the annotation is the tissue rather than the
  noise and red-cell flicker of particular frames.
- *centreline density* and *pieces per Mpx*: how much is found, and how
  fragmented it is.
- *edge sharpness*: median steepest contrast gradient across vessel edges,
  per unit of that vessel's depth (1/edge width).
- *background noise*: robust SD of the finest detail on bare tissue.
- for the output image, how much of the **consensus-mask skeleton**, the
  annotation the stabilization stage already gave, the new annotation also
  contains (3 px tolerance).

## §L5 Validation

**Ground-truth benchmark** (`tests/test_lucky_synthetic.py`). Synthetic
bursts, 1000×600, 80 frames, at the density of the reference bursts: 6
venules 8–16 px wide, 40 small vessels 3–6 px and 160 capillaries 1–2 px,
drawn along known polylines that are the true centrelines, plus 2 wide
venules 24–40 px wide. The bursts add
static tissue texture, smooth illumination, eye motion, red-cell flicker
and shot noise of ~1% per frame (the reference bursts measure 0.7–1.4%).
Defocus sweeps across the field over time (σ 0.6–4.5 px along a tilted,
moving focal plane) in two bursts, and is constant (σ 1.6 px) in a control.
Each burst is stabilized by the real translation pipeline, then fused and
annotated. Centreline F1 against truth, 2 px tolerance:

| | sweep a | sweep b | constant focus |
|---|---|---|---|
| consensus-mask skeleton (what stabilization gives) | 0.217 | 0.204 | 0.292 |
| annotation of the plain mean (p = 0) | 0.818 | 0.766 | 0.837 |
| annotation of the lucky image (p = 1) | 0.842 | 0.791 | 0.837 |

Passing criteria: the annotator beats the consensus skeleton by > 0.3 F1 on
every burst; lucky fusion gains ≥ 0.015 F1 where focus varies and loses no
more than 0.01 where it doesn't.

**What didn't work**, each measured on the benchmark and abandoned (items
1–7 on the benchmark before wide vessels were added, with the ridge
annotator):

1. *Thresholding vesselness, then skeletonizing*: F1 0.51 on the mean.
   The mask merges adjacent vessels (§L3), and ridge detection replaced it.
2. *Energy weights* (the frame's own band energy, p = 2–8): with thresholds
   set for the mean, recall rose from 0.74 to 0.86 but precision fell from
   0.93 to 0.62. On real bursts the weights tracked noise, not sharpness
   (§L2).
3. *Noise-calibrated significance*: vesselness as curvature in units of the
   noise measured from the split-half difference, to hold every image to
   one false-alarm rate. Noise shrinks at large scales, so every place's
   maximum moved to the largest scale and ridges ran along the halos. F1
   dropped to 0.55–0.63.
4. *Band restoration*: dividing each fused band by the frames' mean relative
   amplitude, returning it to the sharpest frame's level (a multi-frame
   deconvolution). The signal-to-noise ratio is unchanged, but the boosted
   texture and noise cost more precision than the recall gained: average F1
   0.79 against 0.85.
5. *Wiener deconvolution* of the fused image (Gaussian PSF σ 1–2 px): at
   best no better than without (0.850 vs 0.848 at σ = 1), and worse at the
   larger PSFs.
6. *Flicker enhancement*: subtracting the noise-corrected temporal SD from
   the mean, to darken vessels that red cells were passing through. Static
   texture was unaffected, but precision fell and F1 dropped from 0.83 to
   0.79–0.81.
7. *Isolated-piece filtering* by mean vesselness: no gain on the mean.
8. *Other scale normalisations* (σ^1.75, σ^2, an extra σ = 36 scale): no
   better on wide vessels and worse overall (F1 0.71–0.77 vs 0.81); the
   wide-vessel failure is fixed by line points instead (§L3).

A pattern runs through 2, 4, 5 and 6: anything that sharpens the image also
sharpens the tissue texture, and the annotator starts tracing it. With
thresholds re-tuned per image, p = 2 and p = 4 reached F1 0.865 and 0.870 on
the sweeps. They lost 0.004–0.007 on constant focus, and per-image tuning
isn't available on real data. p = 1 at the mean's thresholds is the choice
that never lost.

**Reference bursts** (lucky 0.1.0, alignment from the non-rigid result,
translation for `15-40-55`). Centreline density is in px per Mpx of valid
field. Split-half F1 compares the annotations of two independent halves of
the frames. "Recovered" is the share of the consensus-mask skeleton that the
lucky annotation also contains (3 px):

| Burst | frames | centreline: consensus skeleton | mean | lucky | split-half F1: mean | lucky | edge sharpness: mean | lucky | recovered |
|---|---|---|---|---|---|---|---|---|---|
| `15-50-52` | 138 | 4 328 | 24 959 | 25 215 | 0.981 | 0.982 | 0.110 | 0.112 | 0.80 |
| `15-45-12` | 75 | 4 791 | 20 249 | 20 531 | 0.979 | 0.979 | 0.085 | 0.086 | 0.78 |
| `15-31-37` | 158 | 4 762 | 13 886 | 13 973 | 0.991 | 0.990 | 0.091 | 0.092 | 0.68 |
| `12-57-20` (strip) | 296 | 14 611 | 82 982 | 83 269 | 0.979 | 0.978 | 0.231 | 0.233 | 0.94 |
| `15-40-55` (hard) | 14 | 4 561 | 31 333 | 34 570 | 0.517 | 0.572 | 0.133 | 0.141 | 0.43 |

What they show:

- **The annotator is the large change.** The consensus-mask skeleton
  traces the few widest vessels. The new annotation traces 3–6× more
  centreline, and two independent halves of each good burst reproduce it
  with split-half F1 ≈ 0.98. In the overlays inspected, the consensus
  skeleton it doesn't recover is mostly skeleton spurs, and centrelines of
  out-of-focus deep vessels that the consensus mask's coarse
  90th-percentile envelope includes.
- **Lucky fusion gains little on the good bursts, because they give it
  little to work with.** The relative amplitude of the 1–2 px and 2–4 px
  bands per 240×300 px tile varies by only about ±15–20% across frames
  (10th–90th percentile 0.8–1.2). The fluctuations are only weakly shared
  across the field (tile-to-tile correlation 0.13–0.48). This is far less
  than the benchmark's focus sweep. The lucky image is close to the mean
  there: centreline +1%, edge sharpness +1–2%, reproducibility unchanged.
  Alignment is not what limits these bursts: 64 px patches of single frames
  sit a median 0.12 px (90th percentile 0.25 px) from the mean.
- **Where frames do differ, it helps.** `15-40-55` registers only 14
  frames. Its halves disagree (split-half F1 0.52), so its annotation is
  not to be trusted yet. But lucky fusion raised that reproducibility to
  0.57 (p = 2: 0.59), with sharper edges and 10% more centreline.

## §L6 Outputs

Per burst, in `<stabilization>/lucky/<burst>/`:

| File | Contents |
|---|---|
| `lucky.tif` | the fused image at p = `output` (float32, NaN where < 50% coverage) |
| `mean.tif` | the plain mean, fused identically (the p = 0 case) |
| `centreline.tif` | vessel centrelines of `lucky.tif` (0/255, 1 px wide) |
| `vessel_mask.tif` | half-maximum vessel mask (0/255) |
| `vessel_width.tif` | half-maximum width in px at each centreline pixel, 0 elsewhere |
| `vesselness.tif` | the vesselness the centrelines were traced on |
| `segments.csv` | one row per segment: length, chord, tortuosity, widths, end points |
| `compare.png` | top: mean, lucky image; bottom: consensus-mask skeleton (red) on the mean, lucky annotation (green) on the lucky image; magenta = no data |
| `lucky.json` | every metric of §L4 for p = 0, 1, 2 and the consensus skeleton, every parameter, software versions |

`summary.csv` has one row per burst. A result counts as current, and is
skipped on re-runs, only when its `lucky_version` and `params_hash` match
the code being run.
