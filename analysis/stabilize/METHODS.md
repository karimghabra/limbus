# Stabilization methods

How and why each stage of `stabilize` works. Section numbers (§) are cited by
the code and by `config.py`, so every parameter traces back to its reasoning.
Written for a scientific reader; no prior image-processing background assumed.
§12 records how the method was validated — including the approaches that
failed, because those are what justify the final design.

---

## §1 Image formation and red-cell flicker

**Why vessels are visible.** Haemoglobin absorbs strongly, with a sharp peak
near 415 nm and smaller peaks around 540–577 nm. Light crossing a vessel full
of red cells is attenuated before scattering back from the white sclera, so
vessels appear as **dark lines on a bright background**. (Measured on these
data: dark-ridge responses roughly twice bright-ridge ones.) This is also why
green illumination usually gives the best vessel contrast.

**Why vessel interiors flicker.** In capillaries and small venules, red cells
travel in trains separated by plasma gaps. A cell absorbs; a gap does not. The
*interior* of a vessel therefore brightens and darkens as cells pass, even when
the vessel is perfectly still. The vessel's **walls** are fixed tissue
geometry; its **contents** move independently. This motivates registering on
the vessel envelope rather than raw intensity.

## §2 Working scale

Frames are processed at half resolution (`working_scale = 0.5`). The vessels
here span many pixels, so no meaningful structure is lost, and computation
drops 4×. Downscaling uses area averaging (`INTER_AREA`), which low-pass
filters *before* discarding pixels; naive subsampling would alias fine detail
into false patterns (the Nyquist limit). Short strips (< 400 rows) are
processed at full resolution, or the larger filter scales would not fit.

All displacements are reported in **full-resolution pixels**.

## §3 Frame quality gate

Blinks, blur and lighting jumps break every registration method, so they are
found first. Frames are **flagged, never deleted**.

**Sharpness from the Laplacian.** The Laplacian ∇²I = ∂²I/∂x² + ∂²I/∂y²
responds to fine detail; its variance measures high-frequency energy. Blur is
a low-pass filter — in Fourier terms it multiplies the spectrum by the optical
transfer function — so blurred frames score low, and a blink (a featureless
eyelid) scores very low. Sharpness varies multiplicatively and is skewed, so
its logarithm is used.

**Robust z-scores.** An ordinary z-score uses mean and standard deviation,
which the very blinks being hunted would inflate, hiding themselves. Instead:

    z = (x − median(x)) / (1.4826 · MAD)

The median and median absolute deviation (MAD) tolerate up to 50% outliers.
The factor 1.4826 makes the MAD equal the standard deviation for normally
distributed data, so z keeps its usual meaning.

**Statistical *and* physical.** A frame is rejected only if it is both
statistically unusual *and* meaningfully different:

- blink/blur: log-sharpness z < −3.5 **and** sharpness < 80% of the median;
- lighting: brightness |z| > 3.5 **and** > 10% from the median.

Without the physical floor, a very steady burst — whose MAD is tiny — rejects
frames over trivial wobbles (the synthetic motionless burst lost a good frame
that way). Real blinks clear both easily: sharpness 6 against a median of 241.

## §4 Glare and the observed mask

Specular reflection off the tear film obeys angle of incidence = angle of
reflection, so a highlight's position is fixed by the **light source and camera
geometry, not the tissue**. As the eye moves, glare stays put — left in, it
would be a high-contrast feature voting for "no motion". It also saturates the
sensor, and a clipped pixel carries no information. Pixels ≥ 98% of full scale
are excluded, dilated by 30 px with a round kernel (bloom spreads radially).

**Glare is unknown, not "not vessel".** Each frame carries an *observed* mask.
Because glare is fixed to the camera, once frames are aligned to the tissue its
hole falls on **different tissue in every frame**. Treating those hidden pixels
as "not vessel" would score them as disagreement between frames, and would
drag templates toward zero there. Instead they count as unobserved: they are
left out of templates, of the metrics' coverage, and of the stabilized mean
(where alignment would otherwise smear saturated values across the tissue).

## §5 Vesselness: the Hessian

Expand intensity around a point to second order:

    I(x + δ) ≈ I(x) + ∇I·δ + ½ δᵀ H δ

The **Hessian H** (matrix of second derivatives) describes local curvature. Its
eigenvectors point along the directions of greatest and least curvature; its
eigenvalues λ₁, λ₂ (ordered |λ₁| ≤ |λ₂|) give the curvature along each.

| Structure | λ₁ (along) | λ₂ (across) |
|---|---|---|
| Flat background / noise | ≈ 0 | ≈ 0 |
| **Tube (vessel)** | **≈ 0** | **large** |
| Blob | large | large |

A **dark** vessel is an intensity valley, so λ₂ > 0 across it; pixels with
λ₂ ≤ 0 are set to zero, enforcing polarity. Frangi's vesselness combines

    R_b = λ₁/λ₂        (line vs blob)
    S   = √(λ₁² + λ₂²) (structure vs flat noise)
    V   = exp(−R_b² / 2β²) · (1 − exp(−S² / 2c²))

The first factor is near 1 only for elongated structures; the second only
where curvature rises above noise. β = 0.5.

**Scale.** Derivatives are taken of a Gaussian-blurred image, so the blur σ
acts as a ruler: a vessel responds most strongly when σ is comparable to its
width. Scales σ = 4, 8, 16, 32 px (full resolution) are combined by taking the
maximum — the real vessels here were still strengthening at 16 px. Each
response is multiplied by σ² (**scale normalisation**) so scales compete
fairly; blurring otherwise shrinks derivatives. Because the normalised response
is scale-invariant, the large scales are computed on a 2×-downsampled image
and upsampled, giving the same answer much faster.

Second derivatives ignore constant brightness and linear gradients, so uneven
illumination and vignetting do not create false vessels.

**One contrast constant per burst.** c defines what counts as "strong"
curvature. It is set **once per burst** (from the 99th percentile of S in a
reference frame), never per frame: per-frame normalisation rescales each map
differently, so a vessel's value changes between frames for no physical
reason. On these data that broke matching between distant frames (response
0.04 vs 0.37 with a burst-wide c).

## §6 Vessel envelope mask

The soft map V is thresholded at the burst-wide 90th percentile (estimated
from 24 evenly spaced good frames), then a 3×3 morphological opening removes
isolated noise pixels. The result M is the **vessel envelope**.

**Why a binary mask survives illumination change.** Thresholding discards
**amplitude** and keeps only **geometry** — vessel or not. Lighting changes
alter amplitude, not where the vessels are. On a burst whose brightness varied
25%, the mask matched distant frames far better than raw intensity (phase-
correlation response 0.195 vs 0.035).

**The cost is quantisation.** A mask edge is a hard 0/1 step that moves only in
whole pixels, blunting sub-pixel precision. Hence registration is coarse-to-
fine (§8): the mask gives robust capture range, the soft map gives precision.

## §7 Phase correlation

**The Fourier shift theorem.** An image is a sum of sinusoids of every spatial
frequency. Shifting it rigidly by **d** leaves every frequency's amplitude
unchanged and rotates each one's phase by an amount proportional to that
frequency:

    f(x − d)  ⟷  F(k) · exp(−i2π k·d)

So a single displacement dictates *all* the phase changes at once: plotted
across frequency space, the phase differences form a tilted plane, and **the
tilt of that plane is the displacement**. Normalising the cross-power spectrum
isolates it:

    R(k) = F·G* / |F·G*| = exp(i2π k·d)

The inverse Fourier transform of that pure phase ramp is a single sharp spike
at **d** — all frequencies reinforce at the true displacement and cancel
elsewhere.

**Why many frequencies.** One sinusoid of wavelength 20 px cannot distinguish a
5 px shift from 25 px — the phase wraps. Adding more frequencies leaves only
one displacement consistent with all of them, like a vernier scale.

**Whitening.** Natural images concentrate energy in low frequencies — here, the
smooth illumination. Without normalising, lighting would dominate and
registration would track the light, not the tissue. Dividing by |FG*| gives
every frequency an equal vote, and a global gain change (auto-exposure) cancels.

**Peak height is confidence.** A pure rigid shift puts all the energy in one
spike (response near 1). Anything that is not a rigid shift — flicker, noise,
blur, content leaving the field — scatters the phases and shrinks the spike.

**Windowing.** The Fourier transform treats an image as periodic, so its
mismatched borders would correlate at zero shift regardless of true motion. A
Hanning window tapers each image to zero at its edges.

**Sub-pixel precision without interpolation.** The coarse stage uses OpenCV's
`phaseCorrelate`, whose sub-pixel centroid is slightly biased toward whole
pixels (~0.1 px measured). The fine stage instead uses **upsampled-DFT
refinement** (Guizar-Sicairos, Thurman & Fienup, *Opt. Lett.* 2008): a sub-pixel
shift is exact in the Fourier domain — just a phase ramp — so the correlation
peak is evaluated on a grid 20× finer than the pixels, directly from the
spectra, by matrix multiplication. **No frame is warped to measure it.**
Warping first (bilinear pre-shift) blurs by an amount that depends on each
frame's fractional shift, adding noise that grows with motion.

**Partial whitening (α = 0.5).** The fine stage divides by |FG*|^0.5 rather
than |FG*|. Full whitening gives the noisiest high frequencies — red cells,
sensor noise — the same vote as stable vessel geometry; half-whitening still
suppresses illumination while letting geometry dominate. Confidence is still
read from the fully whitened surface, so it stays comparable across frames.

Estimator bake-off on synthetic frames against the true template:

| Estimator | Still, RMS | 45 px motion, RMS |
|---|---|---|
| OpenCV + bilinear pre-shift | 0.151 px | 0.521 px |
| Upsampled DFT, full whitening | 0.039 px | 0.124 px |
| **Upsampled DFT, α = 0.5** | 0.055 px | **0.082 px** |
| Upsampled DFT + pre-shift, 2 steps | 0.045 px | 0.181 px |

**Convention.** `phaseCorrelate(reference, moved)` returns +d for content moved
by +d (verified empirically). Stabilizing a frame means warping it by −d.

## §8 Groupwise registration

**Why not chain consecutive shifts.** Adding up frame-to-frame shifts
accumulates error as a random walk: with independent per-step error σ,

    σ_total ≈ σ · √N

— about 9.5 px after 1000 frames at 0.3 px per step. Worse, one bad step during
a saccade or blink offsets every later frame. Chaining is locally precise,
though, so it **seeds** the trajectory.

**Template registration.** Every frame is measured directly against a common
template, so errors do not accumulate. The template depends on the alignment,
so the two are refined alternately — the structure of expectation–maximisation
and Procrustes analysis:

1. build templates from the current alignment, using only confidently
   registered frames: the soft mask consensus p (fraction of observing frames
   calling each pixel vessel) and the mean soft map, both coverage-normalised;
2. for each good frame, generate **candidates** and let the fine stage decide
   (below);
3. re-centre the trajectory on its median so the template does not drift;
4. repeat until the RMS change < 0.1 px, at most 8 iterations.

**Candidates, not a hand-off.** For each frame:

- the **current estimate** (initially from chaining) is always a candidate;
- the **coarse estimate** — phase correlation of the frame's binary mask
  against p, with full capture range — is added when it lands more than 6 px
  away;
- each candidate is refined by the fine stage (§7), searching its integer peak
  within 6 px, and **the candidate with the higher fine confidence wins**;
- a frame whose best confidence is < 0.10 is marked unregistered, keeps its
  previous estimate rather than adopting a guess, and is left out of the next
  template.

This design came from a failure (§12): on real data, the coarse response alone
could not flag wrong peaks — outliers 200 px off scored like good frames — so
letting the coarse stage hand its estimate straight to refinement let a few
wrong frames corrupt the template, which then misled more frames, and the loop
diverged. Fine confidence separates right from wrong cleanly (~0.4 vs ~0.001).
Now a coarse outlier cannot override a good prior, while a chain broken by a
saccade or blink can still be rescued by the global coarse search.

The model is **translation only**; §10 checks whether that suffices.

## §9 Stability metrics

Two families, kept separate so eye motion and correction quality are never
conflated.

### Motion — how unstable the burst was

From the recovered trajectory of registered frames, in full-resolution pixels:

- **Displacement** from the median position: median, 95th percentile, max.
- **Jitter** as a *speed*, px/s. Bursts here run from 32 to 149 fps, and the
  step between frames scales with the time between them, so only a rate is
  comparable across bursts. Times come from the camera's own per-frame clock
  (`frames.csv`) when available, so dropped frames leave real gaps.
- **Saccades**: frame-to-frame speeds above median + 6·MAD. Jitter RMS is
  reported excluding them, so one saccade does not dominate it.
- **Drift**: slope of a straight-line fit of position against time.

### Quality — how well stabilization worked

Judged from the vessel masks themselves, **never from the registration's own
estimates**, so the method does not grade its own homework. Computed before
(masks as captured) and after (masks aligned), over the same registered frames.

**Vessel overlap** — the probability that a pixel labelled vessel in one frame
is also vessel in another frame. Per pixel, with k frames calling it vessel out
of the n frames that *observed* it:

    overlap = Σ k(k − 1) / Σ k(n − 1)

an exact pairwise probability excluding self-pairs. A still vessel concentrates
k at n and scores near 1; motion smears the same vessel thinly over many pixels,
spreading k and driving overlap down.

It replaced an earlier *persistence* = mean |2p − 1| over pixels with p > 0.05,
which the synthetic test showed to be wrong: pixels a smeared vessel covered
only occasionally (p ≈ 0.1) scored |2p − 1| = 0.8 as strong *agreement*, so large
motion inflated the metric instead of lowering it.

**Dice** — overlap of each frame's mask with the consensus mask (p ≥ 0.5),
compared only where that frame observed:

    Dice(A, B) = 2|A ∩ B| / (|A| + |B|)

Two details keep both metrics honest:

- **Re-binarise after warping.** A sub-pixel shift interpolates mask edges to
  fractions like 0.3 and 0.7; summing those would blur the masks and penalise
  any burst that moved, even when its motion was removed perfectly. Each warped
  mask is re-thresholded at 0.5, keeping the sub-pixel edge position.
- **Coverage.** Only pixels observed by ≥ 50% of frames count, so borders
  shifted in from outside the field — and glare holes (§4) — cannot distort the
  score.

**Residual step** — phase correlation between consecutive *stabilized* frames,
which should now differ by ~0 px.

**Closure error** — shifts must be transitive, so

    ε = |d(i→i+1) + d(i+1→i+2) − d(i→i+2)|

is pure estimator error, measurable without ground truth (like closing a
surveying loop). It is a **lower bound**: correlated errors can cancel. That is
why `tests/test_synthetic.py` also checks against known shifts.

## §10 Stability index and limitations

**Stability index = vessel overlap after stabilization (0–1)**, always read
beside the usable-frame fraction.

- **Not an absolute.** Plasma gaps make real vessel pixels read "not vessel" in
  some frames, so overlap stays below 1 even under perfect alignment; blur and
  noise lower the achievable ceiling further. Compare bursts processed with the
  same parameters. Grade bands should be set from the observed distribution,
  not assumed.
- **Translation only.** The rotation diagnostic re-registers the four quadrants
  of each stabilized frame against the template's quadrants. Pure translation
  leaves every quadrant near 0 px; rotation moves opposite quadrants in opposite
  directions. On the first real bursts it read 0.8–1.4 px against residual steps
  of 0.2–0.4 px: a small but real non-rigid component, plausible because the
  globe is curved, so its motion under the camera is not a pure translation.
  If this matters for an analysis, extend to rigid registration — the standard
  route is the **Fourier–Mellin transform**, which resamples the spectrum
  magnitude into log-polar coordinates so rotation and scale become shifts
  measurable by the same phase correlation — or to piecewise (tiled)
  registration for curvature.
- **Strips.** A 100-row strip gives little vertical capture range; phase
  correlation cannot resolve vertical motion beyond about half the height.
- **Unregistrable stretches.** Frames showing tissue outside the template (large
  gaze shifts) or with strong lighting change cannot be matched, and are left
  out rather than guessed; the usable fraction reports how many.

## §11 Outputs and reproducibility

Per burst and method, in `<out>/<method>/<burst>/` — raw bursts are never
written to. `<out>` defaults to a `stabilization` folder beside the recordings
folder, which is also where the camera app's Review tab looks. Every method
writes the same files, so anything that reads one reads the other:

| File | Contents |
|---|---|
| `transforms.csv` | per frame: time, dx/dy (full-res px), coarse/fine responses, gate result and reason, registered flag, sharpness/brightness z, clipped fraction |
| `metrics.json` | stability index, usable fraction, motion and quality metrics, diagnostics, and **every parameter plus software versions** |
| `mean_raw.tif` | mean of the registered frames, uncorrected (float32) |
| `mean_stabilized.tif`, `std_stabilized.tif` | mean and temporal SD after stabilization, glare excluded (float32; NaN where coverage < 50%) |
| `consensus_mask.tif` | vessel envelope consensus at full resolution (0/255) |
| `qc.png` | raw vs stabilized mean on a shared contrast scale, trajectory, headline metrics |
| `fields.npz` | non-rigid only: each frame's displacement field in compact form (§13) |

`metrics.json` names the `method`, whether it is `experimental`, the
`stabilize_version` and a `params_hash` — a fingerprint of every parameter.
A result counts as current, and is skipped on re-runs, only when both match
the code being run: a changed setting must invalidate old results, which file
times alone can't detect. For the non-rigid method, `registered` in
`transforms.csv` means *used by the non-rigid result*; the translation stage's
flag is kept as `translation_registered`.

In `qc.png`, **magenta marks regions with no valid data** — never black, which
would be indistinguishable from a vessel — and trajectory lines **break** where
frames were not used, rather than drawing motion that was never measured.

Stabilized projections use a **single resampling pass** from each original
frame; warping repeatedly would blur fine detail. Stabilized frame stacks are
not written — they can be regenerated at any time from `transforms.csv` or
`fields.npz` (the Review tab does exactly that for playback).
Per method, `summary.csv` and `summary.html` rank every burst with a result
on disk by stability index.

## §12 Validation

**Synthetic ground truth** (`tests/test_synthetic.py`). Vessel scenes with known
sub-pixel motion, red-cell flicker, shot noise, blinks and camera-fixed glare.
Passing criteria and results (stabilize 0.2.0):

| Case | Motion | RMS error (< 0.2 px) | Overlap after | Perfect-registration ceiling |
|---|---|---|---|---|
| still | 0 px | 0.000 | 0.792 | 0.792 |
| gentle | 8 × 7 px | 0.071 | 0.767 | 0.768 |
| rough | 35 × 45 px + saccade | 0.064 | 0.738 | 0.739 |
| rough + blinks + glare | same | 0.116 | 0.755 | 0.756 |

The **ceiling** is the overlap of the same masks aligned with the *true*
motion. It is the right yardstick; a motionless burst's score is not, because
the generator creates motion with bilinear warps, making moving frames ~12%
blurrier than still ones — no registration could reach the still burst's score.

**What validation caught**, each fixed and re-verified:

1. the first fine stage (bilinear pre-shift) erred by 0.4–0.5 px → upsampled DFT (§7);
2. *persistence* rewarded smeared vessels → vessel overlap (§9);
3. interpolated masks deflated overlap for moving bursts → re-binarisation (§9);
4. glare holes were scored as disagreement → observed masks (§4);
5. the gate rejected good frames in steady bursts → physical floors (§3);
6. on **real** data, the template loop diverged (RMS change 21 → 86 px per
   iteration; Dice *fell*) because a few coarse outliers corrupted the template —
   invisible in synthetic data → candidate selection by fine confidence (§8);
7. OpenCV 5.0's `phaseCorrelate` multiplies its **input arrays** by the window,
   in place. Arrays passed more than once were eroded toward their centre:
   the coarse template once per frame of every iteration, and the on-disk
   vesselness maps during chaining and the closure check. Caught when the
   non-rigid smoke test invented 218 px of deformation on a *still* burst, as
   each shared template patch was re-windowed for every frame → every call
   now correlates private copies (`register.phase_correlate`). Translation
   results before stabilize 0.2.0 were affected; the synthetic test passed
   both before and after (the fine stage has its own Fourier transform), so
   real bursts should be re-run. It may also explain the coarse outliers of
   item 6, which appeared on a template eroded to its centre.

**Real-data acceptance.** `15-50-52` (clean): converges 0.82 → 0.07 px; overlap
0.497 → 0.905, Dice 0.587 → 0.933, 99% of frames usable (0.2.0; identical
headline metrics under 0.1.0). `15-22-26` (blinks, 25% lighting swings), under
0.1.0: overlap 0.221 → 0.621, Dice 0.024 → 0.695, 59% usable.

## §13 Non-rigid refinement (experimental)

**Why.** On wide-field bursts, translation left the stabilized mean *doubled*
toward the corners while the centre was sharp. Registering each tile of each
frame separately showed why: after translation, residual tile shifts were
~0.3 px at the centre but up to 3 px at the corners, with opposite corners
moving in opposite directions — rotation and magnification between fixations,
which one shift per frame cannot remove (§10). Clustering the frames by these
residuals found two distinct poses.

**Model.** Each frame's correction is a displacement field (`fields.py`):

    output(x) = frame(x + d(x)),   d(x) = A·[x, y, 1] + L(x)

- **A**, a 2×3 affine per frame: translation, rotation, magnification, shear.
  It is evaluated *exactly* at every pixel. (The prototype interpolated it
  from the patch centres and held it constant beyond the outermost ones,
  which under-corrected the corners — where rotation displaces most.)
- **L**, a local residual known at the patch centres and interpolated
  bicubically: whatever the affine can't describe, bounded to ±8 px so a bad
  patch match can't tear the image, and 3×3-median smoothed.

When the patch grid has only two rows (e.g. 1920×500), vertical gradients are
barely constrained, so the per-frame model is a **similarity** (rotation and
uniform magnification only). Frames too short for a 2×2 grid (strips) **fall
back to translation**, stored in the same layout.

**Measurement.** 320 px patches at 160 px stride (at working scale), each
phase-correlated (§7) against the same patch of a template. A patch counts if
its response ≥ 0.05 and its shift < a quarter patch; a frame needs ≥ 6 such
patches or its field is left unchanged. The affine is fitted to the patch
shifts by **iteratively reweighted least squares with Huber weights**:
ordinary least squares would let one wrong patch — a glare edge, a vessel
that changed — tilt the whole frame, whereas Huber weights shrink the
influence of residuals beyond 1.5× their median.

**Composition.** Each iteration measures a correction *u* on the frame as
currently aligned by *f*. The new field is exactly `g(x) = u(x) + f(x + u(x))`:
for the affines, `M = M_u + M_f + M_f·M_u` and `b = b_u + b_f + M_f·b_u`; the
local residual is resampled at the moved grid nodes. Each original frame is
therefore resampled **once**, with its final field, never warp upon warp.

**Single-pose reference.** A template averaged over two poses is itself
doubled, and matching against it is ambiguous — either copy fits, pulling
frames toward the *average* pose instead of one pose. So the first two
iterations build the template only from the most self-consistent run of 25
consecutive frames (highest mean correlation with their own mean): one
fixation, one pose. Later iterations use every frame.

**Convergence** is tested on the **90th percentile** of per-frame updates
(< 0.25 px), not the median: the frames that cause doubling are the minority
still moving, and a median test stopped after one round while they still
needed 5 px more. At most 6 iterations.

**Rejection.** Frames whose aligned vesselness still correlates poorly with
the final template — robust z < −3.5 *and* more than 0.05 below the median
correlation — are left out of the result, and counted in `metrics.json`.

**Scoring** uses exactly the metrics of translation (§9), on the same frames,
with the non-rigid warp: vessel overlap and Dice before, after translation,
and after non-rigid; residual step; quadrant disagreement; and the per-tile
residual spread that exposed the doubling, for both methods.

**Status: experimental.** Results so far:

- `15-50-52` (full frame, one saccade): 6×11 patch grid, affine, converged in
  3 iterations, no frames rejected. Vessel overlap 0.905 (translation) →
  0.917 on the same 138 frames. Maximum per-tile residual spread 3.16 → 0.34
  px by the pipeline's measure. An independent check — warping the raw frames
  with `fields.npz`, recomputing vesselness, re-registering 3×4 tiles — gives
  4.22 → 0.45 px (0.45 in one tile, ≤ 0.34 elsewhere), and the best k-means
  silhouette of per-frame tile residuals falls from 0.835 (two clear poses,
  18 and 120 frames) to 0.380: the second pose is largely, not entirely,
  removed.
- Synthetic smoke test: a 0.6° pose change is recovered as 0.598°, tile
  spread 2.91 → 0.22 px; a still burst gains median 0.08 px (max 0.15 px) of
  non-translational field — the patch-shift noise floor, no larger than
  translation's own error — with no loss of tile alignment.

The smoke test
(`tests/test_nonrigid_smoke.py`) checks that a still burst gains no
deformation, that a 0.6° pose change is recovered and removes the doubling,
and that strips fall back. What is *not* yet done is the ground-truth
validation translation has (§12): known, spatially varying deformations with
flicker, noise, blinks and glare, and the accuracy of the recovered fields.
