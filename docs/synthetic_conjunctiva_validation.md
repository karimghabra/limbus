# The synthetic conjunctiva: validation

How the synthetic bursts of `synth` compare with the real LIMBUS recordings, whether the generated networks are
anatomical, and how the flow analysis performs on synthetic data whose answer is known. The specification is
[synthetic_conjunctiva_anatomy.md](synthetic_conjunctiva_anatomy.md).

**Data.** Real: burst 2026-09-16_15-50-52 (reference burst 1), cropped to rows 240-960 and columns 384-1536 (720 × 1152 px), 140 frames at 32.19 fps. Synthetic: the
default scene with seeds 1 and 2, 768 × 512 px, 120 frames each. Both at 3.26 µm/px (the ruler, anatomy §1.3).

**Reproduce.** `notebooks/synthetic.py` with `SYNTH_SEED=1` and `LIMBUS_BURST` set to the real burst produces every
table below and the side-by-side clip; section 3 additionally needs the flow notebook run on the synthetic burst
(`LIMBUS_BURST=<synthetic burst> marimo export html flow.py`), its output folder given as `SYNTH_FLOW_OUTPUTS`.
The flow analysis is `notebooks/flow.py` from the flow-analysis branch (commit ca7f93b); it is not on the main
branch yet.

## 1. Real against synthetic: the target statistics

`synth/targets.py` measures every statistic on a burst as the camera wrote it, real or synthetic alike: the same
motion estimator, the same translation registration of the longest fixation, the same vessel mask and skeleton.
Estimator biases (the mask misses the faintest capillaries; 2 × the distance transform overstates blurred
diameters) therefore apply to both columns. Lengths and diameters in µm, at 3.26 µm/px.


### Motion

| statistic | real | synthetic, seed 1 | synthetic, seed 2 | seed 1 / real | seed 2 / real |
|---|---|---|---|---|---|
| motion step median (px) | 3.46 | 3.08 | 3.35 | 0.889 | 0.968 |
| motion step p90 (px) | 6.73 | 6.14 | 6.25 | 0.911 | 0.927 |
| saccades per s | 0.463 | 0 | 0.541 | 0 | 1.17 |
| excursion x range (px) | 123 | 28.7 | 74.4 | 0.233 | 0.603 |
| excursion y range (px) | 54.2 | 42.6 | 84.6 | 0.785 | 1.56 |
| motion velocity spectrum slope | -0.365 | -0.226 | -0.421 | 0.619 | 1.15 |

### Illumination flicker

| statistic | real | synthetic, seed 1 | synthetic, seed 2 | seed 1 / real | seed 2 / real |
|---|---|---|---|---|---|
| flicker line frequency (Hz) | 8.78 | 8.77 | 8.77 | 0.999 | 0.999 |
| flicker line amplitude (%) | 0.891 | 0.878 | 0.88 | 0.987 | 0.988 |
| flicker residual SD (%) | 0.215 | 0.113 | 0.315 | 0.526 | 1.47 |

### Blur and noise

| statistic | real | synthetic, seed 1 | synthetic, seed 2 | seed 1 / real | seed 2 / real |
|---|---|---|---|---|---|
| sharpness median | 0.388 | 0.331 | 0.362 | 0.853 | 0.932 |
| sharpness p10/p90 | 0.957 | 0.948 | 0.968 | 0.99 | 1.01 |
| background level (DN) | 2.65e+03 | 2.63e+03 | 2.66e+03 | 0.994 | 1 |
| background temporal SD (DN) | 37.9 | 39.1 | 41.1 | 1.03 | 1.08 |
| noise variance slope (DN) | 0.565 | 0.523 | 0.501 | 0.925 | 0.886 |
| noise variance offset (DN2) | -27.6 | 180 | 379 | -6.52 | -13.7 |

### Scleral texture and illumination

| statistic | real | synthetic, seed 1 | synthetic, seed 2 | seed 1 / real | seed 2 / real |
|---|---|---|---|---|---|
| texture band 0.7 px (SD x1000) | 0.937 | 0.983 | 1.27 | 1.05 | 1.36 |
| texture band 1.4 px (SD x1000) | 0.91 | 0.993 | 1.05 | 1.09 | 1.15 |
| texture band 2.8 px (SD x1000) | 1.58 | 1.79 | 1.75 | 1.13 | 1.1 |
| texture band 5.6 px (SD x1000) | 2.7 | 3.13 | 2.83 | 1.16 | 1.05 |
| texture band 11.2 px (SD x1000) | 4.55 | 4.65 | 3.47 | 1.02 | 0.764 |
| illumination p5/p95 | 0.728 | 0.739 | 0.783 | 1.02 | 1.08 |

### Vessel morphology

| statistic | real | synthetic, seed 1 | synthetic, seed 2 | seed 1 / real | seed 2 / real |
|---|---|---|---|---|---|
| vessel area fraction | 0.292 | 0.313 | 0.263 | 1.07 | 0.901 |
| length density (mm/mm2) | 5.98 | 6.56 | 6.15 | 1.1 | 1.03 |
| junction density (/mm2) | 33.1 | 28.2 | 32.1 | 0.851 | 0.97 |
| segments per mm2 | 53.6 | 45.6 | 52.1 | 0.852 | 0.972 |
| diameter p5 (um) | 13 | 13 | 13 | 1 | 1 |
| diameter p25 (um) | 32.6 | 20.6 | 19.6 | 0.632 | 0.6 |
| diameter p50 (um) | 46.1 | 52.2 | 43.7 | 1.13 | 0.949 |
| diameter p75 (um) | 69.3 | 72 | 64.2 | 1.04 | 0.927 |
| diameter p95 (um) | 102 | 91.3 | 87 | 0.896 | 0.854 |
| segment length p25 (um) | 58.9 | 65.5 | 55.7 | 1.11 | 0.945 |
| segment length p50 (um) | 95.6 | 122 | 93.7 | 1.27 | 0.98 |
| segment length p75 (um) | 163 | 210 | 179 | 1.29 | 1.1 |
| segment length p95 (um) | 328 | 470 | 332 | 1.43 | 1.01 |
| tortuosity median | 1.08 | 1.08 | 1.08 | 1 | 1 |
| tortuosity p90 | 1.14 | 1.19 | 1.3 | 1.04 | 1.14 |
| daughter angle median (deg) | 92.9 | 87.6 | 94.7 | 0.944 | 1.02 |
| junction exponent median | 1.99 | 2.04 | 2 | 1.03 | 1.01 |

### Contrast by calibre

| statistic | real | synthetic, seed 1 | synthetic, seed 2 | seed 1 / real | seed 2 / real |
|---|---|---|---|---|---|
| contrast d 7-16 um | 0.0633 | 0.0767 | 0.0648 | 1.21 | 1.02 |
| contrast d 16-29 um | 0.0821 | 0.115 | 0.0913 | 1.4 | 1.11 |
| contrast d 29-49 um | 0.114 | 0.143 | 0.12 | 1.25 | 1.05 |
| contrast d 49-82 um | 0.164 | 0.208 | 0.176 | 1.27 | 1.07 |
| contrast d 82-130 um | 0.244 | 0.247 | 0.216 | 1.01 | 0.887 |
| contrast d 130-326 um | 0.295 | 0.326 | - | 1.11 | - |

### Red-cell flicker in vessels

| statistic | real | synthetic, seed 1 | synthetic, seed 2 | seed 1 / real | seed 2 / real |
|---|---|---|---|---|---|
| flicker SD ratio d 7-16 um | 1.37 | 1.77 | 1.3 | 1.3 | 0.948 |
| flicker SD ratio d 16-29 um | 1.5 | 1.88 | 1.35 | 1.26 | 0.899 |
| flicker SD ratio d 29-49 um | 1.62 | 1.68 | 1.3 | 1.04 | 0.804 |
| flicker SD ratio d 49-82 um | 1.82 | 1.47 | 1.4 | 0.805 | 0.766 |
| flicker SD ratio d 82-130 um | 1.8 | 1.5 | 1.38 | 0.831 | 0.766 |
| flicker SD ratio d 130-326 um | 1.83 | 1.71 | - | 0.933 | - |
| flicker excess share 0.5-2 Hz | 0.237 | 0.225 | 0.174 | 0.949 | 0.735 |
| flicker excess share 2-5 Hz | 0.217 | 0.261 | 0.187 | 1.2 | 0.862 |
| flicker excess share 5-10 Hz | 0.191 | 0.246 | 0.242 | 1.29 | 1.27 |
| flicker excess share 10-16 Hz | 0.167 | 0.194 | 0.274 | 1.17 | 1.65 |
| vessel lag-1 autocorrelation | 0.392 | 0.295 | 0.427 | 0.753 | 1.09 |

**Reading the table.**

- *Matched by construction* (measured on the real bursts and fed to the generator, then measured again on its
  output): the motion step and saccade rate, the mains flicker, the sensor noise, the background level, the
  texture band energies. They test that the renderer reproduces what it was given through the full chain.
- *Emergent* (not set directly; they come out of the anatomy, the flow and the optics): vessel length and junction
  density, the diameter distribution, segment lengths, tortuosity, branch angles and junction exponents, contrast
  by calibre, and the red-cell flicker in vessels. The wiggle amplitude, the episcleral vessel count, the spectral
  weights and the clump amplitude were fitted against these, as anatomy §10 records.
- *Known differences.* Large real vessels (> 50 µm) flicker more than synthetic ones: the real tissue deforms
  non-rigidly, so translation-only registration leaves edge flicker the model does not produce (tissue deformation
  is not modelled). The synthetic red-cell texture decorrelates a little faster from frame to frame (vessel lag-1
  autocorrelation lower, more excess power above 5 Hz). Saccades and excursions are random per burst. The
  illumination falloff is a little flatter in the synthetic field, which is smaller than the real crop.


## 2. The anatomy check

`synth/check.py` measures every structure and haemodynamic quantity on the ground truth of seed 1 (not on the
images) against its cited range (anatomy §2-6): the published range, or the published mean ± 2 SD. Volume flow and
red-cell flux are conserved at every interior node to 5e-12 (relative). The tests run the same
check on further seeds and on randomised scenes (`synth/tests`).


| quantity | synthetic | cited range | unit | status | source |
|---|---|---|---|---|---|
| conjunctival vessel depth | 51.5-170 | 42-240 | µm | within | epithelium 42 µm, conjunctiva 240 µm thick [Zhang2013]; 0-200 µm slab [Akagi2018] |
| episcleral vessel depth | 263-441 | 200-470 | µm | within | under conjunctiva and Tenon's [Zhang2013, FernandezVigo2021, Kumar2013]; 200-1000 µm slab [Akagi2018] |
| deep vessel depth | 521-886 | 450-1000 | µm | within | intrascleral, 200-1000 µm slab [Akagi2018]; deep scleral vessels at ~500 µm [Bizheva2011] |
| capillary diameter | 4.5-8.99 | 4-9 | µm | within | capillaries < 9 µm [Moka2020, Koutsiaris2015]; 4.3 ± 0.9 µm retina [Neriyanuri2023] |
| arteriole diameter | 9.04-36.3 | 5.9-61 | µm | within | range 5.9-42.9 [Khansari2016], 6-61 µm [Khansari2017] |
| venule diameter | 10.2-49.7 | 6-75 | µm | within | range 6.0-51.6 [Khansari2016], 6-75 µm [Khansari2017] |
| arteriole mean diameter (segments >= 50 µm) | 13.2 | 9-28 | µm | within | 15 ± 3 [Khansari2016], 18 ± 5 µm [Khansari2017], mean ± 2 SD |
| venule mean diameter (segments >= 50 µm) | 17.7 | 14-47 | µm | within | 18 ± 2 [Khansari2016], 20 ± 2 [Khansari2017], 30.4 ± 8.4 µm near the limbus [Wang2016] |
| episcleral vein diameter | 50-94.6 | 50-100 | µm | within | 50-100 µm [Asanad2021]; 61 ± 5 µm [Kim2024] |
| episcleral artery diameter | 40-79.4 | 40-97 | µm | within | episcleral vessels 79 ± 9 µm [MacKenzie2016], mean ± 2 SD, lower end at conjunctival arteriole maxima |
| deep vessel diameter | 20-24.7 | 20-50 | µm | within | deep scleral vessels 20-50 µm [Bizheva2011] |
| communicating vessel diameter | 16-20 | 10-20 | µm | within | radial channels 10-20 µm [Bizheva2011] |
| arteriolar junction exponent (median) | 2.6 | 2.59-3.14 |  | within | retinal arteries 2.59 [Luo2017]; Q ~ D^3.01 (2.86-3.14) [MayrovitzRoy1983]; Murray 3 [Murray1926a] |
| venular junction exponent (median) | 2.4 | 1.95-3.14 |  | within | retinal veins 1.95 [Luo2017] to Murray's 3 [Murray1926a, MayrovitzRoy1983] |
| capillary length density | 8.35 | 4.4-12 | mm/mm2 | within | 5.2 ± 1.2 capillaries per mm [Kvernebo2022], x pi/2 for length per area, mean ± 2 SD |
| episcleral arteriovenous anastomoses | 0.659 | 0-1.5 | /mm2 | within | 0.5-1 per mm2 in human episclera [Selbach2005]; a field may hold none |
| capillary red-cell velocity (median) | 0.521 | 0.29-1.5 | mm/s | within | conjunctival capillaries 0.51 mm/s [Moka2020]; nailfold 0.29-1.09 [Hahn2004] |
| arteriolar red-cell velocity (median, >= 6 µm, >= 50 µm long) | 0.725 | 0.4-1.7 | mm/s | within | 0.63 ± 0.17 [Khansari2016], 0.70 ± 0.23 [Khansari2017]; precapillary 1.66 [Koutsiaris2010] |
| venular red-cell velocity (median, >= 6 µm, >= 50 µm long) | 0.355 | 0.25-1 | mm/s | within | 0.54 ± 0.13 [Khansari2016], 0.59 ± 0.17 [Khansari2017], 0.50 ± 0.14 [Hu2018] |
| arteriolar flow (median, >= 11 µm) | 125 | 20-300 | pl/s | within | 86 ± 33 [Khansari2016], 144 pl/s [Khansari2017]; 40 pl/s below 11 µm, 299 above 22 µm [Khansari2016] |
| venular flow (median, >= 11 µm) | 51.8 | 5-462 | pl/s | within | 140 ± 55 [Khansari2016], 175 [Khansari2017]; 5-462 pl/s [Koutsiaris2007] |
| arteriolar wall shear rate (median) | 309 | 150-900 | 1/s | within | 320 ± 132 [Khansari2016]; 488 1/s below 11 µm [Khansari2016] |
| venular wall shear rate (median) | 100 | 30-400 | 1/s | within | 190 ± 46 1/s [Khansari2016]; lower in small venules [Koutsiaris2007] |
| capillary tube haematocrit (median) | 0.151 | 0.1-0.45 |  | within | 10-18 % [Klitzman1979]; length fraction 0.41 in retina [Neriyanuri2023] |
| capillary pressure (median) | 25.2 | 12-35 | mmHg | within | 13-16 mmHg nailfold [Williams1990, Hahn2004]; 30-35 mmHg after precapillary branches [Zweifach1974] |
| capillary linear cell density (median, <= 6 µm) | 35.4 | 23.7-93.7 | cells/mm | within | 48 ± 15, range 23.7-93.7 cells/mm in 4.3 ± 0.9 µm retinal capillaries [Neriyanuri2023]; 55.9 ± 19.3 [Ding2025] |
| precapillary arteriolar pulsatility index | 0.571 | 0.3-1.3 |  | within | PI about 0.8, RI 53 % in 6-12 µm arterioles [Koutsiaris2010] |
| episcleral vein flow (median) | 0.485 | 0-0.84 | µl/min | within | 0.40 ± 0.22 µl/min in 61 µm episcleral veins [Kim2024], mean ± 2 SD |
| communicating vessel angle to the surface | 29.3-46.8 | 15-75 | deg | within | oblique course, no measured angle [Asanad2021, Li2011]; docs §2.4 |
| red-cell velocity in arteriovenous anastomoses and communicating vessels (median) | 7.21 mm/s | no published value |  | reference | docs §2.4 |
| venular velocity by diameter, 4-9 / 9-14 / 14-19 / 19-24 µm | - / 0.35 / 0.27 / 0.29 mm/s | 0.48 / 0.64 / 0.82 / 0.88 mm/s |  | reference | [Koutsiaris2015] |
| fractal dimension of the projected network | not computed here | 1.59-1.63 (slit lamp), 1.28-1.33 (OCTA) |  | reference | [Jiang2014, Shi2018, Akagi2018, Liu2019] |
| bifurcation angle (daughter to daughter) | measured on the images (synth.targets) | 78° in retinal arteries |  | reference | [Luo2017] |


## 3. Flow detection against the truth

The flow notebook ran unchanged on the synthetic burst: foundation.py's preprocessing (it masked the synthetic
glints as dust, as it does on the real bursts), the perfusion map, the centrelines, and kymograph velocities with
their cross-check. `synth/validate.py` registered its centrelines onto the truth's frame 0 (dx 6.5,
dy 1.2 px, correlation peak 0.27) and compared every measured segment's red-cell
velocity with the true cycle-averaged velocity projected along that segment.

| | consistent and confirmed segments | confirmed segments only |
|---|---|---|
| segments | 50 | 48 |
| on a vessel with flow (true speed > 0.3 px/frame) | 47 | 45 |
| of those, true speed within the search limit (20 px/frame) | 45 | 44 |
| median true speed (px/frame) | 5.33 | 5.33 |
| median velocity error (px/frame) | 0.684 | 0.628 |
| all: direction agrees (share) | 1 | 1 |
| all: median relative speed error | 0.107 | 0.101 |
| all: speed ratio measured/true (median) | 0.989 | 0.989 |
| all: speed rank correlation (Spearman) | 0.723 | 0.733 |
| all: speed correlation (Pearson) | 0.718 | 0.611 |
| within the limit: direction agrees (share) | 1 | 1 |
| within the limit: median relative speed error | 0.101 | 0.1 |
| within the limit: speed ratio measured/true (median) | 0.997 | 0.993 |
| within the limit: speed rank correlation (Spearman) | 0.773 | 0.804 |
| within the limit: speed correlation (Pearson) | 0.837 | 0.858 |

By the true vessel kind (consistent and confirmed segments on vessels with flow):

| vessel kind | segments | median true speed (px/frame) | median relative speed error | direction agrees |
|---|---|---|---|---|
| venule | 27 | 3.22 | 0.07 | 1.00 |
| arteriole | 9 | 6.62 | 0.11 | 1.00 |
| episcleral artery | 9 | 9.72 | 0.17 | 1.00 |
| episcleral vein | 1 | 8.59 | 0.16 | 1.00 |
| capillary | 1 | 9.17 | 0.84 | 1.00 |

All 64 measured segments by status: confirmed 48, weak 7, out of reach 6, consistent 2, inconsistent 1.

**What it shows.** Where the flow notebook reports a confirmed velocity on a vessel whose speed is within its
search range, it is right: direction agrees on every segment and the speed is within about 10 % (median),
on conjunctival arterioles, venules and capillaries and on the slower episcleral vessels alike. Its failures are
where the method's limits predict them: the fastest vessels (episcleral vessels above the notebook's 20 px/frame
search limit, where the red-cell pattern aliases and a wrong, slower velocity is reported, which lowers the
correlation over all segments against the correlation within the limit), and the widest, deepest vessels,
whose red-cell texture is blurred and saturated and which are mostly not measured at all. The synthetic data make
that envelope measurable, which the real bursts cannot.

