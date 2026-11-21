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
| motion step median (px) | 3.46 | 2.98 | 3.34 | 0.86 | 0.963 |
| motion step p90 (px) | 6.73 | 5.24 | 5.7 | 0.778 | 0.847 |
| saccades per s | 0.463 | 0 | 0.271 | 0 | 0.584 |
| excursion x range (px) | 123 | 38.1 | 42.7 | 0.309 | 0.346 |
| excursion y range (px) | 54.2 | 34.7 | 37.9 | 0.64 | 0.699 |
| motion velocity spectrum slope | -0.365 | -0.0687 | -0.17 | 0.188 | 0.466 |

### Illumination flicker

| statistic | real | synthetic, seed 1 | synthetic, seed 2 | seed 1 / real | seed 2 / real |
|---|---|---|---|---|---|
| flicker line frequency (Hz) | 8.78 | 8.77 | 8.77 | 0.999 | 0.999 |
| flicker line amplitude (%) | 0.891 | 0.916 | 0.917 | 1.03 | 1.03 |
| flicker residual SD (%) | 0.215 | 0.173 | 0.137 | 0.805 | 0.637 |

### Blur and noise

| statistic | real | synthetic, seed 1 | synthetic, seed 2 | seed 1 / real | seed 2 / real |
|---|---|---|---|---|---|
| sharpness median | 0.388 | 0.302 | 0.371 | 0.779 | 0.957 |
| sharpness p10/p90 | 0.957 | 0.951 | 0.95 | 0.993 | 0.993 |
| background level (DN) | 2.65e+03 | 2.51e+03 | 2.58e+03 | 0.949 | 0.976 |
| background temporal SD (DN) | 37.9 | 38.5 | 39.4 | 1.02 | 1.04 |
| noise variance slope (DN) | 0.565 | 0.459 | 0.482 | 0.811 | 0.853 |
| noise variance offset (DN2) | -27.6 | 337 | 326 | -12.2 | -11.8 |

### Scleral texture and illumination

| statistic | real | synthetic, seed 1 | synthetic, seed 2 | seed 1 / real | seed 2 / real |
|---|---|---|---|---|---|
| texture band 0.7 px (SD x1000) | 0.937 | 1.05 | 1.14 | 1.13 | 1.22 |
| texture band 1.4 px (SD x1000) | 0.91 | 1.18 | 1.03 | 1.3 | 1.13 |
| texture band 2.8 px (SD x1000) | 1.58 | 2.23 | 1.68 | 1.41 | 1.06 |
| texture band 5.6 px (SD x1000) | 2.7 | 4.07 | 2.64 | 1.51 | 0.978 |
| texture band 11.2 px (SD x1000) | 4.55 | 7.57 | 4.36 | 1.67 | 0.96 |
| illumination p5/p95 | 0.728 | 0.755 | 0.767 | 1.04 | 1.05 |

### Vessel morphology

| statistic | real | synthetic, seed 1 | synthetic, seed 2 | seed 1 / real | seed 2 / real |
|---|---|---|---|---|---|
| vessel area fraction | 0.292 | 0.334 | 0.329 | 1.14 | 1.13 |
| length density (mm/mm2) | 5.98 | 7.81 | 7.57 | 1.31 | 1.27 |
| junction density (/mm2) | 33.1 | 44 | 46.5 | 1.33 | 1.4 |
| segments per mm2 | 53.6 | 74.2 | 79.4 | 1.38 | 1.48 |
| diameter p5 (um) | 13 | 13 | 13 | 1 | 1 |
| diameter p25 (um) | 32.6 | 26.9 | 23.5 | 0.825 | 0.721 |
| diameter p50 (um) | 46.1 | 41.2 | 41.7 | 0.894 | 0.906 |
| diameter p75 (um) | 69.3 | 55.7 | 65.5 | 0.804 | 0.945 |
| diameter p95 (um) | 102 | 84.8 | 90.6 | 0.832 | 0.889 |
| segment length p25 (um) | 58.9 | 62.8 | 60.1 | 1.07 | 1.02 |
| segment length p50 (um) | 95.6 | 93.7 | 91.6 | 0.98 | 0.959 |
| segment length p75 (um) | 163 | 171 | 141 | 1.05 | 0.862 |
| segment length p95 (um) | 328 | 310 | 297 | 0.947 | 0.907 |
| daughter angle median (deg) | 92.9 | 88.2 | 89.7 | 0.95 | 0.966 |
| junction exponent median | 1.99 | 1.61 | 2.14 | 0.809 | 1.08 |

### Tortuosity, curvature and irregularity

| statistic | real | synthetic, seed 1 | synthetic, seed 2 | seed 1 / real | seed 2 / real |
|---|---|---|---|---|---|
| tortuosity p25 | 1.05 | 1.06 | 1.06 | 1.01 | 1 |
| tortuosity median | 1.08 | 1.08 | 1.08 | 1 | 1 |
| tortuosity p75 | 1.1 | 1.1 | 1.1 | 1 | 1.01 |
| tortuosity p90 | 1.14 | 1.17 | 1.16 | 1.02 | 1.01 |
| curvature p25 (rad/mm) | 7.02 | 6.8 | 7.07 | 0.969 | 1.01 |
| curvature p50 (rad/mm) | 8.91 | 9.26 | 8.85 | 1.04 | 0.993 |
| curvature p75 (rad/mm) | 10.9 | 10.6 | 10.9 | 0.971 | 0.995 |
| curvature p90 (rad/mm) | 12.7 | 14.3 | 13.9 | 1.12 | 1.09 |
| centreline jitter median (um) | 1.14 | 1.2 | 1.09 | 1.05 | 0.953 |
| width variation along vessels (CV, median) | 0.142 | 0.141 | 0.145 | 0.993 | 1.02 |

### Vessels side by side (arteriole-venule pairs)

| statistic | real | synthetic, seed 1 | synthetic, seed 2 | seed 1 / real | seed 2 / real |
|---|---|---|---|---|---|
| paired length share | 0.445 | 0.493 | 0.36 | 1.11 | 0.81 |
| pair spacing median (um) | 85.3 | 84.8 | 81.8 | 0.994 | 0.958 |
| pair calibre ratio median | 1.5 | 1.5 | 1.5 | 1 | 1 |

### Large vessels and darkness

| statistic | real | synthetic, seed 1 | synthetic, seed 2 | seed 1 / real | seed 2 / real |
|---|---|---|---|---|---|
| vessel area share > 50 um | 0.174 | 0.146 | 0.181 | 0.838 | 1.04 |
| length density > 50 um (mm/mm2) | 2.84 | 2.67 | 3.16 | 0.94 | 1.11 |
| vessel area share > 130 um | 0.00371 | 2.7e-05 | 2.99e-05 | 0.00728 | 0.00808 |
| length density > 130 um (mm/mm2) | 0.0459 | 0.000919 | 0.000918 | 0.02 | 0.02 |
| vessel optical density p50 | 0.122 | 0.149 | 0.109 | 1.22 | 0.894 |
| vessel optical density p90 | 0.27 | 0.261 | 0.214 | 0.968 | 0.792 |
| vessel optical density p99 | 0.381 | 0.409 | 0.377 | 1.07 | 0.99 |
| edge steepness > 50 um (1/um) | 0.0253 | 0.0304 | 0.027 | 1.2 | 1.07 |

### Contrast by calibre

| statistic | real | synthetic, seed 1 | synthetic, seed 2 | seed 1 / real | seed 2 / real |
|---|---|---|---|---|---|
| contrast d 7-16 um | 0.0633 | 0.0988 | 0.0724 | 1.56 | 1.14 |
| contrast d 16-29 um | 0.0821 | 0.122 | 0.101 | 1.49 | 1.23 |
| contrast d 29-49 um | 0.114 | 0.155 | 0.104 | 1.35 | 0.912 |
| contrast d 49-82 um | 0.164 | 0.194 | 0.146 | 1.19 | 0.893 |
| contrast d 82-130 um | 0.244 | 0.255 | 0.196 | 1.05 | 0.806 |
| contrast d 130-326 um | 0.295 | - | - | - | - |

### Red-cell flicker in vessels

| statistic | real | synthetic, seed 1 | synthetic, seed 2 | seed 1 / real | seed 2 / real |
|---|---|---|---|---|---|
| flicker SD ratio d 7-16 um | 1.37 | 1.43 | 1.43 | 1.04 | 1.04 |
| flicker SD ratio d 16-29 um | 1.5 | 1.39 | 1.55 | 0.925 | 1.03 |
| flicker SD ratio d 29-49 um | 1.62 | 1.39 | 1.29 | 0.857 | 0.796 |
| flicker SD ratio d 49-82 um | 1.82 | 1.41 | 1.32 | 0.775 | 0.727 |
| flicker SD ratio d 82-130 um | 1.8 | 1.56 | 1.45 | 0.867 | 0.805 |
| flicker excess share 0.5-2 Hz | 0.237 | 0.303 | 0.228 | 1.28 | 0.962 |
| flicker excess share 2-5 Hz | 0.217 | 0.209 | 0.216 | 0.964 | 0.993 |
| flicker excess share 5-10 Hz | 0.191 | 0.178 | 0.241 | 0.932 | 1.26 |
| flicker excess share 10-16 Hz | 0.167 | 0.145 | 0.213 | 0.869 | 1.28 |
| vessel lag-1 autocorrelation | 0.392 | 0.418 | 0.285 | 1.06 | 0.725 |

**Reading the table.**

- *Matched by construction* (measured on the real bursts and fed to the generator, then measured again on its
  output): the motion step and saccade rate, the mains flicker, the sensor noise, the background level, the
  texture band energies. They test that the renderer reproduces what it was given through the full chain.
- *Emergent* (not set directly; they come out of the anatomy, the flow and the optics): vessel length and junction
  density, the diameter distribution, segment lengths, tortuosity, branch angles and junction exponents, contrast
  by calibre, and the red-cell flicker in vessels. The wiggle amplitude, the episcleral vessel count, the spectral
  weights and the clump amplitude were fitted against these, as anatomy §10 records.
- *Darkness.* On one shared optical-density scale (`synth.media`) the real and synthetic vessels are equally dark:
  the darkness percentiles of vessel pixels and the contrast by calibre of the real burst lie between those of the
  two synthetic seeds, as do the area and length of vessels wider than 50 µm, and the synthetic large vessels are
  not blurrier than the real ones (edge steepness). Real and synthetic frames shown raw, each stretched between its
  own percentiles, differ in apparent darkness because their illumination falloff and range differ, not their
  vessels; compare them in optical density. Vessels wider than 130 µm (by the blurred-mask measure) cover 0.4 % of
  the real field and almost none of the synthetic one, whose widest vessels are the cited 50-100 µm episcleral
  veins.
- *Tortuosity and pairs.* The tortuosity and curvature distributions, the fine jitter of the centrelines and the
  variation of calibre along the vessels match the real burst; so do the share of vessel length running beside a
  parallel vessel and the calibre ratio across such pairs.
- *Known differences.* The synthetic fields hold about 1.3 times the real vessel length and junctions per mm²
  detected by the mask (mostly the companion venules and fine conjunctival branches), and their 25th percentile of
  diameter is lower. A communicating vessel that rises steeply can be seen nearly end-on and then shows as a small
  very dark blob, which the real fields rarely show. Large real vessels (> 50 µm) flicker more than synthetic ones: the real tissue deforms
  non-rigidly, so translation-only registration leaves edge flicker the model does not produce (tissue deformation
  is not modelled). The synthetic red-cell texture decorrelates a little faster from frame to frame (vessel lag-1
  autocorrelation lower, more excess power above 5 Hz). Saccades and excursions are random per burst. The
  illumination falloff is a little flatter in the synthetic field, which is smaller than the real crop.


## 2. The anatomy check

`synth/check.py` measures every structure and haemodynamic quantity on the ground truth of seed 1 (not on the
images) against its cited range (anatomy §2-6): the published range, or the published mean ± 2 SD. Volume flow and
red-cell flux are conserved at every interior node to 8e-11 (relative). The tests run the same
check on further seeds and on randomised scenes (`synth/tests`).


| quantity | synthetic | cited range | unit | status | source |
|---|---|---|---|---|---|
| conjunctival vessel depth | 51.4-173 | 42-240 | µm | within | epithelium 42 µm, conjunctiva 240 µm thick [Zhang2013]; 0-200 µm slab [Akagi2018] |
| episcleral vessel depth | 261-432 | 200-470 | µm | within | under conjunctiva and Tenon's [Zhang2013, FernandezVigo2021, Kumar2013]; 200-1000 µm slab [Akagi2018] |
| deep vessel depth | 663-787 | 450-1000 | µm | within | intrascleral, 200-1000 µm slab [Akagi2018]; deep scleral vessels at ~500 µm [Bizheva2011] |
| capillary diameter | 4.5-8.97 | 4-9 | µm | within | capillaries < 9 µm [Moka2020, Koutsiaris2015]; 4.3 ± 0.9 µm retina [Neriyanuri2023] |
| arteriole diameter | 9.03-34 | 5.9-61 | µm | within | range 5.9-42.9 [Khansari2016], 6-61 µm [Khansari2017] |
| venule diameter | 10-55.4 | 6-75 | µm | within | range 6.0-51.6 [Khansari2016], 6-75 µm [Khansari2017] |
| arteriole mean diameter (segments >= 50 µm) | 12.8 | 9-28 | µm | within | 15 ± 3 [Khansari2016], 18 ± 5 µm [Khansari2017], mean ± 2 SD |
| venule mean diameter (segments >= 50 µm) | 20.9 | 14-47 | µm | within | 18 ± 2 [Khansari2016], 20 ± 2 [Khansari2017], 30.4 ± 8.4 µm near the limbus [Wang2016] |
| episcleral vein diameter | 50-91.6 | 50-100 | µm | within | 50-100 µm [Asanad2021]; 61 ± 5 µm [Kim2024] |
| episcleral artery diameter | 40-74.1 | 40-97 | µm | within | episcleral vessels 79 ± 9 µm [MacKenzie2016], mean ± 2 SD, lower end at conjunctival arteriole maxima |
| deep vessel diameter | 24.7-36.2 | 20-50 | µm | within | deep scleral vessels 20-50 µm [Bizheva2011] |
| communicating vessel diameter | 20-20 | 10-20 | µm | within | radial channels 10-20 µm [Bizheva2011] |
| arteriolar junction exponent (median) | 2.6 | 2.59-3.14 |  | within | retinal arteries 2.59 [Luo2017]; Q ~ D^3.01 (2.86-3.14) [MayrovitzRoy1983]; Murray 3 [Murray1926a] |
| venular junction exponent (median) | 2.7 | 1.95-3.14 |  | within | retinal veins 1.95 [Luo2017] to Murray's 3 [Murray1926a, MayrovitzRoy1983] |
| capillary length density | 9.47 | 4.4-12 | mm/mm2 | within | 5.2 ± 1.2 capillaries per mm [Kvernebo2022], x pi/2 for length per area, mean ± 2 SD |
| episcleral arteriovenous anastomoses | 0.659 | 0-1.5 | /mm2 | within | 0.5-1 per mm2 in human episclera [Selbach2005]; a field may hold none |
| capillary red-cell velocity (median) | 0.651 | 0.29-1.5 | mm/s | within | conjunctival capillaries 0.51 mm/s [Moka2020]; nailfold 0.29-1.09 [Hahn2004] |
| arteriolar red-cell velocity (median, >= 6 µm, >= 50 µm long) | 1.49 | 0.4-1.7 | mm/s | within | 0.63 ± 0.17 [Khansari2016], 0.70 ± 0.23 [Khansari2017]; precapillary 1.66 [Koutsiaris2010] |
| venular red-cell velocity (median, >= 6 µm, >= 50 µm long) | 0.388 | 0.25-1 | mm/s | within | 0.54 ± 0.13 [Khansari2016], 0.59 ± 0.17 [Khansari2017], 0.50 ± 0.14 [Hu2018] |
| arteriolar flow (median, >= 11 µm) | 213 | 20-300 | pl/s | within | 86 ± 33 [Khansari2016], 144 pl/s [Khansari2017]; 40 pl/s below 11 µm, 299 above 22 µm [Khansari2016] |
| venular flow (median, >= 11 µm) | 69.4 | 5-462 | pl/s | within | 140 ± 55 [Khansari2016], 175 [Khansari2017]; 5-462 pl/s [Koutsiaris2007] |
| arteriolar wall shear rate (median) | 629 | 150-900 | 1/s | within | 320 ± 132 [Khansari2016]; 488 1/s below 11 µm [Khansari2016] |
| venular wall shear rate (median) | 109 | 30-400 | 1/s | within | 190 ± 46 1/s [Khansari2016]; lower in small venules [Koutsiaris2007] |
| capillary tube haematocrit (median) | 0.188 | 0.1-0.45 |  | within | 10-18 % [Klitzman1979]; length fraction 0.41 in retina [Neriyanuri2023] |
| capillary pressure (median) | 17.4 | 12-35 | mmHg | within | 13-16 mmHg nailfold [Williams1990, Hahn2004]; 30-35 mmHg after precapillary branches [Zweifach1974] |
| capillary linear cell density (median, <= 6 µm) | 30.2 | 23.7-93.7 | cells/mm | within | 48 ± 15, range 23.7-93.7 cells/mm in 4.3 ± 0.9 µm retinal capillaries [Neriyanuri2023]; 55.9 ± 19.3 [Ding2025] |
| precapillary arteriolar pulsatility index | 0.57 | 0.3-1.3 |  | within | PI about 0.8, RI 53 % in 6-12 µm arterioles [Koutsiaris2010] |
| episcleral vein flow (median) | 0.385 | 0-0.84 | µl/min | within | 0.40 ± 0.22 µl/min in 61 µm episcleral veins [Kim2024], mean ± 2 SD |
| companion venule calibre over its arteriole's (median) | 1.58 | 1.27-1.93 |  | within | artery:vein calibre 0.653 ± 0.068 [Yusufu2025], mean ± 2 SD |
| calibre variation along segments >= 15 µm (CV, median) | 0.112 | 0.05-0.4 |  | within | about 0.1 above 15 µm, up to 0.4 below [Kiani1993]; ~0.19 in conjunctiva [Shahidi2010] |
| communicating vessel angle to the surface | 32.2-59.9 | 15-75 | deg | within | oblique course, no measured angle [Asanad2021, Li2011]; docs §2.4 |
| red-cell velocity in arteriovenous anastomoses and communicating vessels (median) | 9.75 mm/s | no published value |  | reference | docs §2.4 |
| venular velocity by diameter, 4-9 / 9-14 / 14-19 / 19-24 µm | - / 0.37 / 0.43 / 0.20 mm/s | 0.48 / 0.64 / 0.82 / 0.88 mm/s |  | reference | [Koutsiaris2015] |
| fractal dimension of the projected network | not computed here | 1.59-1.63 (slit lamp), 1.28-1.33 (OCTA) |  | reference | [Jiang2014, Shi2018, Akagi2018, Liu2019] |
| bifurcation angle (daughter to daughter) | measured on the images (synth.targets) | 78° in retinal arteries |  | reference | [Luo2017] |


Over 25 seeds, every checked quantity is within its cited range in 25 of 25 default scenes and in
23 of 25 randomised scenes (`synth.randomize`, whose appearance ranges are wide on purpose and whose
anatomical ranges reach the edges of the cited ones); the failures are single quantities just past a bound (for
example a median arteriolar speed of 1.7-2 mm/s, or a capillary length density of 12-13 mm/mm2). Training data can
use `random_scene(seed, anatomical=True)`, which redraws until the check passes.


## 3. Flow detection against the truth

The flow notebook ran unchanged on the synthetic burst: foundation.py's preprocessing (it masked the synthetic
glints as dust, as it does on the real bursts), the perfusion map, the centrelines, and kymograph velocities with
their cross-check. `synth/validate.py` registered its centrelines onto the truth's frame 0 (dx 5.0,
dy 3.8 px, correlation peak 0.22) and compared every measured segment's red-cell
velocity with the true cycle-averaged velocity projected along that segment (seed 2: dx 3.1, dy
-2.0 px, peak 0.29). Both columns count the consistent and confirmed segments.

| | seed 1 | seed 2 |
|---|---|---|
| segments | 68 | 39 |
| on a vessel with flow (true speed > 0.3 px/frame) | 68 | 39 |
| of those, true speed within the search limit (20 px/frame) | 54 | 34 |
| median true speed (px/frame) | 8.04 | 5.76 |
| median velocity error (px/frame) | 1.59 | 0.897 |
| all: direction agrees (share) | 0.765 | 0.846 |
| all: median relative speed error | 0.21 | 0.14 |
| all: speed ratio measured/true (median) | 0.887 | 0.907 |
| all: speed rank correlation (Spearman) | 0.514 | 0.479 |
| all: speed correlation (Pearson) | 0.167 | 0.0754 |
| within the limit: direction agrees (share) | 0.926 | 0.971 |
| within the limit: median relative speed error | 0.178 | 0.119 |
| within the limit: speed ratio measured/true (median) | 0.92 | 0.91 |
| within the limit: speed rank correlation (Spearman) | 0.773 | 0.873 |
| within the limit: speed correlation (Pearson) | 0.757 | 0.911 |

By the true vessel kind, both seeds together (consistent and confirmed segments on vessels with flow):

| vessel kind | segments | median true speed (px/frame) | median relative speed error | direction agrees |
|---|---|---|---|---|
| venule | 65 | 5.29 | 0.14 | 0.95 |
| episcleral artery | 20 | 28.71 | 0.79 | 0.30 |
| arteriole | 11 | 12.03 | 0.36 | 0.73 |
| episcleral vein | 9 | 11.78 | 0.09 | 1.00 |
| capillary | 2 | 3.92 | 2.02 | 0.00 |

All 170 measured segments of both seeds by status: confirmed 76, consistent 31, weak 28, inconsistent 19, out of reach 16.

**What it shows.** Where the flow notebook reports a velocity on a vessel whose speed is within its search range,
it is mostly right: direction agrees on more than nine segments in ten and the speed is within about 12-18 %
(median), best on the conjunctival venules, which dominate the measured segments. Its failures are where the
method's limits predict them: the fastest vessels (episcleral arteries and some arterioles above the notebook's
20 px/frame search limit, where the red-cell pattern aliases and a wrong, slower or reversed velocity is reported,
which is why the correlation over all segments is much lower than within the limit), and the widest, deepest
vessels, whose red-cell texture is blurred and saturated and which are mostly not measured at all. A saccade in the
analysed fixation, as in seed 1, also reduces the number of segments it can measure. The synthetic data make that
envelope measurable, which the real bursts cannot.

