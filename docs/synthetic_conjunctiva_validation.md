# The synthetic conjunctiva: validation

How the synthetic bursts of `synth` compare with the real LIMBUS recordings, whether the generated networks are
anatomical, and how the flow analysis performs on synthetic data whose answer is known. The specification is
[synthetic_conjunctiva_anatomy.md](synthetic_conjunctiva_anatomy.md).

**Data.** Real: burst 2026-09-16_15-50-52 (reference burst 1), cropped to rows 240-960 and columns 384-1536 (720 × 1152 px), 140 frames at 32.19 fps. Synthetic: the
default scene with seeds 1 and 2, 768 × 512 px, 120 frames each. Both at 3.26 µm/px (the ruler, anatomy §1.3).
The synthetic figures below were computed with the fixed 350 µm margin and the earlier draw of the eye motion; a
burst now draws its motion first and widens the margin to hold it (anatomy §1.2), so seeds 1 and 2 grow other
networks of the same statistics and a re-run gives figures that differ within the seed-to-seed spread shown here.

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
| motion step median (px) | 3.46 | 4.12 | 3.57 | 1.19 | 1.03 |
| motion step p90 (px) | 6.73 | 7.26 | 7.38 | 1.08 | 1.1 |
| saccades per s | 0.463 | 0.812 | 1.62 | 1.75 | 3.5 |
| excursion x range (px) | 123 | 126 | 147 | 1.02 | 1.19 |
| excursion y range (px) | 54.2 | 77.3 | 87.4 | 1.43 | 1.61 |
| motion velocity spectrum slope | -0.365 | -0.422 | -0.375 | 1.16 | 1.03 |

### Illumination flicker

| statistic | real | synthetic, seed 1 | synthetic, seed 2 | seed 1 / real | seed 2 / real |
|---|---|---|---|---|---|
| flicker line frequency (Hz) | 8.78 | 8.77 | 8.77 | 0.999 | 0.999 |
| flicker line amplitude (%) | 0.891 | 0.864 | 0.863 | 0.97 | 0.969 |
| flicker residual SD (%) | 0.215 | 0.368 | 0.258 | 1.72 | 1.2 |

### Blur and noise

| statistic | real | synthetic, seed 1 | synthetic, seed 2 | seed 1 / real | seed 2 / real |
|---|---|---|---|---|---|
| sharpness median | 0.388 | 0.302 | 0.381 | 0.777 | 0.982 |
| sharpness p10/p90 | 0.957 | 0.975 | 0.951 | 1.02 | 0.994 |
| background level (DN) | 2.65e+03 | 2.62e+03 | 2.69e+03 | 0.99 | 1.02 |
| background temporal SD (DN) | 37.9 | 39.5 | 39.7 | 1.04 | 1.05 |
| noise variance slope (DN) | 0.565 | 0.436 | 0.401 | 0.771 | 0.709 |
| noise variance offset (DN2) | -27.6 | 439 | 522 | -15.9 | -18.9 |

### Scleral texture and illumination

| statistic | real | synthetic, seed 1 | synthetic, seed 2 | seed 1 / real | seed 2 / real |
|---|---|---|---|---|---|
| texture band 0.7 px (SD x1000) | 0.937 | 1.49 | 1.54 | 1.59 | 1.64 |
| texture band 1.4 px (SD x1000) | 0.91 | 1.3 | 1.19 | 1.43 | 1.31 |
| texture band 2.8 px (SD x1000) | 1.58 | 2.12 | 1.76 | 1.34 | 1.11 |
| texture band 5.6 px (SD x1000) | 2.7 | 3.32 | 2.66 | 1.23 | 0.985 |
| texture band 11.2 px (SD x1000) | 4.55 | 4.56 | 3.9 | 1 | 0.858 |
| illumination p5/p95 | 0.728 | 0.766 | 0.756 | 1.05 | 1.04 |

### Vessel morphology

| statistic | real | synthetic, seed 1 | synthetic, seed 2 | seed 1 / real | seed 2 / real |
|---|---|---|---|---|---|
| vessel area fraction | 0.292 | 0.349 | 0.289 | 1.19 | 0.989 |
| length density (mm/mm2) | 5.98 | 7.58 | 6.89 | 1.27 | 1.15 |
| junction density (/mm2) | 33.1 | 40.9 | 42.6 | 1.23 | 1.29 |
| segments per mm2 | 53.6 | 67.6 | 69.9 | 1.26 | 1.31 |
| diameter p5 (um) | 13 | 13 | 9.22 | 1 | 0.707 |
| diameter p25 (um) | 32.6 | 29.2 | 19.6 | 0.894 | 0.6 |
| diameter p50 (um) | 46.1 | 41.2 | 41.2 | 0.894 | 0.894 |
| diameter p75 (um) | 69.3 | 66.5 | 58.7 | 0.959 | 0.847 |
| diameter p95 (um) | 102 | 94.9 | 80.6 | 0.932 | 0.792 |
| segment length p25 (um) | 58.9 | 57.1 | 52.8 | 0.969 | 0.896 |
| segment length p50 (um) | 95.6 | 99.6 | 93.1 | 1.04 | 0.974 |
| segment length p75 (um) | 163 | 165 | 146 | 1.01 | 0.891 |
| segment length p95 (um) | 328 | 305 | 311 | 0.929 | 0.95 |
| junction exponent median | 1.99 | 2.08 | 1.56 | 1.05 | 0.786 |

### Vessel shape: straightness, curvature spectrum, orientation, branching

| statistic | real | synthetic, seed 1 | synthetic, seed 2 | seed 1 / real | seed 2 / real |
|---|---|---|---|---|---|
| straightness p10 | 0.86 | 0.83 | 0.827 | 0.966 | 0.962 |
| straightness p25 | 0.911 | 0.897 | 0.914 | 0.985 | 1 |
| straightness p50 | 0.926 | 0.92 | 0.927 | 0.994 | 1 |
| straight segment share (>= 0.98, length-weighted) | 0.0202 | 0 | 0.0132 | 0 | 0.653 |
| curvature power share 20-60 um | 0.609 | 0.631 | 0.665 | 1.04 | 1.09 |
| curvature power share 60-180 um | 0.324 | 0.316 | 0.237 | 0.974 | 0.73 |
| curvature power share 180-600 um | 0.0667 | 0.0533 | 0.0986 | 0.799 | 1.48 |
| orientation order S | 0.148 | 0.161 | 0.379 | 1.09 | 2.56 |
| length within 20 deg of the dominant orientation | 0.346 | 0.275 | 0.503 | 0.796 | 1.45 |
| daughter angle p25 (deg) | 86 | 77.5 | 78.1 | 0.9 | 0.908 |
| daughter angle p50 (deg) | 95.1 | 91.4 | 89.5 | 0.961 | 0.94 |
| daughter angle p75 (deg) | 114 | 104 | 107 | 0.913 | 0.938 |
| branch turn from the parent's line, median (deg) | 67.8 | 50.8 | 52 | 0.749 | 0.767 |

### Tortuosity, curvature and irregularity

| statistic | real | synthetic, seed 1 | synthetic, seed 2 | seed 1 / real | seed 2 / real |
|---|---|---|---|---|---|
| tortuosity p25 | 1.05 | 1.07 | 1.06 | 1.02 | 1.01 |
| tortuosity median | 1.08 | 1.08 | 1.08 | 1.01 | 1 |
| tortuosity p75 | 1.1 | 1.11 | 1.09 | 1.01 | 0.997 |
| tortuosity p90 | 1.14 | 1.15 | 1.15 | 1.01 | 1 |
| curvature p25 (rad/mm) | 7.02 | 8.02 | 6.79 | 1.14 | 0.968 |
| curvature p50 (rad/mm) | 8.91 | 9.96 | 8.8 | 1.12 | 0.987 |
| curvature p75 (rad/mm) | 10.9 | 11.8 | 11 | 1.08 | 1.01 |
| curvature p90 (rad/mm) | 12.7 | 14 | 13.7 | 1.1 | 1.08 |
| curvature power share 20-60 um | 0.609 | 0.631 | 0.665 | 1.04 | 1.09 |
| curvature power share 60-180 um | 0.324 | 0.316 | 0.237 | 0.974 | 0.73 |
| curvature power share 180-600 um | 0.0667 | 0.0533 | 0.0986 | 0.799 | 1.48 |
| centreline jitter median (um) | 1.14 | 1.43 | 0.978 | 1.26 | 0.858 |
| width variation along vessels (CV, median) | 0.142 | 0.143 | 0.146 | 1 | 1.03 |

### Vessels side by side (arteriole-venule pairs)

| statistic | real | synthetic, seed 1 | synthetic, seed 2 | seed 1 / real | seed 2 / real |
|---|---|---|---|---|---|
| paired length share | 0.445 | 0.454 | 0.463 | 1.02 | 1.04 |
| pair spacing median (um) | 85.3 | 85.8 | 81.2 | 1.01 | 0.952 |
| pair calibre ratio median | 1.5 | 1.41 | 2.69 | 0.943 | 1.79 |

### Large vessels and darkness

| statistic | real | synthetic, seed 1 | synthetic, seed 2 | seed 1 / real | seed 2 / real |
|---|---|---|---|---|---|
| vessel area share > 50 um | 0.174 | 0.183 | 0.159 | 1.05 | 0.915 |
| length density > 50 um (mm/mm2) | 2.84 | 2.93 | 2.73 | 1.03 | 0.959 |
| vessel area share > 130 um | 0.00371 | 0.00164 | 0 | 0.442 | 0 |
| length density > 130 um (mm/mm2) | 0.0459 | 0.0402 | 0 | 0.875 | 0 |
| vessel optical density p50 | 0.122 | 0.159 | 0.113 | 1.3 | 0.926 |
| vessel optical density p90 | 0.27 | 0.275 | 0.208 | 1.02 | 0.769 |
| vessel optical density p99 | 0.381 | 0.401 | 0.305 | 1.05 | 0.801 |
| edge steepness > 50 um (1/um) | 0.0253 | 0.0303 | 0.03 | 1.2 | 1.19 |

### Contrast by calibre

| statistic | real | synthetic, seed 1 | synthetic, seed 2 | seed 1 / real | seed 2 / real |
|---|---|---|---|---|---|
| contrast d 7-16 um | 0.0633 | 0.0856 | 0.0641 | 1.35 | 1.01 |
| contrast d 16-29 um | 0.0821 | 0.119 | 0.092 | 1.46 | 1.12 |
| contrast d 29-49 um | 0.114 | 0.154 | 0.123 | 1.35 | 1.08 |
| contrast d 49-82 um | 0.164 | 0.197 | 0.155 | 1.2 | 0.947 |
| contrast d 82-130 um | 0.244 | 0.237 | 0.198 | 0.974 | 0.815 |
| contrast d 130-326 um | 0.295 | 0.242 | - | 0.822 | - |

### Red-cell flicker in vessels

| statistic | real | synthetic, seed 1 | synthetic, seed 2 | seed 1 / real | seed 2 / real |
|---|---|---|---|---|---|
| flicker SD ratio d 7-16 um | 1.37 | 1.22 | 1.38 | 0.892 | 1.01 |
| flicker SD ratio d 16-29 um | 1.5 | 1.21 | 1.47 | 0.805 | 0.978 |
| flicker SD ratio d 29-49 um | 1.62 | 1.23 | 1.61 | 0.76 | 0.991 |
| flicker SD ratio d 49-82 um | 1.82 | 1.2 | 1.34 | 0.661 | 0.733 |
| flicker SD ratio d 82-130 um | 1.8 | 1.18 | 1.47 | 0.652 | 0.816 |
| flicker SD ratio d 130-326 um | 1.83 | 1.3 | - | 0.711 | - |
| flicker excess share 0.5-2 Hz | 0.237 | 0.273 | 0.466 | 1.15 | 1.97 |
| flicker excess share 2-5 Hz | 0.217 | 0.296 | 0.193 | 1.36 | 0.888 |
| flicker excess share 5-10 Hz | 0.191 | 0.215 | 0.191 | 1.12 | 0.996 |
| flicker excess share 10-16 Hz | 0.167 | 0.192 | 0.15 | 1.15 | 0.901 |
| vessel lag-1 autocorrelation | 0.392 | 0.588 | 0.568 | 1.5 | 1.45 |

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
  diameter is lower. Large real vessels (> 50 µm) flicker more than synthetic ones: the real tissue deforms
  non-rigidly, so translation-only registration leaves edge flicker the model does not produce (tissue deformation
  is not modelled). The synthetic red-cell texture decorrelates a little faster from frame to frame (vessel lag-1
  autocorrelation lower, more excess power above 5 Hz). Saccades and excursions are random per burst. The
  illumination falloff is a little flatter in the synthetic field, which is smaller than the real crop.


## 2. The anatomy check

`synth/check.py` measures every structure and haemodynamic quantity on the ground truth of seed 1 (not on the
images) against its cited range (anatomy §2-6): the published range, or the published mean ± 2 SD. Volume flow and
red-cell flux are conserved at every interior node to 2e-12 (relative). The tests run the same
check on further seeds and on randomised scenes (`synth/tests`).


| quantity | synthetic | cited range | unit | status | source |
|---|---|---|---|---|---|
| conjunctival vessel depth | 51.9-174 | 42-240 | µm | within | epithelium 42 µm, conjunctiva 240 µm thick [Zhang2013]; 0-200 µm slab [Akagi2018] |
| episcleral vessel depth | 251-439 | 200-470 | µm | within | under conjunctiva and Tenon's [Zhang2013, FernandezVigo2021, Kumar2013]; 200-1000 µm slab [Akagi2018] |
| deep vessel depth | 789-815 | 450-1000 | µm | within | intrascleral, 200-1000 µm slab [Akagi2018]; deep scleral vessels at ~500 µm [Bizheva2011] |
| capillary diameter | 4.51-9 | 4-9 | µm | within | capillaries < 9 µm [Moka2020, Koutsiaris2015]; 4.3 ± 0.9 µm retina [Neriyanuri2023] |
| arteriole diameter | 9-47.6 | 5.9-61 | µm | within | range 5.9-42.9 [Khansari2016], 6-61 µm [Khansari2017] |
| venule diameter | 10-46.2 | 6-75 | µm | within | range 6.0-51.6 [Khansari2016], 6-75 µm [Khansari2017] |
| arteriole mean diameter (segments >= 50 µm) | 13.1 | 9-28 | µm | within | 15 ± 3 [Khansari2016], 18 ± 5 µm [Khansari2017], mean ± 2 SD |
| venule mean diameter (segments >= 50 µm) | 18.3 | 14-47 | µm | within | 18 ± 2 [Khansari2016], 20 ± 2 [Khansari2017], 30.4 ± 8.4 µm near the limbus [Wang2016] |
| episcleral vein diameter | 50-92.3 | 50-100 | µm | within | 50-100 µm [Asanad2021]; 61 ± 5 µm [Kim2024] |
| episcleral artery diameter | 40-77.1 | 40-97 | µm | within | episcleral vessels 79 ± 9 µm [MacKenzie2016], mean ± 2 SD, lower end at conjunctival arteriole maxima |
| deep vessel diameter | 45.1-45.1 | 20-50 | µm | within | deep scleral vessels 20-50 µm [Bizheva2011] |
| communicating vessel diameter | 20-20 | 10-20 | µm | within | radial channels 10-20 µm [Bizheva2011] |
| arteriolar junction exponent (median) | 2.6 | 2.59-3.14 |  | within | retinal arteries 2.59 [Luo2017]; Q ~ D^3.01 (2.86-3.14) [MayrovitzRoy1983]; Murray 3 [Murray1926a] |
| venular junction exponent (median) | 2.7 | 1.95-3.14 |  | within | retinal veins 1.95 [Luo2017] to Murray's 3 [Murray1926a, MayrovitzRoy1983] |
| capillary length density | 9.9 | 4.4-12 | mm/mm2 | within | 5.2 ± 1.2 capillaries per mm [Kvernebo2022], x pi/2 for length per area, mean ± 2 SD |
| episcleral arteriovenous anastomoses | 0 | 0-1.5 | /mm2 | within | 0.5-1 per mm2 in human episclera [Selbach2005]; a field may hold none |
| capillary red-cell velocity (median) | 0.688 | 0.29-1.5 | mm/s | within | conjunctival capillaries 0.51 mm/s [Moka2020]; nailfold 0.29-1.09 [Hahn2004] |
| arteriolar red-cell velocity (median, >= 6 µm, >= 50 µm long) | 0.981 | 0.4-1.7 | mm/s | within | 0.63 ± 0.17 [Khansari2016], 0.70 ± 0.23 [Khansari2017]; precapillary 1.66 [Koutsiaris2010] |
| venular red-cell velocity (median, >= 6 µm, >= 50 µm long) | 0.324 | 0.25-1 | mm/s | within | 0.54 ± 0.13 [Khansari2016], 0.59 ± 0.17 [Khansari2017], 0.50 ± 0.14 [Hu2018] |
| arteriolar flow (median, >= 11 µm) | 231 | 20-300 | pl/s | within | 86 ± 33 [Khansari2016], 144 pl/s [Khansari2017]; 40 pl/s below 11 µm, 299 above 22 µm [Khansari2016] |
| venular flow (median, >= 11 µm) | 70.8 | 5-462 | pl/s | within | 140 ± 55 [Khansari2016], 175 [Khansari2017]; 5-462 pl/s [Koutsiaris2007] |
| arteriolar wall shear rate (median) | 314 | 150-900 | 1/s | within | 320 ± 132 [Khansari2016]; 488 1/s below 11 µm [Khansari2016] |
| venular wall shear rate (median) | 104 | 30-400 | 1/s | within | 190 ± 46 1/s [Khansari2016]; lower in small venules [Koutsiaris2007] |
| capillary tube haematocrit (median) | 0.233 | 0.1-0.45 |  | within | 10-18 % [Klitzman1979]; length fraction 0.41 in retina [Neriyanuri2023] |
| capillary pressure (median) | 21 | 12-35 | mmHg | within | 13-16 mmHg nailfold [Williams1990, Hahn2004]; 30-35 mmHg after precapillary branches [Zweifach1974] |
| capillary linear cell density (median, <= 6 µm) | 42.2 | 23.7-93.7 | cells/mm | within | 48 ± 15, range 23.7-93.7 cells/mm in 4.3 ± 0.9 µm retinal capillaries [Neriyanuri2023]; 55.9 ± 19.3 [Ding2025] |
| precapillary arteriolar pulsatility index | 0.577 | 0.3-1.3 |  | within | PI about 0.8, RI 53 % in 6-12 µm arterioles [Koutsiaris2010] |
| episcleral vein flow (median) | 0.919 | 0-0.84 | µl/min | outside | 0.40 ± 0.22 µl/min in 61 µm episcleral veins [Kim2024], mean ± 2 SD |
| companion venule calibre over its arteriole's (median) | 1.53 | 1.27-1.93 |  | within | artery:vein calibre 0.653 ± 0.068 [Yusufu2025], mean ± 2 SD |
| calibre variation along vessels >= 15 µm (CV, median) | 0.0987 | 0.05-0.4 |  | within | about 0.1 above 15 µm, up to 0.4 below [Kiani1993]; ~0.19 in conjunctiva [Shahidi2010] |
| communicating vessel angle to the surface | 33.7-48.6 | 15-75 | deg | within | oblique course, no measured angle [Asanad2021, Li2011]; docs §2.4 |
| red-cell velocity in arteriovenous anastomoses and communicating vessels (median) | 6.99 mm/s | no published value |  | reference | docs §2.4 |
| venular velocity by diameter, 4-9 / 9-14 / 14-19 / 19-24 µm | - / 0.34 / 0.27 / 0.22 mm/s | 0.48 / 0.64 / 0.82 / 0.88 mm/s |  | reference | [Koutsiaris2015] |
| fractal dimension of the projected network | not computed here | 1.59-1.63 (slit lamp), 1.28-1.33 (OCTA) |  | reference | [Jiang2014, Shi2018, Akagi2018, Liu2019] |
| bifurcation angle (daughter to daughter) | measured on the images (synth.targets) | 78° in retinal arteries |  | reference | [Luo2017] |


Over 25 seeds, every checked quantity is within its cited range in 19 of 25 default scenes and in
18 of 25 randomised scenes (`synth.randomize`, whose appearance ranges are wide on purpose and whose
anatomical ranges reach the edges of the cited ones); the failures are single quantities just past a bound (for
example a median arteriolar speed of 1.7-2 mm/s, or a capillary length density of 12-13 mm/mm2). Training data can
use `random_scene(seed, anatomical=True, n_frames=...)`, which redraws until the check passes on the network that
`simulate(scene, n_frames)` renders (the scene comes back with the margin its burst's eye motion needs, anatomy §1.2).


## 3. Flow detection against the truth

The flow notebook ran unchanged on the synthetic burst: foundation.py's preprocessing (it masked the synthetic
glints as dust, as it does on the real bursts), the perfusion map, the centrelines, and kymograph velocities with
their cross-check. `synth/validate.py` registered its centrelines onto the truth's frame 0 (dx 7.2,
dy 6.2 px, correlation peak 0.21) and compared every measured segment's red-cell
velocity with the true cycle-averaged velocity projected along that segment (seed 2: dx 119.4, dy
42.0 px, peak 0.14). Both columns count the consistent and confirmed segments.

| | seed 1 | seed 2 |
|---|---|---|
| segments | 32 | 27 |
| on a vessel with flow (true speed > 0.3 px/frame) | 32 | 27 |
| of those, true speed within the search limit (20 px/frame) | 31 | 27 |
| median true speed (px/frame) | 7.2 | 2.92 |
| median velocity error (px/frame) | 1.04 | 0.395 |
| all: direction agrees (share) | 0.969 | 0.963 |
| all: median relative speed error | 0.121 | 0.16 |
| all: speed ratio measured/true (median) | 0.994 | 0.992 |
| all: speed rank correlation (Spearman) | 0.905 | 0.814 |
| all: speed correlation (Pearson) | 0.874 | 0.663 |
| within the limit: direction agrees (share) | 0.968 | 0.963 |
| within the limit: median relative speed error | 0.117 | 0.16 |
| within the limit: speed ratio measured/true (median) | 1.01 | 0.992 |
| within the limit: speed rank correlation (Spearman) | 0.896 | 0.814 |
| within the limit: speed correlation (Pearson) | 0.886 | 0.663 |

By the true vessel kind, both seeds together (consistent and confirmed segments on vessels with flow):

| vessel kind | segments | median true speed (px/frame) | median relative speed error | direction agrees |
|---|---|---|---|---|
| venule | 36 | 2.87 | 0.14 | 0.94 |
| episcleral artery | 12 | 7.23 | 0.23 | 1.00 |
| arteriole | 9 | 9.36 | 0.11 | 1.00 |
| capillary | 1 | 3.90 | 0.10 | 1.00 |
| episcleral vein | 1 | 12.13 | 0.25 | 1.00 |

All 112 measured segments of both seeds by status: confirmed 52, inconsistent 35, out of reach 9, weak 9, consistent 7.

**What it shows.** Where the flow notebook reports a velocity on a vessel whose speed is within its search range,
it is mostly right: direction agrees on more than nine segments in ten and the speed is within about 12-18 %
(median), best on the conjunctival venules, which dominate the measured segments. Its failures are where the
method's limits predict them: the fastest vessels (episcleral arteries and some arterioles above the notebook's
20 px/frame search limit, where the red-cell pattern aliases and a wrong, slower or reversed velocity is reported,
which is why the correlation over all segments is much lower than within the limit), and the widest, deepest
vessels, whose red-cell texture is blurred and saturated and which are mostly not measured at all. A saccade in the
analysed fixation, as in seed 1, also reduces the number of segments it can measure. The synthetic data make that
envelope measurable, which the real bursts cannot.

