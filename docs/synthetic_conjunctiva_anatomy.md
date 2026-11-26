# The synthetic conjunctiva: anatomy, physiology and optics

This document is the specification of the `synth` package. It gives the anatomical, physiological and optical
facts the generator is built on, each with its source, and says for every parameter of the generator where its
value comes from. The parameters themselves live in `synth/anatomy.py`, each with a comment naming the section or
reference here; the anatomy check (`synth/check.py`) holds every generated network to the ranges in this document;
and the target statistics (`synth/targets.py`) compare every synthetic burst with the real recordings through the
same estimator.

Every number carries one of four tags:

| tag | meaning |
|---|---|
| [Key] | a published value, from the reference list at the end (each DOI checked against PubMed, Crossref or DataCite) |
| measured | measured on the real LIMBUS bursts with `synth/targets.py` (section 8) or, where stated, with the millimetre ruler |
| fitted | a generator setting with no direct anatomical counterpart, tuned so that a cited or measured quantity comes out right; the quantity it is fitted to is named |
| assumed | a modelling choice with no source; each one is listed in section 9 |

Contents: 1 imaging geometry and the globe · 2 layers in depth · 3 calibres · 4 branching · 5 density and tortuosity ·
6 haemodynamics · 7 optics and the camera · 8 targets measured on the real bursts · 9 assumptions and gaps ·
10 parameter table · 11 the whole anterior eye · 12 the patient, the eye in the head, and mapping · references.

## 1. Imaging geometry

### 1.1 Coordinates

The generator works in tissue micrometres: x along the limbus, y radially away from it, z depth below the
conjunctival epithelial surface (positive downwards). The camera looks along z at the centre of its field and
sees the tissue through the curvature of the globe (section 1.4), so a tissue point's pixel is not a plain division
by the pixel scale: the mapping from (x, y, z) to the pixel of frame 0 is given in the `synth.simulate` docstring;
later frames add the eye's motion (section 7.6).

### 1.2 The field

The default field is 768 × 512 pixels (2.50 × 1.67 mm at s = 3.26 µm/px), generated inside a domain at least 350 µm
larger on every side, so that vessels at the image border continue naturally and eye motion never reveals an edge.
`Scene.width_px`, `height_px` and `margin_um` change it. A burst draws its eye motion first (from the scene's seed
and the number of frames) and widens the margin where the motion needs more than `margin_um`: to the largest
excursion, saccades and rotation included, as the deepest vessels show it on the globe (they are drawn nearer the
field's centre, section 1.4), plus 30 µm (`synth.camera.cover_motion`; the whole eye's views and the slit lamp's
bursts do the same). At the default size a burst that moves more than about 60 px needs more than 350 µm, which
most 120-frame bursts do, so for those seeds `simulate(scene)` grows a different (larger) network than
`network.generate(scene)` with the default margin, and the same seed recorded for a different number of frames
may too; `Result.scene` and `truth/scene.json` hold the margin used. The statistics inside the field do not depend
on the margin. The field models bulbar conjunctiva about 3 mm from the limbus, behind
the palisade zone (section 2.3).

### 1.3 Pixel scale (measured)

The recordings do not store the optical magnification. A snapshot of a millimetre ruler taken through the same
camera (`recordings/scale.png`, 1920 × 1200, 2026-11-04) shows six millimetre ticks. Their centroids along three
row bands lie 307.6, 307.1 and 305.9 px apart (least-squares fit over the six ticks; residuals within ±14 px,
from the printing of the ticks), which gives **s = 3.26 ± 0.01 µm/px**, a field of 6.26 × 3.91 mm at full frame.

Caveats: the ruler was imaged two months after the September bursts, and not necessarily at the working distance
of the eye; if the optics were refocused or re-zoomed in between, s differs. Two independent checks agree with
3.26 µm/px: the median red-cell speed measured by the flow notebook on burst 2026-09-16_15-50-52, 177 px/s, is
0.58 mm/s at this scale, inside the published 0.5–0.7 mm/s of conjunctival vessels [Khansari2016, Khansari2017,
Hu2018]; and the finest vessels the camera resolves, about 3 px FWHM (section 8), are 10 µm, the size of
post-capillary venules and precapillary arterioles [Koutsiaris2010, Moka2020]. A calibration target imaged with
each recording session would remove the doubt.

### 1.4 The globe

The conjunctiva covers a curved surface. The anterior scleral radius is 13.12 ± 0.80 mm by AS-OCT 2 mm beyond the
limbus (13.33 ± 1.12 mm nasally, 12.32 ± 0.77 mm temporally; n = 24) [Choi2014], and 11.2 ± 0.3 mm by fitting a
sphere to 3D profilometry (n = 23 emmetropes) [Jesus2017]; the globe is not quite spherical [Atchison2005,
vanVught2021]. The renderer takes a sphere of `eye_radius_um` = 12.5 mm. The generator's surface coordinates (x, y)
are arc lengths from the centre of the camera's field and z is depth along the radius; the camera looks along the
surface normal at the field's centre, optionally tilted (`camera_tilt_deg`). A point at arc distance r and depth z
lies at polar angle θ = r / R and is seen at (R − z) sin θ from the axis, at a depth R (1 − cos θ) + z cos θ
below the tangent plane (`camera.Renderer._globe`). Over a 2.5 × 1.7 mm field the surface falls 90 µm below the
tangent plane at the corners, which the defocus model carries (§7.2); the lateral compression is under 1 %
at the surface and up to about 7 % for the deepest vessels (the deep plexus at the corners), which sit nearer the
eye's centre.

## 2. Layers in depth (3D)

### 2.1 Thicknesses

| layer | thickness | source |
|---|---|---|
| bulbar conjunctival epithelium | 42.4 ± 7.4 µm (n = 711); 47 ± 8 µm; 45 ± 3 µm | [Zhang2013]; [Zhang2011]; [FengSimpson2008] |
| conjunctival stroma (substantia propria) | 197.7 ± 32.5 µm (n = 711) | [Zhang2013] |
| whole conjunctiva | 240.1 ± 29.8 µm | [Zhang2013] |
| conjunctiva + Tenon's capsule, 0–3 mm from the scleral spur | 180–200 µm temporal, 190–200 µm nasal (n = 596) | [FernandezVigo2021] |
| conjunctiva + Tenon's capsule, 3 mm above the superior limbus | 393 ± 67 µm | [Howlett2014] |
| Tenon's capsule alone | 0.21 ± 0.07 mm | [Kumar2013] |
| anterior sclera, 1–3 mm from the spur | 513–590 µm (n = 605) | [FernandezVigo2022] |
| sclera at the limbus (histology) | 0.53 ± 0.14 mm; 0.50 ± 0.11 mm | [Olsen1998]; [Vurgese2012] |
| episclera as a separate layer | not found in any indexed source | section 9 |

### 2.2 The plexuses and their depths

Anterior-segment OCT angiography separates two vascular layers: a superficial slab from the epithelium to about
200 µm holding the conjunctival vessels, and a deep slab from 200 to 1000 µm holding the episcleral and
intrascleral vessels; the vascular pattern changes at about 200 µm [Akagi2018]. Superficial conjunctival vessels
run radially, centrifugally from the limbus, with smaller vessels spread evenly between them; the deep plexus is
dense near the limbus with fewer, Y-shaped segmental vessels running outward [Akagi2018]. Tenon's capsule lies
between the subconjunctival and the episcleral vessel layers [Kemenes2025].

The generator's three layers (`Anatomy.conjunctival_plexus`, `episcleral_plexus`, `deep_plexus`):

| layer | depth below the surface | basis |
|---|---|---|
| conjunctival plexus | 50–230 µm; capillaries just under the epithelium (50–90 µm), arterioles and venules deeper as their calibre grows | epithelium 42 µm and stroma 198 µm [Zhang2013]; superficial slab 0–200 µm [Akagi2018] |
| episcleral plexus | 260–440 µm | beneath the 240 µm conjunctiva and the Tenon's layer [Zhang2013, FernandezVigo2021, Kumar2013]; deep slab [Akagi2018] |
| deep (intrascleral) plexus | 520–900 µm | within the 0.5 mm limbal sclera [Olsen1998, Vurgese2012]; deep scleral vessels 20–50 µm wide at about 500 µm [Bizheva2011]; deep slab 200–1000 µm [Akagi2018] |

Within the conjunctival layer the depth of a vessel rises with its calibre (`network._conjunctival_depth`), and
every centreline undulates smoothly in depth by a few micrometres, so no layer is a flat sheet. Vessels of
different layers cross over and under one another in projection without meeting.

### 2.3 The limbus: arcades, palisades and the anterior ciliary circulation

These structures bound the field rather than appear in it, but they set its plumbing:

- The anterior ciliary arteries (ACAs) leave the rectus muscles at their insertions, 5.4 mm (medial) and 6.8 mm
  (lateral) from the limbus in children [You2026], typically two per rectus and one on the lateral rectus
  (2.05–2.25 major ACAs per horizontal rectus [Johnson2009]). They run forward in the episclera and feed an
  anterior episcleral arterial circle with superficial and deep parts, which supplies the anterior conjunctival and
  episcleral circulations and the limbal arcades [MeyerWatson1987, VanBuskirk1989]. The larger branches perforate
  the sclera to join the intramuscular circle of the ciliary body [MorrisonVanBuskirk1983]. Where the
  superficial circle is missing, single vessels rise from the deep segments to the conjunctival arterioles
  [MeyerWatson1987].
- Flow: 25 of 40 episcleral arteries flowed away from scleral perforations close to the limbus, and all episcleral
  veins studied drained away from the limbus [Meyer1988]. The generator's episcleral arteries therefore run towards
  the limbus and its veins away from it (section 6.2).
- The palisades of Vogt are radial stromal ridges with hairpin vessel loops, about 72 µm wide and 7.4 per mm of
  limbus [Haagdorens2017], in a zone 1.5–2.1 mm long [Kemenes2025]. Superficial palisade vessels and the deep,
  Y-shaped episcleral vessels collect into an arcade at the palisades and a second, less complete episcleral
  arcade about 2 mm behind the limbus, and merge into the marginal corneal arcade [Kemenes2025]; capillary loops
  occur only in the limbal arcades [Selbach2005].

The generator models the bulbar field behind the palisade zone (about 3 mm from the limbus), so it
draws no palisade loops or marginal arcade; the whole eye (section 11) adds the limbal arcade, the palisade hairpins
and the limbal venous ring at its coarse level.

### 2.4 Vessels between the layers

- **Episcleral arteriovenous anastomoses.** The episclera has no capillary bed: its arteries open into an
  interlacing venous network through arteriovenous anastomoses, 0.5–1 per mm² in human episclera [Selbach2005],
  about the size of small arterioles and venules and sitting 1–3 mm behind the limbus in primates [FunkRohen1996].
  In rabbits and dogs an anastomosis is a short arteriolar segment with smooth muscle opening into a wide,
  funnel-like venous segment [RohenFunk1994]. The generator draws `episcleral_avas_per_mm2` = 0.75 per mm²
  (Poisson), each joining an episcleral artery to the nearest episcleral vein 40–400 µm away, with the lumen of
  its arteriolar segment 5–8 µm (`ava_diameter`; assumed: a segment at resting tone, which keeps the shunt flow a
  fraction of the vein's own, section 6.5).
- **Conjunctival–episcleral communicating vessels.** On OCT angiography most deep vessels show no direct link to
  superficial ones but some do; nobody has counted them [Akagi2018]. Ultra-high-resolution OCT shows radial
  channels 10–20 µm wide joining vessels 100–150 µm deep to deeper scleral vessels [Bizheva2011]. Episcleral veins
  surface gradually as they run away from the limbus [Asanad2021] and an aqueous vein has been traced obliquely
  across consecutive B-scans [Li2011]: the course is oblique. The generator draws `communicating_per_mm2` = 1.2 per
  mm² (fitted, section 9), alternately arterial (rising from an episcleral artery to become the root of a
  conjunctival arteriolar tree) and venous (draining a conjunctival venular tree into an episcleral vein), with a
  diameter in the cited 10–20 µm, at an angle to the scleral surface drawn from 20–60° (assumed: no measured
  angle exists, section 9). A vessel 200 µm deep therefore runs 120–550 µm sideways while it rises.
- **Perforating vessels.** Perforators per mm of limbus, their angle of entry and their calibre are not found in
  any indexed source (section 9). The generator draws `perforator_per_mm2` = 0.25 per mm² perforating vessels
  between the episcleral and deep plexuses at 30–70° to the surface, with the calibre of the deep vessels they
  join [Bizheva2011].

## 3. Calibres

| vessel | diameter | source |
|---|---|---|
| capillaries | < 9 µm; 4.4–8.9 µm class; retina 4.3 ± 0.9 µm | [Moka2020, Koutsiaris2015]; [Neriyanuri2023] |
| precapillary (terminal) arterioles | 6–12 µm | [Koutsiaris2010] |
| post-capillary venules | 9–14 and 14–24 µm classes | [Moka2020, Koutsiaris2015] |
| conjunctival arterioles | 15 ± 3 µm (range 5.9–42.9); 18 ± 5 µm (range 6–61) | [Khansari2016]; [Khansari2017] |
| conjunctival venules | 18 ± 2 µm (range 6.0–51.6); 20 ± 2 µm (range 6–75); 30.4 ± 8.4 µm about 1 mm from the limbus | [Khansari2016]; [Khansari2017]; [Wang2016] |
| the visible conjunctival network | 5–70 µm | [Wang2016] |
| episcleral veins | 50–100 µm; 61 ± 5 µm | [Asanad2021]; [Kim2024] |
| episcleral vessels (arteries and veins together) | 78.9 ± 8.7 µm | [MacKenzie2016] |
| deep scleral vessels | 20–50 µm | [Bizheva2011] |
| communicating channels | 10–20 µm | [Bizheva2011] |
| anterior ciliary arteries | no measured lumen diameter found | section 9 |

Arterioles are fewer than venules: about 13 against 52 measured per subject, 1:4 as sampled, though arterioles
are also harder to see [Khansari2017], and about 1:2 in another cohort [Cheung2001]. The venule-to-arteriole
calibre ratio is 1.1–1.2 [Khansari2016, Khansari2017].

In the generator the terminal arterioles take 9–12 µm and the post-capillary venules 10–16 µm (uniform), the mesh
capillaries 4.5–9 µm, and every larger conjunctival vessel takes its calibre from Murray's law upwards (section
4), so the largest arteriole and venule of a tree follow from the tree's size; the anatomy check confirms they
stay within the cited ranges. Episcleral arteries take 40–80 µm, veins 50–100 µm, deep vessels 20–50 µm where a
named vessel starts; past each side branch it narrows by Murray's law, and the side branch takes Murray's share of
its parent, neither falling below its class's cited minimum.

## 4. Branching

- Murray's law, parent d^γ = Σ daughter d^γ with γ = 3, follows from minimum work [Murray1926a] and from uniform
  wall shear stress [Sherman1981, KassabFung1995]. Measured: flow ∝ D^3.01 (99.9 % CI 2.86–3.14) in arterioles
  6–108 µm [MayrovitzRoy1983]; retinal junction exponents 2.59 for arteries and 1.95 for veins, with a bifurcation
  angle of 78° [Luo2017]. In the conjunctiva, flow scales as D^2.62 in one subject [Shahidi2010] and as D² on the
  post-capillary side [Koutsiaris2007].
- Asymmetric bifurcations: d₁/d₀ = 1/b and d₂/d₀ = α/b with α = d₂/d₁ and b = (1 + α³)^(1/3); the larger daughter
  makes the smaller angle with the parent [Zamir1978].

The generator uses γ = 2.6 for arterioles and 2.7 for venules (`murray_exponent_arterial`, `_venous`), inside the
cited 2.59–3.14 and 1.95–3.14. Trees are grown by space colonisation [Runions2007]: a tip grows on towards the
attractors nearest to it, and a node may sprout one side branch, whose first step leaves at 20–70° to the parent's
continuation (an acute junction, the child carrying on roughly in the parent's direction; the larger daughter makes
the smaller angle with the parent [Zamir1978]) and which then bends towards its attractors as it grows. No side
branch leaves within 160 µm of another branch point of the same tree (`branch_spacing_um`), and every unbranched
terminal branch is cut to 450 µm from its branch point (`max_terminal_branch_um`; the capillary mesh serves the
rest): left alone, colonisation sends a trunk through a field of attractors sprouting a comb of parallel unbranched
side branches 50–100 µm apart, which the recordings do not show; both values are fitted by eye and to the measured
length density. Each path between branch points is then straightened towards its chord, keeping 40 % of its
deviation (`path_straightening`), because growth bends a branch round in a circular arc, and the meander of
section 5 is added instead: the recorded vessels bend gently and continuously along their length, in a hierarchy of
branching trees rather than straight rods [Akagi2018, Kemenes2025, Li2011]. The branching angles that result are
measured on the images (section 8).

### 4.1 Arterioles and venules side by side

Arterioles in microvascular beds are commonly accompanied by a venule running beside them; in rat mesentery a
venule counts as paired when it lies within 15 µm of the arteriole [NelloreHarris2002]. Within a pair the venule is
the wider: the retinal artery-to-vein calibre ratio is 0.653 ± 0.068 in 10 151 healthy people [Yusufu2025] (0.66 ±
0.1 [Garg2022]), so a venule is 1.3–1.8 times its arteriole (mean ± 2 SD); across conjunctival vessels as sampled,
venules are 1.1–1.2 times wider than arterioles [Khansari2016, Khansari2017]. No conjunctival measurement of how
often arterioles are accompanied, or how far apart the pair runs, was found. On the real bursts, 44 % of the
skeleton length (away from junctions) has a parallel vessel 13–115 µm away, with a calibre ratio across the pair of
1.5 (measured with `targets.pairing_stats`, section 8).

In the generator every arteriole whose calibre reaches 25 µm (`pair_min_diameter`) gets a companion venule
(`pair_fraction` = 1, fitted to the measured paired share): an offset copy of the arteriole's course on one side,
at an edge-to-edge gap drawn from 20–60 µm that varies slowly along it (fitted to the pair spacing measured on the
recordings, 85 µm centre to centre; the mesentery's definition of a pair, within 15 µm [NelloreHarris2002], drew
the two touching, as crisp doubled tracks the recordings do not show), 5–15 µm deeper, and 1.3–1.5 times the
arteriole's local calibre, the lower part of the cited 1.3–1.8 [Yusufu2025], and no wider than 40 µm (wider
companions, which drain only their arteriole's own territory, carried so little flow that their
cells stagnated into visible clumps), so it is widest at the arteriole's proximal end, where it
leaves. It stops short of the arteriole's terminal end, drains the capillary mesh through a capillary every
100–160 µm (from a mesh node reached at no more than about 34° to the surface, since a capillary diving straight
down to it would be seen end-on as a dark dot), and leaves at the arteriole's proximal end: at the domain border,
into the nearest venule at least 70 % of its calibre, or else through a communicating vein (within the angles of
§2.4) down to the nearest episcleral vein, the way conjunctival veins drain [Meyer1988]; its blood therefore runs
back beside the arteriole that brought it. A companion keeps only its longest stretch whose lumen stays clear of
every other arteriole and of the companions drawn before it, and that neither runs backwards nor comes closer to
its arteriole than planned (on the inside of a bend tighter than the offset, an offset copy folds into a loop), so
two vessels never share lumen space. The ground truth records which arteriole each
companion accompanies (`paired_with_vessel`). In addition, a fifth of the venular growth attractors lie 50–110 µm beside
the arterioles (about the measured pair spacing), so the rest of the venular tree also tends to follow the arterioles.

## 5. Density, mesh and tortuosity

- Functional capillary density in the bulbar conjunctiva: 5.2 ± 1.2 capillaries per mm of test line, against
  11.2 ± 1.8 at the limbus [Kvernebo2022]. By stereology (length per area = π/2 × crossings per length) that is
  about 8 mm of capillary per mm² (my conversion, not a published number).
- Fractal dimension of the network: 1.59–1.63 by slit-lamp imaging [Jiang2014, Shi2018]; 1.28–1.33 by OCTA
  [Akagi2018, Liu2019]. AS-OCTA vessel area density 27.4 % superficial and 23.0 % deep [Akagi2018].
- Conjunctival vessel length per area in mm/mm²: not found (published "length density" is a skeleton-pixel share).
- Measurable venule segments near the limbus: at least 6 per mm² [Wang2016].
- Tortuosity: no absolute values for healthy bulbar conjunctiva were found. Conjunctival tortuosity is reported as
  curve energy, rising with diabetes in capillary-sized vessels and falling in vessels ≥ 40 µm [Owen2008]; arc/chord
  is 1.41 ± 0.13 in a mostly diabetic set of vessels ≥ 40 µm, with diabetic eyes about 0.18 higher than controls
  [Naim2026]. For context, retinal arteries and veins have arc/chord 1.09 and 1.08 in a population of 10 151
  [Yusufu2025], and retinal tortuosity indices are compiled in [CheungCY2011, Khansari2019]. The real bursts give a
  median arc/chord of 1.075 (p90 1.14) and a median mean curvature of 8.9 rad/mm (section 8); the generator is
  calibrated to these measured distributions, not to a literature value.
- Calibre varies along a microvessel: the coefficient of variation along a segment is about 0.1 for vessels over
  15 µm and up to 0.4 for smaller ones in hamster cremaster [Kiani1993], and about 0.19 along 9–24 µm human
  conjunctival vessels, measurement noise included [Shahidi2010].

In the generator:

- The capillary mesh is the Voronoi tessellation of a jittered hexagonal lattice (vertices of degree three, like
  capillary junctions), `capillary_mesh_spacing` = 285 µm between cell centres with 18 % of the ridges absent,
  fitted so that the capillary length density lands near the 8 mm/mm² derived from [Kvernebo2022] (the anatomy
  check accepts 4.4–12 mm/mm², the cited mean ± 2 SD).
- The trees' density is set by the colonisation's kill distance (`growth_kill` = 210 µm) and attractor density
  (120 per mm²), fitted so that the capillary red-cell velocity and the arteriolar flow come out at the cited
  values (section 6.5): the number of terminal arterioles per capillary decides how long the capillary paths
  are, and hence how fast blood crosses them.
- `arterial_trees_per_mm2` = 0.4 and `venous_trees_per_mm2` = 0.4 roots per mm² (fitted): as many arteriolar as
  venular trees, not the arteriole-to-venule ratio of section 3. Their number sets how large each tree grows, and
  so the calibre of its largest vessels, which the anatomy check holds within the cited ranges.
- Episcleral vessels: `episcleral_vessels_per_mm` = 2.2 vessels per mm along the limbus (fitted to the measured
  length density and diameter distribution, section 8; the literature gives no count), running radially [Akagi2018,
  Kemenes2025] with a heading spread of 0.8 rad (`episcleral_heading_spread`) and a random curvature of SD
  0.0008 rad/µm (`episcleral_curvature`, correlation 250 µm) pulled back towards the initial heading over about
  1.2 mm: long runs that bend gently and continuously. Side branches (0.6 per mm) leave at 20–45°, and past each one
  the vessel narrows by Murray's law (never below its cited calibre range), so it tapers along its course.
  Episcleral veins start at a calibre drawn about the measured 61 µm (SD 8 µm, `episcleral_vein_mean_sd` [Kim2024])
  within the cited 50–100 µm [Asanad2021], so the widest trunks stay within the widths seen in the recordings. Deep
  vessels: 0.6 per mm (`deep_vessels_per_mm`).
- Tortuosity is multi-scale: the colonisation paths and a smooth lateral wiggle give each vessel its meander
  (`Anatomy.wiggle`: arterioles 36 µm and venules 40 µm over 220 µm, capillaries a 4 µm bow over 110 µm); on top
  of it vessels above capillary size get gentle bends of 100–300 µm (`bend_wiggle`, 5 µm over 70 µm) and a slight
  lateral jitter (`fine_jitter`, 0.3 µm over 6 µm), tapered to nothing at the junctions;
  capillaries get neither jitter nor bends, so the mesh is a soft polygon of slightly bowed sides. The settings are fitted so that the straightness, tortuosity and curvature
  distributions, the curvature spectrum and the centreline jitter measured on the real bursts are reproduced
  through the same estimator, and by eye against the real frames (section 8).
- The calibre varies along each named vessel as one smooth profile, continuous through every junction it passes
  (`network.assign_calibres`): the nominal diameters of its successive segments are joined by interpolation, so a
  vessel narrows gradually past each branch instead of in a step; a gentle taper of 5–25 % end to end (`taper`);
  a smooth random variation whose amplitude and wavelength depend on the calibre - coefficient of variation 0.18
  over 100 µm up to 12 µm (`width_variation`), 0.08 over 300 µm from 30 µm (`width_variation_large`),
  log-interpolated between - and, on vessels above capillary size, a fine unevenness of the lumen (3 % over 14 µm,
  `wall_irregularity`). Measured along whole vessels the variation is about 0.08 above 15 µm, inside the cited
  0.1–0.4 [Kiani1993] range at its low end and below the conjunctival 0.19 [Shahidi2010], which includes
  measurement noise. An earlier version drew the profile per segment, normalised each segment to its own mean and
  tapered each by 15–60 %, which made the widest vessels look like chains of wedges, so the large-vessel setting
  is deliberately low and long. The course of every vessel also carries a lateral jitter that grows with its
  calibre (`large_jitter`: 0.03 µm per µm of calibre over 0.5 µm per µm, on top of `fine_jitter`), so wide vessels
  are not perfectly smooth. The red cells follow the local lumen: they sit within it and speed up as the square of
  its narrowing, so the flux is the same all along (the flow model uses the nominal diameter).

## 6. Haemodynamics

### 6.1 The flow model

Each segment is a Poiseuille tube, conductance g = π D⁴ / (128 μ L), with the apparent viscosity μ = μ_plasma ×
μ_rel(D, H_D) of the in vivo law [Pries1994], which includes the Fåhræus–Lindqvist effect and the endothelial
surface layer [PriesSecomb2005]. Interior node pressures solve Σⱼ gᵢⱼ (pᵢ − pⱼ) = 0, so volume flow is conserved at
every interior node to machine precision (the anatomy check and the tests verify it). Plasma viscosity is
1.25 mPa s.

Relative apparent viscosity, D in µm [Pries1994, as written in SecombPries2013]:

    μ_rel = [1 + (μ*₀.₄₅ − 1) ((1 − H_D)^C − 1) / ((1 − 0.45)^C − 1) (D / (D − 1.1))²] (D / (D − 1.1))²
    μ*₀.₄₅ = 6 e^(−0.085 D) + 3.2 − 2.44 e^(−0.06 D^0.645)
    C = (0.8 + e^(−0.075 D)) (−1 + 1 / (1 + 10⁻¹¹ D¹²)) + 1 / (1 + 10⁻¹¹ D¹²)

### 6.2 Pressures at the boundary

The field is a window onto a larger network, so wherever a vessel leaves the domain its end takes a pressure
from its side of the circulation and its calibre, log-linear in diameter between the values at 10 and 100 µm:

| boundary | pressure | source |
|---|---|---|
| arterial, 10 µm | 30 mmHg | pressure falls to 30–35 mmHg where 40–50 µm arterioles give off precapillary branches [Zweifach1974] |
| arterial, 100 µm | 50 mmHg | 45–65 mmHg in 40–50 µm arterioles [Zweifach1974] |
| venous, 10 µm | 14 mmHg | within the capillary pressure of 13–16 mmHg in human nailfold capillaries [Williams1990, Hahn2004] |
| venous, 100 µm | 8 mmHg | episcleral venous pressure 7.6 ± 1.3 [Zeimer1983], 6.3–9.6 [Sit2011], 6.4–7.8 mmHg [Arora2017] |
| capillary | none | the capillary mesh is closed at the domain's edge (outside the camera's field), so capillary blood enters and leaves only through arterioles and venules; open capillary ends at a fixed pressure acted as sinks that drained the arterioles |

Episcleral and deep vessels carry blood along the radius, arteries towards the limbus and veins away from it
[Meyer1988]; their boundary pressures therefore carry a gradient of 0.15 mmHg per mm along y
(`radial_pressure_gradient_mmhg_per_mm`, fitted so that episcleral-vein flow lands in the 0.40 ± 0.22 µl/min
measured by erythrocyte-mediated angiography [Kim2024]; the anatomy check holds it there).

### 6.3 Haematocrit, phase separation and the pulse

- Discharge haematocrit entering the field: 0.40 (systemic 40–45 % [SecombPries2013]; microvascular discharge
  haematocrit is below systemic, [Pries1990]).
- Tube haematocrit from discharge haematocrit (Fåhræus effect) [Pries1992, as written in Fedosov2010]:

      H_T / H_D = H_D + (1 − H_D) (1 + 1.7 e^(−0.35 D) − 0.6 e^(−0.01 D))

- Phase separation at diverging bifurcations [Pries1990, with the parameters of PriesSecomb2005]:

      logit FQ_E = A + B logit((FQ_B − X₀) / (1 − 2 X₀)),  FQ_E = 0 below X₀ and 1 above 1 − X₀
      A = −13.29 ((D_α²/D_β² − 1) / (D_α²/D_β² + 1)) (1 − H_F) / D_F
      B = 1 + 6.98 (1 − H_F) / D_F
      X₀ = 0.964 (1 − H_F) / D_F

  FQ_B is the daughter's share of the blood flow and FQ_E its share of the red cells; D_F and H_F are the parent's
  diameter and discharge haematocrit. Junctions with more than two outflows split off the largest-flow daughter
  against the rest, repeatedly. The discharge haematocrit is propagated downstream node by node in order of
  falling pressure, so red-cell flux is conserved at every junction; flow and haematocrit are iterated to a fixed
  point (damped, typically 50–300 iterations to a change below 10⁻⁵).
- Measured capillary haematocrits: tube haematocrit 10–18 % in hamster muscle [Klitzman1979]; a length-fraction
  of 0.41 in human retinal capillaries [Neriyanuri2023].
- The cardiac pulse: heart rate 1.1 Hz (cardiac period 909 ± 86 ms [Neriyanuri2023]). The arterial inlets carry
  a pressure wave of a generic arterial shape (a systolic peak at 17 % of the period and a dicrotic wave at 42 %,
  `pulse_shape`, expanded in four harmonics) whose half pulse pressure is 12 % of the mean (`arterial_pulsatility`,
  fitted). The vessels are distensible: every node stores a volume C dp/dt, C being half the volume of the vessels
  meeting there times their volume distensibility - 0.002 per mmHg on the arterial side, 0.03 in the capillaries
  and 0.3 on the venous side (`distensibility_per_mmhg`, fitted; veins are far more distensible than arteries). With
  the viscosities frozen at their mean values the system is linear, and each harmonic k solves
  Σ_j g_ij (P_i − P_j) + i k ω C_i P_i = 0 with the inlet wave at the arterial boundaries. The pulse is therefore
  strongest in the arterioles, damped and delayed through the capillaries, and small in the venules; volume is
  conserved at every node with the stored volume included, and exactly over the cycle (tested). Six default scenes
  give a pulsatility index of 0.5–1.3 in 6–12 µm arterioles against the published 0.8 (resistive index 53 %)
  [Koutsiaris2010], 0.04–0.10 in venules of 15 µm and more (venous flow pulses far less than arterial: retinal
  veins against arteries by laser Doppler holography [Puyo2019]), and a capillary pulse pressure of 0.9–3.9 mmHg
  against 1.5–4 mmHg [Zweifach1974, Hahn2004]. The venous peak follows the arteriolar one by about a tenth of the
  period. An earlier version modulated only the arterially driven part of the flow, in phase everywhere, which
  pulsed the venules as much as the arterioles.

### 6.4 Red cells

| quantity | value | source |
|---|---|---|
| red-cell volume | 90 µm³; 75 ± 25 fL in retinal capillaries | [Bosschaart2014]; [Ding2025] |
| red-cell diameter | 7.82 µm | [Fedosov2010, from Evans & Fung] |
| cell-free layer | 0.8 µm at 10 µm rising to 3.1 µm at 50 µm, fluctuating 30–70 % | [Kim2007] |
| velocity profile | v_centre / v_mean = 1.39–1.54, median 1.50 | [Tangelder1986] |
| single-file flow | in capillaries up to about 8 µm | [SecombPries2013] |
| capillary inter-cell distance | 20.5 ± 8.6 µm (CV 0.42); 48 ± 15 cells/mm | [Ding2025]; [Neriyanuri2023] |
| axial correlation of red-cell structure | 18–28 µm at H_T 0.30–0.45; CFL correlation about 30 µm | [Fedosov2010]; [Kim2007] |

In the generator every segment holds N = H_T A L / V_rbc cells. A cell has an axial position s and a position
(r, θ) in the circular cross-section, uniform by area inside the core left by the cell-free layer (linear in
diameter between the cited 0.8 and 3.1 µm); in capillaries narrower than a red cell plus 1 µm the cells run on the
axis. Speeds follow u(ρ) = u_max (1 − ρ^K) with K = 4, which gives the cited v_centre / v_mean = (K + 2) / K = 1.5,
scaled so that the mean over the cells equals the segment's red-cell velocity at that instant. Axial gaps are
gamma distributed with the cited CV 0.42, modulated by a log-normal density field with a 25 µm correlation length
(cited) and a relative amplitude of 0.25 (fitted to the measured flicker, section 8): aggregates and plasma gaps.
At a junction, arriving cells are shared among the outgoing segments by a largest-remainder quota on the
phase-separated red-cell fluxes, so the split is exact; cells leave at outflow boundaries and enter at inflow
boundaries at the segment's red-cell flux.

### 6.5 Velocities and flows (the targets of the fitted settings)

| vessel | red-cell velocity | flow | wall shear rate | source |
|---|---|---|---|---|
| arterioles | 0.63 ± 0.17 mm/s; 0.70 ± 0.23 mm/s | 86 ± 33 pl/s; 144 pl/s | 320 ± 132 s⁻¹ | [Khansari2016]; [Khansari2017] |
| arterioles < 11 µm / > 22 µm | 0.70 / 0.87 mm/s | 40 / 299 pl/s | 488 s⁻¹ (< 11 µm) | [Khansari2016] |
| precapillary arterioles 6–12 µm | cycle mean 1.66 mm/s (0.52–3.26) | | | [Koutsiaris2010] |
| venules | 0.54 ± 0.13; 0.59 ± 0.17; 0.50 ± 0.14 mm/s | 140 ± 55; 175 pl/s | 190 ± 46 s⁻¹ | [Khansari2016]; [Khansari2017]; [Hu2018] |
| venules 4–9 / 9–14 / 14–19 / 19–24 µm | 0.48 / 0.64 / 0.82 / 0.88 mm/s | 5–462 pl/s, ∝ D² | | [Koutsiaris2015]; [Koutsiaris2007] |
| capillaries | 0.51 mm/s (conjunctiva); 0.29–1.09 mm/s (nailfold) | | | [Moka2020]; [Hahn2004] |
| episcleral veins (61 µm) | about 2.3 mm/s (derived from flow and calibre) | 0.40 ± 0.22 µl/min | | [Kim2024] |
| arteriovenous anastomoses, communicating vessels | no published value | | | |

## 7. Optics and the camera

### 7.1 Absorption

Whole blood at haematocrit 45 %, absorption coefficient μa (mm⁻¹, oxy / deoxy) [Bosschaart2014]: 450 nm 31.5 / 33.8;
470 nm 17.1 / 8.3; 500 nm 11.1 / 9.5; 520 nm 13.3 / 14.0; 540 nm 27.1 / 22.1; 550 nm 23.9 / 26.0; 570 nm 24.4 / 23.3;
580 nm 25.8 / 18.9; 600 nm 2.6 / 7.5; 625 nm 0.35 / 2.5; 650 nm 0.16 / 1.8. Molar extinction coefficients of
haemoglobin [Prahl1998]: 53 236 / 46 592 cm⁻¹ M⁻¹ at 540 nm, near-isosbestic points at 530, 545 and 570 nm.
Conjunctival imaging systems use 530 or 540 nm filters for vessel contrast [Shahidi2010, Felder2016].

The LIMBUS illumination's spectrum is not recorded. The measured contrast of the real vessels says it is
broadband: contrast rises with calibre but saturates near 0.3–0.5 even in 100 µm vessels (section 8), whereas
green light would make a 100 µm vessel nearly black (single-pass transmission exp(−27 mm⁻¹ × 0.1 mm × 0.8) ≈ 0.11,
and less again in double pass). With broadband light the red part, which haemoglobin barely absorbs, passes even
a thick vessel. The renderer therefore splits the detected light into nine spectral bands at 450, 470, 500, 520,
545, 575, 600, 625 and 660 nm, each with the mean oxy/deoxy whole-blood absorption coefficient above, and weights
(`Optics.spectrum`) fitted to the measured contrast by calibre. Each red cell adds an optical depth
τ = μa V_rbc / (0.45 A_pixel) per band, spread over its footprint; the detected transmission of a slab is
Σ_k w_k exp(−2 τ_k) (light passes the vessel on its way to the sclera and back).

### 7.2 Scattering and depth

The sclera reflects more than 40 % of visible light diffusely, and transmits only 6 % at 442 nm [Vogel1991]; the
conjunctiva and sclera have been characterised from 300 to 1200 nm [Nemati1996], though no visible-wavelength
values for human conjunctiva were found: a recent Monte-Carlo model of the eye wall had to borrow rabbit values
[Grippo2025], and tissue optics reviews give no conjunctival entry [Jacques2013] (section 9). The transport mean free path of blood at 540 nm is 0.4–0.8 mm [Bosschaart2014].

The renderer models two consequences of scattering, each with one fitted length:

- Not all detected light reaches a vessel: a share 1 − exp(−z / l) turns back above depth z
  (`scatter_veil_length_um` = 650 µm, fitted to the measured contrast of the largest vessels). Deeper vessels
  therefore lose contrast.
- Light from deeper is blurred by the tissue above: σ_scatter = 16 µm × (z / 500 µm)^1.5 (the power of the
  forward-scattering regime; `scatter_blur_um_at_500`, fitted to the measured sharpness), added in quadrature to
  the camera's point spread (σ = 0.9 px, below the ≤ 1.2 px bound measured on the real bursts) and to the defocus
  (σ = 0.06 × |z − z_focus|, with the focal plane at 250 µm, between the conjunctival and episcleral plexuses:
  fitted to the edge steepness of the fine vessels, 8–25 µm, which are softer in the recordings than a focus on
  the conjunctiva gives).

The cells are splatted into nine depth slabs (`camera.SLAB_EDGES`), each blurred with its own σ, and the slabs'
transmissions multiply: a deep vessel shows blurred and faint beneath a sharp superficial one, as in the
recordings. Because the globe curves away from the camera (§1.4), the distance from the focal plane grows towards
the field's edges: each slab is blurred at four levels of the surface's sagitta and the results blended pixel by
pixel, so focus falls off across the field as well as with depth.

At a junction the lumens merge: the last stretch of a narrower vessel lies inside the widest one. A vessel that is
neither the widest at a junction of three or more nor that vessel's straightest continuation therefore does not
draw its cells within the widest vessel's radius of the junction; otherwise the cell clouds of all the vessels
meeting there would add up into a dark blob that no real junction shows. Companion venules likewise keep only the
longest stretch whose lumen stays clear of other arterioles and companions (§4.1).

### 7.3 Scleral texture and illumination

The vessel-free background of the real mean frames has band-pass standard deviations of 0.94, 0.91, 1.58, 2.70
and 4.55 × 10⁻³ (ln units) in the bands between Gaussian scales 0, 0.7, 1.4, 2.8, 5.6 and 11.2 px (measured,
section 8). The renderer's texture is a static sum of independent band-limited Gaussian fields fixed to the tissue,
whose amplitudes (`Optics.texture_amplitudes`) are the non-negative least-squares fit that reproduces those five
numbers once sensor noise, registration and averaging are included (`camera.calibrate_texture`). The illumination
falls off smoothly from a centre near the middle of the field and is fixed to the camera, and flickers at
|120 − 4 × 32.19| = 8.77 Hz, the alias of 120 Hz mains at the frame rate (measured: 8.78 Hz, 0.89 % amplitude; set
to 1.11 % in linear light, since the camera's gamma 0.8 scales a small relative change by 0.8), plus a white
0.12 % (fitted: the measured 0.21 % residual also contains the effect of eye motion on the frame mean).

### 7.4 Motion blur

Each frame averages three sub-exposures spread over the 12.8 ms exposure (measured from the recordings'
manifests), so cells and the eye smear as far as they move during the exposure.

### 7.5 The sensor

Sony IMX249 behind the camera's gamma LUT, from photon-transfer measurements of the real bursts (measured): gain
1.344 DN per electron at 20.4 dB, read noise 6.8 e⁻, gamma 0.8 applied to the 12-bit linear value, ceiling
4091 DN, even/odd row offset 3.56 DN; background 2650 DN (measured on the 2026-09-16 bursts). The renderer draws
Poisson electrons, adds Gaussian read noise, applies the gain, the LUT and integer rounding.

### 7.6 Eye and camera motion

Fixational eye movements: microsaccades of 20′ ± 7′ at 1.3 per second, drift of 52′/s with the head fixed and
about 200′/s head-free, tremor at about 84 Hz below 0.5′ [Cherici2012, and the sources in the physiology sheet];
1′ of rotation is about 3.5 µm at the conjunctiva (about one pixel). During conjunctival slit-lamp recording with
a chin rest the image moves with an SD of 176 ± 24 µm over 5 s [Felder2016].

The renderer draws the motion statistics measured on the real bursts (section 8): a mean-reverting random walk
whose median frame-to-frame step is 4.1 px (the estimator measures 3.46 px on the real burst, and on synthetic
bursts, whose motion is known, it under-reads the step by about 15 %, so 4.1 px reads as the measured 3.46 px),
saccades at 0.46 per second of 40–130 px, never two within 0.25 s (assumed; microsaccades follow one another
after hundreds of milliseconds), a slow rotation
of about 0.05°, and a focal plane that wanders smoothly by ±25 µm about 250 µm (assumed amplitude; it makes the
sharpness of each layer change from frame to frame).

## 8. Targets measured on the real bursts

`synth/targets.py` measures every statistic below on a burst as the camera wrote it; the same code runs on the
synthetic bursts, and `notebooks/synthetic.py` prints the two side by side. The real values are from burst
2026-09-16_15-50-52, cropped to 720 × 1152 px (the crop the flow notebook uses), 140 frames at 32.19 fps, at
3.26 µm/px; the longest fixation (91 frames) is registered by translation and averaged for the still statistics.

The real values and their synthetic counterparts, statistic by statistic, are in
[synthetic_conjunctiva_validation.md](synthetic_conjunctiva_validation.md) and in §7 of `notebooks/synthetic.py`.
The values the generator takes directly from them: frame rate and exposure (manifest); motion step, saccade rate
and excursion; the mains flicker and its white residual; photon-transfer gain, read noise, gamma and row offset;
the background level; the texture band energies and the illumination falloff. The values it was fitted against:
tortuosity, the diameter distribution and vessel density, contrast by calibre, sharpness and red-cell flicker.

## 9. Assumptions and gaps

| item | status | what the generator does |
|---|---|---|
| episclera thickness | not found | the episcleral plexus sits beneath conjunctiva + Tenon's (260–440 µm) |
| perforator count, angle and calibre; ACA perforation distance and calibre | not found (a 2–5 mm band from the limbus is a bound from muscle-insertion distances, not a measurement) | 0.25 perforators per mm², 30–70°, calibre of the deep vessels |
| conjunctival–episcleral communicating vessels per mm² | not counted in any source [Akagi2018] | 1.2 per mm², fitted so that a typical field holds a few |
| angle of communicating vessels | oblique course reported, no angle measured [Asanad2021, Li2011] | 20–60° to the surface |
| arteriole–venule pairing frequency in the conjunctiva | not found | every arteriole of 25 µm or more accompanied, fitted to the measured paired share and by eye (companions of smaller arterioles drew crisp double tracks the fine net of the recordings does not show); gap fitted to the measured pair spacing (the mesentery definition, within 15 µm [NelloreHarris2002], drew them touching) |
| episcleral vessel count per mm of limbus | not found | 2.2 per mm, fitted to the measured diameter distribution |
| episcleral and ACA velocities and pressures | not found | radial pressure gradient fitted to keep large-vessel speeds within the measured range |
| conjunctival tortuosity in healthy eyes | not found (only diabetic sets) | meander, jitter and bends fitted to the measured tortuosity, curvature and jitter |
| illumination spectrum | not recorded | spectral weights fitted to the measured contrast by calibre |
| visible-wavelength optics of human conjunctiva and sclera | not found | veil length and scattering blur fitted to measured contrast and sharpness |
| depth of focus and focal wander | not recorded | defocus 0.06 µm/µm, focus 250 µm, wander ±25 µm |
| fractal dimension and bifurcation angles | published for slit lamp and retina | not imposed; measured on the images |
| geometric drawing details | no anatomical counterpart | episcleral and deep curvature (0.0008 rad/µm, correlation 250 µm), side-branch angles (20-45° off the episcleral and deep vessels, 20-70° in the conjunctival trees), depth undulation of centrelines (5-10 µm), mesh lattice jitter (0.3 of the spacing) and stretch (1.0-1.3), anastomosis lengths (60-350 µm), conjunctival depth by calibre (`network._conjunctival_depth`), the renderer's nine depth slabs and three sub-exposures, glint sizes (1.2-3 px); all set by eye against the recordings and the measured targets |

## 10. Parameter table

Every field of `synth.anatomy.Anatomy`, `Physiology` and `Optics`, its default and its basis. Sections refer to
this document.

| parameter | default | basis |
|---|---|---|
| um_per_px | 3.26 | measured, §1.3 |
| eye_radius_um, camera_tilt_deg | 12.5 mm, (0, 0) | [Choi2014, Jesus2017], §1.4 |
| epithelium | 45 µm | [Zhang2013, Zhang2011, FengSimpson2008], §2.1 |
| conjunctival_plexus | 50–230 µm | [Zhang2013, Akagi2018], §2.2 |
| episcleral_plexus | 260–440 µm | [Zhang2013, FernandezVigo2021, Kumar2013, Akagi2018], §2.2 |
| deep_plexus | 520–900 µm | [Olsen1998, Vurgese2012, Bizheva2011, Akagi2018], §2.2 |
| capillary_diameter | 4.5–9 µm | [Moka2020, Koutsiaris2015, Neriyanuri2023], §3 |
| terminal_arteriole_diameter | 9–12 µm | within 6–12 µm [Koutsiaris2010], §3 |
| postcapillary_venule_diameter | 10–16 µm | within the 9–14 and 14–24 µm classes [Moka2020, Koutsiaris2015], §3 |
| episcleral_artery_diameter | 40–80 µm | [MacKenzie2016], §3 |
| episcleral_vein_diameter | 50–100 µm | [Asanad2021, Kim2024], §3 |
| episcleral_vein_mean_sd | 61 ± 8 µm | drawn about the measured mean [Kim2024], within the range above, §3 |
| max_arteriole_diameter, max_venule_diameter | 58 µm, 72 µm | inside the largest conjunctival arteriole (61 µm) and venule (75 µm) [Khansari2017], §3 |
| deep_vessel_diameter | 20–50 µm | [Bizheva2011], §3 |
| communicating_diameter | 10–20 µm | [Bizheva2011], §2.4 |
| perforating_artery_diameter | 20–50 µm | [Bizheva2011], §2.4 |
| murray_exponent_arterial | 2.6 | retinal arteries 2.59 [Luo2017]; within [Murray1926a, MayrovitzRoy1983], §4 |
| murray_exponent_venous | 2.7 | within 1.95–3.14 [Luo2017, Murray1926a], §4 |
| capillary_mesh_spacing | 285 µm | fitted to capillary length density [Kvernebo2022], §5 |
| capillary_edge_drop | 0.18 | assumed (open polygons), §5 |
| radial_bias | 0.15 | radial course of conjunctival vessels [Akagi2018], strength fitted to the measured tortuosity, §5 |
| arterial_trees_per_mm2, venous_trees_per_mm2 | 0.4, 0.4 | fitted, §5 |
| av_pair_offset, av_pair_fraction | 50–110 µm, 0.2 | assumed, §4.1 and §9 |
| pair_fraction, pair_min_diameter, pair_gap_um, pair_calibre_ratio, pair_max_venule_diameter | 1.0, 25 µm, 20–60 µm, 1.3–1.5, 40 µm | fitted; fitted; fitted to the measured pair spacing; within [Yusufu2025]; fitted, §4.1 |
| fine_jitter, large_jitter, bend_wiggle, width_variation, width_variation_large, wall_irregularity, taper, episcleral_curvature, episcleral_heading_spread, path_straightening, branch_spacing_um, max_terminal_branch_um | 0.3 µm / 6 µm; 0.03 / 0.5 per µm of calibre; 5 µm / 70 µm; CV 0.18 / 100 µm; CV 0.08 / 300 µm; 3 % / 14 µm; 5–25 % per vessel; 0.0008 rad/µm; 0.8 rad; 0.4; 160 µm; 450 µm | fitted to the measured curvature, jitter and width variation; width CV within [Kiani1993, Shahidi2010], §5 |
| anastomoses_per_mm2 | 2.0 | assumed (loops between trees), §5 |
| episcleral_vessels_per_mm | 2.2 | fitted to the measured diameters, §5 |
| episcleral_avas_per_mm2, ava_diameter | 0.75, 5–8 µm | [Selbach2005]; structure [RohenFunk1994], lumen assumed, §2.4 |
| deep_vessels_per_mm | 0.6 | assumed, §2.2 |
| communicating_per_mm2, communicating_angle_deg | 1.2, 20–60° | fitted / assumed, §2.4 and §9 |
| perforator_per_mm2, perforator_angle_deg | 0.25, 30–70° | assumed, §2.4 and §9 |
| wiggle | per kind | fitted to the measured tortuosity, §5 |
| growth_step, growth_influence, growth_kill, attractors_per_mm2, growth_persistence | 18 µm, 400 µm, 210 µm, 120 /mm², 0.8 | fitted to capillary velocity and arteriolar flow, §5 and §6.5 |
| inflow_discharge_haematocrit | 0.40 | [SecombPries2013, Pries1990], §6.3 |
| plasma_viscosity_mpas | 1.25 | [Pries1992], §6.1 |
| rbc_volume_um3, rbc_diameter_um, rbc_thickness_um | 90, 7.8, 2.2 | [Bosschaart2014, Fedosov2010], §6.4 |
| arterial_pressure_at_10um, _at_100um | 30, 50 mmHg | [Zweifach1974], §6.2 |
| venous_pressure_at_10um, _at_100um | 14, 8 mmHg | [Williams1990, Hahn2004, Zeimer1983, Sit2011, Arora2017], §6.2 |
| radial_pressure_gradient_mmhg_per_mm | 0.15 | direction [Meyer1988], value fitted, §6.2 |
| heart_rate_hz, arterial_pulsatility, pulse_shape, pulse_harmonics, distensibility_per_mmhg | 1.1 Hz, 0.12, systolic + dicrotic wave, 4 harmonics, 0.002 / 0.03 / 0.3 per mmHg | [Neriyanuri2023]; fitted to [Koutsiaris2010, Zweifach1974, Hahn2004, Puyo2019], §6.3 |
| cell_free_layer_um | 0.8–3.1 µm | [Kim2007], §6.4 |
| profile_bluntness | 4 | [Tangelder1986], §6.4 |
| aggregation_cv | 0.42 | [Ding2025], §6.4 |
| clump_length_um, clump_amplitude | 25 µm, 0.25 | [Fedosov2010, Kim2007]; fitted to the measured flicker, §6.4 |
| spectrum | nine bands | absorption [Bosschaart2014]; weights fitted to measured contrast, §7.1 |
| double_pass | 2 | §7.1 |
| scatter_veil_length_um, scatter_blur_um_at_500 | 650 µm, 16 µm | fitted to contrast and edge steepness, §7.2 |
| psf_sigma_px | 0.9 | measured bound ≤ 1.2 px, §7.2 |
| defocus_um_per_um, focus_depth_um, focus_wander_um | 0.06, 250 µm, 25 µm | fitted to the fine vessels' edge steepness; assumed; §7.2 and §7.6 |
| illumination_p5_p95, background_dn | 0.64 (fitted to the measured 0.73), 2650 DN | measured, §7.3 and §7.5 |
| flicker_hz, flicker_amplitude, flicker_white | 8.77 Hz, 1.11 %, 0.12 % | measured (amplitude converted to linear light); white fitted, §7.3 |
| texture_amplitudes | fitted amplitudes of the measured bands | measured, §7.3 |
| exposure_s, fps | 12.838 ms, 32.19 | measured (manifest), §7.4 |
| gain_dn_per_e, read_noise_e, gamma, ceiling_dn, row_offset_dn | 1.344, 6.8 e⁻, 0.8, 4091, 3.56 DN | measured, §7.5 |
| motion_step_px, motion_bound_px, saccade_rate_hz, saccade_px, rotation_deg | 4.1 px, 60 px, 0.46 /s, 40–130 px, 0.05° | measured (step corrected for the estimator's under-read), §7.6 |
| tube_um_per_px | 6 µm/px | at this magnification or coarser vessels are drawn as tubes of red-cell density (assumed), §11.9 |
| glints | 4 | measured: a few eye-locked specular reflections per frame |

## 11. The whole anterior eye and its levels of detail

`synth/eye.py` grows the network of the whole anterior eye from where its blood enters and leaves, over the curved
globe, and lets a camera look at any region of it. Positions on the globe are given by the distance s from the
limbus along the surface, the clock angle (superior 0°, temporal 90°, inferior 180°, nasal 270°, a left eye seen
from in front) and the depth below the surface.

### 11.1 The globe and the cornea

Axial length 23.49 ± 1.35 mm, corneal radius 7.69 ± 0.28 mm and white-to-white diameter 11.80 ± 0.42 mm in a pooled
global sample [GaneshLin2023]; a cadaver corneal diameter of 11.7 × 10.7 mm [Kim2024Tillaux]. The anterior scleral
radius is §1.4's (12.5 mm). The scleral angle at the corneoscleral junction is 35–39° at a 15 mm chord, flatter
nasally [Ritzmann2018]. The model's limbus is the circle of radius 5.9 mm on the sphere (`limbus_radius_um`).

### 11.2 The rectus insertions (the spiral of Tillaux)

| muscle | AS-OCT in adults [Pihlblad2016] | cadaver, calipers [Kim2024Tillaux] | textbook |
|---|---|---|---|
| medial (nasal) | 5.7 ± 0.8 mm | 5.28 mm | 5.5 mm |
| inferior | 6.0 ± 0.6 mm | 5.72 mm | 6.5 mm |
| lateral (temporal) | 6.8 ± 0.7 mm | 6.40 mm | 6.9 mm |
| superior | 6.8 ± 0.6 mm | 6.78 mm | 7.7 mm |

The model uses the in vivo AS-OCT distances (`insertion_mm`) and the tendon widths of [Mistry2026] (superior 10.3,
lateral and medial 8.5, inferior 6.4 mm; `insertion_width_mm`); limbus-to-insertion distances in adult strabismus
patients are similar [Liu2011].

### 11.3 The anterior ciliary arteries and the episcleral circle

The textbook count is two anterior ciliary arteries per rectus and one on the lateral rectus [AshtonSmith1953];
measured counts differ: 2.25 ± 0.79 major arteries on the lateral and 2.05 ± 0.65 on the medial rectus
[Johnson2009], and among 742 muscles the medial rectus carried 1, 2 or ≥ 3 major bundles in 6.8, 46.4 and 46.7 %,
the lateral rectus in 41.2, 39.9 and 18.9 % [PanYang2020]. The model draws each eye's horizontal-rectus counts from
those measured shares and gives each vertical rectus two (`arteries_per_muscle`). No measured lumen diameter or
perforation distance was found; the arteries take 70–110 µm (assumed, wider than the 40–80 µm episcleral arteries
they feed), run forward in the episclera from the insertion and send a perforating branch through the sclera
halfway to the circle (an outflow to the ciliary body at an assumed 45 mmHg). They join the episcleral arterial
circle, which the model places 2 mm behind the limbus (deep arcades about 2–3 mm from the limbus [Kemenes2025]; the
circle has superficial and deep parts and is incomplete in places [MeyerWatson1987]: 15 % of its arcs are left out,
an assumed share).

### 11.4 The conjunctival arterial supply and the watershed

Anterior conjunctival arteries arise from the circle and run forward to the limbal arcades and palisades and back
into the bulbar conjunctiva; posterior conjunctival arteries come from the marginal and peripheral palpebral arcades
(about 3.0 mm from the upper lid margin [KakizakiJCM2025], 3–4 mm from the lower [Kakizaki2026]) through the
fornices; the two territories meet at a watershed [MeyerWatson1987] that a review places at about 4 mm from the
limbus (lid branches supply the conjunctiva from 4 mm on [Sun2022]). In the model, anterior arteries rise from the
circle every 1.8 mm (scaled by the quadrant's density) and posterior arteries enter from the fornix every 2.5 mm,
and both grow by space colonisation [Runions2007] over the conjunctival band at trunk scale (branches down to
20–28 µm, at least 0.4 mm between branch points, unbranched terminal branches cut to 1.5 mm, the patch level's
pull along the meridian (`coarse_radial_bias` 0.15 [Akagi2018]) and a persistence of heading of 0.6
(`coarse_growth_persistence`), both fitted so that a 2.5 mm field shows the spread of vessel orientations and the
junction density of the recordings (§7: orientation order 0.15, 33 junctions per mm²); with a stronger pull the
trunks crossed every field as parallel radial lines), so they meet and interleave around the watershed rather than
at a fixed line. Their paths get the meander and bends of section 5, and its calibre
variation along each vessel (`assign_calibres`, whose amplitude falls with calibre: trunks vary less in proportion
than small vessels [Kiani1993]). Each anterior artery also
sends a short branch forward to the limbal arcade, which drains through palisade hairpins 1.5–2.1 mm long
[Kemenes2025, Haagdorens2017] into a limbal venous ring (35 µm); the forward branches' ends are joined around the
limbus into the arterial arcade (18–28 µm) [VanBuskirk1989, Kemenes2025], so both rings run round the cornea.

### 11.5 The venous drainage

Episcleral veins drain away from the limbus [Meyer1988] to the muscular and vortex veins (4–8 vortex veins per
eye [Kutoglu2005]; 8.0 ± 2.1 ampullae on widefield angiography [Verma2020]); conjunctival veins drain to the
palpebral veins. Aqueous enters episcleral veins at the limbus through aqueous veins; no human count was found, and
nasal outflow dominated in live aqueous angiography [Huang2017] but not consistently in enucleated eyes
[Saraswathy2016]. The model: episcleral veins (0.6 per mm of limbus, 50–100 µm) leave the limbal venous ring and
run away from the limbus past the insertions in long, gently bending runs with the patch level's course: a heading
drawn about the meridian with a spread of 0.8 rad (at most 1.2 rad, so none spirals round the globe) and a random
curvature of SD 0.0008 rad/µm pulled back towards that initial heading (§5), with side branches leaving at 25–50°; conjunctival venular
trees, half of whose growth follows the arteries so that veins accompany them, drain to outlets at the fornix and
into the limbal ring. Aqueous inflow is not modelled.

### 11.6 Pressures and flow

Ophthalmic artery systolic pressure is 0.68 ± 0.04 times brachial [StraussKedra1987] (93/59 against 136/87 mmHg
[daSilvaKrieglstein1980]); mean central retinal artery pressure is about 0.73 of mean arterial pressure seated
[Kostic2021]; intraocular pressure about 14 mmHg [Pakravan2017, Kawase2008]; episcleral venous pressure 7.6 ± 1.3
[Zeimer1983] to 9.1–9.5 mmHg [Selbach2005b]. No anterior ciliary artery pressure was found. The model sets the
inlets of the anterior ciliary arteries at 55 mmHg (0.6 of a mean brachial pressure of 93 mmHg: they lie downstream
of the ophthalmic artery through the muscular arteries; fitted so that the conjunctival red-cell velocities fall
within the cited ranges once the fine vessels and the capillary plexus carry the flow - at 65 mmHg, 0.7 of
brachial, the arterioles ran at about twice the cited speed), the palpebral inlets of the posterior conjunctival
arteries at 50 mmHg (assumed), and every venous outlet at 9 mmHg; the network's flows and velocities then follow
from the anatomy. Each anterior conjunctival artery leaves the circle at the calibre of these arteries (25–40 µm),
and the conjunctival trunks are held below 40 µm (arteries) and 48 µm (veins), the widest sharp conjunctival
vessels of the recordings (`coarse_artery_max_um`, `coarse_vein_max_um`; reported ranges reach 61 and 75 µm
[Khansari2017], but wider trunks at conjunctival depth rendered far darker than any real vessel). Episcleral arteries open into episcleral veins through 0.75 arteriovenous anastomoses per mm² (§2.4). Where
the fine level is grown (the whole conjunctiva by default, §11.9) the conjunctival trunks drain through the fine
trees and the capillary plexus; elsewhere the fine vessels each coarse artery feeds are lumped into one equivalent
resistance every 0.5 mm, joining it to the nearest conjunctival vein, and every trunk tip drains through such a bed
of its own (`bed_conductance_scale`, fitted so that the coarse conjunctival vessels carry the cited velocities). Result (seed 0, the whole eye
with its fine level): median red-cell speed 0.93 mm/s in arterioles (interquartile 0.51–1.51), 0.88 in capillaries
(0.40–1.70) and 0.53 in venules (0.23–1.01), against 0.63–0.70 mm/s in conjunctival arterioles and 0.54–0.59 in
venules [Khansari2016, Khansari2017] and 0.29–1.5 mm/s in capillaries [Moka2020, Hahn2004]; 1.6 mm/s in the
conjunctival arterial and 0.8 mm/s in the venous trunks; median capillary pressure 26 mmHg (§6.2); episcleral vein
flow 0.06–0.23 µl/min (interquartile) against 0.40 ± 0.22 µl/min [Kim2024]; pulsatility index 0.54 in 6–12 µm
arterioles, 0.28 in capillaries, 0.10 in venules of 15 µm and more and 0.01 in episcleral veins.

### 11.7 Lids and fornices

In 380 adults the upper lid margin lies 3.5 mm above and the lower 5.2 mm below the pupil centre, the fissure is
8.7 mm high and 25.6 mm wide [Rana2023]: the upper lid covers about 1.8 mm of the superior cornea and the lower lid
sits at the inferior limbus, so the exposed bulbar conjunctiva is mostly nasal and temporal. Fornix depths are
15.3–16.1 mm (upper) and 10.1–10.9 mm (lower) across four populations [Jutley2016, Khan2014, Tan2025, Bulut2024],
from a reference the abstracts do not state; canthal fornix depths 3.0 mm medially and 5.2 mm laterally
[Kawakita2009]. The model ends the bulbar conjunctiva at 15 mm superiorly, 10.5 mm inferiorly and (assumed) 12 mm
temporally and 8 mm nasally, and marks the palpebral fissure (`fissure_half_width_mm`, `upper_lid_mm`,
`lower_lid_mm`); the lids that cover the eye at a given gaze are drawn by `synth.gaze` (§12).

### 11.8 Regional variation

Deep vessel density is about 25 % nasally and temporally against about 21 % superiorly and inferiorly [Akagi2018];
superficial nasal and temporal densities are similar [Schuerch2020]; palisades show superiorly and inferiorly but
rarely in the horizontal meridian [Kemenes2025]; arteries are sparse over the lateral rectus and veins concentrate
in the vertical meridian [Meyer1988]; the sclera is thicker [FernandezVigo2022] and flatter [Choi2014] nasally. All
slit-lamp microvessel studies image the temporal conjunctiva. The model scales the density of its anterior
conjunctival arteries and of the coarse growth by quadrant (`quadrant_density`: 1.08 nasal and temporal, 0.92
superior and inferior, interpolated around the clock).

### 11.9 Levels of detail

The eye is one network, generated and solved once from its seed; the levels of detail are selections of it, so a
vessel has the same path and the same flow at every magnification and in every view.

- **Coarse** (§11.3–11.6): the anterior ciliary arteries, the episcleral circle and veins, the conjunctival trunks
  and the limbal rings, about 8 500 segments.
- **Fine trees** (per tile of about 1 mm × 1 mm on the globe): the conjunctival arteriolar and venular trees and
  their anastomoses (§4–5), grown from roots placed along the coarse conjunctival vessels that cross the tile
  (leaving them at 30–85°) and, as at the patch level, from communicating vessels rising obliquely from the
  episcleral vessels beneath it (`communicating_per_mm2`, §2.4), from a seed made of the eye's seed and the tile's
  position only; about 18 000 segments over the whole conjunctiva.
- **Capillary plexus**: one mesh over the whole conjunctiva, from the limbus to the fornices - the Voronoi
  tessellation, on the sphere, of a jittered lattice of sites laid tile by tile (§5) - into which every terminal
  arteriole and post-capillary venule opens (one or two links each), and every coarse trunk that ends there (three
  links); about 31 000 segments. Earlier versions grew a separate capillary mesh inside each tile, closed at the
  tile's edge, and solved each tile on its own from the coarse pressures at its roots: the mesh shrank away from
  every tile border when its dangling ridges were pruned, which drew the capillaries as isolated clusters with
  empty gutters between them (a construction artefact, not anatomy), and trees hanging from a single root carried
  no flow (about a fifth of the fine vessel length). The coarse level's lumped beds (§11.6) now stand only for the
  part of the conjunctiva without fine vessels (`Eye(fine_region=...)` grows the fine level over part of the eye,
  for tests); over the whole eye there are none.
- **The flow** is solved once over the whole network (about 58 000 segments) from the true inlets and outlets
  (§11.6); the haematocrit iteration stops after 100 rounds, when the flows of all but a few near-stagnant loops have
  settled (they differ from 400 rounds by 0.04 % median, 0.7 % at the 99th percentile), and the final pass conserves
  flow and red cells exactly. Seed 0: 98.7 % of the coarse, 98.7 % of the fine-tree and 98.2 % of the capillary
  length carries red cells moving at 5 µm/s or more; the rest are dead ends and plasma-only capillaries (phase
  separation, §6.3). Generating and solving a whole eye takes about 2–3 minutes and 0.4 GB.
- **The view** (the camera's field and a margin): every vessel of the eye that falls in it, cut at the field's edge,
  each piece carrying the flow, haematocrit and pulse of the edge it came from (tested at 1, 3.26, 6.5 and 12 µm/px:
  every piece's flow equals its edge's, and a view draws every vessel in its field). The lumped beds (§11.6, present
  only in an eye grown with `Eye(fine_region=...)`) stay in the flow solve but are not vessels: views, their truth
  (`graph.json`, the masks) and the browser page leave them out. At low magnification only the
  coarse level may be drawn; blood that leaves a drawn vessel through one the view does not draw takes its red
  cells with it, and where blood joins through one, its red cells enter as at an inflow boundary (`synth.cells`), so
  cells neither pile up nor thin out in blocks along a vessel. A vessel without red-cell flux never takes cells.
- **Rendering by magnification**: below `tube_um_per_px` (6 µm/px) every red cell is transported and drawn; at
  coarser magnification each vessel is a tube of its red-cell density, modulated along its length by the axial
  structure of the red cells (aggregates and plasma gaps) carried downstream at the vessel's pulsatile red-cell
  speed, so flow shows at every magnification at which that structure spans a pixel.
- **Interactive view**: `synth.eyeview.write_viewer` writes a page that draws the whole eye in the browser as the
  camera sees it (Beer-Lambert absorption of every vessel over the scleral texture, its calibre along its length,
  the scattering veil and blur by depth, the red cells as a texture moving at each vessel's pulsatile speed), at
  any magnification from the whole eye to single capillaries.
- **Pose**: the camera's position on the globe (`Pose`: s and clock angle), magnification (`um_per_px`), field size,
  focus (`focus_depth_um`) and tilt (`camera_tilt_deg`) are explicit parameters, so the same eye can be recorded
  from any position.

## 12. The patient, the eye in the head, and mapping

`synth.gaze` and `synth.mapping` simulate imaging a patient's eye the way the vasculature would be mapped in the
clinic: the patient rests on a chin rest, the camera is fixed, and the patient is asked to look at a sequence of
targets so that different parts of the conjunctiva come under the camera.

- **A patient from a seed** (`patient`): the globe radius (12.2 ± 0.6 mm, within 11.2–13.1 mm [Jesus2017,
  Choi2014]), limbal radius (white to white 11.80 ± 0.42 mm [GaneshLin2023]), the palpebral fissure (25.6 mm wide,
  lid margins about 4.1 mm above and 5.9 mm below the corneal centre [Rana2023], spread assumed), the heart rate
  (cardiac period 909 ± 86 ms [Neriyanuri2023]) and the discharge haematocrit (0.40 ± 0.025, assumed) are drawn from
  the seed, and the whole network is grown from the same seed: the same seed is always the same eye (tested).
- **Gaze**: the eye turns about the globe's centre (which stands for its centre of rotation) by Listing's law - from
  the primary position about the axis perpendicular to the primary and the new direction - so a gaze carries no
  torsion of its own. The camera's axis meets the globe at a fixed head direction (`SlitLamp.aim`); for a gaze, the
  field's centre on the eye and the roll of the image about it follow from the rotation.
- **Fixational movements** (in every frame of a burst): drift as a random walk of mean speed 52′/s (head fixed
  [Cherici2012]); microsaccades at 1.3 per second of 20′ ± 7′ [Cherici2012] (1–2 per second [MartinezConde2004]),
  directed back towards the fixation target (they correct the drift); the landing point of each fixation scattered
  by 0.25° about its target (assumed); the head on the chin rest moving as a mean-reverting random walk with an SD of
  176 µm and a 2 s correlation time (SD over 5 s on a chin rest [Felder2016]; time assumed). 1′ of rotation is about
  3.5 µm at the conjunctiva, so a microsaccade moves the image by some 20 pixels at 3.26 µm/px.
- **Saccades** between targets take the main-sequence duration, 2.7 ms per degree [Baloh1975] plus 22 ms (assumed),
  and the recording starts once the eye has settled (0.2 s, assumed).
- **The lids**: the upper lid follows the eye's vertical rotation, one to one downwards and somewhat less upwards
  [BeckerFuchs1988] (0.8, assumed), the lower lid less (0.3, assumed); the fissure keeps its shape across the face.
  Where a lid covers the field the camera sees skin (brighter than sclera, with its own texture) and, at its edge,
  the dark lid margin and lashes over about 0.6 mm (assumed). Blinks come at 14 per minute (primary gaze 8.0–21.0
  per minute [Doughty2001]) and close the lids over 0.3 s (assumed).
- **A burst** (`SlitLamp.record`): the view assembled at the field's pose in frame 0, the frames rendered with the
  image motion that the fixational movements and the head produce (`camera.Motion`), the lids drawn over the frames
  they cover, and the cardiac phase carried on from burst to burst; its truth is every frame's gaze, field pose,
  lid cover and blink state, on top of the usual truth of the burst.
- **Mapping** (`synth.mapping`, a proof of concept): `plan` turns a region of the conjunctiva into fixation targets
  whose fields tile it (spaced by the field's shorter side, since the image lies at any roll on the globe);
  `Mapper.add` builds the map from each burst's frames, the commanded gaze and the camera's fixed geometry only
  (never the truth). It drops frames a lid or blink covers (brighter than the sclera of the earlier bursts, or
  without the vessels' fine structure), registers the rest by phase correlation, flat-fields their mean by a robust
  quadratic fit of the illumination, and measures vesselness and red-cell flicker. It predicts the field's place
  on the globe from the commanded gaze (Listing's law through the fixed camera), moved by the head's drift on the
  chin rest as the last accepted registrations measured it, then registers the burst's vesselness against each earlier burst
  whose field overlaps (one at a time, not the blended map, which one misplaced burst would ghost) by masked
  normalised cross-correlation [Padfield2012], the best match deciding, a large correction needing a correlation of
  0.75 and a small one (under 120 µm) 0.5. Fields are matched through the camera's own projection (orthographic
  along its axis, `synth.camera`) on the sphere of the vessels' typical depth (150 µm), which keeps the parallax
  between overlapping fields small; matching surface points instead biased every pairwise measurement by a few tens
  of micrometres. After every burst all bursts' places are re-solved at once from every trusted pairwise
  measurement (weighted least squares on the chart), so the drift of a chain of registrations is spread out, and
  the bursts are added into a chart of the globe's surface (polar and clock angle about the corneal axis, 4 µm per
  pixel), weighted towards each field's centre. The vessel map keeps the vesselness above 0.6 of Otsu's threshold. `Mapper.evaluate` scores the
  map against the eye's network: the share of the region covered, the error of every burst's placement (absolute,
  and relative to the map's common offset - a map built without an absolute reference sits on the globe within
  the first bursts' error), and, after that common offset, the precision and recall of the vessel map's
  centrelines against the true vessels' centrelines within 8 µm.
- **A demonstration** (patient 7, the camera aimed 40° temporal of the primary gaze, 22 fixations planned over the
  temporal conjunctiva from 72° to 108° and 1.4–5.0 mm from the limbus with 35 % overlap, 16 frames each at
  3.26 µm/px; seven fixations partly under a lid): the map covers 77 % of the region; every burst is placed within
  13 µm (median; 32 µm at the 90th percentile) of its true place relative to the map's common offset of 90 µm (the
  head's position on the chin rest at the first burst); 96 % of the mapped vessel centreline lies on a true
  vessel, and the map finds 52 % of the centreline of the vessels of 15 µm and more, 31 % of the conjunctival
  vessels of 8 µm and more, and 26 % of all vessels including the capillaries, most of which are too faint at this
  magnification to stand out from a burst of 16 frames. These placement figures were computed before
  `Mapper.evaluate` counted a chart column for the arc it spans on the globe (`um_per_column`): they took every
  column as 4 µm, which overstates the along-limbus part of each error by up to about a third at the limbal edge
  of this region, so the figures above are upper bounds.

## References

Each DOI was checked against PubMed, Crossref or DataCite (title, authors, year). Most numbers were read from the
full text; some only from the abstract (among them Zhang2013, FernandezVigo2021 and 2022, Selbach2005, Johnson2009,
Koutsiaris2007, 2010 and 2015, Moka2020, Zweifach1974, Williams1990, Hahn2004, Zeimer1983, Sit2011, Tangelder1986,
Kim2007, Kim2024, MacKenzie2016). Two formulas are used as written by later papers that reproduce them: the Pries
viscosity laws as written by SecombPries2013, and the 1992 Fåhræus relation as written by Fedosov2010; the phase-
separation parameters of PriesSecomb2005 as reproduced by later network studies.

- [Akagi2018] Akagi T, Uji A, Huang AS, et al. Conjunctival and intrascleral vasculatures assessed using anterior segment optical coherence tomography angiography in normal eyes. Am J Ophthalmol 2018;196:1-9. doi:10.1016/j.ajo.2018.08.009.
- [Arora2017] Arora N, McLaren JW, Hodge DO, Sit AJ. Effect of Body Position on Epsicleral Venous Pressure in Healthy Subjects. Invest Ophthalmol Vis Sci 2017;58:5151-5156. doi:10.1167/iovs.17-22154.
- [Asanad2021] Asanad S, Park A, Pottenburgh J, et al. Erythrocyte-mediated angiography: quantifying absolute episcleral blood flow in humans. Ophthalmology 2021;128:799-801. doi:10.1016/j.ophtha.2020.09.010.
- [AshtonSmith1953] Ashton N, Smith R. Anatomical study of Schlemm's canal and aqueous veins by means of neoprene casts. III. Arterial relations of Schlemm's canal. Br J Ophthalmol 1953;37:577-586. doi:10.1136/bjo.37.10.577.
- [Atchison2005] Atchison DA, Pritchard N, Schmid KL, et al. Shape of the retinal surface in emmetropia and myopia. Invest Ophthalmol Vis Sci 2005;46:2698-2707. doi:10.1167/iovs.04-1506.
- [Baloh1975] Baloh RW, Sills AW, Kumley WE, Honrubia V. Quantitative measurement of saccade amplitude, duration, and velocity. Neurology 1975;25:1065-1070. doi:10.1212/wnl.25.11.1065.
- [BeckerFuchs1988] Becker W, Fuchs AF. Lid-eye coordination during vertical gaze changes in man and monkey. J Neurophysiol 1988;60:1227-1252. doi:10.1152/jn.1988.60.4.1227.
- [Bizheva2011] Bizheva K, Hutchings N, Sorbara L, Moayed AA, Simpson T. In vivo volumetric imaging of the human corneo-scleral limbus with spectral domain OCT. Biomed Opt Express 2011;2:1794-1802. doi:10.1364/BOE.2.001794.
- [Bosschaart2014] Bosschaart N, Edelman GJ, Aalders MC, van Leeuwen TG, Faber DJ. A literature review and novel theoretical approach on the optical properties of whole blood. Lasers Med Sci 2014;29:453-79. doi:10.1007/s10103-013-1446-7.
- [Bulut2024] Bulut O, Kaplan A, Furundaoturan O, Kose T, Barut Selver O. Identification of normal fornix depth dataset generated with a validated fornix meter in healthy Turkish population. Ocul Immunol Inflamm 2024;32:713-717. doi:10.1080/09273948.2023.2199055.
- [Cherici2012] Cherici C, Kuang X, Poletti M, Rucci M. Precision of sustained fixation in trained and untrained observers. J Vis 2012;12(6):31. doi:10.1167/12.6.31.
- [Cheung2001] Cheung AT, Ramanujam S, Greer DA, Kumagai LF, Aoki TT. Microvascular abnormalities in the bulbar conjunctiva of patients with type 2 diabetes mellitus. Endocr Pract 2001;7:358-63. doi:10.4158/EP.7.5.358.
- [CheungCY2011] Cheung CY, Zheng Y, Hsu W, et al. Retinal vascular tortuosity, blood pressure, and cardiovascular risk factors. Ophthalmology 2011;118:812-818. doi:10.1016/j.ophtha.2010.08.045.
- [Choi2014] Choi HJ, Lee SM, Lee JY, et al. Measurement of anterior scleral curvature using anterior segment OCT. Optom Vis Sci 2014;91:793-802. doi:10.1097/OPX.0000000000000298.
- [daSilvaKrieglstein1980] da Silva FA, Krieglstein GK. [Measurement of ophthalmic blood pressure with the Mikuni dynamometer]. Klin Monbl Augenheilkd 1980;176:764-772. doi:10.1055/s-2008-1057550.
- [Ding2025] Ding Y, Metha A, Bedggood P. Red Blood Cell Distribution and Volume Changes at Capillary Junctions in the Living Human Retina. Invest Ophthalmol Vis Sci 2025;66(14):65. doi:10.1167/iovs.66.14.65.
- [Doughty2001] Doughty MJ. Consideration of three types of spontaneous eyeblink activity in normal humans: during reading and video display terminal use, in primary gaze, and while in conversation. Optom Vis Sci 2001;78:712-725. doi:10.1097/00006324-200110000-00011.
- [Fedosov2010] Fedosov DA, Caswell B, Popel AS, Karniadakis GE. Blood flow and cell-free layer in microvessels. Microcirculation 2010;17:615-28. doi:10.1111/j.1549-8719.2010.00056.x.
- [Felder2016] Felder AE, Mercurio C, Wanek J, Ansari R, Shahidi M. Automated Real-Time Conjunctival Microvasculature Image Stabilization. IEEE Trans Med Imaging 2016;35:1670-5. doi:10.1109/TMI.2016.2522918.
- [FengSimpson2008] Feng Y, Simpson TL. Corneal, limbal, and conjunctival epithelial thickness from optical coherence tomography. Optom Vis Sci 2008;85:E880-E883. doi:10.1097/OPX.0b013e318185272d.
- [FernandezVigo2021] Fernández-Vigo JI, Shi H, Burgos-Blasco B, et al. Impact of age, sex and refractive error on conjunctival and Tenon's capsule thickness dimensions by swept-source optical coherence tomography in a large population. Int Ophthalmol 2021;41:3687-3698. doi:10.1007/s10792-021-01928-5.
- [FernandezVigo2022] Fernández-Vigo JI, Shi H, Burgos-Blasco B, et al. Anterior scleral thickness dimensions by swept-source optical coherence tomography. Clin Exp Optom 2022;105:13-19. doi:10.1080/08164622.2021.1924629.
- [FunkRohen1996] Funk RH, Rohen JW. Scanning electron microscopic study of episcleral arteriovenous anastomoses in the owl and cynomolgus monkey. Curr Eye Res 1996;15:321-327. doi:10.3109/02713689609007627.
- [GaneshLin2023] Ganesh D, Lin SR. Global metrics on ocular biometry: representative averages and standard deviations across ten countries from four continents. Eye (Lond) 2023;37:511-515. doi:10.1038/s41433-022-01961-3.
- [Garg2022] Garg G, Venkatesh P, Chawla R, et al. Normative data of retinal arteriolar and venular calibre measurements determined using confocal scanning laser ophthalmoscopy system. Indian J Ophthalmol 2022;70:1657-1663. doi:10.4103/ijo.IJO_2162_21.
- [Grippo2025] Grippo TM, LaMarche K, Davreux AM, Stauffer J, Stacey S, et al. Thermal Tissue Modeling of the Ciliary Body With Continuous Wave Transscleral Cyclophotocoagulation and Transscleral Laser Therapy Using MicroPulse Technology. Transl Vis Sci Technol 2025;14(11):32. doi:10.1167/tvst.14.11.32.
- [Haagdorens2017] Haagdorens M, Behaegel J, Rozema J, et al. A method for quantifying limbal stem cell niches using OCT imaging. Br J Ophthalmol 2017;101:1250-1255. doi:10.1136/bjophthalmol-2016-309549.
- [Hahn2004] Hahn M, Jünger M, Shore AC. The effect of prostaglandin E1 on nailfold capillary blood pressure and red blood cell velocity in humans. Clin Hemorheol Microcirc 2004;31:227-34. No DOI in PubMed or Crossref search (PMID 15322328; abstract).
- [Howlett2014] Howlett J, Vahdani K, Rossiter J. Bulbar conjunctival and Tenon's layer thickness measurement using optical coherence tomography. J Curr Glaucoma Pract 2014;8:63-66. doi:10.5005/jp-journals-10008-1163.
- [Hu2018] Hu L, Shi C, Jiang H, et al. Factors affecting microvascular responses in the bulbar conjunctiva in habitual contact lens wearers. Invest Ophthalmol Vis Sci 2018;59:4108-4114. doi:10.1167/iovs.18-24216.
- [Huang2017] Huang AS, Li M, Yang D, Wang H, Wang N, Weinreb RN. Aqueous angiography in living nonhuman primates shows segmental, pulsatile, and dynamic angiographic aqueous humor outflow. Ophthalmology 2017;124:793-803. doi:10.1016/j.ophtha.2017.01.030.
- [Jacques2013] Jacques SL. Optical properties of biological tissues: a review. Phys Med Biol 2013;58:R37-R61. doi:10.1088/0031-9155/58/11/R37.
- [Jesus2017] Jesus DA, Kedzia R, Iskander DR. Precise measurement of scleral radius using anterior eye profilometry. Cont Lens Anterior Eye 2017;40:47-52. doi:10.1016/j.clae.2016.11.003.
- [Jiang2014] Jiang H, Zhong J, DeBuc DC, Tao A, Xu Z, Lam BL, et al. Functional slit lamp biomicroscopy for imaging bulbar conjunctival microvasculature in contact lens wearers. Microvasc Res 2014;92:62-71. doi:10.1016/j.mvr.2014.01.005.
- [Johnson2009] Johnson MS, Christiansen SP, Rath PP, et al. Anterior ciliary circulation from the horizontal rectus muscles. Strabismus 2009;17:45-48. doi:10.1080/09273970802678180.
- [Jutley2016] Jutley G, Carpenter D, Hau S, et al. Upper and lower conjunctival fornix depth in healthy white caucasian eyes: a method of objective assessment. Eye (Lond) 2016;30:1351-1358. doi:10.1038/eye.2016.128.
- [Kakizaki2026] Kakizaki H, Edward DP, Takahashi Y. Position of lower marginal arterial arcade in relation to lower tarsal plate. Ophthalmic Plast Reconstr Surg 2026;42:483-485. doi:10.1097/IOP.0000000000003055.
- [KakizakiJCM2025] Kakizaki H, Edward DP, Takahashi Y. Marginal artery position on upper eyelid. J Cosmet Med 2025;9:79-81. doi:10.25056/jcm.2025.9.2.79.
- [KassabFung1995] Kassab GS, Fung YC. The pattern of coronary arteriolar bifurcations and the uniform shear hypothesis. Ann Biomed Eng 1995;23:13-20. doi:10.1007/BF02368296.
- [Kawakita2009] Kawakita T, Kawashima M, Murat D, Tsubota K, Shimazaki J. Measurement of fornix depth and area: a novel method of determining the severity of fornix shortening. Eye (Lond) 2009;23:1115-1119. doi:10.1038/eye.2008.205.
- [Kawase2008] Kawase K, Tomidokoro A, Araie M, Iwase A, Yamamoto T; Tajimi Study Group. Ocular and systemic factors related to intraocular pressure in Japanese adults: the Tajimi study. Br J Ophthalmol 2008;92:1175-1179. doi:10.1136/bjo.2007.128819.
- [Kemenes2025] Kemenes G, Rebenku I, Bajdik B, et al. Correlation of multimodal clinical imaging with the whole-slide- and superresolution-based immunohistological structure of the corneal limbal stroma. Invest Ophthalmol Vis Sci 2025;66(9):54. doi:10.1167/iovs.66.9.54.
- [Khan2014] Khan IJ, Ghauri AJ, Hodson J, et al. Defining the limits of normal conjunctival fornix anatomy in a healthy South Asian population. Ophthalmology 2014;121:492-497. doi:10.1016/j.ophtha.2013.09.033.
- [Khansari2016] Khansari MM, Wanek J, Felder AE, Camardo N, Shahidi M. Automated Assessment of Hemodynamics in the Conjunctival Microvasculature Network. IEEE Trans Med Imaging 2016;35:605-11. doi:10.1109/TMI.2015.2486619.
- [Khansari2017] Khansari MM, Wanek J, Tan M, Joslin CE, Kresovich JK, Camardo N, Blair NP, Shahidi M. Assessment of Conjunctival Microvascular Hemodynamics in Stages of Diabetic Microvasculopathy. Sci Rep 2017;7:45916. doi:10.1038/srep45916.
- [Khansari2019] Khansari MM, Garvey SL, Farzad S, Shi Y, Shahidi M. Relationship between retinal vessel tortuosity and oxygenation in sickle cell retinopathy. Int J Retina Vitreous 2019;5:47. doi:10.1186/s40942-019-0198-3.
- [Kiani1993] Kiani MF, Cokelet GR, Sarelius IH. Effect of diameter variability along a microvessel segment on pressure drop. Microvasc Res 1993;45:219-232. doi:10.1006/mvre.1993.1020.
- [Kim2007] Kim S, Kong RL, Popel AS, Intaglietta M, Johnson PC. Temporal and spatial variations of cell-free layer width in arterioles. Am J Physiol Heart Circ Physiol 2007;293:H1526-35. doi:10.1152/ajpheart.01090.2006.
- [Kim2024] Kim S, Chen V, Pottenburgh J, et al. Netarsudil 0.02% alters episcleral venous flowrates: a clinical trial using erythrocyte-mediated angiography. Ophthalmol Sci 2024;4:100533. doi:10.1016/j.xops.2024.100533.
- [Kim2024Tillaux] Kim EJ, Rana VK, Barton A, Khatri S, et al. The spiral of Tillaux: dead on accurate; a cadaveric study. Strabismus 2024;32:85-90. doi:10.1080/09273972.2024.2346548.
- [Klitzman1979] Klitzman B, Duling BR. Microvascular hematocrit and red cell flow in resting and contracting striated muscle. Am J Physiol 1979;237:H481-90. doi:10.1152/ajpheart.1979.237.4.H481.
- [Kostic2021] Kostic M, Gordon P, Monsalve P, et al. Non-invasive assessment of central retinal artery pressure: age and posture-dependent changes. Curr Eye Res 2021;46:135-139. doi:10.1080/02713683.2020.1772833.
- [Koutsiaris2007] Koutsiaris AG, Tachmitzi SV, Batis N, Kotoula MG, Karabatsas CH, Tsironi E, Chatzoulis DZ. Volume flow and wall shear stress quantification in the human conjunctival capillaries and post-capillary venules in vivo. Biorheology 2007;44:375-86. doi:10.1177/0006355X2007044005006008 (no DOI in PubMed; DOI verified via Crossref; abstract).
- [Koutsiaris2010] Koutsiaris AG, Tachmitzi SV, Papavasileiou P, Batis N, Kotoula MG, Giannoukas AD, Tsironi E. Blood velocity pulse quantification in the human conjunctival pre-capillary arterioles. Microvasc Res 2010;80:202-8. doi:10.1016/j.mvr.2010.05.001.
- [Koutsiaris2015] Koutsiaris AG. Correlation of axial blood velocity to venular and arteriolar diameter in the human eye in vivo. Clin Hemorheol Microcirc 2015;61:429-38. doi:10.3233/CH-141888.
- [Kumar2013] Kumar DA, Agarwal A, Karnathi S, et al. Anterior segment optical coherence tomography for imaging the sub-Tenon space. Ophthalmic Res 2013;50:231-234. doi:10.1159/000354381.
- [Kutoglu2005] Kutoglu T, Yalcin B, Kocabiyik N, Ozan H. Vortex veins: anatomic investigations on human eyes. Clin Anat 2005;18:269-273. doi:10.1002/ca.20092.
- [Kvernebo2022] Kvernebo AK, Moe MC, Wikslund LK, Capone L, et al. Human ocular surface microcirculation quantified by in vivo computer assisted video microscopy and diffuse reflectance spectroscopy. Exp Eye Res 2022;224:109232. doi:10.1016/j.exer.2022.109232.
- [Li2011] Li P, An L, Reif R, Shen TT, Johnstone M, Wang RK. In vivo microstructural and microvascular imaging of the human corneo-scleral limbus using optical coherence tomography. Biomed Opt Express 2011;2:3109-3118. doi:10.1364/BOE.2.003109.
- [Liu2011] Liu X, Wang F, Xiao Y, Ye X, et al. Measurement of the limbus-insertion distance in adult strabismus patients with anterior segment optical coherence tomography. Invest Ophthalmol Vis Sci 2011;52:8370-8373. doi:10.1167/iovs.11-7752.
- [Liu2019] Liu Z, Wang H, Jiang H, et al. Quantitative analysis of conjunctival microvasculature imaged using optical coherence tomography angiography. Eye Vis (Lond) 2019;6:5. doi:10.1186/s40662-019-0130-9.
- [Luo2017] Luo T, Gast TJ, Vermeer TJ, Burns SA. Retinal vascular branching in healthy and diabetic subjects. Invest Ophthalmol Vis Sci 2017;58:2685-2694. doi:10.1167/iovs.17-21653.
- [MacKenzie2016] MacKenzie LE, Choudhary TR, McNaught AI, Harvey AR. In vivo oximetry of human bulbar conjunctival and episcleral microvasculature using snapshot multispectral imaging. Exp Eye Res 2016;149:48-58. doi:10.1016/j.exer.2016.06.008.
- [MartinezConde2004] Martinez-Conde S, Macknik SL, Hubel DH. The role of fixational eye movements in visual perception. Nat Rev Neurosci 2004;5:229-240. doi:10.1038/nrn1348.
- [MayrovitzRoy1983] Mayrovitz HN, Roy J. Microvascular blood flow: evidence indicating a cubic dependence on arteriolar diameter. Am J Physiol 1983;245:H1031-H1038. doi:10.1152/ajpheart.1983.245.6.H1031.
- [Meyer1988] Meyer PA. Patterns of blood flow in episcleral vessels studied by low-dose fluorescein videoangiography. Eye (Lond) 1988;2(Pt 5):533-546. doi:10.1038/eye.1988.104.
- [MeyerWatson1987] Meyer PA, Watson PG. Low dose fluorescein angiography of the conjunctiva and episclera. Br J Ophthalmol 1987;71:2-10. doi:10.1136/bjo.71.1.2.
- [Mistry2026] Mistry V, Ali N, Katti K, et al. Morphometric variations of extraocular muscles in a Caucasian cohort: a pilot study on the surgical implications of ethnic diversity. Cureus 2026;18:e104472. doi:10.7759/cureus.104472.
- [Moka2020] Moka S, Koutsiaris AG, Garas A, Messinis I, Tachmitzi SV, Giannoukas A, Tsironi EE. Blood flow velocity comparison in the eye capillaries and postcapillary venules between normal pregnant and non-pregnant women. Microvasc Res 2020;127:103926. doi:10.1016/j.mvr.2019.103926.
- [MorrisonVanBuskirk1983] Morrison JC, Van Buskirk EM. Anterior collateral circulation in the primate eye. Ophthalmology 1983;90:707-715. doi:10.1016/s0161-6420(83)34501-2.
- [Murray1926a] Murray CD. The physiological principle of minimum work: I. The vascular system and the cost of blood volume. Proc Natl Acad Sci U S A 1926;12:207-214. doi:10.1073/pnas.12.3.207.
- [Naim2026] Mohamed Naim A, Iroshan Kondarage A, Liyanaarachchi R, Jayasinghe S, De Silva AC. Topology-preserving, diameter-specific framework for conjunctival vessel segmentation and tortuosity analysis in diabetes. Sci Rep 2026;16:20675. doi:10.1038/s41598-026-48680-3.
- [NelloreHarris2002] Nellore K, Harris NR. L-arginine and antineutrophil serum enable venular control of capillary perfusion in hypercholesterolemic rats. Microcirculation 2002;9:477-485. doi:10.1038/sj.mn.7800162.
- [Nemati1996] Nemati B, Rylander HG III, Welch AJ. Optical properties of conjunctiva, sclera, and the ciliary body and their consequences for transscleral cyclophotocoagulation. Appl Opt 1996;35:3321-7. doi:10.1364/AO.35.003321.
- [Neriyanuri2023] Neriyanuri S, Bedggood P, Symons RCA, Metha AB. Flow Heterogeneity and Factors Contributing to the Variability in Retinal Capillary Blood Flow. Invest Ophthalmol Vis Sci 2023;64(10):15. doi:10.1167/iovs.64.10.15.
- [Olsen1998] Olsen TW, Aaberg SY, Geroski DH, Edelhauser HF. Human sclera: thickness and surface area. Am J Ophthalmol 1998;125:237-241. doi:10.1016/s0002-9394(99)80096-8.
- [Owen2008] Owen CG, Newsom RSB, Rudnicka AR, et al. Diabetes and the tortuosity of vessels of the bulbar conjunctiva. Ophthalmology 2008;115:e27-e32. doi:10.1016/j.ophtha.2008.02.009.
- [Padfield2012] Padfield D. Masked object registration in the Fourier domain. IEEE Trans Image Process 2012;21:2706-2718. doi:10.1109/TIP.2011.2181402.
- [Pakravan2017] Pakravan M, Javadi MA, Yazdani S, et al. Distribution of intraocular pressure, central corneal thickness and vertical cup-to-disc ratio in a healthy Iranian population: the Yazd Eye Study. Acta Ophthalmol 2017;95:e144-e151. doi:10.1111/aos.13231.
- [PanYang2020] Pan MH, Yang M. [Morphological observation on the anterior ciliary vessels in the horizontal rectus muscles of patients with concomitant horizontal strabismus]. Zhonghua Yan Ke Za Zhi 2020;56:197-204. doi:10.3760/cma.j.issn.0412-4081.2020.03.007.
- [Pihlblad2016] Pihlblad MS, Erenler F, Sharma A, Manchandia A, et al. Anterior segment optical coherence tomography of the horizontal and vertical extraocular muscles with measurement of the insertion to limbus distance. J Pediatr Ophthalmol Strabismus 2016;53:141-145. doi:10.3928/01913913-20160405-05.
- [Prahl1998] Prahl S. Tabulated molar extinction coefficient for hemoglobin in water (compiled from W.B. Gratzer and N. Kollias), Oregon Medical Laser Center, 1998. https://omlc.org/spectra/hemoglobin/summary.html (web page, read in full).
- [Pries1990] Pries AR, Secomb TW, Gaehtgens P, Gross JF. Blood flow in microvascular networks. Experiments and simulation. Circ Res 1990;67:826-34. doi:10.1161/01.res.67.4.826.
- [Pries1992] Pries AR, Neuhaus D, Gaehtgens P. Blood viscosity in tube flow: dependence on diameter and hematocrit. Am J Physiol 1992;263:H1770-8. doi:10.1152/ajpheart.1992.263.6.H1770.
- [Pries1994] Pries AR, Secomb TW, Gessner T, Sperandio MB, Gross JF, et al. Resistance to blood flow in microvessels in vivo. Circ Res 1994;75:904-15. doi:10.1161/01.res.75.5.904.
- [PriesSecomb2005] Pries AR, Secomb TW. Microvascular blood viscosity in vivo and the endothelial surface layer. Am J Physiol Heart Circ Physiol 2005;289:H2657-64. doi:10.1152/ajpheart.00297.2005.
- [Puyo2019] Puyo L, Paques M, Fink M, Sahel JA, Atlan M. Waveform analysis of human retinal and choroidal blood flow with laser Doppler holography. Biomed Opt Express 2019;10:4942-4963. doi:10.1364/BOE.10.004942.
- [Rana2023] Rana K, Beecher MB, Caltabiano C, Zhao Y, Verjans J, Selva D. Normal periocular anthropometric measurements in an Australian population. Int Ophthalmol 2023;43:2695-2701. doi:10.1007/s10792-023-02669-3.
- [Ritzmann2018] Ritzmann M, Caroline PJ, Börret R, Korszen E. An analysis of anterior scleral shape and its role in the design and fitting of scleral contact lenses. Cont Lens Anterior Eye 2018;41:205-213. doi:10.1016/j.clae.2017.10.010.
- [RohenFunk1994] Rohen JW, Funk RH. Functional morphology of the episcleral vasculature in rabbits and dogs: presence of arteriovenous anastomoses. J Glaucoma 1994;3:51-57. doi:10.1097/00061198-199400310-00010.
- [Runions2007] Runions A, Lane B, Prusinkiewicz P. Modeling trees with a space colonization algorithm. Eurographics Workshop on Natural Phenomena 2007:63-70. doi:10.2312/NPH/NPH07/063-070.
- [Saraswathy2016] Saraswathy S, Tan JCH, Yu F, et al. Aqueous angiography: real-time and physiologic aqueous humor outflow imaging. PLoS One 2016;11:e0147176. doi:10.1371/journal.pone.0147176.
- [Schuerch2020] Schuerch K, Frech H, Zinkernagel M. Conjunctival microangiopathy in diabetes mellitus assessed with optical coherence tomography angiography. Transl Vis Sci Technol 2020;9(6):10. doi:10.1167/tvst.9.6.10.
- [SecombPries2013] Secomb TW, Pries AR. Blood viscosity in microvessels: experiment and theory. C R Phys 2013;14:470-478. doi:10.1016/j.crhy.2013.04.002.
- [Selbach2005] Selbach JM, Rohen JW, Steuhl KP, Lütjen-Drecoll E. Angioarchitecture and innervation of the primate anterior episclera. Curr Eye Res 2005;30:337-344. doi:10.1080/02713680590934076.
- [Selbach2005b] Selbach JM, Posielek K, Steuhl KP, Kremmer S. Episcleral venous pressure in untreated primary open-angle and normal-tension glaucoma. Ophthalmologica 2005;219:357-361. doi:10.1159/000088378.
- [Shahidi2010] Shahidi M, Wanek J, Gaynes B, Wu T. Quantitative assessment of conjunctival microvascular circulation of the human eye. Microvasc Res 2010;79:109-113. doi:10.1016/j.mvr.2009.12.003.
- [Sherman1981] Sherman TF. On connecting large vessels to small. The meaning of Murray's law. J Gen Physiol 1981;78:431-453. doi:10.1085/jgp.78.4.431.
- [Shi2018] Shi Y, Hu L, Chen W, et al. Evaluated conjunctival blood flow velocity in daily contact lens wearers. Eye Contact Lens 2018;44(Suppl 1):S238-S243. doi:10.1097/ICL.0000000000000389.
- [Sit2011] Sit AJ, Ekdawi NS, Malihi M, McLaren JW. A novel method for computerized measurement of episcleral venous pressure in humans. Exp Eye Res 2011;92:537-44. doi:10.1016/j.exer.2011.03.018.
- [StraussKedra1987] Strauss AL, Kedra AW. Experiences with a new procedure for the measurement of the ophthalmic artery pressure: ophthalmomanometry-Doppler. Med Instrum 1987;21:255-261. No DOI; PMID 2960872.
- [Sun2022] Sun Z, Li Y, Liu R, et al. Progress of bulbar conjunctival microcirculation alterations in the diagnosis of ocular diseases. Dis Markers 2022;2022:4046809. doi:10.1155/2022/4046809.
- [Tan2025] Tan ST, Htoon HM, Davidson M, Rauz S, Mehta JS, Ong HS. Defining the limits of upper and lower conjunctival fornix depths in a healthy Asian Chinese population. Eye (Lond) 2025;39:2289-2297. doi:10.1038/s41433-025-03828-9.
- [Tangelder1986] Tangelder GJ, Slaaf DW, Muijtjens AM, Arts T, oude Egbrink MG, et al. Velocity profiles of blood platelets and red blood cells flowing in arterioles of the rabbit mesentery. Circ Res 1986;59:505-14. doi:10.1161/01.res.59.5.505.
- [VanBuskirk1989] Van Buskirk EM. The anatomy of the limbus. Eye (Lond) 1989;3(Pt 2):101-108. doi:10.1038/eye.1989.16.
- [vanVught2021] van Vught L, Shamonin DP, Luyten GPM, Stoel BC, Beenakker JM. MRI-based 3D retinal shape determination. BMJ Open Ophthalmol 2021;6:e000855. doi:10.1136/bmjophth-2021-000855.
- [Verma2020] Verma A, Maram J, Alagorie AR, et al. Distribution and location of vortex vein ampullae in healthy human eyes as assessed by ultra-widefield indocyanine green angiography. Ophthalmol Retina 2020;4:530-534. doi:10.1016/j.oret.2019.11.009.
- [Vogel1991] Vogel A, Dlugos C, Nuffer R, Birngruber R. Optical properties of human sclera, and their consequences for transscleral laser applications. Lasers Surg Med 1991;11:331-40. doi:10.1002/lsm.1900110404.
- [Vurgese2012] Vurgese S, Panda-Jonas S, Jonas JB. Scleral thickness in human eyes. PLoS One 2012;7:e29692. doi:10.1371/journal.pone.0029692.
- [Wang2016] Wang L, Yuan J, Jiang H, et al. Vessel sampling and blood flow velocity distribution with vessel diameter for characterizing the human bulbar conjunctival microvasculature. Eye Contact Lens 2016;42:135-140. doi:10.1097/ICL.0000000000000146.
- [Williams1990] Williams SA, Boolell M, MacGregor GA, Smaje LH, Wasserman SM, et al. Capillary hypertension and abnormal pressure dynamics in patients with essential hypertension. Clin Sci (Lond) 1990;79:5-8. doi:10.1042/cs0790005.
- [You2026] You S, Wu Y, Cai X, et al. Anterior horizontal rectus anatomy in children with concomitant strabismus. BMC Ophthalmol 2026;26. doi:10.1186/s12886-026-05090-3.
- [Yusufu2025] Yusufu M, Vingrys AJ, Shang X, et al. Population-based normative reference for retinal microvascular atlas. Ophthalmol Sci 2025;5:100723. doi:10.1016/j.xops.2025.100723.
- [Zamir1978] Zamir M. Nonsymmetrical bifurcations in arterial branching. J Gen Physiol 1978;72:837-845. doi:10.1085/jgp.72.6.837.
- [Zeimer1983] Zeimer RC, Gieser DK, Wilensky JT, Noth JM, Mori MM, et al. A practical venomanometer. Measurement of episcleral venous pressure and assessment of the normal range. Arch Ophthalmol 1983;101:1447-9. doi:10.1001/archopht.1983.01040020449024.
- [Zhang2011] Zhang X, Li Q, Liu B, et al. In vivo cross-sectional observation and thickness measurement of bulbar conjunctiva using optical coherence tomography. Invest Ophthalmol Vis Sci 2011;52:7787-7791. doi:10.1167/iovs.11-7749.
- [Zhang2013] Zhang X, Li Q, Xiang M, Zou H, Liu B, et al. Bulbar conjunctival thickness measurements with optical coherence tomography in healthy chinese subjects. Invest Ophthalmol Vis Sci 2013;54:4705-9. doi:10.1167/iovs.12-11003.
- [Zweifach1974] Zweifach BW. Quantitative studies of microcirculatory structure and function. I. Analysis of pressure distribution in the terminal vascular bed in cat mesentery. Circ Res 1974;34:843-57 (PubMed pagination; Crossref gives 841-857). doi:10.1161/01.res.34.6.841 (DOI via Crossref; abstract, served by Consensus).
