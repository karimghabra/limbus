# Identifying the vessels in a burst

### What the stage delivers today, what is wrong with it, and what to do about it

*Second draft. The first draft's diagnosis was wrong in two places and its
measuring instrument was wrong in a third; §3.4 records what happened, because
the way it failed is the most transferable thing in this document.*

---

## 0. Summary

The identification stage takes the averaged stabilized frame of a burst and
returns a set of vessels, each a continuous centreline with a radius profile
and an identifier that is supposed to mean the same thing in every frame of
the burst and on every rerun. The brief for this week was the two situations
where it is known to struggle: vessels close together in parallel, and
crossings or bifurcations crowded together.

Both are real and both are measured below. But the week turned up something
that outranks them, and it was not in the brief:

> **The stage does not currently deliver its primary product.** Vessel
> identifiers are not stable: under a perturbation smaller than the difference
> between two halves of one burst, **not one vessel in the crop keeps its id**.
> A quarter of the exported geometry on one crop belongs to two vessels at
> once. And only 58 % of the threaded centreline — the thing all the junction
> work exists to produce — ever reaches the stage that measures velocity.

These are defects in what is already built, they are cheap to fix, and until
they are fixed an improvement in resolving power cannot be banked: a better
detector that renumbers every vessel is not usable for the per-vessel,
per-burst comparison the whole pipeline exists to support.

**On the two failures in the brief, the honest answer is that one is mostly
not fixable on this data and the other is not fixed by the obvious means.**
Four candidate improvements were built and measured; three failed, and the
fourth is still being measured at proper sample size:

| built | result |
|---|---|
| SE(2) sub-Riemannian pairing of junction ends | worse than the existing turn rule everywhere; **0 %** at a 35 px junction gap |
| gating a merged fit's occupied zone | no effect at any separation — the trigger never fires |
| one-versus-two model competition on the cross-profile | works on clean profiles, **collapses on real sclera**: a true pair at 4–5 px scores below the single-vessel null |
| orientation-resolved evidence | fills the crossing hole, but at matched precision **loses** right-angle crossings, 0.00 against 1.00 |
| complete-linkage junction clustering | 0.17 → 0.67 on the worst crowding case at n = 6; high-powered run in progress |

That is not a happy table, and it is the most useful thing in the report: it
says the remaining gains are not in the detector, and it says why for each
route, so the next attempt does not spend the week the same way.

The measured state of the two failures, at 42–48 planted copies per cell with
bootstrap intervals:

| | resolved |
|---|---|
| parallel pair r = 2, 6 px apart | 0.04 [0.00, 0.10] |
| parallel pair r = 2, 8 px | 0.29 [0.17, 0.42] |
| parallel pair r = 2, 10 px | 0.87 [0.77, 0.96] |
| parallel pair r = 4, 10 px | 0.90 [0.81, 0.98] |
| crossing at 90° | 0.78 [0.61, 0.91] |
| crossing at 40° | 0.47 [0.29, 0.65] |
| crossing at 15° | 0.00 |
| thin vessel over a thick one, 25° | 0.05 [0.00, 0.12] |
| bifurcation at 35° / 60° / 90° | 0.98–1.00 (two independent runs) |
| bifurcation at 20° | 0.19 [0.08, 0.31] |
| trunk, 3 branches in 60 px | 0.61 [0.45, 0.77] |
| **trunk, 3 branches in 30 px** | **0.24 [0.13, 0.38]** |
| 3 vessels braided 6 px apart | 0.00 |

Read together with §5.3b and §6.3 these say something simple. **Bifurcations
are handled.** What fails is any configuration in which two vessels run nearly
parallel and close — a shallow crossing, a shallow bifurcation, a braid, a
close pair — and, separately, any place where junctions crowd.

Of those two, the first is largely **not recoverable on the averaged frame**.
Sclera texture is correlated at the same spatial scale as the gap between two
close vessels, so a true pair 4–5 px apart produces a weaker signal than a
genuine single vessel does (§5.3b). The 4.25 px figure the optics allow is
noise-free; the texture-limited limit on this data is nearer 6–8 px, and the
detector's ~9 px is close to it. The factor of two that looked like method
failure is mostly not there.

The second has a candidate fix that is **not** a net gain, which took the full
sweep to establish. Junction ends are clustered by single-linkage union-find,
which is transitive, so crowded branches chain into one node of degree 7-8
that the classifier cannot read - it handles degree <= 4 and leaves everything
above it unpaired. Bounding a node's diameter (complete linkage) does exactly
what that mechanism predicts in the tightest case and the opposite one cell
away:

| trunk scenario | single | complete |
|---|---|---|
| 2 branches in 60 px | 0.71 [0.56, 0.83] | 0.72 [0.60, 0.85] |
| 3 branches in 60 px | 0.61 [0.45, 0.77] | 0.52 [0.35, 0.68] |
| **3 branches in 30 px** | 0.24 [0.13, 0.38] | **0.64 [0.50, 0.79]** |
| **4 branches in 40 px** | **0.87 [0.77, 0.97]** | 0.56 [0.42, 0.72] |
| bifurcation at 20 deg | 0.21 [0.10, 0.33] | 0.23 [0.12, 0.35] |

Two non-overlapping differences in opposite directions, a mean of 0.61 either
way, and complete linkage merging more (0.11 against 0.03 merges per copy at
four branches). On the real networks it gives 17 % fewer vessel objects for
1.4 % less centreline with the median vessel 30 % longer - which is either
better threading or wrong merging, and the scenarios say some of it is the
latter. **The shipped default is therefore unchanged**, with complete linkage
available and its trade-off documented.

*(An earlier draft of this section, and the commit that introduced the change,
claimed it as a win on the strength of the first three cells. The fourth
reversed it.)*

The fix that addresses the mechanism without that side effect — reading a
meeting point of degree >= 5 instead of abandoning it — was then built, and is
worse still. Pairing those ends greedily by turn and calibre takes the same
crowded case from **0.23 to 0.14** and raises merges from 0.02 to **0.59 per
copy**: at a crowded node a trunk and its own branch look exactly like a
continuation, so the rule fuses them. Note that `span` went *up* over the same
change, 0.86 to 0.91 — the metric of §3.4, had it not been fixed, would have
called this an improvement.

So the crowded-junction mechanism is established and neither repair works.
What both share is that they decide from local geometry, which at a crowded
node cannot distinguish a trunk continuing from a trunk branching.

**Calibre cannot rescue it either**, which is the obvious next idea and is
worth ruling out explicitly: by Murray's law a trunk that sheds a symmetric
branch continues at *exactly* the branch's radius — r = 2.5 sheds r = 1.98 and
carries on at 1.98, a ratio of 1.00. The two are equal by construction. Only
an asymmetric split separates them at all (a 30 % branch leaves a ratio of
1.33).

What does separate them is **direction** — the trunk runs on, the branch turns
away — and that is already the test. It fails at a crowded node for a
mechanical reason: the pieces there are short, and an end's tangent measured
over 10 px of a 15 px piece is mostly noise. So the repair to try is a tangent
measured over an arc that provably excludes the junction, with a refusal to
pair when no such arc exists. That is not built, and it is the single most
specific thing this report can hand to whoever picks it up.

**And one plain bug, found last and worth more than most of the above.**
`graph._tangent` measured the direction a piece points at its ends. The start
branch used the requested arc; the end branch indexed from the wrong side and
took the chord from 10 px past the *start* to the end — 84 px of a 94 px arc
on a curved piece, returning 140° where the vessel runs at 180°. That
direction decides every pairing at every junction: continuation, crossing,
which child a parent continues into, `join_ends`' 40° gate, and
`attach_to_body`. It has been wrong in all of them since arc-length tangents
were introduced.

Corrected by interpolating the centreline at the two arc lengths the tangent
spans, so there is no index arithmetic left to get wrong. On the real crops,
with the surrogate false-alarm rate unchanged at 3.3 % on 1522R:

| crop | vessels | px/MP | longest | median |
|---|---|---|---|---|
| 1522R | 31 → 30 | 17 728 → 17 726 | **681 → 972** | 151 → 141 |
| 1522L | 52 → 51 | 20 588 → 20 308 | 941 → 942 | **108 → 128** |

The same evidence and the same detected length, threaded better. It was found
by accident — a feature flag that should have been a no-op was not — which is
an argument for making new options exactly inert by default and checking that
they are.

**The planted scenarios do not see it**, and that is worth recording as a
limitation of the benchmark rather than of the fix. Across every parallel
separation the means are 0.405 against 0.398, and the crossings read
0.00/0.00, 0.26/0.26, 0.47/0.47 and 0.79/0.75 — unchanged, with merges at 60°
halved (0.12 against 0.25). The scenarios plant vessels that are straight or
gently curving, and on a straight piece the old whole-piece chord happened to
give nearly the right direction. The bug only bites on curvature, which the
real networks have and the harness does not. Adding a high-curvature family is
the obvious repair, and is not done.

The rest of what was tried failed, and the failures are the useful part:

| built | result |
|---|---|
| SE(2) sub-Riemannian pairing of junction ends | worse than the existing turn rule at every angle; **0 %** at a 35 px junction gap |
| gating a merged fit's occupied zone | no effect at any separation — the trigger never fires |
| one-versus-two model competition on the cross-profile | works on clean profiles, collapses on real sclera |
| orientation-resolved evidence | fills a real crossing hole, but at matched precision **loses** right-angle crossings, 0.00 against 1.00 |
| **complete-linkage junction clustering** | **adopted** — see above |

There is one genuine evidence defect that remains unfixed: at a 90° crossing
the Hessian keeps only **0.12** of a vessel's off-junction strength, against a
single-vessel null of 0.99. The pipeline survives it because hysteresis
bridges the gap, which works and is luck rather than design. The obvious
repair — an orientation-resolved filter, which does read 1.00 there — costs
more at matched precision than it returns.

---

## 1. What "good" means here, in order

1. **Identity.** One real vessel, one object, one identifier, meaning the same
   thing on every rerun and in every frame. Everything downstream is indexed by
   it.
2. **Correctness of what is claimed.** A merged pair reports one vessel with
   the wrong calibre where there are two.
3. **Completeness.** A missing faint vessel costs a measurement.

The first draft of this report put continuity first and identity inside it.
That was a mistake of emphasis: continuity is a means, and §4 shows the stage
achieving reasonable continuity while failing identity outright.

---

## 2. The current method

Absorbance → ridge evidence → tracing → gap joining → model fit → junction
reading and consolidation → growth → identifiers. `analysis/vessels/README.md`
has the detail.

One structural fact, measured rather than asserted. The same growth code with
the same thresholds, run at the end of the pipeline and from confident seeds
at the start:

| growth runs… | real centreline gained | texture surrogate gained |
|---|---|---|
| after the full pipeline | +0.2 % | +3.0 % |
| from confident seeds only | **+44 %** / **+64 %** | **0 %** |

A factor of two hundred. By the time the late stages run, the thresholded
decision has been taken and there is nothing left to find.

**Correction to the first draft.** I drew from this the conclusion that "a
repair belonging in the evidence or the tracing must be made there", and used
it to argue that the fit and graph stages could only rearrange what they were
given. That is too strong: the stages after evidence cannot *create* a vessel,
but they can and do destroy one - §4.2 and §4.3 show geometry being dropped and
duplicated between the threading and the export, and §7 shows the junction
clustering losing whole trunks where branches crowd. "Cannot introduce" is
right; "can only rearrange" was not.

---

## 3. The instrument

### 3.1 Why the existing benchmark could not see either failure

It plants isolated blurred cylinders — avoiding everything already detected and
enforcing a 10 px gap between plants — and asks what fraction of each planted
centreline lies within 2.5 px of some detection. No configuration it produces
contains two close vessels, a crossing or a junction, and coverage is satisfied
by a fragmented network, a fused pair, or a swapped crossing.

### 3.2 Scenarios

Seven families with controlled geometry, planted into the absorbance of a real
crop: `parallel` (separation 3–20 px at r = 2), `parallel_thick` (6–24 px at
r = 4), `cross` (15–90°), `cross_uneven` (r = 2 over r = 5), `bifurcation`
(20–90°, Murray radii), `ladder` (2–4 branches in a 30–60 px window), `braid`
(3–5 converging vessels).

### 3.3 Metrics

- **resolved** — the scenario came back right: every truth vessel recovered
  whole by a single detection **and** no detection covering two of them. This
  is the headline number, and it is the conjunction of two things that must be
  measured separately.
- **span** — largest fraction of a truth vessel covered by one detection.
- **cover** — fraction covered by everything together; the old metric, kept for
  comparison.
- **parts** — detections in a minimal cover.
- **merges** — detections covering ≥ 35 % of two different truth vessels.

Coverage uses a tolerance that scales with how close the truth lines actually
are, `tol(p) = clip(0.6 · d_nn(p), 1.5, 8)` — more than half the local
separation and less than all of it — so a single midline detection covers both
lines of a pair and is reported as a merge, while two separated detections each
cover their own and not the other.

Statistics are per **copy**, not per line: the two lines of a pair share one
detection and one fate. 48 copies per configuration, bootstrap 95 % intervals.

### 3.4 The first version of this instrument was wrong in the same way as the one it replaced

The first version assigned every detected point to the nearest truth line by a
hard Voronoi partition and counted it only for that one, on the reasoning that
a line down the middle of a pair would be split between them and register as a
merge. It does the opposite: the whole midline falls on one side of the
partition, so a merged detection scored as a clean hit on one vessel and a
total miss on the other.

Measured consequence: at 3, 4 and 5 px separation the detector emitted one
midline detection every time — the same behaviour — and the metric called it
"both found", "one found, one lost", and "both missed" respectively, purely
according to where the tolerance fell relative to half the separation. From
that I wrote a headline about the detector "tracing one of two vessels and
silently dropping the other", and a section about a non-monotonic "dead zone"
that was the boundary of my own tolerance.

This is the second time in this project that a metric has been unable to see
the defect it was built for. The lesson I am taking is narrower and more
useful than "be careful": **a metric must be validated against known
configurations before it is used to draw conclusions.** The current one is —
`score2.py` is checked against a resolved pair, a merged midline, a
single-vessel detection and a four-way shatter, at every separation, and gives
1.00/1.00 + no merge, 1.00/1.00 + merge, 1.00/0.00, and span 0.27 with cover
1.00 respectively. That table should exist for any metric before it is
believed.

---

## 4. What the stage delivers today

Three defects in the shipped path, found by review and confirmed by
measurement. None is about resolving power.

### 4.1 Identifiers are not stable

`network._thread` assigns ids by enumerating a depth-first order over the
hierarchy, roots sorted by `(-calibre, -length)`. **Every id is a global rank.**
Insert one vessel, or change one calibre, and every id after it moves. (The
README's claim that ids are "by position, row band then x" describes
`network.group`, which runs only on the `--no-graph` path.)

The existing determinism test compares two runs on the identical array, which
tests reproducibility, not stability. Measured stability, on 1522R, against
perturbations smaller than the difference between two halves of a burst:

| perturbation | matched | same id | same label | mean id change |
|---|---|---|---|---|
| identical rerun | — | **93 %** | 93 % | 0.1 |
| half-pixel shift | 27/31 | **0 %** | 4 % | 7.1 |
| noise at one frame's level | 25/31 | **0 %** | 0 % | 7.8 |

Verified with a second, stricter matcher on the twelve longest vessels: id 1 →
19, 28 → 30, 22 → 15, 3 → 10, 23 → 31, with centroid movements of 0.0–2.5 px
and lengths unchanged (399 → 399 px in one case). The vessel *count* is not
stable either: 31 → 33 under noise alone.

**An id must be content-addressed** — a geometric fingerprint in a
burst-independent frame, with an explicit matching stage between runs — and a
split must be reported as provenance (`id → id.a, id.b`) rather than as a
renumbering.

### 4.2 A quarter of the exported geometry belongs to two vessels

`graph.split_at_touches` copies a segment with `dict(s)`, so both halves share
one `fit` object by reference. When the halves land in different chains, both
vessels export the same piece — byte-identical point sets:

| crop | vessels | exported pieces | shared between two vessels |
|---|---|---|---|
| 1522L | 52 | 66 | **17 (26 %)** |
| 1522R | 31 | 45 | 5 (11 %) |

Because `profiles.build` samples `pieces[0]`, two different vessel ids can be
measured along the same line and return identical kymographs and velocities.

### 4.3 Most of the threaded centreline never reaches the consumer

`__main__.py` exports `id, anchor, length_px, radius_px, n_pieces, pieces`. The
threaded `centreline`, and the `label`, `parent_chain`, `strahler` and
`junctions` that the junction work produces, are computed and discarded. Then
`profiles.build` filters on `length_px` — the *threaded* length — and samples
`pieces[0]`, the longest single fitted piece.

| crop | threaded | reaches velocity | |
|---|---|---|---|
| 1522R | 7 176 px | 4 164 px | **58 %** |

Per vessel it is worse: the longest vessel on 1522R is 681 px threaded and
contributes 120 px — **18 %**. So a vessel qualifies for measurement on its
threaded length and is then measured over a fifth of it.

**This is the highest-value repair available**, because the junction and
threading work from the previous round is already built and is simply not
being read.

---

## 5. Resolving two vessels that run together

### 5.1 The limit

A cylinder's chord profile has variance `r²/4`, so after optics of width
σ_psf the observed profile is near-Gaussian of width
`σ_eff = √(σ_psf² + r²/4)`, and the Sparrow criterion for two equal Gaussian
lines is `s_min = 2σ_eff`. **This approximation is only good for r ≲ 2.5**: the
chord profile is four times less curved at its peak than a variance-matched
Gaussian, so the formula is optimistic by 10 % at r = 3, 19 % at r = 4 and 37 %
at r = 12. Exact values for the blurred chord at σ_psf = 1.8:

| r | 1 | 2 | 3 | 4 | 5 | 12 |
|---|---|---|---|---|---|---|
| s_min | 3.77 | 4.25 | 5.21 | 6.62 | 8.25 | 19.95 |

Sparrow is also noise-free, and at 5 % contrast the binding constraint may be
SNR. It is a floor, not a prediction.

### 5.2 What the detector achieves

1522R, 48 copies per configuration, `resolved` with bootstrap 95 % interval:

| separation | r = 2 | r = 4 |
|---|---|---|
| 3 px | 0.00 [0.00, 0.00] | — |
| 4 px | 0.00 [0.00, 0.00] | — |
| 5 px | 0.00 [0.00, 0.00] | — |
| 6 px | 0.04 [0.00, 0.10] | 0.00 [0.00, 0.00] |
| 8 px | 0.29 [0.17, 0.42] | 0.17 [0.08, 0.27] |
| 10 px | 0.87 [0.77, 0.96] | 0.90 [0.81, 0.98] |
| 12–14 px | 0.98 [0.94, 1.00] | 0.98 [0.94, 1.00] |
| 20–24 px | 1.00 | 1.00 |

Monotonic, with a half-resolution point near **9 px for both calibres**.
Against the exact limits that is **2.1× at r = 2** and **1.3× at r = 4**: the
deficit is worst where vessels are thinnest. Below 6 px every copy is merged
(merges per copy ≈ 1.0).

`cover` stays at 0.79–1.00 across this whole table while `resolved` runs 0.00
to 1.00 — the old metric's blindness, shown rather than argued.

### 5.3 A mechanism that looked right and is not: the fit deleting the neighbour

A merged pair is fitted as one wide vessel, and the fitted radius is not a
measurement of anything: a pair of r = 2 vessels 5 px apart fits at r = 6.0,
at 8 px apart at r = 7.75. `fit.run` marks that vessel's lumen plus
`0.2r + psf` as occupied and cuts every later candidate inside it
(`split_outside`), so an inflated radius sterilises a zone far wider than the
true separation. Both a reviewer and I derived that the twin must therefore be
removed before it is ever fitted, and a first measurement agreed: with the fit
disabled, span rose from 0.62 to 0.81 at 6 px and 0.72 to 0.88 at 8 px.

**That measurement used the broken metric of §3.4.** Repeated with the
corrected one, fit on versus fit off shows no signal at any separation —
0/4 against 0/4 at 5 and 6 px, 1/3 against 2/3 at 8 px. Gating the occupancy
explicitly (`merged_occupancy = core | off`) was then implemented and changed
nothing at any separation, because its trigger never fires: `single_peaked`
reads 0.954 against its own 0.88 threshold on a fitted pair 5 px apart. The
knob has been removed.

The arithmetic behind the mechanism is still sound; it simply is not what
limits the detector. What does is §5.3b.

### 5.3b The gap to the Sparrow limit is mostly not recoverable

The obvious reading of §5.1 and §5.2 — the optics allow 4.25 px, the detector
manages 9, so there is a factor of two to be won — is wrong, and I built the
thing that would have won it in order to find that out.

`vessels/pairs.py` decides once per candidate, on its own cross-profile,
between one blurred cylinder and two, with the amplitudes and background
solved in closed form and the separation scanned on a grid. On **clean
profiles with white noise** it works very well:

| truth | statistic | separation recovered |
|---|---|---|
| two r = 2 at s = 4 | 0.276 | 4.0 |
| two r = 2 at s = 5 | **0.797** | **5.0** |
| two r = 2 at s = 6 | 0.927 | 6.0 |
| one r = 2 | 0.051 | — |
| one r = 3 | 0.190 | — |
| one r = 4 | 0.028 | — |

A threshold at 0.4 separates a true pair at 5 px from any single vessel, and
the separation comes back to a tenth of a pixel.

**On real sclera it collapses.** The same statistic, with the nulls measured
rather than assumed:

| | median statistic |
|---|---|
| null — phase-randomised surrogate, no vessel | 0.246 (99th pct **0.907**) |
| null — the detector's own accepted vessels | 0.352 (max **0.739**) |
| true pair at s = 4 | **0.190** |
| true pair at s = 5 | **0.244** |
| true pair at s = 6 | 0.687 |
| true pair at s = 8 | 0.907 |

True pairs 4–5 px apart score *below* the single-vessel null. A threshold that
does not split real vessels (> 0.74) would fire only from s ≈ 8, which is
where the shipped detector already begins to work. And at s = 3 — below the
model's own floor, where a split would be invention — it fires 67 % of the
time: it is most confident exactly where it must not be.

Averaging further along the vessel does not rescue it. Across along-sample
counts of 60, 200 and 600 and candidate lengths of 140 and 300 px, the
single-vessel null ranges 0.15–0.45 and the s = 4 and s = 5 signals range
0.21–0.59, overlapping completely and not ordering with separation. That
disorder is itself the finding.

**The reason generalises beyond this method.** Sclera texture is correlated at
the same spatial scale as the dip being measured, so averaging along a vessel
does not suppress it the way independent noise would. The Sparrow figure in
§5.1 is a *noise-free* limit; the texture-limited limit on this data is
somewhere around 6–8 px, and the detector's ~9 px is much closer to it than
the first reading suggested.

So the parallel failure is, to a first approximation, **not a method failure
that better modelling of the averaged frame can fix**. The remaining honest
options are to report the ambiguity (§7), or to bring in information that is
not in the averaged frame at all — and the only such information available is
flow, which cannot enter identity but could serve as an oracle (§9).

### 5.4 Two things I checked and was wrong about

**Scale selection is real but not sufficient.** A transverse scale of
σ = 0.7–1.0 separates an r = 2 pair completely at 5 px and half at 4 px, while
the shipped γ-normalised largest-response selection resolves nothing closer
than 6. That measurement stands. But selecting the finest significant scale
per pixel is non-monotonic (it resolves 4 px and fails at 6, because a
spatially mixed scale map is not a clean ridge), and unioning the scales is
worse: at 5 px the union contains the two fine-scale vessels *and* the coarse
scale's spurious midline, and the 2×2 dilation that closes NMS breaks fuses
all three. Scale evidence must be arbitrated between objects, not per pixel —
and that is a larger change than the first draft implied.

**An anisotropic orientation bank does not help parallel pairs.** Measured: the
same answer as the Hessian. The limiting factor is the transverse scale, not
the filter's shape.

---

## 6. Crossings and junctions

### 6.1 There is an evidence hole at steep crossings, and I originally measured it backwards

The first draft reported no hole, from a probe that took the **maximum over a
±8 px window** centred on the junction — a window that straddles the crossing
and picks up the other vessel's ridge. The reported value was a function of
window width alone (0.12 at w = 0, 1.08 at w = 4).

Measured at the junction, with a single-vessel null confirming the estimator
reads 0.99–1.00 where there is no crossing:

| crossing angle | Hessian (r = 2) | Hessian (r = 4) | orientation plane |
|---|---|---|---|
| 90° | **0.12** | **0.07** | 1.00 |
| 60° | 0.94 | 0.92 | 1.01 |
| 40° | 1.37 | 1.41 | 1.04 |
| 25° | 1.69 | 1.73 | 1.18 |
| 15° | 1.89 | 1.84 | 1.43 |

So the two curvatures do cancel at a right-angle crossing, the evidence
collapses to 7–12 % of its off-junction value, and an orientation-resolved
filter removes the hole entirely. At shallow angles the opposite happens — the
two vessels superpose and the evidence nearly doubles, which is the fusion
problem again, and the orientation filter reduces but does not remove it.

The hole survives today because hysteresis with `reach = 60` bridges it. That
works, and it is fragile: it depends on a faint-ridge threshold being crossed
either side of a gap whose depth is 88 % of the signal.

### 6.2 What the pipeline does with junctions

| scenario | resolved |
|---|---|
| crossing 90° | 0.78 [0.61, 0.91] |
| crossing 60° | 0.75 [0.54, 0.92] |
| crossing 40° | 0.47 [0.29, 0.65] |
| crossing 25° | 0.26 [0.13, 0.39] |
| crossing 15° | 0.00 [0.00, 0.00] |
| thin over thick, 90° | 0.96 [0.87, 1.00] |
| thin over thick, 25° | 0.05 [0.00, 0.12] |
| **bifurcation 90°** | **1.00 [1.00, 1.00]** |
| **bifurcation 60°** | **1.00 [1.00, 1.00]** |
| **bifurcation 35°** | **1.00 [1.00, 1.00]**, 0.98 [0.93, 1.00] on a rerun |
| **bifurcation 20°** | **0.19 [0.08, 0.31]** |
| trunk, 2 branches in 60 px | 0.50 [0.28, 0.72] |
| trunk, 3 branches in 60 px | 0.29 [0.12, 0.53] |
| trunk, 3 branches in 30 px | 0.29 [0.12, 0.53] |
| 3 vessels braided, 6 px apart | 0.00 [0.00, 0.00] |

Even a right-angle crossing survives intact only 78 % of the time, and
crowding branches onto a trunk roughly halves the success rate — the user's
stated complaint, reproduced and quantified.

**Bifurcations are handled well, and an earlier draft said the opposite.** It
reported 0.00 at every angle, which was a third metric error: at a bifurcation
the parent *continuing into one child* is one vessel and is the right answer,
and the merge test was counting it as a failure. The tell was that the number
was 0.00 at 20°, 35° and 60° alike while span sat at 0.98 — too uniform to be
physical. With parent-child continuations exempted, bifurcations are perfect
at 35° and 60° and fail only at 20°.

### 6.3 The shallow-angle failures are all one failure

Crossing at 15–25°, bifurcation at 20°, and the braid are the same thing:
two vessels running nearly parallel, close together, over a long stretch. At
15° a crossing pair sits within 6 px of each other for ~46 px either side of
the meeting point. **Every junction failure in the table above is a fusion
failure wearing a different hat**, and the one detector change that would
address all of them is raising the parallel resolving power of §5.

### 6.4 Pairing ends at a junction is **not** the problem

I proposed replacing the turn-angle pairing rule with the closed-form
sub-Riemannian distance on SE(2), on the strength of a published 99.72 % versus
89.99 % on retinal crossing disambiguation. I tested it before building it, on
four-end junctions with realistic position and tangent noise:

| crossing angle | max-turn correct | SE(2) correct |
|---|---|---|
| 10° | 93 % | 73 % |
| 15° | 100 % | 84 % |
| 25° | 100 % | 93 % |
| 40–90° | 100 % | 99–100 % |

and at a larger junction gap (35 px) SE(2) drops to **0 %** at every angle,
because sub-Riemannian *distance* grows with path length and at a crossing the
correct pairing spans the longest gap while the wrong ones span less. The
published baseline was position-only fast marching; the existing code already
uses tangents, and is near-perfect. **The existing rule is not the weak link,
and P3 is withdrawn.**

The weak link is upstream of it: at 15° two vessels sit within 6 px of each
other for ~46 px either side of the meeting point, so no four-way junction
ever forms — there is a fused stretch instead. And where junctions are crowded,
`graph.build` clusters ends by single-linkage union-find with
`tol = node_tol + r_a + r_b`, which is transitive, so three bifurcations within
30 px collapse into one node of degree 7–8; `classify` handles degree ≤ 4 and
returns `unresolved` with no pairs above that, leaving every end in the cluster
unpaired.

---

## 7. What to do, in order

**Stage 0 — make the stage deliver what it already computes.** Nothing here is
new science and all of it is measured above.

1. **Export the threaded centreline, label, parent and junctions**, and make
   `profiles.py` sample the threaded centreline instead of `pieces[0]`.
   Recovers 42 % of the centreline on 1522R, and up to 82 % on the longest
   vessels.
2. **Stop sharing `fit` objects across chains** in `split_at_touches`; copy the
   piece. Removes 26 % duplicate geometry on 1522L.
3. **Content-addressed identifiers**, with an explicit matching stage between
   runs and provenance for splits and merges. Add a stability test — half-pixel
   shift, frame-level noise — to the test suite as a gate.

**Stage 1 — raising the parallel resolving power was the main event, and it
is closed.** §6.3 shows every junction failure is a fusion failure, so moving
the limit would have improved all of them at once. §5.3b reports the attempt
and its failure: on real sclera a true pair 4–5 px apart produces a weaker
signal than a genuine single vessel, and the texture-limited limit is around
6–8 px against the ~9 px already achieved. **There is much less here to win
than the optical limit suggests, and I do not propose pursuing it further on
the averaged frame.**

What follows from that is a change of terminal behaviour rather than of
resolving power:

4. **Report the ambiguity.** A candidate whose profile is consistent with two
   vessels, at a separation the model cannot support, should carry
   `possibly_two` and its measured separation into `vessels.json` — and, since
   nothing downstream currently consumes a flag, the velocity stage should
   decline to report a trusted speed for such a vessel rather than averaging
   two lumens. §1 ranks a wrong claim above a missing one, and a fused pair
   reporting one confident speed is a wrong claim.

**Stage 2 — crossings: the evidence hole is real, and the obvious fix is
worse than the problem.**

The orientation stack does remove the hole (§4.4, 0.12 → 1.00), and on real
crops it finds 30–43 % more centreline. Almost all of that is texture:

| crop | evidence | real px/MP | surrogate px/MP | false |
|---|---|---|---|---|
| 1522R | hessian | 17 728 | 1 605 | **9.1 %** |
| 1522R | orientation | 25 294 | 13 518 | **53.4 %** |

Taking the maximum over sixteen orientation planes is a multiple-comparisons
problem — the robust-z threshold was calibrated for one orientation, and with
sixteen chances to exceed it the operating point moves. So the comparison has
to be at matched false-alarm rate:

| orientation t_hi / t_lo | real px/MP | surrogate | false |
|---|---|---|---|
| 3.0 / 1.5 | 25 294 | 13 081 | 51.7 % |
| 4.0 / 2.0 | 22 188 | 7 434 | 33.5 % |
| 5.0 / 2.5 | 19 056 | 6 072 | 31.9 % |
| 6.0 / 3.0 | 18 188 | 1 528 | 8.4 % |
| **7.0 / 3.5** | **16 513** | **0** | **0.0 %** |
| 8.0 / 4.0 | 16 489 | 0 | 0.0 % |

against the Hessian's 17 728 real and 578 surrogate (3.3 %) on the same
image. At t = 7.0 the orientation evidence returns 93 % of the Hessian's
length with no measurable false alarms at all — a better operating point on
precision, at a small cost in completeness, and exactly the shift sixteen
planes should require.

*(An earlier version of this section read the first three rows and concluded
that the false fraction plateaued around a third, and that this was a worse
ROC rather than a mis-set threshold. The next two rows contradict that. It was
a conclusion drawn from a partial curve — the same error this report spends
§3.4 and §5.4 documenting.)*

Whether that precision translates into crossings surviving is the question the
filter exists to answer, and it is being measured at the matched thresholds
now.

5. **Flag, don't delete.** `single_peaked` drops a merged large-pass
   candidate; keep it, carrying the measured dip. §1 ranks a wrong claim above
   a missing one.
6. **Bound a junction's diameter — adopted.** `graph.build` clustered ends by
   single-linkage union-find, which is transitive, so branches crowded onto a
   trunk chain into one node of degree 7–8 that `classify` cannot read (it
   handles degree ≤ 4 and returns "unresolved" above that, leaving every end
   in the cluster unpaired). Measured, the same three branches resolve 0.64 in
   a 60 px window and **0.25** in a 30 px one — crowding, not the number of
   branches, is what breaks it.

   Complete linkage keeps a node's own diameter bounded. On the **real**
   networks it is a clear gain in continuity, which §1 ranks first:

   | crop | clustering | vessels | px/MP | longest | median | > 300 px |
   |---|---|---|---|---|---|---|
   | 1522R | single | 31 | 17 728 | 681 | 151 | 9 |
   | 1522R | complete | **29** | 17 728 | **849** | 151 | 8 |
   | 1522L | single | 52 | 20 588 | 941 | 108 | 8 |
   | 1522L | complete | **43** | 20 292 | 942 | **140** | **10** |

   Fewer, longer vessels at essentially unchanged total length: 17 % fewer
   objects on 1522L for 1.4 % less centreline, with the median vessel 30 %
   longer.

   And on the planted scenarios it buys exactly what the mechanism predicts,
   where it predicts it:

   | trunk scenario | single | complete |
   |---|---|---|
   | 2 branches in 60 px | 0.71 [0.56, 0.83] | 0.72 [0.60, 0.85] |
   | 3 branches in 60 px | 0.61 [0.45, 0.77] | 0.52 [0.35, 0.68] |
   | **3 branches in 30 px** | **0.24 [0.13, 0.38]** | **0.64 [0.50, 0.79]** |

   n ≈ 45 per cell. The crowded case — the one this report started from —
   improves 2.7-fold with non-overlapping intervals; the uncrowded cases are
   unchanged or slightly worse with intervals that overlap heavily. That is
   the right shape: the change costs little where junctions are separated and
   buys a great deal where they crowd.

**Not proposed any more.** Gating the fit's occupancy on a merged profile was
built and measured: no effect at any separation, because the trigger never
fires. The mechanism was plausible and the arithmetic supporting it was sound;
it simply is not what happens.

---

## 8. How it will be judged

Every change, against a frozen upstream, reported together:

1. **The hard scenarios** — `resolved` with bootstrap CIs, stratified by
   separation and angle. Primary.
2. **Identifier stability** — the §4.1 test. A change that improves resolution
   and renumbers the network has failed.
3. **The existing injection benchmark** — thin-vessel recall must not fall.
4. **The texture surrogate** — false-alarm length per megapixel must not rise.
5. **Real-crop network statistics** — vessel count, total length, longest
   vessel.
6. **Planting realism**, which is still not run and gates every number in §5–6:
   a matched-filter detector should score comparably on planted and on real
   vessels. If plants are much easier, the numbers are optimistic.

Order matters: P1-type changes multiply fragments and would flatter or spoil
any pairing change measured after them, so each is evaluated against a frozen
upstream before they are combined.

---

## 8b. What has been built, and what was built and then discarded

**Done and committed** (`8b649fb`, `6939f7b`, `4b323d2`):

- The export now carries the threaded centreline, radius profile, label,
  parent and junctions, and `profiles.py` samples it. A merged piece keeps
  every member's fit; a split piece takes only the fits within the span it
  kept. Velocity sees **7310 px of 1522R where it saw 4332 (+69 %)**; on
  1522L it reads 9089 px where it read 9801, the old figure having exceeded
  the 8995 px actually threaded because it counted shared geometry twice.
  Duplicated pieces on 1522L: 17 → 3.
- `vessels/scenarios` — the geometries and the scorer — with the scorer's
  validation against known configurations in the test suite, so the failure in
  §3.4 cannot recur silently.
- `vessels/experiments/id_stability.py`, which produces the §4.1 numbers.

**Built, measured, and not adopted.** Both are recorded because the cost of
finding out was the point:

- **SE(2) sub-Riemannian pairing.** Tested on four-end junctions before being
  built into the pipeline. The existing turn-angle rule is 92–100 % correct at
  every angle; the sub-Riemannian distance is worse everywhere and **0 % at a
  35 px junction gap**, because distance grows with path length and at a
  crossing the correct pairing spans the longest gap while the wrong ones span
  less. The published 99.7 % vs 89.99 % compared against *position-only* fast
  marching; this code already uses tangents. Withdrawn.
- **Gating the fit's occupancy on a merged profile.** Implemented as
  `merged_occupancy = full | core | off`. Measured on planted pairs: **no
  difference at any separation**, because the trigger never fires —
  `single_peaked` on a *fitted* merged pair returns "single" (centre/peak
  0.954 against its 0.88 threshold at s = 5). This is the same error as §5's
  first diagnosis: a fix wired to a trigger that does not fire. The earlier
  "disabling the fit helps" result was measured with the broken metric and is
  being re-measured before any occupancy claim is made.

---

## 9. Known weaknesses of this report

- **σ_psf is treated as a constant 1.8 px** and it is neither. Residual
  non-rigid stabilization error adds to it, varies across the frame, and is
  largest on bare sclera where the registration has nothing to lock to. The
  displacement fields are on disk and this is computable; until it is, §5.1's
  limits are a lower bound and planted vessels are sharper than real ones.
- **The scale is unknown in µm.** The calibration bursts give 27.46 and
  24.99 px per grid period but the target pitch has not been supplied, so
  "two vessels 4 px apart" cannot be interpreted physiologically — and whether
  such a pair is two vessels, one vessel and its paired venule, or two plexus
  layers projected together changes what the right answer *is*.
- **One crop carries the statistics.** 1522L has 31 % free ground and could
  place only 8–9 copies, with five configurations unplaceable; the third crop
  is 17 % free and unusable. The scenarios also sit in the emptiest sclera by
  construction, which is the opposite of the crowded case they describe.
- **No real-data measurement of either failure.** Everything is planted.
  Velocity could serve as a held-out oracle — a fused pair carrying two
  different flows gives an incoherent kymograph where a genuine wide vessel
  does not — provided it is fixed before tuning and never becomes an input.
- **No runtime budget has been set**, so the cost column is decoration.
