"""Builds limbus_vessel_workflow.ipynb (run: python build_notebook.py)."""
import nbformat as nbf

cells = []
M = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
C = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

# =============================================================================
M(r"""
# Limbal vessel annotation, morphometry & blood-flow velocimetry
### A step-by-step, *show-the-maths* walkthrough

**Input**
* `mean_stabilized.tif` – a registered mean image of the limbus / sclera (the *reference*), 1920 × 500 px
* `burst_2026-09-16_15-22-26/` – the raw 12-bit video it came from: 1105 frames at ≈ 74 fps (Basler acA1920-40um)

**Output**
* every visible vessel as a **smoothing-spline centre-line** with a **mask**, **diameter profile**, **tortuosity** and **red-blood-cell (RBC) velocity**
* a **directed graph** of the network in which every edge points along the direction of blood flow
* files for the annotation software (`outputs/*.json`, `.graphml`, `.csv`) and **validation videos**

| step | what | key idea |
|---|---|---|
| 1 | data | timestamps, auto-exposure, blinks |
| 2 | registration | SIFT + RANSAC similarity, ECC refinement |
| 3 | vessel enhancement | Hessian eigenvalues → Frangi vesselness |
| 3b | functional detection | temporal flicker of the registered video, jitter-corrected → thin perfused vessels |
| 4 | segmentation | hysteresis threshold on structural + functional vesselness → skeleton |
| 5 | graph | pixel topology → nodes & segments, spur pruning, crossings |
| 6 | splines | parametric cubic B-splines, arc length, Frenet frame, curvature |
| 7 | diameter | cross-sectional profiles, FWHM vs model fitting |
| 8 | masks | centre-line ± d(s)/2 |
| 9 | tortuosity | DM, SOAM, κ², inflections |
| 10 | velocity | kymographs → LSPIV / structure tensor / time of flight |
| 11 | directed graph | flow direction, mass conservation, node types, export |
| 12 | **interactive explorer** | pick a vessel → mask + all metrics |
| 13 | **video validation** | tracers advected at the measured speed, co-moving kymographs |
| 14 | summary & limitations | |

Every measurement step ends with a **ground-truth check** on synthetic data with known answers.
All of the code lives in the `limbusflow/` package; the notebook calls it one step at a time so each intermediate result can be seen.
""")

C(r"""
%matplotlib inline
import os, sys, json, time, warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.abspath("."))

import numpy as np, pandas as pd, cv2
import matplotlib.pyplot as plt
import ipywidgets as W
from IPython.display import display, Video, Markdown
from scipy import ndimage as ndi

from limbusflow import io, register as rg, vesselness as vs, graph as gr, spline as spl
from limbusflow import morphometry as mo, velocity as ve, network as nw, viz, validate as va

# ---------------------------------------------------------------- configuration
def find_data(name, bases=(".", os.path.join("..", "..", "reference_data"))):
    # look next to the notebook first, then in the repository's reference_data/ folder
    for base in bases:
        if os.path.exists(os.path.join(base, name)):
            return os.path.join(base, name)
    return name

REFERENCE = find_data("mean_stabilized.tif")
BURST     = find_data("burst_2026-09-16_15-22-26")
CACHE, OUT = "cache", "outputs"
os.makedirs(CACHE, exist_ok=True); os.makedirs(OUT, exist_ok=True)

# Spatial calibration. The sensor pixel pitch is 5.86 um (IMX249), but the optical
# magnification of the imaging system is not recorded in the manifest, so all
# lengths are reported in PIXELS unless you set this (e.g. 5.86 / magnification).
UM_PER_PX = None
RECOMPUTE_REGISTRATION = False       # True: re-register all 1105 frames (~5 min)

plt.rcParams.update({"figure.dpi": 90, "axes.titlesize": 10, "font.size": 9})
pd.set_option("display.width", 200, "display.max_columns", 40, "display.precision", 3)
""")

# =============================================================================
M(r"""
## 1 · The data

The recorder writes one TIFF per frame plus per-frame *chunk data* straight from the camera:
a hardware timestamp (ns), the exposure time and the gain. Auto-exposure/auto-gain were **on**, so
the overall brightness can step between frames; blinks and tear-film reflections produce outliers.
Everything that follows must be robust to these.
""")

C(r"""
ref, ref_valid = io.load_reference(REFERENCE)
burst = io.load_burst(BURST, cache_dir=CACHE)
cam = burst.manifest.get("camera", {}); cap = burst.manifest.get("capture", {})
print(f"reference   : {ref.shape}, {ref_valid.mean()*100:.1f}% valid pixels (NaN border from stabilisation)")
print(f"burst       : {len(burst)} frames of {burst.frames.shape[1:]} uint16 (12-bit), camera {cam.get('vendor')} {cam.get('model')}")
print(f"frame rate  : {burst.fps:.2f} fps from camera timestamps; duration {burst.t[-1]:.2f} s; dropped {cap.get('frames_dropped')}")
print(f"exposure    : {burst.exposure_s.min()*1e3:.2f}-{burst.exposure_s.max()*1e3:.2f} ms  "
      f"(= {burst.exposure_s.mean()*burst.fps*100:.0f}% of the frame period -> expect motion blur)")

fig, ax = plt.subplots(1, 2, figsize=(15, 2.8), gridspec_kw=dict(width_ratios=[3, 1]))
viz.show(ax[0], ref, "reference image (mean_stabilized.tif)")
ax[1].imshow(ref_valid, cmap="gray"); ax[1].set_title("valid-pixel mask"); ax[1].axis("off")
plt.show()
""")

C(r"""
t = burst.t
mean_int = np.array([np.asarray(f)[::8, ::8].mean() for f in burst.frames])
fig, ax = plt.subplots(1, 4, figsize=(16, 2.8))
ax[0].hist(np.diff(t) * 1e3, 50); ax[0].set_xlabel("frame interval (ms)"); ax[0].set_title("timestamp jitter")
ax[1].plot(t, burst.exposure_s * 1e3); ax[1].set_xlabel("t (s)"); ax[1].set_ylabel("exposure (ms)"); ax[1].set_title("auto exposure")
ax[2].plot(t, burst.gain_db); ax[2].set_xlabel("t (s)"); ax[2].set_ylabel("gain (dB)"); ax[2].set_title("auto gain")
ax[3].plot(t, mean_int); ax[3].set_xlabel("t (s)"); ax[3].set_ylabel("mean DN"); ax[3].set_title("mean intensity (dips = blinks)")
plt.tight_layout(); plt.show()
""")

M(r"""
Scrub through the raw video. The eye drifts by hundreds of pixels, makes saccades, blinks, and some
frames are motion-blurred. (Sliders work in a live kernel; the static figure below shows a few frames.)
""")

C(r"""
fig, axs = plt.subplots(2, 3, figsize=(16, 3.6))
for a, i in zip(axs.ravel(), [40, 240, 480, 640, 840, 1080]):
    viz.show(a, burst.frames[i], f"frame {i}  (t = {t[i]:.2f} s)")
plt.tight_layout(); plt.show()

@W.interact(i=W.IntSlider(0, 0, len(burst) - 1, 1, description="frame", continuous_update=False,
                          layout=W.Layout(width="600px")))
def _scrub(i):
    fig, a = plt.subplots(figsize=(15, 4)); viz.show(a, burst.frames[i], f"raw frame {i}, t = {t[i]:.3f} s"); plt.show()
""")

# =============================================================================
M(r"""
## 2 · Registration: map every frame onto the reference

Every measurement is made in the reference coordinate system, so each frame $F_t$ needs a transform
$T_t$ with $F_t(\mathbf{x}) \approx R(T_t\mathbf{x})$. Over this small field the globe is close to flat, so a
**similarity transform** (4 degrees of freedom) is enough:

$$
T_t:\;\begin{pmatrix}x_R\\y_R\end{pmatrix}=s\begin{pmatrix}\cos\theta&-\sin\theta\\ \sin\theta&\cos\theta\end{pmatrix}\begin{pmatrix}x_F\\y_F\end{pmatrix}+\begin{pmatrix}t_x\\t_y\end{pmatrix}
$$

1. **Flat-field + stretch.** Divide by a heavily blurred copy, $\tilde I = I/(G_{25}*I)$, so uneven illumination does not dominate.
2. **SIFT keypoints.** Blob-like extrema of the difference-of-Gaussians scale space, found on vessel bends and branch points, each with a 128-D gradient-histogram descriptor.
3. **Matching.** Nearest neighbour in descriptor space, kept only if it is clearly better than the 2nd-nearest (Lowe ratio < 0.75).
4. **RANSAC.** Repeatedly fit $T$ to 2 random matches, count *inliers* within 3 px, keep the best consensus and refit on all inliers. Wrong matches cannot pull the solution.
5. **ECC refinement.** Maximise the enhanced correlation coefficient
   $\rho(T)=\frac{\langle \bar R,\ \bar F\circ T\rangle}{\|\bar R\|\,\|\bar F\circ T\|}$ (zero-mean, band-passed images) over a full affine $T$ by Gauss–Newton iterations, for sub-pixel accuracy.
6. **Score.** The NCC between the warped frame and the reference; frames with few inliers, low NCC or implausible scale/rotation are rejected (blinks, heavy blur).
""")

C(r"""
R = rg.Registrar(ref, ref_valid)
i_demo = 600
frame = np.asarray(burst.frames[i_demo])
A, kp, des = R.keypoints(frame)
good = R.match(des)
M0, inl = R.ransac(kp, good)
print(f"frame {i_demo}: {len(kp)} keypoints, {len(good)} ratio-test matches, {inl.sum()} RANSAC inliers")

fig, ax = plt.subplots(2, 2, figsize=(16, 5))
viz.show(ax[0, 0], frame, "raw frame"); viz.show(ax[0, 1], A, "flat-fielded, 8-bit, half resolution (SIFT input)")
ax[1, 0].imshow(R.Rs, cmap="gray"); ax[1, 0].scatter(*np.array([k.pt for k in R.kr]).T, s=2, c="lime")
ax[1, 0].set_title(f"reference keypoints ({len(R.kr)})"); ax[1, 0].axis("off")
# matches: draw inliers (green) and outliers (red) as lines from frame position to reference position
ax[1, 1].imshow(R.Rs, cmap="gray", alpha=0.9)
for g, ok in zip(good, inl):
    p = np.array(kp[g.queryIdx].pt); q = np.array(R.kr[g.trainIdx].pt)
    ax[1, 1].plot([p[0], q[0]], [p[1], q[1]], "-", color="lime" if ok else "red", lw=0.6, alpha=0.8)
ax[1, 1].set_title("match vectors frame → reference: green = RANSAC inliers, red = rejected"); ax[1, 1].axis("off")
plt.tight_layout(); plt.show()
s_, th_ = np.hypot(M0[0, 0], M0[1, 0]), np.degrees(np.arctan2(M0[1, 0], M0[0, 0]))
print(f"RANSAC similarity: scale {s_:.4f}, rotation {th_:+.3f} deg, translation ({M0[0,2]:+.1f}, {M0[1,2]:+.1f}) px")
""")

C(r"""
M1 = R.ecc_refine(frame, M0)
def composite(Mx):
    w = rg.warp(R._bandpass(frame), Mx, ref.shape, border=0)
    a, b = viz.stretch(R.Rbp, 2, 98), viz.stretch(w, 2, 98)
    return np.dstack([a, b, b])            # reference = red, frame = cyan; aligned -> grey
fig, ax = plt.subplots(2, 1, figsize=(16, 5))
ax[0].imshow(composite(M0)[:, 500:1300]); ax[0].set_title(f"after RANSAC (NCC {R.ncc(frame, M0)[0]:.4f})  red = reference, cyan = frame"); ax[0].axis("off")
ax[1].imshow(composite(M1)[:, 500:1300]); ax[1].set_title(f"after ECC refinement (NCC {R.ncc(frame, M1)[0]:.4f})"); ax[1].axis("off")
plt.tight_layout(); plt.show()
print("change of the transform by ECC (px at the image corners):",
      np.round(np.abs((M1 - M0) @ np.array([[0, 1919, 0, 1919], [0, 0, 499, 499], [1, 1, 1, 1]])).max(), 2))
""")

M(r"""
### Whole burst
Registering all 1105 frames takes a few minutes (8 threads), so the result is cached in `cache/registration.npz`.
Set `RECOMPUTE_REGISTRATION = True` to redo it.
""")

C(r"""
reg_path = os.path.join(CACHE, "registration.npz")
if RECOMPUTE_REGISTRATION or not os.path.exists(reg_path):
    t0 = time.time(); reg = R.register_burst(burst.frames); reg.save(reg_path); print(f"{time.time()-t0:.0f} s")
else:
    reg = rg.Registration.load(reg_path)
tab = reg.table
runs = reg.good_runs()
print(f"good frames: {tab.good.sum()} / {len(tab)}")
print("longest runs of consecutive good frames:", [(a, b, f"{(b-a+1)/burst.fps:.2f}s") for a, b in runs[:5]])

fig, ax = plt.subplots(4, 1, figsize=(15, 8), sharex=True)
bad = ~tab.good.values
for a in ax:
    a.fill_between(t, 0, 1, where=bad, transform=a.get_xaxis_transform(), color="red", alpha=0.12, lw=0)
ax[0].plot(t, np.where(tab.good, tab.tx, np.nan), label="t_x"); ax[0].plot(t, np.where(tab.good, tab.ty, np.nan), label="t_y")
ax[0].set_ylabel("translation (px)"); ax[0].legend()
ax[1].plot(t, np.where(tab.good, tab.rot_deg, np.nan)); ax[1].set_ylabel("rotation (deg)")
ax[2].plot(t, tab.ncc); ax[2].set_ylabel("NCC"); ax[2].set_ylim(0, 1)
ax[3].plot(t, tab.n_inlier); ax[3].set_ylabel("RANSAC inliers"); ax[3].set_xlabel("time (s)")
a, b = runs[0]; ax[0].axvspan(t[a], t[b], color="green", alpha=0.1, label="velocity window")
ax[0].set_title("eye motion recovered by registration (red = rejected frames, green = window used for velocimetry)")
plt.tight_layout(); plt.show()

idx = np.arange(runs[0][0], runs[0][1] + 1)        # the longest unbroken run is used for velocimetry
print(f"velocity window: frames {idx[0]}-{idx[-1]} ({len(idx)} frames, {len(idx)/burst.fps:.2f} s)")
""")

C(r"""
@W.interact(i=W.IntSlider(int(idx[100]), 0, len(burst) - 1, 1, description="frame", continuous_update=False,
                          layout=W.Layout(width="600px")))
def _reg_view(i):
    fig, ax = plt.subplots(1, 2, figsize=(16, 2.8))
    viz.show(ax[0], burst.frames[i], f"raw frame {i}")
    if np.isfinite(reg.M[i]).all():
        w = rg.warp(R._bandpass(np.asarray(burst.frames[i])), reg.M[i], ref.shape, border=0)
        ax[1].imshow(np.dstack([viz.stretch(R.Rbp, 2, 98), viz.stretch(w, 2, 98), viz.stretch(w, 2, 98)]))
    ax[1].set_title(f"registered (cyan) over reference (red); good = {bool(tab.good[i])}, NCC = {tab.ncc[i]:.3f}")
    ax[1].axis("off"); plt.show()
""")

M(r"""
### Validation 1: can we rebuild the reference?
If the transforms are right, the mean of the registered good frames should reproduce `mean_stabilized.tif`
(which the recorder made independently). We also re-measure the **residual local misalignment** by
phase-correlating 128-px patches of registered frames against the reference.
""")

C(r"""
our_mean, count = rg.registered_mean(burst.frames, reg)
ok = np.isfinite(our_mean) & ref_valid & (count > 20)
ncc_means = np.corrcoef(R._bandpass(np.nan_to_num(our_mean, nan=np.nanmean(our_mean)))[ok], R.Rbp[ok])[0, 1]
diff = (our_mean / np.nanmedian(our_mean) - ref / np.median(ref))
fig, ax = plt.subplots(3, 1, figsize=(15, 7))
viz.show(ax[0], np.nan_to_num(our_mean, nan=np.nanmean(our_mean)), f"our registered mean of {tab.good.sum()} frames")
viz.show(ax[1], ref, "provided mean_stabilized.tif")
im = ax[2].imshow(np.where(ok, diff, np.nan), cmap="RdBu_r", vmin=-0.08, vmax=0.08); ax[2].axis("off")
ax[2].set_title(f"relative difference (band-passed NCC = {ncc_means:.4f})"); plt.colorbar(im, ax=ax[2], fraction=0.01)
plt.tight_layout(); plt.show()

def residual_shifts(i, patch=128):
    w = rg.warp(R._bandpass(np.asarray(burst.frames[i])), reg.M[i], ref.shape, border=0)
    out = []
    for y in range(40, 500 - patch, 96):
        for x in range(40, 1920 - patch, 128):
            a, b = R.Rbp[y:y+patch, x:x+patch], w[y:y+patch, x:x+patch]
            if (b == 0).mean() > 0.05: continue
            (dx, dy), r = cv2.phaseCorrelate(a.astype(np.float64), b.astype(np.float64),
                                            cv2.createHanningWindow((patch, patch), cv2.CV_64F))
            if r > 0.2: out.append(np.hypot(dx, dy))
    return out
res = np.concatenate([residual_shifts(i) for i in idx[::20]])
print(f"residual local misalignment in the velocity window: median {np.median(res):.2f} px, 90th pct {np.percentile(res, 90):.2f} px")
""")

# =============================================================================
M(r"""
## 3 · Vessel enhancement: the Hessian and Frangi vesselness

Near a point $\mathbf{x}_0$, the second-order Taylor expansion of the image is
$I(\mathbf{x}_0+\mathbf{d}) \approx I + \mathbf{d}^\top\nabla I + \tfrac12\mathbf{d}^\top H\,\mathbf{d}$ with the **Hessian**
$H=\begin{pmatrix}I_{xx}&I_{xy}\\I_{xy}&I_{yy}\end{pmatrix}$. Its eigenvalues $|\lambda_1|\le|\lambda_2|$ are the principal curvatures of the
intensity landscape, and its eigenvectors give their directions.

A **dark vessel on a bright background is a valley**: strong positive curvature *across* it ($\lambda_2 \gg 0$) and almost none *along* it ($\lambda_1\approx 0$).
Derivatives are computed at scale $\sigma$ by convolving with derivatives of a Gaussian, and multiplied by $\sigma^2$ so that responses at different scales are comparable.
The Frangi measure combines

$$
R_b=\frac{\lambda_1}{\lambda_2}\ (\text{blob vs line}),\qquad S=\sqrt{\lambda_1^2+\lambda_2^2}\ (\text{structure vs flat}),\qquad
V_\sigma = e^{-R_b^2/2\beta^2}\left(1-e^{-S^2/2c^2}\right)\ \ [\lambda_2>0],
$$

taking the maximum over scales, $V=\max_\sigma V_\sigma$.
""")

C(r"""
P = nw.Params()
net = nw.Network(ref=ref, valid=ref_valid, params=P, um_per_px=UM_PER_PX)
net.compute_flicker(burst.frames, reg, idx)      # temporal flicker map, explained in section 3b
net.enhance(keep_stack=True)
fig, ax = plt.subplots(2, 1, figsize=(15, 5))
viz.show(ax[0], ref, "reference: strong illumination gradient")
viz.show(ax[1], net.ff, r"flat-fielded  $I / (G_{30} * I)$")
plt.tight_layout(); plt.show()
""")

C(r"""
ys, xs = slice(120, 330), slice(560, 900)          # a zoom with branches and a crossing
crop = net.ff[ys, xs]
sig = 3.0
Ixx, Ixy, Iyy = vs.hessian(crop, sig)
l1, l2, vx, vy = vs.hessian_eigen(Ixx, Ixy, Iyy)
fig, ax = plt.subplots(2, 4, figsize=(16, 6))
for a, im, ttl in zip(ax[0], [crop, Ixx, Ixy, Iyy], ["crop", r"$\sigma^2 I_{xx}$", r"$\sigma^2 I_{xy}$", r"$\sigma^2 I_{yy}$"]):
    a.imshow(im, cmap="gray" if ttl == "crop" else "RdBu_r"); a.set_title(ttl); a.axis("off")
m = np.abs(l2).max()
ax[1, 0].imshow(l1, cmap="RdBu_r", vmin=-m, vmax=m); ax[1, 0].set_title(r"$\lambda_1$ (along vessel) ≈ 0")
ax[1, 1].imshow(l2, cmap="RdBu_r", vmin=-m, vmax=m); ax[1, 1].set_title(r"$\lambda_2$ (across) > 0 in dark vessels")
ax[1, 2].imshow(vs.frangi_single(l1, l2, 0.5, P.frangi_c), cmap="magma"); ax[1, 2].set_title(rf"$V_\sigma$ at $\sigma$={sig}")
ax[1, 3].imshow(crop, cmap="gray"); step = 8
yy, xx = np.mgrid[0:crop.shape[0]:step, 0:crop.shape[1]:step]
msk = (l2[::step, ::step] > 0.3 * m)
ax[1, 3].quiver(xx[msk], yy[msk], vx[::step, ::step][msk], vy[::step, ::step][msk], color="yellow", angles="xy",
                scale_units="xy", scale=1 / 7, width=0.006, headwidth=0, headlength=0, headaxislength=0, pivot="mid")
ax[1, 3].set_title("eigenvector of λ₁ = local vessel direction")
for a in ax[1]: a.axis("off")
plt.tight_layout(); plt.show()
""")

C(r"""
st = net.frangi_stack
fig, ax = plt.subplots(2, 4, figsize=(16, 3.6))
for a, s, V in zip(ax.ravel(), st["sigmas"], st["V"]):
    a.imshow(V, cmap="magma"); a.set_title(rf"$V_\sigma$, $\sigma$ = {s}"); a.axis("off")
Vmax = st["V"].max(0); best_sigma = st["sigmas"][st["V"].argmax(0)]
ax.ravel()[-2].imshow(Vmax, cmap="magma"); ax.ravel()[-2].set_title(r"$V=\max_\sigma V_\sigma$"); ax.ravel()[-2].axis("off")
ax.ravel()[-1].imshow(np.where(Vmax > 0.3, best_sigma, np.nan), cmap="viridis"); ax.ravel()[-1].axis("off")
ax.ravel()[-1].set_title(r"arg-max $\sigma$ (∝ vessel radius)")
plt.tight_layout(); plt.show()
""")

C(r"""
@W.interact(sigma=W.FloatSlider(3, min=1, max=10, step=0.5, continuous_update=False),
            beta=W.FloatSlider(0.5, min=0.1, max=2, step=0.1, continuous_update=False),
            c=W.FloatLogSlider(0.02, base=10, min=-3, max=0, continuous_update=False))
def _frangi_play(sigma, beta, c):
    l1, l2, _, _ = vs.hessian_eigen(*vs.hessian(crop, sigma))
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
    ax[0].imshow(crop, cmap="gray"); ax[0].axis("off")
    ax[1].imshow(vs.frangi_single(l1, l2, beta, c), cmap="magma", vmin=0, vmax=1); ax[1].axis("off")
    ax[1].set_title(f"V(σ={sigma}, β={beta}, c={c:.3g}) — small σ: capillaries; large σ: big vessels")
    plt.show()
""")

# =============================================================================
M(r"""
## 3b · Functional detection: where does the video *flicker*?

Many thin vessels are barely visible in the mean image, but blood moving through them makes their lumen **flicker**:
red-cell aggregates and plasma gaps change the intensity from frame to frame, while static tissue does not.
The temporal standard deviation of the registered video is a map of **perfused** vessels. This is the same idea as motion-contrast angiography (OCT-A).

For a pixel $\mathbf x$ of the registered, brightness-normalised video,

$$F_t(\mathbf x)=m(\mathbf x)+\underbrace{b_t(\mathbf x)}_{\text{blood flicker}}+\underbrace{\nabla m(\mathbf x)\cdot\boldsymbol\delta_t}_{\text{residual jitter}}+\underbrace{n_t(\mathbf x)}_{\text{sensor noise}} .$$

After removing slow changes (a running temporal mean, $\sigma_t$ = 10 frames), the variance is

$$\operatorname{var}_t F \approx \operatorname{var}(b) + \sigma_\delta^2\,|\nabla m|^2 + a + c\,m .$$

The **jitter** term (strong at the edges of big vessels) and the **noise** terms are fitted by robust least squares over background pixels and subtracted:

$$\text{flicker}(\mathbf x)=\frac{\sqrt{\max\big(\operatorname{var}_t F-\hat\sigma_\delta^2|\nabla m|^2-\hat a-\hat c\,m,\;0\big)}}{m}.$$

The fitted $\hat\sigma_\delta$ is the residual misregistration. It is a free cross-check against the patch-based measurement in section 2.
Running means use *normalised convolution*, $(G*(Fw))/(G*w)$ with $w=0$ where a pixel was out of view or specular, so partly covered pixels are not biased.
""")

C(r"""
fp_ = net.fl_parts
cover_ok = fp_["cover"] >= P.flicker_min_cover
raw_fl = np.where(cover_ok, np.sqrt(fp_["var"]) / np.maximum(fp_["mean"], 1e-6), 0)
bg = fp_["bg_mask"]
fig = plt.figure(figsize=(16, 8.5)); g = fig.add_gridspec(3, 3, width_ratios=[3, 3, 1.6])
a = fig.add_subplot(g[0, :2]); a.imshow(viz.stretch(raw_fl, 1, 99.5), cmap="gray"); a.axis("off")
a.set_title("raw temporal std / mean: big-vessel edges and jitter dominate")
a = fig.add_subplot(g[1, :2]); a.imshow(viz.stretch(net.fl, 1, 99.5), cmap="gray"); a.axis("off")
a.set_title("jitter- and noise-corrected flicker map = perfused vessels")
a = fig.add_subplot(g[2, :2]); viz.show(a, net.ff, "the mean image, for comparison")
a = fig.add_subplot(g[0:2, 2])
rs = np.random.default_rng(0).choice(np.flatnonzero(bg), 20000, replace=False)
a.scatter(fp_["grad2"].ravel()[rs], fp_["var"].ravel()[rs], s=1, alpha=0.3, label="background pixels")
gg = np.linspace(0, np.percentile(fp_["grad2"][bg], 99.5), 50); co = fp_["coef"]
a.plot(gg, co["a"] + co["c"] * np.median(fp_["mean"][bg]) + co["sigma_delta2"] * gg, "r-", lw=2,
       label=rf"fit: $\hat\sigma_\delta$ = {np.sqrt(co['sigma_delta2']):.2f} px")
a.set_xlim(0, gg[-1]); a.set_ylim(0, np.percentile(fp_["var"][bg], 99.5))
a.set_xlabel(r"$|\nabla m|^2$"); a.set_ylabel(r"var$_t$ F"); a.legend(fontsize=8); a.set_title("jitter model on background")
a = fig.add_subplot(g[2, 2]); a.imshow(fp_["cover"], cmap="viridis", vmin=0, vmax=1); a.axis("off"); a.set_title("fraction of frames in view")
plt.tight_layout(); plt.show()
""")

C(r"""
fig, ax = plt.subplots(2, 3, figsize=(16, 5.5))
for j, (ys_, xs_) in enumerate([(slice(250, 480), slice(440, 720)), (slice(120, 470), slice(1020, 1300)), (slice(60, 330), slice(20, 330))]):
    ax[0, j].imshow(viz.stretch(net.ff[ys_, xs_]), cmap="gray"); ax[0, j].set_title("mean image"); ax[0, j].axis("off")
    ax[1, j].imshow(viz.stretch(net.fl[ys_, xs_], 1, 99.5), cmap="gray"); ax[1, j].set_title("flicker map"); ax[1, j].axis("off")
plt.tight_layout(); plt.show()
""")

M(r"""
### Combining structure and function
Two vesselness maps, each normalised to [0, 1] by its 99.5th percentile, are combined by their maximum:
* **structural**: dark ridges of the flat-fielded mean (Frangi, section 3), now including $\sigma$ = 1 px for capillaries;
* **functional**: *bright* ridges of the flicker map (the same Hessian analysis with the sign flipped).

Residual edge flicker makes thin parallel ridges *inside* thick vessels. These are not separate vessels, so the functional term is ignored inside thick structural vessels
(the structural mask opened by a disc of radius 4 px). The band is deliberately not dilated, so thin branches keep their connection to the parent vessel.
""")

C(r"""
fig, ax = plt.subplots(4, 1, figsize=(16, 13))
viz.show(ax[0], net.V_struct, "structural vesselness (mean image)", cmap="magma")
viz.show(ax[1], net.V_func, "functional vesselness (flicker map)", cmap="magma")
ax[2].imshow(net.thick, cmap="gray"); ax[2].axis("off"); ax[2].set_title("thick vessels: functional term ignored here")
viz.show(ax[3], net.V, "combined  V = max(V_struct, V_func outside thick vessels)", cmap="magma")
plt.tight_layout(); plt.show()
""")

# =============================================================================
M(r"""
## 4 · Segmentation and skeleton

**Hysteresis thresholding.** Pixels with $V$ above a *high* threshold seed vessels; connected pixels above a *low*
threshold are added. Weak but connected vessel parts are kept while isolated noise is rejected. Small islands are removed and small holes filled.

**Skeletonisation.** Iterative thinning reduces the mask to a 1-px-wide, 8-connected centre-line that keeps its topology.
""")

C(r"""
net.segment()
lo, hi = net.seg_thresholds
fig = plt.figure(figsize=(16, 6.5)); g = fig.add_gridspec(2, 3, width_ratios=[1, 3, 3])
a = fig.add_subplot(g[:, 0])
a.hist(net.V[net.valid_e].ravel(), 200, log=True, color="0.5"); a.axvline(lo, c="C0", label="low"); a.axvline(hi, c="C3", label="high")
a.set_xlabel("vesselness V"); a.legend(); a.set_title("hysteresis thresholds")
a = fig.add_subplot(g[0, 1:]); viz.show(a, net.ff); a.imshow(np.ma.masked_where(~net.mask, net.mask), cmap="autumn", alpha=0.45)
a.set_title(f"vessel mask ({net.mask.mean()*100:.1f}% of pixels)")
a = fig.add_subplot(g[1, 1:]); viz.show(a, net.ff)
yy, xx = np.nonzero(net.skel); a.scatter(xx, yy, s=0.3, c="yellow"); a.set_title(f"skeleton ({net.skel.sum()} px)")
plt.tight_layout(); plt.show()

# what did the functional channel add? structural-only segmentation for comparison
m_struct, _ = vs.segment(net.V_struct, net.valid_e, P.comb_low_pct, P.comb_high_pct, P.comb_min_size)
sk_struct = vs.skeletonize(m_struct)
added = net.skel & ~ndi.binary_dilation(sk_struct, iterations=4)
print(f"centre-line length: structural only {sk_struct.sum()} px, structural + functional {net.skel.sum()} px "
      f"(+{added.sum()} px of centre-line found only through flicker)")
fig, a = plt.subplots(figsize=(16, 4.3)); viz.show(a, net.ff)
yy, xx = np.nonzero(sk_struct); a.scatter(xx, yy, s=0.3, c="cyan", label="found in the mean image")
yy, xx = np.nonzero(added); a.scatter(xx, yy, s=0.6, c="orangered", label="found only through flicker")
a.legend(loc="lower right", markerscale=12); a.set_title("vessels added by functional detection"); plt.show()
""")

C(r"""
nb = gr.classify_pixels(net.skel)
zy, zx = slice(0, 200), slice(180, 460)
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
viz.show(ax[0], net.ff[zy, zx], "zoom")
ax[1].imshow(net.ff[zy, zx], cmap="gray")
for val, c, lab in [(1, "lime", "n=1 end point"), (2, "yellow", "n=2 interior"), (3, "red", "n≥3 junction")]:
    m = (nb[zy, zx] == val) if val < 3 else (nb[zy, zx] >= 3)
    yy, xx = np.nonzero(m); ax[1].scatter(xx, yy, s=4 if val == 2 else 18, c=c, label=lab)
ax[1].legend(loc="lower right"); ax[1].axis("off"); ax[1].set_title("pixel classes by 8-neighbour count n(p)")
plt.tight_layout(); plt.show()
""")

# =============================================================================
M(r"""
## 5 · From pixels to a graph

* **Nodes**: clusters of junction pixels (8-connected) and end points.
* **Edges (segments)**: pixel chains between nodes, traced in order.

Clean-up, iterated until nothing changes:
1. **spur pruning**: a dangling segment shorter than $\max(15\,\text{px},\ 2\,r_{\text{parent}})$ is a thinning artefact from a bumpy vessel edge, because a "branch" that barely reaches past the wall of its parent is not a vessel;
2. **degree-2 merge**: a node left with exactly two segments is no longer a junction, so concatenate them;
3. **junction merge**: two junctions joined by a segment shorter than $\max(10\,\text{px},\ r_u+r_v)$ ($r$ = local vessel radius from the distance transform of the mask) are one branch point, or one crossing of two thick vessels, split by thinning into two T-junctions;
4. **crossings**: in the limbus, conjunctival vessels pass *over* episcleral ones. In 2-D this looks like a 4-way junction, but no blood is exchanged.
   If a degree-4 node's segments form two nearly straight pairs (outgoing tangents within 35° of anti-parallel), the node is split and each vessel passes straight through;
5. **border nodes**: end points near the edge of the field of view are where blood enters or leaves the image.
""")

C(r"""
G_raw = gr.skeleton_to_graph(net.skel)
net.build_graph()
G = net.graph
kinds = pd.Series([d["kind"] for d in G.nodes.values()]).value_counts()
n_cross = sum(len(d.get("crossings", [])) for d in G.edges.values())
print(f"raw graph  : {len(G_raw.nodes)} nodes, {len(G_raw.edges)} segments")
print(f"clean graph: {len(G.nodes)} nodes, {len(G.edges)} segments, {n_cross} crossings resolved;", dict(kinds))

def draw_graph(ax, Gx, title):
    viz.show(ax, net.ff, title)
    rng = np.random.default_rng(3)
    for e, d in Gx.edges.items():
        ax.plot(d["path"][:, 0], d["path"][:, 1], lw=1.5, color=plt.cm.tab20(rng.integers(20)))
        for cx, cy in d.get("crossings", []):
            ax.scatter(cx, cy, marker="x", s=60, c="magenta", zorder=6)
    col = dict(junction="red", end="lime", border="deepskyblue", loop="orange")
    for n, d in Gx.nodes.items():
        ax.scatter(*d["pos"], s=14, c=col.get(d["kind"], "w"), zorder=5, edgecolors="k", linewidths=0.3)
fig, ax = plt.subplots(2, 1, figsize=(16, 9))
draw_graph(ax[0], G_raw, "raw skeleton graph (every thinning spur is a segment)")
draw_graph(ax[1], G, "cleaned graph: red junction, green end, blue border (leaves FOV), magenta × crossing")
plt.tight_layout(); plt.show()
""")

C(r"""
# illustrate the crossing test on the first crossing found
ecr = next(e for e, d in G.edges.items() if d.get("crossings"))
cx, cy = G.edges[ecr]["crossings"][0]
near = [e for e, d in G.edges.items() if np.min(np.hypot(d["path"][:, 0] - cx, d["path"][:, 1] - cy)) < 3]
fig, ax = plt.subplots(figsize=(6, 5))
ax.imshow(net.ff, cmap="gray"); ax.set_xlim(cx - 60, cx + 60); ax.set_ylim(cy + 50, cy - 50)
for e in near:
    p = G.edges[e]["path"]; k = np.argmin(np.hypot(p[:, 0] - cx, p[:, 1] - cy))
    ax.plot(p[:, 0], p[:, 1], lw=2)
    for j in (k - 15, k + 15):
        if 0 <= j < len(p):
            v = p[j] - p[k]; ax.annotate("", xy=p[k] + v, xytext=p[k], arrowprops=dict(arrowstyle="->", color="yellow", lw=1.5))
ax.set_title("crossing: two straight-through pairs of tangents → split into 2 vessels"); ax.axis("off"); plt.show()
""")

# =============================================================================
M(r"""
## 6 · Centre-lines as smoothing splines

A skeleton segment is an ordered chain of integer pixel centres $\mathbf p_0,\dots,\mathbf p_{N-1}$. It is jagged
(a staircase) and cannot be differentiated. We replace it by a **parametric cubic B-spline**

$$\mathbf r(u)=\sum_i \mathbf c_i\,B_{i,3}(u),\qquad u\in[0,1],$$

whose control points $\mathbf c_i$ and knots are chosen to minimise roughness subject to
$\sum_j w_j^2\,\|\mathbf r(u_j)-\mathbf p_j\|^2 \le s$, with $s = N\sigma^2$ ($\sigma\approx 0.8$ px, the expected skeleton jitter).
The end points get larger weights so the spline stays attached to the graph nodes.

**Arc-length re-parameterisation.** $u$ is not proportional to distance. We integrate the speed
$s(u)=\int_0^u\|\mathbf r'(v)\|\,dv$, invert it numerically and resample every 1 px along the curve.

**Frenet frame and curvature.**
$\mathbf T=\mathbf r'/\|\mathbf r'\|$, $\ \mathbf N = (-T_y, T_x)$, $\ \kappa=\dfrac{x'y''-y'x''}{\|\mathbf r'\|^3}$ (signed, 1/px).
""")

C(r"""
net.fit_splines()
print(f"{len(net.vessels)} vessels (segments ≥ {P.min_len_measure} px), numbered by decreasing length")
cand = [v for v in net.vessels.values() if v.length > 150]
VID_T = max(cand, key=lambda v: np.nan_to_num(mo.tortuosity(v.spline)["DM"])).id      # a tortuous demo vessel
ves = net.vessels[VID_T]; sp = ves.spline
fig, ax = plt.subplots(1, 2, figsize=(16, 5))
for a in ax:
    a.imshow(net.ff, cmap="gray"); viz.zoom_to(a, net, VID_T, 25)
ax[0].plot(sp.raw[:, 0], sp.raw[:, 1], "s", ms=3, mfc="none", color="orange", label="skeleton pixels")
ax[0].plot(sp.xy[:, 0], sp.xy[:, 1], "c-", lw=1.5, label="smoothing spline r(s)")
cp = sp.control_points; ax[0].plot(cp[:, 0], cp[:, 1], "m.--", lw=0.6, ms=6, label=f"{len(cp)} control points")
ax[0].legend(); ax[0].set_title(f"{VID_T}: pixel chain vs spline")
k = 12
ax[1].plot(sp.xy[:, 0], sp.xy[:, 1], "c-")
ax[1].quiver(sp.xy[::k, 0], sp.xy[::k, 1], sp.T[::k, 0], sp.T[::k, 1], color="yellow", angles="xy", scale=18, label="T")
ax[1].quiver(sp.xy[::k, 0], sp.xy[::k, 1], sp.N[::k, 0], sp.N[::k, 1], color="red", angles="xy", scale=18, label="N")
ax[1].legend(); ax[1].set_title("Frenet frame every 12 px of arc length")
plt.tight_layout(); plt.show()
""")

C(r"""
@W.interact(sigma=W.FloatSlider(0.8, min=0.1, max=4, step=0.1, description="σ (px)", continuous_update=False))
def _smooth_play(sigma):
    s2 = spl.fit_spline(sp.raw, sigma=sigma)
    fig, ax = plt.subplots(1, 2, figsize=(16, 3.5))
    ax[0].imshow(net.ff, cmap="gray"); viz.zoom_to(ax[0], net, VID_T, 25)
    ax[0].plot(s2.raw[:, 0], s2.raw[:, 1], "s", ms=2, mfc="none", color="orange"); ax[0].plot(s2.xy[:, 0], s2.xy[:, 1], "c-")
    ax[0].set_title(f"s = N σ² = {len(s2.raw) * sigma**2:.0f}: {len(s2.control_points)} control points")
    ax[1].plot(s2.s, s2.kappa); ax[1].axhline(0, c="k", lw=0.5); ax[1].set_xlabel("arc length s (px)"); ax[1].set_ylabel("κ (1/px)")
    ax[1].set_title("curvature: too little smoothing → pixel-staircase noise; too much → flattened bends"); plt.show()
""")

M(r"""
### Validation 2: curvature against an analytic curve
A sine wave $y=A\sin(2\pi x/\lambda)$ has known curvature $\kappa = y''/(1+y'^2)^{3/2}$ and known length.
We rasterise it to integer pixels (as a skeleton would be) and run the same spline fit.
""")

C(r"""
sv = va.sine_curve_test(A=20, lam=160)
s_ = sv["spline"]
fig, ax = plt.subplots(1, 2, figsize=(16, 3.2))
ax[0].plot(sv["pixels"][:, 0], sv["pixels"][:, 1], "s", ms=2, mfc="none", color="orange"); ax[0].plot(s_.xy[:, 0], s_.xy[:, 1], "c-")
ax[0].set_aspect("equal"); ax[0].invert_yaxis(); ax[0].set_title("rasterised sine + fitted spline")
ax[1].plot(s_.xy[:, 0], sv["kappa_true"], "k-", lw=2, label="analytic κ"); ax[1].plot(s_.xy[:, 0], s_.kappa, "C1-", label="spline κ")
ax[1].legend(); ax[1].set_xlabel("x (px)"); ax[1].set_title("curvature")
plt.tight_layout(); plt.show()
rmse = np.sqrt(np.mean((s_.kappa[10:-10] - sv["kappa_true"][10:-10]) ** 2))
print(f"length: spline {s_.length:.1f} px vs analytic {sv['L_true']:.1f} px ({(s_.length/sv['L_true']-1)*100:+.2f}%)")
print(f"DM: spline {s_.length / np.linalg.norm(s_.xy[-1]-s_.xy[0]):.4f} vs analytic {sv['DM_true']:.4f};  κ RMSE {rmse:.4f} 1/px (peak |κ| {np.abs(sv['kappa_true']).max():.4f})")
""")

# =============================================================================
M(r"""
## 7 · Diameter along the vessel

At every arc-length sample we read the image along the normal, $\mathbf q(t)=\mathbf r(s_i)+t\,\mathbf N(s_i)$, $t\in[-h,h]$ (bilinear interpolation),
and average neighbouring profiles (±3 px) to reduce noise. Blood absorbs the light, so a vessel is a **dip**.

* **FWHM (primary, model-free).** Baseline $b$ = median of the tails, depth $A=b-\min P$; the width is the distance between the two crossings of $b-A/2$.
* **Model fit.** A vessel of width $w$ at offset $c$, with edge softness $\sigma$, on a linear background:
  $$P(t)=b_0+b_1t-\tfrac{A}{2}\left[\operatorname{erf}\!\frac{t-c+w/2}{\sqrt2\,\sigma}-\operatorname{erf}\!\frac{t-c-w/2}{\sqrt2\,\sigma}\right]$$
  fitted by non-linear least squares (analytic Jacobian). The fit also re-centres the mask ($c$) and acts as a **shape check**: samples whose profile is not a clean dip ($R^2<0.8$), or that lie near junctions and crossings, are excluded.
""")

C(r"""
VID_D = VID_T
ves = net.vessels[VID_D]; sp = ves.spline
junc = [tuple(n["pos"]) for n in G.nodes.values() if n["kind"] == "junction"]
dm = mo.measure_diameter(net.ff, sp, ves.d_guess, exclude_pts=junc + ves.crossings, exclude_r=0.8 * ves.d_guess + 3)
Pm, tt = dm["profiles"], dm["t"]
fig, ax = plt.subplots(1, 2, figsize=(16, 4.2), gridspec_kw=dict(width_ratios=[1.2, 1]))
ax[0].imshow(net.ff, cmap="gray"); viz.zoom_to(ax[0], net, VID_D, 30)
for i in range(0, len(sp.s), 6):
    a, b = sp.xy[i] - tt[-1] * sp.N[i], sp.xy[i] + tt[-1] * sp.N[i]
    ax[0].plot([a[0], b[0]], [a[1], b[1]], "y-", lw=0.6)
ax[0].plot(sp.xy[:, 0], sp.xy[:, 1], "c-"); ax[0].set_title("profile lines along the normals")
ax[1].imshow(Pm.T, aspect="auto", cmap="gray", extent=[0, sp.length, tt[-1], tt[0]])
ax[1].plot(sp.s, np.where(dm["valid"], dm["d"] / 2, np.nan), "c.", ms=2); ax[1].plot(sp.s, np.where(dm["valid"], -dm["d"] / 2, np.nan), "c.", ms=2)
ax[1].set_xlabel("s (px)"); ax[1].set_ylabel("t across (px)"); ax[1].set_title("straightened vessel P(s, t) with ± FWHM/2")
plt.tight_layout(); plt.show()
""")

C(r"""
@W.interact(i=W.IntSlider(len(sp.s) // 2, 0, len(sp.s) - 1, 1, description="sample s", continuous_update=False,
                          layout=W.Layout(width="600px")))
def _profile(i):
    p = Pm[i]
    wf, cf, dep = mo.fwhm_width(p, tt)
    f = mo.fit_width(p, tt, wf if np.isfinite(wf) else ves.d_guess, cf if np.isfinite(cf) else 0)
    fig, ax = plt.subplots(figsize=(8, 3.3))
    ax.plot(tt, p, "k.", ms=4, label="profile (avg ±3 px)")
    if f["params"] is not None:
        ax.plot(tt, mo.box_blur_model(tt, *f["params"]), "C1-", label=f"model: w={f['w']:.1f}, σ={f['sig']:.1f}, R²={f['r2']:.3f}")
    if np.isfinite(wf):
        k = max(2, len(p) // 5); base = np.median(np.r_[p[:k], p[-k:]])
        ax.axhline(base - dep / 2, color="C0", ls="--", lw=0.8)
        ax.axvspan(cf - wf / 2, cf + wf / 2, color="C0", alpha=0.15, label=f"FWHM = {wf:.1f} px")
    ax.set_xlabel("t (px across)"); ax.legend(fontsize=8); ax.set_title(f"{VID_D}, s = {sp.s[i]:.0f} px, valid = {dm['valid'][i]}")
    plt.show()
""")

C(r"""
net.measure_morphology(progress=False)
fig, ax = plt.subplots(figsize=(14, 3.2))
v_ = net.vessels[VID_D]
ax.plot(v_.spline.s, np.where(v_.diam["valid"], v_.diam["d"], np.nan), ".", ms=3, label="FWHM (valid)")
ax.plot(v_.spline.s, np.where(~v_.diam["valid"], v_.diam["d"], np.nan), ".", ms=2, c="0.7", label="excluded (junction/crossing/outlier)")
ax.plot(v_.spline.s, v_.d_smooth, "k-", label="smoothed d(s) → mask")
ax.set_xlabel("s (px)"); ax.set_ylabel("d (px)"); ax.legend(); ax.set_title(f"{VID_D}: diameter along the vessel"); plt.show()
""")

M(r"""
### Validation 3: diameter phantoms, and why FWHM is the primary estimate
Straight synthetic vessels of known width, blurred ($\sigma$ = 1.5 px) with noise, drawn with two profile shapes:
a **box** (strongly absorbing) and a **projected cylinder** (weakly absorbing, dip ∝ chord length $2\sqrt{r^2-t^2}$).
""")

C(r"""
tb_box = pd.DataFrame(va.diameter_phantom_test(model="box", n_rep=8)).set_index("true_d")
tb_cyl = pd.DataFrame(va.diameter_phantom_test(model="cylinder", n_rep=8)).set_index("true_d")
fig, ax = plt.subplots(1, 2, figsize=(14, 3.6))
for a, tb, name in [(ax[0], tb_box, "box vessel"), (ax[1], tb_cyl, "cylinder vessel")]:
    d = tb.index.values
    a.plot(d, d, "k--", lw=0.8, label="identity")
    a.errorbar(d, tb.fwhm, tb.fwhm_sd, fmt="o-", label="FWHM"); a.errorbar(d, tb.fit, tb.fit_sd, fmt="s-", label="box-model fit")
    a.set_xlabel("true diameter (px)"); a.set_ylabel("measured (px)"); a.set_title(name); a.legend()
plt.tight_layout(); plt.show()
display(pd.concat({"box": tb_box, "cylinder": tb_cyl}, axis=1).round(2))
""")

M(r"""
**Reading the phantom results**
* Box vessels: FWHM is exact from ≈ 6 px up. Below that, blur makes FWHM overestimate while the model fit still tracks the truth.
* Cylinder vessels: FWHM ≈ 0.85·d, the theoretical $\sqrt3/2$ for a semi-ellipse.

On the **real** image both models fit equally well ($R^2$ identical to 3 decimals), with an edge softness $\sigma\approx0.3\,d$ that *grows with vessel size*.
The softness therefore comes mostly from the vessel's own rounded absorption profile and from light scattering in the sclera, not from a fixed optical PSF.
Width and softness trade off, so a model-based width is model-dependent. The cell below fits both models to a real vessel.
FWHM is reported as the primary diameter because it is model-free and monotonic. **Absolute diameters are good to about ±15 %; relative changes are much more reliable.**
""")

C(r"""
from scipy.optimize import least_squares
def cyl_model(t, b0, b1, A, c, w, sg):
    dt = 0.1; u = np.arange(-w / 2 - 4 * sg - 1, w / 2 + 4 * sg + 1 + dt, dt)
    e = ndi.gaussian_filter1d(np.sqrt(np.clip(1 - (2 * u / w) ** 2, 0, None)), max(sg, 1e-3) / dt, mode="constant")
    return b0 + b1 * t - A * np.interp(t - c, u, e, left=0, right=0)
rows = []
for vid in list(net.vessels)[:10]:
    v_ = net.vessels[vid]; Pp, t_ = mo.cross_profiles(net.ff, v_.spline, max(8, 1.6 * v_.d_guess), 0.5); Pp = mo.smooth_along(Pp, 3)
    for i in np.flatnonzero(v_.diam["valid"])[::15]:
        p = Pp[i]; fb = mo.fit_width(p, t_, v_.d_guess)
        b0 = np.median(np.r_[p[:5], p[-5:]])
        rc = least_squares(lambda q: cyl_model(t_, *q) - p, [b0, 0, b0 - p.min(), 0, max(v_.d_guess, 3), 1.0],
                           bounds=([-np.inf, -np.inf, 0, -t_[-1] / 2, 1, 0.3], [np.inf, np.inf, np.inf, t_[-1] / 2, 1.8 * t_[-1], 6]))
        ss = np.sum((p - p.mean()) ** 2)
        rows.append(dict(vessel=vid, fwhm=mo.fwhm_width(p, t_)[0], box_w=fb["w"], box_sigma=fb["sig"], box_R2=fb["r2"],
                         cyl_w=rc.x[4], cyl_sigma=rc.x[5], cyl_R2=1 - np.sum(rc.fun ** 2) / ss))
display(pd.DataFrame(rows).groupby("vessel", sort=False).median().round(3))
""")

# =============================================================================
M(r"""
## 8 · Vessel masks

The outline of each vessel is the centre-line offset by half the smoothed diameter along the normal, re-centred by the fitted offset $c(s)$:
$\ \mathbf r(s)+\big(c(s)\pm d(s)/2\big)\mathbf N(s)$. The polygon is filled (with round end caps) to make the mask.
Masks are stored as polygons, so they stay exact at any zoom.
""")

C(r"""
lab = net.label_image()
rng = np.random.default_rng(0); pal = rng.uniform(0.25, 1, (lab.max() + 1, 3)); pal[0] = 0
fig, ax = plt.subplots(2, 1, figsize=(16, 8.5))
viz.show(ax[0], net.ff); ax[0].imshow(np.dstack([pal[lab], (lab > 0) * 0.55]))
ax[0].set_title(f"all {len(net.vessels)} vessel masks (label image; one colour per vessel)")
viz.show(ax[1], net.ff); viz.draw_vessel(ax[1], net, VID_D); viz.zoom_to(ax[1], net, VID_D, 60)
ax[1].set_title(f"{VID_D}: polygon mask, centre-line and crossings")
plt.tight_layout(); plt.show()
""")

# =============================================================================
M(r"""
## 9 · Tortuosity

| metric | definition | meaning |
|---|---|---|
| **DM** | $L / C$ (arc length / chord) | ≥ 1; 1 = straight |
| **SOAM** | $\frac1L\int|\kappa|\,ds$ | total turning per unit length (rad/px) |
| **κ²** | $\frac1L\int\kappa^2\,ds$ | bending energy per unit length |
| **ICM** | $(n_{\text{infl}}+1)\cdot DM$ | DM weighted by the number of *inflections* (sign changes of κ) |

DM cannot tell one large arc from many small wiggles; SOAM and ICM can. All are computed from the spline, never from the pixel staircase.
""")

C(r"""
T = net.table()
cols = ["length_px", "d_median_px", "DM", "SOAM", "KSQ", "n_inflections", "ICM"]
display(T[cols].sort_values("DM", ascending=False).head(10))
from matplotlib.collections import LineCollection
fig, ax = plt.subplots(figsize=(16, 4.6))
viz.show(ax, net.ff, "centre-lines coloured by |curvature| (1/px)")
segs, cvals = [], []
for v in net.vessels.values():
    xy = v.spline.xy
    segs += [xy[j:j + 2] for j in range(len(xy) - 1)]; cvals += list(np.abs(v.spline.kappa[:-1]))
lc = LineCollection(segs, cmap="plasma", norm=plt.Normalize(0, 0.05), linewidths=2.2); lc.set_array(np.array(cvals))
ax.add_collection(lc); plt.colorbar(lc, ax=ax, fraction=0.015)
plt.show()
""")

# =============================================================================
M(r"""
## 10 · Blood velocity from the video

### 10.1 Kymographs
For every registered frame $t$ we read the intensity along the vessel centre-line, averaged across the central half of the lumen:

$$K(t,s)=\operatorname*{mean}_{|w|<d/4} F_t\big(T_t^{-1}[\mathbf r(s)+w\,\mathbf N(s)]\big).$$

Samples come straight from the **raw** frame through the inverse registration transform, so each pixel is interpolated only once.
Blood is not uniform at this resolution: red-cell aggregates and plasma gaps make the lumen **flicker**. These patterns are carried by the flow,
so in $K$ they draw oblique streaks, $K(t,s)\approx f(s-vt)$, whose **slope $ds/dt$ is the velocity**.

All vessels are sampled in a single pass over the frames of the velocity window.
""")

C(r"""
t0 = time.time()
net.measure_velocity(burst.frames, reg, idx, burst.fps, progress=True)
print(f"kymographs + 3 velocity estimators for {len(net.vessels)} vessels: {time.time()-t0:.0f} s")
rel = [v for v in net.vessels.values() if v.vel["reliable"]]
VID_V = max([v for v in rel if v.vel["scale"] == 1], key=lambda v: v.vel["quality"] * v.length).id   # a clear demo vessel
print("demo vessel for velocity:", VID_V)
ves = net.vessels[VID_V]
pts = ve.lumen_points(ves.spline, ves.d_median, P.lumen_frac)
fig, ax = plt.subplots(1, 2, figsize=(16, 4))
ax[0].imshow(net.ff, cmap="gray"); viz.zoom_to(ax[0], net, VID_V, 30)
ax[0].plot(pts[::3, :, 0].T, pts[::3, :, 1].T, "y-", lw=0.8); ax[0].set_title(f"{VID_V}: lumen samples (central half)")
ax[1].imshow(viz.stretch(ves.K), aspect="auto", cmap="gray", extent=[0, ves.length, len(idx) / burst.fps, 0])
ax[1].set_xlabel("arc length s (px)"); ax[1].set_ylabel("time (s)"); ax[1].set_title("raw kymograph K(t, s)")
plt.tight_layout(); plt.show()
""")

M(r"""
### 10.2 Keep only what moves with the blood
1. $K_1=K-G_{\sigma_t}*_tK$: subtract a **running temporal mean** ($\sigma_t$ = 10 frames). This removes static anatomy *and* anything that changes slowly without moving (slow registration drift, illumination). A pattern moving ≥ 1 px/frame moves many px within the window, so it survives.
2. $K_2=K_1-\langle K_1\rangle_s$: remove each row's mean (global brightness flicker from auto-gain and blinks).
3. High-pass along $s$ ($\sigma$ = 10 px) and light smoothing along $s$ **only**. Smoothing along $t$ would correlate neighbouring rows and bias the displacement towards zero.
""")

C(r"""
K = ves.K; Kf = np.where(np.isfinite(K), K, np.nanmean(K))
K1 = Kf - ndi.gaussian_filter1d(Kf, P.t_window, axis=0, mode="nearest")
K1b = K1 - K1.mean(1, keepdims=True)
K2 = ves.vel["K2"]
fig, ax = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
ext = [0, ves.length, len(idx) / burst.fps, 0]
for a, im, ttl in zip(ax, [Kf, K1, K1b, K2], ["K (raw)", "− running temporal mean", "− row mean", "high-pass in s → K₂"]):
    a.imshow(viz.stretch(im[:150], 1, 99), aspect="auto", cmap="gray", extent=[0, ves.length, 150 / burst.fps, 0])
    a.set_title(ttl); a.set_xlabel("s (px)")
ax[0].set_ylabel("time (s)"); plt.tight_layout(); plt.show()
""")

M(r"""
### 10.3 Estimator 1: spatio-temporal correlation (LSPIV)
Line-scan particle image velocimetry: cross-correlate each kymograph row with the row `lag` frames later,

$$C(\text{lag},\delta)=\Big\langle \frac{\sum_s K_2(t,s)\,K_2(t+\text{lag},s+\delta)}{\|K_2(t,\cdot)\|\,\|K_2(t+\text{lag},\cdot)\|}\Big\rangle_t .$$

A moving pattern draws a straight **ridge** $\delta = v\cdot\text{lag}$ through the origin. We **track** the ridge from lag 1 upwards, searching near the predicted $\delta$, and fit $v$ by weighted least squares.
The peak is located to sub-pixel precision with a 3-point Gaussian fit:
$\hat\delta = k + \frac{\ln C_{k-1}-\ln C_{k+1}}{2(\ln C_{k-1}-2\ln C_k+\ln C_{k+1})}$.
""")

C(r"""
rid = ves.vel["ridge"]; Cm, sh = ves.vel["Cmap"], ves.vel["shifts"]
lag_demo = ves.vel["lag"]
one = ve.lspiv(K2, lag_demo)
fig, ax = plt.subplots(1, 3, figsize=(16, 3.8))
r0 = 200
ax[0].plot(K2[r0], label=f"row t={r0}"); ax[0].plot(K2[r0 + lag_demo], label=f"row t={r0}+{lag_demo}")
ax[0].set_xlabel("s (px)"); ax[0].legend(); ax[0].set_title("two kymograph rows: the pattern has shifted")
ax[1].plot(one["shifts"], one["curve"]); ax[1].axvline(one["v"] * lag_demo, color="r", ls="--")
ax[1].set_xlabel("shift δ (px)"); ax[1].set_title(f"mean correlation at lag {lag_demo}: peak δ = {one['v']*lag_demo:.2f} px")
ax[2].imshow(Cm, aspect="auto", cmap="magma", extent=[sh[0], sh[-1], len(Cm) + 0.5, 0.5])
ax[2].plot(np.where(rid["used"], rid["disp"], np.nan), rid["lags"], "c.", ms=8, label="tracked ridge")
ax[2].plot(rid["v"] * rid["lags"], rid["lags"], "c-", label=f"fit δ = v·lag, v = {rid['v']:.2f} px/frame")
ax[2].set_xlabel("δ (px)"); ax[2].set_ylabel("lag (frames)"); ax[2].legend(fontsize=8); ax[2].set_title("C(lag, δ)")
plt.tight_layout(); plt.show()
""")

M(r"""
### 10.4 Estimator 2: structure tensor (local streak orientation)
$$J=G_\rho*\begin{pmatrix}K_s^2&K_sK_t\\K_sK_t&K_t^2\end{pmatrix}.$$
The eigenvector of the *smaller* eigenvalue points **along** the streaks, $(e_s,e_t)\Rightarrow v=e_s/e_t$.
The coherence $\big((\mu_1-\mu_2)/(\mu_1+\mu_2)\big)^2$ measures how line-like the neighbourhood is.
This gives a dense map $v(t,s)$. For fast flow the $s$ axis is first compressed so the streaks sit near 45°, where the derivatives are well sampled.
""")

C(r"""
stv = ve.structure_tensor_velocity(K2, rid["v"])
fig, ax = plt.subplots(1, 3, figsize=(16, 3.6), sharey=True)
ax[0].imshow(viz.stretch(K2, 1, 99), aspect="auto", cmap="gray", extent=ext); ax[0].set_title("K₂")
vm = np.nanpercentile(np.abs(stv["v_map"]), 95)
im = ax[1].imshow(stv["v_map"], aspect="auto", cmap="coolwarm", vmin=-vm, vmax=vm, extent=ext); plt.colorbar(im, ax=ax[1], label="px/frame")
ax[1].set_title(f"v(t, s): weighted median {stv['v']:.2f} px/frame")
im = ax[2].imshow(stv["coherence"], aspect="auto", cmap="viridis", vmin=0, vmax=1, extent=ext); plt.colorbar(im, ax=ax[2])
ax[2].set_title("coherence (weight)")
for a in ax: a.set_xlabel("s (px)")
ax[0].set_ylabel("time (s)"); plt.tight_layout(); plt.show()
""")

M(r"""
### 10.5 Estimator 3: time of flight ("flicker" correlation)
The intensity fluctuation at $s_1$ reappears at $s_2=s_1+D$ after a delay $\tau$:
$R_D(\tau)=\langle K_2(t,s_1)\,K_2(t+\tau,s_1+D)\rangle$. Repeating this for several separations and fitting $D = v\,\tau$ gives $v$.
It is the temporal dual of LSPIV and uses the signal differently (time series rather than spatial profiles).
""")

C(r"""
tof = ves.vel["tof"]
fig, ax = plt.subplots(1, 2, figsize=(14, 3.6))
for j, (D, c) in enumerate(zip(tof["separations"], tof["curves"])):
    ax[0].plot(tof["lags"], c, color=plt.cm.viridis(j / (len(tof["curves"]) - 1)), label=f"D = {D:.0f} px")
ax[0].set_xlabel("delay τ (frames)"); ax[0].set_ylabel("correlation"); ax[0].legend(fontsize=7); ax[0].set_title("R_D(τ): the peak moves with D")
u = tof["used"]
ax[1].plot(tof["tau"][u], tof["separations"][u], "o", label="peak delays"); ax[1].plot(tof["tau"][~u], tof["separations"][~u], "x", c="0.6")
tt_ = np.linspace(min(0, tof["tau"].min()), max(0, tof["tau"].max()), 10); ax[1].plot(tt_, tof["v"] * tt_, "r-", label=f"D = v τ, v = {tof['v']:.2f} px/frame")
ax[1].set_xlabel("τ (frames)"); ax[1].set_ylabel("D (px)"); ax[1].legend(); plt.tight_layout(); plt.show()
""")

M(r"""
### 10.6 The Fourier view
A pattern $f(s-vt)$ has the 2-D spectrum $\hat K(k,\omega)=\hat f(k)\,\delta(\omega+vk)$: all its energy lies on the **line $\omega=-vk$**.
Static structure sits on $\omega=0$ and global flicker on $k=0$, which is exactly what the preprocessing removed.
""")

C(r"""
Fk = np.fft.fftshift(np.abs(np.fft.fft2(K2 * np.hanning(len(K2))[:, None] * np.hanning(K2.shape[1])[None, :])) ** 2)
kk = np.fft.fftshift(np.fft.fftfreq(K2.shape[1])); ww = np.fft.fftshift(np.fft.fftfreq(len(K2)))
fig, ax = plt.subplots(figsize=(7, 5))
ax.imshow(np.log10(Fk + 1e-12), aspect="auto", cmap="inferno", extent=[kk[0], kk[-1], ww[-1], ww[0]])
k_ = np.linspace(kk[0], kk[-1], 10); v_ = ves.vel["v_time"]
ax.plot(k_, -v_ * k_, "c--", lw=1.5, label=f"ω = −v k,  v = {v_:.2f} px/frame")
ax.set_xlim(-0.25, 0.25); ax.set_ylim(-0.5, 0.5); ax.set_xlabel("k (cycles / px)"); ax.set_ylabel("ω (cycles / frame)")
ax.legend(); ax.set_title("log power spectrum of K₂"); plt.show()
""")

M(r"""
### 10.7 Velocity over time and the reported value
Sliding-window LSPIV (24 frames ≈ 0.32 s, at a lag giving ≈ 8 px displacement) gives $v(t)$.
The **reported velocity is the median of $v(t)$**, a robust time average. On the flow phantom (10.9) it was the most accurate estimator.
A vessel is marked **reliable** only if the correlation ridge is clear and at least 2 of the 3 independent estimators (ridge fit, structure tensor, time of flight) agree with it in sign and within 35 %.
(Slow vessels are analysed on time-binned kymographs with an extra persistence test; see 10.10.)

We also check whether the cardiac pulse is visible. The spectra of all reliable vessels are averaged, since the heartbeat is common to all vessels while noise is not.
Caveat: 5.6 s holds only ~5–7 beats, so the frequency resolution is ≈ 0.18 Hz.
""")

C(r"""
fig, ax = plt.subplots(1, 2, figsize=(16, 3.6))
ax[0].plot(ves.vel["t_s"], ves.vel["v_t"] * burst.fps, label=VID_V)
ax[0].axhline(ves.vel["v_time"] * burst.fps, c="k", ls="--", label="median (reported)")
ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("velocity (px/s)"); ax[0].legend(); ax[0].set_title("v(t)")
specs = []
for v in rel:
    p = ve.pulsatility(v.vel["t_s"], v.vel["v_t"])
    if p["power"] is not None:
        specs.append(np.interp(np.linspace(0, 4, 200), p["freqs"], p["power"] / (p["power"][p["freqs"] > 0.3].sum() + 1e-12)))
fq = np.linspace(0, 4, 200); spec = np.mean(specs, 0)
ax[1].plot(fq, spec); band = (fq > 0.7) & (fq < 3)
f0 = fq[band][np.argmax(spec[band])]
ax[1].axvline(f0, c="r", ls="--", label=f"peak in 0.7–3 Hz: {f0:.2f} Hz = {60*f0:.0f} /min")
ax[1].set_xlabel("frequency (Hz)"); ax[1].set_ylabel("mean normalised power"); ax[1].legend()
ax[1].set_title(f"network-average spectrum of v(t) over {len(specs)} reliable vessels"); plt.tight_layout(); plt.show()
""")

M(r"""
### 10.8 Do the three estimators agree across the network?
""")

C(r"""
T = net.table()
ok = T.velocity_reliable.astype(bool)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
vt_ = T.v_time_px_f * burst.fps
for a, col, name in [(ax[0], "v_ridge_px_f", "ridge fit"), (ax[1], "v_st_px_f", "structure tensor"), (ax[2], "v_tof_px_f", "time of flight")]:
    y = T[col] * burst.fps
    a.scatter(vt_[~ok], y[~ok], s=12, c="0.7", label="not reliable"); a.scatter(vt_[ok], y[ok], s=16, c="C0", label="reliable")
    lim = np.nanmax(np.abs(np.r_[vt_[ok], y[ok]])) * 1.1
    a.plot([-lim, lim], [-lim, lim], "k--", lw=0.8); a.set_xlim(-lim, lim); a.set_ylim(-lim, lim)
    r = np.corrcoef(vt_[ok], y[ok])[0, 1]
    a.set_xlabel("median v(t)  (px/s)"); a.set_ylabel(f"{name} (px/s)"); a.set_title(f"{name}: r = {r:.3f} (reliable)"); a.legend(fontsize=7)
plt.tight_layout(); plt.show()
print(f"reliable velocity in {ok.sum()} of {len(T)} vessels; speeds {np.nanmin(np.abs(vt_[ok])):.0f}-{np.nanmax(np.abs(vt_[ok])):.0f} px/s")
""")

M(r"""
### 10.9 Validation 4: flow phantom with known velocities
Take the **real** reference image and the **real** centre-lines of 8 vessels. Inside each lumen, add a random red-cell pattern
that moves at a **known** velocity (0.6–11 px/frame, both directions), integrated over the 95 % exposure (motion blur).
Each frame is then displaced by random eye jitter (σ = 6 px, 0.3°) and gets sensor noise.
The synthetic burst goes through the **same** registration → kymograph → velocity chain as the real data.
""")

C(r"""
ph_vids = [v.id for v in sorted(rel, key=lambda v: -v.length)[:8]]
v_true = np.array([1.0, -2.0, 3.0, -0.6, 5.0, -8.0, 11.0, 4.0])[:len(ph_vids)]
t0 = time.time()
ph_frames, M_true = va.make_flow_phantom(net, ph_vids, v_true, n_frames=120)
ph_reg = rg.Registrar(net.ref, net.valid).register_burst(ph_frames.astype(np.uint16), progress=False)
corners = np.array([[0, 1919, 0, 1919], [0, 0, 499, 499], [1, 1, 1, 1]])
reg_err = [np.abs(ph_reg.M[i] @ corners - M_true[i] @ corners).max() for i in range(len(ph_frames))]
ph_pts = [ve.lumen_points(net.vessels[v].spline, net.vessels[v].d_median, P.lumen_frac) for v in ph_vids]
ph_K = ve.kymographs(ph_frames, ph_reg, np.arange(len(ph_frames)), ph_pts, progress=False)
rows = []
for vid, vt, Kx in zip(ph_vids, v_true, ph_K):
    o = net.velocity_for(Kx)
    rows.append(dict(vessel=vid, true=vt, reported_median_vt=o["v_time"], ridge=o["v_ridge"], structure_tensor=o["v_st"],
                     time_of_flight=o["v_tof"], reliable=o["reliable"]))
ph = pd.DataFrame(rows).set_index("vessel")
ph["error_%"] = (ph.reported_median_vt / ph.true - 1) * 100
print(f"phantom: {time.time()-t0:.0f} s; registration error vs known jitter: median {np.median(reg_err):.2f} px, max {np.max(reg_err):.2f} px")
display(ph.round(3))
fig, ax = plt.subplots(figsize=(5, 4.5))
for col, mk in [("reported_median_vt", "o"), ("ridge", "s"), ("structure_tensor", "^"), ("time_of_flight", "x")]:
    ax.plot(ph.true, ph[col], mk, label=col)
ax.plot([-12, 12], [-12, 12], "k--", lw=0.8); ax.set_xlabel("true v (px/frame)"); ax.set_ylabel("estimated"); ax.legend(fontsize=8)
ax.set_title("flow phantom"); plt.show()
""")


M(r"""
### 10.10 Slow flow in thin vessels
With a 10-frame running mean, a pattern must move at least ≈ 0.5–1 px/frame to survive step 1 of the preprocessing.
Thin vessels can flow far more slowly than that. Three additions make slow flow measurable **without** letting drift pass as flow:

1. **Remove the static anatomy explicitly.** Before anything else, each kymograph row is fitted by least squares as
   $K(t,s)\approx c_0+c_1\,\bar K(s)$, where $\bar K$ is the temporal-mean profile, and the fit is subtracted. This absorbs gain and illumination changes of the static anatomy at every time scale.
2. **Time-binning pyramid.** Averaging $k$ consecutive frames turns $v$ px/frame into $kv$ px/bin and reduces noise by ≈ $\sqrt k$.
   All three estimators run unchanged on binned kymographs with $k\in\{1,3,9\}$, and the scale whose displacement per bin lies in their accurate range (0.7–12 px) is kept.
3. **Find the flow line from its asymmetry.** Static structure and registration jitter give correlations that are *symmetric* in the shift $\delta$
   (a stationary pattern looks the same shifted left or right). Directed flow does not. The antisymmetric part
   $$A(\text{lag},\delta)=\tfrac12\big[C(\text{lag},\delta)-C(\text{lag},-\delta)\big]$$
   is integrated along every line through the origin, $\delta=v\cdot\text{lag}$ (a Radon/Hough transform restricted to such lines),
   and the best $v$ seeds the ridge tracker. Slow flow no longer hides under the symmetric static ridge at $\delta=0$.

**Guarding against false slow flow.** Slow registration drift can mimic slow motion. Binned (k > 1) results must therefore also pass a
**persistence test**: the first and second halves of the recording give the same velocity (same sign, within 35 %).
The **negative control** below measures the false-positive rate on lines through static tissue, where the true velocity is zero.
""")

C(r"""
slow_v = [v for v in net.vessels.values() if v.vel.get("reliable") and v.vel["scale"] > 1]
print(f"{len(slow_v)} vessels are measurable only after time-binning:",
      ", ".join(f"{v.id} ({v.speed * burst.fps:.1f} px/s, d={v.d_median:.1f}px, k={v.vel['scale']})" for v in slow_v))
if slow_v:
    VID_S = min(slow_v, key=lambda v: v.speed).id
    vs_ = net.vessels[VID_S]; k = vs_.vel["scale"]
    Kr = ve.remove_static(vs_.K)
    K2_1, _ = ve.preprocess(Kr, P.t_window, P.highpass_s)
    Cm1, sh1 = ve.correlation_map(K2_1, P.max_lag)
    Ck, shk = vs_.vel["Cmap"], vs_.vel["shifts"]
    Ak = 0.5 * (Ck - Ck[:, ::-1])
    fig, ax = plt.subplots(1, 5, figsize=(17, 3.8))
    ax[0].imshow(viz.stretch(K2_1, 1, 99), aspect="auto", cmap="gray"); ax[0].set_title(f"{VID_S}: K₂ at k = 1"); ax[0].set_ylabel("frame")
    ax[1].imshow(viz.stretch(vs_.vel["K2"], 1, 99), aspect="auto", cmap="gray"); ax[1].set_title(f"K₂ binned ×{k}: streaks appear"); ax[1].set_ylabel(f"bin ({k} frames)")
    ax[2].imshow(Cm1, aspect="auto", cmap="magma", extent=[sh1[0], sh1[-1], len(Cm1) + .5, .5]); ax[2].set_title("C at k = 1: static ridge at δ = 0")
    ax[3].imshow(Ck, aspect="auto", cmap="magma", extent=[shk[0], shk[-1], len(Ck) + .5, .5]); ax[3].set_title(f"C at k = {k}")
    r_ = vs_.vel["ridge"]; ax[3].plot(np.where(r_["used"], r_["disp"], np.nan), r_["lags"], "c.")
    am = np.abs(Ak).max()
    ax[4].imshow(Ak, aspect="auto", cmap="RdBu_r", vmin=-am, vmax=am, extent=[shk[0], shk[-1], len(Ak) + .5, .5])
    ax[4].plot(vs_.vel["v_ridge"] * k * r_["lags"], r_["lags"], "k--", lw=1); ax[4].set_title("antisymmetric part A: the flow line")
    for a in ax[2:]: a.set_xlabel("δ (px)")
    plt.tight_layout(); plt.show()
    display(pd.DataFrame(vs_.vel["per_scale"]).set_index("k").round(3))
    fig = viz.vessel_dashboard(net, VID_S); plt.show()
""")

M(r"""
#### Validation 5: negative control (static tissue, true velocity = 0)
For every vessel we draw a line parallel to it, 8 px beyond its wall, and keep only stretches that avoid all detected vessels and all perfused pixels of the flicker map.
Kymographs along these lines go through exactly the same estimator. Every "reliable" velocity found there is a false positive.
""")

C(r"""
ctrl, ctrl_lines = va.negative_control(net, burst.frames, reg, idx)
n_fp = int(ctrl.reliable.sum())
print(f"{len(ctrl)} control lines, {n_fp} false positives ({100 * n_fp / max(len(ctrl), 1):.1f} %)")
if n_fp: display(ctrl[ctrl.reliable].round(3))
fig, ax = plt.subplots(figsize=(16, 4.5))
ax.imshow(net.fl, cmap="gray", vmin=0, vmax=np.percentile(net.fl[net.fl > 0], 99.5)); ax.axis("off")
ax.set_title("control lines (green) over the flicker map; red = reported a velocity")
for sp_, (_, r) in zip(ctrl_lines, ctrl.iterrows()):
    ax.plot(sp_.xy[:, 0], sp_.xy[:, 1], "-", color="red" if r.reliable else "lime", lw=1.2)
plt.show()
""")

M(r"""
Any remaining hits should be inspected. In this recording they were fast (2–3 px/frame), and their kymographs show real streaks: those control lines had run into
faint or defocused vessels that segmentation had missed, not into static tissue. No control line produced a *slow* false positive. The persistence test is what removed them
(without it, 9 of 101 control lines reported slow "flow").

#### Validation 6: slow flow in thin vessels with known velocity
The flow phantom from 10.9 again, now in **thin** vessels (< 8 px), with velocities from 0.05 to 1 px/frame, over the full window.
It compares the multi-scale estimator with the original single-scale one.
""")

C(r"""
thin_v = sorted([v for v in net.vessels.values() if v.d_median < 8 and v.length > 100], key=lambda v: -v.length)[:8]
th_ids = [v.id for v in thin_v]
v_slow = np.array([0.05, 0.1, -0.15, 0.2, 0.3, -0.4, 0.6, 1.0])[:len(th_ids)]
t0 = time.time()
sf, sM = va.make_flow_phantom(net, th_ids, v_slow, n_frames=len(idx), grain=4.0)
sreg = rg.Registrar(net.ref, net.valid).register_burst(sf.astype(np.uint16), progress=False)
sK = ve.kymographs(sf, sreg, np.arange(len(sf)), [ve.lumen_points(net.vessels[v].spline, net.vessels[v].d_median, P.lumen_frac)
                                                   for v in th_ids], progress=False)
rows = []
for vid, vt, Kx in zip(th_ids, v_slow, sK):
    a_, b_ = net.velocity_for(Kx), net.velocity_for_single_scale(Kx)
    rows.append(dict(vessel=vid, d_px=net.vessels[vid].d_median, true=vt, multiscale=a_["v"], time_bin=a_["scale"],
                     single_scale=b_["v"]))
sp_df = pd.DataFrame(rows).set_index("vessel"); sp_df["error_%"] = (sp_df.multiscale / sp_df.true - 1) * 100
print(f"slow phantom: {time.time() - t0:.0f} s")
display(sp_df.round(3))
del sf, sK
""")

# =============================================================================
M(r"""
## 11 · The directed flow graph

1. **Orient** every reliably measured vessel along its velocity sign.
2. **Volumetric flow**: $Q = \bar v\,\pi d^2/4$ with $\bar v = v_{RBC}/1.6$ (centre-line red-cell velocity → lumen-mean velocity, Baker & Wayland 1974).
3. **Mass conservation** (Kirchhoff) at true junctions, $\sum_{in}Q=\sum_{out}Q$, is used
   (a) to **infer** the direction of a vessel whose velocity could not be measured, when it is the only unknown at a junction (iterated), and
   (b) as a **consistency check**: the residual $(\sum_{in}Q-\sum_{out}Q)/\max(\sum_{in}Q, \sum_{out}Q)$ at junctions where every flow was measured.
4. **Node types**: *divergent* (1 in → 2+ out, arteriolar branching), *convergent* (2+ in → 1 out, venular confluence),
   *inlet/outlet* (where blood enters or leaves the field of view), *source/sink* (physically inconsistent, flagged for review).

The result is a `networkx.MultiDiGraph` (multi, because two vessels can join the same pair of nodes).
""")

C(r"""
net.orient()
D = net.dg
print(f"directed graph: {D.number_of_nodes()} nodes, {D.number_of_edges()} vessel edges")
print("edge directions:", pd.Series([d["direction"] for *_, d in D.edges(data=True)]).value_counts().to_dict())
print("node types     :", pd.Series([d["flow_type"] for _, d in D.nodes(data=True)]).value_counts().to_dict())
fig, ax = plt.subplots(figsize=(17, 5.5))
viz.show(ax, net.ff); viz.draw_network(ax, net, color_by="speed", labels=True)
ax.legend(loc="lower right", fontsize=7, framealpha=0.8)
ax.set_title("directed network: colour = RBC speed, arrows = flow direction, dashed grey = direction unknown")
plt.show()
""")

C(r"""
res = pd.DataFrame([dict(node=n, x=d["x"], y=d["y"], type=d["flow_type"], Q_in=d["Q_in"], Q_out=d["Q_out"],
                         residual=d["conservation_residual"]) for n, d in D.nodes(data=True)
                    if np.isfinite(d["conservation_residual"])]).set_index("node")
fig, ax = plt.subplots(1, 2, figsize=(16, 4), gridspec_kw=dict(width_ratios=[1, 3]))
ax[0].hist(res.residual, bins=np.linspace(-1, 1, 11)); ax[0].set_xlabel("mass-balance residual"); ax[0].set_title(f"{len(res)} fully measured junctions")
viz.show(ax[1], net.ff); viz.draw_network(ax[1], net, color_by="none", arrows=True, lw=1, nodes=False, cbar=False)
sc = ax[1].scatter(res.x, res.y, c=res.residual, cmap="RdBu_r", vmin=-1, vmax=1, s=120, edgecolors="k", zorder=9)
for n, r in res.iterrows():
    if abs(r.residual) > 0.5: ax[1].annotate(f"node {n}", (r.x, r.y), color="yellow", fontsize=8, xytext=(5, -8), textcoords="offset points")
plt.colorbar(sc, ax=ax[1], fraction=0.015); ax[1].set_title("residual per junction; |residual| > 0.5 flagged for review")
plt.tight_layout(); plt.show()
display(res.round(3).sort_values("residual"))
""")

M(r"""
**Interpreting the residuals.** Good junctions have residuals of a few tens of percent: the measurement uncertainty of $Q\propto v\,d^2$ is about 30–40 %,
because the diameter error (±15 %) counts twice. Residuals near ±1 mean *everything flows in* or *everything flows out*. That is physically impossible for a
real junction, so it points to (i) a crossing or overlap misread as a junction (vessels at different depths meeting in a T),
(ii) a vessel that leaves the focal plane, or (iii) a wrong direction on one small vessel.
These nodes are exactly what an annotator should review, and the software can list them automatically.

### Graph queries
Because the result is a real directed graph, questions about the network are one-liners:
""")

C(r"""
import networkx as nx
inlets = [n for n, d in D.nodes(data=True) if d["flow_type"].startswith("inlet")]
outlets = [n for n, d in D.nodes(data=True) if d["flow_type"].startswith("outlet")]
known = nx.MultiDiGraph([(a, b, k) for a, b, k, d in D.edges(keys=True, data=True) if d["direction"] == "measured"])
print(f"{len(inlets)} inlets, {len(outlets)} outlets; weakly connected components: {nx.number_weakly_connected_components(D)}")
best = None
for a in inlets:
    if a not in known: continue
    for b in nx.descendants(known, a):
        if b in outlets:
            path = nx.shortest_path(known, a, b)
            if best is None or len(path) > len(best): best = path
if best:
    vids_path = [next(iter(known.get_edge_data(u, v))) for u, v in zip(best[:-1], best[1:])]
    print("longest inlet→outlet path through vessels with measured velocity:", " → ".join(vids_path))
    print("transit time along it:", f"{sum(net.vessels[v].length / net.vessels[v].speed for v in vids_path) / burst.fps:.2f} s",
          f"over {sum(net.vessels[v].length for v in vids_path):.0f} px")
e = next(iter(D.edges(keys=True, data=True)))
print("\nexample edge record:", {k: (np.round(v, 3) if isinstance(v, float) else v) for k, v in e[3].items() if k != "xy"})
""")

C(r"""
files = net.export(os.path.join(OUT, "limbus_network"))
print("written:", *files, sep="\n  ")
js = json.load(open(files[0]))
print("\nJSON top-level keys:", list(js.keys()))
print("graph metadata:", {k: (f"{len(v)} frame indices" if k == "frames" else v) for k, v in js["graph"].items()})
lk = js["links"][0]
print("one link (vessel) has keys:", list(lk.keys()))
display(viz.plotly_network(net))
""")

# =============================================================================
M(r"""
## 12 · Interactive vessel explorer

Pick a vessel from the list (sort by length, diameter, speed or tortuosity; optionally only vessels with reliable velocity).
The dashboard shows the **mask** on the image with flow arrows, the **diameter profile**, **curvature and inflections**, the summary table,
the **raw and moving-component kymographs** with the measured slope, the **spatio-temporal correlation ridge**, **v(t)**, and the **time-of-flight** curves.
Arrows in the list: → measured direction, ⇢ inferred from conservation, ? unknown.
*(This needs a live kernel; a static example is rendered below.)*
""")

C(r"""
ui = viz.explorer(net, default=VID_V)
""")

C(r"""
fig = viz.vessel_dashboard(net, VID_T); plt.show()
""")

# =============================================================================
M(r"""
## 13 · Validation against the video

**(a) Co-moving kymograph.** If $v$ is right, then in a frame moving with the blood, $\tilde K(t,s)=K_2(t,\,s+\int_0^t v\,dt')$, the streaks become **vertical**.
Here the kymograph is sheared by each vessel's own measured $v(t)$.

**(b) Tracer video.** The registered video with every vessel outline and red **tracer dots** advected along each centre-line at that vessel's measured $v(t)$.
Watch whether the dots ride with the dark red-cell aggregates. Tracers moving too fast or too slow would visibly slide past them. The raw camera frame is shown on top.

**(c) Kymograph video.** One vessel: the registered crop (with tracers) and its moving-component kymograph written line by line in real time.
""")

C(r"""
def comoving(v):
    k = v.vel["scale"]                     # rows of K2 are bins of k frames
    K2x = v.vel["K2"]; vt = k * np.interp(np.arange(len(K2x)), v.vel["t_s"] * burst.fps / k,
                                          np.where(np.isfinite(v.vel["v_t"]), v.vel["v_t"], v.vel["v_time"]))
    disp = np.concatenate([[0], np.cumsum(vt[:-1])])
    s = np.arange(K2x.shape[1])
    return np.array([np.interp(s + d, s, row, left=np.nan, right=np.nan) for row, d in zip(K2x, disp)])
show_v = [v for v in sorted(rel, key=lambda v: -v.vel["quality"] * v.length)][:4]
fig, ax = plt.subplots(2, 4, figsize=(16, 6))
for j, v in enumerate(show_v):
    # show only as many frames as the blood needs to cross ~70 % of the vessel (afterwards it has left the kymograph)
    nr = int(np.clip(0.7 * v.length / abs(v.vel["v_time"] * v.vel["scale"]), 30, 200))
    ax[0, j].imshow(viz.stretch(v.vel["K2"][:nr], 1, 99), aspect="auto", cmap="gray"); ax[0, j].set_title(f"{v.id}: K₂ (lab frame)")
    cm_ = comoving(v)[:nr]
    ax[1, j].imshow(viz.stretch(np.nan_to_num(cm_, nan=np.nanmean(cm_)), 1, 99), aspect="auto", cmap="gray")
    ax[1, j].set_title(f"co-moving at measured v(t) → vertical"); ax[1, j].set_xlabel("s (px)")
ax[0, 0].set_ylabel("frame"); ax[1, 0].set_ylabel("frame"); plt.tight_layout(); plt.show()
""")

C(r"""
t0 = time.time()
ov_path = va.render_overlay_video(burst.frames, reg, idx, net, os.path.join(OUT, "validation_tracers.mp4"), fps_out=25)
ky_path = va.render_kymograph_video(burst.frames, reg, idx, net, VID_V, os.path.join(OUT, f"validation_kymograph_{VID_V}.mp4"), fps_out=25)
print(f"rendered in {time.time()-t0:.0f} s (played at 25 fps = {burst.fps/25:.1f}x slow motion):\n  {ov_path}\n  {ky_path}")
display(Video(ov_path, width=960, html_attributes="controls loop"))
display(Video(ky_path, width=720, html_attributes="controls loop"))
""")

# =============================================================================
M(r"""
## 14 · Summary

| stage | method | validation |
|---|---|---|
| registration | SIFT + RANSAC similarity + ECC affine | synthetic jitter recovered to ~0.3 px; the registered mean reproduces `mean_stabilized.tif` |
| detection | structural Frangi (mean image) + functional Frangi (jitter-corrected flicker map) | flicker adds the thin perfused vessels; the fitted jitter σ matches the registration residual |
| centre-lines | hysteresis → skeleton → graph → smoothing B-splines | length error 0.3 % and κ within ~10 % on an analytic curve |
| diameter | FWHM of normal profiles (model fit as shape check) | exact ≥ 6 px on box phantoms; model dependence ≈ ±15 % on real vessels |
| tortuosity | DM, SOAM, κ², ICM from the spline | analytic sine |
| velocity | static-anatomy regression → time-binning pyramid (k = 1, 3, 9) → median sliding LSPIV; reliability = 2-of-3 agreement (+ persistence test when binned) | phantoms: 0.6–11 px/frame (normal vessels) and 0.05–1 px/frame (thin vessels) within a few %; negative control in static tissue; co-moving kymographs; tracer video |
| flow graph | velocity sign + Kirchhoff inference | mass-balance residuals; inconsistent junctions flagged |

**Limitations and next steps**
* **Calibration.** Set `UM_PER_PX` (5.86 µm / optical magnification) to get µm, mm/s and nl/s. The graph and all relative measures do not depend on it.
* **Velocity range.** On phantoms, from 0.05 px/frame (≈ 4 px/s, thin vessels, time-binned) up to ≈ ⅓ of the vessel length per frame.
  On real data the slow end is limited by drift rather than by the estimator. Slow results are accepted only if they persist across both halves of the recording,
  so a vessel with genuinely intermittent slow flow may be left "unknown". Short vessels (< ~40 px) and defocused vessels (right side of this field)
  often give no reliable velocity. They are flagged, not guessed.
* **Longer windows for slow flow.** The slowest measurable speed scales with 1 / (window length). Correlating rows across short gaps
  (using the camera timestamps) would allow every good frame to be used (≈ 13.6 s here instead of 5.6 s).
* **Window.** Velocities come from the longest unbroken run of registered frames (5.6 s here). Longer bursts would resolve the cardiac cycle better.
* **Depth.** Crossings are detected geometrically. A T-shaped overlap of vessels at different depths can still be misread as a junction, and the mass-balance residual flags these.
* **Product integration.** `outputs/limbus_network_graph.json` holds, per vessel, the B-spline (knots, coefficients, degree), the outline polygon, the diameter profile,
  all metrics and the oriented edge. That is everything needed to redraw and edit the annotation in the acquisition software. Re-running `net.orient()` after a user
  edits a direction re-propagates the conservation inference.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3 (limbus)", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.write(nb, "limbus_vessel_workflow.ipynb")
print("wrote limbus_vessel_workflow.ipynb with", len(cells), "cells")
