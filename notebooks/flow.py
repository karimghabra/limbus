import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Flow: where blood moves, and how fast

    Subtracting each pixel's mean over time from a stabilized burst leaves a video in which the flow is much easier to see than in the burst itself. This notebook measures why, and builds on it to find and measure flow without any vessel annotation:

    | § | Step | What it gives |
    |---|---|---|
    | 0 | Setup: run the preprocessing (`foundation.py`) and take one fixation's stabilized frames | `density`, `log_frames`, the moving and static reference pixels |
    | 1 | **Why the mean-subtracted video shows flow**: signal, noise and clutter, before and after | the contrast table, the verdict, the side-by-side video |
    | 2 | **Where flow is**: the perfusion map, from the variance of the mean-subtracted video in a band of temporal frequencies | `perfusion_z`, `perfused` |
    | 3 | The centrelines of the perfused vessels, by skeletonising the perfusion map | `segments` |
    | 4 | **How fast**: a kymograph along every centreline, and the red-cell velocity from the slope of its streaks | `velocities` |
    | 5 | A cross-check of the velocities by 2D cross-correlation of the frames | `velocities["v_xcorr"]` |
    | 6 | The velocity map, the per-segment table, and what the frame rate allows | |
    | 7 | Save the videos, figures and tables to `flow_outputs/<burst>/`, and the runtime | |

    Nothing here uses a vessel annotation: the vessels are found from what moves. Lengths are in pixels and times in frames unless `UM_PER_PX` is set (§0); a velocity in px/frame times the frame rate is px/s.

    **Conventions.** The frames are those of one fixation (the anchor, unless `FIXATION` says otherwise), in its own reference frame, as `foundation.py` leaves them. `NaN` is no data. A velocity is positive when the blood runs from a centreline's first point to its last. *Moving pixels* and *static pixels* are the foundation's own judgement from §11 (`changing_pixels`, and `quiet_pixels` that are also background): they are the reference regions that contrast is measured against, and the static pixels also set the level and the noise scale of the perfusion map (§2), so its thresholds are in units of their spread. The moving pixels are never used to find or measure flow.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 0 · Setup

    The first cells run `foundation.py` inside this notebook (`app.embed()`), on the burst it finds or the one `LIMBUS_BURST` names, and take what this notebook reads from it. Its printed lines show here; its figures and previews do not (open `foundation.py` for those). So *Run all* runs the preprocessing first; running this notebook and the foundation at once on the live server takes the memory of two runs.

    | Setting | Default | What it does |
    |---|---|---|
    | `FIXATION` | `None` | the fixation to analyse; `None` is the anchor, the fixation with the most frames |
    | `MIN_COVERAGE`, `EDGE` | `0.95`, `6` | a pixel is analysed when this share of the fixation's frames has data there, and it lies this many pixels inside that area |
    | `DRIFT_WINDOW` | `15` frames | the rolling mean or median that §1 compares with the plain temporal mean, and the high-pass of the perfusion map (§2): what changes more slowly than this is drift, not flow |
    | `SPATIAL_BLUR` | `1.0` px | Gaussian blur of every frame before the perfusion map, the kymographs and the cross-correlation |
    | `PERFUSION_LEVELS` | `(4, 8)` | hysteresis thresholds on the perfusion map, in robust standard deviations of the static pixels |
    | `MIN_REGION`, `MAX_HOLE` | `150`, `300` px | perfused regions smaller than this are dropped, holes smaller than this filled |
    | `MIN_SEGMENT` | `30` px | shorter centreline segments are not measured |
    | `MAX_VELOCITY`, `MAX_LAG`, `MIN_OVERLAP` | `20` px/frame, `8`, `12` px | the slope search of §4 |
    | `BLOCKS` | `4` | the burst is split into this many blocks of time to give each velocity its uncertainty |
    | `UM_PER_PX` | `None` | micrometres per pixel at the eye (5.86 µm sensor pitch divided by the optical magnification); `None` keeps lengths in pixels |
    | `OUTPUT_DIR` | `flow_outputs` | where §7 writes, in the working directory, one folder per burst |
    """)
    return


@app.cell
def _():
    import base64
    import csv
    import io
    import resource
    import time
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    from scipy.ndimage import convolve, find_objects, gaussian_filter1d, label, median_filter, uniform_filter1d
    from skimage.filters import apply_hysteresis_threshold
    from skimage.morphology import skeletonize

    from foundation import app as foundation

    notebook_start = time.perf_counter()

    FIXATION = None               # None: the anchor fixation
    MIN_COVERAGE = 0.95           # share of the fixation's frames that must have data at an analysed pixel
    EDGE = 6                      # px kept clear of the edge of that area
    DRIFT_WINDOW = 15             # frames; the rolling background, and the perfusion map's high-pass
    SPATIAL_BLUR = 1.0            # px
    PERFUSION_LEVELS = (4.0, 8.0) # hysteresis on the perfusion map, in robust SDs of the static pixels
    MIN_REGION = 150              # px
    MAX_HOLE = 300                # px
    MIN_SEGMENT = 30              # px
    MAX_VELOCITY = 20.0           # px/frame searched
    MAX_LAG = 8                   # frames
    MIN_OVERLAP = 12              # px
    BLOCKS = 4
    UM_PER_PX = None              # e.g. 5.86 / magnification
    OUTPUT_DIR = Path("flow_outputs")
    return (
        BLOCKS,
        DRIFT_WINDOW,
        EDGE,
        FIXATION,
        Image,
        MAX_HOLE,
        MAX_LAG,
        MAX_VELOCITY,
        MIN_COVERAGE,
        MIN_OVERLAP,
        MIN_REGION,
        MIN_SEGMENT,
        OUTPUT_DIR,
        PERFUSION_LEVELS,
        SPATIAL_BLUR,
        ThreadPoolExecutor,
        UM_PER_PX,
        apply_hysteresis_threshold,
        base64,
        convolve,
        csv,
        cv2,
        find_objects,
        foundation,
        gaussian_filter1d,
        io,
        label,
        median_filter,
        notebook_start,
        np,
        plt,
        resource,
        skeletonize,
        time,
        uniform_filter1d,
    )


@app.cell
async def _(foundation, time):
    import warnings as _warnings

    import IPython.display as _ipython_display
    import matplotlib as _matplotlib
    import matplotlib.pyplot as _pyplot

    # runs foundation.py, sections 0-17; its §17 saves its result as usual. Its figures and previews are those of
    # foundation.py and are not repeated here: while it runs, figures go to a backend that shows nothing and
    # previews to a display that shows nothing, and both are put back afterwards.
    _backend, _display = _matplotlib.get_backend(), _ipython_display.display
    _start = time.perf_counter()
    try:
        _matplotlib.use("agg")
        _ipython_display.display = lambda *objects, **options: None
        with _warnings.catch_warnings():
            _warnings.filterwarnings("ignore", message=".*non-interactive.*")
            preprocessing = await foundation.embed()
    finally:
        _pyplot.close("all")
        _ipython_display.display = _display
        _matplotlib.use(_backend)
    foundation_seconds = time.perf_counter() - _start
    print(f"The preprocessing ran in {foundation_seconds:.0f} s; its figures and previews are in foundation.py")
    return foundation_seconds, preprocessing


@app.cell
def _(preprocessing):
    # what this notebook reads from the preprocessing
    INK_MUTED = preprocessing.defs["INK_MUTED"]
    SERIES = preprocessing.defs["SERIES"]
    anchor = preprocessing.defs["anchor"]
    blur_valid = preprocessing.defs["blur_valid"]
    changing_pixels = preprocessing.defs["changing_pixels"]
    final_background_fits = preprocessing.defs["final_background_fits"]
    final_density_fixations = preprocessing.defs["final_density_fixations"]
    fps = preprocessing.defs["fps"]
    quiet_pixels = preprocessing.defs["quiet_pixels"]
    robust_sd = preprocessing.defs["robust_sd"]
    stabilized_fixations = preprocessing.defs["stabilized_fixations"]
    target_folder = preprocessing.defs["target_folder"]
    workers = preprocessing.defs["workers"]
    return (
        INK_MUTED,
        SERIES,
        anchor,
        blur_valid,
        changing_pixels,
        final_background_fits,
        final_density_fixations,
        fps,
        quiet_pixels,
        robust_sd,
        stabilized_fixations,
        target_folder,
        workers,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### One fixation's frames

    Everything below works on one fixation, in its own reference frame: within a fixation the stabilized pixels show the same spot of tissue in every frame, which is what a temporal mean per pixel needs.

    - `density` is the foundation's final optical density, $D_t = B - \ln I_t - c_t$ (§12 of `foundation.py`): each frame measured against the fitted background $B$ and centred on 0 over the background pixels, which removes the flicker of the automatic exposure (one factor per frame).
    - `log_frames` is the stabilized frame itself in log units with only that flicker removed, $\ln I_t + c_t = B - D_t$: the burst as recorded, with its uneven illumination, vessels and fixed pattern.

    In log units a change of 0.01 is a 1% change of brightness. Because $B$ is the same in every frame, subtracting the temporal mean gives the same video from either: $\ln I_t + c_t - \overline{\ln I + c} = -(D_t - \bar D)$.

    | Output | Content |
    |---|---|
    | `number`, `frame_count` | the fixation analysed, and its number of frames |
    | `density`, `log_frames` | `(frames, rows, columns)` `float32`, `NaN` where a frame has no data |
    | `analysed` | the pixels analysed (`MIN_COVERAGE`, `EDGE`) |
    | `moving_reference`, `static_reference` | the foundation's changing pixels, and its quiet background pixels, within `analysed` |
    | `still_picture` | the density of the fixation's mean frame, for the figures |
    | `frame_period`, `exposure_s` | from the frame rate and the recorder's manifest |
    """)
    return


@app.cell
def _(
    EDGE,
    FIXATION,
    MIN_COVERAGE,
    anchor,
    changing_pixels,
    cv2,
    final_background_fits,
    final_density_fixations,
    fps,
    np,
    quiet_pixels,
    target_folder,
):
    import json as _json

    number = anchor if FIXATION is None else FIXATION
    density = final_density_fixations[number]
    _fit = final_background_fits[number]
    frame_count = len(density)
    log_frames = (_fit["B"][None] - density).astype(np.float32)

    _coverage = np.isfinite(density).sum(axis=0)
    analysed = (_coverage >= MIN_COVERAGE * frame_count) & _fit["ok"]
    analysed = cv2.erode(analysed.astype(np.uint8), np.ones((2 * EDGE + 1, 2 * EDGE + 1), np.uint8)).astype(bool)
    moving_reference = changing_pixels[number] & analysed
    static_reference = quiet_pixels[number] & _fit["bg"] & analysed
    still_picture = np.where(analysed, _fit["OD"], np.nan)

    frame_period = 1.0 / fps
    exposure_s = None
    _manifest = target_folder / "manifest.json"
    if _manifest.exists():
        _state = _json.loads(_manifest.read_text(encoding="utf-8")).get("camera_state_at_start", {})
        if "exposure_us" in _state:
            exposure_s = _state["exposure_us"] * 1e-6
    print(f"Fixation {number}: {frame_count} frames of {density.shape[1]} x {density.shape[2]} at {fps:.2f} fps "
          f"({frame_period * 1e3:.1f} ms apart)"
          + (f", exposure {exposure_s * 1e3:.1f} ms ({exposure_s / frame_period:.0%} of the frame period)" if exposure_s else ""))
    print(f"Analysed: {analysed.mean():.0%} of the field | moving reference {moving_reference.sum()} px, "
          f"static reference {static_reference.sum()} px")
    return (
        analysed,
        density,
        exposure_s,
        frame_count,
        frame_period,
        log_frames,
        moving_reference,
        number,
        static_reference,
        still_picture,
    )


@app.cell
def _(OUTPUT_DIR, io, mo, plt, target_folder):
    output_folder = OUTPUT_DIR / target_folder.name


    def show_figure(figure, name, dpi=100):
        """Save a figure to output_folder/<name>.png, close it, and show that PNG (small enough for marimo to show)."""
        output_folder.mkdir(parents=True, exist_ok=True)
        path = output_folder / f"{name}.png"
        figure.savefig(path, dpi=dpi)
        plt.close(figure)
        mo.output.append(mo.image(io.BytesIO(path.read_bytes())))

    return output_folder, show_figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Why the mean-subtracted video shows flow

    Write what a pixel records in frame $t$ as three parts,

    $$
    X_t(\mathbf{x}) \;=\; S(\mathbf{x}) \;+\; m_t(\mathbf{x}) \;+\; n_t(\mathbf{x}),
    $$

    - $S$, the **static** part: the illumination, the tissue, the vessel walls and the column of blood as far as it does not change, the sensor's fixed pattern. It is large: in log units it spans tenths, from the bright background to the dark vessels.
    - $m_t$, the **moving** part: red-cell aggregates and plasma gaps passing through the pixel. It is small, about a percent of the brightness, and it exists only inside perfused vessels.
    - $n_t$, the **noise**: shot noise and read noise, new in every frame.

    The temporal mean of a pixel over $N$ frames is $S + \bar m + \bar n$, so subtracting it leaves $m_t - \bar m + n_t - \bar n$: the static part is gone exactly, the moving part loses only its own average (nothing, for flow that keeps moving), and the noise is the same noise, with its variance multiplied by $1 - 1/N$.

    **So the subtraction cannot raise the signal-to-noise ratio of a pixel in a frame**: the fluctuation over time, signal and noise, is untouched by subtracting a constant per pixel. What it changes is what the moving part has to compete with on the screen. Before, the moving part sits on top of $S$, whose spatial structure (vessel edges, uneven light) is typically far larger, and a display stretched to the frame's range leaves the moving part only a few gray levels (the table below gives both for this burst). After, the display range is set by the noise and the moving part themselves, and the moving part fills it. That is a **contrast gain against clutter**, not a signal-to-noise gain.

    The true signal-to-noise gains come from averaging: over neighbouring pixels (a blur the size of a red-cell aggregate), over many frames (the temporal variance map below, and §2), and from removing what is noise-like but not noise, such as the frame-to-frame flicker and slow drift. A rolling mean or median over `DRIFT_WINDOW` frames instead of the mean over the whole burst also follows slow drift (tear film, focus, residual motion), at the cost of the part of the flow that changes more slowly than the window.

    **What is measured.** For each version of the video, over the moving and the static reference pixels, on every second row and column:

    | Measure | Definition |
    |---|---|
    | moving signal $a$ | $\sqrt{\langle\sigma_t^2\rangle_\text{moving} - \langle\sigma_t^2\rangle_\text{static}}$: the fluctuation over time in the vessels in excess of that on the static tissue |
    | noise $n$ | $\sqrt{\langle\sigma_t^2\rangle_\text{static}}$: the fluctuation over time on the static tissue |
    | SNR | $a / n$: the moving signal against the noise, in one pixel and one frame |
    | clutter $c$ | the spatial standard deviation of the version's temporal mean over the analysed field: what stays on the screen and does not move |
    | fine clutter $c_f$ | the same for structure finer than ~8 px (the temporal mean minus its 4 px blur), next to the moving pixels: vessel edges, the scale of red-cell aggregates |
    | CNR | $a / \sqrt{c^2 + n^2}$: the moving signal against everything else in a frame; CNR (fine) uses $c_f$ |
    | gray levels | $a$ in 8-bit gray levels when the video is shown stretched from its 0.5th to its 99.5th percentile, as a viewer would show it |
    """)
    return


@app.cell
def _(DRIFT_WINDOW, cv2, median_filter, np, uniform_filter1d):
    def fill_with_mean(video):
        """Return a complete copy of a (frames, rows, columns) video: a pixel's missing frames get its temporal mean."""
        mean = np.nanmean(video, axis=0)
        return np.where(np.isfinite(video), video, np.nan_to_num(mean)[None]).astype(np.float32)


    def remove_rolling_mean(video, window=DRIFT_WINDOW, chunk=64):
        """Return video minus each pixel's centred running mean over `window` frames (over the frames with data),
        row block by row block so that only one extra stack is held. NaN stays NaN."""
        out = np.empty_like(video)
        for row in range(0, video.shape[1], chunk):
            part = video[:, row:row + chunk]
            has_data = np.isfinite(part)
            total = uniform_filter1d(np.where(has_data, part, 0), window, axis=0, mode="nearest")
            count = uniform_filter1d(has_data.astype(np.float32), window, axis=0, mode="nearest")
            out[:, row:row + chunk] = np.where(has_data, part - total / np.maximum(count, 1e-3), np.nan)
        return out


    def remove_rolling_median(video, window=DRIFT_WINDOW):
        """Return a complete video minus each pixel's centred running median over `window` frames."""
        return video - median_filter(video, size=(window, 1, 1), mode="nearest")


    def contrast_measures(video, moving, static, field):
        """The measures of §1 for a complete (frames, rows, columns) video and three masks of its pixels."""
        mean = video.mean(axis=0)
        variance = video.var(axis=0)
        noise = float(np.sqrt(variance[static].mean()))
        signal = float(np.sqrt(max(variance[moving].mean() - noise ** 2, 0.0)))
        clutter = float(mean[field].std())
        weight = field.astype(np.float32)
        smooth = cv2.GaussianBlur(np.where(field, mean, 0), (0, 0), 4) / np.maximum(cv2.GaussianBlur(weight, (0, 0), 4), 1e-3)
        near = cv2.dilate(moving.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & field
        fine_clutter = float((mean - smooth)[near].std())
        low, high = np.percentile(video[::3][:, field], [0.5, 99.5])
        return {
            "moving signal": signal, "noise": noise, "SNR": signal / noise,
            "clutter": clutter, "fine clutter": fine_clutter,
            "CNR": signal / np.hypot(clutter, noise), "CNR (fine)": signal / np.hypot(fine_clutter, noise),
            "gray levels": signal / (high - low) * 255,
        }

    return (
        contrast_measures,
        fill_with_mean,
        remove_rolling_mean,
        remove_rolling_median,
    )


@app.cell
def _(
    DRIFT_WINDOW,
    analysed,
    blur_valid,
    contrast_measures,
    fill_with_mean,
    log_frames,
    moving_reference,
    np,
    number,
    remove_rolling_mean,
    remove_rolling_median,
    stabilized_fixations,
    static_reference,
):
    _step = (slice(None), slice(None, None, 2), slice(None, None, 2))  # every second row and column
    _moving, _static, _field = moving_reference[::2, ::2], static_reference[::2, ::2], analysed[::2, ::2]

    contrast_table = {}
    _dn = stabilized_fixations[number][_step].astype(np.float32)
    _dn[_dn == 0] = np.nan                                     # 0 is no data in the stabilized frames
    _dn = fill_with_mean(_dn)
    contrast_table["stabilized frame (DN)"] = contrast_measures(_dn, _moving, _static, _field)
    contrast_table["DN − temporal mean"] = contrast_measures(_dn - _dn.mean(axis=0), _moving, _static, _field)
    del _dn

    for _scale, _label in ((0.0, ""), (1.5, ", blurred 1.5 px")):
        if _scale:
            _video = fill_with_mean(np.stack([blur_valid(_frame, _scale)[::2, ::2] for _frame in log_frames]))
        else:
            _video = fill_with_mean(log_frames[_step])
        contrast_table[f"log frame{_label}"] = contrast_measures(_video, _moving, _static, _field)
        contrast_table[f"log − temporal mean{_label}"] = contrast_measures(_video - _video.mean(axis=0), _moving, _static, _field)
        contrast_table[f"log − rolling mean ({DRIFT_WINDOW}){_label}"] = contrast_measures(
            remove_rolling_mean(_video), _moving, _static, _field)
        contrast_table[f"log − rolling median ({DRIFT_WINDOW}){_label}"] = contrast_measures(
            remove_rolling_median(_video), _moving, _static, _field)
    del _video

    _columns = list(next(iter(contrast_table.values())))
    _rows = ["| version | " + " | ".join(_columns) + " |", "|---" * (len(_columns) + 1) + "|"]
    for _name, _values in contrast_table.items():
        _rows.append(f"| {_name} | " + " | ".join(f"{_values[_column]:.3g}" for _column in _columns) + " |")
    contrast_markdown = "\n".join(_rows)
    return contrast_markdown, contrast_table


@app.cell
def _(DRIFT_WINDOW, contrast_markdown, contrast_table, mo):
    _before, _after = contrast_table["log frame"], contrast_table["log − temporal mean"]
    _rolling = contrast_table[f"log − rolling mean ({DRIFT_WINDOW})"]
    _blurred = contrast_table["log − temporal mean, blurred 1.5 px"]
    _rolling_blurred = contrast_table[f"log − rolling mean ({DRIFT_WINDOW}), blurred 1.5 px"]
    _dn_before, _dn_after = contrast_table["stabilized frame (DN)"], contrast_table["DN − temporal mean"]
    _median = contrast_table[f"log − rolling median ({DRIFT_WINDOW})"]
    _median_blurred = contrast_table[f"log − rolling median ({DRIFT_WINDOW}), blurred 1.5 px"]

    _level = "below" if _after["SNR"] < 1 else "not below"
    if _dn_after["SNR"] < _after["SNR"]:
        _dn_note = (f"its SNR ({_dn_after['SNR']:.2f}) is lower than in the log frames ({_after['SNR']:.2f}), which fits the "
                    "flicker of the automatic exposure that the DN frames still carry: a fluctuation of the whole frame "
                    "that the density removes, and removing it is a genuine noise reduction")
    else:
        _dn_note = (f"its SNR ({_dn_after['SNR']:.2f}) is not lower than in the log frames ({_after['SNR']:.2f}), so removing "
                    "the flicker of the automatic exposure, as the density does, gains no measurable noise reduction here")
    if _before["fine clutter"] > _before["moving signal"]:
        _fine_note = (f"Even the fine clutter alone, the vessel edges at the scale of the moving pattern "
                      f"({_before['fine clutter']:.3f}), is larger than the moving signal ({_before['moving signal']:.3f}).")
    else:
        _fine_note = (f"The fine clutter alone, the vessel edges at the scale of the moving pattern "
                      f"({_before['fine clutter']:.3f}), is not larger than the moving signal "
                      f"({_before['moving signal']:.3f}): it is the coarse clutter that hides the flow.")

    _rolling_loses = _rolling["SNR"] < _after["SNR"]
    _rolling_wins_pooled = _rolling_blurred["SNR"] > _blurred["SNR"]
    if _rolling_loses:
        _pixel_note = (f"At one pixel it lowers the SNR ({_rolling['SNR']:.2f} against {_after['SNR']:.2f}): the flow lost "
                       "costs more than the drift removed gains, as happens when the pixel noise dominates the static "
                       "tissue's fluctuation.")
    else:
        _pixel_note = (f"At one pixel it does not lower the SNR ({_rolling['SNR']:.2f} against {_after['SNR']:.2f}): the "
                       "drift removed is worth at least the flow lost.")
    if _rolling_wins_pooled:
        _pooled_note = (f"Pooled over 1.5 px it raises the SNR ({_rolling_blurred['SNR']:.2f} against {_blurred['SNR']:.2f}): "
                        "the pixel noise averages down, the drift, which is smooth in space, does not, and removing it wins.")
    else:
        _pooled_note = (f"Pooled over 1.5 px it does not raise the SNR either ({_rolling_blurred['SNR']:.2f} against "
                        f"{_blurred['SNR']:.2f}): the drift it removes is worth no more than the flow it loses.")
    _single_choice = "the plain temporal mean is as good" if _rolling_loses else "the rolling mean is at least as good"
    _pooled_choice = ("the rolling background is better" if _rolling_wins_pooled
                      else "the rolling background is no better on this burst")
    mo.md(rf"""
    {contrast_markdown}

    **The verdict on this burst.** Subtracting the temporal mean leaves the signal-to-noise ratio of a pixel in a frame where it was ({_before["SNR"]:.2f} before, {_after["SNR"]:.2f} after: the moving signal is *{_level}* the noise in one pixel of one frame). It takes the contrast of the moving signal against everything else on the screen from {_before["CNR"]:.2f} to {_after["CNR"]:.2f}, {_after["CNR"] / _before["CNR"]:.3g} times, because the clutter it competes with goes from {_before["clutter"]:.3f} to {_after["clutter"]:.1g} in log units; shown stretched to its range, the moving signal goes from {_before["gray levels"]:.0f} to {_after["gray levels"]:.0f} gray levels. In the recorded units (DN): CNR {_dn_before["CNR"]:.2f} → {_dn_after["CNR"]:.2f}, {_dn_before["gray levels"]:.0f} → {_dn_after["gray levels"]:.0f} gray levels; {_dn_note}. **Whatever is gained here is a contrast gain from removing static clutter, not a signal-to-noise gain.** {_fine_note}

    What the eye then does is where a signal-to-noise gain can come from: it pools neighbouring pixels and successive frames. A 1.5 px blur alone takes the SNR from {_after["SNR"]:.2f} to {_blurred["SNR"]:.2f}; the variance over all the frames (below, and §2) pools the frames as well. The rolling mean over {DRIFT_WINDOW} frames takes out the slow drift, and with it the part of the flow that changes more slowly than the window. {_pixel_note} {_pooled_note} The rolling median gives an SNR of {_median["SNR"]:.2f} at one pixel and {_median_blurred["SNR"]:.2f} pooled. So for watching single frames {_single_choice}; for maps made from many frames and pooled in space (§2) {_pooled_choice}.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Motion-contrast maps.** One picture from all the frames: each pixel's temporal standard deviation, of the video (blurred by 1.5 px) minus its temporal mean and of the same video minus the rolling mean, multiplied by the square root of the mean brightness so that shot noise, which is larger where the image is darker, gives one level everywhere. Each is divided by its median on the static pixels and given as how many robust standard deviations of the static pixels a pixel stands above that level. The second figure plots the separation against the number of frames averaged, next to $\sqrt{N}$, the growth that averaging independent frames gives; this is the kind of signal-to-noise gain that the eye gets from watching the video. Below, one frame of each version of the video next to the two maps, on a zoomed part of the field.
    """)
    return


@app.cell
def _(
    DRIFT_WINDOW,
    INK_MUTED,
    blur_valid,
    density,
    fill_with_mean,
    frame_count,
    log_frames,
    moving_reference,
    np,
    plt,
    remove_rolling_mean,
    remove_rolling_median,
    robust_sd,
    show_figure,
    static_reference,
):
    def motion_contrast(deviation, brightness, static):
        """Temporal SD map, levelled for shot noise and scaled to robust SDs of the static pixels above their median."""
        spread = np.sqrt(np.nanmean(deviation ** 2, axis=0)) * brightness
        level = spread / np.nanmedian(spread[static])
        return (level - 1) / robust_sd(level[static][np.isfinite(level[static])])


    _brightness = np.sqrt(np.exp(np.nanmean(log_frames[:, ::2, ::2], axis=0)))
    _static, _moving = static_reference[::2, ::2], moving_reference[::2, ::2]
    _video = np.stack([blur_valid(_frame, 1.5)[::2, ::2] for _frame in log_frames])
    _mean_removed = _video - np.nanmean(_video, axis=0)
    _rolling_removed = remove_rolling_mean(_video)
    sd_map_mean = motion_contrast(_mean_removed, _brightness, _static)
    sd_map_rolling = motion_contrast(_rolling_removed, _brightness, _static)
    frames_averaged = [_n for _n in (2, 4, 8, 16, 32, 64) if _n < frame_count] + [frame_count]
    separation_by_frames = {
        _name: [float(np.nanmedian(motion_contrast(_dev[:_n], _brightness, _static)[_moving])) for _n in frames_averaged]
        for _name, _dev in (("minus temporal mean", _mean_removed), (f"minus rolling mean ({DRIFT_WINDOW})", _rolling_removed))
    }

    # a window of up to 256 x 256 px around the most moving pixels, for the frame comparison
    _height, _width = min(256, density.shape[1]), min(256, density.shape[2])
    _left = int(np.clip(np.argmax(np.convolve(moving_reference.sum(axis=0), np.ones(_width), "same")) - _width // 2,
                        0, density.shape[2] - _width))
    _top = int(np.clip(np.argmax(np.convolve(moving_reference.sum(axis=1), np.ones(_height), "same")) - _height // 2,
                       0, density.shape[1] - _height))
    zoom = (slice(_top, _top + _height), slice(_left, _left + _width))
    _t = frame_count // 2
    _window = log_frames[:, zoom[0], zoom[1]]
    _complete = fill_with_mean(_window)
    _panels = [
        ("log frame", _window[_t]),
        ("− temporal mean", _complete[_t] - _complete.mean(axis=0)),
        (f"− rolling mean ({DRIFT_WINDOW})", remove_rolling_mean(_window)[_t]),
        (f"− rolling median ({DRIFT_WINDOW})", remove_rolling_median(_complete)[_t]),
    ]
    _figure, _axes = plt.subplots(1, 6, figsize=(18, 3.4))
    for _axis, (_title, _image) in zip(_axes, _panels):
        _low, _high = np.nanpercentile(_image, [0.5, 99.5])
        _axis.imshow(_image, cmap="gray", vmin=_low, vmax=_high)
        _axis.set_title(f"frame {_t}: {_title}")
    for _axis, (_title, _map) in zip(_axes[4:], (("SD map, − temporal mean", sd_map_mean),
                                                 (f"SD map, − rolling mean", sd_map_rolling))):
        _axis.imshow(np.repeat(np.repeat(_map, 2, 0), 2, 1)[zoom], cmap="magma", vmin=0, vmax=20)
        _axis.set_title(_title)
    for _axis in _axes:
        _axis.axis("off")
    _figure.suptitle("each frame panel stretched from its own 0.5th to 99.5th percentile; the maps from 0 to 20 robust SDs "
                     "of the static pixels", fontsize=9, color=INK_MUTED)
    _figure.tight_layout()
    show_figure(_figure, "frame_comparison")

    _figure, _axis = plt.subplots(figsize=(6, 3.4))
    for _name, _values in separation_by_frames.items():
        _axis.plot(frames_averaged, _values, "o-", label=_name)
    _axis.plot(frames_averaged, np.array(frames_averaged) ** 0.5 * separation_by_frames["minus temporal mean"][0] / 2 ** 0.5,
               ":", color=INK_MUTED, label=r"$\propto\sqrt{N}$")
    _axis.set_xscale("log", base=2)
    _axis.set_xlabel("frames averaged")
    _axis.set_ylabel("median of the moving pixels\n(robust SDs of the static pixels)")
    _axis.set_title("separation of flow from static tissue in the SD map")
    _axis.legend()
    _figure.tight_layout()
    show_figure(_figure, "separation_by_frames")
    print("SD map, median over the moving pixels in robust SDs of the static ones: "
          + ", ".join(f"{_name} {_values[-1]:.1f}" for _name, _values in separation_by_frames.items()))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **The video.** The fixation's frames side by side, at half size: left the log frame (the stabilized burst with the flicker removed), middle minus each pixel's temporal mean, right minus the rolling mean over `DRIFT_WINDOW` frames. Each panel is stretched from its own 0.5th to 99.5th percentile over the whole video, so every panel uses the full gray scale; pixels without data in a frame are black. It is saved at that size as WebM (VP8, which browsers play) to `flow_outputs/<burst>/`; the preview here is an animated WebP of at most 150 frames, narrowed until it is under 2.5 MB so that marimo can show it.
    """)
    return


@app.cell
def _(
    Image,
    base64,
    cv2,
    fps,
    io,
    log_frames,
    mo,
    np,
    output_folder,
    remove_rolling_mean,
):
    PREVIEW_MAX_BYTES = 2_500_000  # an inline preview is embedded in the output (base64, a third larger); marimo refuses ~10 MB


    def save_video(frames, path, rate):
        """Save a (frames, rows, columns) uint8 video as WebM (VP8), which browsers play; return the path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"VP80"), rate, (frames.shape[2], frames.shape[1]), False)
        if not writer.isOpened():
            raise RuntimeError(f"OpenCV cannot write VP8 video to {path}")
        for _frame in frames:
            writer.write(_frame)
        writer.release()
        return path


    def preview(frames, rate, shown_width, max_frames=150):
        """An inline animated WebP of a uint8 video: at most `max_frames` frames (one in every few, at the same speed),
        made narrower until it is under PREVIEW_MAX_BYTES, and shown `shown_width` pixels wide."""
        stride = -(-len(frames) // max_frames)
        frames, rate = frames[::stride], rate / stride
        width = frames.shape[2]
        for _attempt in range(5):
            height = max(2, round(width * frames.shape[1] / frames.shape[2]))
            images = [Image.fromarray(cv2.resize(_frame, (width, height), interpolation=cv2.INTER_AREA)) for _frame in frames]
            buffer = io.BytesIO()
            images[0].save(buffer, format="WEBP", save_all=True, append_images=images[1:], duration=int(1000 / rate),
                           loop=0, quality=75, method=4)
            if buffer.tell() <= PREVIEW_MAX_BYTES or width <= 128:
                break
            width = max(128, int(width * (PREVIEW_MAX_BYTES / buffer.tell()) ** 0.5))
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return mo.Html(f'<img src="data:image/webp;base64,{encoded}" width="{shown_width}">')


    def to_gray(video):
        """Stretch a video from its 0.5th to its 99.5th percentile to 8 bits; no data is black."""
        low, high = np.nanpercentile(video[::4], [0.5, 99.5])
        return (np.clip((np.nan_to_num(video, nan=low) - low) / (high - low), 0, 1) * 255).astype(np.uint8)


    _half = np.stack([cv2.resize(_frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) for _frame in log_frames])
    _panels = [to_gray(_half), to_gray(_half - np.nanmean(_half, axis=0)), to_gray(remove_rolling_mean(_half))]
    _labels = ("recorded (log, flicker removed)", "minus temporal mean", "minus rolling mean")
    for _panel, _label in zip(_panels, _labels):
        for _frame in _panel:
            cv2.putText(_frame, _label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2, cv2.LINE_AA)
    _video = np.concatenate(_panels, axis=2)
    side_by_side_path = save_video(_video, output_folder / "recorded_vs_mean_subtracted.webm", fps)
    print(f"Saved {side_by_side_path} ({side_by_side_path.stat().st_size / 1e6:.1f} MB, {_video.shape[2]} x {_video.shape[1]})")
    _shown = preview(_video, fps, 1000)
    del _half, _panels, _video
    _shown
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Where flow is: the perfusion map

    The perfusion map is the motion-contrast map of §1 made at full resolution, with these choices, which §1 and the spectrum below test on this burst:

    1. Every frame of the log video is blurred by `SPATIAL_BLUR` pixels (over the pixels with data), which pools a red-cell aggregate's pixels.
    2. Each pixel's temporal mean is subtracted (`mean_removed`, the video of §1) and then its rolling mean over `DRIFT_WINDOW` frames (`high_passed`): a high-pass in time. The reason is drift: blood adds power in the moving pixels across the temporal frequencies, but at the lowest ones the static tissue drifts too, and the drift differs from place to place, which makes a map of the plain variance patchy. The spectrum below shows both for this burst.
    3. The temporal standard deviation of `high_passed`, times the square root of the mean brightness (shot noise), divided by its median over the static pixels, is the **perfusion level**; minus 1, in robust standard deviations of the static pixels, it is the **perfusion map** `perfusion_z`.
    4. Blurred by 1.5 px, it is thresholded by hysteresis at `PERFUSION_LEVELS` (pixels above the higher level, and those above the lower one connected to them), small regions dropped and small holes filled: the **perfused regions** `perfused`.

    The table below the spectrum shows the choice of the window on every second row and column: the spread of the map on the static pixels (lower is a cleaner background) and the median of the moving pixels in that spread.

    | Output | Content |
    |---|---|
    | `mean_removed`, `high_passed` | `(frames, rows, columns)` `float32`: the blurred log video minus its temporal mean, and minus its rolling mean as well |
    | `perfusion_z` | `(rows, columns)`: the perfusion map, `NaN` outside `analysed` |
    | `perfused` | `(rows, columns)` `bool`: the perfused regions |
    """)
    return


@app.cell
def _(
    DRIFT_WINDOW,
    SPATIAL_BLUR,
    blur_valid,
    log_frames,
    np,
    remove_rolling_mean,
):
    mean_removed = np.stack([blur_valid(_frame, SPATIAL_BLUR) for _frame in log_frames]).astype(np.float32)
    mean_removed -= np.nanmean(mean_removed, axis=0)
    high_passed = remove_rolling_mean(mean_removed, DRIFT_WINDOW)
    print(f"mean_removed and high_passed: {mean_removed.shape}, {2 * mean_removed.nbytes / 1e6:.0f} MB together")
    return high_passed, mean_removed


@app.cell
def _(
    DRIFT_WINDOW,
    SERIES,
    analysed,
    fps,
    frame_count,
    mean_removed,
    moving_reference,
    np,
    plt,
    remove_rolling_mean,
    robust_sd,
    show_figure,
    static_reference,
):
    _sample = mean_removed[:, ::2, ::2]
    _moving, _static = moving_reference[::2, ::2], static_reference[::2, ::2]
    _spectrum = np.abs(np.fft.rfft(np.nan_to_num(_sample[:, analysed[::2, ::2]]), axis=0)) ** 2
    spectrum_frequencies = np.fft.rfftfreq(frame_count, 1 / fps)
    _which = _moving[analysed[::2, ::2]], _static[analysed[::2, ::2]]
    spectrum_moving, spectrum_static = (_spectrum[:, _w].mean(axis=1) for _w in _which)
    del _spectrum

    _figure, _axes = plt.subplots(1, 2, figsize=(12, 3.4))
    _axes[0].semilogy(spectrum_frequencies[1:], spectrum_moving[1:], color=SERIES[1], label="moving pixels")
    _axes[0].semilogy(spectrum_frequencies[1:], spectrum_static[1:], color=SERIES[0], label="static pixels")
    _axes[0].axvline(fps / DRIFT_WINDOW, color="#898781", ls=":", label=f"fps / DRIFT_WINDOW")
    _axes[0].set_xlabel("frequency (Hz)")
    _axes[0].set_ylabel("power (log units², per frequency)")
    _axes[0].set_title("temporal power spectrum of the mean-subtracted video")
    _axes[0].legend()
    _axes[1].plot(spectrum_frequencies[1:], spectrum_moving[1:] / spectrum_static[1:], color=SERIES[2])
    _axes[1].set_xlabel("frequency (Hz)")
    _axes[1].set_ylabel("moving / static")
    _axes[1].set_title("excess power of the moving pixels")
    _figure.tight_layout()
    show_figure(_figure, "temporal_spectrum")

    window_rows = []
    for _window in (None, 5, 9, 15, 25):
        _video = _sample if _window is None else remove_rolling_mean(_sample, _window)
        _spread = np.sqrt(np.nanmean(_video ** 2, axis=0))
        _level = _spread / np.nanmedian(_spread[_static])
        _scale = robust_sd(_level[_static])
        window_rows.append((_window, _scale, float(np.nanmedian((_level[_moving] - 1) / _scale))))
    print("window (frames) | spread on static pixels | median of moving pixels (in that spread)")
    for _window, _scale, _median in window_rows:
        print(f"{'whole burst' if _window is None else _window:>15} | {_scale:23.3f} | {_median:.1f}")
    print(f"Moving pixels have {spectrum_moving[1] / spectrum_static[1]:.0f}x the power of static pixels at "
          f"{spectrum_frequencies[1]:.1f} Hz and {spectrum_moving[-1] / spectrum_static[-1]:.1f}x at the Nyquist "
          f"frequency, {spectrum_frequencies[-1]:.1f} Hz")
    return spectrum_frequencies, spectrum_moving, spectrum_static


@app.cell
def _(
    MAX_HOLE,
    MIN_REGION,
    PERFUSION_LEVELS,
    analysed,
    apply_hysteresis_threshold,
    blur_valid,
    high_passed,
    label,
    log_frames,
    moving_reference,
    np,
    robust_sd,
    static_reference,
):
    def drop_small(mask, size):
        """Remove the connected regions of a boolean mask that have `size` pixels or fewer."""
        labels, _ = label(mask)
        counts = np.bincount(labels.ravel())
        keep = counts > size
        keep[0] = False
        return keep[labels]


    _brightness = np.sqrt(np.exp(np.nanmean(log_frames, axis=0)))
    _spread = np.sqrt(np.nanmean(high_passed ** 2, axis=0)) * _brightness
    _level = _spread / np.nanmedian(_spread[static_reference])
    perfusion_z = np.where(analysed, (_level - 1) / robust_sd(_level[static_reference]), np.nan).astype(np.float32)

    _smooth = np.nan_to_num(blur_valid(perfusion_z, 1.5), nan=0.0)
    perfused = apply_hysteresis_threshold(_smooth, PERFUSION_LEVELS[0], PERFUSION_LEVELS[1]) & analysed
    perfused = drop_small(perfused, MIN_REGION)
    perfused = (perfused | ~drop_small(~perfused, MAX_HOLE)) & analysed  # small holes filled
    print(f"Perfused: {perfused[analysed].mean():.0%} of the analysed field | "
          f"{perfused[moving_reference].mean():.0%} of the foundation's changing pixels, "
          f"{perfused[static_reference].mean():.2%} of its static background pixels | "
          f"perfusion map on the static pixels: 99th percentile {np.nanpercentile(perfusion_z[static_reference], 99):.1f}, "
          f"on the changing pixels: median {np.nanmedian(perfusion_z[moving_reference]):.1f}")
    return perfused, perfusion_z


@app.cell
def _(np, perfused, perfusion_z, plt, show_figure, still_picture):
    still_range = (0, float(np.nanpercentile(still_picture, 99.5)))  # the gray scale of the still picture in the figures
    still_gray = plt.cm.gray.with_extremes(bad="black")             # pixels not analysed are black
    _figure, _axes = plt.subplots(1, 2, figsize=(16, 5.4))
    _axes[0].imshow(still_picture, cmap=still_gray, vmin=still_range[0], vmax=still_range[1])
    _shown = _axes[0].imshow(perfusion_z, cmap="magma", vmin=0, vmax=30, alpha=0.65)
    _axes[0].set_title("perfusion map over the still picture (density of the mean frame)")
    _figure.colorbar(_shown, ax=_axes[0], fraction=0.03, label="robust SDs of the static pixels")
    _axes[1].imshow(still_picture, cmap=still_gray, vmin=still_range[0], vmax=still_range[1])
    _axes[1].contour(perfused, [0.5], colors="#1baf7a", linewidths=0.7)
    _axes[1].set_title("perfused regions")
    for _axis in _axes:
        _axis.axis("off")
    _figure.tight_layout()
    show_figure(_figure, "perfusion_map")
    return still_gray, still_range


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Centrelines

    The perfused regions are thinned to their skeleton, one pixel wide, and cut at every junction (a skeleton pixel with three or more neighbours, and the pixels next to it). Each piece between junctions or ends is a **segment**: its pixels are put in order from one end, smoothed along their length by a 2 px Gaussian and resampled every pixel. Segments shorter than `MIN_SEGMENT` are left out (they are mostly spurs from the edges of wide regions), and so are closed loops.

    Two numbers summarise the perfused network, as perfusion is summarised in capillaroscopy: the share of the field that is perfused, and the **perfused length density**, the length of centreline per area (the microscopy literature's functional capillary density, here for every perfused vessel, not only capillaries).

    | Output | Content |
    |---|---|
    | `segments` | list of `(length, 2)` arrays of `(row, column)`, one point per pixel of length |
    | `centrelines()` | the function, for any boolean mask |
    """)
    return


@app.cell
def _(convolve, cv2, find_objects, gaussian_filter1d, label, np, skeletonize):
    def _order(points):
        """Order the pixels of a thin open curve from one end to the other; None for a closed loop."""
        remaining = {tuple(p) for p in points}

        def neighbours(p):
            return [(p[0] + a, p[1] + b) for a in (-1, 0, 1) for b in (-1, 0, 1)
                    if (a or b) and (p[0] + a, p[1] + b) in remaining]

        ends = [p for p in remaining if len(neighbours(p)) == 1]
        if not ends:
            return None
        current, path, seen = ends[0], [ends[0]], {ends[0]}
        while True:
            following = sorted((q for q in neighbours(current) if q not in seen),
                               key=lambda q: abs(q[0] - current[0]) + abs(q[1] - current[1]))
            if not following:
                return np.array(path, dtype=float)
            current = following[0]
            path.append(current)
            seen.add(current)


    def centrelines(mask, min_length, smooth=2.0):
        """The segments of a mask's skeleton between junctions, as (length, 2) arrays of (row, column) every pixel."""
        skeleton = skeletonize(mask)
        neighbours = convolve(skeleton.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - 1
        junctions = cv2.dilate((skeleton & (neighbours >= 3)).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        pieces, _ = label(skeleton & ~junctions, structure=np.ones((3, 3)))
        found = []
        for index, box in enumerate(find_objects(pieces), start=1):
            inside = pieces[box] == index
            if inside.sum() < min_length:
                continue
            path = _order(np.argwhere(inside) + [box[0].start, box[1].start])
            if path is None:
                continue
            path = gaussian_filter1d(path, smooth, axis=0, mode="nearest")
            arc = np.r_[0, np.cumsum(np.hypot(*np.diff(path, axis=0).T))]
            if arc[-1] < min_length:
                continue
            at = np.arange(0, arc[-1], 1.0)
            found.append(np.stack([np.interp(at, arc, path[:, 0]), np.interp(at, arc, path[:, 1])], axis=1))
        return found

    return (centrelines,)


@app.cell
def _(
    MIN_SEGMENT,
    UM_PER_PX,
    analysed,
    centrelines,
    np,
    perfused,
    plt,
    show_figure,
    skeletonize,
    still_gray,
    still_picture,
    still_range,
):
    segments = centrelines(perfused, MIN_SEGMENT)
    _lengths = np.array([len(_s) for _s in segments])
    _skeleton_length = skeletonize(perfused).sum()
    _density = _skeleton_length / analysed.sum()
    if segments:
        print(f"{len(segments)} segments of at least {MIN_SEGMENT} px: median {np.median(_lengths):.0f} px, longest "
              f"{_lengths.max():.0f} px, {_lengths.sum()} px in all ({_lengths.sum() / _skeleton_length:.0%} of the skeleton)")
    else:
        print(f"No segment of at least {MIN_SEGMENT} px: nothing to measure in §4 to §6")
    print(f"Perfused length density: {_density:.4f} px of centreline per px of field"
          + (f" = {_density / (UM_PER_PX * 1e-3):.1f} mm/mm²" if UM_PER_PX else ""))

    _figure, _axis = plt.subplots(figsize=(14, 9))
    _axis.imshow(still_picture, cmap=still_gray, vmin=still_range[0], vmax=still_range[1])
    _colors = plt.cm.tab20(np.arange(len(segments)) % 20)
    for _index, _segment in enumerate(segments):
        _axis.plot(_segment[:, 1], _segment[:, 0], color=_colors[_index], lw=1.2)
        _axis.text(_segment[len(_segment) // 2, 1], _segment[len(_segment) // 2, 0], str(_index), fontsize=6, color="w")
    _axis.set_title("centreline segments, numbered")
    _axis.axis("off")
    _figure.tight_layout()
    show_figure(_figure, "centrelines")
    return (segments,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · How fast: kymographs and the slope of their streaks

    **The kymograph.** For a segment with centreline $\mathbf{r}(s)$, one point per pixel of length, the kymograph is the mean-subtracted video read along it, averaged over the centreline and the lines 1 px to either side. Each frame is one row. The temporal mean of every pixel is already gone, so the vessel walls and anything else that does not move leave nothing behind. It is read from `high_passed`, with the rolling mean over `DRIFT_WINDOW` frames removed as well: read from `mean_removed`, patches of the vessel that brighten and darken slowly draw vertical streaks, which can put the peak at $v = 0$ on a segment in which the cross-correlation finds clear flow. The price is that flow slower than about a pattern length per `DRIFT_WINDOW` frames is weakened (0.2 px/frame for a 3 px pattern and the default window; §6 measures the pattern length). Each row's mean along the segment is subtracted as well, which removes what changes the whole segment at once. Red-cell aggregates and plasma gaps carried by the flow then draw streaks: $K(t, s) \approx f(s - v t)$, and the slope $ds/dt$ of the streaks is the velocity $v$ in px/frame.

    **The slope search.** The Radon transform integrates $K$ along every line $s = s_0 + v t$; along the streaks' slope the integrals vary most from one $s_0$ to the next. The variance of those integrals is a sum, over every pair of rows $t, t'$, of the correlation of row $t$ with row $t'$ shifted by $v (t' - t)$. The search keeps the part of that sum that carries the flow:

    $$
    C(v) \;=\; \frac{1}{\text{lags}}\sum_{\ell = 1}^{\texttt{MAX\_LAG}} \frac{\langle K(t, s)\, K(t + \ell, s + v\ell) \rangle_{t, s}}{\langle K^2 \rangle},
    $$

    leaving out lag 0 (each row with itself, which is the noise) and the lags at which a shifted row overlaps the segment by fewer than `MIN_OVERLAP` px: on a short segment a fast pattern leaves after a few frames, and a plain Radon integral along the whole burst would run out of the segment. $v$ is searched up to `MAX_VELOCITY`, and on each segment only as far as two lags still fit, $(L - \texttt{MIN\_OVERLAP})/2$ for a segment of length $L$: the **reach** of that segment. The peak of $C$ is found on a 0.25 px/frame grid, refined on a 0.05 px/frame grid around it, and interpolated by a parabola. Using several lags at once also resolves the ambiguity of a pattern that repeats along the vessel, which a single pair of frames cannot: a false match one repeat further on does not line up across the lags.

    **Uncertainty.** The burst is split into `BLOCKS` blocks of time and the slope searched in each; the uncertainty is the standard error of the block estimates, $\mathrm{sd}/\sqrt{\texttt{BLOCKS}}$. It includes the real changes of flow over the burst (the pulse).

    A velocity is **positive when the blood runs from the segment's first point to its last**; the direction in the image is given as well. The synthetic test runs the same code on kymographs of known velocity, with a random pattern of 3 px correlation length and noise as large as the pattern (σ = 1) or twice as large (σ = 2), on a 40 px segment.
    """)
    return


@app.cell
def _(MAX_LAG, MIN_OVERLAP, cv2, gaussian_filter1d, np):
    def kymograph(video, line, half_width=1):
        """K(t, s): the video read along a centreline (length, 2) of (row, column), averaged across +-half_width px."""
        tangent = np.gradient(line, axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
        normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
        points = line[None] + np.arange(-half_width, half_width + 1)[:, None, None] * normal[None]
        map_x, map_y = points[..., 1].astype(np.float32), points[..., 0].astype(np.float32)
        return np.stack([cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=float("nan")).mean(axis=0) for frame in video])


    def clean_kymograph(K):
        """Subtract the mean over time of every position and the mean along the segment of every frame; return the
        result with the missing values set to 0, and the share that was missing."""
        missing = float(np.isnan(K).mean())
        K = K - np.nanmean(K, axis=0, keepdims=True)
        K = K - np.nanmean(K, axis=1, keepdims=True)
        return np.nan_to_num(K).astype(np.float32), missing


    def slope_scores(K, velocities, max_lag=MAX_LAG, min_overlap=MIN_OVERLAP):
        """C(v) for each slope v (px/frame): the mean over the lags that fit of the correlation of each row with the
        row `lag` frames later, that one shifted by v * lag along the segment. NaN where no lag fits."""
        frames, length = K.shape
        velocities = np.asarray(velocities, dtype=float)
        power = float(np.mean(K ** 2)) or 1.0
        positions = np.arange(length)
        total = np.zeros(len(velocities))
        lags = np.zeros(len(velocities))
        for lag in range(1, min(max_lag, frames - 1) + 1):
            target = positions[None, :] + velocities[:, None] * lag          # (velocities, length)
            inside = (target >= 0) & (target <= length - 1)
            count = inside.sum(axis=1)
            fits = count >= min_overlap   # fewer positions fit at every longer lag, so this only ever switches off
            if not fits.any():
                break
            lower = np.clip(np.floor(target).astype(int), 0, length - 1)
            upper = np.minimum(lower + 1, length - 1)
            fraction = (target - lower).astype(np.float32)
            later = K[lag:][:, lower] * (1 - fraction) + K[lag:][:, upper] * fraction   # (frames - lag, velocities, length)
            products = np.einsum("tl,tvl->vl", K[:-lag], later) * inside
            total += np.where(fits, products.sum(axis=1) / np.maximum(count, 1) / (frames - lag) / power, 0.0)
            lags += fits
        return np.where(lags > 0, total / np.maximum(lags, 1), np.nan)


    def _parabola_peak(velocities, scores):
        best = int(np.nanargmax(scores))
        if 0 < best < len(scores) - 1 and np.all(np.isfinite(scores[best - 1: best + 2])):
            left, middle, right = scores[best - 1: best + 2]
            curvature = left - 2 * middle + right
            if curvature < 0:
                return velocities[best] + 0.5 * (left - right) / curvature * (velocities[1] - velocities[0]), middle
        return velocities[best], scores[best]


    def find_slope(K, reach):
        """The slope of the streaks within +-reach px/frame: coarse search, fine search around its peak, parabola.
        Returns the velocity, the peak of C, and the coarse grid with its scores."""
        coarse = np.arange(-reach, reach + 1e-9, 0.25)
        coarse_scores = slope_scores(K, coarse)
        if not np.isfinite(coarse_scores).any():
            return np.nan, np.nan, coarse, coarse_scores
        centre = coarse[int(np.nanargmax(coarse_scores))]
        fine = np.arange(max(-reach, centre - 0.5), min(reach, centre + 0.5) + 1e-9, 0.05)
        velocity, height = _parabola_peak(fine, slope_scores(K, fine))
        return float(velocity), float(height), coarse, coarse_scores


    def synthetic_kymograph(velocity, frames=118, length=40, correlation=3.0, noise=1.0, seed=0):
        """A random pattern of unit variance moving at `velocity` px/frame, plus white noise of SD `noise`."""
        rng = np.random.default_rng(seed)
        span = length + 2 * abs(velocity) * frames + 40
        pattern = gaussian_filter1d(rng.normal(size=int(span)), correlation)
        pattern /= pattern.std()
        start = abs(velocity) * frames + 20
        rows = [np.interp(np.arange(length) - velocity * t + start, np.arange(len(pattern)), pattern) for t in range(frames)]
        return np.array(rows) + noise * rng.normal(size=(frames, length))

    return (
        clean_kymograph,
        find_slope,
        kymograph,
        synthetic_kymograph,
    )


@app.cell
def _(
    MIN_OVERLAP,
    clean_kymograph,
    find_slope,
    frame_count,
    synthetic_kymograph,
):
    print("true v | noise | slope search   (px/frame, 40 px segment, "
          f"{frame_count} frames, reach {(40 - MIN_OVERLAP) / 2:.0f} px/frame)")
    synthetic_results = []
    for _true in (0.2, 1.0, 3.0, -5.0, 8.0, 12.0):
        for _noise in (1.0, 2.0):
            _K, _ = clean_kymograph(synthetic_kymograph(_true, frames=frame_count, noise=_noise))
            _found = find_slope(_K, (40 - MIN_OVERLAP) / 2)[0]
            synthetic_results.append((_true, _noise, _found))
            print(f"{_true:6.1f} | {_noise:5.1f} | {_found:12.2f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · The cross-check: 2D cross-correlation of the frames

    An estimate that does not go through the kymograph or the centreline's exact position: ensemble particle image velocimetry on the frames themselves. The segment is cut into pieces of at most 48 px. For each piece, every frame of `high_passed` inside a tube 3 px either side of the centreline is cross-correlated with the next frame within `MAX_VELOCITY` + 1 px around it, and the correlations of all pairs of frames are added up before the peak is looked for (the ensemble average, which finds a peak where single pairs are too noisy). `high_passed` rather than `mean_removed`, so that slow drift adds no peak at zero shift. The peak, interpolated by a parabola in each direction, is the displacement per frame; its component along the piece is the velocity, and its component across the piece should be close to 0. The pieces' velocities are averaged, weighted by length.

    | Output | Content |
    |---|---|
    | `velocities` | one row per measured segment (§4, §5): the columns are explained in §6 |
    """)
    return


@app.cell
def _(cv2, np):
    def tube(shape, line, half_width):
        mask = np.zeros(shape, np.uint8)
        cv2.polylines(mask, [np.round(line[:, ::-1]).astype(np.int32).reshape(-1, 1, 2)], False, 1, thickness=2 * half_width + 1)
        return mask.astype(bool)


    def ensemble_shift(video, line, reach, half_width=3, lag=1):
        """The displacement (rows, columns) per frame of the summed cross-correlation of the tube around `line` in each
        frame with its surroundings `lag` frames later, and the peak's height as a correlation coefficient."""
        rows, columns = video.shape[1:]
        margin = int(np.ceil(reach)) + half_width + 2
        top, left = np.maximum(np.floor(line.min(axis=0)).astype(int) - margin, 0)
        bottom, right = np.minimum(np.ceil(line.max(axis=0)).astype(int) + margin, (rows, columns))
        part = video[:, top:bottom, left:right]
        has_data = np.isfinite(part)
        part = np.where(has_data, part, 0)
        inner = tube(part.shape[1:], line - [top, left], half_width)
        outer = cv2.dilate(inner.astype(np.uint8), np.ones((2 * margin + 1,) * 2, np.uint8)).astype(bool)
        size = (2 * part.shape[1], 2 * part.shape[2])
        first = part[:-lag] * inner
        second = part[lag:] * outer
        # the cross-power spectra of all the pairs of frames, added up: one inverse transform for the whole ensemble
        cross = (np.conj(np.fft.rfft2(first, size)) * np.fft.rfft2(second, size)).sum(axis=0)
        total = np.fft.irfft2(cross, size)
        energy_inner, energy_outer = float((first ** 2).sum()), float((second ** 2).sum())
        if energy_inner == 0:
            return np.array([np.nan, np.nan]), np.nan
        correlation = np.fft.fftshift(total)
        centre = (size[0] // 2, size[1] // 2)
        r = int(np.ceil(reach))
        window = correlation[centre[0] - r: centre[0] + r + 1, centre[1] - r: centre[1] + r + 1]
        window = window / np.sqrt(energy_inner * energy_outer * inner.sum() / outer.sum())
        best = np.unravel_index(np.argmax(window), window.shape)
        shift = np.array(best, dtype=float) - r
        for axis in (0, 1):
            if 0 < best[axis] < window.shape[axis] - 1:
                step = np.eye(2, dtype=int)[axis]
                left_value, middle, right_value = window[tuple(best - step)], window[best], window[tuple(best + step)]
                curvature = left_value - 2 * middle + right_value
                if curvature < 0:
                    shift[axis] += 0.5 * (left_value - right_value) / curvature
        return shift / lag, float(window[best])


    def segment_cross_check(video, line, reach, piece=48):
        """The velocity along `line` (px/frame, positive towards its end) and across it, from ensemble_shift on pieces."""
        count = max(1, int(np.ceil(len(line) / piece)))
        bounds = np.linspace(0, len(line), count + 1).astype(int)
        along, across, weights = [], [], []
        for start, stop in zip(bounds[:-1], bounds[1:]):
            part = line[start:stop]
            if len(part) < 8:
                continue
            direction = part[-1] - part[0]
            direction /= np.linalg.norm(direction)
            shift, height = ensemble_shift(video, part, reach)
            if not np.isfinite(height):
                continue
            along.append(shift @ direction)
            across.append(shift @ np.array([-direction[1], direction[0]]))
            weights.append(len(part))
        if not weights:
            return np.nan, np.nan
        return float(np.average(along, weights=weights)), float(np.average(across, weights=weights))

    return (segment_cross_check,)


@app.cell
def _(
    BLOCKS,
    MAX_VELOCITY,
    MIN_OVERLAP,
    ThreadPoolExecutor,
    clean_kymograph,
    find_slope,
    high_passed,
    kymograph,
    np,
    segment_cross_check,
    segments,
    time,
    workers,
):
    def measure(index):
        """Everything §4 and §5 measure on one segment, or None when its kymograph misses too much data."""
        line = segments[index]
        K, missing = clean_kymograph(kymograph(high_passed, line))
        if missing > 0.1:
            return None
        reach = min(MAX_VELOCITY, (len(line) - MIN_OVERLAP) / 2)
        velocity, height, grid, scores = find_slope(K, reach)
        if not np.isfinite(velocity):
            return None
        mirror = float(np.interp(-velocity, grid, np.nan_to_num(scores)))
        edges = np.linspace(0, len(K), BLOCKS + 1).astype(int)
        blocks = np.array([find_slope(K[a:b], reach)[0] for a, b in zip(edges[:-1], edges[1:])])
        along, across = segment_cross_check(high_passed, line, MAX_VELOCITY + 1)
        middle = len(line) // 2
        tangent = line[min(middle + 3, len(line) - 1)] - line[max(middle - 3, 0)]
        tangent = tangent / np.linalg.norm(tangent) * np.sign(velocity)
        return {
            "segment": index, "length_px": len(line), "reach": reach,
            "v": velocity, "v_se": float(np.std(blocks, ddof=1) / np.sqrt(BLOCKS)), "blocks": blocks,
            "peak": height, "mirror": mirror,
            "v_xcorr": along, "across_xcorr": across,
            # the flow's direction in the picture, in degrees counter-clockwise from the +x axis (rows grow downwards)
            "direction_deg": float(np.degrees(np.arctan2(-tangent[0], tangent[1]))),
            "kymograph": K, "grid": grid, "scores": scores,
        }


    _start = time.perf_counter()
    with ThreadPoolExecutor(workers) as _pool:
        measurements = [_m for _m in _pool.map(measure, range(len(segments))) if _m is not None]
    print(f"Measured {len(measurements)} of {len(segments)} segments in {time.perf_counter() - _start:.0f} s "
          f"({len(segments) - len(measurements)} miss more than 10% of their kymograph)")
    return (measurements,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · The velocities

    **Which measurements count.** A segment's velocity is

    - **out of reach** when the peak lies within 0.5 px/frame of the segment's reach: the flow may be faster than the segment can show;
    - **weak** when the peak correlation $C$ is below 0.1, or less than 0.05 above $C$ at the opposite velocity (no clear direction);
    - **inconsistent** when it is neither, but the blocks of time do not all give the peak's direction, or the standard error is above both 0.3 px/frame and 30% (or cannot be given, because a block found no slope);
    - **consistent** when it is none of these: every block of time gives the same direction, and the standard error is at most 0.3 px/frame or 30%;
    - **confirmed** when it is consistent and the cross-correlation of §5 agrees within 0.5 px/frame or 30%.

    The table has one row per measured segment. `v` is the slope search in px/frame (positive from the segment's first point to its last) with its standard error `v_se`, also in px/s and, when `UM_PER_PX` is set, in mm/s; `direction_deg` is the direction of the flow in the picture, counter-clockwise from the +x axis; `peak` is $C$ at the peak and `mirror` $C$ at the opposite velocity, so `peak` well above `mirror` means the direction is clear; `v_xcorr` is the cross-correlation's estimate of §5, and `across_xcorr` its displacement across the vessel.
    """)
    return


@app.cell
def _(UM_PER_PX, fps, measurements, mo, np):
    def status(row):
        if abs(row["v"]) > row["reach"] - 0.5:
            return "out of reach"
        if row["peak"] < 0.1 or row["peak"] - row["mirror"] < 0.05:
            return "weak"
        same_direction = np.all(np.sign(row["blocks"]) == np.sign(row["v"]))
        if not (same_direction and row["v_se"] <= max(0.3, 0.3 * abs(row["v"]))):
            return "inconsistent"
        if np.isfinite(row["v_xcorr"]) and abs(row["v_xcorr"] - row["v"]) <= max(0.5, 0.3 * abs(row["v"])):
            return "confirmed"
        return "consistent"


    velocities = []
    for _row in measurements:
        _entry = {
            "segment": _row["segment"], "length_px": _row["length_px"], "status": status(_row),
            "v_px_frame": round(_row["v"], 2), "v_se": round(_row["v_se"], 2),
            "v_px_s": round(_row["v"] * fps, 1), "v_se_px_s": round(_row["v_se"] * fps, 1),
            "direction_deg": round(_row["direction_deg"]),
            "peak": round(_row["peak"], 3), "mirror": round(_row["mirror"], 3),
            "v_xcorr": round(_row["v_xcorr"], 2),
            "across_xcorr": round(_row["across_xcorr"], 2), "reach": round(_row["reach"], 1),
        }
        if UM_PER_PX:
            _entry["v_mm_s"] = round(_row["v"] * fps * UM_PER_PX / 1000, 3)
            _entry["v_se_mm_s"] = round(_row["v_se"] * fps * UM_PER_PX / 1000, 3)
        velocities.append(_entry)
    _counts = {_s: sum(_v["status"] == _s for _v in velocities) for _s in ("confirmed", "consistent", "inconsistent", "weak", "out of reach")}
    print(" | ".join(f"{_s}: {_n}" for _s, _n in _counts.items()))
    mo.ui.table(velocities, page_size=15, selection=None)
    return (velocities,)


@app.cell
def _(INK_MUTED, SERIES, measurements, np, plt, show_figure, velocities):
    _by_segment = {_row["segment"]: _row for _row in measurements}
    _confirmed = sorted((_v for _v in velocities if _v["status"] == "confirmed"), key=lambda _v: abs(_v["v_px_frame"]))
    # six examples spread over the range of speeds
    _picked = [_confirmed[int(round(_q * (len(_confirmed) - 1)))] for _q in np.linspace(0, 1, min(6, len(_confirmed)))]
    _picked = list({_v["segment"]: _v for _v in _picked}.values())
    if _picked:
        _figure, _axes = plt.subplots(2, len(_picked), figsize=(3.2 * len(_picked), 7.5), squeeze=False,
                                      gridspec_kw={"height_ratios": [3, 1]})
        for _column, _entry in enumerate(_picked):
            _row = _by_segment[_entry["segment"]]
            _K, _v = _row["kymograph"], _row["v"]
            _limit = np.percentile(np.abs(_K), 99)
            _axis = _axes[0, _column]
            _axis.imshow(_K, cmap="gray", vmin=-_limit, vmax=_limit, aspect="auto", interpolation="nearest")
            for _s0 in np.linspace(-abs(_v) * len(_K), _K.shape[1] + abs(_v) * len(_K), 14):
                _t = np.array([0, len(_K) - 1])
                _axis.plot(_s0 + _v * _t, _t, color=SERIES[1], lw=0.8, alpha=0.7)
            _axis.set_xlim(-0.5, _K.shape[1] - 0.5)
            _axis.set_ylim(len(_K) - 0.5, -0.5)
            _axis.set_title(f"segment {_entry['segment']}: {_v:.2f} ± {_row['v_se']:.2f} px/frame\n"
                            f"cross-correlation {_row['v_xcorr']:.2f}", fontsize=9)
            _axis.set_xlabel("position along the segment (px)")
            _axis.set_ylabel("frame" if _column == 0 else "")
            _axis.grid(False)
            _bottom = _axes[1, _column]
            _bottom.plot(_row["grid"], _row["scores"], color=SERIES[0])
            _bottom.axvline(_v, color=SERIES[1], lw=0.8)
            _bottom.set_xlabel("v (px/frame)")
            _bottom.set_ylabel("C(v)" if _column == 0 else "")
        _figure.suptitle("kymographs of the mean-subtracted video with the fitted slope (orange), and the slope search "
                         "below each", fontsize=10, color=INK_MUTED)
        _figure.tight_layout()
        show_figure(_figure, "kymographs")
    else:
        print("No confirmed segment to show")
    return


@app.cell
def _(INK_MUTED, SERIES, np, plt, show_figure, velocities):
    _colors = {"confirmed": SERIES[2], "consistent": SERIES[0], "inconsistent": SERIES[3], "weak": INK_MUTED, "out of reach": SERIES[7]}
    _figure, _axis = plt.subplots(figsize=(6, 4.6))
    for _status, _color in _colors.items():
        _rows = [_v for _v in velocities if _v["status"] == _status]
        _axis.scatter([_v["v_px_frame"] for _v in _rows], [_v["v_xcorr"] for _v in _rows], s=12, color=_color,
                      label=f"{_status} ({len(_rows)})")
    _top = max(abs(_v["v_px_frame"]) for _v in velocities) + 1 if velocities else 1
    _axis.plot([-_top, _top], [-_top, _top], ":", color=INK_MUTED)
    _axis.set_xlim(-_top, _top)
    _axis.set_ylim(-_top, _top)
    _axis.set_xlabel("slope search, v (px/frame)")
    _axis.set_ylabel("2D cross-correlation (§5) (px/frame)")
    _axis.legend(fontsize=8)
    _figure.tight_layout()
    show_figure(_figure, "velocity_agreement")
    _agreeing = [_v for _v in velocities if _v["status"] in ("confirmed", "consistent") and np.isfinite(_v["v_xcorr"])]
    if _agreeing:
        _difference = np.array([_v["v_xcorr"] - _v["v_px_frame"] for _v in _agreeing])
        print(f"Consistent segments: cross-correlation minus slope search, median {np.median(_difference):+.2f} px/frame, "
              f"median absolute {np.median(np.abs(_difference)):.2f} px/frame; displacement across the vessel, median "
              f"absolute {np.median([abs(_v['across_xcorr']) for _v in _agreeing]):.2f} px/frame")
    return


@app.cell
def _(
    UM_PER_PX,
    fps,
    np,
    plt,
    segments,
    show_figure,
    still_gray,
    still_picture,
    still_range,
    velocities,
):
    _figure, _axis = plt.subplots(figsize=(15, 9.5))
    _axis.imshow(still_picture, cmap=still_gray, vmin=still_range[0], vmax=still_range[1])
    _scale = fps * UM_PER_PX / 1000 if UM_PER_PX else fps
    _unit = "mm/s" if UM_PER_PX else "px/s"
    _measured = [_v for _v in velocities if _v["status"] in ("confirmed", "consistent")]
    _top = np.percentile([abs(_v["v_px_frame"]) * _scale for _v in _measured], 95) if _measured else 1
    _map = plt.cm.viridis
    for _v in velocities:
        _line = segments[_v["segment"]]
        if _v not in _measured:
            _axis.plot(_line[:, 1], _line[:, 0], color="#c3c2b7", lw=0.6, alpha=0.6)
            continue
        _speed = abs(_v["v_px_frame"]) * _scale
        _axis.plot(_line[:, 1], _line[:, 0], color=_map(min(_speed / _top, 1)), lw=2.2 if _v["status"] == "confirmed" else 1.2)
        _middle = len(_line) // 2
        _step = (_line[min(_middle + 2, len(_line) - 1)] - _line[max(_middle - 2, 0)]) * np.sign(_v["v_px_frame"])
        _axis.annotate("", xy=(_line[_middle, 1] + _step[1] * 2, _line[_middle, 0] + _step[0] * 2),
                       xytext=(_line[_middle, 1], _line[_middle, 0]),
                       arrowprops={"arrowstyle": "-|>", "color": "w", "lw": 0.8, "mutation_scale": 9})
    _bar = _figure.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, _top), cmap=_map), ax=_axis, fraction=0.025)
    _bar.set_label(f"red-cell speed ({_unit})")
    _axis.set_title("velocity map: consistent segments (thick: confirmed by the cross-correlation), arrows along the flow; "
                    "gray: not measured reliably")
    _axis.axis("off")
    _figure.tight_layout()
    show_figure(_figure, "velocity_map")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What the frame rate allows

    Three limits decide which velocities this recording can measure:

    - **Fast flow.** A pattern has to stay in the segment for at least two frames: each segment's reach, $(L - \texttt{MIN\_OVERLAP})/2$ px/frame, caps what it can show. Within a frame the exposure smears the pattern by $v \times$ exposure: once the smear is longer than the pattern itself (its correlation length along the vessel, measured on the kymographs), the streaks lose contrast. The smear is the same in every frame, so it does not bias the displacement, but it leaves less pattern to follow.
    - **The sampling theorem.** At a fixed pixel, a pattern of correlation length $\lambda$ passing at $v$ px/frame changes at about $v / \lambda$ cycles per frame, so the time series of one pixel is sampled without aliasing only while $v < \lambda / 2$ (the Nyquist frequency, half the frame rate). The perfusion map's variance does not mind aliasing (it counts the power wherever it lands); the slope search and the cross-correlation compare whole stretches of the vessel between frames, and only a pattern that repeats along the vessel at the distance it travels in one frame can fool them, which the several lags guard against.
    - **Slow flow.** Over the burst a pattern has to move by a few pixels to draw a measurable slope; the standard errors give the precision.
    """)
    return


@app.cell
def _(
    SERIES,
    exposure_s,
    fps,
    frame_count,
    frame_period,
    measurements,
    mo,
    np,
    plt,
    show_figure,
    spectrum_frequencies,
    spectrum_moving,
    spectrum_static,
    velocities,
):
    def correlation_length(K):
        """Half width at half maximum of the kymograph's autocorrelation along the segment (px)."""
        spectrum = np.abs(np.fft.rfft(K, n=2 * K.shape[1], axis=1)) ** 2
        autocorrelation = np.fft.irfft(spectrum.mean(axis=0))[: K.shape[1]]
        autocorrelation /= autocorrelation[0]
        below = np.nonzero(autocorrelation < 0.5)[0]
        if not len(below):
            return np.nan
        k = below[0]
        return float(k - 1 + (autocorrelation[k - 1] - 0.5) / (autocorrelation[k - 1] - autocorrelation[k]))


    _good = {_v["segment"] for _v in velocities if _v["status"] in ("confirmed", "consistent")}
    _rows = [_m for _m in measurements if _m["segment"] in _good]
    _lengths = np.array([correlation_length(_m["kymograph"]) for _m in _rows])
    pattern_length = float(np.nanmedian(_lengths)) if len(_lengths) else np.nan
    _duty = exposure_s / frame_period if exposure_s else np.nan
    _speeds = np.array([abs(_m["v"]) for _m in _rows])
    _errors = np.array([_m["v_se"] for _m in _rows])
    _reaches = np.array([_m["reach"] for _m in measurements])
    _out_of_reach = sum(_v["status"] == "out of reach" for _v in velocities)
    _nyquist = spectrum_frequencies[-1]

    _figure, _axis = plt.subplots(figsize=(8, 3.6))
    if len(_speeds):
        _axis.hist(_speeds, bins=np.arange(0, max(_speeds.max(), 1) + 0.5, 0.5), color=SERIES[0], alpha=0.8,
                   label="consistent segments")
    _marks = [(np.median(_errors) if len(_errors) else np.nan, "median uncertainty", SERIES[2]),
              (pattern_length / 2, "pattern length / 2 (Nyquist at one pixel)", SERIES[3]),
              (pattern_length / _duty if np.isfinite(_duty) else np.nan, "smear = pattern length", SERIES[7]),
              (np.median(_reaches) if len(_reaches) else np.nan, "median reach of a segment", SERIES[1])]
    for _value, _name, _color in _marks:
        if np.isfinite(_value):
            _axis.axvline(_value, color=_color, ls="--", lw=1, label=f"{_name}: {_value:.1f}")
    _axis.set_xlabel("speed (px/frame)")
    _axis.set_ylabel("segments")
    _axis.legend(fontsize=8)
    _figure.tight_layout()
    show_figure(_figure, "speed_limits")

    _faster_than_nyquist = float(np.mean(_speeds > pattern_length / 2)) if len(_speeds) else np.nan
    _smear_limit = pattern_length / _duty if np.isfinite(_duty) else np.nan
    _smeared = float(np.mean(_speeds > _smear_limit)) if len(_speeds) and np.isfinite(_smear_limit) else np.nan
    _sentences = [f"{frame_count} frames at {fps:.1f} fps, {frame_period * 1e3:.1f} ms apart"
                  + (f", each exposed for {exposure_s * 1e3:.1f} ms ({_duty:.0%} of the period): a pattern moving $v$ "
                     f"px/frame is smeared over {_duty:.2f} $v$ px within a frame" if exposure_s else "") + "."]
    if len(_speeds):
        _sentences.append(
            f"The flow patterns along the vessels have a correlation length of {pattern_length:.1f} px (median over the "
            f"consistent segments), so a single pixel's time series is free of aliasing only below "
            f"{pattern_length / 2:.1f} px/frame ({pattern_length / 2 * fps:.0f} px/s); {_faster_than_nyquist:.0%} of the "
            "consistent segments flow faster.")
        if np.isfinite(_smear_limit):
            _sentences.append(
                f"The smear is as long as the pattern itself at {_smear_limit:.1f} px/frame; {_smeared:.0%} of the "
                "consistent segments flow faster than that"
                + (" and are measured all the same." if _smeared > 0 else "."))
        _sentences.append(
            f"Speeds measured: median {np.median(_speeds):.2f} px/frame, 90th percentile {np.percentile(_speeds, 90):.2f} "
            f"px/frame, fastest {_speeds.max():.2f} px/frame; median uncertainty {np.median(_errors):.2f} px/frame.")
    else:
        _sentences.append("No segment was measured consistently, so there is no pattern length and no speed to report.")
    _sentences.append(
        f"At the Nyquist frequency, {_nyquist:.1f} Hz, the spectrum of §2 gives the moving pixels "
        f"{spectrum_moving[-1] / spectrum_static[-1]:.1f} times the power of the static ones; a ratio above 1 there means "
        "the flow has power up to the Nyquist frequency, and whatever it has above it the frame rate folds back.")
    if len(_reaches):
        _sentences.append(
            f"The reach of a segment is {np.median(_reaches):.0f} px/frame in the median, and {_out_of_reach} segments "
            "peak at their reach" + (": their flow may be faster than they can show." if _out_of_reach else "."))
    _sentences.append("Faster flow needs a higher frame rate: a shorter region of interest raises it (the reference bursts "
                      "include strips at 74 and 149 fps), and a shorter exposure reduces the smear.")
    mo.md("**On this burst.** " + " ".join(_sentences))
    return (pattern_length,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 · Save the results, and the runtime

    Everything goes to `flow_outputs/<burst folder>/` in the working directory (git ignores `notebooks/flow_outputs/`):

    | File | Content |
    |---|---|
    | `recorded_vs_mean_subtracted.webm` | the side-by-side video of §1 |
    | `frame_comparison.png`, `separation_by_frames.png` | §1 |
    | `temporal_spectrum.png`, `perfusion_map.png`, `centrelines.png` | §2, §2, §3 |
    | `kymographs.png`, `velocity_agreement.png`, `velocity_map.png`, `speed_limits.png` | §6 |
    | `velocities.csv` | the table of §6 |
    | `flow_maps.npz` | `perfusion_z`, `perfused`, `analysed`, the centrelines (`segment_points`, rows of `(segment, row, column)`), `fps`, `fixation`, `burst`, `pattern_length` (px, the median over the consistent segments of §6; `NaN` when there is none) |
    | `contrast.csv` | the contrast table of §1 |
    """)
    return


@app.cell
def _(
    analysed,
    contrast_table,
    csv,
    foundation_seconds,
    fps,
    notebook_start,
    np,
    number,
    output_folder,
    pattern_length,
    perfused,
    perfusion_z,
    resource,
    segments,
    target_folder,
    time,
    velocities,
):
    output_folder.mkdir(parents=True, exist_ok=True)
    if velocities:
        with open(output_folder / "velocities.csv", "w", newline="", encoding="utf-8") as _file:
            _writer = csv.DictWriter(_file, fieldnames=list(velocities[0]))
            _writer.writeheader()
            _writer.writerows(velocities)
    with open(output_folder / "contrast.csv", "w", newline="", encoding="utf-8") as _file:
        _writer = csv.writer(_file)
        _columns = list(next(iter(contrast_table.values())))
        _writer.writerow(["version", *_columns])
        for _name, _values in contrast_table.items():
            _writer.writerow([_name, *(f"{_values[_c]:.6g}" for _c in _columns)])
    _points = np.concatenate([np.column_stack([np.full(len(_s), _i), _s]) for _i, _s in enumerate(segments)]) \
        if segments else np.zeros((0, 3))
    np.savez_compressed(output_folder / "flow_maps.npz", perfusion_z=perfusion_z, perfused=perfused, analysed=analysed,
                        segment_points=_points.astype(np.float32), fps=float(fps), fixation=number, burst=str(target_folder),
                        pattern_length=pattern_length)
    for _path in sorted(output_folder.iterdir()):
        print(f"{_path.name:40s} {_path.stat().st_size / 1e6:6.2f} MB")
    _elapsed = time.perf_counter() - notebook_start
    _peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # kB on Linux
    print(f"\nThis notebook ran in {_elapsed:.0f} s, of which the preprocessing took {foundation_seconds:.0f} s; "
          f"peak memory of this process {_peak:.1f} GB")
    return


if __name__ == "__main__":
    app.run()
