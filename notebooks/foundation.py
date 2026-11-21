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
    # Preprocessing: the burst, held still and cleaned

    This notebook takes one burst of the limbus camera from its TIFF frames to the pictures that vessel annotation starts from. It drops the blurred frames, follows the eye's motion, holds the vessels still, removes dust and glare, fits the background and measures every pixel's optical density against it, separates what changes over time from what does not, and combines the still picture with the picture of what moves. §17 saves the result to one file, so that an annotation notebook can start from it without running this one.

    There is one section per step. Each starts with what the step does, the settings it uses and the names it hands on; each fixation's frames are kept as a stack of their own (`raw_fixations`, `stabilized_fixations`, …), static check figures stand in for most of the videos, and the remaining previews are small. The vessel annotation that used to follow §16 is in [`archive_vessel_annotation.py`](archive_vessel_annotation.py).

    ### The pipeline at a glance

    | § | Step | Main outputs |
    |---|---|---|
    | 0 | Setup | imports, settings, `create_animation()` |
    | 1 | Load the burst | `raw_burst`, `fps`, `target_folder` |
    | 2 | Drop the blurred frames | `gated_frames`, `gated_indices`, `frame_times` |
    | 3 | Measure the motion between frames | `step_sizes`, `frame_positions` |
    | 4 | Split the burst into fixations | `fixation_members`, **`raw_fixations`** |
    | 5 | Align the frames of each fixation | `fixations` |
    | 6 | Align the fixations to each other | `anchor`, `aligned_fixation_averages` |
    | 7 | Remove dust and glare | `sensor_map`, `glare_pixels`, `clean_frame()` |
    | 8 | Stabilize every frame | **`stabilized_fixations`**, `stabilized_frames` |
    | 9 | The mean frame of each fixation | `fixation_means` |
    | 10 | Fit the background, measure the density | `background_fits`, `density_fixations`, `density_means` |
    | 11 | Find the quiet pixels | `fluctuation_ratio`, `quiet_pixels`, `changing_pixels` |
    | 12 | Refit the background on the quiet pixels | `final_background_fits`, **`final_density_fixations`** |
    | 13 | Follow the shading from frame to frame | `frame_shadings`, `temporal_frame()` |
    | 14 | The smear of the temporal burst | `burst_smears` |
    | 15 | Clean the smear's background | `smear_fits`, `cleaned_smears` |
    | 16 | Combine the two vessel pictures | **`combined_sums`** |
    | 17 | Save the result | `preprocessed/<burst>.npz` |

    Ingestion and stabilization proper are §1–§8. §9–§16 turn the stabilized stacks into the pictures that vessel annotation reads.

    ### The background comes first

    The background is fitted right after stabilization, before any temporal mean is subtracted. Flicker multiplies a whole frame by one factor, which is a constant in log space, and centring each frame's density on its background pixels removes it (§10 shows the check). So there is no flicker step, and the temporal mean is subtracted in density instead (§11, §13).

    ### Conventions

    - **Fixation**: a run of frames between two saccades, in which the eye only jitters. Per-fixation outputs are lists indexed by the fixation's number `n`.
    - **Reference frame**: the one frame of a fixation that its other frames are aligned to. Every per-fixation stack and picture lies in its own fixation's reference frame.
    - **Anchor**: the fixation with the most frames (`anchor`). Its reference frame is the common frame of the whole burst; `stabilized_frames` and the previews use it.
    - **Frame numbers** are positions in `gated_frames`, not in `raw_burst`. `gated_indices[i]` is frame `i`'s number in the recording.
    - **Shifts** are `(dy, dx)` in pixels, rows first. **Maps** such as `fixations[n]["to_anchor"]` act on `(x, y)`, as OpenCV's do.
    - **No data** is `0` in the integer stacks and `NaN` in the float ones.
    - **DN** (digital number) is the camera's raw unit. **Optical density** is $B - \ln I$: how far a pixel's log brightness lies below the fitted background $B$. It is about 0 on background and positive in vessels.
    - **Quiet pixel**: a pixel whose brightness changes over time no more than the camera noise does, i.e. background.

    Nothing runs when the notebook is opened: press *Run all*, or run the cells from top to bottom. On a whole 1920 × 1200 burst of 140 frames the notebook takes about 2½ minutes on 6 threads.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 0 · Setup

    Imports, the notebook's few settings, and `create_animation`, which plays any `(frames, rows, columns)` stack as a looping preview.

    | Setting | Default | What it does |
    |---|---|---|
    | `SHOW_PREVIEWS` | `True` | `False` skips every animated preview; the check figures are still drawn |
    | `PREVIEW_FORMAT` | `"webp"` | `"webp"` (small), `"gif"` (plays everywhere, ~15× larger) or `"mp4"` (needs ffmpeg) |
    | `PREVIEW_WIDTH` | `480` | pixels across a preview |
    | `PREVIEW_MAX_MB` | `2.5` | the largest file a preview may be; a larger one is rendered again, narrower, until it fits |
    | `PREVIEW_DIR` | `previews` | where the previews are cached, in the working directory |
    | `workers` | half the cores | threads for the per-frame work |

    Everything the notebook writes (`previews/`, `vessel_pictures/`, `preprocessed/`) goes to the working directory it was started in.

    `create_animation` changes what you see, never the data. It stretches the stack's full range to 8 bits, shrinks the frames to `width` pixels across, and caches the file in `previews/` under a hash of the frames, so each stack is rendered only once. A preview is embedded in the notebook's output, and marimo refuses an output much over 10 MB, so a preview whose file is larger than `PREVIEW_MAX_MB` is rendered again at the width that should bring it under that size; noisy stacks, such as §13's side-by-side comparison, need this. Previews are animated WebP, about 15× smaller than the same frames as a GIF, which keeps this notebook light. If your viewer won't play them, set `PREVIEW_FORMAT = "gif"`.
    """)
    return


@app.cell(hide_code=True)
def _():
    import base64
    import functools
    import hashlib
    import json
    import os
    import subprocess
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.fft
    from IPython.display import HTML, Image as IPImage, Video, display
    from PIL import Image
    from scipy.ndimage import distance_transform_edt, percentile_filter
    from skimage.filters import apply_hysteresis_threshold, frangi, laplace
    from skimage.transform import downscale_local_mean

    print(f"Working in {Path.cwd()}")

    SHOW_PREVIEWS = True            # False skips every animated preview; the check figures are still drawn
    PREVIEW_FORMAT = "webp"         # "webp" (small), "gif" (plays everywhere, ~15x larger) or "mp4" (needs ffmpeg)
    PREVIEW_WIDTH = 480             # pixels; previews are rendered this wide
    PREVIEW_MAX_MB = 2.5            # a preview file larger than this is rendered again, narrower: it is embedded in the output
    PREVIEW_DIR = Path("previews")  # where the previews are cached, under a hash of their content
    workers = max(1, (os.cpu_count() or 2) // 2)  # threads for the per-frame work: half the cores

    # plots: recessive axes and grid, and one fixed order of colors for categories (fixation 0, 1, ...)
    SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
    INK, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
    plt.rcParams.update({
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7",
        "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
        "axes.titlesize": 10, "axes.titlecolor": INK, "axes.labelcolor": INK_SECONDARY,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "legend.frameon": False,
    })


    def fixation_color(number):
        """The color of fixation `number` in the plots: the categories in their fixed order, gray beyond them."""
        return SERIES[number] if number < len(SERIES) else INK_MUTED


    def create_animation(frames, name, fps=30.0, width=PREVIEW_WIDTH, embed_html5=True):
        """Save a grayscale (frames, rows, columns) stack as a cached looping preview and display it."""
        frames = np.asarray(frames)
        if frames.ndim != 3 or len(frames) == 0:
            raise ValueError("frames must be a non-empty NumPy array with shape (N, H, W)")
        if fps <= 0:
            raise ValueError("fps must be greater than zero")
        ext = "." + PREVIEW_FORMAT.lower()
        if ext not in {".webp", ".gif", ".mp4"}:
            raise ValueError("PREVIEW_FORMAT must be 'webp', 'gif' or 'mp4'")

        def render(width):
            """Render the preview `width` pixels wide (once; it is cached) and return its path."""
            # at most `width` pixels wide, with even dimensions as H.264 requires
            width = max(2, 2 * round(min(width, frames.shape[2]) / 2))
            height = max(2, 2 * round(width * frames.shape[1] / frames.shape[2] / 2))

            # SHA-1 is only a cache key here; it hashes the stack several times faster than SHA-256
            digest = hashlib.sha1(usedforsecurity=False)
            digest.update(memoryview(np.ascontiguousarray(frames)))
            digest.update(f"{frames.shape}:{frames.dtype}:{fps}:{ext}:{width}x{height}".encode("utf-8"))
            PREVIEW_DIR.mkdir(exist_ok=True)
            path = PREVIEW_DIR / f"{name}_{digest.hexdigest()[:16]}{ext}"

            if path.exists():
                print(f"Found cached preview: {path}")
                return path
            print(f"Rendering {len(frames)} frames at {fps:.2f} FPS...")
            # frames are scaled to 8 bits over the stack's full range and handed straight to the encoder
            low, high = float(frames.min()), float(frames.max())
            scale = 255.0 / max(high - low, 1e-8)
            preview = np.empty((len(frames), height, width), dtype=np.uint8)
            for index, frame in enumerate(frames):
                small = cv2.resize(frame.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA)
                preview[index] = np.clip((small - low) * scale + 0.5, 0, 255)
            if ext == ".mp4":
                subprocess.run(
                    ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "gray",
                     "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
                     "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(path)],
                    input=preview.tobytes(), check=True,
                )
            else:
                images = [Image.fromarray(frame) for frame in preview]
                options = {"quality": 80, "method": 4} if ext == ".webp" else {}
                images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0, **options)
            print(f"Saved preview to {path} ({path.stat().st_size / 1e6:.1f} MB)")
            return path

        # the file is embedded in the output (as base64, a third larger), so it is kept under PREVIEW_MAX_MB: the
        # size goes about with the number of pixels, so each retry narrows it by the square root of the excess
        shown_width = max(2, 2 * round(min(width, frames.shape[2]) / 2))  # a narrowed preview is still shown this wide
        path = render(width)
        for _attempt in range(4):
            size = path.stat().st_size
            if size <= PREVIEW_MAX_MB * 1e6 or width <= 64:
                break
            width = max(64, int(width * (PREVIEW_MAX_MB * 1e6 / size) ** 0.5))
            print(f"{size / 1e6:.1f} MB is more than PREVIEW_MAX_MB = {PREVIEW_MAX_MB} MB; rendering it again {width} px wide")
            path = render(width)

        if ext == ".mp4":
            display(Video(str(path), embed=embed_html5, width=640))
        elif ext == ".webp":
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            display(HTML(f'<img src="data:image/webp;base64,{encoded}" width="{shown_width}">'))
        else:
            display(IPImage(filename=str(path)))

    return (
        INK_MUTED,
        INK_SECONDARY,
        Image,
        PREVIEW_WIDTH,
        Path,
        SERIES,
        SHOW_PREVIEWS,
        ThreadPoolExecutor,
        apply_hysteresis_threshold,
        create_animation,
        cv2,
        distance_transform_edt,
        downscale_local_mean,
        fixation_color,
        frangi,
        functools,
        json,
        laplace,
        np,
        os,
        percentile_filter,
        plt,
        scipy,
        workers,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Load the burst

    The burst is the first `burst_*` folder found in the working directory or one of its parent folders. Its TIFF frames are read in name order into one array and left untouched: these are the camera's own 12-bit values. The frame rate comes from the recorder's `manifest.json`. It gives the frame times and plays the previews at the speed of the recording. To load a particular burst instead, set the environment variable `LIMBUS_BURST` to its folder before starting the notebook.

    | Output | Shape · type | Content |
    |---|---|---|
    | `raw_burst` | `(frames, rows, columns)` · `uint16` | every recorded frame, as recorded |
    | `fps` | `float` | frames per second of the recording |
    | `target_folder` | `Path` | the burst's folder |
    """)
    return


@app.cell(hide_code=True)
def _(Image, Path, json, np, os):
    project_dir = Path.cwd()
    burst_choice = os.environ.get("LIMBUS_BURST")  # a burst folder to load, instead of the first one found
    search_roots = () if burst_choice else (project_dir, *project_dir.parents)
    burst_folders = [Path(burst_choice)] if burst_choice else []
    for search_root in search_roots:
        burst_folders = sorted(path for path in search_root.glob("burst_*") if path.is_dir())
        if burst_folders:
            project_dir = search_root
            break
    if not burst_folders:
        raise FileNotFoundError(
            f"No burst_* directory found in {Path.cwd()} or its parent directories. "
            "Set LIMBUS_BURST to a burst folder, or use a local Python kernel with the burst folder in the workspace."
        )

    target_folder = burst_folders[0]
    tiff_files = sorted(target_folder.glob("*.tif*"))
    if not tiff_files:
        raise FileNotFoundError(f"No TIFF files found in {target_folder}")

    print(f"Loading {len(tiff_files)} TIFF frames from {target_folder}...")
    with Image.open(tiff_files[0]) as _image:
        first_frame = np.asarray(_image)
        raw_burst = np.empty((len(tiff_files), *first_frame.shape), dtype=first_frame.dtype)
        raw_burst[0] = first_frame

    for _index, file_path in enumerate(tiff_files[1:], start=1):
        with Image.open(file_path) as _image:
            raw_burst[_index] = np.asarray(_image)

    print(f"Loaded burst with shape {raw_burst.shape} and dtype {raw_burst.dtype}.")

    manifest_path = target_folder / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as manifest_file:
            manifest_data = json.load(manifest_file)
        capture_data = manifest_data.get("capture", {})
        requested_data = manifest_data.get("acquisition_requested", {})
        fps = capture_data.get("effective_fps", requested_data.get("requested_fps", 15.0))
        print(f"Using {fps:.3f} FPS from the capture manifest.")
    else:
        fps = 15.0
        print(f"manifest.json not found in {target_folder}; using {fps:.1f} FPS.")
    return fps, raw_burst, target_folder


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The raw burst. Watch for the jitter, and for the one large jump: a saccade.
    """)
    return


@app.cell
def _(SHOW_PREVIEWS, create_animation, fps, raw_burst):
    if SHOW_PREVIEWS:
        create_animation(raw_burst, "raw_burst", fps=fps)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Drop the blurred frames

    Some frames are blurred, for example when the eye moves during the exposure. They are dropped before anything is aligned or averaged. Each frame gets two scores:

    - **Sharpness**: the variance of the frame's Laplacian. It is high when edges are crisp.
    - **Vesselness**: the mean Frangi response to dark ridges. It is high when vessels stand out.

    Both are computed on a reduced copy of the frame (the mean of every 4 × 4 block), divided by its mean brightness so that exposure changes do not move them. The reduction matters: at full resolution both scores measure sensor noise and do not respond to blur. A frame is kept when **both** scores reach 80% of the burst's median.

    | Output | Shape · type | Content |
    |---|---|---|
    | `gated_frames` | `(kept, rows, columns)` · `uint16` | the kept frames, in their recorded order |
    | `gated_indices` | `(kept,)` · `int` | each kept frame's number in `raw_burst` |
    | `frame_times` | `(kept,)` · `float` | when each kept frame was recorded, in seconds from the first frame; the gaps are longer where frames were dropped |
    | `sharpness`, `vesselness` | `(kept,)` · `float` | the kept frames' scores |
    | `all_sharpness`, `all_vesselness` | `(frames,)` · `float` | every recorded frame's scores |
    | `thresholds` | `(2,)` | the two cut-offs |

    From here on, a frame number is a position in `gated_frames`.
    """)
    return


@app.cell
def _(
    INK_MUTED,
    INK_SECONDARY,
    SERIES,
    ThreadPoolExecutor,
    downscale_local_mean,
    fps,
    frangi,
    laplace,
    np,
    plt,
    raw_burst,
    workers,
):
    GATE_DOWNSAMPLE = 4       # frames are scored on the means of blocks this many pixels on a side
    GATE_SIGMAS = (1, 2, 3)   # reduced pixels; the scales of the Frangi filter
    SHARPNESS_CUTOFF = 0.8    # a frame is kept when both of its scores reach this fraction of the burst's median
    VESSELNESS_CUTOFF = 0.8


    def score_frame(frame, downsample=GATE_DOWNSAMPLE, sigmas=GATE_SIGMAS):
        """Return the Laplacian-variance sharpness and mean Frangi vesselness of one frame."""
        # on downsample x downsample block means: at full resolution both scores measure sensor noise and do not
        # respond to blur. sigmas are in downsampled pixels.
        small = downscale_local_mean(frame, (downsample, downsample))
        small = small / small.mean()  # divide out brightness so exposure changes don't move the scores
        return laplace(small).var(), frangi(small, sigmas=sigmas, black_ridges=True).mean()


    print(f"Scoring {len(raw_burst)} frames for sharpness and vesselness on {workers} threads...")
    with ThreadPoolExecutor(max_workers=workers) as _pool:
        all_sharpness, all_vesselness = np.array(list(_pool.map(score_frame, raw_burst))).T

    thresholds = (SHARPNESS_CUTOFF * np.median(all_sharpness), VESSELNESS_CUTOFF * np.median(all_vesselness))
    _keep = (all_sharpness >= thresholds[0]) & (all_vesselness >= thresholds[1])
    gated_indices = np.flatnonzero(_keep)
    gated_frames = raw_burst[gated_indices]
    sharpness, vesselness = all_sharpness[gated_indices], all_vesselness[gated_indices]
    frame_times = gated_indices / fps  # seconds from the first frame of the burst; dropped frames leave gaps

    print(f"Recorded frames: {len(raw_burst)} | kept: {len(gated_frames)} | dropped: {np.flatnonzero(~_keep).tolist()}")
    print(f"Sharpness threshold: {thresholds[0]:.3e} | Vesselness threshold: {thresholds[1]:.3e}")

    # every recorded frame's two scores against their cut-offs; the dropped frames are marked
    recorded = np.arange(len(raw_burst))
    _figure, _axes = plt.subplots(1, 2, figsize=(14, 3.2))
    for _axis, scores, threshold, _name in zip(_axes, (all_sharpness, all_vesselness), thresholds, ("sharpness", "vesselness")):
        _axis.plot(recorded, scores, color=INK_SECONDARY, linewidth=1.5, label="score")
        _axis.axhline(threshold, color=INK_MUTED, linestyle="--", linewidth=1, label="cut-off")
        _axis.plot(recorded[~_keep], scores[~_keep], "o", color=SERIES[1], markersize=7, label="dropped")
        _axis.set_title(f"{_name} of every recorded frame")
        _axis.set_xlabel("recorded frame")
    _axes[0].legend()
    _figure.tight_layout()
    plt.show()
    return frame_times, gated_frames, gated_indices, sharpness, vesselness


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Measure the motion between frames

    Motion is measured by **phase correlation**. Shifting an image by $\mathbf{s}$ leaves the magnitude of its Fourier transform alone and multiplies it by a phase ramp $e^{-i\,\mathbf{k}\cdot\mathbf{s}}$ (the Fourier shift theorem). The normalized cross-power spectrum of two frames is therefore that ramp alone,

    $$
    R(\mathbf{k}) \;=\; \frac{F_\text{moving}(\mathbf{k})\,F^*_\text{reference}(\mathbf{k})}{\bigl|F_\text{moving}(\mathbf{k})\,F^*_\text{reference}(\mathbf{k})\bigr|} \;=\; e^{-i\,\mathbf{k}\cdot\mathbf{s}},
    $$

    and its inverse transform is a single peak at $\mathbf{s}$.

    1. `frame_spectrum` removes the frame's mean, tapers its edges with a Hann window and takes the FFT.
    2. `phase_shift` forms $R$ and low-passes it with a Gaussian of `PEAK_SIGMA` pixels. This drops the noise-dominated frequencies and gives the peak a Gaussian shape. It then transforms back.
    3. A three-point fit along each axis places the peak to a fraction of a pixel.

    `phase_shift` returns the shift `(dy, dx)` and a **coherence** between 0 and 1: the height of the peak relative to that of a perfect match. Here the shift is measured between every frame and the one before it, and adding the steps up traces the eye's path.

    | Output | Shape | Content |
    |---|---|---|
    | `step_shifts` | `(kept, 2)` | `(dy, dx)` of each frame from the one before it (row 0 is zero) |
    | `step_sizes` | `(kept,)` | their lengths, in pixels |
    | `frame_positions` | `(kept, 2)` | the running sum of the steps: where the eye was |
    | `frame_spectrum()`, `phase_shift()` | | the two functions; §5 and §6 use them again |
    """)
    return


@app.cell
def _(ThreadPoolExecutor, functools, gated_frames, np, scipy, workers):
    PEAK_SIGMA = 1.5  # pixels; width given to the correlation peak by low-passing the phase ramp

    num_frames, frame_height, frame_width = gated_frames.shape
    frame_shape = (frame_height, frame_width)


    @functools.lru_cache(maxsize=None)
    def correlation_filters(shape):
        """Return the Hann window, the low-pass for the phase ramp and its peak height for images of one shape."""
        window = np.outer(np.hanning(shape[0]), np.hanning(shape[1])).astype(np.float32)
        frequency_y = scipy.fft.fftfreq(shape[0])[:, None]
        frequency_x = scipy.fft.rfftfreq(shape[1])[None, :]
        lowpass = np.exp(-2 * np.pi**2 * PEAK_SIGMA**2 * (frequency_y**2 + frequency_x**2)).astype(np.float32)
        return window, lowpass, scipy.fft.irfft2(lowpass, s=shape)[0, 0]


    # compute the fast fourier transform of all frames
    # (a spectrum takes ~0.05 s, so each is recomputed where it is used instead of holding 1.2 GB of them)
    def frame_spectrum(frame):
        """Return the half-plane FFT of a mean-subtracted, Hann-windowed frame."""
        frame = frame.astype(np.float32)
        return scipy.fft.rfft2((frame - frame.mean()) * correlation_filters(frame.shape)[0])


    # use fourier shift theorem and normalized cross power spectrum to
    # compute the phase ramp, and use the inverse fourier transform to compute the subpixel shift between frames
    def phase_shift(reference_spectrum, moving_spectrum, shape=None):
        """Return the (dy, dx) shift of the moving image from the reference, and the peak coherence (0 to 1)."""
        shape = shape or frame_shape  # the images are full frames unless a tile shape is given
        _, lowpass, peak_scale = correlation_filters(shape)
        cross_power = moving_spectrum * np.conj(reference_spectrum)
        cross_power /= np.abs(cross_power) + 1e-12  # unit magnitude leaves only the phase ramp of the shift
        # Low-passing the ramp drops the noise-dominated frequencies and turns the correlation
        # peak into a Gaussian, which a 3-point fit along each axis then locates below a pixel.
        correlation = scipy.fft.irfft2(cross_power * lowpass, s=shape)
        peak = np.unravel_index(np.argmax(correlation), shape)

        shift = np.zeros(2)
        for axis, size in enumerate(shape):
            line = correlation[:, peak[1]] if axis == 0 else correlation[peak[0], :]
            samples = line.take([peak[axis] - 1, peak[axis], peak[axis] + 1], mode="wrap")
            below, centre, above = np.log(np.maximum(samples, 1e-12))
            curvature = below - 2 * centre + above
            offset = 0.5 * (below - above) / curvature if curvature < 0 else 0.0
            position = peak[axis] + float(np.clip(offset, -0.5, 0.5))
            shift[axis] = position - size if position > size / 2 else position
        return shift, float(correlation[peak] / peak_scale)


    def step_shift(index):
        return phase_shift(frame_spectrum(gated_frames[index - 1]), frame_spectrum(gated_frames[index]))


    with ThreadPoolExecutor(max_workers=workers) as _pool:
        steps = list(_pool.map(step_shift, range(1, num_frames)))
    step_shifts = np.array([(0.0, 0.0), *(shift for shift, _ in steps)])  # row i: shift of frame i from frame i - 1
    step_sizes = np.hypot(step_shifts[:, 0], step_shifts[:, 1])
    frame_positions = np.cumsum(step_shifts, axis=0)
    return (
        frame_height,
        frame_positions,
        frame_shape,
        frame_spectrum,
        frame_width,
        num_frames,
        phase_shift,
        step_shifts,
        step_sizes,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Split the burst into fixations

    Ordinary steps are a few pixels long. A step longer than `GREAT_SHIFT_FACTOR` (10) times the median step, and longer than `GREAT_SHIFT_FLOOR` (20) pixels, is a saccade. The frame sequence is cut at every saccade, and each piece is one fixation.

    | Output | Shape · type | Content |
    |---|---|---|
    | `fixation_members` | list of `(frames,)` · `int` | the frame numbers of each fixation; they are consecutive |
    | `raw_fixations` | list of `(frames, rows, columns)` · `uint16` | each fixation's frames as recorded: not aligned, not cleaned |

    `raw_fixations[n]` is a view into `gated_frames`, not a copy. Copy it before changing it in place.
    """)
    return


@app.cell(hide_code=True)
def _(
    INK_MUTED,
    fixation_color,
    frame_positions,
    gated_frames,
    np,
    num_frames,
    plt,
    step_sizes,
):
    GREAT_SHIFT_FACTOR = 10.0  # a frame-to-frame shift this many times the burst's median shift is "great"...
    GREAT_SHIFT_FLOOR = 20.0   # ...as long as it is also at least this many pixels

    # when subpixel shifts are great, break the frame sequence into multiple fixations
    great_shift = max(GREAT_SHIFT_FACTOR * np.median(step_sizes[1:]), GREAT_SHIFT_FLOOR)
    fixation_members = np.split(np.arange(num_frames), np.flatnonzero(step_sizes > great_shift))
    print(f"Frame-to-frame shift: median {np.median(step_sizes[1:]):.2f} px, largest {step_sizes.max():.2f} px | "
          f"great-shift limit {great_shift:.1f} px -> {len(fixation_members)} fixation(s)")

    raw_fixations = [gated_frames[_members[0]:_members[-1] + 1] for _members in fixation_members]  # views, not copies
    print("raw_fixations: " + ", ".join(f"{frames.shape}" for frames in raw_fixations))

    # the eye's steps against the saccade limit, and its path, both colored by fixation
    _figure, _axes = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={"width_ratios": (1.6, 1)})
    for _number, _members in enumerate(fixation_members):
        stepped = _members[_members > 0]  # frame 0 has no step
        _axes[0].plot(stepped, step_sizes[stepped], "o-", color=fixation_color(_number), markersize=4, linewidth=1,
                     label=f"fixation {_number}")
        _axes[1].plot(frame_positions[_members, 1], frame_positions[_members, 0], "o-", color=fixation_color(_number),
                     markersize=4, linewidth=1, label=f"fixation {_number}")
    _axes[0].axhline(great_shift, color=INK_MUTED, linestyle="--", linewidth=1, label="saccade limit")
    _axes[0].set_yscale("log")
    _axes[0].set_xlabel("frame")
    _axes[0].set_ylabel("step from the frame before (px)")
    _axes[0].set_title("frame-to-frame steps")
    _axes[0].legend()
    _axes[1].invert_yaxis()  # rows grow downwards, as in the frames
    _axes[1].set_aspect("equal", adjustable="datalim")
    _axes[1].set_xlabel("x (px)")
    _axes[1].set_ylabel("y (px)")
    _axes[1].set_title("path of the eye")
    _figure.tight_layout()
    plt.show()
    return fixation_members, raw_fixations


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Align the frames of each fixation

    Within a fixation the eye only jitters, so a shift is enough to line the frames up. For each fixation:

    1. **Choose the reference frame.** It is the frame with the best quality, where quality is sharpness plus vesselness, each standardized over the gated burst. If no frame of the fixation beats the burst's average, the frame closest to the fixation's median position is used instead.
    2. **Measure each frame's shift from the reference** by phase correlation directly against the reference frame. Adding up the steps of §3 would also add up their errors.
    3. **Average the aligned frames.** Each frame is shifted back onto the reference with cubic interpolation. A frame counts only where it really has data, so the border of the average is made from fewer frames.

    **Output:** `fixations`, one dictionary per fixation. §6 adds three more keys.

    | Key | Content |
    |---|---|
    | `frames` | its frame numbers (positions in `gated_frames`) |
    | `times` | when each of its frames was recorded, in seconds |
    | `reference` | the frame number of its reference frame |
    | `shifts` | `(dy, dx)` of each frame from the reference |
    | `coherence` | how well each frame matched the reference, from 0 to 1 |
    | `average` | the mean of the aligned frames (`float32`) |
    | `coverage` | how many frames were averaged at each pixel |
    """)
    return


@app.cell
def _(
    ThreadPoolExecutor,
    cv2,
    fixation_members,
    frame_height,
    frame_positions,
    frame_shape,
    frame_spectrum,
    frame_times,
    frame_width,
    gated_frames,
    np,
    phase_shift,
    sharpness,
    vesselness,
    workers,
):
    # choose the reference frame according to the highest sharpness and vesselness score, or the median frame if all scores are low
    # (both scores are standardized over the gated burst and summed; "low" means no frame of the fixation beats
    # the burst average, and the median frame is then the one closest to the fixation's median position)
    def standardize(scores):
        return (scores - scores.mean()) / (scores.std() + 1e-12)


    def covered_area(dy, dx, margin=0):
        """Return the (rows, columns) of the reference frame that a frame shifted by (dy, dx) really covers."""
        rows = slice(max(0, int(np.ceil(margin - dy))),
                     min(frame_height, int(np.floor(frame_height - 1 - margin - dy)) + 1))
        columns = slice(max(0, int(np.ceil(margin - dx))),
                        min(frame_width, int(np.floor(frame_width - 1 - margin - dx)) + 1))
        return rows, columns


    frame_quality = standardize(sharpness) + standardize(vesselness)

    fixations = []
    for _members in fixation_members:
        _reference = _members[np.argmax(frame_quality[_members])]
        if frame_quality[_reference] <= 0:
            distances = np.linalg.norm(frame_positions[_members] - np.median(frame_positions[_members], axis=0), axis=1)
            _reference = _members[np.argmin(distances)]

        # each fixation is a set of frames that are aligned to a single reference frame, and the fixations are then aligned to each other
        reference_spectrum = frame_spectrum(gated_frames[_reference])
        with ThreadPoolExecutor(max_workers=workers) as _pool:
            matches = list(_pool.map(
                lambda index: phase_shift(reference_spectrum, frame_spectrum(gated_frames[index])), _members
            ))
        shifts = np.array([shift for shift, _ in matches])

        # compute the average of each fixation, once fixation frames are aligned to their reference
        _total = np.zeros(frame_shape, dtype=np.float32)
        coverage = np.zeros(frame_shape, dtype=np.uint16)
        for _index, (_dy, _dx) in zip(_members, shifts):
            _aligned = cv2.warpAffine(
                gated_frames[_index].astype(np.float32), np.float32([[1, 0, -_dx], [0, 1, -_dy]]),
                (frame_width, frame_height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
            )
            # Only the pixels that the shifted frame really covers count towards the average.
            _area = covered_area(_dy, _dx)
            _total[_area] += _aligned[_area]
            coverage[_area] += 1

        fixations.append({
            "frames": _members,  # indices into gated_frames
            "times": frame_times[_members],  # seconds; when each frame was recorded (gaps where frames were dropped)
            "reference": _reference,
            "shifts": shifts,  # (dy, dx) of each frame from the reference
            "coherence": np.array([coherence for _, coherence in matches]),
            "average": _total / np.maximum(coverage, 1),
            "coverage": coverage,  # number of frames averaged at each pixel
        })
        print(f"Fixation {len(fixations) - 1}: frames {_members[0]}-{_members[-1]} ({len(_members)} frames), "
              f"reference {_reference}, largest shift from reference {np.linalg.norm(shifts, axis=1).max():.1f} px")
    return covered_area, fixations


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · Align the fixations to each other

    After a saccade the view changes by more than a shift, so a translation alone does not line two fixations up. The fixation with the most frames is the **anchor**, and every other fixation's average is mapped onto the anchor's average:

    1. Phase correlation gives a starting translation.
    2. **ECC** (enhanced correlation coefficient, `cv2.findTransformECC`) refines it into an affine map. ECC sees only the fine detail of the averages: the structure smaller than about `DETAIL_SIGMA` pixels. Between fixations the thick out-of-focus vessels shift against the sharp ones and would otherwise pull the fit away from them.
    3. The fit is judged by its **local misfit**: the averages are compared tile by tile, and the misfit is the median shift still left between them. If the affine map leaves more than `MISFIT_TOLERANCE` pixels, a homography (8 parameters) is tried as well. A fit is kept only if it lowers the misfit.

    | Output | Content |
    |---|---|
    | `anchor` | the anchor fixation's number |
    | `aligned_fixation_averages` | `(fixations, rows, columns)`: every fixation's average, resampled into the anchor's frame |
    | `fixations[n]["to_anchor"]` | the map from a point `(x, y)` in the anchor's frame to the same point in fixation `n`'s reference frame: 2 × 3 for a translation or an affine map, 3 × 3 for a homography |
    | `fixations[n]["model"]` | `"translation"`, `"affine"` or `"homography"`, whichever was kept |
    | `fixations[n]["misfit"]` | its median local misfit, in pixels |
    """)
    return


@app.cell
def _(
    cv2,
    fixations,
    frame_height,
    frame_shape,
    frame_spectrum,
    frame_width,
    np,
    phase_shift,
):
    DETAIL_SIGMA = 2.0      # pixels; ECC fits the structure finer than this, i.e. the in-focus vessels
    MISFIT_TILE = 256       # pixels; side of the tiles on which the leftover local shift is measured
    MISFIT_TOLERANCE = 0.5  # pixels; a median leftover shift above this makes the affine insufficient

    # then try to identify an affine transformation that aligns average fixation frames to each other
    # (every average is mapped onto the fixation with the most frames: phase correlation gives the starting
    # translation and ECC refines it. ECC is given only the fine detail, because between fixations the thick
    # out-of-focus vessels shift against the sharp ones and would otherwise pull the fit away from them.)
    ECC_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
    anchor = int(np.argmax([len(_fixation["frames"]) for _fixation in fixations]))
    anchor_average = fixations[anchor]["average"]
    anchor_spectrum = frame_spectrum(anchor_average)


    def fine_detail(image):
        """Keep the structure finer than DETAIL_SIGMA, which is what the frame alignment locks onto."""
        return image - cv2.GaussianBlur(image, (0, 0), DETAIL_SIGMA)


    def warp_to_anchor(average, transform):
        """Resample an average into the anchor's frame; return it with its median local misfit in pixels."""
        warp = cv2.warpAffine if len(transform) == 2 else cv2.warpPerspective
        size = (frame_width, frame_height)
        warped = warp(average, transform, size, flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)
        inside = warp(
            np.ones(frame_shape, dtype=np.uint8), transform, size, flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP
        ) > 0
        # The misfit is the shift still left between the two images, measured on half-overlapping tiles.
        residuals = []
        for top in range(0, frame_height - MISFIT_TILE + 1, MISFIT_TILE // 2):
            for left in range(0, frame_width - MISFIT_TILE + 1, MISFIT_TILE // 2):
                tile = (slice(top, top + MISFIT_TILE), slice(left, left + MISFIT_TILE))
                if inside[tile].all():
                    shift, _ = phase_shift(
                        frame_spectrum(anchor_average[tile]), frame_spectrum(warped[tile]), (MISFIT_TILE, MISFIT_TILE)
                    )
                    residuals.append(np.hypot(shift[0], shift[1]))
        return warped, float(np.median(residuals)) if residuals else np.inf


    anchor_detail = fine_detail(anchor_average)
    aligned_fixation_averages = np.empty((len(fixations), *frame_shape), dtype=np.float32)
    for _number, _fixation in enumerate(fixations):
        _average = _fixation["average"]
        (_dy, _dx), _ = phase_shift(anchor_spectrum, frame_spectrum(_average))
        transform, _model = np.float32([[1, 0, _dx], [0, 1, _dy]]), "translation"  # maps anchor (x, y) to this average's (x, y)
        _aligned, misfit = warp_to_anchor(_average, transform)
        if _number != anchor:
            average_detail = fine_detail(_average)
            _tried = [f"translation {misfit:.2f} px"]
            # if the affine map is insufficient, we can use an enhanced cross correlation to find a warping transform
            # (ECC is already what fits the affine, so the richer warp it is asked for is its 8-parameter homography;
            # each fit is kept only if it lowers the misfit)
            for candidate_model, motion in (("affine", cv2.MOTION_AFFINE), ("homography", cv2.MOTION_HOMOGRAPHY)):
                if candidate_model == "homography" and misfit <= MISFIT_TOLERANCE:
                    break
                start = transform if motion == cv2.MOTION_AFFINE else np.vstack([transform, [0, 0, 1]]).astype(np.float32)
                try:
                    _, _candidate = cv2.findTransformECC(
                        anchor_detail, average_detail, start.copy(), motion, ECC_CRITERIA, None, 5
                    )
                except cv2.error:
                    _tried.append(f"{candidate_model} did not converge")
                    continue
                candidate_aligned, candidate_misfit = warp_to_anchor(_average, _candidate)
                _tried.append(f"{candidate_model} {candidate_misfit:.2f} px")
                if candidate_misfit < misfit:
                    transform, _model, _aligned, misfit = _candidate, candidate_model, candidate_aligned, candidate_misfit
            _verdict = "within" if misfit <= MISFIT_TOLERANCE else "still above"
            print(f"Fixation {_number} -> fixation {anchor}: local misfit {' -> '.join(_tried)}; "
                  f"kept the {_model}, {_verdict} the {MISFIT_TOLERANCE} px tolerance")
        aligned_fixation_averages[_number] = _aligned
        _fixation.update({"to_anchor": transform, "model": _model, "misfit": misfit})
    return aligned_fixation_averages, anchor


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Blinking between the aligned averages checks the alignment by eye: whatever jumps between the images is not aligned.
    """)
    return


@app.cell
def _(SHOW_PREVIEWS, aligned_fixation_averages, create_animation):
    if SHOW_PREVIEWS:
        create_animation(aligned_fixation_averages, "fixation_averages", fps=2, width=960)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 · Remove dust and glare

    Two things in the frames are not tissue. Both are dealt with before the frames are stabilized, so that every later step works on clean frames.

    - **Dust on the sensor** sits on the same sensor pixels in every frame, while the tissue moves under it. It dims the light by a fixed fraction, so it can be divided out.
    - **Glare** is light reflected off the wet surface. Where it clips, the pixel holds no tissue signal at all, so glare cannot be corrected. It is left out instead: those pixels count as *no data* in that frame.

    **Glare first.** In each frame a pixel is glare when

    - it is clipped, at `SATURATION_LEVEL` of the burst's largest value or above, or
    - it is more than `GLARE_EXCESS` brighter than the bright tissue around it. That level (`bright_level`) is the 95th percentile of the brightness in a window around the pixel. Vessels are darker than it and open tissue just reaches it, so only glare stands clear of it.

    Glare is then grown by `GLARE_GROW` pixels to take in its halo. A pixel that is too bright in nearly every frame is not glare but a stuck sensor pixel, and is kept apart.
    """)
    return


@app.cell
def _(
    ThreadPoolExecutor,
    cv2,
    frame_height,
    frame_shape,
    frame_width,
    gated_frames,
    np,
    num_frames,
    percentile_filter,
    workers,
):
    SATURATION_LEVEL = 0.95  # fraction of the burst's largest value; from here up a pixel counts as clipped
    GLARE_EXCESS = 0.08      # a pixel this much brighter than the bright tissue around it is glare as well
    GLARE_GROW = 12          # pixels by which glare is grown, to take in its halo
    STUCK_FRACTION = 0.9     # a pixel that is too bright in this fraction of the frames is stuck, not glare
    BRIGHT_SCALE = 8         # the bright level of the tissue is read from frames reduced by this factor...
    BRIGHT_WINDOW = 11       # ...over a window of this many reduced pixels each way

    clipped_level = SATURATION_LEVEL * gated_frames.max()


    def disc_kernel(radius):
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


    def bright_level(frame):
        """Return a smooth picture of how bright the tissue between the vessels is around each pixel."""
        reduced = cv2.resize(
            frame, (frame_width // BRIGHT_SCALE, frame_height // BRIGHT_SCALE), interpolation=cv2.INTER_AREA
        )
        # the 95th percentile sits on the open tissue: vessels are darker, and glare fills too little of the window
        level = cv2.GaussianBlur(percentile_filter(reduced, 95, size=BRIGHT_WINDOW, mode="nearest"), (0, 0), 1.5)
        return cv2.resize(level, (frame_width, frame_height), interpolation=cv2.INTER_CUBIC)


    def too_bright(index):
        """Return the pixels of a frame that are clipped or stand out above the tissue, as flat indices."""
        frame = gated_frames[index].astype(np.float32)
        stands_out = cv2.GaussianBlur(frame, (0, 0), 1) > (1 + GLARE_EXCESS) * bright_level(frame)
        return np.flatnonzero((frame >= clipped_level) | stands_out)


    with ThreadPoolExecutor(max_workers=workers) as _pool:
        bright_pixels = list(_pool.map(too_bright, range(num_frames)))

    # a pixel that is too bright in nearly every frame is the sensor's doing (a hot pixel), not glare
    times_bright = np.bincount(np.concatenate(bright_pixels), minlength=frame_height * frame_width)
    stuck_pixels = (times_bright >= STUCK_FRACTION * num_frames).reshape(frame_shape)

    glare_pixels = []  # per frame: the pixels left out as glare, as flat indices into the frame
    for bright in bright_pixels:
        glare = np.zeros(frame_shape, dtype=np.uint8)
        glare.flat[bright] = 1
        glare[stuck_pixels] = 0
        glare_pixels.append(np.flatnonzero(cv2.dilate(glare, disc_kernel(GLARE_GROW))))

    bright_counts = [len(bright) - stuck_pixels.sum() for bright in bright_pixels]
    glare_counts = [len(glare) for glare in glare_pixels]
    print(f"Glare: {np.median(bright_counts):.0f} pixels per frame are clipped or stand out above the tissue "
          f"({min(bright_counts)} to {max(bright_counts)}); grown by {GLARE_GROW} px, "
          f"{np.median(glare_counts):.0f} pixels per frame are left out "
          f"({np.median(glare_counts) / (frame_height * frame_width) * 100:.2f}% of a frame)")
    print(f"Stuck pixels (too bright in at least {STUCK_FRACTION:.0%} of the frames): {stuck_pixels.sum()}"
          + "".join(f" at (x {x}, y {y})" for y, x in np.argwhere(stuck_pixels)[:5]))
    return disc_kernel, glare_pixels, stuck_pixels


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Then the sensor.** If a frame showed nothing but tissue, it would equal the tissue picture, moved to where that frame saw it and scaled by the frame's brightness. So each frame is divided by exactly that, and the ratios are averaged pixel by pixel *on the sensor*. The tissue cancels, because it lies somewhere else on the sensor in every frame. What is left is each sensor pixel's own response: 1 for a sound pixel, less under dust.

    The tissue picture is the average of a fixation's frames, each shifted onto the reference frame as in §5, with the glare left out. Dust leaves a dark trail on that picture, so the calculation runs twice; the second round also leaves out the sensor pixels that the first found flawed.

    - Where the response is off by more than `DUST_DEPTH`, the spot is grown by `DUST_GROW` pixels and the frames are divided by the response there. Everywhere else the frames stay exactly as recorded.
    - A pixel that passes less than `DARKEST_GAIN` of the light cannot be restored. It is left out in every frame, like the stuck pixels.

    | Output | Content |
    |---|---|
    | `sensor_map` | `(rows, columns)`: the response of every sensor pixel, exactly 1 where the sensor is sound |
    | `unusable_pixels` | `(rows, columns)`, boolean: the sensor pixels that hold no data in any frame |
    | `glare_pixels` | per frame: the pixels left out as glare, as flat indices into the frame |
    | `stuck_pixels` | `(rows, columns)`, boolean: the stuck sensor pixels |
    | `clean_frame(i)` | gated frame `i` with the sensor map divided out, and 0 where it holds no data. Cleaned frames are computed when asked for instead of being stored as one more stack. |
    | `tissue`, `frame_factors` | each fixation's tissue picture (`NaN` where no frame saw the tissue clear) and each frame's brightness factor |
    | `viewing_tissue` | the tissue pictures with those `NaN` spots smoothed over, for the previews only |

    From here on, **0 in a frame means no data**.
    """)
    return


@app.cell
def _(
    cv2,
    disc_kernel,
    fixations,
    frame_height,
    frame_shape,
    frame_width,
    gated_frames,
    glare_pixels,
    np,
    num_frames,
    stuck_pixels,
):
    DUST_DEPTH = 0.03    # a sensor pixel whose response is off by more than this is flawed
    DUST_GROW = 4        # pixels by which the flawed spots are grown, to take in their soft edges
    DARKEST_GAIN = 0.25  # a pixel that passes less light than this cannot be restored and is left out instead
    MIN_VIEWS = 0.25     # a pixel's response is trusted when this fraction of the frames could measure it


    def glare_mask(index):
        """Return where a frame is left out as glare, as a boolean image."""
        mask = np.zeros(frame_shape, dtype=bool)
        mask.flat[glare_pixels[index]] = True
        return mask


    def moved_to_reference(image, dy, dx):
        """Move a frame's image to where it lies in its fixation's reference frame (0 where the frame has nothing)."""
        return cv2.warpAffine(image, np.float32([[1, 0, -dx], [0, 1, -dy]]), (frame_width, frame_height))


    def seen_from_frame(picture, dy, dx):
        """Move a picture in a fixation's reference frame to where a frame shifted by (dy, dx) saw it (NaN beyond it)."""
        return cv2.warpAffine(
            picture, np.float32([[1, 0, dx], [0, 1, dy]]), (frame_width, frame_height), borderValue=np.nan
        )


    def tissue_pictures(left_out):
        """Average each fixation's frames on the tissue, without glare and without the sensor pixels in `left_out`."""
        pictures = []
        for fixation in fixations:
            total = np.zeros(frame_shape, dtype=np.float32)
            weight = np.zeros(frame_shape, dtype=np.float32)
            for index, (dy, dx) in zip(fixation["frames"], fixation["shifts"]):
                good = ~(left_out | glare_mask(index))
                total += moved_to_reference(np.where(good, gated_frames[index], 0).astype(np.float32), dy, dx)
                weight += moved_to_reference(good.astype(np.float32), dy, dx)
            pictures.append(np.where(weight >= 1, total / np.maximum(weight, 1), np.nan).astype(np.float32))
        return pictures


    def sensor_response(pictures):
        """Return each sensor pixel's mean of frame / expected tissue, and every frame's brightness factor."""
        total = np.zeros(frame_shape, dtype=np.float32)
        views = np.zeros(frame_shape, dtype=np.float32)
        factors = np.ones(num_frames)
        sample = (slice(None, None, 4), slice(None, None, 4))  # every 4th pixel is plenty to fit one number per frame
        for fixation, picture in zip(fixations, pictures):
            for index, (dy, dx) in zip(fixation["frames"], fixation["shifts"]):
                expected = seen_from_frame(picture, dy, dx)  # the tissue as this frame saw it; NaN where unknown
                frame = gated_frames[index].astype(np.float32)
                usable = np.isfinite(expected) & ~glare_mask(index)
                expected[~usable] = 1
                fit = usable[sample]
                factors[index] = (np.dot(frame[sample][fit], expected[sample][fit])
                                  / np.dot(expected[sample][fit], expected[sample][fit]))
                total += np.where(usable, frame / (factors[index] * expected), 0)
                views += usable
        response = np.where(views >= MIN_VIEWS * num_frames, total / np.maximum(views, 1), 1)
        return response.astype(np.float32), factors


    # The tissue picture is needed to find the flawed sensor pixels, and the flawed pixels should stay out of the
    # tissue picture, because dust leaves a dark trail on it. Two rounds settle this: the second leaves out what
    # the first found.
    flawed = stuck_pixels.copy()
    for _ in range(2):
        tissue = tissue_pictures(left_out=flawed)
        _response, frame_factors = sensor_response(tissue)
        flawed = cv2.dilate((np.abs(_response - 1) > DUST_DEPTH).astype(np.uint8), disc_kernel(DUST_GROW)) > 0
        flawed |= stuck_pixels

    sensor_map = np.where(flawed, _response, 1).astype(np.float32)  # 1 wherever the sensor is fine
    unusable_pixels = stuck_pixels | (sensor_map < DARKEST_GAIN)   # sensor pixels that hold no data in any frame
    # for the previews: the tissue pictures, with the few spots that no frame saw clear smoothed over from around them
    viewing_tissue = [
        cv2.inpaint(np.nan_to_num(picture), np.isnan(picture).astype(np.uint8), 5, cv2.INPAINT_TELEA) for picture in tissue
    ]


    def clean_frame(index):
        """Return a gated frame with the sensor's flaws divided out, and 0 where it holds no data."""
        frame = np.rint(gated_frames[index] / sensor_map).astype(gated_frames.dtype)
        frame[unusable_pixels] = 0
        frame.flat[glare_pixels[index]] = 0
        return frame


    off = (np.abs(sensor_map - 1) > DUST_DEPTH).astype(np.uint8)
    spots, _, spot_stats, spot_centres = cv2.connectedComponentsWithStats(off)
    print(f"Sensor map: {spots - 1} spots where the response is off by more than {DUST_DEPTH:.0%}; "
          f"{(sensor_map != 1).sum()} pixels are corrected and {unusable_pixels.sum()} left out as unusable")
    for spot in np.argsort(-spot_stats[1:, cv2.CC_STAT_AREA])[:8] + 1:
        _left, _top, _width, height, _area = spot_stats[spot]
        _inside = sensor_map[_top:_top + height, _left:_left + _width]
        extreme = _inside.flat[np.argmax(np.abs(_inside - 1))]
        print(f"  (x {spot_centres[spot][0]:.0f}, y {spot_centres[spot][1]:.0f}): {_area} px, "
              f"response {extreme:.2f} at its {'darkest' if extreme < 1 else 'brightest'}")
    return clean_frame, frame_factors, sensor_map, viewing_tissue


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A check of the cleaning, without a video:

    - **Top left:** the sensor map, on a scale that makes even the faint dust spots visible.
    - **Top right:** in how many frames each pixel was left out as glare.
    - **Bottom:** a close-up of the darkest dust spot in the anchor's reference frame, as recorded and cleaned. Black pixels in the cleaned close-up are no data.
    """)
    return


@app.cell
def _(
    anchor,
    clean_frame,
    fixations,
    frame_height,
    frame_shape,
    frame_width,
    gated_frames,
    glare_pixels,
    np,
    plt,
    sensor_map,
):
    CLOSEUP = 240  # pixels; side of the close-up, shown at full resolution

    times_glare = np.bincount(np.concatenate(glare_pixels), minlength=frame_height * frame_width).reshape(frame_shape)
    dust_row, dust_column = np.unravel_index(np.argmin(sensor_map), frame_shape)
    _top = int(np.clip(dust_row - CLOSEUP // 2, 0, frame_height - CLOSEUP))
    _left = int(np.clip(dust_column - CLOSEUP // 2, 0, frame_width - CLOSEUP))
    _window = (slice(_top, _top + CLOSEUP), slice(_left, _left + CLOSEUP))
    _shown = fixations[anchor]["reference"]
    as_recorded, _cleaned = gated_frames[_shown][_window], clean_frame(_shown)[_window]
    _low, _high = np.percentile(as_recorded, [0.5, 99.5])

    _figure, _axes = plt.subplots(2, 2, figsize=(14, 9.5))
    _panels = (
        (sensor_map, "sensor map: the response of every sensor pixel (white: 1, sound; black: 0.85 or less)",
         dict(vmin=0.85, vmax=1.0)),
        (times_glare, f"frames in which each pixel was left out as glare (white: {times_glare.max()})",
         dict(vmin=0, vmax=max(1, times_glare.max()))),
        (as_recorded, f"darkest dust spot (x {dust_column}, y {dust_row}) in frame {_shown}, as recorded",
         dict(vmin=_low, vmax=_high)),
        (_cleaned, "the same frame, cleaned", dict(vmin=_low, vmax=_high)),
    )
    for _axis, (_image, _title, scale) in zip(_axes.ravel(), _panels):
        _axis.imshow(_image, cmap="gray", **scale)
        _axis.set_title(_title)
        _axis.axis("off")
    _figure.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8 · Stabilize every frame

    Each frame now carries two pieces of motion: its own shift within its fixation (§5) and its fixation's map onto the anchor (§6). They are combined into one transform, so every cleaned frame is resampled **once**, with cubic interpolation. Where a frame has no data the value is 0: outside the frame, and where §7 left pixels out. The cubic resampling blends a pixel without data into the 2 pixels around it, so those are left out as well.

    The stabilized frames come in two layouts:

    | Output | Shape · type | Content |
    |---|---|---|
    | `stabilized_fixations` | list of `(frames, rows, columns)` · `uint16` | **one stack per fixation, aligned to its own reference frame**: §5's shifts only, without §6's map. A fixation can be worked on without relying on the alignment between fixations. §9–§16 all work on these. |
    | `stabilized_frames` | `(kept, rows, columns)` · `uint16` | every frame in the **anchor's** frame, laid out like `gated_frames`: the burst as a whole |
    | `stabilized_left_out` | per frame of `stabilized_frames` | the pixels inside the frame without data, as flat indices; the preview uses it |

    The anchor's entry in `stabilized_fixations` is a view of `stabilized_frames`, not a second copy. Copy it before changing it in place.
    """)
    return


@app.cell
def _(
    anchor,
    clean_frame,
    cv2,
    fixations,
    frame_height,
    frame_shape,
    frame_width,
    gated_frames,
    np,
    num_frames,
):
    def anchor_to_reference(fixation):
        """Return the 3x3 map from anchor (x, y) to the same point's (x, y) in this fixation's reference frame."""
        to_reference = np.eye(3)
        to_reference[:len(fixation["to_anchor"])] = fixation["to_anchor"]
        return to_reference


    def warp_frame(index, to_frame):
        """Resample a cleaned frame through a 3x3 map from output (x, y) to that frame's (x, y).

        Returns the resampled frame, and where inside that frame there is no data (the frame is 0 there)."""
        frame = clean_frame(index)
        size = (frame_width, frame_height)
        warped = cv2.warpPerspective(frame, to_frame, size, flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)
        # the cubic resampling blends a pixel without data (0) into the 2 pixels around it, so those are left out too
        empty = cv2.warpPerspective(
            (frame == 0).astype(np.uint8), to_frame, size, flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP
        )
        left_out = cv2.dilate(empty, np.ones((5, 5), np.uint8)) > 0
        warped[left_out] = 0
        return warped, left_out


    # a frame's own shift and its fixation's map onto the anchor are combined into one transform, so each frame
    # is resampled only once. Pixels a frame does not cover stay 0.
    stabilized_frames = np.zeros_like(gated_frames)
    stabilized_fixations = []
    stabilized_left_out = [None] * num_frames  # per frame of stabilized_frames: the pixels inside the frame without data
    for _number, _fixation in enumerate(fixations):
        _members = _fixation["frames"]
        _to_reference = anchor_to_reference(_fixation)
        if _number == anchor:
            # The anchor's own reference is the common frame, so its entry is a view of stabilized_frames
            # rather than a second copy; copy it before changing it in place.
            own_frames = stabilized_frames[_members[0]:_members[-1] + 1]
        else:
            own_frames = np.zeros((len(_members), *frame_shape), dtype=gated_frames.dtype)
        for _position, (_index, (_dy, _dx)) in enumerate(zip(_members, _fixation["shifts"])):
            to_frame = np.array([[1, 0, _dx], [0, 1, _dy], [0, 0, 1.0]])  # this fixation's reference (x, y) -> this frame
            stabilized_frames[_index], _left_out = warp_frame(_index, to_frame @ _to_reference)
            stabilized_left_out[_index] = np.flatnonzero(_left_out)
            if _number != anchor:
                own_frames[_position], _ = warp_frame(_index, to_frame)
        stabilized_fixations.append(own_frames)

    print(f"stabilized_frames: {stabilized_frames.shape} | "
          f"stabilized_fixations: {[frames.shape for frames in stabilized_fixations]}")
    return (
        anchor_to_reference,
        stabilized_fixations,
        stabilized_frames,
        stabilized_left_out,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Did it work?** The plain mean of the anchor's raw frames against the mean of its stabilized frames. Without stabilization the jitter smears every vessel; with it, the fine vessels come out sharp. The bottom row shows the middle of the field at full resolution.
    """)
    return


@app.cell
def _(
    anchor,
    frame_height,
    frame_shape,
    frame_width,
    np,
    plt,
    raw_fixations,
    stabilized_fixations,
):
    # the plain mean of the anchor's raw frames against the mean of its stabilized frames (over the frames with data)
    raw_mean = raw_fixations[anchor].mean(axis=0, dtype=np.float64)
    _total = np.zeros(frame_shape, dtype=np.float64)
    _count = np.zeros(frame_shape, dtype=np.uint16)
    for _frame in stabilized_fixations[anchor]:
        _total += _frame
        _count += _frame > 0
    stabilized_mean = np.where(_count > 0, _total / np.maximum(_count, 1), np.nan)

    _middle = (slice(frame_height // 2 - 160, frame_height // 2 + 160), slice(frame_width // 2 - 240, frame_width // 2 + 240))
    _low, _high = np.nanpercentile(stabilized_mean[::4, ::4], [0.5, 99.5])
    _figure, _axes = plt.subplots(2, 2, figsize=(14, 9.5))
    for _column, (_image, _name) in enumerate(((raw_mean, "mean of the raw frames"),
                                            (stabilized_mean, "mean of the stabilized frames"))):
        for _row, part in enumerate((np.s_[:, :], _middle)):
            _axes[_row, _column].imshow(_image[part], cmap="gray", vmin=_low, vmax=_high)
            _axes[_row, _column].set_title(f"fixation {anchor}: {_name}" + (", middle at full resolution" if _row else ""))
            _axes[_row, _column].axis("off")
    _figure.tight_layout()
    plt.show()
    del raw_mean, _total, _count, stabilized_mean
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The stabilized burst, in the anchor's frame. The vessels should stay in place while the edge of each frame moves around them. For viewing only, a pixel that §7 left out shows what the other frames saw at that spot of tissue; in `stabilized_frames` it is 0.
    """)
    return


@app.cell
def _(
    PREVIEW_WIDTH,
    SHOW_PREVIEWS,
    anchor,
    anchor_to_reference,
    create_animation,
    cv2,
    fixations,
    fps,
    frame_factors,
    frame_height,
    frame_width,
    np,
    num_frames,
    stabilized_frames,
    stabilized_left_out,
    viewing_tissue,
):
    # for viewing only, a pixel that was left out shows what the other frames saw at that spot of tissue
    if SHOW_PREVIEWS:
        view_height = round(PREVIEW_WIDTH * frame_height / frame_width)
        _preview_frames = np.empty((num_frames, view_height, PREVIEW_WIDTH), dtype=np.uint16)
        for _number, _fixation in enumerate(fixations):
            _picture = viewing_tissue[_number]
            if _number != anchor:  # this fixation's tissue picture, mapped into the anchor's frame
                _picture = cv2.warpPerspective(
                    _picture, anchor_to_reference(_fixation), (frame_width, frame_height),
                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                )
            for _index in _fixation["frames"]:
                _shown = stabilized_frames[_index].astype(np.float32)
                _left_out = stabilized_left_out[_index]
                _shown.flat[_left_out] = frame_factors[_index] * _picture.flat[_left_out]
                _preview_frames[_index] = cv2.resize(_shown, (PREVIEW_WIDTH, view_height), interpolation=cv2.INTER_AREA)
        create_animation(_preview_frames, "stabilized_burst", fps=fps)
        del _preview_frames
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9 · The mean frame of each fixation

    After stabilization a pixel shows the same spot of tissue in every frame of a fixation, so its mean over time is the static picture: the anatomy. §10 fits the background on it. Nothing is subtracted here. The temporal mean is taken out later, in density, once the background is gone (§11, §13).

    A frame counts only where it has data. That leaves out a margin of `EDGE_MARGIN` pixels inside its border, where the resampling mixed in the empty outside, and the pixels that §7 marked as no data.

    | Output | Shape · type | Content |
    |---|---|---|
    | `fixation_means` | list of `(rows, columns)` · `float32` | each fixation's mean frame in DN, `NaN` where no frame has data |
    | `frame_area(n, i)` | | the rows and columns of fixation `n`'s reference frame that its frame `i` covers, inside the margin |
    """)
    return


@app.cell
def _(covered_area, fixations, frame_shape, np, stabilized_fixations):
    EDGE_MARGIN = 2  # pixels dropped inside each frame's border, where the cubic resampling mixed in the empty outside


    def frame_area(number, position):
        """Return the (rows, columns) of fixation `number`'s reference frame that its frame `position` really covers,
        EDGE_MARGIN pixels inside the frame's border."""
        dy, dx = fixations[number]["shifts"][position]
        return covered_area(dy, dx, EDGE_MARGIN)


    fixation_means = []  # per fixation: (rows, cols) mean over the frames that cover each pixel, NaN where none does
    for _number, frames in enumerate(stabilized_fixations):
        _total = np.zeros(frame_shape, dtype=np.float32)
        _count = np.zeros(frame_shape, dtype=np.uint16)
        for _position, _frame in enumerate(frames):
            _area = frame_area(_number, _position)
            _total[_area] += _frame[_area]
            _count[_area] += _frame[_area] > 0  # 0 marks the pixels left out as glare or as unusable
        fixation_means.append(np.where(_count > 0, _total / np.maximum(_count, 1), np.nan).astype(np.float32))
        print(f"Fixation {_number}: mean of {len(frames)} frames, "
              f"{_count.sum() / _count.size / len(frames) * 100:.1f}% of pixels have data")
    return fixation_means, frame_area


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10 · Fit the background and measure the density

    The illumination is uneven: on this burst the background is about twice as bright in some places as in others, and the vessels sit on top of it. This step fits the background and measures every pixel against it.

    It works on the logarithm of the image. There, a vessel that absorbs a fixed fraction of the light lowers the value by a fixed amount, whatever the brightness behind it:

    - $L = \ln I$ is the log image;
    - $B$ is the fitted background in the same units: what $L$ would be without the vessels;
    - $D = B - L$ is the **optical density**: about 0 on background and positive in vessels. Where vessels overlap, their densities add.

    **Why the background comes before any temporal mean.** Flicker multiplies a whole frame $t$ by a factor $f_t$ close to 1. In log space that factor is a constant, so it shifts every pixel's density by the same amount:

    $$
    D_t \;=\; B - \ln\!\left(f_t\, I_t\right) \;=\; \left(B - \ln I_t\right) - \ln f_t .
    $$

    Each frame is centred on 0 over the background pixels: its **offset**, its median density there, is subtracted. The shift $-\ln f_t$ is part of that offset, so the flicker leaves with it and needs no step of its own. A camera gamma does not change this, since $(f_t I)^{\gamma} = f_t^{\gamma} I^{\gamma}$ is still one factor per frame. The check at the end of this section measures it.

    `fit_background` works in four stages:

    1. **First guess.** An upper envelope of $L$: a grey closing fills in every dark structure that a disc of `radius` pixels cannot fit inside, so the vessels disappear. A pixel counts as background when its density against this envelope is below the median plus `cutoff` robust standard deviations.
    2. **Refit**, `iterations` times. A mask of everything that stands out against the local noise is built from the current density and widened by `dilation` pixels. $B$ is refitted as a local mean of $L$ over the pixels outside the mask, with wider windows bridging the wide vessels.
    3. **Density refit.** Every pixel whose density is above `od_threshold` times the noise joins the mask, and $B$ is refitted.
    4. **Glare refit.** Pixels far brighter than the background (density below minus `glare_threshold` times the noise) are dropped, and $B$ is refitted.

    **The final fits are thin plates, not local means.** The refits that only build the mask use the local mean (`masked_mean`, `sigma` wide), which is cheap. But a local mean over 20% of the pixels is a mosaic: wherever the background pixels sit more than a window apart, it degenerates to the nearest pixel's value and shows cell edges. Every refit that produces a background that is used (the last of the `iterations`, the density refit, the glare refit, and §12's quiet refit) is instead a **thin plate** (`thin_plate_background`): the surface through the background pixels with the least bending energy, $\sum_{bg} (B - L)^2 + \lambda \sum |\nabla^2 B|^2$, solved on a grid `THIN_PLATE_BLOCK` pixels coarse. $\lambda$ is set by `wavelength`: the fit follows the background down to about that wavelength and bridges anything narrower, with no edges. On this burst, scored on background tiles held out of the fit, a 30 px thin plate interpolates 16 px tiles with an error of 0.0048 against the local mean's 0.0060, and 48 px tiles with 0.0075 against 0.0105 (in optical density; the noise of the mean is 0.0016). Simply widening the local mean makes both worse: the background has real structure at 15 to 40 px.

    The functions below implement it; their docstrings give the details. `burst_density` measures every frame of a stack against one fitted background, and `saturated_pixels` finds the pixels that saturate in any recorded frame, which `fit_background` leaves out.
    """)
    return


@app.cell
def _(
    ThreadPoolExecutor,
    apply_hysteresis_threshold,
    cv2,
    distance_transform_edt,
    downscale_local_mean,
    np,
    scipy,
):
    # background estimation in log space: a first guess of the background from
    # an upper envelope, and the first guess of which pixels are background
    #
    #   L   = ln I, the log image
    #   B0  = upper envelope of L: L with every dark line narrower than the
    #         closing disc filled in
    #   OD0 = B0 - L, the first optical density: ~0 on background, positive in
    #         vessels (overlapping vessels add)
    #
    # then the refit (fit_background): a vessel mask from the contrast-to-noise
    # map takes the vessels out, and B is fitted on the background pixels alone.
    # frame_density applies a fitted B to single frames.


    def log_image(image, valid=None, saturation=4095):
        '''
        natural log of the image. empty (<= 0), saturated (>= saturation) and
        invalid pixels are not data; they take the log of the nearest good pixel

        input: 2d image, optional mask of valid pixels, saturation level (4095
        for 12-bit camera data)

        output: L = ln I as float32, boolean mask of the good pixels
        '''
        image = np.asarray(image, dtype=np.float64)
        ok = np.isfinite(image) & (image > 0) & (image < saturation)
        if valid is not None:
            ok &= valid
        if not ok.any():
            raise ValueError('no usable pixels')

        L = np.log(np.where(ok, image, 1))
        if not ok.all():
            # indices of the nearest good pixel, for every pixel
            nearest = distance_transform_edt(~ok, return_distances=False, return_indices=True)
            L = L[tuple(nearest)]

        return L.astype(np.float32), ok


    def upper_envelope(L, radius=60, block=4, smooth=None):
        '''
        first guess of the background: grey closing (max filter, then min filter
        over a disc) fills every dark structure the disc can't fit inside, so
        vessels disappear and the background around them remains. the result sits
        slightly above the true background (the max filter latches onto noise
        peaks), and below L only on bright details smaller than the block grid
        and smoothing

        for speed the closing runs on a grid `block` times smaller, then is
        smoothed and resized back

        input: log image, disc radius in full-size pixels (by the usual definition
        of closing, dark structures up to about 2 x radius wide get filled), block
        size, smoothing sigma in grid pixels (defaults to radius / block / 2, as
        in the report)

        output: B0, same shape as L
        '''
        rows, cols = L.shape

        # pad to a multiple of the block by reflection so block averages stay honest
        pad_rows, pad_cols = (-rows) % block, (-cols) % block
        padded = np.pad(L, ((0, pad_rows), (0, pad_cols)), mode='reflect')
        small = downscale_local_mean(padded, (block, block)).astype(np.float32)

        grid_radius = max(1, round(radius / block))
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grid_radius + 1,) * 2)
        closed = cv2.morphologyEx(small, cv2.MORPH_CLOSE, disc, borderType=cv2.BORDER_REFLECT)

        if smooth is None:
            smooth = grid_radius / 2
        closed = cv2.GaussianBlur(closed, (0, 0), smooth, borderType=cv2.BORDER_REFLECT)

        full = cv2.resize(closed, (padded.shape[1], padded.shape[0]),
                          interpolation=cv2.INTER_LINEAR)
        return full[:rows, :cols]


    def robust_sd(x):
        '''
        standard deviation estimated from the median absolute deviation, so a
        minority of extreme values (vessels) barely affect it
        '''
        return 1.4826 * np.median(np.abs(x - np.median(x)))


    def first_background(image, valid=None, radius=60, block=4, cutoff=2, saturation=4095):
        '''
        the report's first guess of the background and of which pixels are
        background

        a pixel counts as background when its (lightly blurred) optical density is
        below median + cutoff x robust sd: most pixels are background, so the
        median is typical background and vessels are the outliers above it

        input: 2d image, optional valid mask, envelope settings (see
        upper_envelope), cutoff in robust standard deviations, saturation level

        output: dict with L (log image), ok (usable pixels), B0 (first background,
        log units), OD0 (B0 - L) and bg (first background pixels)
        '''
        L, ok = log_image(image, valid, saturation)
        B0 = upper_envelope(L, radius, block)
        OD0 = B0 - L

        S = cv2.GaussianBlur(OD0, (0, 0), 1)
        values = S[ok]
        bg = ok & (S < np.median(values) + cutoff * robust_sd(values))

        return {'L': L, 'ok': ok, 'B0': B0, 'OD0': OD0, 'bg': bg}


    # background fitting, the rest of the report's step 1. the background is
    # refitted on background pixels only, after a vessel mask built from the
    # contrast-to-noise map (the report's step 2) has removed the vessels


    def blur(x, sigma):
        '''
        gaussian blur with reflected borders. wide blurs are smooth, so above
        8 px they run on a smaller grid and are resized back, which is much faster
        and visually identical
        '''
        x = np.asarray(x, dtype=np.float32)
        if sigma <= 8:
            return cv2.GaussianBlur(x, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)

        factor = int(sigma // 4)
        rows, cols = x.shape
        small = cv2.resize(x, (max(1, cols // factor), max(1, rows // factor)),
                           interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), sigma / factor, borderType=cv2.BORDER_REFLECT)
        return cv2.resize(small, (cols, rows), interpolation=cv2.INTER_LINEAR)


    def masked_mean(x, weight, sigma, octaves=4, falloff=0.03):
        '''
        local mean of x over the pixels where weight is 1 ("normalized
        convolution"). wider windows, each counting falloff times less than the
        one before, fill holes the narrow one can't reach, so wide vessels are
        bridged from the background around them

        input: 2d image, boolean / 0-1 weight, base sigma in pixels

        output: local mean, same shape
        '''
        weight = np.asarray(weight, dtype=np.float32)
        x = np.asarray(x, dtype=np.float32)
        numerator = np.zeros_like(x)
        denominator = np.zeros_like(x)
        for k in range(octaves):
            numerator += falloff ** k * blur(x * weight, sigma * 2 ** k)
            denominator += falloff ** k * blur(weight, sigma * 2 ** k)

        # with no background anywhere near, fall back to the global background mean
        overall = x[weight > 0].mean() if (weight > 0).any() else x.mean()
        eps = falloff ** octaves
        return (numerator + eps * overall) / (denominator + eps)


    THIN_PLATE_BLOCK = 4  # pixels; the thin plate is solved on a grid this many pixels coarse


    def grid_laplacian(x):
        """5-point Laplacian on a grid, with reflecting borders (a border pixel's missing neighbour is itself)."""
        out = -4.0 * x
        out[1:] += x[:-1]
        out[:-1] += x[1:]
        out[:, 1:] += x[:, :-1]
        out[:, :-1] += x[:, 1:]
        out[0] += x[0]
        out[-1] += x[-1]
        out[:, 0] += x[:, 0]
        out[:, -1] += x[:, -1]
        return out


    def thin_plate_background(x, weight, wavelength=30.0, block=THIN_PLATE_BLOCK, tolerance=1e-6, max_iterations=500):
        '''
        the smoothest surface through the pixels where weight is 1: the thin
        plate that minimizes sum_bg (B - x)^2 + lambda sum |laplacian B|^2. it
        bridges holes of any width without the edges a local mean shows where
        the background pixels are sparse

        the fit runs on a grid `block` pixels coarse (block means of x over the
        weighted pixels, weighted by their share of the block) and is brought
        back to full size with cubic interpolation. lambda comes from
        `wavelength`: the surface follows the data down to about that
        wavelength (half power at wavelength = 2 pi lambda^(1/4) grid pixels)

        the normal equations (W + lambda L^T L) B = W x are solved by conjugate
        gradients, preconditioned by the same operator with uniform weights,
        which a cosine transform diagonalizes. a few hundred cheap iterations
        reach the direct solution to about 1e-4 of the density noise

        input: 2d image, boolean / 0-1 weight, wavelength in full-size pixels,
        block size, relative tolerance, iteration cap

        output: the fitted surface, same shape as x, float32
        '''
        x = np.asarray(x, dtype=np.float32)
        rows, cols = x.shape
        pad_rows, pad_cols = (-rows) % block, (-cols) % block
        padded = np.pad(x, ((0, pad_rows), (0, pad_cols)), mode='reflect')
        w = np.pad(np.asarray(weight, dtype=np.float32), ((0, pad_rows), (0, pad_cols)), mode='constant')
        w_small = downscale_local_mean(w, (block, block)).astype(np.float64)
        y = downscale_local_mean(padded * w, (block, block)) / np.maximum(w_small, 1e-9)
        y = np.where(w_small > 0, y, 0.0).astype(np.float64)
        m, n = y.shape
        mean = (w_small * y).sum() / w_small.sum()  # solved as a deviation from the mean: the Laplacian ignores constants
        y -= mean
        lam = (wavelength / (2 * np.pi * block)) ** 4

        # the preconditioner: the operator with every weight at the average, diagonal in the cosine basis
        k, l = np.arange(m)[:, None], np.arange(n)[None, :]
        mu = -4 * np.sin(np.pi * k / (2 * m)) ** 2 - 4 * np.sin(np.pi * l / (2 * n)) ** 2  # eigenvalues of grid_laplacian
        spectrum = w_small.mean() + lam * mu ** 2

        def apply(v):
            return w_small * v + lam * grid_laplacian(grid_laplacian(v))

        def precondition(r):
            return scipy.fft.idctn(scipy.fft.dctn(r, type=2, norm='ortho') / spectrum, type=2, norm='ortho')

        b = w_small * y
        b_norm = np.sqrt((b * b).sum())
        solution = precondition(b)
        residual = b - apply(solution)
        z = precondition(residual)
        direction = z.copy()
        rz = (residual * z).sum()
        for _ in range(max_iterations):
            a_direction = apply(direction)
            step = rz / (direction * a_direction).sum()
            solution += step * direction
            residual -= step * a_direction
            if np.sqrt((residual * residual).sum()) <= tolerance * b_norm:
                break
            z = precondition(residual)
            rz_new = (residual * z).sum()
            direction = z + (rz_new / rz) * direction
            rz = rz_new

        full = cv2.resize((solution + mean).astype(np.float32), (padded.shape[1], padded.shape[0]),
                          interpolation=cv2.INTER_CUBIC)
        return full[:rows, :cols]


    def local_rms(d, bg, window):
        '''
        typical size of d near each pixel, measured on background pixels only.
        values beyond 3 robust standard deviations are capped, and the result is
        floored at 5 % of the global level

        input: 2d signal, boolean background mask, window sigma in pixels

        output: local rms, same shape
        '''
        level = 1.4826 * np.median(np.abs(d[bg]))
        capped = np.minimum(d ** 2, (3 * level) ** 2)
        return np.sqrt(np.maximum(masked_mean(capped, bg, window), (0.05 * level) ** 2))


    def band_cnr(od, bg, bands=(1, 2, 4, 8, 16, 32)):
        '''
        contrast against the local noise (the report's step 2): at each size s,
        centre minus a surround twice as wide, divided by its local noise on
        background pixels. a thin vessel stands out at a small size, a wide one at
        a large size

        input: optical density, boolean background mask, band sizes in pixels

        output: Z, the contrast-to-noise ratio of the best band
        '''
        z = np.full(od.shape, -np.inf, dtype=np.float32)
        for s in bands:
            d = blur(od, s) - blur(od, 2 * s)
            z = np.maximum(z, d / local_rms(d, bg, max(24, 4 * s)))
        return z


    def fit_background(image, valid=None, radius=60, block=4, cutoff=2, iterations=2,
                       z_threshold=2, z_low=1, noise_threshold=3, dilation=5, close_radius=0,
                       od_threshold=2, glare_threshold=5, sigma=8, wavelength=30.0, noise_window=48,
                       bands=(1, 2, 4, 8, 16, 32), saturation=4095):
        """
        the report's full step 1, with a wider vessel mask: start from the upper
        envelope (first_background), then repeat `iterations` times:
          1. Z = contrast-to-noise of the current OD (band_cnr)
          2. vessel mask M = Z above z_threshold, plus weaker Z above z_low that
             connects to it (hysteresis: faint stretches of a visible vessel count,
             isolated texture doesn't), plus lightly blurred OD above
             noise_threshold x the local noise. optionally closed with a disc of
             close_radius (fills a gap along an artery's central light reflex),
             then dilated by `dilation` px so the vessels' soft edges don't leak
             into the background
          3. refit the background B on the pixels outside M: a local mean
             (masked_mean, sigma) while the mask is still being built, a thin
             plate (thin_plate_background, wavelength) for every background
             that is used: the last iteration and the refits below. with
             wavelength=None those are local means too (§15's smear)
          4. OD = B - L
        then two optional extra refits:
          - od refit: add every pixel whose OD is above od_threshold x noise to
            the mask, refit (the report's "change 1": the density itself finds
            the faint vessel stretches the contrast map missed)
          - glare: pixels far brighter than the background (OD below
            -glare_threshold x noise) are not background either; drop them and
            refit

        on the test burst, the wider mask (hysteresis, 5 px dilation, od refit)
        keeps ~40 % more density on faint vessels and removes the dip of B under
        wide vessels, while the background residual gets flatter, not worse

        input: 2d image, optional valid mask (pass ~saturated, from the frames:
        a pixel saturated in some frames averages below the saturation level),
        first-guess settings (see first_background), number of refits, mask
        thresholds (z_low=None for a plain threshold, od_threshold=None or
        glare_threshold=None to skip those refits), mask closing and dilation,
        background sigma (mask-building refits), thin-plate wavelength
        (final refits; None for a local mean of `sigma` there as well),
        noise window, contrast bands, saturation level (all sizes in pixels)

        output: dict with L, ok, B0, OD0, bg0 (the first guess and its background
        pixels), B (fitted background, log units), OD (B - L, the target),
        M (pixels kept out of the fit: deliberately generous, not a vessel
        segmentation), bg (background
        pixels), Z (contrast-to-noise), noise (local noise of OD), glare (pixels
        dropped as glare), and history: one dict per refit with its name and that
        refit's M, bg, B and OD
        """
        first = first_background(image, valid, radius, block, cutoff, saturation)
        L, ok, bg = first['L'], first['ok'], first['bg']
        B, od = first['B0'], first['OD0']
        vessels, z = ~bg, None
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation + 1,) * 2)
        history = []

        def refit(name, final=False):
            nonlocal bg, B, od
            bg = ok & ~vessels
            B = thin_plate_background(L, bg, wavelength) if final and wavelength else masked_mean(L, bg, sigma)
            od = B - L
            history.append({'name': name, 'M': vessels, 'bg': bg, 'B': B, 'OD': od})

        def noise_of(od, bg):
            smooth = blur(od, 1)
            return smooth, local_rms(smooth - np.median(smooth[bg]), bg, noise_window)

        smooth, noise = noise_of(od, bg)
        for i in range(iterations):
            z = band_cnr(od, bg, bands)
            smooth, noise = noise_of(od, bg)
            if z_low is None:
                vessels = z > z_threshold
            else:
                vessels = apply_hysteresis_threshold(z, z_low, z_threshold)
            vessels |= smooth > noise_threshold * noise
            if close_radius:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_radius + 1,) * 2)
                vessels = cv2.morphologyEx(vessels.astype(np.uint8), cv2.MORPH_CLOSE,
                                           kernel).astype(bool)
            vessels = cv2.dilate(vessels.astype(np.uint8), disc).astype(bool)
            refit(f'refit {i + 1}', final=(i == iterations - 1))

        if od_threshold is not None:
            vessels = vessels | (blur(od, 1) > od_threshold * noise)
            refit('od refit', final=True)

        glare = np.zeros(od.shape, bool)
        if glare_threshold is not None:
            glare = ok & (od < -glare_threshold * noise)
            if glare.any():
                ok = ok & ~glare
                refit('glare refit', final=True)

        if not ok.all():
            nearest = distance_transform_edt(~ok, return_distances=False, return_indices=True)
            od = od[tuple(nearest)]
            if history:
                history[-1]['OD'] = od

        return {**first, 'bg0': first['bg'], 'ok': ok, 'B': B, 'OD': od, 'M': vessels,
                'bg': bg, 'Z': z, 'noise': noise, 'glare': glare, 'history': history}


    def frame_density(frame, background, background_pixels=None, saturation=4095):
        """
        optical density of a single stabilized frame against a background
        fitted on the fixation's mean frame, so every frame gets exactly the
        same subtraction. if background_pixels is given, the frame's median
        density on those pixels is subtracted, so a frame exposed a little
        differently from the average still centres on zero

        input: 2d frame in the fixation's reference frame, B from
        fit_background, optional background-pixel mask, saturation level

        output: OD of the frame (nan where the frame isn't data), the offset
        that was subtracted
        """
        frame = np.asarray(frame, dtype=np.float32)
        ok = np.isfinite(frame) & (frame > 0) & (frame < saturation)
        L = np.log(np.where(ok, frame, 1))

        od = np.where(ok, background - L, np.nan).astype(np.float32)
        offset = 0.0
        if background_pixels is not None:
            values = od[background_pixels & np.isfinite(od)]
            if values.size:
                offset = float(np.median(values))
                od -= offset
        return od, offset


    def burst_density(frames, background, background_pixels=None, saturation=4095, workers=None):
        """
        optical density of every frame of a stabilized stack against one
        fitted background: frame_density applied to each frame, so the vessels
        of every frame are measured against the same background. the result is
        a density burst in which the vessels sit still and the background is
        flat: the burst to watch, average, or follow over time

        input: stack of shape (frames, rows, cols), B from fit_background,
        optional background-pixel mask (each frame is centred on zero over it),
        saturation level, number of threads

        output: density burst as float32 (nan where a frame has no data), and the
        (frames,) offsets subtracted
        """
        density = np.empty(frames.shape, dtype=np.float32)
        offsets = np.zeros(len(frames))

        def one(i):
            density[i], offsets[i] = frame_density(frames[i], background, background_pixels, saturation)

        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(one, range(len(frames))))

        return density, offsets


    def saturated_pixels(frames, transforms, saturation=4095):
        """
        mask of the pixels saturated in any frame of a fixation, in its
        reference frame, for fit_background's `valid`. tested on the raw frames: the
        stabilized frames are interpolated, which overshoots near bright spots
        and would flag pixels that never saturated, and the average can't tell,
        since a pixel saturated in some frames averages below the saturation level

        input: the fixation's frames as recorded, of shape (frames, rows, cols),
        the (frames, 2, 3) maps from the reference frame to each frame,
        saturation level

        output: boolean mask, same shape as one frame
        """
        rows, cols = frames.shape[1:]
        saturated = np.zeros((rows, cols), bool)
        for frame, transform in zip(frames, transforms):
            hit = frame >= saturation
            if hit.any():
                warped = cv2.warpAffine(hit.astype(np.uint8), transform, (cols, rows),
                                        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                saturated |= warped.astype(bool)
        return saturated

    return (
        blur,
        burst_density,
        fit_background,
        masked_mean,
        robust_sd,
        saturated_pixels,
        thin_plate_background,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each fixation gets its own background, fitted on its mean frame by `fit_background` alone, without temporal data: the mask is built with local means `BACKGROUND_SIGMA` wide, the background itself is a thin plate that follows the tissue down to `BACKGROUND_WAVELENGTH` pixels. §12 refits it on the quiet pixels. The illumination is fixed to the camera while stabilization fixes the tissue, so after a saccade the lighting pattern lies somewhere else on the tissue. That is why each fixation needs its own fit. Pixels that saturate in any frame are kept out of the fit; they are looked up in `gated_frames`, before any resampling.

    Every stabilized frame is then measured against its fixation's $B$ and centred on 0 over the background pixels (`measure_density`). `temporal_mean` gives the mean density over time. §11 finds the quiet pixels from these, and §12 measures every frame again against the refitted background.

    | Output | Content |
    |---|---|
    | `background_fits` | per fixation, the dictionary from `fit_background`, in the fixation's own reference frame: `B` the fitted background, `OD` the density of the mean frame, `bg` the background pixels, `M` the pixels kept out of the fit, `ok` the usable pixels, and `history` with every refit |
    | `density_fixations` | per fixation, `(frames, rows, columns)` `float32` optical density, `NaN` where a frame has no data |
    | `density_offsets` | per fixation, the offset subtracted from each frame |
    | `density_means` | per fixation, `(rows, columns)`: the mean density over the frames that have data at each pixel |
    | `measure_density()`, `temporal_mean()` | the two functions; §12 uses them again |
    """)
    return


@app.cell
def _(
    burst_density,
    fit_background,
    fixation_means,
    fixations,
    frame_area,
    frame_shape,
    gated_frames,
    np,
    saturated_pixels,
    stabilized_fixations,
    workers,
):
    BACKGROUND_SIGMA = 8        # pixels; width of the local means that build the vessel mask inside fit_background
    BACKGROUND_WAVELENGTH = 30  # pixels; the thin-plate background follows the tissue down to about this wavelength


    def measure_density(number, log_background, background):
        """Return the optical density of every stabilized frame of fixation `number` against a fitted background, each
        frame centred on 0 over the background pixels and NaN where it has no data, and the offsets that were subtracted."""
        density, offsets = burst_density(stabilized_fixations[number], log_background, background_pixels=background,
                                         workers=workers)
        for position, frame in enumerate(density):
            # the data area of the mean frame: without the rim where the resampling blended in the empty outside
            inside = np.zeros(frame_shape, dtype=bool)
            inside[frame_area(number, position)] = True
            frame[~inside] = np.nan
        return density, offsets


    def temporal_mean(stack):
        """Return a stack's mean over time, over the frames that have data at each pixel (NaN where none has)."""
        total = np.zeros(frame_shape, dtype=np.float32)
        count = np.zeros(frame_shape, dtype=np.uint16)
        for frame in stack:
            total += np.nan_to_num(frame)
            count += np.isfinite(frame)
        return np.where(count > 0, total / np.maximum(count, 1), np.nan).astype(np.float32)


    background_fits = []    # per fixation: the dict from fit_background (B, OD, bg, M, ...), in the fixation's own frame
    density_fixations = []  # per fixation: (frames, rows, cols) float32 optical density, NaN where a frame has no data
    density_offsets = []    # per fixation: what was subtracted from each frame to centre its background on 0
    density_means = []      # per fixation: (rows, cols) mean density over the frames that have data at each pixel
    for _number, _fixation in enumerate(fixations):
        _members = _fixation["frames"]
        # a pixel that saturates in any frame is not background data; these are looked up in the frames as recorded
        to_frames = np.float64([[[1, 0, dx], [0, 1, dy]] for dy, dx in _fixation["shifts"]])  # reference (x, y) -> frame
        saturated = saturated_pixels(gated_frames[_members[0]:_members[-1] + 1], to_frames)
        _fit = fit_background(fixation_means[_number], valid=~saturated, sigma=BACKGROUND_SIGMA,
                             wavelength=BACKGROUND_WAVELENGTH)
        _density, _offsets = measure_density(_number, _fit["B"], _fit["bg"])
        background_fits.append(_fit)
        density_fixations.append(_density)
        density_offsets.append(_offsets)
        density_means.append(temporal_mean(_density))
        print(f"Fixation {_number}: background fitted on {_fit['bg'].mean() * 100:.0f}% of the pixels "
              f"({saturated.sum()} saturated and {_fit['glare'].sum()} glare pixels left out) | "
              f"frame offsets {_offsets.min():+.4f} to {_offsets.max():+.4f}")
    return (
        BACKGROUND_WAVELENGTH,
        background_fits,
        density_fixations,
        density_means,
        density_offsets,
        measure_density,
        temporal_mean,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **No flicker left.** Each frame's whole-field level, before and after the background. It is the mean over every pixel with data, vessels included, not only the background pixels the offsets were taken on, of

    - $\ln(\text{frame} / \text{mean frame})$ for the stabilized frame against its fixation's mean (before), and
    - $\bar{D} - D_t$ for the frame's density against its fixation's mean density (after),

    both in log units, which are fractions of the brightness, and both relative to the fixation's own average. Before, whole frames swing by up to 1.5% from one frame to the next; after, they stay within about 0.1%.
    """)
    return


@app.cell
def _(
    INK_MUTED,
    SERIES,
    density_fixations,
    density_means,
    fixation_means,
    fixation_members,
    fixations,
    np,
    num_frames,
    plt,
    stabilized_fixations,
):
    # each frame's whole-field level, on every 4th pixel along each axis, before the background (the stabilized frame
    # against its fixation's mean frame) and after it (the frame's density against the fixation's mean density)
    _sample = (slice(None, None, 4), slice(None, None, 4))
    before, _after = np.zeros(num_frames), np.zeros(num_frames)
    for _number, _fixation in enumerate(fixations):
        _mean, _mean_density = fixation_means[_number][_sample], density_means[_number][_sample]
        for _position, _index in enumerate(_fixation["frames"]):
            _density = density_fixations[_number][_position][_sample]
            _has_data = np.isfinite(_density)
            _frame = stabilized_fixations[_number][_position][_sample][_has_data].astype(np.float64)
            before[_index] = np.mean(np.log(_frame / _mean[_has_data]))
            _after[_index] = np.mean(_mean_density[_has_data] - _density[_has_data])
        for _level in (before, _after):
            _level[_fixation["frames"]] -= _level[_fixation["frames"]].mean()  # relative to the fixation's own average
    print(f"Whole-field level from frame to frame (rms): {np.sqrt(np.mean(before ** 2)) * 100:.2f}% of the brightness "
          f"before the background, {np.sqrt(np.mean(_after ** 2)) * 100:.3f}% after")

    _figure, _axis = plt.subplots(figsize=(14, 3.2))
    for _members in fixation_members[1:]:
        _axis.axvline(_members[0] - 0.5, color="#c3c2b7", linestyle=":", linewidth=1)  # where a new fixation starts
    _axis.plot(before * 100, color=INK_MUTED, linewidth=1.5, label="stabilized frame / mean frame (before the background)")
    _axis.plot(_after * 100, color=SERIES[0], linewidth=1.5, label="mean density - frame density (after)")
    _axis.set_xlabel("frame")
    _axis.set_ylabel("whole-field level (% of brightness)")
    _axis.set_title("flicker: each frame's whole-field level, relative to its fixation's average")
    _low, _high = _axis.get_ylim()
    _axis.set_ylim(_low, _high + 0.45 * (_high - _low))  # room for the legend above the lines
    _axis.legend(loc="upper center", ncol=2)
    _figure.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11 · Find the quiet pixels

    §12 refits the background through the pixels that do not change over time. Blood moving through a vessel makes a pixel's brightness fluctuate from frame to frame by more than the camera noise does, while tissue without flow shows only the noise. Pixels that show only the noise are **quiet**.

    The fluctuation is measured on the **temporal burst**, each frame's density minus its fixation's mean density, $\bar{D} - D_t$. The background is the same in every frame, so it drops out. What is left is the frame's relative change in brightness, in log units, with the flicker already gone. For each frame:

    1. The frame's own smooth part is removed (`remove_smooth_part`): structure larger than about `SMOOTH_SIGMA` pixels, the slow shading that drifts as the eye moves. §13 follows that shading properly.
    2. The rest is blurred by `FLUCTUATION_BLUR` pixels to beat the pixel noise.
    3. **The slow movement is removed.** The pixel's running mean over `SLOW_WINDOW` frames (centred on the frame) is subtracted. Blood shows as changes that last a frame or a few; what remains after a quarter of a second is not flow. On this burst, judged by the total fluctuation, two thirds of the field "changed", in large patches between the vessels whose traces wander slowly by 1 to 2% over seconds: a third of that is locked to the eye's drift (parallax, or motion the per-frame translation cannot follow), the rest slow and patchy, tear film or focus. Neither is blood.
    4. What is left is squared, multiplied by the brightness and added up in the anchor's frame.

    Shot noise makes a relative change of about $1/\sqrt{\text{brightness}}$, so multiplying by the brightness makes the result level wherever only noise acts. That level is the **noise floor**, read off as the peak of the histogram. `fluctuation_ratio` is the fluctuation divided by the noise floor: about 1 on quiet tissue, and 2 to 4 in vessels on this burst.

    A pixel **changes** when its ratio is above `ACTIVE_LEVELS[1]`, or above `ACTIVE_LEVELS[0]` and connected to a pixel that is (hysteresis: faint stretches of something that changes count, isolated specks don't). A pixel is **quiet** when it lies more than `ACTIVE_GROW` pixels from every pixel that changes, so that the soft edges of vessels are not taken for background. Where fewer than `MIN_FRAMES` frames have data a pixel is not judged; it is neither.

    With the slow movement removed, this burst's fixation 0 goes from 67% changing and 17% quiet (`SLOW_WINDOW = 0`, the total fluctuation) to 48% changing and 33% quiet with a 9-frame window, or 40% and 40% with a 5-frame window. Vessels without countable cells (deep and blurred, or packed) show no fast change either; §12 keeps them out of the background with a guard.

    | Output | Content |
    |---|---|
    | `fluctuation_ratio` | `(rows, columns)`: the fluctuation over the noise floor in the anchor's frame, `NaN` where not judged |
    | `changing_pixels`, `quiet_pixels` | per fixation, boolean `(rows, columns)` in the fixation's own reference frame |
    | `remove_smooth_part()`, `blur_valid()` | two helpers that ignore the pixels without data; §14 uses `blur_valid` again |
    | `SLOW_WINDOW` | the window of the slow movement; §13 uses it again for the preview |
    """)
    return


@app.cell
def _(
    anchor,
    anchor_to_reference,
    apply_hysteresis_threshold,
    cv2,
    density_fixations,
    density_means,
    fixation_means,
    fixations,
    frame_height,
    frame_shape,
    frame_width,
    np,
    plt,
):
    FLUCTUATION_BLUR = 2.0        # pixels; each frame's change is blurred by this to beat the pixel noise
    ACTIVE_LEVELS = (1.25, 1.5)   # x the noise floor; a pixel changes above the second, or above the first next to one that does
    ACTIVE_GROW = 5               # pixels; a quiet pixel is farther than this from every pixel that changes
    MIN_FRAMES = 40               # a pixel is judged only where at least this many frames have data
    SLOW_WINDOW = 9               # frames; a pixel's running mean over this many frames is its slow movement, and is removed (0: keep it)

    SMOOTH_SIGMA = 32             # pixels; structure larger than about this counts as a frame's smooth part
    SMOOTH_SCALE = 8              # the smooth part is computed on frames reduced by this factor


    def remove_smooth_part(frame):
        """Subtract a frame's own large-scale part, ignoring the pixels without data."""
        reduced = (frame_width // SMOOTH_SCALE, frame_height // SMOOTH_SCALE)
        has_data = np.isfinite(frame)
        values = cv2.resize(np.where(has_data, frame, 0), reduced, interpolation=cv2.INTER_AREA)
        weight = cv2.resize(has_data.astype(np.float32), reduced, interpolation=cv2.INTER_AREA)
        sigma = SMOOTH_SIGMA / SMOOTH_SCALE
        smooth = cv2.GaussianBlur(values, (0, 0), sigma) / np.maximum(cv2.GaussianBlur(weight, (0, 0), sigma), 1e-3)
        return frame - cv2.resize(smooth, (frame_width, frame_height), interpolation=cv2.INTER_CUBIC)


    def blur_valid(frame, sigma):
        """Gaussian blur that ignores the pixels without data, which stay NaN."""
        has_data = np.isfinite(frame)
        values = cv2.GaussianBlur(np.where(has_data, frame, 0), (0, 0), sigma)
        weight = cv2.GaussianBlur(has_data.astype(np.float32), (0, 0), sigma)
        return np.where(has_data, values / np.maximum(weight, 1e-3), np.nan)


    def fast_changes(density, mean_density):
        """Yield each frame's change from the mean with its smooth part, the pixel noise and the slow movement taken out.

        The slow movement is the pixel's running mean over SLOW_WINDOW frames centred on the frame; the first and last
        SLOW_WINDOW // 2 frames of a fixation have no centred window and are skipped."""
        window = []  # the processed changes of the last SLOW_WINDOW frames
        total = np.zeros(frame_shape, dtype=np.float32)
        count = np.zeros(frame_shape, dtype=np.float32)
        for frame in density:
            change = mean_density - frame  # the temporal burst: the frame's relative change from the mean, in log units
            change = blur_valid(remove_smooth_part(change), FLUCTUATION_BLUR)
            if SLOW_WINDOW < 2:
                yield change
                continue
            window.append(change)
            total += np.nan_to_num(change)
            count += np.isfinite(change)
            if len(window) > SLOW_WINDOW:
                gone = window.pop(0)
                total -= np.nan_to_num(gone)
                count -= np.isfinite(gone)
            if len(window) == SLOW_WINDOW:
                slow = np.where(count > 0, total / np.maximum(count, 1), np.nan)
                yield window[SLOW_WINDOW // 2] - slow


    # the fluctuation at each spot of tissue, added up over the frames of all fixations in the anchor's frame
    squares = np.zeros(frame_shape, dtype=np.float32)
    frames_counted = np.zeros(frame_shape, dtype=np.float32)
    for _number, (_fixation, _mean, _density, _mean_density) in enumerate(zip(fixations, fixation_means, density_fixations,
                                                                          density_means)):
        _to_reference = anchor_to_reference(_fixation)
        for _change in fast_changes(_density, _mean_density):
            _has_data = np.isfinite(_change)
            energy = np.where(_has_data, _change**2 * _mean, 0)  # times the brightness, which levels the shot noise
            _has_data = _has_data.astype(np.float32)
            if _number != anchor:
                energy, _has_data = (
                    cv2.warpPerspective(image, _to_reference, (frame_width, frame_height),
                                        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
                    for image in (energy, _has_data)
                )
            squares += energy
            frames_counted += _has_data

    _judged = frames_counted >= MIN_FRAMES
    activity = np.sqrt(squares / np.maximum(frames_counted, 1))
    # the noise floor is where the quiet pixels sit: the peak of the histogram
    _histogram, _edges = np.histogram(activity[_judged], bins=400, range=(0, np.percentile(activity[_judged], 99)))
    noise_floor = _edges[np.argmax(_histogram):][:2].mean()
    fluctuation_ratio = np.where(_judged, activity / noise_floor, np.nan).astype(np.float32)

    changing = apply_hysteresis_threshold(np.nan_to_num(fluctuation_ratio), *ACTIVE_LEVELS)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ACTIVE_GROW + 1,) * 2)
    quiet = _judged & ~cv2.dilate(changing.astype(np.uint8), disc).astype(bool)

    changing_pixels = []  # per fixation: True where the tissue changes over time, in the fixation's own frame
    quiet_pixels = []     # per fixation: True where it is quiet, clear of everything that changes
    for _number, _fixation in enumerate(fixations):
        masks = [changing, quiet]
        if _number != anchor:
            masks = [
                cv2.warpPerspective(mask.astype(np.uint8), anchor_to_reference(_fixation), (frame_width, frame_height),
                                    flags=cv2.INTER_NEAREST).astype(bool)
                for mask in masks
            ]
        changing_pixels.append(masks[0])
        quiet_pixels.append(masks[1])

    print(f"Fluctuation measured on up to {frames_counted.max():.0f} frames per pixel"
          f"{f', slow movement over {SLOW_WINDOW} frames removed' if SLOW_WINDOW >= 2 else ''}, {_judged.mean() * 100:.1f}% of the "
          f"pixels judged | of those, {changing[_judged].mean() * 100:.1f}% change and {quiet[_judged].mean() * 100:.1f}% are quiet")

    # show the fluctuation and the quiet pixels, in the anchor's frame
    _figure, _axes = plt.subplots(1, 2, figsize=(16, 5.3))
    _axes[0].imshow(np.nan_to_num(fluctuation_ratio), cmap="gray", vmin=0.8, vmax=2.5)
    _axes[0].set_title("fluctuation over time / noise floor (black: 0.8 or less, white: 2.5 or more)")
    _axes[1].imshow(quiet, cmap="gray", vmin=0, vmax=1)
    _axes[1].set_title("quiet pixels (white)")
    for _axis in _axes:
        _axis.axis("off")
    _figure.tight_layout()
    plt.show()
    return SLOW_WINDOW, blur_valid, changing_pixels, fluctuation_ratio, quiet_pixels


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 12 · Refit the background on the quiet pixels

    The background of §10 was fitted from the mean frame alone, judging a pixel by how much it stands out from its surroundings. Now the temporal data decide what is background. It is fitted once more as a thin plate (`thin_plate_background`, `BACKGROUND_WAVELENGTH`), but this time a pixel is background when it is quiet (§11). So that there is background close to the vessels as well, a pixel that §10 took as background stays background unless it changes. Where a pixel was not judged, §10's choice stays.

    **The guard.** A quiet pixel is not background when it stands out spatially as a vessel: `blur(OD, 1)` above `VESSEL_GUARD` times the local noise, `fit_background`'s own criterion. Vessels without countable cells show no fast change: the packed cores of the big vessels, and the deep, blurred vessels below the focal plane. On this burst the guard removes about a fifth of the quiet pixels; without it the background would be fitted through those vessels and take them out of the density.

    Every frame is then measured again against the refitted $B$, and `density_means` is made again. Each fixation's first measurement is freed before the second is made, so the two never sit in memory together: §10's lists are refitted in place. The cell gives them back under names of their own, `final_…`, and every later section reads those, so marimo knows that what follows depends on this cell. The cell can be run again: it always refits from §10's fit.

    | Output | Content |
    |---|---|
    | `final_background_fits` | `background_fits`, updated: `B`, `OD`, `bg` and `M` are now the quiet refit's. Its `history` ends with a `"quiet refit"` entry, and the entry before that is §10's fit. |
    | `final_density_fixations`, `final_density_offsets`, `final_density_means` | `density_fixations`, `density_offsets` and `density_means`, measured again against the refitted background: **the final density** |
    """)
    return


@app.cell
def _(
    BACKGROUND_WAVELENGTH,
    background_fits,
    blur,
    changing_pixels,
    density_fixations,
    density_means,
    density_offsets,
    distance_transform_edt,
    fixations,
    measure_density,
    quiet_pixels,
    robust_sd,
    temporal_mean,
    thin_plate_background,
):
    VESSEL_GUARD = 3  # x the local noise; a quiet pixel whose density stands out by this much is a vessel, not background

    for _number in range(len(fixations)):
        _fit = background_fits[_number]
        _fit["history"] = [entry for entry in _fit["history"] if entry["name"] != "quiet refit"]  # when run again
        spatial = _fit["history"][-1]  # §10's fit, made from the mean frame alone
        # quiet refit: one more refit like those inside fit_background, with the temporal data deciding what is
        # background. The quiet pixels are background, and so are the pixels fit_background chose unless they change
        # (which is also what stays where a pixel was not judged).
        usable = _fit["ok"]
        # the guard: a quiet pixel that stands out as a vessel holds blood that shows no fast change (packed, deep or blurred)
        _vessel = blur(spatial["OD"], 1) > VESSEL_GUARD * _fit["noise"]
        guarded = (quiet_pixels[_number] & _vessel).sum() / max(quiet_pixels[_number].sum(), 1)
        _background = usable & ((quiet_pixels[_number] & ~_vessel) | (spatial["bg"] & ~changing_pixels[_number]))
        log_background = thin_plate_background(_fit["L"], _background, BACKGROUND_WAVELENGTH)
        _mean_density = log_background - _fit["L"]
        if not usable.all():
            # as in fit_background: a pixel that is not data takes the density of the nearest one that is
            nearest = distance_transform_edt(~usable, return_distances=False, return_indices=True)
            _mean_density = _mean_density[tuple(nearest)]
        from_quiet = (_background & quiet_pixels[_number]).sum() / _background.sum()
        _fit.update({"B": log_background, "OD": _mean_density, "bg": _background, "M": ~_background})
        _fit["history"].append({"name": "quiet refit", "M": _fit["M"], "bg": _background, "B": log_background,
                               "OD": _mean_density})

        density_fixations[_number] = None  # free the first measurement before the second is made
        density_fixations[_number], density_offsets[_number] = measure_density(_number, _fit["B"], _fit["bg"])
        density_means[_number] = temporal_mean(density_fixations[_number])
        _offsets = density_offsets[_number]
        print(f"Fixation {_number}: background refitted on {_background.mean() * 100:.0f}% of the pixels, "
              f"{from_quiet * 100:.0f}% of them quiet pixels (the guard kept {guarded * 100:.0f}% of the quiet pixels out); "
              f"density of the mean on them "
              f"{robust_sd(_fit['OD'][_fit['bg']]):.4f} (robust sd) | frame offsets {_offsets.min():+.4f} to {_offsets.max():+.4f}")
    final_background_fits, final_density_fixations = background_fits, density_fixations
    final_density_offsets, final_density_means = density_offsets, density_means
    return (
        final_background_fits,
        final_density_fixations,
        final_density_means,
        final_density_offsets,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A check of each fit. From left to right: the fixation's mean frame, the fitted background on the same gray scale, the density of the mean frame, and the pixels kept out of the fit (white). The background should be smooth and free of vessels, and the density flat between the vessels. The pixels kept out are everything that changes, with a margin. They are not a vessel segmentation.
    """)
    return


@app.cell
def _(final_background_fits, fixation_means, fixations, np, plt):
    DENSITY_RANGE = (-0.1, 0.5)  # optical density shown from black to white, here and in the preview below

    # show each fit: the fixation's mean, the fitted background, the density of the mean and the pixels kept out of the fit
    _figure, _axes = plt.subplots(len(fixations), 4, figsize=(16, 2.9 * len(fixations)), squeeze=False)
    for _number, (_fit, _mean, _row) in enumerate(zip(final_background_fits, fixation_means, _axes)):
        brightness = np.nanpercentile(_mean[::4, ::4], [0.5, 99.5])
        _panels = (
            (f"fixation {_number}: mean frame", _mean, brightness),
            ("fitted background, exp(B)", np.exp(_fit["B"]), brightness),
            ("density of the mean, B - ln(mean)", _fit["OD"], DENSITY_RANGE),
            ("kept out of the fit", ~_fit["bg"], (0, 1)),
        )
        for _axis, (_title, _image, (_low, _high)) in zip(_row, _panels):
            _axis.imshow(_image, cmap="gray", vmin=_low, vmax=_high)
            _axis.set_title(_title)
            _axis.axis("off")
    _figure.tight_layout()
    plt.show()
    return (DENSITY_RANGE,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The density burst, in the anchor's frame. The vessels are now bright on a flat, dark background. Black to white is `DENSITY_RANGE`; pixels without data show as 0, the background level. `preview_tile` prepares one frame at preview size, and §13 uses it again.
    """)
    return


@app.cell
def _(
    DENSITY_RANGE,
    PREVIEW_WIDTH,
    SHOW_PREVIEWS,
    anchor,
    anchor_to_reference,
    create_animation,
    cv2,
    final_density_fixations,
    fixations,
    fps,
    frame_height,
    frame_width,
    np,
    num_frames,
):
    preview_height = round(PREVIEW_WIDTH * frame_height / frame_width)


    def preview_tile(frame, number, low, high):
        """Return a frame of fixation `number` in the anchor's frame at preview size, 8-bit with low..high as black..white."""
        if number != anchor:
            frame = cv2.warpPerspective(
                frame, anchor_to_reference(fixations[number]), (frame_width, frame_height),
                flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderValue=np.nan,
            )
        frame = np.clip(np.nan_to_num(frame), low, high)  # pixels without data show as 0
        small = cv2.resize(frame, (PREVIEW_WIDTH, preview_height), interpolation=cv2.INTER_AREA)
        return np.uint8((small - low) * (255 / (high - low)) + 0.5)


    if SHOW_PREVIEWS:
        _preview_frames = np.empty((num_frames, preview_height, PREVIEW_WIDTH), dtype=np.uint8)
        for _number, (_fixation, _density) in enumerate(zip(fixations, final_density_fixations)):
            for _index, _frame in zip(_fixation["frames"], _density):
                _preview_frames[_index] = preview_tile(_frame, _number, *DENSITY_RANGE)
        create_animation(_preview_frames, "density_burst", fps=fps)
        del _preview_frames
    return preview_height, preview_tile


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 13 · Follow the shading from frame to frame

    The **temporal burst** is each frame's density minus its fixation's mean density, `final_density_means` (§12). A fixation's background $B$ is the same in every frame, so it drops out of this difference. That also means it no longer matters which pixels the background was fitted on: fitted on the quiet pixels (§12) or without any temporal data (§10), the two backgrounds differ by one constant per frame. The cell prints how large that constant is.

    What is left in the temporal burst are large soft shadows that drift across the picture. They are not vessels. The illumination is fixed to the camera, and within a fixation the eye still moves by tens of pixels, so the lighting pattern slides over the tissue from frame to frame. One background per fixation cannot follow it.

    **The slow movement.** Finer than the shading, patches of the field drift by 1 to 2% over a second or two (§11). `slow_movement` is each pixel's running mean of the temporal burst over `SLOW_WINDOW` frames, and `fast_frame` takes it out, keeping only what changes within a few frames: the blood. It is a guard for looking, as §11's is for judging; `final_density_fixations` itself is unchanged.

    **A background for every frame removes the shadows.** Each frame's density is brought to 0 on the background pixels around each pixel, not only over the frame as a whole. The local mean of the frame's density over the background pixels, `SHADING_SIGMA` pixels wide, is the frame's **shading**. Under a vessel it is bridged from the background on either side, so what happens inside the vessel is left alone. No temporal data is used here: the background and its pixels are those `fit_background` found from the mean frame alone, before the quiet refit, and each frame's shading is measured in that frame by itself.

    | Output | Content |
    |---|---|
    | `density_noise`, `density_limit` | the noise of the temporal burst, and the ± range (`DISPLAY_SIGMAS` times the noise) that the previews show |
    | `frame_shadings` | per fixation, `(frames, rows / SHADING_SCALE, columns / SHADING_SCALE)`: what each frame's shading adds to the fixation's average shading, on a coarse grid. `enlarge` brings one to full size. |
    | `frame_centres` | per fixation, `(frames,)`: the constant that goes with each frame's shading |
    | `temporal_frame(n, i)` | frame `i` of fixation `n`: its density with the moving shading taken out. Positive in vessels, about 0 on background. `final_density_fixations` itself is unchanged. |
    | `slow_movement(n, i)`, `fast_frame(n, i)` | the running mean of the temporal burst (`temporal_frame` minus `final_density_means`) over `SLOW_WINDOW` frames around frame `i`, §11's slow movement per pixel; and `temporal_frame` with it taken out: what moves from one frame to the next |
    """)
    return


@app.cell
def _(
    SLOW_WINDOW,
    ThreadPoolExecutor,
    anchor,
    changing_pixels,
    cv2,
    final_background_fits,
    final_density_fixations,
    final_density_means,
    fixations,
    frame_height,
    frame_width,
    masked_mean,
    np,
    quiet_pixels,
    workers,
):
    DISPLAY_SIGMAS = 4.0  # the previews show this many noise standard deviations either side of 0

    density_noise = 1.4826 * np.nanmedian(
        np.abs(final_density_fixations[anchor][::8, ::8, ::8] - final_density_means[anchor][::8, ::8])
    )
    density_limit = DISPLAY_SIGMAS * density_noise
    print(f"The temporal burst has a noise of {density_noise:.4f} optical density; "
          f"the previews show -{density_limit:.3f} to +{density_limit:.3f}")

    SHADING_SIGMA = 32  # pixels; shading wider than about this is followed from frame to frame
    SHADING_SCALE = 8   # the shading is smooth, so it is worked out and kept on a grid this many times coarser


    def frame_shading(density, background):
        """Return one density frame's local mean over its background pixels, on the coarse grid: that frame's shading."""
        weight = (background & np.isfinite(density)).astype(np.float32)
        size = (frame_width // SHADING_SCALE, frame_height // SHADING_SCALE)
        values = cv2.resize(np.nan_to_num(density) * weight, size, interpolation=cv2.INTER_AREA)
        weight = cv2.resize(weight, size, interpolation=cv2.INTER_AREA)  # share of background pixels in each block
        return masked_mean(values / np.maximum(weight, 1e-6), weight, SHADING_SIGMA / SHADING_SCALE)


    def enlarge(shading):
        """Bring a shading from the coarse grid to full size."""
        return cv2.resize(shading, (frame_width, frame_height), interpolation=cv2.INTER_CUBIC)


    def spatial_frame(number, position):
        """Return what centres one frame on the background fitted without temporal data, and the frame's shading."""
        fit = final_background_fits[number]
        spatial = fit["history"][-2]  # the last fit made from the mean frame alone, before the quiet refit
        density = final_density_fixations[number][position] + (spatial["B"] - fit["B"])
        centre = np.median(density[spatial["bg"] & np.isfinite(density)])  # centred on 0 over its background, as in §12
        return centre, frame_shading(density - centre, spatial["bg"])


    def temporal_frame(number, position):
        """Return one frame of a fixation with the shading that moves from frame to frame taken out: its density,
        positive in vessels, about 0 on background, NaN where the frame has no data."""
        shading = enlarge(frame_shadings[number][position]) + np.float32(frame_centres[number][position])
        return final_density_fixations[number][position] - shading


    def slow_movement(number, position):
        """Return each pixel's slow movement at frame `position`: the running mean of the temporal burst (`temporal_frame`
        minus the fixation's mean density) over SLOW_WINDOW frames centred on the frame, fewer at the ends of the
        fixation. About 0 wherever a pixel only flickers; the slow wander of the patches between the vessels."""
        count = len(fixations[number]["frames"])
        first = max(0, min(position - SLOW_WINDOW // 2, count - SLOW_WINDOW))
        window = np.stack([temporal_frame(number, p) for p in range(first, min(first + SLOW_WINDOW, count))])
        with np.errstate(invalid="ignore"):
            return np.nanmean(window, axis=0) - final_density_means[number]


    def fast_frame(number, position):
        """Return `temporal_frame` with its slow movement taken out: the frame's density, with only the changes that come
        and go within a few frames left in it."""
        frame = temporal_frame(number, position)
        return frame - slow_movement(number, position) if SLOW_WINDOW >= 2 else frame


    frame_shadings = []  # per fixation: (frames, rows / 8, cols / 8), each frame's shading minus the fixation's average
    frame_centres = []   # per fixation: (frames,), the constant that goes with each frame's shading
    largest_step = 0.0
    for _number, _fixation in enumerate(fixations):
        with ThreadPoolExecutor(max_workers=workers) as _pool:
            results = list(_pool.map(lambda position: spatial_frame(_number, position), range(len(_fixation["frames"]))))
        centres = np.array([centre for centre, _ in results])
        shadings = np.array([shading for _, shading in results])
        shadings -= shadings.mean(axis=0)
        frame_shadings.append(shadings)
        centres -= centres.mean()  # over all frames; at the rim, where some frames have no data, off by under a gray level
        frame_centres.append(centres)
        largest_step = max(largest_step, np.abs(centres).max())

    _moving = np.sqrt(np.mean(np.concatenate([shadings.ravel() for shadings in frame_shadings]) ** 2))
    print(f"Without the temporal data, a frame differs from §12's by one constant: at most {largest_step:.4f} "
          f"of density, against a gray scale of -{density_limit:.3f} to +{density_limit:.3f}")
    print(f"Shading that moves from frame to frame: {_moving:.4f} of density (rms), about {_moving * 100:.1f}% of the "
          f"brightness")

    # how much the slow movement takes out of the temporal burst, on the anchor's quiet and changing pixels
    _sample = range(SLOW_WINDOW // 2, len(fixations[anchor]["frames"]) - SLOW_WINDOW // 2, 8)
    slow_quiet, slow_changing, all_quiet, all_changing = [], [], [], []
    for _position in _sample:
        burst = temporal_frame(anchor, _position) - final_density_means[anchor]  # the temporal burst, as the preview shows it
        slow = slow_movement(anchor, _position)
        for mask, slow_part, whole in ((quiet_pixels[anchor], slow_quiet, all_quiet),
                                       (changing_pixels[anchor], slow_changing, all_changing)):
            _values = mask & np.isfinite(burst) & np.isfinite(slow)
            slow_part.append(np.mean((slow[_values] - slow[_values].mean()) ** 2))
            whole.append(np.mean((burst[_values] - burst[_values].mean()) ** 2))
    print(f"Slow movement over {SLOW_WINDOW} frames: on the quiet pixels it holds {np.mean(slow_quiet) / np.mean(all_quiet) * 100:.0f}% "
          f"of the temporal burst's variance, on the changing pixels {np.mean(slow_changing) / np.mean(all_changing) * 100:.0f}% "
          f"(sampled every 8th frame of fixation {anchor})")
    return density_limit, enlarge, fast_frame, frame_shadings, temporal_frame


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The temporal burst, negated so that dark means less light, on one gray scale of ± `density_limit`. **Left:** the background refitted to every frame (`temporal_frame`), as before. **Right:** the same with the slow movement taken out (`fast_frame`): each pixel's running mean over `SLOW_WINDOW` frames is gone, so only what moves within a few frames is left. The drifting patches between the vessels should be on the left and gone on the right, and the blood the same on both.
    """)
    return


@app.cell
def _(
    PREVIEW_WIDTH,
    SHOW_PREVIEWS,
    create_animation,
    density_limit,
    fast_frame,
    final_density_means,
    fixations,
    fps,
    np,
    num_frames,
    preview_height,
    preview_tile,
    temporal_frame,
):
    if SHOW_PREVIEWS:
        comparison_frames = np.empty((num_frames, preview_height, 2 * PREVIEW_WIDTH), dtype=np.uint8)
        for _number, _fixation in enumerate(fixations):
            for _position, _index in enumerate(_fixation["frames"]):
                # the frame minus the mean over time, negated so dark means less light
                followed = final_density_means[_number] - temporal_frame(_number, _position)
                fast = final_density_means[_number] - fast_frame(_number, _position)
                comparison_frames[_index, :, :PREVIEW_WIDTH] = preview_tile(followed, _number, -density_limit, density_limit)
                comparison_frames[_index, :, PREVIEW_WIDTH:] = preview_tile(fast, _number, -density_limit, density_limit)
        create_animation(comparison_frames, "per_frame_background_and_slow_movement_removed", fps=fps, width=2 * PREVIEW_WIDTH)
        del comparison_frames
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 14 · The smear of the temporal burst

    Three pictures per fixation, in the fixation's own reference frame:

    - **Left: the mean density.** This is what was subtracted: the static picture, in which the vessels are bright.
    - **Middle: the average of the temporal burst over time.** It is 0, because each pixel's own mean over time is exactly what was subtracted. The cell prints how far from 0 it gets; the shading leaves a trace at the rim, where some frames have no data.
    - **Right: the smear.** How far the temporal burst swings around 0 at each pixel: its standard deviation over time. Each frame is first blurred by `SMEAR_BLUR` pixels to beat the pixel noise, and the result is multiplied by the square root of the brightness, which evens out the shot noise. Vessels with flow stand out; tissue without flow stays at the noise level.

    | Output | Content |
    |---|---|
    | `burst_smears` | per fixation, `(rows, columns)`: the smear, `NaN` where fewer than `SMEAR_MIN_FRAMES` of the fixation's frames have data |
    """)
    return


@app.cell
def _(
    DENSITY_RANGE,
    anchor,
    blur_valid,
    density_limit,
    enlarge,
    final_density_fixations,
    final_density_means,
    fixation_means,
    fixations,
    frame_shadings,
    frame_shape,
    np,
    plt,
):
    SMEAR_BLUR = 2.0         # pixels; each frame is blurred by this before its fluctuation is added up
    SMEAR_MIN_FRAMES = 0.25  # a pixel needs data in this fraction of its fixation's frames to appear in the smear

    gray = plt.get_cmap("gray").copy()
    gray.set_bad("black")  # pixels without a value are drawn black

    burst_smears = []  # per fixation: the temporal burst's standard deviation over time, shot noise evened out
    burst_averages = []
    for _number, (_density, _mean_density, _changes) in enumerate(zip(final_density_fixations, final_density_means, frame_shadings)):
        summed, soft_sum, soft_squares = (np.zeros(frame_shape, dtype=np.float64) for _ in range(3))
        _seen = np.zeros(frame_shape, dtype=np.uint16)
        for _frame, _change in zip(_density, _changes):
            temporal = _mean_density - _frame + enlarge(_change)  # negated as in the videos: dark means less light
            soft = np.nan_to_num(blur_valid(temporal, SMEAR_BLUR))
            summed += np.nan_to_num(temporal)
            soft_sum += soft
            soft_squares += soft**2
            _seen += np.isfinite(temporal)
        frames_seen = np.maximum(_seen, 1)
        burst_averages.append(np.where(_seen > 0, summed / frames_seen, np.nan))
        # the spread around each pixel's own mean, times the square root of the brightness to even out the shot noise
        spread = np.sqrt(np.maximum(soft_squares / frames_seen - (soft_sum / frames_seen) ** 2, 0))
        enough = _seen >= SMEAR_MIN_FRAMES * len(_density)
        burst_smears.append(
            np.where(enough, spread * np.sqrt(np.nan_to_num(fixation_means[_number])), np.nan).astype(np.float32)
        )
        print(f"Fixation {_number}: {len(_density)} frames | the temporal burst averages to 0 within "
              f"{np.nanmax(np.abs(burst_averages[_number])):.0e} of density")

    smear_range = np.nanpercentile(burst_smears[anchor], [1, 99.7])  # one gray scale for the smears of all fixations
    _figure, _axes = plt.subplots(len(fixations), 3, figsize=(16, 3.8 * len(fixations)), squeeze=False)
    for _number, _row in enumerate(_axes):
        _shown = (
            (f"fixation {_number}: mean density, which was subtracted", final_density_means[_number], DENSITY_RANGE),
            ("average of the temporal burst over time: 0", burst_averages[_number], (-density_limit, density_limit)),
            ("smear of the temporal burst: its fluctuation over time", burst_smears[_number], smear_range),
        )
        for _axis, (_title, _image, (_low, _high)) in zip(_row, _shown):
            _axis.imshow(_image, cmap=gray, vmin=_low, vmax=_high)
            _axis.set_title(_title, fontsize=10)
            _axis.axis("off")
    _figure.tight_layout()
    plt.show()
    del burst_averages
    return burst_smears, gray


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 15 · Clean the smear's background

    The smear has a background of its own. On tissue without flow it sits at the noise floor, not at 0, and that level is not perfectly even. It is taken out with the same fit that cleans the average frame, `fit_background`.

    That fit expects dark vessels on a bright background, in log space, so it is given $\exp(-\text{smear} / \text{noise floor})$, whose log is minus the smear in multiples of the noise floor. The density it returns is then

    $$
    \text{cleaned smear} \;=\; \frac{\text{smear} - \text{fitted background of the smear}}{\text{noise floor}}:
    $$

    0 where nothing flows and, on a vessel, the fluctuation above the background in multiples of the noise floor. The smear's background is a local mean, `SMEAR_SIGMA` pixels wide, in every refit: `fit_background` is called with `wavelength=None`, so it fits no thin plate here, unlike §10 with its `BACKGROUND_WAVELENGTH` of 30 pixels. The noise floor is nearly flat, and a fit that follows the picture that closely creeps up the widest vessels and hollows them out.

    The cleaned smear is compared with the **background-cleaned average frame**, `final_background_fits[n]["OD"]`. The average frame shows how much light a place takes away on average; the smear shows how much that changes from frame to frame. A pixel *stands out* in either picture when it is more than `STANDS_OUT` times that picture's background noise above 0.

    Two figures: **the cleaning** per fixation (the smear, its fitted background, what is left, and the pixels kept out of the fit), and **the comparison** per fixation. In the comparison's last panel the average frame is magenta and the smear green: white is a place both mark, magenta one that takes light away without changing, green one that only changes.

    | Output | Content |
    |---|---|
    | `smear_fits` | per fixation, the dictionary from `fit_background`, fitted on the smear |
    | `cleaned_smears` | per fixation, `(rows, columns)`: the smear with its background removed, in multiples of its noise floor (`NaN` where the smear has no value) |
    """)
    return


@app.cell
def _(
    DENSITY_RANGE,
    anchor,
    burst_smears,
    final_background_fits,
    fit_background,
    fixations,
    gray,
    np,
    plt,
    robust_sd,
):
    SMEAR_SIGMA = 64  # pixels; width of the local mean that is the smear's background. The noise floor is nearly flat,
                      # and a narrower one (8, as for the average frame) creeps up the widest vessels and hollows them out
    STANDS_OUT = 5    # a pixel stands out when it is this many times its picture's background noise above 0

    smear_fits = []      # per fixation: the dict from fit_background, fitted on the smear
    cleaned_smears = []  # per fixation: the smear minus its fitted background, in multiples of the smear's noise floor
    compared = []        # per fixation: what the figures below show
    for _number, smear in enumerate(burst_smears):
        # the noise floor is where the pixels without flow sit: the peak of the smear's histogram
        _values = smear[np.isfinite(smear)]
        _histogram, _edges = np.histogram(_values, bins=400, range=(0, np.percentile(_values, 99)))
        _floor = _edges[np.argmax(_histogram):][:2].mean()
        # exp(-smear / floor) is bright where the smear is low and dark on the vessels; its log is -smear / floor
        _fit = fit_background(np.exp(-smear / _floor), sigma=SMEAR_SIGMA, wavelength=None)
        _seen = np.isfinite(smear)
        _cleaned = np.where(_seen, _fit["OD"], np.nan)
        smear_fits.append(_fit)
        cleaned_smears.append(_cleaned)

        _average_fit = final_background_fits[_number]
        _average = np.where(_seen, _average_fit["OD"], np.nan)  # the background-cleaned average frame, where the smear has data
        average_noise = robust_sd(_average[_average_fit["bg"] & _seen])
        smear_noise = robust_sd(_cleaned[_fit["bg"] & _seen])
        dark = _seen & (_average > STANDS_OUT * average_noise)   # takes light away on average
        _moving = _seen & (_cleaned > STANDS_OUT * smear_noise)   # changes from frame to frame
        compared.append((smear / _floor, -_fit["B"], _cleaned, ~_fit["bg"], _average))
        print(f"Fixation {_number}: background of the smear fitted on {_fit['bg'].mean() * 100:.0f}% of the pixels; the cleaned "
              f"smear has a background noise of {smear_noise:.2f} noise floors | correlation with the background-cleaned "
              f"average frame {np.corrcoef(_average[_seen], _cleaned[_seen])[0, 1]:.2f}")
        print(f"   stands out ({STANDS_OUT} x its own background noise): in the average frame on {dark.sum() / _seen.sum() * 100:.0f}% "
              f"of the field, in the smear on {_moving.sum() / _seen.sum() * 100:.0f}%, in both on "
              f"{(dark & _moving).sum() / _seen.sum() * 100:.0f}% | where both do, the average frame is "
              f"{np.median(_average[dark & _moving]) / average_noise:.0f} x and the smear "
              f"{np.median(_cleaned[dark & _moving]) / smear_noise:.0f} x its noise (median)")

    # show the cleaning of each smear: the smear, its fitted background, what is left, and the pixels kept out of the fit
    smear_top = np.nanpercentile(compared[anchor][0], 99.7)
    _figure, _axes = plt.subplots(len(fixations), 4, figsize=(16, 2.9 * len(fixations)), squeeze=False)
    for _number, (_row, (smear, _background, _cleaned, kept_out, _average)) in enumerate(zip(_axes, compared)):
        _shown = (
            (f"fixation {_number}: smear, in noise floors", smear, (0, smear_top)),
            ("fitted background of the smear", _background, (0, smear_top)),
            ("smear with its background removed", _cleaned, (-1, smear_top - 1)),
            ("kept out of the fit", kept_out, (0, 1)),
        )
        for _axis, (_title, _image, (_low, _high)) in zip(_row, _shown):
            _axis.imshow(_image, cmap=gray, vmin=_low, vmax=_high)
            _axis.set_title(_title, fontsize=10)
            _axis.axis("off")
    _figure.tight_layout()
    plt.show()

    # show the comparison: the background-cleaned average frame, the cleaned smear, and both in one picture.
    # In the last one the average frame is magenta and the smear green, each up to its 99.5th percentile, so a place
    # that both mark comes out white, one that takes light away without changing magenta, one that only changes green.
    _figure, _axes = plt.subplots(len(fixations), 3, figsize=(16, 3.8 * len(fixations)), squeeze=False)
    for _number, (_row, (smear, _background, _cleaned, kept_out, _average)) in enumerate(zip(_axes, compared)):
        average_share = np.clip(np.nan_to_num(_average) / np.nanpercentile(_average, 99.5), 0, 1)
        smear_share = np.clip(np.nan_to_num(_cleaned) / np.nanpercentile(_cleaned, 99.5), 0, 1)
        _shown = (
            (f"fixation {_number}: background-cleaned average frame", _average, DENSITY_RANGE),
            ("smear with its background removed", _cleaned, (-1, smear_top - 1)),
            ("magenta: average frame, green: smear, white: both", np.dstack([average_share, smear_share, average_share]), (0, 1)),
        )
        for _axis, (_title, _image, (_low, _high)) in zip(_row, _shown):
            _axis.imshow(_image, cmap=gray, vmin=_low, vmax=_high)
            _axis.set_title(_title, fontsize=10)
            _axis.axis("off")
    _figure.tight_layout()
    plt.show()
    del compared
    return STANDS_OUT, cleaned_smears, smear_fits


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 16 · Combine the two vessel pictures

    The background-cleaned average frame and the cleaned smear, put together for each fixation. The two are in different units (optical density, noise floors), so each is first divided by its own background noise; a value then says how many times the noise a pixel stands above 0.

    - **The sum** adds the two. A vessel that either picture shows is in it, and one that both show is twice as strong.
    - **The average** is half the sum: the same picture on half the scale.

    All four panels share one gray scale, that of the two pictures on their own, so the average can be read against either of them directly. Each panel is also saved at full resolution in `vessel_pictures/`.

    | Output | Content |
    |---|---|
    | `combined_sums` | per fixation, `(rows, columns)`: the sum, in multiples of one picture's background noise (`NaN` where the smear has no value). The average is `combined_sums[n] / 2`. |
    """)
    return


@app.cell
def _(
    Image,
    STANDS_OUT,
    anchor,
    cleaned_smears,
    final_background_fits,
    fixations,
    gray,
    np,
    os,
    plt,
    robust_sd,
    smear_fits,
):
    combined_sums = []  # per fixation: average frame / its background noise + cleaned smear / its background noise
    shown_together = []
    for _number, _cleaned in enumerate(cleaned_smears):
        _seen = np.isfinite(_cleaned)
        _average_fit, smear_fit = final_background_fits[_number], smear_fits[_number]
        in_average = np.where(_seen, _average_fit["OD"], np.nan) / robust_sd(_average_fit["OD"][_average_fit["bg"] & _seen])
        in_smear = _cleaned / robust_sd(_cleaned[smear_fit["bg"] & _seen])
        combined_sums.append(in_average + in_smear)
        shown_together.append((in_average, in_smear, combined_sums[_number], combined_sums[_number] / 2))

        # where both pictures see background the sum is noise alone; that is the level the sum is judged against
        sum_noise = robust_sd(combined_sums[_number][_average_fit["bg"] & smear_fit["bg"] & _seen])
        _share = lambda marked: (marked & _seen).sum() / _seen.sum() * 100
        print(f"Fixation {_number}: stands out ({STANDS_OUT} x the background noise) in the sum on "
              f"{_share(combined_sums[_number] > STANDS_OUT * sum_noise):.0f}% of the field | in the average frame alone on "
              f"{_share(in_average > STANDS_OUT):.0f}%, in the smear alone on {_share(in_smear > STANDS_OUT):.0f}%, "
              f"in either on {_share((in_average > STANDS_OUT) | (in_smear > STANDS_OUT)):.0f}% | "
              f"background noise of the sum {sum_noise:.2f} (two independent pictures would give 1.41)")

    # one gray scale for every panel: up to the 99.5th percentile of the fainter of the two pictures in the anchor,
    # so that the fine vessels show; the widest vessels of the other picture then go white
    top = min(np.nanpercentile(picture, 99.5) for picture in shown_together[anchor][:2])
    _figure, _axes = plt.subplots(len(fixations), 4, figsize=(16, 2.9 * len(fixations)), squeeze=False)
    for _number, (_row, _pictures) in enumerate(zip(_axes, shown_together)):
        titles = (f"fixation {_number}: background-cleaned average frame", "smear with its background removed",
                  "sum of the two", "average of the two")
        for _axis, _title, _image in zip(_row, titles, _pictures):
            _axis.imshow(_image, cmap=gray, vmin=0, vmax=top)
            _axis.set_title(_title, fontsize=10)
            _axis.axis("off")
    _figure.suptitle(f"each picture in multiples of its own background noise; black to white is 0 to {top:.0f}", fontsize=10)
    _figure.tight_layout()
    plt.show()

    # the figure shows the panels much reduced; save each one at full resolution too, on the same gray scale,
    # and the average also inverted (dark vessels on a light background)
    os.makedirs("vessel_pictures", exist_ok=True)
    for _number, _pictures in enumerate(shown_together):
        for _name, _picture in zip(("average_frame", "smear", "sum", "average"), _pictures):
            shade = (np.clip(np.nan_to_num(_picture) / top, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(shade).save(f"vessel_pictures/fixation_{_number}_{_name}.png")
            if _name == "average":
                Image.fromarray(255 - shade).save(f"vessel_pictures/fixation_{_number}_{_name}_inverted.png")
    print(f"Saved {5 * len(shown_together)} pictures at full resolution in the folder vessel_pictures")
    del shown_together
    return combined_sums, top


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The average of the two for fixation 1, inverted on the gray scale above, so the vessels come out dark on a light background, the way the frames themselves look.
    """)
    return


@app.cell
def _(combined_sums, fixations, np, plt, top):
    _number = min(1, len(fixations) - 1)  # fixation 1, or fixation 0 when the burst has only one
    inverted_average = 1 - np.clip(np.nan_to_num(combined_sums[_number] / 2) / top, 0, 1)  # 1 is background, 0 the top of the scale

    _figure, _axis = plt.subplots(figsize=(16, 10))
    _axis.imshow(inverted_average, cmap="gray", vmin=0, vmax=1)
    _axis.set_title(f"fixation {_number}: average of the two, inverted")
    _axis.axis("off")
    _figure.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 17 · Save the result

    What vessel annotation needs from this notebook goes into one file, `preprocessed/<burst folder>.npz` in the working directory, so that an annotation notebook can start from it without running this one. The per-fixation pictures are stacked along a first axis of length *fixations*; each lies in its own fixation's reference frame, and `to_anchor[n]` maps the anchor's frame onto fixation `n`'s.

    | Key | Shape · type | Content |
    |---|---|---|
    | `burst`, `fps`, `anchor` | scalars | the burst's folder, its frame rate, the anchor fixation's number |
    | `to_anchor` | `(fixations, 3, 3)` | the map from the anchor's `(x, y)` to each fixation's (§6) |
    | `densities` | `(fixations, rows, columns)` · `float32` | **the still picture**: the optical density of the mean frame, $B - \ln(\text{mean})$, vessels bright on 0 (§12) |
    | `usable_pixels`, `background_pixels` | same · `bool` | the pixels with data, and those the background was fitted on (§12) |
    | `quiet_pixels` | same · `bool` | what does not change over time (§11) |
    | `cleaned_smears` | same · `float32` | the fluctuation over time with its background removed (§14, §15) |
    | `combined_sums` | same · `float32` | the still picture and the cleaned smear together, in multiples of the background noise (§16) |

    `NaN` marks where the smear has no value, in `cleaned_smears` and `combined_sums`. `densities` has no `NaN`: a pixel without data is filled from the nearest usable one, so mask it with `usable_pixels`. `combined_sums` contains that filled density wherever the smear has a value, so its data area is `np.isfinite(combined_sums) & usable_pixels`. To read the result elsewhere:

    ```python
    with np.load("preprocessed/burst_2026-09-16_15-50-52.npz") as saved:
        anchor = int(saved["anchor"])
        still_picture = np.where(saved["usable_pixels"][anchor], saved["densities"][anchor], np.nan)
    ```
    """)
    return


@app.cell
def _(
    Path,
    anchor,
    anchor_to_reference,
    cleaned_smears,
    combined_sums,
    final_background_fits,
    fixations,
    fps,
    np,
    quiet_pixels,
    target_folder,
):
    SAVE_DIR = Path("preprocessed")  # in the working directory

    def _stacked(pictures, dtype=np.float32):
        return np.stack([np.asarray(picture, dtype=dtype) for picture in pictures])

    _result = {
        "burst": str(target_folder),
        "fps": float(fps),
        "anchor": anchor,
        "to_anchor": np.stack([anchor_to_reference(_fixation) for _fixation in fixations]),
        "densities": _stacked(_fit["OD"] for _fit in final_background_fits),
        "usable_pixels": _stacked((_fit["ok"] for _fit in final_background_fits), bool),
        "background_pixels": _stacked((_fit["bg"] for _fit in final_background_fits), bool),
        "quiet_pixels": _stacked(quiet_pixels, bool),
        "cleaned_smears": _stacked(cleaned_smears),
        "combined_sums": _stacked(combined_sums),
    }

    SAVE_DIR.mkdir(exist_ok=True)
    _saved_path = SAVE_DIR / f"{target_folder.name}.npz"
    _partial = SAVE_DIR / f"{target_folder.name}.partial.npz"  # renamed once complete, so a broken save is never read
    np.savez_compressed(_partial, **_result)
    _partial.replace(_saved_path)
    print(f"Saved {len(_result)} arrays to {_saved_path.resolve()} ({_saved_path.stat().st_size / 1e6:.0f} MB)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What is in memory.** All of the notebook's names stay available to cells added below. The main ones:

    | Stage | Variables |
    |---|---|
    | The recording | `raw_burst`, `fps`, `gated_frames`, `gated_indices`, `frame_times` |
    | The motion | `step_shifts`, `frame_positions`, `fixation_members`, `fixations`, `anchor` |
    | Frames, per fixation | `raw_fixations` → `stabilized_fixations` → `final_density_fixations` |
    | Frames, whole burst | `stabilized_frames`, in the anchor's frame |
    | Pictures, per fixation | `fixation_means`, `final_density_means`, `final_background_fits[n]["OD"]`, `burst_smears`, `cleaned_smears`, `combined_sums` |
    | Masks, per fixation | `quiet_pixels`, `changing_pixels`, `final_background_fits[n]["bg"]`, `final_background_fits[n]["ok"]` |
    | Functions | `phase_shift`, `frame_spectrum`, `clean_frame`, `anchor_to_reference`, `frame_area`, `fit_background`, `measure_density`, `temporal_mean`, `masked_mean`, `robust_sd`, `blur`, `blur_valid`, `enlarge`, `temporal_frame`, `fast_frame`, `create_animation` |

    Memory is the tight resource: every full stack of `float32` frames is about 1.3 GB, so work on frames or pairs of frames as you go rather than building another stack. The cell below lists the arrays with their shapes, types and sizes in memory.
    """)
    return


@app.cell
def _(
    aligned_fixation_averages,
    burst_smears,
    changing_pixels,
    cleaned_smears,
    combined_sums,
    final_density_fixations,
    final_density_means,
    fixation_means,
    fluctuation_ratio,
    frame_shadings,
    gated_frames,
    quiet_pixels,
    raw_burst,
    raw_fixations,
    sensor_map,
    stabilized_fixations,
    stabilized_frames,
):
    OUTPUTS = {"raw_burst": raw_burst, "gated_frames": gated_frames, "raw_fixations": raw_fixations,
               "stabilized_frames": stabilized_frames, "stabilized_fixations": stabilized_fixations,
               "final_density_fixations": final_density_fixations, "aligned_fixation_averages": aligned_fixation_averages,
               "fixation_means": fixation_means, "final_density_means": final_density_means, "burst_smears": burst_smears,
               "cleaned_smears": cleaned_smears, "combined_sums": combined_sums, "quiet_pixels": quiet_pixels,
               "changing_pixels": changing_pixels, "fluctuation_ratio": fluctuation_ratio, "sensor_map": sensor_map,
               "frame_shadings": frame_shadings}

    held = 0
    print(f"{'name':28s} {'shape':40s} {'dtype':8s} {'MB':>7s}")
    for _name, value in OUTPUTS.items():
        arrays = value if isinstance(value, list) else [value]
        _shapes = ", ".join(str(array.shape) for array in arrays)
        views = sum(array.base is not None for array in arrays)  # a view shows memory that another output holds
        size = sum(array.nbytes for array in arrays if array.base is None) / 1e6
        held += size
        note = f"  + {views} view{'s' if views > 1 else ''} into another output" if views else ""
        print(f"{_name:28s} {_shapes:40s} {str(arrays[0].dtype):8s} {size:7.0f}{note}")
    print(f"{'':78s} {held:7.0f} MB in all")
    return



if __name__ == "__main__":
    app.run()
