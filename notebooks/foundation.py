import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Karim's attempt: the burst, held still and cleaned

    This notebook runs everything that comes **before the neural fields**. It reads one burst, holds the vessels still, cleans the frames, and makes the vessel pictures that the neural fields read. It stops where section 18 of [`notebooks/development copy 2.ipynb`](notebooks/development%20copy%202.ipynb) begins; the neural fields are yours to write at the bottom.

    The computation follows the development notebook's sections 1–17, with the same functions and the same settings, and with one change of order, described below. The layout is new. There is one section per step, each fixation's frames are kept as a stack of their own (`raw_fixations`, `stabilized_fixations`, …), static check figures stand in for most of the videos, and the remaining previews are small.

    ### The pipeline at a glance

    | § | Step | Main outputs | Dev § |
    |---|---|---|---|
    | 1 | Load the burst | `raw_burst`, `fps` | 1 |
    | 2 | Drop the blurred frames | `gated_frames`, `gated_indices`, `frame_times` | 3 |
    | 3 | Measure the motion between frames | `step_shifts`, `frame_positions` | 4 |
    | 4 | Split the burst into fixations | `fixation_members`, **`raw_fixations`** | 5 |
    | 5 | Align the frames of each fixation | `fixations` | 6 |
    | 6 | Align the fixations to each other | `anchor`, `aligned_fixation_averages` | 7 |
    | 7 | Remove dust and glare | `sensor_map`, `glare_pixels`, `clean_frame()` | 8 |
    | 8 | Stabilize every frame | **`stabilized_fixations`**, `stabilized_frames` | 9 |
    | 9 | The mean frame of each fixation | `fixation_means` | 10 |
    | 10 | Fit the background, measure the density | `background_fits`, `density_fixations`, `density_means` | 12 |
    | 11 | Find the quiet pixels | `fluctuation_ratio`, `quiet_pixels`, `changing_pixels` | 12 |
    | 12 | Refit the background on the quiet pixels | `final_background_fits`, **`final_density_fixations`** | 12 |
    | 13 | Follow the shading from frame to frame | `frame_shadings`, `temporal_frame()` | 13–14 |
    | 14 | The smear of the temporal burst | `burst_smears` | 15 |
    | 15 | Clean the smear's background | `smear_fits`, `cleaned_smears` | 16 |
    | 16 | Combine the two vessel pictures | **`combined_sums`** | 17 |

    Ingestion and stabilization proper are §1–§8. §9–§16 turn the stabilized stacks into the pictures that the neural fields read.

    ### The background comes first

    The development notebook subtracts each fixation's temporal mean in DN (its §10), then evens out the flicker (§11), and only then fits the background (§12). Here the background is fitted right after stabilization, before any temporal mean is subtracted. Flicker multiplies a whole frame by one factor, which is a constant in log space, and centring each frame's density on its background pixels removes it (§10 shows the check). So there is no flicker step, and the temporal mean is subtracted in density instead (§11, §13).

    ### Conventions

    - **Fixation**: a run of frames between two saccades, in which the eye only jitters. Per-fixation outputs are lists indexed by the fixation's number `n`.
    - **Reference frame**: the one frame of a fixation that its other frames are aligned to. Every per-fixation stack and picture lies in its own fixation's reference frame.
    - **Anchor**: the fixation with the most frames (`anchor`). Its reference frame is the common frame of the whole burst; `stabilized_frames` and the previews use it.
    - **Frame numbers** are positions in `gated_frames`, not in `raw_burst`. `gated_indices[i]` is frame `i`'s number in the recording.
    - **Shifts** are `(dy, dx)` in pixels, rows first. **Maps** such as `fixations[n]["to_anchor"]` act on `(x, y)`, as OpenCV's do.
    - **No data** is `0` in the integer stacks and `NaN` in the float ones.
    - **DN** (digital number) is the camera's raw unit. **Optical density** is $B - \ln I$: how far a pixel's log brightness lies below the fitted background $B$. It is about 0 on background and positive in vessels.
    - **Quiet pixel**: a pixel whose brightness changes over time no more than the camera noise does, i.e. background.

    Run the cells from top to bottom, on a local kernel or a Colab one (see §0). Locally the whole notebook takes about three minutes once Google Drive has the TIFFs cached. On Colab, reading them through Drive the first time adds a few minutes, and a free server's two cores make the rest slower too.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 0 · Setup

    Imports, the notebook's few settings, and `create_animation`, which plays any `(frames, rows, columns)` stack as a looping preview.

    **On a Colab kernel**, the Colab server cannot see this computer's disk. The cell therefore mounts Google Drive (the first time it asks you to sign in) and works in this notebook's folder there, `/content/drive/MyDrive/segmentation/attempt5000`. Everything below then runs as it does locally. If the notebook lives somewhere else on your Drive, change `COLAB_NOTEBOOK_FOLDER`.

    `create_animation` changes what you see, never the data. It stretches the stack's full range to 8 bits, shrinks the frames to `width` pixels across, and caches the file in `previews/` under a hash of the frames, so each stack is rendered only once. Previews are animated WebP, about 15× smaller than the same frames as a GIF, which keeps this notebook light. If your viewer won't play them, set `PREVIEW_FORMAT = "gif"`.
    """)
    return


@app.cell
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
    import scipy.interpolate
    import scipy.linalg
    import scipy.ndimage
    import scipy.sparse
    import scipy.sparse.linalg
    import scipy.spatial
    import scipy.special
    from IPython.display import HTML, Image as IPImage, Video, display
    from PIL import Image
    from scipy.ndimage import distance_transform_edt, percentile_filter
    from skimage.filters import apply_hysteresis_threshold, frangi, laplace
    from skimage.transform import downscale_local_mean

    # On a Colab kernel the local disk is out of reach, so this notebook's folder is opened through Google Drive
    # instead: the burst, the cached previews and the saved pictures are then found where they are locally.
    COLAB_NOTEBOOK_FOLDER = Path("/content/drive/MyDrive/segmentation/attempt5000")
    try:
        from google.colab import drive as colab_drive
    except ImportError:
        colab_drive = None  # a local kernel, already working in the notebook's folder
    if colab_drive is not None:
        if not COLAB_NOTEBOOK_FOLDER.exists():
            colab_drive.mount("/content/drive")
        os.chdir(COLAB_NOTEBOOK_FOLDER)
    print(f"Working in {Path.cwd()}")

    SHOW_PREVIEWS = True            # False skips every animated preview; the check figures are still drawn
    PREVIEW_FORMAT = "webp"         # "webp" (small), "gif" (plays everywhere, ~15x larger) or "mp4" (needs ffmpeg)
    PREVIEW_WIDTH = 480             # pixels; previews are rendered this wide
    PREVIEW_DIR = Path("previews")  # where the previews are cached, under a hash of their content
    # threads for the per-frame work: half the cores on this machine, all of them on a Colab server, which runs nothing else
    workers = (os.cpu_count() or 2) if colab_drive is not None else max(1, (os.cpu_count() or 2) // 2)

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
        else:
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

        if ext == ".mp4":
            display(Video(str(path), embed=embed_html5, width=640))
        elif ext == ".webp":
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            display(HTML(f'<img src="data:image/webp;base64,{encoded}" width="{width}">'))
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
        hashlib,
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

    The burst is the first `burst_*` folder found next to this notebook or in one of its parent folders. Its TIFF frames are read in name order into one array and left untouched: these are the camera's own 12-bit values. The frame rate comes from the recorder's `manifest.json`. It gives the frame times and plays the previews at the speed of the recording. To load a particular burst instead, set the environment variable `LIMBUS_BURST` to its folder before starting the notebook.

    | Output | Shape · type | Content |
    |---|---|---|
    | `raw_burst` | `(frames, rows, columns)` · `uint16` | every recorded frame, as recorded |
    | `fps` | `float` | frames per second of the recording |
    | `target_folder` | `Path` | the burst's folder |
    """)
    return


@app.cell
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
    return fps, raw_burst


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
    return frame_times, gated_frames, sharpness, vesselness


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


@app.cell
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

    The functions below are those of `background.py`, unchanged; their docstrings give the details. They mention functions of that pipeline (`register_burst`, `stabilize_burst`, `vessels.vessel_seeds`) that this notebook does not have.
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
    # background estimation in log space, following step 1 of the pipeline report
    # (pipeline_report.html): a first guess of the background from an upper
    # envelope, and the first guess of which pixels are background
    #
    #   L   = ln I, the log image
    #   B0  = upper envelope of L: L with every dark line narrower than the
    #         closing disc filled in
    #   OD0 = B0 - L, the first optical density: ~0 on background, positive in
    #         vessels (overlapping vessels add)
    #
    # then the refit (fit_background): a vessel mask from the contrast-to-noise
    # map (the report's step 2) takes the vessels out, and B is fitted on the
    # background pixels alone. frame_density applies a fitted B to single frames.


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
        segmentation; see vessels.vessel_seeds for that), bg (background
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


    def frame_density(frame, background, transform=None, background_pixels=None,
                      saturation=4095):
        """
        optical density of a single frame against a background fitted on the
        stabilized average. the background is warped into the frame with the
        frame's own registration transform (the inverse of what stabilized it),
        so every frame gets exactly the same subtraction. if background_pixels
        is given, the frame's median density on those pixels is subtracted, so a
        frame exposed a little differently from the average still centres on zero

        input: 2d frame (raw, in its own coordinates), B from fit_background
        (reference coordinates), the frame's (2, 3) transform from register_burst
        (None if the frame is already in reference coordinates), optional
        background-pixel mask in reference coordinates, saturation level

        output: OD of the frame in its own coordinates (nan where the background
        doesn't cover or the frame isn't data), the offset that was subtracted
        """
        frame = np.asarray(frame, dtype=np.float32)
        rows, cols = frame.shape
        ok = np.isfinite(frame) & (frame > 0) & (frame < saturation)
        L = np.log(np.where(ok, frame, 1))

        if transform is None:
            B, pixels = background, background_pixels
        else:
            # transforms map reference -> frame, which is the forward map here
            B = cv2.warpAffine(background.astype(np.float32), transform, (cols, rows),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                               borderValue=np.nan)
            pixels = None
            if background_pixels is not None:
                pixels = cv2.warpAffine(background_pixels.astype(np.uint8), transform,
                                        (cols, rows), flags=cv2.INTER_NEAREST,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=0).astype(bool)

        od = np.where(ok, B - L, np.nan).astype(np.float32)
        offset = 0.0
        if pixels is not None:
            values = od[pixels & np.isfinite(od)]
            if values.size:
                offset = float(np.median(values))
                od -= offset
        return od, offset


    def burst_density(frames, background, transforms=None, background_pixels=None,
                      saturation=4095, workers=None):
        """
        optical density of every frame of a burst against one fitted background:
        frame_density applied to the whole burst, so the vessels of every frame
        are measured against the same background

        pass the stabilized frames (reference coordinates, from stabilize_burst)
        with transforms=None to get a density burst in reference coordinates,
        where the vessels sit still and the background is flat: the burst to
        watch, average, or follow over time. pass the raw frames with their
        transforms to get each frame's density in its own coordinates instead

        input: burst of shape (frames, rows, cols), B from fit_background, the
        (frames, 2, 3) transforms from register_burst or None, optional
        background-pixel mask (each frame is centred on zero over it), saturation
        level

        output: density burst as float32 (nan where a frame has no data), and the
        (frames,) offsets subtracted
        """
        density = np.empty(frames.shape, dtype=np.float32)
        offsets = np.zeros(len(frames))

        def one(i):
            transform = None if transforms is None else transforms[i]
            density[i], offsets[i] = frame_density(frames[i], background, transform,
                                                   background_pixels, saturation)

        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(one, range(len(frames))))

        return density, offsets


    def saturated_pixels(frames, transforms, saturation=4095):
        """
        reference-coordinate mask of the pixels saturated in any frame of a
        burst, for fit_background's `valid`. tested on the raw frames: the
        stabilized frames are interpolated, which overshoots near bright spots
        and would flag pixels that never saturated, and the average can't tell,
        since a pixel saturated in some frames averages below the saturation level

        input: raw burst of shape (frames, rows, cols), its (frames, 2, 3)
        transforms from register_burst, saturation level

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
    _number = min(1, len(fixations) - 1)  # fixation 1, as in the development notebook
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
    ## What you have now

    | Stage | Variables |
    |---|---|
    | The recording | `raw_burst`, `fps`, `gated_frames`, `gated_indices`, `frame_times` |
    | The motion | `step_shifts`, `frame_positions`, `fixation_members`, `fixations`, `anchor` |
    | Frames, per fixation | `raw_fixations` → `stabilized_fixations` → `final_density_fixations` |
    | Frames, whole burst | `stabilized_frames`, in the anchor's frame |
    | Pictures, per fixation | `fixation_means`, `final_density_means`, `final_background_fits[n]["OD"]`, `burst_smears`, `cleaned_smears`, `combined_sums` |
    | Masks, per fixation | `quiet_pixels`, `changing_pixels`, `final_background_fits[n]["bg"]`, `final_background_fits[n]["ok"]` |
    | Functions | `phase_shift`, `frame_spectrum`, `clean_frame`, `anchor_to_reference`, `frame_area`, `fit_background`, `measure_density`, `temporal_mean`, `masked_mean`, `robust_sd`, `blur`, `blur_valid`, `enlarge`, `temporal_frame`, `create_animation` |

    The cell below lists the arrays with their shapes, types and sizes in memory.
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Your neural fields start here

    The development notebook's neural fields (its §18 onward) read these, all for the anchor's fixation:

    | What | Where |
    |---|---|
    | the still picture: density of the mean frame, vessels bright on 0 | `final_background_fits[anchor]["OD"]` |
    | the data area those pictures share | `np.isfinite(cleaned_smears[anchor]) & final_background_fits[anchor]["ok"]` |
    | background pixels, to calibrate a noise level against | `quiet_pixels[anchor] & final_background_fits[anchor]["bg"]`, within the data area |
    | what flows: the change in density between two frames | `(final_density_fixations[anchor][i + 1] - final_density_fixations[anchor][i]) / np.diff(fixations[anchor]["times"])[i]` |
    | one frame with the drifting shading taken out | `temporal_frame(anchor, i)` |
    | where the tissue fluctuates | `cleaned_smears[anchor]` |
    | the two pictures together | `combined_sums[anchor]` |

    Memory is the tight resource: every full stack of `float32` frames is about 1.3 GB, so work on frames or pairs of frames as you go rather than building another stack, and `del` stacks you no longer need.
    """)
    return


@app.cell
def _():
    # compute the fixation 0 average of the background cleaned, stabilized burst

    # define an on/off center contrast neural field
    # define an orientation / edge detector neural field

    # plot these (make sure you are hashing and saving these things to disk) as heatmaps
    # the actual neural layer should have a tunable set of parameters:
    # the set of gaussian blurs performed on the image
    # the set of kernel sizes used for the convolutions
    # the set of orientations for the edge detector kernels
    # the wavlet that determines kernel values

    # when plotting, we want to see, at each pixel, the strongest responding neuron at each cell.
    # obviously the actual output will be multidimensional; some neurons will respond more strongly to certain features than others.
    # therefore, when visualizing, we may want to collapse across certain dimensions to highlight the strongest responses.

    # in general, the dimensions of the orientation output should include, at each (x,y) (pixel coordinates), the strongest responding orientation neuron, as well as its gaussian blur, kernel size, and optical density at that pixel
    # plot the orientation output as a 3d image. let a cartesian coordinate system represent the (x,y) pixel coordinates, and use the third dimension to represent the strongest responding orientation neuron, its gaussian blur, kernel size, and optical density at that pixel
    # derive a closed form solution for a hyperspline: a multidimensional spline that is fit to the strongest responding orientation neuron, its gaussian blur, kernel size, and optical density at each pixel
    # the idea is that when fitting in this high dimensional space, there should be fewer sharp transitions and more smooth variations across the different dimensions.

    # when tracing a hyperspline along its length in (x,y), call this z, we should be able to observe smooth variations in the strongest responding orientation neuron, its gaussian blur, kernel size, and optical density at each pixel
    # we should then have smooth functions of these parameters along z of the hyperspline.

    # theoretically, vessels that cross over in x,y should be more separable in the high-dimensional hyperspline space, allowing for smoother and more accurate tracking of individual vessels.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 17 · From the fixation-0 average to a vessel graph

    This part turns the fixation-0 average into a set of vessels and the junctions between them, in a few stages that each hand the next one plain variables:

    | § | Stage | Output |
    |---|---|---|
    | 17 | the **orientation field**: a bank of neurons over blurs, kernel sizes and orientations, kept per scale band | `banded_scores`, `banded_scales` |
    | 18 | the **hyperspline**: a closed-form smoothing spline through everything the field says along a vessel | `fit_hyperspline` |
    | 19 | **tracing**: walks along the field, largest vessels first | `traced_vessels` |
    | 20 | the **whole field as one system**: every vessel and every junction fitted to the density at once; faint stretches cut, redundant vessels dropped | `settled_vessels`, `settled_fit` |
    | 21 | **continuations**: every end carried on, joined to another end, snapped onto the vessel it meets, or bridged | `continued_vessels` |
    | 22 | **refinement**: each vessel's centre, width and profile fitted to the density, with sudden changes penalised | `final_vessels`, `final_fit` |
    | 23 | **coarse to fine**: the haze between the vessels, then corrections where the residual says the fit is wrong, at finer and finer knots | `haze`, `fine_vessels`, `fine_fit` |
    | 24 | **flow**: each pixel's variance over time, a picture of the moving blood alone; every vessel's flow along it | `flows`, `flow_vessels`, `flow_amplitude_fit` |
    | 25 | the **vessel graph**: vessels, junctions, and what the model still misses | `vessel_table`, `junctions`, `vessel_model` |

    The exploratory versions of these steps live in `karim_attempt_exploration.ipynb`: the on/off-centre field, the 3D views, the walker comparison at crossings, the 2D signal comparisons, and greedy subtraction.

    **Input.** The average over time of `final_density_fixations[0]` is `final_density_means[0]`, from §12: optical density, with vessels bright on a background at 0. Glare leaves a few small holes inside the data; each is filled with the mean of the data around it. Outside the data the input is 0, the background's level, and neurons within reach of the data's outer edge stay silent. The background's spread, `nf_noise`, is measured on the quiet background pixels of §11–§12. It is the noise level that later tests are stated in.

    **Saved to disk.** The orientation field is saved in `neural_cache/` under a hash of its input image, its settings and the source code of the functions that compute it. A later run with the same three loads it instead of computing it again. Every figure goes to `vessel_pictures/neural_fields/` under the same hash.
    """)
    return


@app.cell
def _(
    Path,
    cv2,
    final_background_fits,
    final_density_means,
    hashlib,
    json,
    masked_mean,
    np,
    os,
    quiet_pixels,
    robust_sd,
    scipy,
):
    import inspect

    import matplotlib.colors

    NF_FIXATION = 0                            # the fixation whose average the pipeline reads
    NF_BLURS = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0)  # pixels; Gaussian blurs applied to the image before the convolutions
    NF_KERNEL_SIZES = (7, 11, 17, 25, 37, 55)  # pixels; side of the square convolution kernels
    NF_ORIENTATIONS = 16                       # orientation neurons, spread evenly over 180 degrees
    NF_WAVELET = "gabor"                       # orientation kernels: "gabor" or "gauss_d2" (lines), "gabor_odd" or "gauss_d1" (edges)
    NF_BANDS = ((8.0, "large"), (3.5, "medium"), (0.0, "small"))  # (smallest matched width in px, name), largest first
    NF_CACHE = Path("neural_cache")            # where the field is saved, under a hash of its input, settings and code
    NF_FIGURES = Path("vessel_pictures") / "neural_fields"  # where the figures are saved, under the same hash


    def source_of(function):
        """Return a function's source code, so that changing the code changes the hash."""
        try:
            return inspect.getsource(function)
        except (OSError, TypeError):
            return function.__qualname__


    def field_digest(arrays, settings, code):
        """Return a short hash of some arrays, a dict of settings and the source code of some functions."""
        digest = hashlib.sha1(usedforsecurity=False)
        for array in arrays:
            array = np.ascontiguousarray(array)
            digest.update(f"{array.shape}:{array.dtype}".encode("utf-8"))
            digest.update(array.tobytes())
        digest.update(json.dumps(settings, sort_keys=True, default=str).encode("utf-8"))
        for function in code:
            digest.update(source_of(function).encode("utf-8"))
        return digest.hexdigest()[:16]


    def cached(name, compute, arrays, settings, code):
        """Return compute() and its hash, loading it from NF_CACHE when the same input, settings and code ran before."""
        key = field_digest(arrays, settings, code)
        path = NF_CACHE / f"{name}_{key}.npz"
        if path.exists():
            with np.load(path) as saved:
                result = {item: saved[item] for item in saved.files}
            print(f"Loaded {name} from {path}")
        else:
            result = compute()
            NF_CACHE.mkdir(exist_ok=True)
            partial = NF_CACHE / f"{name}_{key}.partial.npz"  # renamed once complete, so an interrupted save is never loaded
            np.savez(partial, **result)
            os.replace(partial, path)
            print(f"Computed {name} and saved it to {path} ({path.stat().st_size / 1e6:.0f} MB)")
        return result, key


    def save_figure(figure, name, key):
        """Save a figure in NF_FIGURES under its name and the hash of what it shows."""
        NF_FIGURES.mkdir(parents=True, exist_ok=True)
        path = NF_FIGURES / f"{name}_{key}.png"
        figure.savefig(path, dpi=150)
        print(f"Saved {path}")


    nf_average = final_density_means[NF_FIXATION]  # the average over time of final_density_fixations[NF_FIXATION] (temporal_mean, §12)
    nf_data = np.isfinite(nf_average) & final_background_fits[NF_FIXATION]["ok"]
    nf_outline = scipy.ndimage.binary_fill_holes(nf_data)  # the data's outline, without the small holes glare leaves inside
    nf_holes = nf_outline & ~nf_data
    nf_image = np.where(nf_data, nf_average, 0).astype(np.float32)  # outside the outline: the background's density, 0
    nf_image = np.where(nf_holes, masked_mean(nf_image, nf_data, 4), nf_image).astype(np.float32)  # a hole: the data around it
    nf_margin = max(NF_KERNEL_SIZES) // 2 + int(np.ceil(2 * max(NF_BLURS)))  # pixels; how far the widest neuron reaches
    nf_inside = cv2.erode(nf_outline.astype(np.uint8), np.ones((2 * nf_margin + 1,) * 2, np.uint8), borderValue=0) > 0
    nf_background = quiet_pixels[NF_FIXATION] & final_background_fits[NF_FIXATION]["bg"] & nf_inside
    nf_noise_pixels = np.flatnonzero(nf_background)[::4]  # every neuron's noise is measured on these pixels
    nf_noise = robust_sd(nf_image.flat[nf_noise_pixels])  # optical density; the spread of the background, for §20's test
    print(f"Fixation {NF_FIXATION}: neurons fire on {nf_inside.mean() * 100:.0f}% of the frame, at least {nf_margin} px from "
          f"where the data ends; noise is measured on {len(nf_noise_pixels)} background pixels, where the density spreads "
          f"by {nf_noise:.4f}")
    return (
        NF_BANDS,
        NF_BLURS,
        NF_FIXATION,
        NF_KERNEL_SIZES,
        NF_ORIENTATIONS,
        NF_WAVELET,
        cached,
        matplotlib,
        nf_background,
        nf_data,
        nf_image,
        nf_inside,
        nf_noise,
        nf_noise_pixels,
        save_figure,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The orientation field

    Every neuron is one kernel applied to one blurred copy of the image. A kernel is an envelope times a profile across its line, made blind to a flat picture (zero sum) and given unit energy:

    | Setting | Meaning |
    |---|---|
    | `NF_BLURS` | Gaussian blurs (σ, px) applied to the image before the convolutions; 0 is the image itself |
    | `NF_KERNEL_SIZES` | side of the square kernels, in px; the kernel's widths scale with it |
    | `NF_ORIENTATIONS` | orientation neurons, spread evenly over 180° |
    | `NF_WAVELET` | the profile across the line: `gabor` or `gauss_d2` answer to lines, `gabor_odd` or `gauss_d1` to edges |
    | `NF_BANDS` | scale bands, by the width a neuron is matched to, $\sqrt{\text{blur}^2 + (\text{kernel size}/8)^2}$ |

    **Responses in noise units.** Every neuron's response is divided by its spread over the quiet background pixels. A response of 5 means five times what noise alone gives, whatever the neuron's blur, size or orientation, so neurons of every scale compete fairly.

    **Kept per band.** With these settings there are 6 blurs × 6 sizes × 16 orientations = 576 neurons at every pixel, far too many maps to keep. For every scale band and orientation, the strongest neuron over that band's blurs and sizes is kept (`banded_scores`), with which blur and size it was (`banded_scales`). Each band then sees its own scale's response, even where a vessel of another scale is stronger. That is what lets §19 trace the large vessels first, without a thin vessel's response pulling the walk away.

    The figure shows the kernels of the wavelet in use at one size. Below them: at every pixel, the strongest neuron over everything, its orientation, its band, and each band's own strongest neuron.

    | Output | Content |
    |---|---|
    | `banded_scores` | `(bands, orientations, rows, columns)`, float16: per band and orientation, the strongest neuron, in noise units |
    | `banded_scales` | the same shape: which blur and size it was, as `b * len(NF_KERNEL_SIZES) + s` |
    | `band_peak`, `strongest_response` | per band, and over all bands, the strongest neuron at every pixel |
    """)
    return


@app.cell
def _(
    NF_BANDS,
    NF_BLURS,
    NF_FIXATION,
    NF_KERNEL_SIZES,
    NF_ORIENTATIONS,
    NF_WAVELET,
    ThreadPoolExecutor,
    cached,
    cv2,
    matplotlib,
    nf_image,
    nf_inside,
    nf_noise_pixels,
    np,
    plt,
    robust_sd,
    save_figure,
    workers,
):
    def kernel_grid(size):
        """Return the x and y offsets of every place in a square kernel of `size` (odd) pixels, from its centre."""
        half = size // 2
        y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float64)
        return x, y


    def balanced(kernel, envelope=None):
        """Make a kernel blind to a flat picture (zero sum) and give it unit energy, so that noise alone gives every kernel
        the same spread. With an envelope, the sum is taken out in the envelope's shape, which keeps the kernel compact."""
        if envelope is None:
            kernel = kernel - kernel.mean()
        else:
            kernel = kernel - envelope * kernel.sum() / envelope.sum()
        return (kernel / np.sqrt(np.sum(kernel ** 2))).astype(np.float32)


    def orientation_kernel(size, k, wavelet=None):
        """Return the kernel of orientation neuron k: it answers best to a line (or an edge) running at
        k x 180 / NF_ORIENTATIONS degrees from the x axis, turning towards +y."""
        wavelet = wavelet or NF_WAVELET
        theta = k * np.pi / NF_ORIENTATIONS
        x, y = kernel_grid(size)
        along = x * np.cos(theta) + y * np.sin(theta)
        across = -x * np.sin(theta) + y * np.cos(theta)
        width, length = size / 8, size / 4  # the envelope's Gaussian widths across and along the line
        envelope = np.exp(-0.5 * (across / width) ** 2 - 0.5 * (along / length) ** 2)
        profiles = {
            "gabor": np.cos(np.pi * across / (2 * width)),      # even: a bright centre between dark flanks, a line
            "gabor_odd": np.sin(np.pi * across / (2 * width)),  # odd: bright on one side, dark on the other, an edge
            "gauss_d2": 1 - (across / width) ** 2,               # minus the second derivative of a Gaussian: a line
            "gauss_d1": -across / width,                         # minus the first derivative of a Gaussian: an edge
        }
        if wavelet not in profiles:
            raise ValueError(f"NF_WAVELET must be one of {sorted(profiles)}")
        return balanced(envelope * profiles[wavelet], envelope)


    def noise_units(response):
        """Divide a neuron's response by its spread over the background pixels, and silence it near the data's edge."""
        response /= robust_sd(response.flat[nf_noise_pixels])
        response[~nf_inside] = 0
        return response


    def blurred_copies(image):
        """Yield each blur of NF_BLURS with the image blurred by it."""
        for blur in NF_BLURS:
            yield blur, image if blur == 0 else cv2.GaussianBlur(image, (0, 0), blur)


    def band_index(blur, size):
        """The scale band of NF_BANDS that the neurons of one blur and kernel size belong to, by their matched width."""
        width = np.hypot(blur, size / 8)
        return next(number for number, (smallest, _) in enumerate(NF_BANDS) if width >= smallest)


    def banded_orientation_field(image):
        """§17's orientation field with its strongest neuron kept per scale band: for every band and orientation, the
        strongest neuron over that band's blurs and sizes ("scores", noise units, float16) and which it was ("scales")."""
        sizes, odd = len(NF_KERNEL_SIZES), NF_WAVELET in ("gabor_odd", "gauss_d1")
        scores = np.zeros((len(NF_BANDS), NF_ORIENTATIONS, *image.shape), np.float16)
        scales = np.zeros((len(NF_BANDS), NF_ORIENTATIONS, *image.shape), np.uint8)
        for b, (blur, blurred) in enumerate(blurred_copies(image)):
            for s, size in enumerate(NF_KERNEL_SIZES):
                band = band_index(blur, size)
                def respond(k):
                    response = noise_units(cv2.filter2D(blurred, cv2.CV_32F, orientation_kernel(size, k),
                                                        borderType=cv2.BORDER_REFLECT))
                    response = (np.abs(response) if odd else np.maximum(response, 0)).astype(np.float16)
                    stronger = response > scores[band, k]
                    scores[band, k][stronger] = response[stronger]
                    scales[band, k][stronger] = b * sizes + s
                with ThreadPoolExecutor(max_workers=workers) as pool:  # each thread writes only its own orientation
                    list(pool.map(respond, range(NF_ORIENTATIONS)))
        return {"scores": scores, "scales": scales}


    banded, banded_key = cached(
        "banded_field", lambda: banded_orientation_field(nf_image), (nf_image, nf_inside, nf_noise_pixels),
        {"blurs": NF_BLURS, "sizes": NF_KERNEL_SIZES, "orientations": NF_ORIENTATIONS, "wavelet": NF_WAVELET,
         "bands": NF_BANDS},
        (banded_orientation_field, band_index, orientation_kernel, balanced, kernel_grid, noise_units, blurred_copies,
         robust_sd))
    banded_scores, banded_scales = banded["scores"], banded["scales"]  # (bands, orientations, rows, columns)
    del banded
    for _band, (_smallest, _name) in enumerate(NF_BANDS):
        _members = [f"{blur:g}/{size}" for blur in NF_BLURS for size in NF_KERNEL_SIZES if band_index(blur, size) == _band]
        print(f"Band {_band} ({_name}, matched width {_smallest:g} px and up): blur/kernel size {', '.join(_members)}")

    # the strongest neuron at every pixel, over every band, orientation, blur and size
    band_peak = banded_scores.max(axis=1)  # (bands, rows, columns): each band's strongest orientation at every pixel
    strongest_band = np.argmax(band_peak, axis=0)
    strongest_response = np.take_along_axis(band_peak, strongest_band[None], axis=0)[0].astype(np.float32)
    strongest_orientation = np.argmax(np.take_along_axis(banded_scores, strongest_band[None, None], axis=0)[0], axis=0)
    fires = strongest_response >= 3.0
    print(f"The strongest neuron fires above 3 noise units on {fires[nf_inside].mean() * 100:.0f}% of the pixels; there, "
          + ", ".join(f"the {name} band is the strongest on {np.mean(strongest_band[fires & nf_inside] == band) * 100:.0f}%"
                      for band, (_, name) in enumerate(NF_BANDS)))

    # the kernels of the wavelet in use, then the field: the strongest neuron, its orientation, its band, and each band
    _middle = NF_KERNEL_SIZES[len(NF_KERNEL_SIZES) // 2]
    _figure = plt.figure(figsize=(16, 8.6))
    for _k in range(NF_ORIENTATIONS):
        _axis = _figure.add_axes((0.05 + _k * 0.9 / NF_ORIENTATIONS, 0.875, 0.8 / NF_ORIENTATIONS, 0.075))
        kernel = orientation_kernel(_middle, _k)
        _axis.imshow(kernel, cmap="gray", vmin=-np.abs(kernel).max(), vmax=np.abs(kernel).max())
        _axis.set_title(f"{_k * 180 / NF_ORIENTATIONS:.0f}°", fontsize=7)
        _axis.axis("off")
    _top = np.percentile(strongest_response[fires], 99)
    hue = matplotlib.colors.hsv_to_rgb(np.dstack([strongest_orientation / NF_ORIENTATIONS,
                                                  np.full(strongest_response.shape, 0.85),
                                                  np.clip(strongest_response / _top, 0, 1)]))
    band_colors = matplotlib.colors.ListedColormap(plt.get_cmap("Blues")(np.linspace(1.0, 0.35, len(NF_BANDS))))
    band_colors.set_bad("black")
    _panels = [("the strongest neuron (noise units)", strongest_response, dict(cmap="gray", vmin=0, vmax=_top)),
              ("its orientation (hue); brightness is its response", hue, {}),
              ("its band", np.where(fires, strongest_band, np.nan),
               dict(cmap=band_colors, vmin=-0.5, vmax=len(NF_BANDS) - 0.5, interpolation="nearest"))]
    _panels += [(f"the {_name} band's strongest neuron (noise units)", band_peak[_band].astype(np.float32),
                dict(cmap="gray", vmin=0, vmax=_top)) for _band, (_, _name) in enumerate(NF_BANDS)]
    for _place, (_title, _picture, _style) in enumerate(_panels):
        _axis = _figure.add_axes((0.01 + (_place % 3) * 0.32, 0.44 - (_place // 3) * 0.425, 0.28, 0.37))
        _image = _axis.imshow(_picture, **_style)
        if _place == 2:
            bar = _figure.colorbar(_image, ax=_axis, ticks=range(len(NF_BANDS)), fraction=0.03, pad=0.01)
            bar.ax.set_yticklabels([_name for _, _name in NF_BANDS])
        elif _place != 1:
            _figure.colorbar(_image, ax=_axis, fraction=0.03, pad=0.01)
        _axis.set_title(_title, fontsize=10)
        _axis.axis("off")
    _figure.suptitle(f"top: the {NF_ORIENTATIONS} orientation kernels of '{NF_WAVELET}' at {_middle} px (white excites, black "
                    f"inhibits); below: the banded orientation field of fixation {NF_FIXATION}", fontsize=10, y=0.995)
    save_figure(_figure, "field", banded_key)
    plt.show()
    del strongest_orientation, hue
    return (
        band_peak,
        banded_key,
        banded_orientation_field,
        banded_scales,
        banded_scores,
        strongest_response,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 18 · The hyperspline

    ### A closed form

    Take the samples along a traced vessel, $(z_i, \mathbf{y}_i)$ for $i = 1 \dots n$. Here $z$ is the length along the vessel in $(x, y)$, and each $\mathbf{y}_i \in \mathbb{R}^d$ stacks what the field says at that point: $x$, $y$, $\cos 2\theta$, $\sin 2\theta$, blur, kernel size, matched width, density and response. Every vessel from §19 on is carried by its hyperspline: its centreline and starting width are read off it. The **hyperspline** is the curve $\mathbf{g} : \mathbb{R} \to \mathbb{R}^d$ that minimizes

    $$
    \sum_{i=1}^{n} w_i \,\bigl\lVert \mathbf{y}_i - \mathbf{g}(z_i) \bigr\rVert^2 \;+\; \lambda \int \bigl\lVert \mathbf{g}''(z) \bigr\rVert^2 \, dz .
    $$

    Both terms are sums over the $d$ coordinates, so the problem splits into $d$ scalar problems that share $z_i$, $w_i$ and $\lambda$. Each scalar minimizer is a **natural cubic spline** with a knot at every $z_i$ (Reinsch, 1967). Such a spline is fixed by its values $g_i$ and its second derivatives $\gamma_i$ at the knots, with $\gamma_1 = \gamma_n = 0$.

    Write $h_i = z_{i+1} - z_i$. Continuity of the first derivative across the knots is the linear condition $Q^\top \mathbf{g} = R\,\boldsymbol{\gamma}$, with $\boldsymbol{\gamma} = (\gamma_2 \dots \gamma_{n-1})$. Here $Q$ is $n \times (n-2)$ and $R$ is $(n-2) \times (n-2)$, both banded:

    $$
    Q_{j-1,j} = \tfrac{1}{h_{j-1}}, \quad Q_{j,j} = -\tfrac{1}{h_{j-1}} - \tfrac{1}{h_j}, \quad Q_{j+1,j} = \tfrac{1}{h_j}; \qquad
    R_{j,j} = \tfrac{h_{j-1} + h_j}{3}, \quad R_{j,j+1} = R_{j+1,j} = \tfrac{h_j}{6}.
    $$

    The penalty is then a quadratic form in the values, $\int g''^2\,dz = \boldsymbol{\gamma}^\top R\,\boldsymbol{\gamma} = \mathbf{g}^\top Q R^{-1} Q^\top \mathbf{g}$. With $W = \operatorname{diag}(w_i)$, the whole objective is quadratic in $\mathbf{g}$, so its minimizer has a closed form:

    $$
    \hat{\mathbf{g}} = \left(W + \lambda\, Q R^{-1} Q^\top\right)^{-1} W\, \mathbf{y} .
    $$

    It is computed without any inverse. Setting $\boldsymbol{\gamma} = R^{-1} Q^\top \mathbf{g}$ in $W(\mathbf{y} - \mathbf{g}) = \lambda Q \boldsymbol{\gamma}$ gives a banded (pentadiagonal) system:

    $$
    \left(R + \lambda\, Q^\top W^{-1} Q\right) \boldsymbol{\gamma} = Q^\top \mathbf{y}, \qquad \hat{\mathbf{g}} = \mathbf{y} - \lambda\, W^{-1} Q\, \boldsymbol{\gamma} .
    $$

    One factorization serves all $d$ coordinates at once ($\mathbf{y}$ is $n \times d$). Between two knots the spline is a cubic, also in closed form. With $s = z - z_i$:

    $$
    g(z) = g_i + b_i s + \tfrac{\gamma_i}{2} s^2 + \tfrac{\gamma_{i+1} - \gamma_i}{6 h_i} s^3, \qquad b_i = \tfrac{g_{i+1} - g_i}{h_i} - \tfrac{h_i}{6}\,(2\gamma_i + \gamma_{i+1}).
    $$

    **Choices.**

    - **$\lambda$ is set by a length.** With samples $\Delta z$ apart, a smoothing spline averages over about $L = (\lambda\,\Delta z)^{1/4}$ along the curve (Silverman, 1984). So `HYPERSPLINE_LENGTH` sets $\lambda = L^4 / \Delta z$.
    - **Orientation enters as $(\cos 2\theta, \sin 2\theta)$.** A line at 179° and one at 1° are then neighbours, and the spline sees no jump at the wrap. The fitted orientation is $\hat\theta = \tfrac{1}{2}\operatorname{atan2}(\hat s, \hat c)$.
    - **Blur and kernel size** can only take the values of the neuron banks. Along $z$, the spline turns their steps into a smooth scale.
    - **Units between coordinates do not matter.** The problem splits by coordinate, so multiplying a coordinate by a constant multiplies its fit by the same constant.

    The cell checks the closed form against `scipy.interpolate.make_smoothing_spline`, which minimizes the same objective by another route.
    """)
    return


@app.cell
def _(np, scipy):
    HYPERSPLINE_LENGTH = 8.0  # pixels along the vessel over which the hyperspline averages


    def knot_matrices(z):
        """Reinsch's banded matrices for knots at the increasing positions z: Q (n x (n - 2)) and R ((n - 2) x (n - 2))."""
        n, h = len(z), np.diff(z)
        if n < 4 or np.any(h <= 0):
            raise ValueError("a hyperspline needs at least 4 samples at strictly increasing z")
        Q = scipy.sparse.diags([1 / h[:-1], -1 / h[:-1] - 1 / h[1:], 1 / h[1:]], [0, -1, -2], shape=(n, n - 2))
        R = scipy.sparse.diags([(h[:-1] + h[1:]) / 3, h[1:-1] / 6, h[1:-1] / 6], [0, 1, -1], shape=(n - 2, n - 2))
        return Q, R


    def natural_spline(z, fitted, second):
        """The natural cubic spline through the rows of `fitted` at the knots z, with second derivatives `second` (zero at
        both ends), piece by piece, as a scipy PPoly."""
        h = np.diff(z)[:, None]
        slope = np.diff(fitted, axis=0) / h - h * (2 * second[:-1] + second[1:]) / 6
        cubic = np.diff(second, axis=0) / (6 * h)
        return scipy.interpolate.PPoly(np.stack([cubic, second[:-1] / 2, slope, fitted[:-1]]), z)


    def fit_hyperspline(z, values, length=HYPERSPLINE_LENGTH, weights=None):
        """Fit the hyperspline: the closed-form natural cubic smoothing spline through the rows of `values` (n x d) at the
        increasing positions z, all d coordinates at once. Returns it as a piecewise cubic (scipy PPoly: call it with z, or
        take .derivative()), and the lambda that the smoothing length gave."""
        z = np.asarray(z, np.float64)
        values = np.asarray(values, np.float64).reshape(len(z), -1)
        n = len(z)
        Q, R = knot_matrices(z)
        weights = np.ones(n) if weights is None else np.asarray(weights, np.float64)
        lam = length ** 4 / np.median(np.diff(z))
        W_inverse = scipy.sparse.diags(1 / weights)
        gamma = scipy.sparse.linalg.spsolve((R + lam * (Q.T @ W_inverse @ Q)).tocsc(), Q.T @ values).reshape(n - 2, -1)
        fitted = values - lam * (W_inverse @ (Q @ gamma))
        second = np.vstack([np.zeros((1, values.shape[1])), gamma, np.zeros((1, values.shape[1]))])
        return natural_spline(z, fitted, second), lam


    # the closed form against scipy's smoothing spline, which minimizes the same thing, on a noisy test curve
    generator = np.random.default_rng(1)
    test_z = np.cumsum(generator.uniform(1.0, 3.0, 200))
    test_values = np.column_stack([np.sin(test_z / 40), np.cos(test_z / 25)]) + 0.1 * generator.standard_normal((200, 2))
    test_spline, test_lam = fit_hyperspline(test_z, test_values)
    _reference = np.column_stack([scipy.interpolate.make_smoothing_spline(test_z, test_values[:, j], lam=test_lam)(test_z)
                                 for j in range(test_values.shape[1])])
    print(f"Closed form against scipy.interpolate.make_smoothing_spline (lambda {test_lam:.3g}): largest difference "
          f"{np.abs(test_spline(test_z) - _reference).max():.1e} on values of order 1")
    return (fit_hyperspline,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 19 · Tracing, largest vessels first

    The bands are taken in order, large to small. In each band:

    1. **Seeds.** The seeds are the band's crest pixels: pixels stronger than both neighbours across their line, at least `SEED_LEVEL` strong, where this band is the vessel's own scale. That means it fires at least `BAND_SHARE` of every other band. A blur of 8 or 16 px makes even a thin vessel answer strongly in the large band, so being strong in a band is not enough. The seeds are taken strongest first, at most `MOST_VESSELS` kept per band.
    2. **Walk.** From each seed not yet explained, a walker steps `TRACE_STEP` px at a time both ways along the line. After each step it re-centres itself across the line, within `TRACE_REACH` px. It reads only the three neurons nearest its own orientation, and turns by at most `TRACE_BEND` degrees per pixel. That is what carries it straight through a crossing, where the strongest neuron would turn it onto the other vessel.
    3. **Scale.** Where another band's neuron of the same orientation has fired more than its own for `ALONG_STEPS` steps, the vessel has widened or narrowed out of the band. The walk moves to that band and goes on, so that a vessel tapering through 3.5 or 8 px stays one vessel.
    4. **Ends.** A walk ends where its neuron fires below `TRACE_STOP`, at the data's edge, or where it has run along a vessel traced before for `ALONG_STEPS` steps (oriented within 20° of it). It has then joined that vessel. This keeps a thin branch from being traced on along the thick vessel it joins.
    5. **Keep.** A walk of at least `MIN_VESSEL` px becomes a vessel, and its hyperspline is fitted. Its **core**, `CORE_WIDTHS` matched widths to either side of the centreline, is marked as explained: no later seed starts inside it.

    | Output | Content |
    |---|---|
    | `traced_vessels` | every vessel traced: the band it started in, its walk (`traced`: x, y, orientation, band, blur, kernel size, response and density per step, and z) and its hyperspline (`spline`) |
    | `traced_orientation` | the orientation of every traced vessel's core, `NaN` elsewhere |
    """)
    return


@app.cell
def _(
    DENSITY_RANGE,
    NF_BANDS,
    NF_BLURS,
    NF_KERNEL_SIZES,
    NF_ORIENTATIONS,
    SERIES,
    band_peak,
    banded_key,
    banded_scales,
    banded_scores,
    cv2,
    fit_hyperspline,
    nf_image,
    nf_inside,
    np,
    plt,
    save_figure,
):
    TRACE_STEP = 2.0    # pixels the walker moves along its line per step
    TRACE_REACH = 3.0   # pixels; how far across its line the walker looks to re-centre itself after each step
    TRACE_BEND = 3.0    # degrees per pixel walked that the walker's orientation may turn by
    TRACE_STOP = 3.0    # noise units; a walk ends where its neuron fires below this
    SEED_LEVEL = 10.0   # noise units; a vessel is started from a crest pixel of its band at least this strong
    BAND_SHARE = 1.0    # a band seeds only where it fires at least this share of every other band; a walk moves to
                        # another band where that band fires more than its own divided by this, for ALONG_STEPS steps
    CORE_WIDTHS = 1.5   # matched widths; a traced vessel's core reaches this far to either side of its centreline
    MIN_VESSEL = 40.0   # pixels; a shorter vessel is not kept
    MOST_VESSELS = 150  # vessels per band at the most
    ALONG_STEPS = 3     # steps in a row along a traced vessel's core that end a walk (or out of its band, that move it)


    def bilinear(picture, xs, ys):
        """Read a picture between its pixels (linear interpolation) at the points (xs, ys)."""
        xs = np.clip(np.asarray(xs, np.float64), 0, picture.shape[1] - 1.001)
        ys = np.clip(np.asarray(ys, np.float64), 0, picture.shape[0] - 1.001)
        x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
        fx, fy = xs - x0, ys - y0
        at = lambda row, column: picture[row, column].astype(np.float64)
        return (at(y0, x0) * (1 - fx) * (1 - fy) + at(y0, x0 + 1) * fx * (1 - fy)
                + at(y0 + 1, x0) * (1 - fx) * fy + at(y0 + 1, x0 + 1) * fx * fy)


    def scores_at(neurons, xs, ys, scores):
        """The responses of some orientation neurons of `scores` (orientations x rows x columns) at the points (xs, ys),
        read between pixels: (neurons, points)."""
        xs = np.clip(np.atleast_1d(np.asarray(xs, np.float64)), 0, scores.shape[2] - 1.001)
        ys = np.clip(np.atleast_1d(np.asarray(ys, np.float64)), 0, scores.shape[1] - 1.001)
        x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
        fx, fy = xs - x0, ys - y0
        neurons = np.asarray(neurons)[:, None]
        at = lambda row, column: scores[neurons, row[None], column[None]].astype(np.float64)
        return (at(y0, x0) * (1 - fx) * (1 - fy) + at(y0, x0 + 1) * fx * (1 - fy)
                + at(y0 + 1, x0) * (1 - fx) * fy + at(y0 + 1, x0 + 1) * fx * fy)


    def inside(x, y):
        """Whether (x, y) lies where the neurons fire (away from the data's edge)."""
        column, row = int(round(x)), int(round(y))
        return 0 <= row < nf_inside.shape[0] and 0 <= column < nf_inside.shape[1] and bool(nf_inside[row, column])


    def neuron_at(x, y, k, field):
        """What one band's field (a pair of its scores and scales) says at (x, y) on orientation neuron k: the orientation
        between the neurons (a parabola through k and its two neighbours), the neuron's response, blur and kernel size, and
        the density there."""
        scores, scales = field
        n = NF_ORIENTATIONS
        below, here, above = scores_at([(k - 1) % n, k, (k + 1) % n], x, y, scores)[:, 0]
        curvature = below - 2 * here + above
        offset = float(np.clip(0.5 * (below - above) / curvature, -0.5, 0.5)) if curvature < 0 else 0.0
        scale = int(scales[k][int(round(y)), int(round(x))])
        return {"x": float(x), "y": float(y), "orientation": ((k + offset) % n) * np.pi / n, "neuron": k,
                "response": float(here), "blur": NF_BLURS[scale // len(NF_KERNEL_SIZES)],
                "kernel_size": NF_KERNEL_SIZES[scale % len(NF_KERNEL_SIZES)], "density": float(bilinear(nf_image, x, y))}


    def walk(point, heading, along):
        """Walk along a line from `point` (as neuron_at gives it, with its band) in the direction `heading` (a unit
        (dx, dy)), and return the points passed. The walker reads its band's three neurons nearest its own orientation and
        turns by at most TRACE_BEND degrees per pixel. Where for ALONG_STEPS steps another band's neuron of the same
        orientation fires more than its own divided by BAND_SHARE, the vessel has widened or narrowed out of the band, and
        the walker moves to that band. It ends where its neuron fires below TRACE_STOP, or where it has run for ALONG_STEPS
        steps along a vessel traced before (`along`: their orientation, NaN elsewhere)."""
        n, band = NF_ORIENTATIONS, point["band"]
        field = (banded_scores[band], banded_scales[band])
        x, y, theta = point["x"], point["y"], point["orientation"]
        offsets = np.linspace(-TRACE_REACH, TRACE_REACH, 13)
        largest_turn = np.radians(TRACE_BEND * TRACE_STEP)
        points, alongside, outscaled = [], 0, 0
        for _ in range(300):
            x, y = x + TRACE_STEP * heading[0], y + TRACE_STEP * heading[1]
            if not inside(x, y):
                break
            xs, ys = x - offsets * heading[1], y + offsets * heading[0]  # across the line
            nearest = int(round(theta * n / np.pi)) % n
            neurons = [(nearest + turn) % n for turn in (-1, 0, 1)]
            responses = scores_at(neurons, xs, ys, field[0])
            best, across = np.unravel_index(np.argmax(responses), responses.shape)
            if responses[best, across] < TRACE_STOP or not inside(xs[across], ys[across]):
                break
            x, y = float(xs[across]), float(ys[across])
            if len(points) > 5 and min(np.hypot(p["x"] - x, p["y"] - y) for p in points[:-5]) < TRACE_STEP / 2:
                break  # it came back on itself
            here = {**neuron_at(x, y, neurons[best], field), "band": band}
            points.append(here)
            traced_theta = along[int(round(y)), int(round(x))]
            same_way = np.isfinite(traced_theta) and np.degrees(
                abs(np.angle(np.exp(2j * (here["orientation"] - traced_theta)))) / 2) < 20
            alongside = alongside + 1 if same_way else 0
            if alongside >= ALONG_STEPS:
                del points[-alongside:]
                break
            others = {number: scores_at([here["neuron"]], x, y, banded_scores[number])[0, 0]
                      for number in range(len(NF_BANDS)) if number != band}
            rival = max(others, key=others.get)
            outscaled = outscaled + 1 if here["response"] < BAND_SHARE * others[rival] else 0
            if outscaled >= ALONG_STEPS:  # the vessel has left its band: the walk goes on in the band it has entered
                band, outscaled = rival, 0
                field = (banded_scores[band], banded_scales[band])
            turn = np.angle(np.exp(2j * (here["orientation"] - theta))) / 2  # to the orientation read, folded to +-90 degrees
            theta += np.clip(turn, -largest_turn, largest_turn)
            direction = np.array([np.cos(theta), np.sin(theta)])
            heading = direction if direction @ heading >= 0 else -direction  # an orientation has two directions: keep going
        return points


    def trace(point, along):
        """Walk both ways from a point (as neuron_at gives it, with its band). Returns the points in order along the line as
        a dict of arrays, with "z" the length along the line in (x, y), 0 at the start."""
        heading = np.array([np.cos(point["orientation"]), np.sin(point["orientation"])])
        backward = walk(point, -heading, along)
        points = backward[::-1] + [point] + walk(point, heading, along)
        traced = {key: np.array([p[key] for p in points]) for key in point}
        z = np.concatenate([[0], np.cumsum(np.hypot(np.diff(traced["x"]), np.diff(traced["y"])))])
        traced["z"] = z - z[len(backward)]
        return traced


    def samples_of(traced):
        """The coordinates the hyperspline is fitted to, one row per step of a walk: x, y, the orientation as (cos 2θ,
        sin 2θ), blur, kernel size, the matched width, the density and the response."""
        width = np.hypot(traced["blur"], traced["kernel_size"] / 8)  # the strongest neuron's width across the line
        return np.column_stack([traced["x"], traced["y"], np.cos(2 * traced["orientation"]),
                                np.sin(2 * traced["orientation"]), traced["blur"], traced["kernel_size"], width,
                                traced["density"], traced["response"]])


    def crest_seeds(scores, allowed):
        """The crest pixels of one band's scores, stronger than both neighbours across their line, at least SEED_LEVEL and
        within `allowed`, strongest first: their x, y and strongest neuron."""
        strongest = np.argmax(scores, axis=0)
        response = np.take_along_axis(scores, strongest[None], axis=0)[0].astype(np.float32)
        theta = (strongest * np.pi / NF_ORIENTATIONS).astype(np.float32)
        rows, columns = response.shape
        grid_x, grid_y = np.meshgrid(np.arange(columns, dtype=np.float32), np.arange(rows, dtype=np.float32))
        across_x, across_y = -np.sin(theta), np.cos(theta)
        crest = ((response >= SEED_LEVEL) & nf_inside & allowed
                 & (response >= cv2.remap(response, grid_x + across_x, grid_y + across_y, cv2.INTER_LINEAR))
                 & (response >= cv2.remap(response, grid_x - across_x, grid_y - across_y, cv2.INTER_LINEAR)))
        ys, xs = np.nonzero(crest)
        order = np.argsort(-response[ys, xs], kind="stable")
        return xs[order], ys[order], strongest[ys[order], xs[order]]


    def mark_core(spline, z_range, picture):
        """Draw a traced vessel's core into `picture`: its centreline, CORE_WIDTHS matched widths to either side, holding
        the vessel's fitted orientation there."""
        values = spline(np.arange(z_range[0], z_range[1], 1.0))
        theta = 0.5 * np.arctan2(values[:, 3], values[:, 2])
        ends = np.round(values[:, :2]).astype(int)
        for i in range(len(values) - 1):
            thickness = max(1, int(round(2 * CORE_WIDTHS * max(values[i, 6], 1.0))))
            cv2.line(picture, tuple(map(int, ends[i])), tuple(map(int, ends[i + 1])), float(theta[i]), thickness)


    traced_orientation = np.full(nf_image.shape, np.nan, np.float32)  # every traced vessel's core: its orientation
    _tried = np.zeros(nf_image.shape, bool)                              # seeds already walked from
    traced_vessels = []
    for _band, (_smallest, _name) in enumerate(NF_BANDS):
        strongest_other = np.max(np.delete(band_peak, _band, axis=0), axis=0)
        xs, ys, ks = crest_seeds(banded_scores[_band], band_peak[_band] >= BAND_SHARE * strongest_other)  # its own scale
        kept = tries = 0
        for _x, _y, _k in zip(xs, ys, ks):
            if kept == MOST_VESSELS or tries == 10 * MOST_VESSELS:
                break
            if _tried[_y, _x] or np.isfinite(traced_orientation[_y, _x]):
                continue
            tries += 1
            seed = {**neuron_at(_x, _y, int(_k), (banded_scores[_band], banded_scales[_band])), "band": _band}
            _traced = trace(seed, traced_orientation)
            for px, py in zip(np.round(_traced["x"]).astype(int), np.round(_traced["y"]).astype(int)):
                _tried[max(py - 2, 0):py + 3, max(px - 2, 0):px + 3] = True  # never walk this stretch again from a seed
            if _traced["z"][-1] - _traced["z"][0] < MIN_VESSEL or len(_traced["z"]) < 6:
                continue
            spline, _ = fit_hyperspline(_traced["z"], samples_of(_traced))
            traced_vessels.append({"band": _band, "traced": _traced, "spline": spline})
            mark_core(spline, (_traced["z"][0], _traced["z"][-1]), traced_orientation)
            kept += 1
        _lengths = [_vessel["traced"]["z"][-1] - _vessel["traced"]["z"][0] for _vessel in traced_vessels if _vessel["band"] == _band]
        print(f"Band {_band} ({_name}): {kept} vessels from {tries} seeds walked ({len(xs)} crest pixels at least "
              f"{SEED_LEVEL:g} where this band is the strongest scale), {sum(_lengths):.0f} px of centreline")
    print(f"{len(traced_vessels)} vessels in all; their cores cover "
          f"{np.isfinite(traced_orientation)[nf_inside].mean() * 100:.0f}% of the field")
    del _tried

    # every vessel over the density, each stretch in the band it was walked in
    BAND_STYLES = (dict(color=SERIES[0], linewidth=2.6), dict(color=SERIES[1], linewidth=1.7),
                   dict(color=SERIES[2], linewidth=1.1))
    _figure, _axis = plt.subplots(figsize=(16, 10.4))
    _axis.imshow(nf_image, cmap="gray", vmin=DENSITY_RANGE[0], vmax=DENSITY_RANGE[1])
    for _vessel in traced_vessels:
        _traced = _vessel["traced"]
        curve = _vessel["spline"](_traced["z"])
        _changes = np.flatnonzero(np.diff(_traced["band"])) + 1
        for run in np.split(np.arange(len(curve)), _changes):
            run = np.arange(run[0], min(run[-1] + 2, len(curve)))  # each stretch reaches the next one's first point
            _axis.plot(curve[run, 0], curve[run, 1], **BAND_STYLES[_traced["band"][run[0]]])
    for _band, (_smallest, _name) in enumerate(NF_BANDS):
        _axis.plot([], [], label=f"{_name} band (matched width {_smallest:g} px and up)", **BAND_STYLES[_band])
    switching = sum(len(np.unique(vessel["traced"]["band"])) > 1 for vessel in traced_vessels)
    _axis.set_title(f"{len(traced_vessels)} vessels traced, largest band first ({switching} of them change band along the way)",
                   fontsize=10)
    _axis.legend(loc="lower left", fontsize=9, frameon=True, facecolor="black", edgecolor="none", framealpha=0.7,
                labelcolor="white")
    _axis.axis("off")
    _figure.tight_layout()
    save_figure(_figure, "traced_vessels", banded_key)
    plt.show()
    return (
        CORE_WIDTHS,
        MIN_VESSEL,
        SEED_LEVEL,
        TRACE_BEND,
        TRACE_REACH,
        TRACE_STEP,
        bilinear,
        inside,
        samples_of,
        traced_vessels,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 20 · The whole field as one system, junctions included

    Every vessel is fitted to the density **together**, as one system. Vessel $k$ adds, at every pixel near it,

    $$
    D_k(\mathbf{p}) = a_k(z)\, g_k(\mathbf{p}), \qquad g_k(\mathbf{p}) = \exp\!\Bigl(-\tfrac{1}{2}\,\bigl|u / \sigma_k(z)\bigr|^{p_k}\Bigr).
    $$

    - $z$ is the pixel's place along the vessel's centreline, and $u$ its distance across it.
    - $\sigma_k(z)$ is the hyperspline's matched width there, smoothed along the vessel over `WIDTH_LENGTH` px.
    - The shape $p_k$ is 2, a Gaussian, until §22 fits each vessel its own.
    - The amplitude $a_k(z)$ is a cubic spline along the vessel with a knot every `KNOT_SPACING` px.
    - The profile reaches `PROFILE_SIGMAS` widths across, and past the vessel's ends it fades out as a Gaussian of the same width.

    ### Junctions: added up, or merged

    Where two vessels' cores overlap, their sum $a_1 g_1 + a_2 g_2$ is right for one case only. In optical density, absorptions add along the line of sight (Beer–Lambert), so two vessels at different depths, one passing over the other, do add up: a **crossing**. Two vessels whose lumens merge, a **joining** (a bifurcation), hold no more blood where they overlap than the deeper of the two. With both lumens centred at the same depth, the line of sight crosses the union of the two, whose thickness is the larger one, not the sum:

    $$
    \max(a_1 g_1,\, a_2 g_2) \;=\; a_1 g_1 + a_2 g_2 - \min(a_1 g_1,\, a_2 g_2).
    $$

    The correction is replaced by its closed form $c\, g_1 g_2$. Along either vessel's centreline, where the other profile is 1, $\min(a_1 g_1, a_2 g_2)$ and $\min(a_1, a_2)\, g_1 g_2$ agree exactly. Between the centrelines the product is a little smaller. For Gaussian profiles crossing at an angle, $g_1 g_2$ is itself a 2D Gaussian blob centred where the centrelines cross. Its shape is fixed by the two widths and the angle. Each junction therefore adds one unknown, linear like the amplitudes:

    $$
    D(\mathbf{p}) = \sum_k a_k(z)\, g_k(\mathbf{p}) \;-\; \sum_{\text{junctions } j} c_j\, g_{1(j)}(\mathbf{p})\, g_{2(j)}(\mathbf{p}) .
    $$

    Every pixel is one equation. The density is linear in all the unknowns, so the whole field's fit is one sparse least-squares problem with a closed form, $\bigl(A^\top A + P^\top P\bigr)\,\mathbf{x} = A^\top \mathbf{d}$. Here $A$ holds every vessel's basis pictures and every junction's product picture, and $P$ the penalties on each amplitude's changes (`AMPLITUDE_STEADINESS`) and bending (`AMPLITUDE_SMOOTHING`) along its vessel. A junction's $c_j$ must lie between 0 (the densities add up) and the fainter vessel's amplitude there (the lumens merge). One that falls outside is held at the bound it crossed, and the rest is solved again. Its **share**, $c_j / \min(a_1, a_2)$, says which it is: below 0.5 a crossing, from 0.5 a joining.

    A junction is found wherever a connected patch of pixels reached by both vessels holds a pixel where $g_1 g_2 \ge$ `JUNCTION_CORES`, which is where the two cores overlap. Its kind follows from how many of the two vessels end there: **X** (neither), **T** (one) or **V** (both).

    ### Settling: cut the faint, drop the redundant

    `settle(vessels)` runs here on the traced vessels, and again after §21, §22 and §23:

    1. **Cut.** The field is fitted, and every vessel is cut back to where its amplitude is at least `CUT_SHARE` of its typical amplitude (the 75th percentile along it). Faint stretches at its ends go. A faint stretch in its middle longer than two knots splits it in two. Pieces shorter than `MIN_VESSEL` go.
    2. **Drop.** A vessel is dropped when less than `UNIQUE_SHARE` of its own picture's energy is something no other vessel could explain. That share is exact for the linear system. It is how much worse the whole field's fit gets without the vessel, every other vessel refitted, $\mathbf{b}^\top M^{-1} \mathbf{b}$, over the energy of the vessel's own picture. Here $\mathbf{b}$ are its coefficients and $M$ their block of $(A^\top A + P^\top P)^{-1}$. A vessel traced twice, or a stub along a bigger vessel, explains almost nothing on its own. Weak vessels are dropped a few at a time, never two that touch in the same batch, and the field is refitted after every batch.

    | Output | Content |
    |---|---|
    | `fit_field(vessels)` | the joint fit as a function: `"model"`, `"amplitudes"` (per vessel, a BSpline along z), `"junctions"`, `"residual"`; with `target=`, fitted to that picture instead of the density (§23 fits the density without the haze) |
    | `settle(vessels)` | fit, cut, drop, fit again: the vessels kept, their fit, and counts |
    | `settled_vessels`, `settled_fit` | §19's vessels after settling |
    | `geometry_of(vessel)`, `vessel_picture(fit, k)` | a vessel's centre, width and shape along it; one vessel's part of a fit's model |
    """)
    return


@app.cell
def _(
    MIN_VESSEL,
    cv2,
    fit_hyperspline,
    nf_image,
    nf_noise,
    np,
    samples_of,
    scipy,
    traced_vessels,
):
    KNOT_SPACING = 8.0          # pixels between the knots of a vessel's amplitude (and, in §22, of its shift and width)
    PROFILE_SIGMAS = 3.0        # a vessel's cross-section is modelled out to this many of its widths
    AMPLITUDE_SMOOTHING = 10.0  # how strongly a vessel's amplitude resists bending along it (relative to the evidence)
    AMPLITUDE_STEADINESS = 1.0  # how strongly it resists any change along it: no sudden brightening or fading
    WIDTH_LENGTH = 40.0         # pixels along a vessel over which the width its walk gave is smoothed
    JUNCTION_CORES = 0.25       # two vessels meet where the product of their profiles reaches this: their cores overlap
    CUT_SHARE = 0.3             # a stretch of a vessel fainter than this share of its typical amplitude is cut away
    UNIQUE_SHARE = 0.3          # a vessel must explain at least this share of its own model's energy that no other could
    RESIDUAL_RANGE = 0.1        # optical density; the residual pictures span this either side of 0


    def vessel_profile(u, sigma, shape):
        """A vessel's cross-section at the distance u across it: exp(-|u / sigma|^shape / 2). Shape 2 is a Gaussian;
        larger shapes have flatter tops and steeper sides, like a wide vessel seen through a little blur."""
        return np.exp(-0.5 * np.abs(u / sigma) ** shape)


    def geometry_of(vessel):
        """A vessel's geometry at every pixel along it: "z", "centre" (n x 2), "width" (its sigma, n) and "shape". Taken
        from its hyperspline, with the width smoothed over WIDTH_LENGTH, until §22 refines it."""
        if "geometry" in vessel:
            return vessel["geometry"]
        traced, spline = vessel["traced"], vessel["spline"]
        z = np.linspace(traced["z"][0], traced["z"][-1], max(4, int(np.ceil(traced["z"][-1] - traced["z"][0])) + 1))
        values = spline(z)
        smooth, _ = fit_hyperspline(z, np.log(np.maximum(values[:, 6], 1.0)), WIDTH_LENGTH)
        return {"z": z, "centre": values[:, :2], "width": np.exp(smooth(z)[:, 0]), "shape": 2.0}


    def normals_of(centre):
        """The unit normal of a polyline (n x 2) at each of its points."""
        slope = np.gradient(centre, axis=0)
        slope /= np.maximum(np.linalg.norm(slope, axis=1, keepdims=True), 1e-9)
        return np.column_stack([-slope[:, 1], slope[:, 0]])


    def vessel_basis(vessel):
        """One vessel's basis: every pixel within PROFILE_SIGMAS widths of its centreline ("pixels", flat indices), its place
        along the vessel ("z"), the vessel's profile there ("profile": the cross-section, fading out within about a width
        past the vessel's ends), and the weight of each amplitude coefficient there ("design": a cubic B-spline along z,
        to be multiplied by the profile), with the vessel's number of coefficients and knots. Past an end the profile
        fades as a Gaussian of the vessel's width, out to PROFILE_SIGMAS widths."""
        geometry = geometry_of(vessel)
        z_line, centre, sigma, shape = geometry["z"], geometry["centre"], geometry["width"], geometry["shape"]
        normal = normals_of(centre)
        tangent = np.column_stack([normal[:, 1], -normal[:, 0]])
        reach = PROFILE_SIGMAS * sigma.max() + 2
        low = np.maximum(np.floor(centre.min(axis=0) - reach), 0).astype(int)
        high = np.minimum(np.ceil(centre.max(axis=0) + reach) + 1, nf_image.shape[::-1]).astype(int)
        band = np.zeros((high[1] - low[1], high[0] - low[0]), np.uint8)  # drawn on a crop around the vessel
        corners = np.round(centre - low).astype(int)
        for i in range(len(z_line) - 1):
            cv2.line(band, tuple(map(int, corners[i])), tuple(map(int, corners[i + 1])), 1,
                     max(1, int(np.ceil(2 * PROFILE_SIGMAS * sigma[i]))))
        rows, columns = np.nonzero(band)
        rows, columns = rows + low[1], columns + low[0]
        points = np.column_stack([columns, rows]).astype(np.float64)
        _, nearest = scipy.spatial.cKDTree(centre).query(points)
        offset = points - centre[nearest]
        across = np.sum(offset * normal[nearest], axis=1)
        z_raw = z_line[nearest] + np.sum(offset * tangent[nearest], axis=1)
        beyond = np.maximum(0, np.maximum(z_line[0] - z_raw, z_raw - z_line[-1]))  # how far past the vessel's ends
        keep = (np.abs(across) <= PROFILE_SIGMAS * sigma[nearest]) & (beyond <= PROFILE_SIGMAS * sigma[nearest])
        width = sigma[nearest][keep]
        count = max(4, int(np.ceil((z_line[-1] - z_line[0]) / KNOT_SPACING)) + 3)
        knots = np.concatenate([[z_line[0]] * 3, np.linspace(z_line[0], z_line[-1], count - 2), [z_line[-1]] * 3])
        z_kept = np.clip(z_raw[keep], z_line[0], z_line[-1] - 1e-9)
        return {"pixels": rows[keep] * nf_image.shape[1] + columns[keep], "z": z_kept,
                "profile": vessel_profile(across[keep], width, shape) * np.exp(-0.5 * (beyond[keep] / width) ** 2),
                "design": scipy.interpolate.BSpline.design_matrix(z_kept, knots, 3).tocoo(), "count": count, "knots": knots}


    def find_junctions(bases, geometries):
        """Every place where two vessels' cores meet: a connected patch of pixels where both vessels reach, holding a
        pixel where the product of their profiles is at least JUNCTION_CORES. Returns the junctions (their two vessels, the
        patch's pixels where the product is at least 0.01 and the product there, the place of its peak, where along each
        vessel that lies, the angle between them there, and their kind: "X" where both go on through it, "T" where one ends
        there, "V" where both do) and every pair of vessels that touch at all."""
        n = len(bases)
        sizes = [len(basis["pixels"]) for basis in bases]
        pixels = np.concatenate([basis["pixels"] for basis in bases])
        which = np.repeat(np.arange(n), sizes)
        profile = np.concatenate([basis["profile"] for basis in bases])
        z = np.concatenate([basis["z"] for basis in bases])
        entry = scipy.sparse.csr_matrix((np.arange(1, len(pixels) + 1, dtype=np.float64), (pixels, which)),
                                        shape=(nf_image.size, n))  # at every pixel, which vessels reach it (entry + 1)
        per_pixel = np.diff(entry.indptr)
        shared = np.flatnonzero(per_pixel >= 2)
        starts, numbers = entry.indptr[shared], per_pixel[shared]
        first, second, place = [], [], []
        for a in range(numbers.max(initial=0)):
            for b in range(a + 1, numbers.max(initial=0)):
                has = numbers > b
                first.append(entry.data[starts[has] + a].astype(np.int64) - 1)
                second.append(entry.data[starts[has] + b].astype(np.int64) - 1)
                place.append(shared[has])
        if not first:
            return [], set()
        first, second, place = np.concatenate(first), np.concatenate(second), np.concatenate(place)
        swap = which[first] > which[second]
        first[swap], second[swap] = second[swap], first[swap]
        product = profile[first] * profile[second]
        pair = which[first] * n + which[second]
        touching = {(int(key // n), int(key % n)) for key in np.unique(pair)}
        meaningful = product >= 0.01
        first, second, place, product, pair = (values[meaningful] for values in (first, second, place, product, pair))
        order = np.argsort(pair, kind="stable")
        boundaries = np.flatnonzero(np.diff(pair[order])) + 1
        junctions = []
        for group in np.split(order, boundaries):
            if len(group) == 0 or product[group].max() < JUNCTION_CORES:
                continue
            i, l = int(which[first[group[0]]]), int(which[second[group[0]]])
            ys, xs = np.divmod(place[group], nf_image.shape[1])
            patch = np.zeros((np.ptp(ys) + 1, np.ptp(xs) + 1), np.uint8)
            patch[ys - ys.min(), xs - xs.min()] = 1
            count, labels = cv2.connectedComponents(patch, connectivity=8)
            label = labels[ys - ys.min(), xs - xs.min()]
            for number in range(1, count):
                member = group[label == number]
                if product[member].max() < JUNCTION_CORES:
                    continue
                peak = member[np.argmax(product[member])]
                junction = {"vessels": (i, l), "pixels": place[member], "product": product[member],
                            "at": np.array(np.divmod(place[peak], nf_image.shape[1])[::-1], float),
                            "z": (float(z[first[peak]]), float(z[second[peak]]))}
                tangents, widths, ends = [], [], []
                for vessel_number, z_here in zip((i, l), junction["z"]):
                    geometry = geometries[vessel_number]
                    k = int(np.clip(np.searchsorted(geometry["z"], z_here), 1, len(geometry["z"]) - 1))
                    tangents.append(geometry["centre"][k] - geometry["centre"][k - 1])
                    widths.append(geometry["width"][k])
                    ends.append(min(z_here - geometry["z"][0], geometry["z"][-1] - z_here))
                reach = 1.5 * sum(widths) + 4  # an end this close to the peak is an end at the junction
                cosine = abs(tangents[0] @ tangents[1]) / max(np.linalg.norm(tangents[0]) * np.linalg.norm(tangents[1]), 1e-12)
                junction["angle"] = float(np.degrees(np.arccos(np.clip(cosine, 0, 1))))
                junction["kind"] = "XTV"[sum(end <= reach for end in ends)]
                junctions.append(junction)
        return junctions, touching


    def fit_field(vessels, bases=None, junctions=True, target=None):
        """Fit the density (or `target`, a picture in its place) with every vessel, and with every junction's overlap term,
        as one sparse least-squares system (see the markdown above). Returns a dict: "model" (the picture), "amplitudes" (per vessel, a BSpline along z),
        "junctions" (each with its fitted overlap "c" and "share", c over the fainter vessel's amplitude there), "pixels"
        (flat indices of the pixels some vessel reaches), "density" and "residual" there, and "A", "H", "blocks", "vector",
        "bases" and "touching" for the functions below."""
        bases = [vessel_basis(vessel) for vessel in vessels] if bases is None else bases
        found, touching = find_junctions(bases, [geometry_of(vessel) for vessel in vessels]) if junctions else ([], set())
        blocks, rows, columns, values, column, penalties = [], [], [], [], 0, []
        for basis in bases:  # every vessel's amplitude coefficients, one block of columns per vessel
            design, count = basis["design"], basis["count"]
            rows.append(basis["pixels"][design.row])
            columns.append(design.col + column)
            values.append(design.data * basis["profile"][design.row])
            blocks.append((column, count, basis["knots"]))
            seen = len(design.data) / count  # pixels per coefficient: the penalties are relative to the evidence
            second = scipy.sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(count - 2, count))
            change = scipy.sparse.diags([-1.0, 1.0], [0, 1], shape=(count - 1, count))
            penalties.append(scipy.sparse.vstack([np.sqrt(AMPLITUDE_SMOOTHING * seen) * second,
                                                  np.sqrt(AMPLITUDE_STEADINESS * seen) * change]))
            column += count
        for junction in found:  # then one column per junction: minus the product of its two vessels' profiles
            rows.append(junction["pixels"])
            columns.append(np.full(len(junction["pixels"]), column))
            values.append(-junction["product"])
            penalties.append(scipy.sparse.csr_matrix([[1e-3 * np.sqrt(np.sum(junction["product"] ** 2))]]))
            column += 1
        used, row_of = np.unique(np.concatenate(rows), return_inverse=True)
        A = scipy.sparse.csr_matrix((np.concatenate(values), (row_of, np.concatenate(columns))), shape=(len(used), column))
        P = scipy.sparse.block_diag(penalties, format="csr")
        d = (nf_image if target is None else target).ravel()[used].astype(np.float64)
        H = (A.T @ A + P.T @ P).tocsc()
        vector = scipy.sparse.linalg.spsolve(H, A.T @ d)  # the whole field as one system
        # a junction's overlap lies between 0 (the densities add up) and the fainter vessel's amplitude there (the lumens
        # merge): one outside is held at the bound it crossed, and the rest of the system is solved again
        first_junction, free = column - len(found), np.ones(column, bool)
        for _ in range(3):
            amplitudes = [scipy.interpolate.BSpline(knots, vector[first:first + count], 3) for first, count, knots in blocks]
            upper = np.array([max(min(amplitudes[i](junction["z"][0]), amplitudes[l](junction["z"][1])), 0)
                              for junction in found for i, l in [junction["vessels"]]]).reshape(-1)
            overlap = vector[first_junction:]
            held = np.clip(overlap, 0, upper)
            crossed = free[first_junction:] & (np.abs(held - overlap) > 1e-9)
            if not crossed.any():
                break
            vector[first_junction:][crossed] = held[crossed]
            free[first_junction + np.flatnonzero(crossed)] = False
            loose, fixed = np.flatnonzero(free), np.flatnonzero(~free)
            A_columns = A.tocsc()
            right = A_columns[:, loose].T @ (d - A_columns[:, fixed] @ vector[fixed])
            vector[loose] = scipy.sparse.linalg.spsolve(H[loose][:, loose].tocsc(), right)
        amplitudes = [scipy.interpolate.BSpline(knots, vector[first:first + count], 3) for first, count, knots in blocks]
        for number, junction in enumerate(found):
            i, l = junction["vessels"]
            junction["c"] = float(vector[first_junction + number])
            junction["held"] = not free[first_junction + number]
            junction["amplitudes"] = (float(amplitudes[i](junction["z"][0])), float(amplitudes[l](junction["z"][1])))
            junction["share"] = junction["c"] / max(min(junction["amplitudes"]), 1e-3)
            junction["effect"] = max(min(junction["amplitudes"]), 0) * float(junction["product"].max())
        model = np.zeros(nf_image.shape, np.float32)
        model.ravel()[used] = A @ vector
        return {"model": model, "pixels": used, "density": d, "residual": d - A @ vector, "amplitudes": amplitudes,
                "junctions": found, "touching": touching, "A": A.tocsc(), "H": H, "blocks": blocks, "vector": vector,
                "bases": bases}


    def vessel_picture(fit, number):
        """One vessel's part of a joint fit's model (its own amplitude times its profile), as a picture."""
        first, count, _ = fit["blocks"][number]
        picture = np.zeros(nf_image.size, np.float32)
        picture[fit["pixels"]] = fit["A"][:, first:first + count] @ fit["vector"][first:first + count]
        return picture.reshape(nf_image.shape)


    def explained(residual, density):
        """The share of the density's energy that a model explains."""
        return 1 - np.sum(residual ** 2) / np.sum(density ** 2)


    def unique_share(fit):
        """Per vessel, the share of its own model's energy that no other vessel could explain: how much worse the whole
        field's penalised fit would get without it (and its junction terms), every other vessel refitted, b' M^-1 b with b
        its coefficients and M their block of the inverse of the system's matrix, over the energy of its own picture. Also
        per junction, the standard error of its share, for noise of nf_noise per pixel."""
        lu = scipy.sparse.linalg.splu(fit["H"])
        total = fit["H"].shape[0]
        first_junction = total - len(fit["junctions"])
        owned = [list(range(first, first + count)) for first, count, _ in fit["blocks"]]
        for number, junction in enumerate(fit["junctions"]):
            for vessel_number in junction["vessels"]:
                owned[vessel_number].append(first_junction + number)
        gain = np.zeros(len(owned))
        for number, columns in enumerate(owned):
            unit = np.zeros((total, len(columns)))
            unit[columns, np.arange(len(columns))] = 1
            block = lu.solve(unit)[columns]
            coefficients = fit["vector"][columns]
            own = fit["A"][:, columns[0]:columns[0] + fit["blocks"][number][1]] @ coefficients[:fit["blocks"][number][1]]
            gain[number] = coefficients @ np.linalg.solve(block, coefficients) / max(own @ own, 1e-12)
        for number, junction in enumerate(fit["junctions"]):
            unit = np.zeros(total)
            unit[first_junction + number] = 1
            error = nf_noise * np.sqrt(max(lu.solve(unit)[first_junction + number], 0))
            junction["share_error"] = error / max(min(junction["amplitudes"]), 1e-3)
        return gain


    def cut(vessel, z0, z1):
        """The stretch of a vessel from z0 to z1 along it, as a vessel."""
        traced = vessel["traced"]
        keep = (traced["z"] >= z0 - 1e-6) & (traced["z"] <= z1 + 1e-6)
        piece = {key: values[keep] for key, values in traced.items()}
        stretch = {**vessel, "traced": piece, "spline": fit_hyperspline(piece["z"], samples_of(piece))[0]}
        if "geometry" in vessel:
            geometry = vessel["geometry"]
            along = (geometry["z"] >= z0 - 1e-6) & (geometry["z"] <= z1 + 1e-6)
            stretch["geometry"] = {key: (values[along] if isinstance(values, np.ndarray) and len(values) == len(along)
                                         else values) for key, values in geometry.items()}
        return stretch


    def cut_faint(vessels, fit):
        """Cut every vessel back to where it is at least CUT_SHARE of its typical amplitude (the 75th percentile along it):
        faint stretches at its ends go, and so does a faint stretch in its middle longer than two knots, which splits it.
        Pieces shorter than MIN_VESSEL go too."""
        pieces = []
        for number, vessel in enumerate(vessels):
            z = geometry_of(vessel)["z"]
            amplitude = fit["amplitudes"][number](z)
            bright = amplitude >= CUT_SHARE * max(np.percentile(amplitude, 75), 1e-6)
            edges = np.flatnonzero(np.diff(np.concatenate([[0], bright.astype(int), [0]])))
            runs = [[start, stop] for start, stop in zip(edges[::2], edges[1::2])]
            merged = runs[:1]
            for start, stop in runs[1:]:  # a short faint stretch between two bright ones is kept
                if z[start] - z[merged[-1][1] - 1] <= 2 * KNOT_SPACING:
                    merged[-1][1] = stop
                else:
                    merged.append([start, stop])
            for start, stop in merged:
                if z[stop - 1] - z[start] < MIN_VESSEL:
                    continue
                pieces.append(vessel if start == 0 and stop == len(z) else cut(vessel, z[start], z[stop - 1]))
        return pieces


    def settle(vessels, target=None):
        """Fit the whole field with its junctions, cut the vessels back to where they are bright, fit again, then drop the
        vessels that explain less than UNIQUE_SHARE of their own picture that no other vessel could, a few at a time,
        refitting after every batch. Returns the vessels kept, their fit, and how many were cut and dropped."""
        before = len(vessels)
        vessels = cut_faint(vessels, fit_field(vessels, target=target))
        after_cutting = len(vessels)
        fit = fit_field(vessels, target=target)
        dropped = 0
        while True:
            score = unique_share(fit)
            batch, neighbours = [], {}
            for i, l in fit["touching"]:
                neighbours.setdefault(i, set()).add(l)
                neighbours.setdefault(l, set()).add(i)
            for number in np.argsort(score):
                if score[number] >= UNIQUE_SHARE:
                    break
                if not neighbours.get(number, set()) & set(batch):  # of two touching weak vessels, the stronger waits
                    batch.append(int(number))
            if not batch:
                break
            kept = [number for number in range(len(vessels)) if number not in set(batch)]
            vessels = [vessels[number] for number in kept]
            fit = fit_field(vessels, [fit["bases"][number] for number in kept], target=target)
            dropped += len(batch)
        fit["score"] = score
        return vessels, fit, {"before": before, "after cutting": after_cutting, "dropped": dropped}


    def junction_table(fit):
        """Print the junctions by kind (X, T, V) and by what the fit makes of them: crossing (the two vessels' densities
        add up, share below 0.5) or joining (their lumens merge, share 0.5 or more). It is unclear where merging could
        change the model by less than 5 noise levels, or where the share's standard error is above 0.25."""
        rows = {}
        for junction in fit["junctions"]:
            clear = junction.get("share_error", 0) <= 0.25 and junction["effect"] >= 5 * nf_noise
            verdict = ("joining" if junction["share"] >= 0.5 else "crossing") if clear else "unclear"
            junction["verdict"] = verdict
            rows.setdefault(junction["kind"], {}).setdefault(verdict, 0)
            rows[junction["kind"]][verdict] += 1
        print(f"   {'kind':>5s} {'crossing':>9s} {'joining':>8s} {'unclear':>8s}")
        for kind, name in (("X", "X"), ("T", "T"), ("V", "V")):
            counts = rows.get(kind, {})
            print(f"   {name:>5s} {counts.get('crossing', 0):9d} {counts.get('joining', 0):8d} {counts.get('unclear', 0):8d}")


    first_fit = fit_field(traced_vessels, junctions=False)
    print(f"{len(traced_vessels)} traced vessels, fitted as one system without junction terms: "
          f"{explained(first_fit['residual'], first_fit['density']) * 100:.1f}% of the density explained")
    settled_vessels, settled_fit, settled_counts = settle(traced_vessels)
    print(f"settled: {len(traced_vessels)} vessels, {settled_counts['after cutting']} pieces after cutting the faint stretches, "
          f"{len(settled_vessels)} after dropping {settled_counts['dropped']} that explained too little of their own; "
          f"{explained(settled_fit['residual'], settled_fit['density']) * 100:.1f}% explained, with "
          f"{len(settled_fit['junctions'])} junctions")
    junction_table(settled_fit)
    return (
        AMPLITUDE_SMOOTHING,
        AMPLITUDE_STEADINESS,
        KNOT_SPACING,
        PROFILE_SIGMAS,
        RESIDUAL_RANGE,
        explained,
        fit_field,
        geometry_of,
        junction_table,
        normals_of,
        settle,
        settled_fit,
        settled_vessels,
        unique_share,
        vessel_basis,
        vessel_picture,
        vessel_profile,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 21 · Continuations: joining, snapping and bridging the ends

    Many vessels end where the picture shows them going on. In a complex junction the orientation or the scale is ambiguous for a stretch, and the walk loses its line there. So every end tries to continue, and the vessels it connects become one.

    **The end's properties.** Each end has an outward direction (the centreline's direction at the end). Over its last `MIN_VESSEL` px it also has an average width, amplitude in §20's fit, and profile shape.

    **1. Continue from the end.** From each end, largest vessels first, a walk steps outward. It reads the density with every *other* vessel of the fit taken out, so that neither a crossing vessel nor a neighbour's tail can pass for this one. At each step it looks up to `TRACE_REACH` px to either side for the vessel's own cross-section, at the vessel's own width and shape. The step counts where two things hold:

    - the cross-section's amplitude is at least `EXTEND_SHARE` of the end's;
    - its shape correlates with the vessel's by at least `EXTEND_MATCH`.

    The walk turns towards what it found, by at most `TRACE_BEND` degrees per pixel. It ends after two steps in a row that do not count, at the data's edge, or after `EXTEND_MOST` px. Where it runs into another vessel's core, there are two cases:

    - **joined:** it has reached one of that vessel's ends, facing it within `FACING` degrees. The two vessels become one, through the walk.
    - **snapped:** it has met the other vessel's side, a junction. It is carried on to that vessel's centreline, to the point nearest straight ahead, within `FACING` of its heading. The branch then ends on its parent's centreline, so §20's junction term sees the whole overlap. A walk with no point of the centreline ahead ran alongside the other vessel rather than into it, and is not snapped.

    **2. Join through a crossing.** Two ends may snap onto one vessel at the same place from either side, facing each other within `FACING` degrees, with about the same width, both arriving at least 30° across it. That is one vessel passing under (or over) the other, cut in two by the walk's end rule. The two ends are joined through the crossing.

    **3. Bridge what is left.** An end that is still free may face a real gap, where the vessel cannot be seen for a stretch. A bridge is tried to every other free end within `BRIDGE_REACH` px that is roughly straight ahead, faces it within `FACING` degrees, and has about the same width and amplitude. The bridge is a cubic that keeps both ends' directions, with width and amplitude interpolated along it. It is compared with the density without every other vessel. It passes when the gap holds between `BRIDGE_SHARE` and its inverse of the bridge's density, and the bridge explains at least `BRIDGE_GAIN` of the energy around it.

    **4. Merge and settle.** Each end joins at most one other, and no join closes a loop. The vessels that joins, crossings and bridges connect become one vessel whose walk runs through them, with a new hyperspline. A continuation that joined nothing lengthens its own vessel. The merged vessels are then settled (§20): fitted with their junctions, cut back where faint, and thinned of the redundant.

    | Output | Content |
    |---|---|
    | `extensions`, `snapped`, `joins`, `crossings_joined`, `bridges` | the continuations of each kind, with their points |
    | `merged_vessels` | the vessels after merging |
    | `continued_vessels`, `continued_fit` | after settling |
    """)
    return


@app.cell
def _(
    CORE_WIDTHS,
    DENSITY_RANGE,
    MIN_VESSEL,
    NF_ORIENTATIONS,
    SERIES,
    TRACE_BEND,
    TRACE_REACH,
    TRACE_STEP,
    banded_key,
    bilinear,
    cv2,
    explained,
    fit_hyperspline,
    geometry_of,
    inside,
    junction_table,
    nf_image,
    normals_of,
    np,
    plt,
    samples_of,
    save_figure,
    scipy,
    settle,
    settled_fit,
    settled_vessels,
    vessel_picture,
    vessel_profile,
):
    EXTEND_SHARE = 0.4    # a step of a continuation counts where the cross-section has this share of the end's amplitude...
    EXTEND_MATCH = 0.5    # ...and matches the vessel's shape with at least this correlation
    EXTEND_MOST = 200.0   # pixels a continuation may run at the most
    FACING = 45.0         # degrees; two ends that are joined must face each other within this
    BRIDGE_REACH = 60.0   # pixels; the longest gap a bridge may span
    BRIDGE_SHARE = 0.35   # a bridge's gap must hold between this share and its inverse of the bridge's interpolated density...
    BRIDGE_GAIN = 0.2     # ...and the bridge must explain at least this share of the energy around it


    def vessel_ends(number, vessel, fit):
        """Both ends of a vessel: where it ends, its outward direction there, and its average width, amplitude (in the
        joint fit), response and shape over its last MIN_VESSEL px."""
        geometry = geometry_of(vessel)
        z, centre, width = geometry["z"], geometry["centre"], geometry["width"]
        tangent = np.column_stack([normals_of(centre)[:, 1], -normals_of(centre)[:, 0]])
        amplitude = fit["amplitudes"][number](z)
        traced = vessel["traced"]
        ends = []
        for side in (0, 1):
            span = z <= z[0] + MIN_VESSEL if side == 0 else z >= z[-1] - MIN_VESSEL
            walked = traced["z"] <= traced["z"][0] + MIN_VESSEL if side == 0 else traced["z"] >= traced["z"][-1] - MIN_VESSEL
            ends.append({"vessel": number, "side": side, "position": centre[0 if side == 0 else -1],
                         "outward": -tangent[0] if side == 0 else tangent[-1], "width": width[span].mean(),
                         "amplitude": max(amplitude[span].mean(), 0.02), "response": traced["response"][walked].mean(),
                         "shape": geometry["shape"]})
        return ends


    def cores_picture(vessels):
        """At every pixel, the number of the vessel whose core (CORE_WIDTHS widths to either side) covers it, -1 elsewhere."""
        labels = np.full(nf_image.shape, -1, np.float32)
        for number, vessel in enumerate(vessels):
            geometry = geometry_of(vessel)
            corners = np.round(geometry["centre"]).astype(int)
            for i in range(len(corners) - 1):
                cv2.line(labels, tuple(map(int, corners[i])), tuple(map(int, corners[i + 1])), float(number),
                         max(1, int(round(2 * CORE_WIDTHS * geometry["width"][i]))))
        return labels.astype(np.int32)


    def extend(end, picture, labels, ends_of):
        """Continue a vessel past one of its ends on `picture` (the density without every other vessel), finding its own
        cross-section at each step. Returns the points it passed and how it ended: ("joins", end index) where it reached
        another vessel's end facing it, ("meets", vessel number) where it reached another vessel's core, or (None, None)."""
        x, y = end["position"]
        heading = end["outward"] / np.linalg.norm(end["outward"])
        width, amplitude, shape = end["width"], end["amplitude"], end["shape"]
        u = np.arange(-2.5 * width, 2.5 * width + 0.25, 0.5)
        template = vessel_profile(u, width, shape)
        centred = template - template.mean()
        offsets = np.linspace(-TRACE_REACH, TRACE_REACH, 13)
        largest_turn = np.radians(TRACE_BEND * TRACE_STEP)
        points, misses, travelled = [], 0, 0.0
        while travelled < EXTEND_MOST:
            ahead = np.array([x, y]) + TRACE_STEP * heading
            travelled += TRACE_STEP
            if not inside(*ahead):
                break
            normal = np.array([-heading[1], heading[0]])
            centres = ahead + offsets[:, None] * normal
            profiles = bilinear(picture, centres[:, 0, None] + u * normal[0], centres[:, 1, None] + u * normal[1])
            strength = profiles @ template / (template @ template)
            varying = profiles - profiles.mean(axis=1, keepdims=True)
            match = varying @ centred / np.maximum(np.linalg.norm(varying, axis=1) * np.linalg.norm(centred), 1e-12)
            candidate = np.where(match >= EXTEND_MATCH, strength, -np.inf)
            best = int(np.argmax(candidate))
            if candidate[best] >= EXTEND_SHARE * amplitude:
                found = centres[best]
                turn = np.angle(np.exp(1j * (np.arctan2(*(found - [x, y])[::-1]) - np.arctan2(heading[1], heading[0]))))
                angle = np.arctan2(heading[1], heading[0]) + np.clip(turn, -largest_turn, largest_turn)
                heading = np.array([np.cos(angle), np.sin(angle)])
                x, y = found
                misses = 0
            else:
                x, y = ahead
                misses += 1
                if misses >= 2:
                    break
            points.append((x, y))
            label = labels[int(round(y)), int(round(x))]
            if label >= 0 and label != end["vessel"] and misses == 0:  # it has reached another vessel
                for other in ends_of[label]:
                    gap = np.hypot(*(ends[other]["position"] - [x, y]))
                    if gap <= max(8.0, 2 * width) and -heading @ ends[other]["outward"] >= np.cos(np.radians(FACING)):
                        return np.array(points), ("joins", other)
                return np.array(points), ("meets", int(label))
        points = points[:len(points) - misses]  # the steps that did not count
        return np.array(points).reshape(-1, 2), (None, None)


    def onto_centreline(points, end, vessel):
        """A continuation that met another vessel's core, carried on to that vessel's centreline: to the point of it nearest
        to straight ahead, within FACING of the continuation's heading and within reach of the core. Returns the points,
        where along the other vessel (z) it arrives and the angle at which it arrives (degrees, 0 to 90), or None where no
        point of the centreline lies ahead: the continuation ran alongside the other vessel rather than into it."""
        geometry = geometry_of(vessel)
        points = points.reshape(-1, 2)
        tip = points[-1] if len(points) else end["position"]
        heading = (points[-1] - points[-2]) if len(points) >= 2 else end["outward"]
        heading = heading / max(np.linalg.norm(heading), 1e-9)
        reach = (CORE_WIDTHS + 1) * geometry["width"].max() + 2 * TRACE_STEP
        near = scipy.spatial.cKDTree(geometry["centre"]).query_ball_point(tip, reach)
        if not near:
            return None
        near = np.array(near)
        towards = geometry["centre"][near] - tip
        distance = np.maximum(np.linalg.norm(towards, axis=1), 1e-9)
        off_heading = np.degrees(np.arccos(np.clip(towards @ heading / distance, -1, 1)))
        if off_heading.min() > FACING:
            return None
        nearest = near[np.argmin(off_heading + 90 * distance / reach)]  # nearest to straight ahead, then nearest
        target = geometry["centre"][nearest]
        tangent = normals_of(geometry["centre"])[nearest][::-1] * [1, -1]
        arrival = np.degrees(np.arccos(np.clip(abs(tangent @ heading), 0, 1)))
        steps = np.linspace(0, 1, max(2, int(np.ceil(np.linalg.norm(target - tip) / TRACE_STEP)) + 1))[1:, None]
        return np.vstack([points, tip + steps * (target - tip)]), float(geometry["z"][nearest]), float(arrival)


    def hermite(a, b, step=1.0):
        """The bridge from end a to end b: a cubic that leaves a along its outward direction and arrives at b against b's.
        Returns its points, its tangent angles and its parameter t from 0 to 1."""
        p0, p1 = a["position"], b["position"]
        gap = np.linalg.norm(p1 - p0)
        m0, m1 = a["outward"] * gap, -b["outward"] * gap
        t = np.linspace(0, 1, max(3, int(np.ceil(gap / step))) + 1)[:, None]
        points = (2 * t**3 - 3 * t**2 + 1) * p0 + (t**3 - 2 * t**2 + t) * m0 + (-2 * t**3 + 3 * t**2) * p1 + (t**3 - t**2) * m1
        slope = (6 * t**2 - 6 * t) * p0 + (3 * t**2 - 4 * t + 1) * m0 + (6 * t - 6 * t**2) * p1 + (3 * t**2 - 2 * t) * m1
        return points, np.arctan2(slope[:, 1], slope[:, 0]), t[:, 0]


    def bridge_test(a, b, picture):
        """Whether a bridge from a to b explains the density of `picture` (without every other vessel) in its gap: its
        interpolated vessel is scaled to the picture in its band (the share), and the share of the band's energy it then
        removes is the gain. The stretch next to either end is left out. Returns share, gain and the bridge's points."""
        points, angle, t = hermite(a, b)
        width = (1 - t) * a["width"] + t * b["width"]
        amplitude = (1 - t) * a["amplitude"] + t * b["amplitude"]
        band = np.zeros(nf_image.shape, np.uint8)
        corners = np.round(points).astype(int)
        for i in range(len(points) - 1):
            cv2.line(band, tuple(map(int, corners[i])), tuple(map(int, corners[i + 1])), 1, max(1, int(np.ceil(5 * width[i]))))
        rows, columns = np.nonzero(band)
        pixel_points = np.column_stack([columns, rows]).astype(np.float64)
        _, nearest = scipy.spatial.cKDTree(points).query(pixel_points)
        offset = pixel_points - points[nearest]
        across = -offset[:, 0] * np.sin(angle[nearest]) + offset[:, 1] * np.cos(angle[nearest])
        arc = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(points, axis=0).T))])
        margin = np.minimum(width[nearest], arc[-1] / 4)
        keep = (arc[nearest] >= margin) & (arc[-1] - arc[nearest] >= margin) & (np.abs(across) <= 2.5 * width[nearest])
        if keep.sum() < 20:  # ends that nearly touch: nothing between them to test
            return np.inf, 1.0, points
        predicted = amplitude[nearest][keep] * vessel_profile(across[keep], width[nearest][keep], a["shape"])
        seen = picture[rows[keep], columns[keep]]
        share = float(seen @ predicted / (predicted @ predicted))
        gain = 1 - np.sum((seen - max(share, 0) * predicted) ** 2) / max(np.sum(seen ** 2), 1e-12)
        return share, float(gain), points


    def path_samples(points, width_start, width_end, response):
        """Samples along a stretch the walker did not take (a continuation or a bridge), as a walk would give them: the
        width interpolated as the blur (kernel size 0), the density read from the picture."""
        points = np.asarray(points, np.float64).reshape(-1, 2)
        if len(points) == 0:
            return None
        angle = np.arctan2(*np.gradient(points, axis=0)[:, ::-1].T) if len(points) > 1 else np.zeros(1)
        t = np.linspace(0, 1, len(points))
        return {"x": points[:, 0], "y": points[:, 1], "orientation": angle % np.pi,
                "neuron": np.round(angle % np.pi * NF_ORIENTATIONS / np.pi).astype(int) % NF_ORIENTATIONS,
                "response": np.full(len(points), response), "blur": (1 - t) * width_start + t * width_end,
                "kernel_size": np.zeros(len(points)), "density": bilinear(nf_image, points[:, 0], points[:, 1])}


    angle_between = lambda u, v: np.degrees(np.arccos(np.clip(u @ v, -1, 1)))
    _vessels, _fit = settled_vessels, settled_fit
    ends = [end for _number, _vessel in enumerate(_vessels) for end in vessel_ends(_number, _vessel, _fit)]
    ends_of = {_number: (2 * _number, 2 * _number + 1) for _number in range(len(_vessels))}
    _labels = cores_picture(_vessels)
    residual_picture = nf_image - _fit["model"]
    parent = list(range(len(_vessels)))


    def root(number):
        """The chain a vessel belongs to (union-find)."""
        while parent[number] != number:
            parent[number] = parent[parent[number]]
            number = parent[number]
        return number


    def link_ends(i, j, points):
        """Join end i to end j through `points` (from i to j): their vessels become one chain."""
        parent[root(ends[i]["vessel"])] = root(ends[j]["vessel"])
        link[i], link[j] = j, i
        connection[i, j], connection[j, i] = points, points[::-1]


    # 1. continue every end from the edge of its vessel, largest vessels first: into another end (a join), onto another
    # vessel's centreline (a junction), or until the vessel's own cross-section is lost
    link, connection, extensions, joins, snapped = {}, {}, {}, [], {}
    for _number in np.argsort([-np.mean(geometry_of(_vessel)["width"]) for _vessel in _vessels]):
        _picture = residual_picture + vessel_picture(_fit, _number)  # the density without every other vessel
        for end_index in ends_of[_number]:
            if end_index in link:
                continue
            points, (outcome, target) = extend(ends[end_index], _picture, _labels, ends_of)
            if outcome == "joins" and target not in link and root(_number) != root(ends[target]["vessel"]):
                link_ends(end_index, target, points)
                joins.append({"ends": (end_index, target), "points": points})
            elif outcome == "meets" and (arrived := onto_centreline(points, ends[end_index], _vessels[target])) is not None:
                points, z_there, arrival = arrived
                snapped[end_index] = (target, z_there, arrival)
                extensions[end_index] = points
            elif len(points) >= 2:
                extensions[end_index] = points

    # 2. two ends that arrive at the same place on one vessel from either side, facing each other with about the same
    # width, both arriving across it (at least 30 degrees from its direction): one vessel passing under (or over) the
    # other. They are joined through it, a crossing.
    crossings_joined = []
    arrivals = sorted(snapped.items(), key=lambda item: item[1])
    for _place, (i, (target_i, z_i, arrival_i)) in enumerate(arrivals):
        for j, (target_j, z_j, arrival_j) in arrivals[_place + 1:]:
            if target_j != target_i or z_j - z_i > 2 * (ends[i]["width"] + ends[j]["width"]) + 4:
                break
            a, b = ends[i], ends[j]
            if (i in link or j in link or a["vessel"] == b["vessel"] or root(a["vessel"]) == root(b["vessel"])
                    or min(arrival_i, arrival_j) < 30
                    or angle_between(a["outward"], -b["outward"]) > FACING
                    or max(a["width"] / b["width"], b["width"] / a["width"]) > 1.6):
                continue
            through = np.vstack([extensions.pop(i), extensions.pop(j)[::-1]])
            link_ends(i, j, through)
            crossings_joined.append({"ends": (i, j), "points": through, "under": target_i})
            del snapped[i], snapped[j]


    def tip_of(i):
        """An end as it stands after its continuation: at the continuation's tip, heading the way the walk last went."""
        end = dict(ends[i])
        if i in extensions:
            points = extensions[i]
            end["position"] = points[-1]
            if len(points) >= 2:
                end["outward"] = (points[-1] - points[-2]) / max(np.linalg.norm(points[-1] - points[-2]), 1e-9)
        return end


    # 3. bridge the ends that are still free, from their continuations' tips, against the density without other vessels
    free = [i for i in range(len(ends)) if i not in link and i not in snapped]
    tree = scipy.spatial.cKDTree(np.array([tip_of(i)["position"] for i in free]).reshape(-1, 2))
    candidates = []
    for _place, i in enumerate(free):
        a = tip_of(i)
        for other_place in tree.query_ball_point(a["position"], BRIDGE_REACH):
            j = free[other_place]
            b = tip_of(j)
            if j <= i or b["vessel"] == a["vessel"]:
                continue
            gap_vector = b["position"] - a["position"]
            gap = np.linalg.norm(gap_vector)
            if gap > 1e-6 and (angle_between(a["outward"], gap_vector / gap) > 2 * FACING / 3
                               or angle_between(b["outward"], -gap_vector / gap) > 2 * FACING / 3):
                continue
            if (angle_between(a["outward"], -b["outward"]) > FACING
                    or max(a["width"] / b["width"], b["width"] / a["width"]) > 1.6
                    or max(a["amplitude"] / b["amplitude"], b["amplitude"] / a["amplitude"]) > 1.6):
                continue
            _picture = residual_picture + vessel_picture(_fit, a["vessel"]) + vessel_picture(_fit, b["vessel"])
            _share, _gain, points = bridge_test(a, b, _picture)
            passes = np.isinf(_share) or (BRIDGE_SHARE <= _share <= 1 / BRIDGE_SHARE and _gain >= BRIDGE_GAIN)
            candidates.append({"ends": (i, j), "share": _share, "gain": _gain, "points": points, "gap": gap, "passes": passes})
    bridges = []
    for _candidate in sorted(candidates, key=lambda candidate: -candidate["gain"]):
        i, j = _candidate["ends"]
        if not _candidate["passes"] or i in link or j in link or root(ends[i]["vessel"]) == root(ends[j]["vessel"]):
            continue
        nothing = np.zeros((0, 2))
        link_ends(i, j, np.vstack([extensions.pop(i, nothing), _candidate["points"], extensions.pop(j, nothing)[::-1]]))
        bridges.append(_candidate)

    # 4. merge: every chain of joined vessels becomes one vessel, lengthened by its free ends' continuations
    KEYS = ("x", "y", "orientation", "neuron", "response", "blur", "kernel_size", "density")
    merged_vessels, visited = [], set()
    for _number in range(len(_vessels)):
        if _number in visited:
            continue
        free_sides = [side for side in (0, 1) if 2 * _number + side not in link]
        if not free_sides:
            continue  # the middle of a chain: reached from one of its ends
        chain, _entry = [], (_number, free_sides[0])
        while True:
            vessel_number, side = _entry
            visited.add(vessel_number)
            chain.append(_entry)
            exit_end = 2 * vessel_number + 1 - side
            if exit_end not in link:
                break
            _entry = (link[exit_end] // 2, link[exit_end] % 2)
        first_end, last_end = 2 * chain[0][0] + chain[0][1], 2 * chain[-1][0] + 1 - chain[-1][1]
        if len(chain) == 1 and first_end not in extensions and last_end not in extensions:
            merged_vessels.append(_vessels[_number])
            continue
        pieces = []
        if first_end in extensions:  # the continuation at the free start, walked inwards
            pieces.append(path_samples(extensions[first_end][::-1], ends[first_end]["width"], ends[first_end]["width"],
                                       ends[first_end]["response"]))
        for _place, (vessel_number, side) in enumerate(chain):
            _traced = _vessels[vessel_number]["traced"]
            order = slice(None) if side == 0 else slice(None, None, -1)
            pieces.append({key: _traced[key][order] for key in KEYS})
            if _place + 1 < len(chain):
                this_end = 2 * vessel_number + 1 - side
                next_end = 2 * chain[_place + 1][0] + chain[_place + 1][1]
                pieces.append(path_samples(connection[this_end, next_end], ends[this_end]["width"], ends[next_end]["width"],
                                           0.5 * (ends[this_end]["response"] + ends[next_end]["response"])))
        if last_end in extensions:  # the continuation at the free end
            pieces.append(path_samples(extensions[last_end], ends[last_end]["width"], ends[last_end]["width"],
                                       ends[last_end]["response"]))
        walk_through = {key: np.concatenate([piece[key] for piece in pieces if piece is not None]) for key in KEYS}
        step = np.hypot(np.diff(walk_through["x"]), np.diff(walk_through["y"]))
        _keep = np.concatenate([[True], step > 0.1])  # no two samples at one place
        walk_through = {key: values[_keep] for key, values in walk_through.items()}
        walk_through["z"] = np.concatenate([[0], np.cumsum(np.hypot(np.diff(walk_through["x"]), np.diff(walk_through["y"])))])
        merged_vessels.append({"band": min(_vessels[vessel_number]["band"] for vessel_number, _ in chain),
                               "traced": walk_through, "spline": fit_hyperspline(walk_through["z"], samples_of(walk_through))[0],
                               "pieces": len(chain)})

    print(f"{len(ends)} vessel ends: {len(extensions) + 2 * len(crossings_joined)} continued, {len(snapped)} of them onto "
          f"another vessel's centreline (a junction), {len(joins)} pairs joined end to end, {len(crossings_joined)} pairs joined "
          f"through a vessel they cross, {len(bridges)} gaps bridged ({len(candidates)} tested)")
    print(f"{len(_vessels)} vessels -> {len(merged_vessels)} after joining "
          f"({sum(vessel.get('pieces', 1) > 1 for vessel in merged_vessels)} of them from "
          f"{sum(vessel.get('pieces', 1) for vessel in merged_vessels if vessel.get('pieces', 1) > 1)} pieces)")
    continued_vessels, continued_fit, continued_counts = settle(merged_vessels)
    print(f"settled: {continued_counts['after cutting']} pieces after cutting the faint stretches, {len(continued_vessels)} "
          f"after dropping {continued_counts['dropped']}; {explained(continued_fit['residual'], continued_fit['density']) * 100:.1f}% "
          f"explained (before continuing: {explained(_fit['residual'], _fit['density']) * 100:.1f}%), with "
          f"{len(continued_fit['junctions'])} junctions")
    junction_table(continued_fit)

    # the whole field: the settled vessels, their continuations, joins, crossings and bridges
    _figure, _axis = plt.subplots(figsize=(16, 10.4))
    _axis.imshow(nf_image, cmap="gray", vmin=DENSITY_RANGE[0], vmax=DENSITY_RANGE[1])
    for _vessel in _vessels:
        _centre = geometry_of(_vessel)["centre"]
        _axis.plot(_centre[:, 0], _centre[:, 1], color=SERIES[0], linewidth=1.2, alpha=0.8)
    layers = (([points for i, points in extensions.items() if i not in snapped], SERIES[2], 2.5, "continued from an end"),
              ([extensions[i] for i in snapped], SERIES[6], 2.5, "continued onto another vessel: a junction"),
              ([join["points"] for join in joins], SERIES[3], 3, "continued into another end: joined"),
              ([item["points"] for item in crossings_joined], SERIES[7], 3, "joined through a vessel it crosses"),
              ([bridge["points"] for bridge in bridges], SERIES[4], 3, "bridged"))
    for paths, color, _width, label in layers:
        for _number, points in enumerate(paths):
            _axis.plot(points[:, 0], points[:, 1], color=color, linewidth=_width, label=label if _number == 0 else None)
    _axis.plot([], [], color=SERIES[0], linewidth=1.2, label="vessels of §20")
    _axis.set_title(f"{len(_vessels)} vessels, continued and joined: {len(merged_vessels)}; settled: {len(continued_vessels)}",
                   fontsize=10)
    _axis.legend(loc="lower left", fontsize=9, frameon=True, facecolor="black", edgecolor="none", framealpha=0.7,
                labelcolor="white")
    _axis.axis("off")
    _figure.tight_layout()
    save_figure(_figure, "continued_vessels", banded_key)
    plt.show()
    return continued_fit, continued_vessels, cores_picture


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 22 · Refining each vessel's geometry against the others

    So far a vessel's centreline and width came from its walk, smoothed by its hyperspline. The walker's samples jump between pixels and between the neurons' few blurs and sizes; the geometry has never been fitted to the density itself. Here every vessel's geometry is fitted to the density, with every other vessel taken into account, and **sudden changes are penalised**. A vessel does not suddenly get wider, narrower, brighter or fainter, nor step sideways, so a fit that makes it do so must pay for it.

    **Each vessel, against the others.** For one vessel, the picture is the density minus every *other* vessel's part of the current model, junction terms included. Every 2 px along the vessel, its cross-section is read along the normal of its walk's centreline. The vessel is described by three cubic splines along it, with a knot every `KNOT_SPACING` px: its centre's shift $s(z)$ from the walk, its log width $\ell(z) = \ln\sigma(z)$, and its amplitude $a(z)$. Penalised Gauss–Newton (with Levenberg–Marquardt damping) fits them together, minimizing

    $$
    \sum_{\text{cross-sections}} w_i \sum_u \Bigl(P_i(u) - a\, e^{-\frac12 |(u - s)/\sigma|^{p}}\Bigr)^2
    \;+\; \lambda_s \lVert D^2 s\rVert^2 + \rho \lVert s \rVert^2
    \;+\; \sum_{q \in \{\ell, a\}} \bigl(\mu_q \lVert D q\rVert^2 + \lambda_q \lVert D^2 q\rVert^2\bigr).
    $$

    Here $D$ and $D^2$ are the first and second differences of the spline coefficients along the vessel.

    - **Width and amplitude:** $\mu_q$ (`STEADINESS`) penalises any change, $\lambda_q$ (`STIFFNESS`) any bending.
    - **Shift:** $\lambda_s$ penalises bending, and $\rho$ (`LEASH`) holds the centre to the walk. Whatever the penalties, no centre ends up more than `LEASH_MOST` widths (at least 2 px) from its walk.
    - **Scale:** each penalty is scaled to the data term of its own kind of coefficient, so the settings are stiffnesses relative to the evidence.
    - **Baseline:** every cross-section gets its own straight-line baseline across the vessel, an offset and a slope, solved away in closed form and never rendered. Haze around a vessel, often more on one side, is then not explained by widening it or by shifting it towards the brighter side.
    - **Crossings:** where another vessel's core crosses a cross-section, its weight $w_i$ is `CROSSING_WEIGHT`, and the width and brightness are carried through from either side.
    - **Shape:** the profile shape $p$ is chosen once per vessel, from `REFINE_SHAPES`, as the one that fits its cross-sections best. 2 is a Gaussian; larger values have flatter tops, as a wide vessel seen through a little blur does.
    - **Faint vessels:** a vessel fainter than 10 noise levels keeps its walk's geometry, because its cross-sections are too noisy to place it better.

    **One vessel at a time, largest first.** Each vessel sees the others' latest models. Its own new model, its amplitude fitted alone with §20's penalties, is written back into the field's model before the next vessel looks (Gauss–Seidel). If every vessel looked at the same leftover at once, neighbours would all widen to soak up the same unexplained density. Then the field is settled once more (§20): every amplitude and junction fitted jointly, faint stretches cut, redundant vessels dropped.

    | Output | Content |
    |---|---|
    | `refined_vessels` | the vessels with their refined geometry (`vessel["geometry"]`: centre, width, shape and shift along z) |
    | `final_vessels`, `final_fit` | after settling: the pipeline's vessels and the joint fit of the field with them |
    """)
    return


@app.cell
def _(
    AMPLITUDE_SMOOTHING,
    AMPLITUDE_STEADINESS,
    KNOT_SPACING,
    RESIDUAL_RANGE,
    banded_key,
    bilinear,
    continued_fit,
    continued_vessels,
    cores_picture,
    explained,
    geometry_of,
    junction_table,
    nf_image,
    nf_noise,
    normals_of,
    np,
    plt,
    save_figure,
    scipy,
    settle,
    vessel_basis,
    vessel_profile,
):
    STIFFNESS = {"shift": 10.0, "width": 100.0, "amplitude": 10.0}  # resistance to bending along the vessel
    STEADINESS = {"width": 100.0, "amplitude": 10.0}                # resistance to any change along the vessel
    LEASH = 0.05                         # how strongly a centre is held to where its walk put it
    LEASH_MOST = 0.5                     # widths (at least 2 px): no centre ends up further than this from its walk
    CROSSING_WEIGHT = 0.1                # how much a cross-section crossed by another vessel's core counts
    REFINE_SHAPES = (2.0, 3.0, 4.0, 6.0)  # profile shapes tried: 2 is a Gaussian, larger is flatter on top


    def without_baseline(u):
        """A function that takes away, from every row of an array over u, its best straight line in u (offset and slope):
        the cross-section's own baseline."""
        basis, _ = np.linalg.qr(np.column_stack([np.ones_like(u), u]))
        return lambda values: values - (values @ basis) @ basis.T


    def choose_shape(profiles, u, sigma):
        """The profile shape of REFINE_SHAPES that fits a vessel's cross-sections best, each with its best shift (up to 3 px),
        width change (up to 25%) and amplitude, every cross-section and template without its own baseline."""
        flatten = without_baseline(u)
        profiles = flatten(profiles)
        energy = np.sum(profiles ** 2, axis=1)
        shifts = np.arange(-3.0, 3.01, 0.5)
        misfit_of = {}
        for shape in REFINE_SHAPES:
            best = np.full(len(profiles), np.inf)
            for change in (0.8, 0.9, 1.0, 1.12, 1.25):
                width = change * sigma[:, None, None]
                distance = u[None, None, :] - shifts[None, :, None]
                template = flatten(np.where(np.abs(distance) <= 3 * width, vessel_profile(distance, width, shape), 0))
                projection = np.einsum("mu,msu->ms", profiles, template)
                misfit = energy[:, None] - projection ** 2 / np.maximum(np.sum(template ** 2, axis=2), 1e-12)
                best = np.minimum(best, misfit.min(axis=1))
            misfit_of[shape] = best.sum()
        return min(misfit_of, key=misfit_of.get)


    def fit_vessel_geometry(walked, start, picture, crossed):
        """Fit one vessel's shift from its walk, log width and amplitude, three cubic splines along it, to `picture` (the
        density without every other vessel) by penalised Gauss-Newton (see the markdown). `walked` is the walk's geometry
        (the frame the shifts are measured in), `start` the curves to start from (at every pixel of the walk), `crossed` a
        picture that is True where another vessel's core lies. Returns the refined geometry."""
        z, centre = walked["z"], walked["centre"]
        normal = normals_of(centre)
        take = np.unique(np.append(np.arange(0, len(z), 2), len(z) - 1))  # a cross-section every 2 px
        count = max(4, int(np.ceil((z[-1] - z[0]) / KNOT_SPACING)) + 3)
        knots = np.concatenate([[z[0]] * 3, np.linspace(z[0], z[-1], count - 2), [z[-1]] * 3])
        basis = scipy.interpolate.BSpline.design_matrix(np.clip(z[take], z[0], z[-1] - 1e-9), knots, 3).toarray()
        leash = np.maximum(2.0, LEASH_MOST * walked["width"])
        reach = 3.5 * start["width"].max() + leash.max()
        u = np.arange(-reach, reach + 0.25, 0.5)
        profiles = bilinear(picture, centre[take, 0, None] + u * normal[take, 0, None],
                            centre[take, 1, None] + u * normal[take, 1, None])
        rows = np.clip(np.round(centre[take, 1]).astype(int), 0, nf_image.shape[0] - 1)
        columns = np.clip(np.round(centre[take, 0]).astype(int), 0, nf_image.shape[1] - 1)
        weight = np.where(crossed[rows, columns], CROSSING_WEIGHT, 1.0)[:, None]
        shape = choose_shape(profiles, u, start["width"][take])
        flatten = without_baseline(u)  # every cross-section's own baseline is solved away, never rendered
        profiles = flatten(profiles)
        least_squares = lambda values: np.linalg.lstsq(basis, values, rcond=None)[0]
        theta = np.concatenate([np.zeros(count), least_squares(np.log(start["width"][take])),
                                least_squares(start["amplitude"][take])])

        def curves(theta):
            shift, log_width, amplitude = (basis @ theta[place * count:(place + 1) * count] for place in range(3))
            return shift, np.exp(log_width), amplitude

        def misfit(theta):
            shift, width, amplitude = curves(theta)
            model = flatten(amplitude[:, None] * vessel_profile(u[None, :] - shift[:, None], width[:, None], shape))
            return np.sum(weight * (profiles - model) ** 2)

        def normal_equations(theta):
            shift, width, amplitude = curves(theta)
            t = (u[None, :] - shift[:, None]) / width[:, None]
            g = np.exp(-0.5 * np.abs(t) ** shape)
            slope = -0.5 * shape * np.abs(t) ** (shape - 1) * np.sign(t) * g  # dg/dt
            jacobians = (flatten(amplitude[:, None] * slope * (-1 / width[:, None])),  # d model / d shift
                         flatten(amplitude[:, None] * slope * (-t)),                   # d model / d log width
                         flatten(g))                                                   # d model / d amplitude
            residual = profiles - flatten(amplitude[:, None] * g)
            hessian = np.zeros((3 * count, 3 * count))
            gradient = np.zeros(3 * count)
            for i, first in enumerate(jacobians):
                gradient[i * count:(i + 1) * count] = basis.T @ np.sum(weight * first * residual, axis=1)
                for j, second in enumerate(jacobians):
                    hessian[i * count:(i + 1) * count, j * count:(j + 1) * count] = (
                        basis.T @ (np.sum(weight * first * second, axis=1)[:, None] * basis))
            return hessian, gradient

        # the penalties, each scaled to the data term of its own kind of coefficient
        hessian, _ = normal_equations(theta)
        scale = [max(np.trace(hessian[i * count:(i + 1) * count, i * count:(i + 1) * count]) / count, 1e-12) for i in range(3)]
        first_difference, second_difference = np.diff(np.eye(count), axis=0), np.diff(np.eye(count), 2, axis=0)
        bend, change = second_difference.T @ second_difference, first_difference.T @ first_difference
        penalty = scipy.linalg.block_diag(scale[0] * (STIFFNESS["shift"] * bend + LEASH * np.eye(count)),
                                          scale[1] * (STIFFNESS["width"] * bend + STEADINESS["width"] * change),
                                          scale[2] * (STIFFNESS["amplitude"] * bend + STEADINESS["amplitude"] * change))
        objective = lambda theta: misfit(theta) + theta @ penalty @ theta
        current, damping = objective(theta), 1e-3
        for _ in range(5):  # Gauss-Newton steps
            hessian, gradient = normal_equations(theta)
            system = hessian + penalty
            for _ in range(8):  # Levenberg-Marquardt: damp until the step lowers the penalised objective
                step = np.linalg.solve(system + damping * np.diag(np.diag(system)), gradient - penalty @ theta)
                trial = objective(theta + step)
                if trial < current:
                    theta, current, damping = theta + step, trial, damping / 3
                    break
                damping *= 10
            else:
                break
        shift = np.clip(scipy.interpolate.BSpline(knots, theta[:count], 3)(z), -leash, leash)
        width = np.clip(np.exp(scipy.interpolate.BSpline(knots, theta[count:2 * count], 3)(z)), 0.8, 40.0)
        return {"z": z, "centre": centre + shift[:, None] * normal, "width": width, "shape": shape, "shift": shift}


    def vessel_part(fit, number):
        """One vessel's part of a joint fit's model: the pixels it reaches (flat indices) and its density there."""
        first, count, _ = fit["blocks"][number]
        block = fit["A"][:, first:first + count]
        rows = np.unique(block.indices)
        return fit["pixels"][rows], (block @ fit["vector"][first:first + count])[rows]


    def fit_one(vessel, picture):
        """Fit one vessel's amplitude alone to `picture` (the density without every other vessel), with the same penalties
        as the joint fit. Returns its part of the model: the pixels it reaches and its density there."""
        basis = vessel_basis(vessel)
        design, count = basis["design"], basis["count"]
        rows, row_of = np.unique(basis["pixels"], return_inverse=True)
        A = scipy.sparse.csr_matrix((design.data * basis["profile"][design.row], (row_of[design.row], design.col)),
                                    shape=(len(rows), count))
        seen = len(design.data) / count
        second = scipy.sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(count - 2, count))
        change = scipy.sparse.diags([-1.0, 1.0], [0, 1], shape=(count - 1, count))
        P = scipy.sparse.vstack([np.sqrt(AMPLITUDE_SMOOTHING * seen) * second, np.sqrt(AMPLITUDE_STEADINESS * seen) * change])
        coefficients = scipy.sparse.linalg.spsolve((A.T @ A + P.T @ P).tocsc(), A.T @ picture.ravel()[rows].astype(np.float64))
        return rows, A @ coefficients


    # one vessel at a time, largest first: each sees the others' latest models, and its own new model is written back into
    # the field's before the next vessel looks (Gauss-Seidel), so no leftover is absorbed twice
    _vessels, _fit = continued_vessels, continued_fit
    walked = [geometry_of(vessel) for vessel in _vessels]
    model_picture = _fit["model"].astype(np.float64)
    _parts = [vessel_part(_fit, _number) for _number in range(len(_vessels))]
    _labels = cores_picture(_vessels)
    refined_vessels, _moved, faint = list(_vessels), np.zeros(len(_vessels)), 0
    for _number in np.argsort([-np.mean(_geometry["width"]) for _geometry in walked]):
        amplitude = _fit["amplitudes"][_number](walked[_number]["z"])
        if np.median(amplitude) < 10 * nf_noise:  # too faint to place better than its walk did
            faint += 1
            continue
        _pixels, _values = _parts[_number]
        _picture = nf_image - model_picture
        _picture.ravel()[_pixels] += _values  # the density without every other vessel
        _geometry = fit_vessel_geometry(walked[_number], {"width": walked[_number]["width"], "amplitude": amplitude}, _picture,
                                       (_labels >= 0) & (_labels != _number))
        refined_vessels[_number] = {**_vessels[_number], "geometry": _geometry}
        _moved[_number] = np.abs(_geometry["shift"]).mean()
        new_pixels, new_values = fit_one(refined_vessels[_number], _picture)
        model_picture.ravel()[_pixels] -= _values
        model_picture.ravel()[new_pixels] += new_values
        _parts[_number] = (new_pixels, new_values)
    del model_picture, _parts
    _shapes = np.array([geometry_of(vessel)["shape"] for vessel in refined_vessels])
    print(f"{len(_vessels) - faint} vessels refined ({faint} too faint kept their walk's geometry): centres moved "
          f"{_moved[_moved > 0].mean():.2f} px from their walks on average | profile shapes "
          + ", ".join(f"{shape:g}: {np.sum(_shapes == shape)}" for shape in REFINE_SHAPES))
    final_vessels, final_fit, final_counts = settle(refined_vessels)
    print(f"settled: {final_counts['after cutting']} pieces after cutting the faint stretches, {len(final_vessels)} after "
          f"dropping {final_counts['dropped']}; {explained(final_fit['residual'], final_fit['density']) * 100:.1f}% explained "
          f"(before refining: {explained(_fit['residual'], _fit['density']) * 100:.1f}%), with {len(final_fit['junctions'])} "
          f"junctions")
    junction_table(final_fit)

    # the residual of the whole field before and after refining
    union = np.union1d(_fit["pixels"], final_fit["pixels"])
    _covered = np.zeros(nf_image.shape, bool)
    _covered.ravel()[union] = True
    _colormap = plt.get_cmap("RdBu_r").copy()
    _colormap.set_bad("#2c2c2a")
    _figure, _axes = plt.subplots(1, 2, figsize=(16, 5.6))
    for _axis, (_title, _model) in zip(_axes, (("density - model, before refining", _fit["model"]),
                                           ("density - model, after refining", final_fit["model"]))):
        _image = _axis.imshow(np.where(_covered, nf_image - _model, np.nan), cmap=_colormap, vmin=-RESIDUAL_RANGE,
                            vmax=RESIDUAL_RANGE)
        _figure.colorbar(_image, ax=_axis, fraction=0.03, pad=0.01)
        _residual = (nf_image - _model).ravel()[union]
        _axis.set_title(f"{_title}: rms {np.sqrt(np.mean(_residual ** 2)):.4f}", fontsize=10)
        _axis.axis("off")
    _figure.tight_layout()
    save_figure(_figure, "refined_residual", banded_key)
    plt.show()
    return CROSSING_WEIGHT, final_fit, final_vessels, vessel_part


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 23 · Coarse to fine: the residual decides where to look closer

    A diffusion model makes an image in steps. At a high noise level only the large layout stands out of the noise, so only that is settled; then the noise is lowered and the next finer detail is settled, down to none. Here the fit is refined on the same kind of schedule. Each level blurs the residual, density − haze − model, by its own blur, tests boxes of its own size, and gives the vessels new freedom only where a box fails, with knots as close together as that blur can resolve.

    **The noise, measured.** A box is judged against the noise in the mean density, measured with the burst's own frames: `nf_null` is half the difference between the mean of the even frames and the mean of the odd ones. Both halves see the same tissue, so what is left is noise alone, spread pixel by pixel as the noise in the mean of all the frames is (flow inside the vessels included). Blurred by $s$, the noise's variance at each pixel is that of the null blurred alike. The noise of the mean is 0.0016 between the vessels and 0.0019 in their cores, three times less than `nf_noise` (0.0049), which also holds what the background fit left.

    **Level 0, the whole field as one box: the haze.** Between the vessels the density is not 0: diffuse density, from vessels too fine or too deep to trace, is left. The model has no term for it, so every vessel near it widens or brightens to take some. The haze is a local mean, `HAZE_SIGMA` wide, of the residual over the pixels clear of every vessel (where their profiles add up to less than `HAZE_CLEAR`), carried on under the vessels. The haze and every amplitude and junction of §20 are fitted in turns.

    **Levels 1 and on**, coarse to fine, each with its blur $s$, box side $B$ and knot spacing $h$ (`LEVELS`):

    1. **Blur.** The residual is blurred by $s$, over the pixels with data alone.
    2. **Boxes.** Boxes of side $B$, one every $B/2$ px, so that each pixel lies in four. A box fails where its blurred residual carries at least `FAIL_RATIO` times the energy that noise alone would carry there. Every level tests all of its boxes, not only those inside a failing box of the level before: a centre 1 px off leaves a pair of lines, one bright and one dark, that an 8 px blur averages away, so passing at a coarse level says nothing about a finer one.
    3. **Freedom where a box fails.** Every vessel gets corrections to its shift, log width and amplitude, each a cubic spline with a knot every $h$ px. A coefficient is free only where its reach touches a failing box and its *score*, $g^2/H$, the misfit that freeing it alone would remove, in noise units, is at least `SCORE_LEVEL`. ($g$ and $H$ are the gradient and curvature of the misfit in that coefficient.) It is the counterpart of §20's unique share: that one asks what a vessel would cost to remove, this one what a coefficient would gain if it were added.
    4. **The fit.** The free coefficients are fitted by penalised Gauss–Newton, one vessel at a time, largest first, each seeing the others' latest models (as in §22). The misfit compares the blurred picture (the density without the haze and every other vessel) with the vessel's own model, blurred alike. The blur is computed exactly; only the derivatives use a closed form. Across a straight vessel, a profile $\sigma$ wide blurred by $s$ is the same profile $\tilde\sigma = \sqrt{\sigma^2 + (s/m)^2}$ wide, its peak lowered by $\sigma/\tilde\sigma$, where $m$ is the profile's spread in widths (1 for a Gaussian). No level moves a centre more than `LEVEL_SHIFT_MOST` px, or its blur if that is larger.
    5. **The penalties, the same at every level.** A correction pays for changing along the vessel faster than `CORRECTION_WAVELENGTH` px, through $\int f''^2\,dz$ of the curve itself, and for departing from §22's geometry at all (`ANCHOR`), each relative to the evidence per px of vessel. Both are stated on the curves, not on the spline coefficients, and on the whole departure from §22, not on one level's part of it. Finer knots therefore let a smooth correction be drawn more exactly but buy no extra wiggle, and every level minimises the same objective. The levels are a route to its minimum, coarse knots first, as a multigrid solver takes one.
    6. **The field again.** Every amplitude and junction is fitted jointly (§20), then the haze again, then the amplitudes again.

    **Why blur, and why not here.** Gauss–Newton only finds a centre that is already within about a width of it. Blurring both the picture and the model by $s$ widens that reach by about $s$, so that coarse levels can bring a geometry close enough for the unblurred levels to finish. Here §22 already put the centres within about half a pixel, so the default `LEVELS` do not blur. The blur is kept as the first number of each level, to try.

    **Then** the field is settled (§20) against the density without the haze. The boxes still failing at the finest level show where the model itself is wrong, not its settings.

    ### What it found

    | | residual rms, field | over §22's vessels | widths change per 10 px |
    |---|---|---|---|
    | §22 | 0.0187 | 0.0214 | 0.9% |
    | level 0: the haze | 0.0162 | 0.0191 | |
    | levels 1–3, settled | 0.0141 | 0.0164 | 3.7% |

    (Optical density, over the field inside the neurons' margin and over the pixels §22's vessels reach; both regions are fixed, so the rows compare.)

    - **Every box fails, at every level.** Even unblurred, a 32 px box's residual carries a median 40 times the noise's energy, and blurred by 8 px it carries thousands of times. The residual is structure the model lacks, not noise, so the boxes do not yet single anything out: they will once the model comes within a few noise levels. Working only on the worst boxes, those holding half of the energy above `FAIL_RATIO`, ended worse than working everywhere.
    - **The haze is the largest single gain**, about half of the whole. It is strongest at the field's edges and where the vessels are densest, where §10's background had the fewest background pixels to go by.
    - **The levels move centres by half a pixel and widths by about 13% from §22's**, slowly along each vessel. Thin vessels (σ under 3 px) came out 23% wider and 15% fainter, their area (amplitude × width) almost unchanged. The density alone cannot say whether that is their lumen or density beside them; §24's flow, a picture of the moving blood alone, has the same width across them (0.89 σ against the density's 0.91), so it is their lumen.
    - **Blurring made it worse**: with blurs of 8 and 4 px on the first two levels the field's residual ended at 0.0151, not 0.0141. The blurred levels moved centres 1.3 px towards blurred structure, and the unblurred level had to bring them back.
    - **Wiggles bought nothing.** With widths allowed to change over 48 px instead of 192 px, the residual was the same, but widths changed 13% every 10 px. The penalties are therefore on the curves, not on the coefficients (step 5).
    - **The levels are a route, not the answer.** Running the finest level four times instead reached the same residual (within 2%), as it should when every level minimises the same objective.
    - **Where the residual lives**, scale by scale (first figure): the haze takes most of it at blurs of 16–32 px, the levels some at 1–8 px. What is left stands out most at a blur of 8 px, the size of the vessels themselves. It lies along the wide vessels and at the field's edges (second figure), where a Gaussian profile does not fit and the background was poorly known.

    | Output | Content |
    |---|---|
    | `nf_null` | noise only: half the difference between the means of the even and the odd frames of the fixation |
    | `haze` | the density between the vessels, carried on under them, as a picture |
    | `fine_vessels`, `fine_fit` | the vessels after the levels and settling, and the joint fit of the density without the haze with them |
    | `level_log` | per level: its blur, boxes and knots, the share of boxes failing before and after it, the coefficients freed, the residual |
    | `misfit_ratio` | per pixel, after settling: the residual's energy over the noise's in the worst finest box it lies in |
    | `haze_residual`, `fine_residual` | density − haze − model after the haze alone, and after the levels and settling |
    """)
    return


@app.cell
def _(
    CROSSING_WEIGHT,
    DENSITY_RANGE,
    INK_MUTED,
    NF_FIXATION,
    PROFILE_SIGMAS,
    RESIDUAL_RANGE,
    SERIES,
    banded_key,
    cores_picture,
    cv2,
    final_density_fixations,
    final_fit,
    final_vessels,
    fit_field,
    geometry_of,
    junction_table,
    matplotlib,
    nf_background,
    nf_data,
    nf_image,
    nf_inside,
    nf_noise,
    normals_of,
    np,
    plt,
    robust_sd,
    save_figure,
    scipy,
    settle,
    unique_share,
    vessel_part,
    vessel_profile,
):
    import time

    LEVELS = ((0.0, 128, 16.0), (0.0, 64, 8.0), (0.0, 32, 4.0))  # coarse to fine: (blur, box side, knot spacing) in px
    FAIL_RATIO = 2.0          # a box fails where its blurred residual carries at least this many times the energy of noise alone
    SCORE_LEVEL = 4.0         # a coefficient is freed where freeing it alone would cut the misfit by this many noise units
    HAZE_SIGMA = 16.0         # pixels; the haze is a local mean, this wide, of the density left between the vessels
    HAZE_CLEAR = 0.1          # a pixel shows the haze where the vessels' profiles there add up to less than this
    HAZE_ROUNDS = 2           # the haze and the amplitudes are fitted in turns this many times
    CORRECTION_WAVELENGTH = {"shift": 48.0, "width": 192.0, "amplitude": 24.0}  # px; corrections changing faster along a vessel are damped
    ANCHOR = {"shift": 0.1, "width": 0.1, "amplitude": 0.1}  # resistance to departing from §22's geometry at all, relative to the evidence
    LEVEL_SHIFT_MOST = 2.0    # pixels (or the level's blur, if larger) a level may move a centre
    GAUSS_NEWTON_STEPS = 4    # Gauss-Newton steps per vessel and level
    SCALES = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)  # blurs at which the residual's energy is compared with the noise's
    FAMILIES = ("shift", "width", "amplitude")      # what a level corrects along a vessel; "width" is its logarithm


    def split_half_null(frames):
        """Half the difference between the mean of the even frames and the mean of the odd frames, each over the frames
        with data at a pixel: noise only, spread pixel by pixel as the noise in the mean of all the frames is."""
        sums = np.zeros((2,) + frames[0].shape)
        counts = np.zeros((2,) + frames[0].shape, np.int32)
        for index, frame in enumerate(frames):
            frame = np.asarray(frame)
            sums[index % 2] += np.nan_to_num(frame)
            counts[index % 2] += np.isfinite(frame)
        means = sums / np.maximum(counts, 1)
        return np.where((counts > 0).all(axis=0) & nf_data, 0.5 * (means[0] - means[1]), 0).astype(np.float32)


    fine_valid = nf_data.astype(np.float32)  # the pixels with data: every blur below is a mean over them alone


    def fine_blur_valid(picture, blur, valid=None):
        """A Gaussian blur of `picture` over the pixels with data alone (normalised convolution); blur 0 changes nothing."""
        if blur == 0:
            return picture
        valid = fine_valid if valid is None else valid
        picture = np.asarray(picture, np.float64)
        return (cv2.GaussianBlur(picture * valid, (0, 0), blur)
                / np.maximum(cv2.GaussianBlur(valid.astype(np.float64), (0, 0), blur), 1e-6))


    def noise_variance(blur):
        """Per pixel, the variance of the noise in the mean density blurred by `blur`: the split-half null blurred alike,
        squared, and averaged over a neighbourhood (4 px, or twice the blur) so that it is a variance, not one sample."""
        return fine_blur_valid(fine_blur_valid(nf_null, blur) ** 2, max(4.0, 2.0 * blur))


    def box_ratios(residual, variance, box):
        """Boxes of side `box`, one every box/2 px: per box, the energy of the (blurred) residual over the energy the noise
        alone would have there, both summed over the box's pixels with data. NaN for a box less than half covered by data."""
        half = box // 2
        rows, columns = -(-nf_image.shape[0] // half), -(-nf_image.shape[1] // half)

        def box_sums(picture):
            padded = np.zeros((rows * half, columns * half))
            padded[:nf_image.shape[0], :nf_image.shape[1]] = picture
            blocks = padded.reshape(rows, half, columns, half).sum(axis=(1, 3))
            return blocks[:-1, :-1] + blocks[1:, :-1] + blocks[:-1, 1:] + blocks[1:, 1:]  # 2 x 2 blocks make one box

        energy, noise, count = box_sums(residual ** 2 * fine_valid), box_sums(variance * fine_valid), box_sums(fine_valid)
        return np.where(count >= 0.5 * box * box, energy / np.maximum(noise, 1e-30), np.nan)


    def ratio_picture(ratio, box):
        """Per pixel, the largest ratio of the boxes it lies in (NaN where none is judged)."""
        half = box // 2
        blocks = np.full((ratio.shape[0] + 1, ratio.shape[1] + 1), -np.inf)
        for down in (0, 1):
            for right in (0, 1):
                part = blocks[down:down + ratio.shape[0], right:right + ratio.shape[1]]
                np.maximum(part, np.nan_to_num(ratio, nan=-np.inf), out=part)
        picture = np.kron(blocks, np.ones((half, half)))[:nf_image.shape[0], :nf_image.shape[1]]
        return np.where(np.isfinite(picture), picture, np.nan).astype(np.float32)


    def failing_pixels(ratio, box):
        """The pixels inside at least one box whose ratio is at least FAIL_RATIO."""
        return np.nan_to_num(ratio_picture(ratio, box), nan=0) >= FAIL_RATIO


    def vessel_cover(fit):
        """Per pixel, the sum of every vessel's profile there (each 1 on its own centreline)."""
        pixels = np.concatenate([basis["pixels"] for basis in fit["bases"]])
        profiles = np.concatenate([basis["profile"] for basis in fit["bases"]])
        return np.bincount(pixels, weights=profiles, minlength=nf_image.size).reshape(nf_image.shape)


    def estimate_haze(residual, clear):
        """The haze: a local mean, HAZE_SIGMA wide, of `residual` over the pixels with data that are clear of every vessel.
        Where such pixels are scarce (fewer than a fifth of the neighbourhood), a mean four times as wide fills in."""
        clear = (clear & nf_data).astype(np.float64)
        residual = np.asarray(residual, np.float64)
        near = cv2.GaussianBlur(clear, (0, 0), HAZE_SIGMA)
        narrow = cv2.GaussianBlur(residual * clear, (0, 0), HAZE_SIGMA) / np.maximum(near, 1e-6)
        wide = (cv2.GaussianBlur(residual * clear, (0, 0), 4 * HAZE_SIGMA)
                / np.maximum(cv2.GaussianBlur(clear, (0, 0), 4 * HAZE_SIGMA), 1e-6))
        trust = np.clip(near / 0.2, 0, 1)
        return np.where(nf_data, trust * narrow + (1 - trust) * wide, 0).astype(np.float32)


    def vessel_band(geometry, reach, stride=1):
        """The pixels (on every `stride`-th row and column) within PROFILE_SIGMAS widths plus `reach` px of a vessel's
        centreline, and as far past its ends: their flat indices ("pixels"), the index of the nearest centreline point
        ("nearest"), the distance across from it ("across") and how far past the vessel's ends they lie ("beyond"), measured
        as §20's vessel_basis measures them."""
        z_line, centre, sigma = geometry["z"], geometry["centre"], geometry["width"]
        normal = normals_of(centre)
        tangent = np.column_stack([normal[:, 1], -normal[:, 0]])
        extent = PROFILE_SIGMAS * sigma + reach
        low = np.maximum(np.floor(centre.min(axis=0) - extent.max()), 0).astype(int)
        high = np.minimum(np.ceil(centre.max(axis=0) + extent.max()) + 1, nf_image.shape[::-1]).astype(int)
        band = np.zeros((high[1] - low[1], high[0] - low[0]), np.uint8)
        corners = np.round(centre - low).astype(int)
        for i in range(len(z_line) - 1):
            cv2.line(band, tuple(map(int, corners[i])), tuple(map(int, corners[i + 1])), 1, 2 * int(np.ceil(extent[i])) + 1)
        rows, columns = np.nonzero(band)
        rows, columns = rows + low[1], columns + low[0]
        on_grid = (rows % stride == 0) & (columns % stride == 0)
        rows, columns = rows[on_grid], columns[on_grid]
        points = np.column_stack([columns, rows]).astype(np.float64)
        _, nearest = scipy.spatial.cKDTree(centre).query(points)
        offset = points - centre[nearest]
        across = np.sum(offset * normal[nearest], axis=1)
        z_raw = z_line[nearest] + np.sum(offset * tangent[nearest], axis=1)
        beyond = np.maximum(0, np.maximum(z_line[0] - z_raw, z_raw - z_line[-1]))
        keep = (np.abs(across) <= extent[nearest]) & (beyond <= extent[nearest])
        return {"pixels": rows[keep] * nf_image.shape[1] + columns[keep], "nearest": nearest[keep], "across": across[keep],
                "beyond": beyond[keep]}


    def render(band, curves, shape, amplitude="amplitude"):
        """A vessel's density on its band, as §20 models it: its amplitude times its profile, `shift` px across from the
        centreline the band was measured from, cut off beyond PROFILE_SIGMAS widths and faded past the vessel's ends.
        `curves` holds the shift, log width and amplitude at every point of the centreline; `amplitude` names the one to
        use (§24 gives a vessel a second one, for the flow)."""
        shift, log_width, height = (curves[name][band["nearest"]] for name in ("shift", "width", amplitude))
        sigma = np.exp(log_width)
        across = band["across"] - shift
        inside = (np.abs(across) <= PROFILE_SIGMAS * sigma) & (band["beyond"] <= PROFILE_SIGMAS * sigma)
        return np.where(inside, height * vessel_profile(across, sigma, shape) * np.exp(-0.5 * (band["beyond"] / sigma) ** 2),
                        0.0)


    def profile_spread(shape):
        """The standard deviation of the profile exp(-|t|^shape / 2) in t: 1 for a Gaussian, less for a flatter top."""
        return float(np.sqrt(2 ** (2 / shape) * scipy.special.gamma(3 / shape) / scipy.special.gamma(1 / shape)))


    def blurred_jacobian(band, curves, shape, blur, amplitude="amplitude"):
        """The derivatives of a vessel's density (with the amplitude curve `amplitude`), blurred by `blur`, at its band's
        pixels, with respect to its shift, log width and amplitude there. In closed form: across a straight vessel, a
        profile of width sigma blurred by b is the same profile sqrt(sigma^2 + (b/m)^2) wide (m its spread in widths), its
        peak lowered by the same factor."""
        nearest = band["nearest"]
        shift, sigma, height = curves["shift"][nearest], np.exp(curves["width"][nearest]), curves[amplitude][nearest]
        spread = blur / profile_spread(shape)
        wide = np.sqrt(sigma ** 2 + spread ** 2)
        t = (band["across"] - shift) / wide
        g = np.exp(-0.5 * np.abs(t) ** shape)
        slope = -0.5 * shape * np.abs(t) ** (shape - 1) * np.sign(t) * g  # dg/dt
        fade = np.exp(-0.5 * (band["beyond"] / wide) ** 2)
        ratio = sigma / wide
        return {"shift": -height * ratio * slope * fade / wide,
                "width": height * fade * (sigma * spread ** 2 / wide ** 3 * g - ratio ** 3 * t * slope),
                "amplitude": ratio * g * fade}


    def refine_on_level(geometry, departure, pictures, failing, labels, number, blur, spacing):
        """Correct one vessel's shift and log width, and its amplitude in each of `pictures`, on one level, each a cubic
        spline with a knot every `spacing` px, against the pictures blurred by `blur`. `pictures` maps a name ("amplitude"
        for the density; §24 adds "flow") to a dict: "picture", the picture without the haze and every other vessel;
        "weight", how much each of its pixels counts (a picture, or one number); "amplitude", the vessel's amplitude in it
        along the centreline. The shift and the width are shared by every picture.
        Where another vessel's core lies (`labels`, from cores_picture, is neither -1 nor `number`), a pixel counts
        CROSSING_WEIGHT, as in §22. `departure` holds the shift and log width the levels before have moved it from where
        the penalties are anchored: they are on the whole departure, not on this level's part of it.
        Only the coefficients whose reach touches a failing box and whose score is at least SCORE_LEVEL are free.
        Returns None if none is, else the new geometry, the new amplitudes along it, the pixels its density reaches and
        its value there in every picture, how many coefficients of each kind were freed, and the misfit it removed."""
        z, centre, width, shape = geometry["z"], geometry["centre"], geometry["width"], geometry["shape"]
        families = ("shift", "width") + tuple(pictures)
        normal = normals_of(centre)
        most = max(LEVEL_SHIFT_MOST, blur)
        stride = max(1, int(blur // 2))  # a blurred picture is compared every blur/2 px
        drawn = vessel_band(geometry, most + 1)                      # where the vessel's density may land
        seen = vessel_band(geometry, most + 2 * blur + 1, stride)    # where its blurred density is compared with the data
        every = np.concatenate([drawn["pixels"], seen["pixels"]])
        ys, xs = np.divmod(every, nf_image.shape[1])
        pad = int(np.ceil(4 * blur)) + 2
        top, left = max(ys.min() - pad, 0), max(xs.min() - pad, 0)
        bottom, right = min(ys.max() + pad + 1, nf_image.shape[0]), min(xs.max() + pad + 1, nf_image.shape[1])
        crop = (slice(top, bottom), slice(left, right))
        valid = fine_valid[crop]

        def local(pixels):
            rows, columns = np.divmod(pixels, nf_image.shape[1])
            return (rows - top) * (right - left) + (columns - left)

        drawn_at, seen_at = local(drawn["pixels"]), local(seen["pixels"])
        core = labels.ravel()[seen["pixels"]]
        crossing = np.where((core >= 0) & (core != number), CROSSING_WEIGHT, 1.0)
        data = np.concatenate([fine_blur_valid(item["picture"][crop], blur, valid).ravel()[seen_at] for item in pictures.values()])
        weight = np.concatenate([crossing * (item["weight"].ravel()[seen["pixels"]] if np.ndim(item["weight"]) else item["weight"])
                                 for item in pictures.values()])
        count = max(4, int(np.ceil((z[-1] - z[0]) / spacing)) + 3)
        knots = np.concatenate([[z[0]] * 3, np.linspace(z[0], z[-1], count - 2), [z[-1]] * 3])
        along = scipy.interpolate.BSpline.design_matrix(np.clip(z, z[0], z[-1] - 1e-9), knots, 3).tocsr()
        at_seen = along[seen["nearest"]]
        start = {"shift": np.zeros(len(z)), "width": np.log(width), **{name: item["amplitude"] for name, item in pictures.items()}}
        low_width, high_width = np.log(0.8), np.log(40.0)

        def curves(theta):
            values = {name: start[name] + along @ theta[place * count:(place + 1) * count] for place, name in enumerate(families)}
            values["shift"] = np.clip(values["shift"], -most, most)
            values["width"] = np.clip(values["width"], low_width, high_width)
            return values

        def blurred(theta):
            values, models = curves(theta), []
            for name in pictures:
                own = np.zeros(valid.shape)
                own.ravel()[drawn_at] = render(drawn, values, shape, name)
                models.append(fine_blur_valid(own, blur, valid).ravel()[seen_at])
            return np.concatenate(models)

        def jacobian(theta):
            values, rows = curves(theta), []
            for name in pictures:  # one band of rows per picture: the shared shift and width, then its own amplitude
                parts = blurred_jacobian(seen, values, shape, blur, name)
                columns = [scipy.sparse.diags(parts[family]) @ at_seen for family in ("shift", "width")]
                columns += [scipy.sparse.diags(parts["amplitude"]) @ at_seen if family == name
                            else scipy.sparse.csr_matrix(at_seen.shape) for family in pictures]
                rows.append(scipy.sparse.hstack(columns))
            return scipy.sparse.vstack(rows).tocsr()

        theta = np.zeros(len(families) * count)
        residual = data - blurred(theta)
        J = jacobian(theta)
        gradient = J.T @ (weight * residual)
        curvature = J.power(2).T @ weight
        oversampling = max(1.0, 4 * np.pi * blur ** 2 / stride ** 2)  # samples per independent one, for a blurred picture
        score = gradient ** 2 / np.maximum(curvature, 1e-30) / oversampling
        touches = (at_seen.T @ failing.ravel()[seen["pixels"]].astype(np.float64)) > 0
        free = (score.reshape(len(families), count) >= SCORE_LEVEL) & touches
        free |= touches & (np.pad(free[:, 1:], ((0, 0), (0, 1))) | np.pad(free[:, :-1], ((0, 0), (1, 0))))  # and neighbours
        if not free.any():
            return None
        index = np.flatnonzero(free.ravel())
        # the penalties, on the curves along the vessel, whatever the knot spacing: their size (ANCHOR) and their bending
        # (exact for the splines, sum of the second difference squared every px), each relative to the evidence per px
        dense = along.toarray()
        size, bending = dense.T @ dense, np.diff(dense, 2, axis=0).T @ np.diff(dense, 2, axis=0)
        blocks, prior = [], []
        for place, name in enumerate(families):
            kind = name if name in ("shift", "width") else "amplitude"  # every picture's amplitude is penalised alike
            evidence = max(curvature[place * count:(place + 1) * count].sum() / len(z), 1e-12)
            length = CORRECTION_WAVELENGTH[kind] / (2 * np.pi)
            blocks.append(evidence * (ANCHOR[kind] * size + length ** 4 * bending))
            prior.append(np.linalg.lstsq(dense, departure[name], rcond=None)[0] if name in departure else np.zeros(count))
        penalty, prior = scipy.linalg.block_diag(*blocks), np.concatenate(prior)
        objective = lambda theta, residual: np.sum(weight * residual ** 2) + (prior + theta) @ penalty @ (prior + theta)
        first = current = objective(theta, residual)
        damping = 1e-3
        for step_number in range(GAUSS_NEWTON_STEPS):
            if step_number:
                J = jacobian(theta)
            hessian = (J.T @ J.multiply(weight[:, None])).toarray()[np.ix_(index, index)] + penalty[np.ix_(index, index)]
            right_side = (J.T @ (weight * residual) - penalty @ (prior + theta))[index]
            for _ in range(8):  # Levenberg-Marquardt: damp until the step lowers the penalised misfit, blurred exactly
                trial = theta.copy()
                trial[index] += np.linalg.solve(hessian + damping * np.diag(np.diag(hessian)), right_side)
                trial_residual = data - blurred(trial)
                value = objective(trial, trial_residual)
                if value < current:
                    theta, residual, current, damping = trial, trial_residual, value, damping / 3
                    break
                damping *= 10
            else:
                break
        final = curves(theta)
        new_geometry = {"z": z, "centre": centre + final["shift"][:, None] * normal, "width": np.exp(final["width"]),
                        "shape": shape, "shift": geometry.get("shift", np.zeros(len(z))) + final["shift"]}
        return {"geometry": new_geometry, "amplitude": {name: final[name] for name in pictures}, "pixels": drawn["pixels"],
                "values": {name: render(drawn, final, shape, name) for name in pictures}, "freed": free.sum(axis=1),
                "coefficients": count, "shift": final["shift"], "gain": (first - current) / oversampling,
                "moved": float(np.abs(final["shift"]).mean())}


    def scale_profile(residual, region):
        """The residual's energy over the noise's, over `region`, at every blur of SCALES."""
        return np.array([np.sum(fine_blur_valid(residual, blur)[region] ** 2) / np.sum(noise_variance(blur)[region])
                         for blur in SCALES])


    def residual_rms(residual, region):
        """The rms of a residual over a region (a boolean picture)."""
        return float(np.sqrt(np.mean(np.asarray(residual)[region] ** 2)))


    def width_drift(vessels):
        """The median, over the vessels, of how much their widths changed from §22's: exp(median |ln(width / §22 width)|) - 1
        along each vessel (the same z points), as a share."""
        changes = [np.median(np.abs(np.log(geometry_of(new)["width"] / geometry_of(old)["width"])))
                   for new, old in zip(vessels, final_vessels) if len(geometry_of(new)["width"]) == len(geometry_of(old)["width"])]
        return float(np.expm1(np.median(changes)))


    def width_roughness(vessels):
        """The median, over the vessels at least 40 px long, of how much their width changes every 10 px along them: the
        rms of ln(width) 10 px on minus ln(width) here."""
        values = []
        for vessel in vessels:
            geometry = geometry_of(vessel)
            z, log_width = geometry["z"], np.log(geometry["width"])
            if z[-1] - z[0] >= 40:
                here = np.arange(z[0], z[-1] - 10, 1.0)
                values.append(np.sqrt(np.mean((np.interp(here + 10, z, log_width) - np.interp(here, z, log_width)) ** 2)))
        return float(np.median(values))


    nf_null = split_half_null(final_density_fixations[NF_FIXATION])
    print(f"Split-half null: the noise of the mean density is {robust_sd(nf_null[nf_background]):.4f} on the background "
          f"pixels (nf_noise, which also holds what the background fit left, is {nf_noise:.4f}) and "
          f"{robust_sd(nf_null[(vessel_cover(final_fit) >= 0.5) & nf_data]):.4f} in the vessels' cores")
    fine_started = time.time()
    fine_region = nf_inside & nf_data                 # every residual below is measured over this fixed region...
    band_region = np.zeros(nf_image.shape, bool)      # ...and over the pixels §22's vessels reach, also fixed
    band_region.ravel()[final_fit["pixels"]] = True
    band_region &= nf_inside

    # level 0, the whole field as one box: the haze and every amplitude and junction, fitted in turns
    _vessels, _fit = list(final_vessels), final_fit
    haze = np.zeros(nf_image.shape, np.float32)
    for _ in range(HAZE_ROUNDS):
        haze = estimate_haze(nf_image - _fit["model"], vessel_cover(_fit) < HAZE_CLEAR)
        _fit = fit_field(_vessels, _fit["bases"], target=nf_image - haze)
    haze_residual = (nf_image - haze - _fit["model"]).astype(np.float32)  # after the haze alone
    level_log = [{"level": 0, "blur": None, "rms": residual_rms(nf_image - haze - _fit["model"], fine_region),
                  "band_rms": residual_rms(nf_image - haze - _fit["model"], band_region), "width_drift": 0.0,
                  "seconds": time.time() - fine_started}]
    print(f"level 0 (the whole field): haze {np.sqrt(np.mean(haze[fine_region] ** 2)):.4f} rms | residual rms over the field "
          f"{residual_rms(nf_image - final_fit['model'], fine_region):.4f} -> {level_log[0]['rms']:.4f}, over §22's vessels "
          f"{residual_rms(nf_image - final_fit['model'], band_region):.4f} -> {level_log[0]['band_rms']:.4f}")

    # levels 1.., coarse to fine: the boxes that fail free the coefficients there; one vessel at a time, largest first
    boxes_seen = []  # per level: its box side and every box's ratio before it, for the figure below
    _departures = [{"shift": np.zeros(len(geometry_of(vessel)["z"])), "width": np.zeros(len(geometry_of(vessel)["z"]))}
                  for vessel in _vessels]  # per vessel: its shift and log width moved from §22's, along it
    for _level, (_blur, _box, _spacing) in enumerate(LEVELS, start=1):
        _started = time.time()
        _variance = noise_variance(_blur)
        inverse_variance = 1 / np.maximum(_variance, 1e-12)
        leftover = (nf_image - haze - _fit["model"]).astype(np.float64)  # the residual; a vessel's own part is added back
        blurred_residual = fine_blur_valid(leftover, _blur)
        ratio = box_ratios(blurred_residual, _variance, _box)
        _failing = failing_pixels(ratio, _box)
        boxes_seen.append((f"before level {_level}: {_box} px boxes", _box, ratio))
        _labels = cores_picture(_vessels)
        freed, _changed, _gain, _moved = np.zeros(3, int), 0, 0.0, []
        for _number in np.argsort([-np.mean(geometry_of(vessel)["width"]) for vessel in _vessels]):
            _geometry = geometry_of(_vessels[_number])
            own_pixels, own_values = vessel_part(_fit, _number)
            leftover.ravel()[own_pixels] += own_values  # the density without the haze and every other vessel
            _z = _geometry["z"]
            _density = {"picture": leftover, "weight": inverse_variance,
                       "amplitude": _fit["amplitudes"][_number](np.clip(_z, _z[0], _z[-1] - 1e-9))}
            _result = refine_on_level(_geometry, _departures[_number], {"amplitude": _density}, _failing, _labels, _number, _blur,
                                     _spacing)
            if _result is None:
                leftover.ravel()[own_pixels] -= own_values
                continue
            leftover.ravel()[_result["pixels"]] -= _result["values"]["amplitude"]  # its new model, before the next vessel looks
            _vessels[_number] = {**_vessels[_number], "geometry": _result["geometry"]}
            _departures[_number] = {"shift": _departures[_number]["shift"] + _result["shift"],
                                  "width": np.log(_result["geometry"]["width"] / geometry_of(final_vessels[_number])["width"])}
            freed += _result["freed"]
            _changed += 1
            _gain += _result["gain"]
            _moved.append(_result["moved"])
        del leftover
        _fit = fit_field(_vessels, target=nf_image - haze)  # every amplitude and junction again, jointly, on the new geometry
        haze = estimate_haze(nf_image - _fit["model"], vessel_cover(_fit) < HAZE_CLEAR)
        _fit = fit_field(_vessels, _fit["bases"], target=nf_image - haze)
        _after = box_ratios(fine_blur_valid(nf_image - haze - _fit["model"], _blur), _variance, _box)
        known = np.isfinite(ratio) & np.isfinite(_after)
        level_log.append({"level": _level, "blur": _blur, "box": _box, "spacing": _spacing, "boxes": int(known.sum()),
                          "failing_before": float(np.mean(ratio[known] >= FAIL_RATIO)),
                          "failing_after": float(np.mean(_after[known] >= FAIL_RATIO)),
                          "median_ratio_before": float(np.median(ratio[known])),
                          "median_ratio_after": float(np.median(_after[known])),
                          "vessels": _changed, "freed": dict(zip(FAMILIES, freed.tolist())), "gain": _gain,
                          "moved": float(np.mean(_moved)) if _moved else 0.0,
                          "rms": residual_rms(nf_image - haze - _fit["model"], fine_region),
                          "band_rms": residual_rms(nf_image - haze - _fit["model"], band_region),
                          "width_drift": width_drift(_vessels), "seconds": time.time() - _started})
        _entry = level_log[-1]
        print(f"level {_level} (blur {_blur:g} px, boxes {_box} px, knots every {_spacing:g} px): boxes failing "
              f"{_entry['failing_before'] * 100:.0f}% -> {_entry['failing_after'] * 100:.0f}% (median ratio "
              f"{_entry['median_ratio_before']:.1f} -> {_entry['median_ratio_after']:.1f}) | {_changed} vessels corrected, "
              f"coefficients freed: " + ", ".join(f"{name} {count}" for name, count in _entry["freed"].items())
              + f" | centres moved {_entry['moved']:.2f} px, widths changed {_entry['width_drift'] * 100:.0f}% from §22's | "
              f"residual rms {_entry['rms']:.4f} over the field, {_entry['band_rms']:.4f} over §22's vessels ({_entry['seconds']:.0f} s)")

    fine_vessels, fine_fit, fine_counts = settle(_vessels, target=nf_image - haze)
    unique_share(fine_fit)
    fine_residual = (nf_image - haze - fine_fit["model"]).astype(np.float32)
    final_variance = noise_variance(LEVELS[-1][0])
    final_ratio = box_ratios(fine_blur_valid(fine_residual, LEVELS[-1][0]), final_variance, LEVELS[-1][1])
    misfit_ratio = ratio_picture(final_ratio, LEVELS[-1][1])  # how far from the noise each place still is
    boxes_seen.append((f"after the levels and settling: {LEVELS[-1][1]} px boxes", LEVELS[-1][1], final_ratio))
    known = np.isfinite(final_ratio)
    print(f"settled: {fine_counts['after cutting']} pieces after cutting the faint stretches, {len(fine_vessels)} after "
          f"dropping {fine_counts['dropped']}; residual rms {residual_rms(fine_residual, fine_region):.4f} over the field "
          f"(§22: {residual_rms(nf_image - final_fit['model'], fine_region):.4f}), {residual_rms(fine_residual, band_region):.4f} "
          f"over §22's vessels (§22: {residual_rms(nf_image - final_fit['model'], band_region):.4f}); widths change "
          f"{width_roughness(fine_vessels) * 100:.1f}% every 10 px along a vessel (§22: {width_roughness(final_vessels) * 100:.1f}%); "
          f"still failing at the finest level: "
          f"{np.mean(final_ratio[known] >= FAIL_RATIO) * 100:.0f}% of the {LEVELS[-1][1]} px boxes, median ratio "
          f"{np.median(final_ratio[known]):.0f} "
          f"({(time.time() - fine_started) / 60:.1f} min in all)")
    junction_table(fine_fit)

    # where the residual lives, scale by scale: its energy over the noise's at each blur, before (§22) and after
    _covered = np.zeros(nf_image.shape, bool)
    _covered.ravel()[np.union1d(final_fit["pixels"], fine_fit["pixels"])] = True
    profiles = {"§22: density - model": scale_profile((nf_image - final_fit["model"]).astype(np.float64), fine_region),
                "§23: density - haze - model": scale_profile(fine_residual.astype(np.float64), fine_region)}
    print("residual energy over noise energy, over the field, at blurs " + ", ".join(f"{blur:g}" for blur in SCALES) + " px:")
    for _name, _values in profiles.items():
        print(f"   {_name:28s} " + " ".join(f"{value:8.1f}" for value in _values))
    _figure, _axis = plt.subplots(figsize=(8, 4))
    _places = np.arange(len(SCALES))
    for _place, (_name, _values) in enumerate(profiles.items()):
        _axis.plot(_places, _values, color=SERIES[_place], linewidth=2, marker="o", markersize=8, label=_name)
    _axis.axhline(1, color=INK_MUTED, linewidth=1, linestyle=":")
    _axis.text(_places[-1], 1.15, "noise alone", color=INK_MUTED, fontsize=9, ha="right", va="bottom")
    _axis.axhline(FAIL_RATIO, color=INK_MUTED, linewidth=1, linestyle="--")
    _axis.text(_places[-1], FAIL_RATIO * 1.15, "a box fails above this", color=INK_MUTED, fontsize=9, ha="right", va="bottom")
    _axis.set_yscale("log")
    _axis.set_xticks(_places, [f"{_blur:g}" for _blur in SCALES])
    _axis.set_xlabel("blur (px)")
    _axis.set_ylabel("residual energy / noise energy")
    _axis.set_title("Where the residual lives: its energy over the noise's, at each blur, over the field")
    _axis.legend(fontsize=9)
    _figure.tight_layout()
    save_figure(_figure, "fine_scales", banded_key)
    plt.show()

    # every level's boxes: each box's residual energy over the noise's before the level, and after the last one
    columns = (len(boxes_seen) + 1) // 2
    _figure, _axes = plt.subplots(2, columns, figsize=(16, 10.4 * 2 / columns), squeeze=False)
    _judged = np.concatenate([ratio[np.isfinite(ratio)] for _, _, ratio in boxes_seen])
    norm = matplotlib.colors.LogNorm(vmin=max(FAIL_RATIO, np.percentile(_judged, 1)), vmax=np.percentile(_judged, 99))
    for _axis, (_title, _box, ratio) in zip(_axes.ravel(), boxes_seen):
        _image = _axis.imshow(ratio_picture(ratio, _box), cmap="Oranges", norm=norm)
        known = np.isfinite(ratio)
        _axis.set_title(f"{_title}: {np.mean(ratio[known] >= FAIL_RATIO) * 100:.0f}% fail, median ratio "
                       f"{np.median(ratio[known]):.0f}", fontsize=10)
        _axis.axis("off")
    for _axis in _axes.ravel()[len(boxes_seen):]:
        _axis.axis("off")
    _figure.colorbar(_image, ax=_axes, fraction=0.02, pad=0.01, label="residual energy / noise energy, per box")
    save_figure(_figure, "fine_boxes", banded_key)
    plt.show()

    # before and after: §22's residual, the haze, and what is left after the levels
    _colormap = plt.get_cmap("RdBu_r").copy()
    _colormap.set_bad("#2c2c2a")
    _figure, _axes = plt.subplots(3, 1, figsize=(11, 20.5))
    for _axis, (_title, _picture) in zip(_axes, (("§22: density - model", nf_image - final_fit["model"]),
                                             ("the haze", haze),
                                             ("§23: density - haze - model", fine_residual))):
        _image = _axis.imshow(np.where(_covered, _picture, np.nan), cmap=_colormap, vmin=-RESIDUAL_RANGE, vmax=RESIDUAL_RANGE)
        _axis.set_title(f"{_title}: rms {np.sqrt(np.mean(_picture[fine_region] ** 2)):.4f} over the field", fontsize=10)
        _axis.axis("off")
        _figure.colorbar(_image, ax=_axis, fraction=0.03, pad=0.01, label="optical density")
    _figure.tight_layout()
    save_figure(_figure, "fine_residual", banded_key)
    plt.show()

    # close-ups where the levels cut the residual most (96 px windows, apart from each other, away from the field's edge):
    # the density, §22's residual, the residual after the haze alone, and after the levels
    inner = cv2.erode(nf_inside.astype(np.uint8), np.ones((97, 97), np.uint8), borderValue=0) > 0
    gain_picture = cv2.blur((haze_residual.astype(np.float64) ** 2 - fine_residual.astype(np.float64) ** 2) * _covered, (96, 96))
    gain_picture[~inner] = -np.inf
    _places = []
    for flat in np.argsort(gain_picture, axis=None)[::-1]:
        _y, _x = np.divmod(flat, nf_image.shape[1])
        if all(np.hypot(_x - px, _y - py) >= 200 for px, py in _places):
            _places.append((_x, _y))
        if len(_places) == 3:
            break
    _figure, _axes = plt.subplots(len(_places), 4, figsize=(12.4, 3.1 * len(_places)), squeeze=False)
    for _row, (_x, _y) in enumerate(_places):
        _window = (slice(_y - 48, _y + 48), slice(_x - 48, _x + 48))
        _panels = (("density", nf_image, "gray", DENSITY_RANGE),
                  ("§22: density - model", nf_image - final_fit["model"], "RdBu_r", (-RESIDUAL_RANGE / 2, RESIDUAL_RANGE / 2)),
                  ("after the haze alone", haze_residual, "RdBu_r", (-RESIDUAL_RANGE / 2, RESIDUAL_RANGE / 2)),
                  ("after the levels", fine_residual, "RdBu_r", (-RESIDUAL_RANGE / 2, RESIDUAL_RANGE / 2)))
        for _column, (_title, _picture, _colormap, (_low, _high)) in enumerate(_panels):
            _axes[_row, _column].imshow(_picture[_window], cmap=_colormap, vmin=_low, vmax=_high)
            _axes[_row, _column].set_title(_title if _row == 0 else "", fontsize=9)
            _axes[_row, _column].axis("off")
    _figure.tight_layout()
    save_figure(_figure, "fine_closeups", banded_key)
    plt.show()
    return (
        HAZE_CLEAR,
        HAZE_ROUNDS,
        box_ratios,
        estimate_haze,
        failing_pixels,
        fine_fit,
        fine_region,
        fine_vessels,
        haze,
        noise_variance,
        refine_on_level,
        residual_rms,
        time,
        vessel_cover,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 24 · Flow: the second picture

    The mean density cannot tell blood from anything else that absorbs: a vessel and the haze beside it look alike. Blood moves, though, and the rest does not. Wherever cells move, each frame differs from the mean, so each pixel's variance over time is a second picture of the vessels: of the moving blood alone.

    **The flow picture.** Each pixel's variance over the frames with data, about its own mean, less the *floor*: the variance that noise alone gives. The floor is a local mean, `FLOW_FLOOR_SIGMA` wide, of the variance over the quiet background pixels of §11, where nothing moves. A pixel that fewer than `FLOW_MIN_FRAMES` of the frames saw, in the halo of a glare, is left out.

    **What it looks like, measured on §23's geometry.**
    - Across a vessel, the flow has the same profile as the density. Their second-moment widths, in units of §23's σ, are 0.89 against 0.91 on thin vessels, 0.86 against 0.95 on medium ones and 0.88 against 0.92 on wide ones. So each vessel's flow is drawn with its own profile, times a flow amplitude along it: a spline, as the density's is. The flow pictures of overlapping vessels add up.
    - Where nothing flows it is quiet. Between the vessels, where there is no haze, the variance is the background pixels' own.
    - Where there is haze it is not. Pixels with a density of only 0.02–0.06 between the vessels vary 25% more than the background, and those of 0.06–0.12 vary 54% more: capillaries too fine to trace. The flow gets its own haze, as the density did.
    - The residual motion left by §8 hardly shows. A term for it, the gradient of the mean squared, explains next to nothing.

    **The joint fit.** Every vessel's shift and width are fitted to both pictures at once, with an amplitude of its own in each, by the machinery of §23 (`refine_on_level` takes a picture per amplitude). Each picture counts by how well the model fits it, one over its mean square misfit, times `FLOW_WEIGHT` for the flow. The penalties are §23's, anchored to §23's geometry.

    **A held-out check.** The fit sees the flow of the even frames only. The odd frames' flow has its own noise, so a geometry that had only fitted the even frames' noise would fit the odd frames no better. Both are reported, for §23's geometry and the joint one.

    ### What it found

    - **The flow picture.** Noise alone gives a variance of 3.0·10⁻⁴ OD² per frame (0.017 OD). In the vessels' cores the flow adds a median 0.84 times that. The picture's own noise, from the odd frames against the even, is 6.7·10⁻⁵.
    - **The vessels explain most of it.** With the flow's haze they explain 81% of the flow picture's energy, and noise is 5%. The rest, about a seventh, is flowing structure that no vessel accounts for (third panel of the first figure): red lines where something flows that was never traced, blue where a vessel was drawn wider or brighter than its blood.
    - **The flow agrees with §23's geometry.** Fitted jointly to both pictures (`FLOW_LEVELS = ((0.0, 32, 8.0),)`), the vessels' widths changed by a median of 0%, within ±5% for 80% of them, and their centres moved 0.1 px. The flow's misfit fell 2% on the held-out odd frames, as much as on the fitted even ones, so the change is real, not noise. But it is small, and the density's misfit rose 1.4%. The joint levels are therefore off by default, and the vessels keep §23's geometry.
    - **Every vessel's flow.** Flow per unit density varies about 8-fold between vessels, from 8.1·10⁻⁴ to 6.1·10⁻³ (10% to 90%). 14 of the 285 flow less than a fifth of the median (second figure). Either they carried little blood during the burst, or they were traced along something that is not a lumen; several lie right beside a large vessel.

    | Output | Content |
    |---|---|
    | `flows` | the flow pictures, variance less floor, of `"all"` the frames, the `"even"` and the `"odd"` ones |
    | `flow_vessels`, `flow_fit`, `joint_haze` | the vessels, the density's fit and the haze that §25 uses: §23's, or with `FLOW_LEVELS`, those of the joint fit after settling |
    | `flow_per_density`, `still` | per vessel: its median flow over its median density amplitude, and whether that is below `NO_FLOW_SHARE` of the median |
    | `flow_amplitude_fit`, `flow_haze` | every vessel's flow amplitude along it (`["amplitudes"]`), fitted on all the frames, and the flow between the vessels |
    | `flow_residual` | the flow the vessels do not explain: flowing structure that was never traced |
    | `flow_check` | the residuals of the density and of the even and odd frames' flow, on §23's geometry and on the joint one |
    """)
    return


@app.cell
def _(
    DENSITY_RANGE,
    HAZE_CLEAR,
    HAZE_ROUNDS,
    NF_FIXATION,
    SERIES,
    banded_key,
    box_ratios,
    cores_picture,
    cv2,
    estimate_haze,
    failing_pixels,
    final_density_fixations,
    fine_fit,
    fine_region,
    fine_vessels,
    fit_field,
    geometry_of,
    haze,
    matplotlib,
    nf_background,
    nf_data,
    nf_image,
    nf_inside,
    noise_variance,
    np,
    plt,
    refine_on_level,
    residual_rms,
    save_figure,
    settle,
    time,
    unique_share,
    vessel_cover,
    vessel_part,
):
    FLOW_FLOOR_SIGMA = 32.0  # pixels; the floor is a local mean, this wide, of the variance on the quiet background pixels
    FLOW_LEVELS = ()         # the joint levels, as §23's LEVELS, e.g. ((0.0, 32, 8.0),); none: §23's geometry is kept
    FLOW_WEIGHT = 1.0        # in the joint levels: how much the flow counts against the density, each relative to its misfit
    NO_FLOW_SHARE = 0.2      # a vessel flowing less than this share of the median flow per unit density is marked as still
    FLOW_MIN_FRAMES = 0.9    # a pixel is in the flow picture only if at least this share of the frames saw it


    def frame_variances(frames):
        """Per pixel, the variance over time of the density frames, over the frames with data there: of all of them, of the
        even ones and of the odd ones, each about its own mean; and how many frames saw it ("frames")."""
        sums = np.zeros((2, 2) + frames[0].shape)  # even and odd frames: their sum and their sum of squares
        counts = np.zeros((2,) + frames[0].shape)
        for index, frame in enumerate(frames):
            frame = np.asarray(frame, np.float64)
            ok = np.isfinite(frame)
            value = np.where(ok, frame, 0)
            sums[index % 2, 0] += value
            sums[index % 2, 1] += value ** 2
            counts[index % 2] += ok

        def variance(total, squares, count):
            return np.where(count > 1, (squares - total ** 2 / np.maximum(count, 1)) / np.maximum(count - 1, 1), np.nan)

        return {"all": variance(sums[:, 0].sum(axis=0), sums[:, 1].sum(axis=0), counts.sum(axis=0)),
                "even": variance(sums[0, 0], sums[0, 1], counts[0]), "odd": variance(sums[1, 0], sums[1, 1], counts[1]),
                "frames": counts.sum(axis=0)}


    def local_mean(values, mask, sigma):
        """A local mean, `sigma` px wide, of `values` over the pixels of `mask`; where those are scarce (under a fifth of the
        neighbourhood), a mean four times as wide fills in."""
        mask = mask.astype(np.float64)
        values = np.nan_to_num(np.asarray(values, np.float64)) * mask
        near = cv2.GaussianBlur(mask, (0, 0), sigma)
        narrow = cv2.GaussianBlur(values, (0, 0), sigma) / np.maximum(near, 1e-6)
        wide = cv2.GaussianBlur(values, (0, 0), 4 * sigma) / np.maximum(cv2.GaussianBlur(mask, (0, 0), 4 * sigma), 1e-6)
        trust = np.clip(near / 0.2, 0, 1)
        return trust * narrow + (1 - trust) * wide


    def flow_fit_of(vessels, bases, flow, rounds=HAZE_ROUNDS):
        """Every vessel's flow amplitude along it, fitted jointly to a flow picture on the vessels' geometry (§20's fit,
        every overlap adding up), in turns with the flow's own haze. Returns the fit and the flow haze."""
        flow_haze = np.zeros(nf_image.shape, np.float32)
        fit = fit_field(vessels, bases, junctions=False, target=flow)
        for _ in range(rounds):
            flow_haze = estimate_haze(flow - fit["model"], vessel_cover(fit) < HAZE_CLEAR)
            fit = fit_field(vessels, fit["bases"], junctions=False, target=flow - flow_haze)
        return fit, flow_haze


    def flow_misfits(name, vessels, fit, flow_fit, flow_haze):
        """The residual rms over the field of the density, and of the even (fitted) and odd (held out) frames' flow."""
        flow_check[name] = {"density": residual_rms(nf_image - joint_haze - fit["model"], fine_region),
                            "even": residual_rms(flows["even"] - flow_haze - flow_fit["model"], fine_region),
                            "odd": residual_rms(flows["odd"] - flow_haze - flow_fit["model"], fine_region)}


    flow_started = time.time()
    variances = frame_variances(final_density_fixations[NF_FIXATION])
    flow_seen = nf_data & (variances.pop("frames") >= FLOW_MIN_FRAMES * len(final_density_fixations[NF_FIXATION]))  # not glare's halo
    flows = {}  # the flow pictures: each pixel's variance over time, less the floor that noise alone gives
    for key, _variance in variances.items():
        _floor = local_mean(_variance, nf_background & flow_seen & np.isfinite(_variance), FLOW_FLOOR_SIGMA)
        flows[key] = np.where(flow_seen & np.isfinite(_variance), _variance - _floor, 0).astype(np.float32)
        if key == "all":
            flow_floor = np.median(_floor[nf_background])
    del variances
    flow_noise = np.sqrt(np.mean((flows["odd"] - flows["even"])[fine_region] ** 2)) / 2  # the noise of the all-frames picture
    cores = (vessel_cover(fine_fit) >= 0.5) & nf_inside
    print(f"Flow picture: noise alone gives a variance of {flow_floor:.2e} OD² per frame; in the vessels' cores the flow adds "
          f"a median {np.median(flows['all'][cores]) / flow_floor:.2f} times that. The picture's own noise (odd frames against "
          f"even) is {flow_noise:.1e}; {np.mean(~flow_seen[fine_region]):.1%} of the field is left out, seen by too few frames "
          f"({time.time() - flow_started:.0f} s)")

    # the flow on §23's geometry: every vessel's flow amplitude, fitted on the even frames, judged on the odd ones
    flow_check = {}
    joint_haze = haze  # §23's haze; the joint levels estimate it again
    even_fit, even_haze = flow_fit_of(fine_vessels, fine_fit["bases"], flows["even"])
    flow_misfits("§23's geometry", fine_vessels, fine_fit, even_fit, even_haze)

    # the joint levels (none by default): every vessel's shift and width fitted to both pictures at once, one vessel at a
    # time, largest first; the density and the even frames' flow each count by how well the model fits it
    _vessels, _fit, flow_haze = list(fine_vessels), fine_fit, even_haze
    _departures = [{"shift": np.zeros(len(geometry_of(_vessel)["z"])), "width": np.zeros(len(geometry_of(_vessel)["z"]))}
                  for _vessel in _vessels]  # moved from §23's geometry
    for _level, (_blur, _box, _spacing) in enumerate(FLOW_LEVELS, start=1):
        assert _blur == 0, "the flow picture is a variance: it is never blurred"
        _started = time.time()
        leftovers = {"amplitude": (nf_image - joint_haze - _fit["model"]).astype(np.float64),
                     "flow": (flows["even"] - flow_haze - even_fit["model"]).astype(np.float64)}
        weights = {"amplitude": 1 / np.mean(leftovers["amplitude"][fine_region] ** 2),
                   "flow": FLOW_WEIGHT / np.mean(leftovers["flow"][fine_region] ** 2)}
        _failing = failing_pixels(box_ratios(leftovers["amplitude"], noise_variance(0.0), _box), _box)
        _labels = cores_picture(_vessels)
        _changed, _moved = 0, []
        for _number in np.argsort([-np.mean(geometry_of(_vessel)["width"]) for _vessel in _vessels]):
            _geometry = geometry_of(_vessels[_number])
            _z = np.clip(_geometry["z"], _geometry["z"][0], _geometry["z"][-1] - 1e-9)
            _parts = {"amplitude": vessel_part(_fit, _number), "flow": vessel_part(even_fit, _number)}
            _pictures = {}
            for _name, (_pixels, _values) in _parts.items():
                leftovers[_name].ravel()[_pixels] += _values  # each picture without every other vessel
                _pictures[_name] = {"picture": leftovers[_name], "weight": weights[_name],
                                  "amplitude": (_fit if _name == "amplitude" else even_fit)["amplitudes"][_number](_z)}
            _result = refine_on_level(_geometry, _departures[_number], _pictures, _failing, _labels, _number, _blur, _spacing)
            if _result is None:
                for _name, (_pixels, _values) in _parts.items():
                    leftovers[_name].ravel()[_pixels] -= _values
                continue
            for _name in _parts:
                leftovers[_name].ravel()[_result["pixels"]] -= _result["values"][_name]
            _vessels[_number] = {**_vessels[_number], "geometry": _result["geometry"]}
            _departures[_number] = {"shift": _departures[_number]["shift"] + _result["shift"],
                                  "width": np.log(_result["geometry"]["width"] / geometry_of(fine_vessels[_number])["width"])}
            _changed += 1
            _moved.append(_result["moved"])
        del leftovers
        _fit = fit_field(_vessels, target=nf_image - joint_haze)
        joint_haze = estimate_haze(nf_image - _fit["model"], vessel_cover(_fit) < HAZE_CLEAR)
        _fit = fit_field(_vessels, _fit["bases"], target=nf_image - joint_haze)
        even_fit, flow_haze = flow_fit_of(_vessels, _fit["bases"], flows["even"])
        print(f"joint level {_level} (knots every {_spacing:g} px): {_changed} vessels corrected, centres moved "
              f"{np.mean(_moved):.2f} px ({time.time() - _started:.0f} s)")
    if FLOW_LEVELS:
        flow_misfits("joint geometry", _vessels, _fit, even_fit, flow_haze)
        flow_vessels, flow_fit, flow_counts = settle(_vessels, target=nf_image - joint_haze)
        unique_share(flow_fit)
    else:
        flow_vessels, flow_fit = fine_vessels, fine_fit
    print("residual rms over the field (the flow's amplitudes fitted on the even frames):")
    for _name, _values in flow_check.items():
        print(f"   {_name:15s}: density {_values['density']:.4f} | flow, even frames (fitted) {_values['even']:.3e}, odd frames "
              f"(held out) {_values['odd']:.3e}")

    # every vessel's flow, from all the frames, and its flow per unit density
    flow_amplitude_fit, flow_haze = flow_fit_of(flow_vessels, flow_fit["bases"], flows["all"])
    flow_residual = (flows["all"] - flow_haze - flow_amplitude_fit["model"]).astype(np.float32)
    explained_flow = 1 - np.mean(flow_residual[fine_region] ** 2) / np.mean(flows["all"][fine_region] ** 2)
    flow_per_density = np.zeros(len(flow_vessels))
    for _number, _vessel in enumerate(flow_vessels):
        _z = geometry_of(_vessel)["z"]
        _z = np.clip(_z, _z[0], _z[-1] - 1e-9)
        flow_per_density[_number] = (np.median(flow_amplitude_fit["amplitudes"][_number](_z))
                                    / max(np.median(flow_fit["amplitudes"][_number](_z)), 1e-4))
    still = flow_per_density < NO_FLOW_SHARE * np.median(flow_per_density)
    print(f"the vessels and the flow's haze explain {explained_flow:.0%} of the flow picture's energy; its noise is "
          f"{flow_noise ** 2 / np.mean(flows['all'][fine_region] ** 2):.0%} of it | flow per unit density: median "
          f"{np.median(flow_per_density):.2e}, from {np.percentile(flow_per_density, 10):.2e} to "
          f"{np.percentile(flow_per_density, 90):.2e} (10% to 90%); {still.sum()} of {len(flow_vessels)} vessels flow less than "
          f"{NO_FLOW_SHARE:g} of the median ({(time.time() - flow_started) / 60:.1f} min in all)")

    # the density, the flow, and the flow the vessels do not explain
    flow_range = np.percentile(flows["all"][cores], 95)
    _figure, _axes = plt.subplots(3, 1, figsize=(11, 20.5))
    for _axis, (_title, _picture, _colormap, (_low, _high)) in zip(_axes, (
            ("the density (time average)", nf_image, "gray", DENSITY_RANGE),
            ("the flow: each pixel's variance over time, less what noise alone gives", flows["all"], "gray", (0, flow_range)),
            ("the flow the vessels and the flow's haze do not explain", flow_residual, "RdBu_r", (-flow_range / 2, flow_range / 2)))):
        _image = _axis.imshow(np.where(nf_data if _axis is _axes[0] else flow_seen, _picture, np.nan), cmap=_colormap, vmin=_low,
                            vmax=_high)
        _axis.set_title(_title, fontsize=10)
        _axis.axis("off")
        _figure.colorbar(_image, ax=_axis, fraction=0.03, pad=0.01, label="OD" if _colormap == "gray" and _axis is _axes[0] else "OD² per frame")
    _figure.tight_layout()
    save_figure(_figure, "flow_pictures", banded_key)
    plt.show()

    # every vessel by its flow per unit density; those that hardly flow stand out
    _figure, _axis = plt.subplots(figsize=(16, 10.4))
    _axis.imshow(nf_image, cmap="gray", vmin=DENSITY_RANGE[0], vmax=DENSITY_RANGE[1])
    flow_colors = plt.get_cmap("Blues")
    _top = np.percentile(flow_per_density, 95)
    for _number, _vessel in enumerate(flow_vessels):
        _centre = geometry_of(_vessel)["centre"]
        if still[_number]:
            _axis.plot(_centre[:, 0], _centre[:, 1], color=SERIES[1], linewidth=2.4)
        else:
            _axis.plot(_centre[:, 0], _centre[:, 1], color=flow_colors(0.35 + 0.65 * np.clip(flow_per_density[_number] / _top, 0, 1)),
                      linewidth=1.4)
    _axis.plot([], [], color=SERIES[1], linewidth=2.4, label=f"flowing less than {NO_FLOW_SHARE:g} of the median ({still.sum()})")
    _figure.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, _top), cmap=matplotlib.colors.ListedColormap(
        flow_colors(np.linspace(0.35, 1, 64)))), ax=_axis, fraction=0.02, pad=0.01, label="flow per unit density")
    _axis.set_title(f"{len(flow_vessels)} vessels by their flow per unit density", fontsize=10)
    _axis.legend(loc="lower left", fontsize=9, frameon=True, facecolor="black", edgecolor="none", framealpha=0.7,
                labelcolor="white")
    _axis.axis("off")
    _figure.tight_layout()
    save_figure(_figure, "flow_vessels", banded_key)
    plt.show()
    return flow_amplitude_fit, flow_fit, flow_per_density, flow_vessels, joint_haze


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 25 · The vessel graph

    The pipeline's result, from §23 and §24, in plain variables:

    | Output | Content |
    |---|---|
    | `vessel_table` | per vessel: its centreline (`centre`, n × 2), `width` (σ, px), profile `shape`, `amplitude` (optical density), `flow` (variance over time) and `z` at every pixel along it, its `length`, and its `flow_per_density` |
    | `junctions` | per junction: its two `vessels` (indices into `vessel_table`), its place `at`, its `kind` (X, T or V), the `angle` between the vessels, its `share` and its `verdict`: crossing, joining or unclear |
    | `vessel_model` | the model of the whole field as a picture: every vessel and junction rendered in optical density; the density is `vessel_model + joint_haze` plus what is left |

    A junction's verdict is **unclear** in either of two cases: merging could change the model there by less than 5 noise levels (the fainter vessel is too faint to tell), or the share's standard error is above 0.25.

    The figures:

    - **The vessel graph.** Every vessel coloured by its width, every junction by its verdict.
    - **Junction close-ups.** The clearest crossings and joinings: the density, the model, and the residual (density − haze − model) with the junction terms and with every overlap simply added up (the same vessels, refitted without junction terms).
    - **The shares** by junction kind.
    - **What the model still misses.** §17's orientation layer run on the residual, density − haze − model, in the same noise units as on the density. Line structure that the vessels do not explain still makes it fire: a vessel never traced, a faint capillary, or a vessel whose width or centre is off (a pair of thin lines along its edges).
    """)
    return


@app.cell
def _(
    DENSITY_RANGE,
    INK_MUTED,
    RESIDUAL_RANGE,
    SEED_LEVEL,
    SERIES,
    banded_key,
    banded_orientation_field,
    fit_field,
    flow_amplitude_fit,
    flow_fit,
    flow_per_density,
    flow_vessels,
    geometry_of,
    joint_haze,
    junction_table,
    matplotlib,
    nf_image,
    nf_inside,
    np,
    plt,
    save_figure,
    strongest_response,
):
    _vessels, junctions = flow_vessels, flow_fit["junctions"]
    vessel_model = flow_fit["model"]
    vessel_table = []
    for _number, _vessel in enumerate(_vessels):
        _geometry = geometry_of(_vessel)
        vessel_table.append({"centre": _geometry["centre"], "width": _geometry["width"], "shape": _geometry["shape"],
                             "amplitude": flow_fit["amplitudes"][_number](_geometry["z"]), "z": _geometry["z"],
                             "flow": flow_amplitude_fit["amplitudes"][_number](_geometry["z"]),
                             "flow_per_density": float(flow_per_density[_number]),
                             "length": float(np.ptp(_geometry["z"]))})
    _lengths = np.array([_row["length"] for _row in vessel_table])
    widths = np.array([np.median(_row["width"]) for _row in vessel_table])
    print(f"{len(_vessels)} vessels, {_lengths.sum():.0f} px of centreline: length median {np.median(_lengths):.0f} px "
          f"(10% shorter than {np.percentile(_lengths, 10):.0f} px), width median {np.median(widths):.1f} px")
    print(f"{len(junctions)} junctions; where the two vessels' cores overlap, the fit decided:")
    junction_table(flow_fit)

    # with and without the junction terms: the same vessels refitted as if every overlap simply added up
    additive_fit = fit_field(_vessels, flow_fit["bases"], junctions=False, target=nf_image - joint_haze)
    at_junctions = np.unique(np.concatenate([junction["pixels"] for junction in junctions]))
    for _title, _model in (("with the junction terms", vessel_model), ("every overlap additive", additive_fit["model"])):
        _residual = (nf_image - joint_haze - _model).ravel()[at_junctions]
        print(f"   {_title:24s}: residual rms {np.sqrt(np.mean(_residual ** 2)):.4f} over the {len(at_junctions)} junction pixels, "
              f"mean {_residual.mean():+.4f}")

    # the vessel graph: every vessel coloured by its width, every junction by the fit's verdict
    VERDICT_STYLES = {"crossing": dict(marker="o", markerfacecolor="none", markeredgecolor=SERIES[3], markersize=7,
                                       markeredgewidth=1.6),
                      "joining": dict(marker="o", markerfacecolor=SERIES[4], markeredgecolor="black", markersize=6,
                                      markeredgewidth=0.6),
                      "unclear": dict(marker="x", color=INK_MUTED, markersize=5)}
    _figure, _axis = plt.subplots(figsize=(16, 10.4))
    _axis.imshow(nf_image, cmap="gray", vmin=DENSITY_RANGE[0], vmax=DENSITY_RANGE[1])
    width_colors = plt.get_cmap("Blues")
    largest = np.percentile(widths, 98)
    for _row in vessel_table:
        _share = np.clip(np.median(_row["width"]) / largest, 0, 1)
        _axis.plot(_row["centre"][:, 0], _row["centre"][:, 1], color=width_colors(0.35 + 0.65 * _share),
                  linewidth=1.0 + 2.0 * _share)
    for _verdict, _style in VERDICT_STYLES.items():
        _places = np.array([junction["at"] for junction in junctions if junction["verdict"] == _verdict]).reshape(-1, 2)
        _axis.plot(_places[:, 0], _places[:, 1], linestyle="none", label=f"{_verdict} ({len(_places)})", **_style)
    _figure.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, largest), cmap=matplotlib.colors.ListedColormap(
        width_colors(np.linspace(0.35, 1, 64)))), ax=_axis, fraction=0.02, pad=0.01, label="median width (sigma, px)")
    _axis.set_title(f"{len(_vessels)} vessels and {len(junctions)} junctions; a crossing's densities add up, a joining's lumens "
                   f"merge", fontsize=10)
    _axis.legend(loc="lower left", fontsize=9, frameon=True, facecolor="black", edgecolor="none", framealpha=0.7,
                labelcolor="white")
    _axis.axis("off")
    _figure.tight_layout()
    save_figure(_figure, "vessel_graph", banded_key)
    plt.show()

    # close-ups of the clearest crossings and joinings, apart from each other: the density, the model, and the residual
    # with and without the junction terms
    def clearest(verdict, count=3, apart=150):
        """The junctions of one verdict where merging would change the model most, at least `apart` px from each other."""
        chosen = []
        for junction in sorted((junction for junction in junctions if junction["verdict"] == verdict),
                               key=lambda junction: -junction["effect"]):
            if all(np.hypot(*(junction["at"] - other["at"])) >= apart for other in chosen):
                chosen.append(junction)
            if len(chosen) == count:
                break
        return chosen



    _shown = clearest("crossing") + clearest("joining")
    _figure, _axes = plt.subplots(len(_shown), 4, figsize=(12.4, 3.1 * len(_shown)), squeeze=False)
    for _row, junction in enumerate(_shown):
        cx, cy = np.round(junction["at"]).astype(int)
        half = int(max(20, 6 * max(np.median(vessel_table[k]["width"]) for k in junction["vessels"])))
        _window = (slice(max(cy - half, 0), cy + half), slice(max(cx - half, 0), cx + half))
        _panels = (("density", nf_image, "gray", DENSITY_RANGE), ("model", vessel_model, "gray", DENSITY_RANGE),
                  ("density - haze - model", nf_image - joint_haze - vessel_model, "RdBu_r", (-RESIDUAL_RANGE, RESIDUAL_RANGE)),
                  ("the same, all additive", nf_image - joint_haze - additive_fit["model"], "RdBu_r",
                   (-RESIDUAL_RANGE, RESIDUAL_RANGE)))
        for _column, (_title, _picture, _colormap, (_low, _high)) in enumerate(_panels):
            _axes[_row, _column].imshow(_picture[_window], cmap=_colormap, vmin=_low, vmax=_high)
            _axes[_row, _column].set_title(_title if _column else f"{junction['verdict']} ({junction['kind']}, "
                                        f"{junction['angle']:.0f}°): share {junction['share']:.2f}", fontsize=9)
            _axes[_row, _column].axis("off")
    _figure.tight_layout()
    save_figure(_figure, "junction_closeups", banded_key)
    plt.show()

    # the fit's share at every junction, by its shape: X (both vessels go on through), T (one ends there), V (both end)
    _figure, _axis = plt.subplots(figsize=(9, 4))
    bins = np.linspace(0, 1, 41)
    for _place, kind in enumerate("XTV"):
        shares = [junction["share"] for junction in junctions if junction["kind"] == kind]
        _axis.hist(np.clip(shares, bins[0], bins[-1]), bins=bins, histtype="step", linewidth=1.8, color=SERIES[_place],
                  label=f"{kind} ({len(shares)})")
    _axis.axvline(0.5, color=INK_MUTED, linewidth=1, linestyle=":")
    _axis.set_xlabel("share: the fitted overlap over the fainter vessel's amplitude (0 adds up, 1 merges)")
    _axis.set_ylabel("junctions")
    _axis.legend(fontsize=9)
    _figure.tight_layout()
    save_figure(_figure, "junction_shares", banded_key)
    plt.show()

    # what the model still misses: the orientation layer of §17 run on the residual, in the same noise units as on the
    # density. Line structure the vessels do not explain (a vessel never traced, or one misplaced) still makes it fire.
    missed = banded_orientation_field((nf_image - joint_haze - vessel_model).astype(np.float32))["scores"].max(axis=(0, 1))
    missed = missed.astype(np.float32)
    for _title, _response in (("on the density", strongest_response), ("on the residual", missed)):
        print(f"the orientation layer {_title:19s}: fires above SEED_LEVEL ({SEED_LEVEL:g} noise units) on "
              f"{np.mean(_response[nf_inside] >= SEED_LEVEL) * 100:4.1f}% of the field, its 99th percentile "
              f"{np.percentile(_response[nf_inside], 99):5.1f}")
    _top = np.percentile(strongest_response[nf_inside], 99)
    _figure, _axes = plt.subplots(1, 2, figsize=(16, 5.4))
    for _axis, (_title, _response) in zip(_axes, (("the orientation layer on the density", strongest_response),
                                              ("the orientation layer on density - haze - model: what the vessels miss", missed))):
        _image = _axis.imshow(_response, cmap="gray", vmin=0, vmax=_top)
        _figure.colorbar(_image, ax=_axis, fraction=0.03, pad=0.01, label="strongest neuron (noise units)")
        _axis.set_title(_title, fontsize=10)
        _axis.axis("off")
    _figure.tight_layout()
    save_figure(_figure, "missed_structure", banded_key)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
