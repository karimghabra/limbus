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
    # Archive: vessel annotation, sections 17–25

    **A different vessel annotation algorithm is planned.** These sections are kept here for reference and are no longer developed. The preprocessing they build on, sections 0–16, is [`foundation.py`](foundation.py), which ends by saving its result for whatever annotation comes next.

    Sections 17–25 are those of the foundation notebook before it was split, unchanged, with their own section numbers: the orientation field (17), the hyperspline (18), tracing (19), the whole field as one system with its junctions (20), continuations (21), refinement (22), coarse to fine (23), flow (24) and the vessel graph (25). Before them come the notes that led into them, also unchanged. Where their texts say §0–§16, they mean the sections of `foundation.py`, whose numbers have not changed.

    **Running it.** The first cells below run `foundation.py` inside this notebook (`app.embed()`) and pass on the 10 names that sections 17–25 read from it; nothing else is added. Every other name of the preprocessing, such as `cleaned_smears`, `combined_sums`, `fixations`, `anchor` or `temporal_frame`, is reached as `preprocessing.defs["cleaned_smears"]` and so on. So *Run all* here runs the preprocessing first, on the burst it finds (or `LIMBUS_BURST`), and then sections 17–25. Running this notebook and the foundation at once on the live server takes the memory of two runs.
    """)
    return


@app.cell
def _():
    # what sections 17-25 import for themselves; the foundation itself is imported, to be run in the next cell
    import hashlib
    import json
    import os
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

    from foundation import app as foundation

    return Path, ThreadPoolExecutor, cv2, foundation, hashlib, json, np, os, plt, scipy


@app.cell
async def _(foundation):
    # runs sections 0-16 of foundation.py, with their figures and previews
    preprocessing = await foundation.embed()
    preprocessing.output
    return (preprocessing,)


@app.cell
def _(preprocessing):
    # the names sections 17-25 read from the preprocessing
    DENSITY_RANGE = preprocessing.defs["DENSITY_RANGE"]
    INK_MUTED = preprocessing.defs["INK_MUTED"]
    SERIES = preprocessing.defs["SERIES"]
    final_background_fits = preprocessing.defs["final_background_fits"]
    final_density_fixations = preprocessing.defs["final_density_fixations"]
    final_density_means = preprocessing.defs["final_density_means"]
    masked_mean = preprocessing.defs["masked_mean"]
    quiet_pixels = preprocessing.defs["quiet_pixels"]
    robust_sd = preprocessing.defs["robust_sd"]
    workers = preprocessing.defs["workers"]
    return (
        DENSITY_RANGE,
        INK_MUTED,
        SERIES,
        final_background_fits,
        final_density_fixations,
        final_density_means,
        masked_mean,
        quiet_pixels,
        robust_sd,
        workers,
    )


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
