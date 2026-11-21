"""Target statistics of a burst, measured the same way on a real recording and on a synthetic one.

Every function here takes frames as the camera wrote them (12-bit DN after the camera's gamma LUT), so the
synthetic clips go through exactly the estimator the real ones do and the comparison table compares like with
like. Nothing here knows whether a burst is real.

measure(frames, fps, um_per_px) returns a flat dict of named statistics plus a few arrays for the figures:

| group      | statistics                                                                              |
|------------|-----------------------------------------------------------------------------------------|
| motion     | frame-to-frame step (median, p90), saccades, cumulative excursion, drift spectrum slope  |
| flicker    | the frame-mean periodogram: strongest line (Hz, amplitude), white residual               |
| noise      | temporal SD on the background against its mean (a photon-transfer line in DN)            |
| blur       | per-frame sharpness spread                                                               |
| texture    | band-pass SDs of the vessel-free background of the mean frame, illumination p5/p95        |
| morphology | vessel area and length density, junction density, diameters, segment lengths,            |
|            | tortuosity, branch angles, the Murray exponent at junctions, contrast by diameter         |
| flow       | the temporal spectrum of the mean-subtracted video in vessels against background         |
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import apply_hysteresis_threshold, sato
from skimage.morphology import remove_small_objects, skeletonize

TEXTURE_SIGMAS = (0.7, 1.4, 2.8, 5.6, 11.2)          # px, the DoG bands of the background texture
DIAMETER_BINS = ((2, 5), (5, 9), (9, 15), (15, 25), (25, 40), (40, 100))   # px
FLICKER_BANDS = ((0.5, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 16.0))         # Hz


@dataclass
class Burst:
    frames: np.ndarray          # (T, H, W) float32 DN
    fps: float
    exposure_s: float | None
    name: str


def load_burst(folder, crop=None, max_frames=None, first=0) -> Burst:
    """Read a recorder burst folder (frame_NNNNNN.tif, manifest.json, optional frames.csv).

    crop is (top, left, height, width) or None. The frame rate is the camera clock's when frames.csv has it,
    else the camera's reported resulting_fps, else the host-time effective rate."""
    import tifffile

    folder = Path(folder)
    files = sorted(folder.glob("frame_*.tif"))[first:]
    if max_frames:
        files = files[:max_frames]
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    fps = None
    table = folder / "frames.csv"
    if table.exists():
        with table.open(newline="") as handle:
            stamps = [float(row["camera_timestamp_ns"]) for row in csv.DictReader(handle) if row.get("camera_timestamp_ns")]
        if len(stamps) > 2:
            fps = 1e9 / float(np.median(np.diff(stamps)))
    state = manifest.get("camera_state_at_start", {})
    fps = fps or state.get("resulting_fps") or manifest.get("capture", {}).get("effective_fps")
    exposure = state.get("exposure_us")
    first_frame = tifffile.imread(files[0])
    if crop is None:
        crop = (0, 0) + first_frame.shape
    top, left, height, width = crop
    frames = np.empty((len(files), height, width), np.float32)
    for index, path in enumerate(files):
        frames[index] = tifffile.imread(path)[top:top + height, left:left + width]
    return Burst(frames, float(fps), exposure * 1e-6 if exposure else None, folder.name)


# ----------------------------------------------------------------------------------------------- helpers

def _bandpass(image, low=2.0, high=12.0):
    image = image.astype(np.float32)
    return cv2.GaussianBlur(image, (0, 0), low) - cv2.GaussianBlur(image, (0, 0), high)


def _blur_masked(image, mask, sigma):
    """Gaussian blur over the pixels in mask only (normalised convolution)."""
    weight = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)
    total = cv2.GaussianBlur(np.where(mask, image, 0).astype(np.float32), (0, 0), sigma)
    return total / np.maximum(weight, 1e-6)


def _robust_sd(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def _pct(values, q):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, q)) if values.size else float("nan")


# ----------------------------------------------------------------------------------------------- motion

def frame_steps(frames, scale=0.5):
    """Translation between consecutive frames by phase correlation of band-passed, downscaled frames.

    Returns (steps (T-1, 2) in full-resolution px as (dx, dy), responses (T-1,))."""
    window = None
    previous = None
    steps, responses = [], []
    for frame in frames:
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        band = _bandpass(np.log(np.maximum(small, 1.0)), 1.0, 8.0)
        if window is None:
            window = cv2.createHanningWindow(band.shape[::-1], cv2.CV_32F)
        if previous is not None:
            (dx, dy), response = cv2.phaseCorrelate(previous, band, window)
            steps.append((dx / scale, dy / scale))
            responses.append(response)
        previous = band
    return np.array(steps, float).reshape(-1, 2), np.array(responses, float)


def fixations(steps, min_frames=20):
    """Split a burst at saccades (a step above median + 5 MAD-SDs, and at least 12 px). Returns [(start, stop)]."""
    size = np.hypot(steps[:, 0], steps[:, 1])
    limit = max(12.0, np.median(size) + 5 * _robust_sd(size))
    cuts = [0] + [int(i) + 1 for i in np.flatnonzero(size > limit)] + [len(steps) + 1]
    runs = [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b - a >= min_frames]
    return runs, limit


def align(frames, reference_index=None, scale=0.5):
    """Register every frame to one reference frame (translation, phase correlation) and warp it there.

    Returns (aligned float32 stack with NaN outside the data, shifts (T, 2))."""
    if reference_index is None:
        reference_index = len(frames) // 2

    def band(frame):
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return _bandpass(np.log(np.maximum(small, 1.0)), 1.0, 8.0)

    reference = band(frames[reference_index])
    window = cv2.createHanningWindow(reference.shape[::-1], cv2.CV_32F)
    height, width = frames.shape[1:]
    aligned = np.empty_like(frames, dtype=np.float32)
    shifts = np.zeros((len(frames), 2))
    ones = np.ones((height, width), np.float32)
    for index, frame in enumerate(frames):
        (dx, dy), _ = cv2.phaseCorrelate(reference, band(frame), window)
        dx, dy = dx / scale, dy / scale
        shifts[index] = dx, dy
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        warped = cv2.warpAffine(frame.astype(np.float32), matrix, (width, height), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        valid = cv2.warpAffine(ones, matrix, (width, height), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP) > 0.999
        aligned[index] = np.where(valid, warped, np.nan)
    return aligned, shifts


def motion_stats(steps, fps):
    size = np.hypot(steps[:, 0], steps[:, 1])
    runs, limit = fixations(steps)
    path = np.vstack([[0.0, 0.0], np.cumsum(steps, axis=0)])
    keep = size <= limit
    out = {
        "motion step median (px)": float(np.median(size[keep])) if keep.any() else float("nan"),
        "motion step p90 (px)": _pct(size[keep], 90),
        "saccades per s": float((~keep).sum() / (len(steps) / fps)),
        "excursion x range (px)": float(np.ptp(path[:, 0])),
        "excursion y range (px)": float(np.ptp(path[:, 1])),
    }
    # spectrum of the drift velocity within fixations: slope of log power against log frequency
    velocity = steps[keep]
    if len(velocity) > 16:
        spectrum = np.abs(np.fft.rfft(velocity - velocity.mean(0), axis=0)) ** 2
        spectrum = spectrum.sum(1)
        frequency = np.fft.rfftfreq(len(velocity), 1 / fps)
        use = frequency > 0
        slope = np.polyfit(np.log(frequency[use]), np.log(spectrum[use] + 1e-12), 1)[0]
        out["motion velocity spectrum slope"] = float(slope)
        out["_motion_spectrum"] = (frequency, spectrum)
    out["_steps"] = steps
    out["_fixations"] = runs
    return out


# ----------------------------------------------------------------------------------------------- flicker

def flicker_stats(frames, fps, pad=16):
    """The illumination flicker: the strongest line in the spectrum of the relative frame mean over the central half
    of the frame. The spectrum is zero-padded pad-fold, so a line between two frequency bins is not under-read
    (without padding a line half a bin off loses a third of its amplitude)."""
    height, width = frames.shape[1:]
    centre = frames[:, height // 4: 3 * height // 4, width // 4: 3 * width // 4]
    means = centre.reshape(len(frames), -1).mean(1)
    relative = means / means.mean() - 1
    n = len(relative)
    t = np.arange(n)
    relative = relative - np.polyval(np.polyfit(t, relative, 1), t)
    transform = np.fft.rfft(relative, n * pad)
    spectrum = np.abs(transform) * 2 / n                      # sinusoid amplitude
    frequency = np.fft.rfftfreq(n * pad, 1 / fps)
    usable = frequency >= 1.5 * fps / n                       # above the slowest drift the detrending leaves
    peak = int(np.flatnonzero(usable)[np.argmax(spectrum[usable])])
    line = spectrum[peak] * np.cos(2 * np.pi * frequency[peak] * t / fps + np.angle(transform[peak]))
    coarse = np.abs(np.fft.rfft(relative)) * 2 / n
    return {
        "flicker line frequency (Hz)": float(frequency[peak]),
        "flicker line amplitude (%)": float(100 * spectrum[peak]),
        "flicker residual SD (%)": float(100 * np.std(relative - line)),
        "_flicker_spectrum": (np.fft.rfftfreq(n, 1 / fps), coarse),
    }


# ----------------------------------------------------------------------------------------------- the still picture

def vessel_picture(mean_frame, valid, closing_radius=35, background_sigma=12):
    """Optical density of the vessels in a mean frame: ln(background) - ln(frame).

    The background is a grey closing (which removes dark structures narrower than about twice closing_radius)
    smoothed over the valid pixels."""
    log = np.log(np.maximum(np.nan_to_num(mean_frame, nan=float(np.nanmedian(mean_frame))), 1.0)).astype(np.float32)
    small = cv2.resize(log, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_radius + 1, closing_radius + 1))
    closed = cv2.resize(cv2.morphologyEx(small, cv2.MORPH_CLOSE, small_kernel), log.shape[::-1], interpolation=cv2.INTER_LINEAR)
    background = _blur_masked(closed, valid, background_sigma)
    od = np.where(valid, background - log, np.nan)
    return od.astype(np.float32), background


def vessel_mask(od, valid, low=3.0, high=7.0, min_size=120):
    """Vessels: hysteresis on the smoothed optical density (thresholds in robust SDs of the darkest-free half of
    the field) gated by a ridge filter, small pieces dropped and small holes filled."""
    smooth = cv2.GaussianBlur(np.nan_to_num(od), (0, 0), 1.0)
    lower = smooth[valid & (smooth < np.nanmedian(smooth[valid]))]
    spread = _robust_sd(np.r_[lower, 2 * np.median(lower) - lower])
    centre = float(np.median(smooth[valid]))
    ridges = sato(smooth, sigmas=(1.5, 3, 6, 10), black_ridges=False)
    ridge_gate = ridges > np.percentile(ridges[valid], 50)
    mask = apply_hysteresis_threshold(np.where(valid, smooth, 0), centre + low * spread, centre + high * spread)
    mask &= ridge_gate | (smooth > centre + 2 * high * spread)
    mask &= valid
    mask = _drop_small(mask, min_size)
    mask = ~_drop_small(~mask, 60)
    return mask


def _drop_small(mask, size):
    """Remove connected pieces smaller than size pixels (scikit-image 0.24 and 0.26 name the argument differently)."""
    try:
        return remove_small_objects(mask, max_size=size - 1)
    except TypeError:
        return remove_small_objects(mask, min_size=size)


_NEIGHBOURS = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])


def skeleton_graph(mask, prune=10):
    """Skeleton of a vessel mask as segments between junctions and ends.

    Returns (skeleton, segments: list of (n, 2) row/col arrays in order, junctions: list of (row, col) centres,
    junction_segments: list of lists of (segment index, which end))."""
    skeleton = skeletonize(mask)
    for _ in range(3):                                     # prune short spurs, three passes
        count = ndi.convolve(skeleton.astype(np.uint8), _NEIGHBOURS, mode="constant") * skeleton
        junction = count >= 3
        pieces, n = ndi.label(skeleton & ~ndi.binary_dilation(junction, np.ones((3, 3))), np.ones((3, 3)))
        if n == 0:
            break
        sizes = ndi.sum(np.ones_like(pieces), pieces, index=np.arange(1, n + 1))
        ends = (count == 1)
        has_end = ndi.maximum(ends, pieces, index=np.arange(1, n + 1)).astype(bool)
        drop = np.flatnonzero(has_end & (sizes < prune)) + 1
        if drop.size == 0:
            break
        skeleton &= ~np.isin(pieces, drop)
        skeleton = skeletonize(skeleton)
    count = ndi.convolve(skeleton.astype(np.uint8), _NEIGHBOURS, mode="constant") * skeleton
    junction_pixels = count >= 3
    junction_zone = ndi.binary_dilation(junction_pixels, np.ones((3, 3))) & skeleton
    clusters, n_junctions = ndi.label(junction_zone, np.ones((3, 3)))
    junctions = ndi.center_of_mass(junction_zone, clusters, range(1, n_junctions + 1)) if n_junctions else []
    pieces, n_pieces = ndi.label(skeleton & ~junction_zone, np.ones((3, 3)))
    segments = []
    junction_segments = [[] for _ in range(n_junctions)]
    objects = ndi.find_objects(pieces)
    grown_clusters = ndi.grey_dilation(clusters, size=(3, 3))
    for label_index, box in enumerate(objects, start=1):
        if box is None:
            continue
        rows, cols = np.nonzero(pieces[box] == label_index)
        points = np.stack([rows + box[0].start, cols + box[1].start], 1)
        ordered = _order_points(points)
        segment_index = len(segments)
        segments.append(ordered)
        for which, (r, c) in ((0, ordered[0]), (1, ordered[-1])):
            cluster = grown_clusters[r, c]
            if cluster:
                junction_segments[cluster - 1].append((segment_index, which))
    return skeleton, segments, [np.array(j) for j in junctions], junction_segments


def _order_points(points):
    """Order the pixels of one skeleton piece from one end to the other (greedy walk from an end)."""
    if len(points) <= 2:
        return points
    lookup = {(int(r), int(c)): i for i, (r, c) in enumerate(points)}
    degree = np.zeros(len(points), int)
    for i, (r, c) in enumerate(points):
        degree[i] = sum((r + dr, c + dc) in lookup for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)
    start = int(np.argmin(degree))
    order, seen = [start], {start}
    current = start
    while True:
        r, c = points[current]
        best = None
        for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            j = lookup.get((int(r + dr), int(c + dc)))
            if j is not None and j not in seen:
                best = j
                break
        if best is None:
            break
        order.append(best)
        seen.add(best)
        current = best
    return points[order]


def _segment_length(points):
    if len(points) < 2:
        return float(len(points))
    return float(np.hypot(*np.diff(points, axis=0).T).sum())


def morphology_stats(od, mask, valid, um_per_px):
    skeleton, segments, junctions, junction_segments = skeleton_graph(mask)
    distance = ndi.distance_transform_edt(mask)
    area_px = float(valid.sum())
    area_mm2 = area_px * (um_per_px * 1e-3) ** 2
    lengths = np.array([_segment_length(s) for s in segments])
    diameters = 2 * distance[skeleton]
    weights = np.ones_like(diameters)
    out = {
        "vessel area fraction": float(mask[valid].mean()),
        "length density (mm/mm2)": float(lengths.sum() * um_per_px * 1e-3 / area_mm2),
        "junction density (/mm2)": float(len(junctions) / area_mm2),
        "segments per mm2": float(len(segments) / area_mm2),
    }
    for q in (5, 25, 50, 75, 95):
        out[f"diameter p{q} (um)"] = float(np.percentile(np.repeat(diameters, weights.astype(int)), q) * um_per_px) if diameters.size else float("nan")
    long_enough = lengths >= 10
    for q in (25, 50, 75, 95):
        out[f"segment length p{q} (um)"] = _pct(lengths[long_enough] * um_per_px, q)
    tortuosity = []
    for points, length in zip(segments, lengths):
        if length >= 20:
            chord = float(np.hypot(*(points[-1] - points[0])))
            if chord > 0:
                tortuosity.append(length / chord)
    out["tortuosity median"] = _pct(tortuosity, 50)
    out["tortuosity p90"] = _pct(tortuosity, 90)
    # branch geometry at three-way junctions
    angles, exponents = [], []
    for centre, members in zip(junctions, junction_segments):
        if len(members) != 3:
            continue
        directions, widths = [], []
        for segment_index, which in members:
            points = segments[segment_index]
            if len(points) < 8:
                break
            run = points[:15] if which == 0 else points[::-1][:15]
            vector = run[-1] - centre
            norm = np.hypot(*vector)
            if norm == 0:
                break
            directions.append(vector / norm)
            widths.append(float(np.median(2 * distance[run[4:15, 0], run[4:15, 1]])) if len(run) > 5 else np.nan)
        if len(directions) != 3:
            continue
        pair = [float(np.degrees(np.arccos(np.clip(directions[i] @ directions[j], -1, 1))))
                for i, j in ((0, 1), (0, 2), (1, 2))]
        widths = np.array(widths)
        order = np.argsort(widths)[::-1]
        parent = order[0]
        # the angle between the two daughters (the pair not involving the widest segment)
        daughters = [i for i in range(3) if i != parent]
        index = {(0, 1): 0, (0, 2): 1, (1, 2): 2}[tuple(sorted(daughters))]
        angles.append(pair[index])
        d0, d1, d2 = widths[order]
        if d2 > 1.5 and d0 < d1 + d2:
            low, high = 1.0, 8.0
            for _ in range(40):
                mid = 0.5 * (low + high)
                if d1 ** mid + d2 ** mid > d0 ** mid:
                    low = mid
                else:
                    high = mid
            exponents.append(0.5 * (low + high))
    out["daughter angle median (deg)"] = _pct(angles, 50)
    out["junction exponent median"] = _pct(exponents, 50)
    contrast = 1 - np.exp(-np.nan_to_num(od[skeleton]))
    diameter_px = 2 * distance[skeleton]
    for low, high in DIAMETER_BINS:
        select = (diameter_px >= low) & (diameter_px < high)
        out[f"contrast d {low * um_per_px:.0f}-{high * um_per_px:.0f} um"] = float(np.median(contrast[select])) if select.sum() > 20 else float("nan")
    out["_skeleton"] = skeleton
    out["_diameters_um"] = diameters * um_per_px
    out["_segments"] = segments
    return out


def texture_stats(mean_frame, mask, valid):
    """Band-pass SDs of ln(mean frame) on the background well away from any vessel, and the illumination."""
    log = np.log(np.maximum(np.nan_to_num(mean_frame, nan=1.0), 1.0)).astype(np.float32)
    background = valid & ~ndi.binary_dilation(mask, iterations=6)
    out = {}
    previous = None
    for sigma in (0.0,) + TEXTURE_SIGMAS:
        level = log if sigma == 0 else _blur_masked(log, background, sigma)
        if previous is not None:
            band = previous - level
            strict = ndi.binary_erosion(background, iterations=max(2, int(round(3 * sigma))))
            out[f"texture band {sigma:g} px (SD x1000)"] = 1000 * _robust_sd(band[strict])
        previous = level
    large = _blur_masked(log, background, 60)
    values = np.exp(large[valid])
    out["illumination p5/p95"] = float(np.percentile(values, 5) / np.percentile(values, 95))
    return out


def noise_stats(aligned, mask, valid):
    """Temporal noise on the background: var of frame-to-frame differences / 2, against the mean level (DN)."""
    background = valid & ~ndi.binary_dilation(mask, iterations=6)
    mean = np.nanmean(aligned, axis=0)
    difference = np.diff(aligned, axis=0)
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            variance = np.nanmean(difference ** 2, axis=0) / 2
    levels = mean[background]
    values = variance[background]
    good = np.isfinite(levels) & np.isfinite(values)
    levels, values = levels[good], values[good]
    edges = np.percentile(levels, np.linspace(2, 98, 9))
    centres, medians = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        select = (levels >= a) & (levels < b)
        if select.sum() > 50:
            centres.append(float(np.median(levels[select])))
            medians.append(float(np.median(values[select])))
    slope, intercept = np.polyfit(centres, medians, 1) if len(centres) >= 3 else (np.nan, np.nan)
    return {
        "background level (DN)": float(np.median(levels)),
        "background temporal SD (DN)": float(np.sqrt(np.median(values))),
        "noise variance slope (DN)": float(slope),
        "noise variance offset (DN2)": float(intercept),
    }


def blur_stats(frames):
    """Sharpness per frame: energy of the finest band (sigma 0.7 to 1.4 px) relative to the 1.4 to 5 px band, so
    illumination and noise level cancel; reported as the spread over the burst."""
    values = []
    for frame in frames[:: max(1, len(frames) // 60)]:
        log = np.log(np.maximum(frame, 1.0)).astype(np.float32)
        fine = cv2.GaussianBlur(log, (0, 0), 0.7) - cv2.GaussianBlur(log, (0, 0), 1.4)
        mid = cv2.GaussianBlur(log, (0, 0), 1.4) - cv2.GaussianBlur(log, (0, 0), 5.0)
        values.append(float(np.std(fine) / np.std(mid)))
    values = np.array(values)
    return {
        "sharpness median": float(np.median(values)),
        "sharpness p10/p90": float(np.percentile(values, 10) / np.percentile(values, 90)),
    }


def flow_signal_stats(aligned, mask, skeleton, valid, fps, um_per_px):
    """The temporal fluctuation of the mean-subtracted video: in vessels by diameter against background, and its
    spectrum (the red-cell flicker)."""
    log = np.log(np.maximum(aligned, 1.0))
    centred = log - np.nanmean(log, axis=0)
    frame_offset = np.nanmedian(centred[:, valid & ~mask], axis=1)      # exposure and illumination flicker
    centred -= frame_offset[:, None, None]
    variance = np.nanmean(centred ** 2, axis=0)
    background = valid & ~ndi.binary_dilation(mask, iterations=6)
    noise = float(np.nanmedian(variance[background]))
    distance = ndi.distance_transform_edt(mask)
    diameter = 2 * distance
    out = {}
    for low, high in DIAMETER_BINS:
        select = skeleton & (diameter >= low) & (diameter < high) & valid
        if select.sum() > 20:
            out[f"flicker SD ratio d {low * um_per_px:.0f}-{high * um_per_px:.0f} um"] = float(np.sqrt(np.nanmedian(variance[select]) / noise))
    vessel = skeleton & valid & np.isfinite(variance)
    series_v = centred[:, vessel]
    series_b = centred[:, background][:, :: max(1, background.sum() // 20000)]
    series_v = np.nan_to_num(series_v[:, :: max(1, series_v.shape[1] // 20000)])
    series_b = np.nan_to_num(series_b)
    frequency = np.fft.rfftfreq(len(log), 1 / fps)
    power_v = (np.abs(np.fft.rfft(series_v, axis=0)) ** 2).mean(1)
    power_b = (np.abs(np.fft.rfft(series_b, axis=0)) ** 2).mean(1)
    excess = power_v - power_b
    for low, high in FLICKER_BANDS:
        select = (frequency >= low) & (frequency < high)
        out[f"flicker excess share {low:g}-{high:g} Hz"] = float(np.clip(excess[select], 0, None).sum() / max(np.clip(excess[1:], 0, None).sum(), 1e-12))
    lag = np.nanmean(series_v[1:] * series_v[:-1]) / np.nanmean(series_v ** 2)
    out["vessel lag-1 autocorrelation"] = float(lag)
    out["_flicker_power"] = (frequency, power_v, power_b)
    return out


def measure(frames, fps, um_per_px, crop_border=24, label=""):
    """Every target statistic of one burst. frames: (T, H, W) DN as the camera wrote them."""
    frames = np.asarray(frames, np.float32)
    steps, _ = frame_steps(frames)
    out = {"label": label, "frames": len(frames), "fps": float(fps)}
    out.update(motion_stats(steps, fps))
    out.update(flicker_stats(frames, fps))
    out.update(blur_stats(frames))
    runs = out["_fixations"] or [(0, len(frames))]
    start, stop = max(runs, key=lambda run: run[1] - run[0])
    aligned, _ = align(frames[start:stop])
    coverage = np.isfinite(aligned).mean(0)
    valid = coverage >= 0.95
    valid[:crop_border] = valid[-crop_border:] = False
    valid[:, :crop_border] = valid[:, -crop_border:] = False
    mean_frame = np.nanmean(aligned, axis=0)
    od, _ = vessel_picture(mean_frame, valid)
    mask = vessel_mask(od, valid)
    out["fixation frames"] = int(stop - start)
    out.update(texture_stats(mean_frame, mask, valid))
    out.update(noise_stats(aligned, mask, valid))
    morphology = morphology_stats(od, mask, valid, um_per_px)
    out.update(morphology)
    out.update(flow_signal_stats(aligned, mask, morphology["_skeleton"], valid, fps, um_per_px))
    out["_mean_frame"] = mean_frame
    out["_od"] = od
    out["_mask"] = mask
    out["_valid"] = valid
    return out


def public(stats):
    """The scalar statistics only (names without a leading underscore)."""
    return {k: v for k, v in stats.items() if not k.startswith("_")}
