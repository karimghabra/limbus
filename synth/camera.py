"""The camera: from red cells in 3D tissue to 12-bit frames, as the recorder writes them.

Image formation (docs §7):

1. Eye and camera motion. The tissue moves relative to the camera by a translation and a small rotation that
   follow the measured fixational motion: a mean-reverting random walk whose frame-to-frame step matches the real
   bursts, with occasional saccades (synth.targets measures both on real and synthetic bursts alike). The focal
   plane wanders in depth as the head and the camera move.
2. Absorption. Every red cell is splatted, at its projected position, into one of several depth slabs on a grid
   twice as fine as the sensor's, spread over the red-cell footprint. Its haemoglobin gives an optical depth per
   spectral band, tau_k = mu_k V_rbc / (H_ref A_pixel) per cell [Prahl, Bosschaart2014], and the light that
   crosses a slab is transmitted by T = sum_k w_k exp(-2 tau_k) (the light goes through the vessel to the sclera
   and back: a double pass). The weights w_k are the detected spectrum (the source times the sensor's response):
   broadband light, of which the weakly absorbed red part passes even a thick vessel, which is why large vessels
   saturate at a contrast near 0.5 rather than going black (measured, docs §7.1).
3. Depth. Detected light returns from a range of depths: a share 1 - exp(-z / l) of it turns back above depth z
   and never meets a vessel there. So deeper vessels lose contrast, and each slab is blurred by the camera's
   point-spread function, by defocus proportional to its distance from the focal plane, and by the scattering of
   the tissue above it, which grows with depth [Jacques2013]. Vessels at different depths therefore cross over
   and under one another with different sharpness, as in the recordings.
4. The scleral texture (a static random field with the band-pass spectrum measured on the real bursts) moves with
   the tissue; the illumination falls off towards the edges of the field and flickers at the mains alias
   measured on the real bursts.
5. Motion blur: each frame averages sub-exposures spread over the exposure time, so moving cells and a moving eye
   both smear.
6. The sensor (IMX249 as measured on the real bursts): photon shot noise and read noise in electrons, the gain in
   DN per electron, the camera's gamma 0.8 lookup table, 12-bit quantisation, the ceiling at 4091 DN and the
   even/odd row offset.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .anatomy import Scene
from .targets import TEXTURE_SIGMAS

SLAB_EDGES = (0.0, 90.0, 140.0, 200.0, 260.0, 340.0, 440.0, 560.0, 720.0, 1100.0)   # µm below the surface
HAEMATOCRIT_REF = 0.45       # the haematocrit at which the whole-blood absorption coefficients are given


@dataclass
class Motion:
    """Per-frame tissue motion relative to the camera, in pixels and radians, and the focal depth."""
    dx: np.ndarray
    dy: np.ndarray
    rotation: np.ndarray
    focus: np.ndarray            # µm below the surface
    saccades: np.ndarray         # frame indices where a saccade happened

    def at(self, t_frames):
        """Interpolated (dx, dy, rotation, focus) at a fractional frame index."""
        i = np.arange(len(self.dx))
        return (float(np.interp(t_frames, i, self.dx)), float(np.interp(t_frames, i, self.dy)),
                float(np.interp(t_frames, i, self.rotation)), float(np.interp(t_frames, i, self.focus)))


def eye_motion(n_frames, scene: Scene, rng) -> Motion:
    """Fixational motion: a mean-reverting random walk with the measured step size, saccades at the measured rate,
    a slow rotation and a wandering focal plane."""
    o = scene.optics
    sigma = o.motion_step_px / np.sqrt(2 * np.log(2))          # per-axis SD whose 2D median step is motion_step_px
    reversion = sigma / max(o.motion_bound_px, 1.0)
    x = np.zeros(n_frames)
    y = np.zeros(n_frames)
    saccades = []
    for k in range(1, n_frames):
        step = rng.normal(0, sigma, 2) - reversion * np.array([x[k - 1], y[k - 1]])
        refractory = saccades and (k - saccades[-1]) < 0.25 * o.fps     # saccades are separated in time (assumed 0.25 s)
        if rng.random() < o.saccade_rate_hz / o.fps and not refractory:
            amplitude = rng.uniform(*o.saccade_px)
            angle = rng.uniform(0, 2 * np.pi)
            step += amplitude * np.array([np.cos(angle), np.sin(angle)])
            saccades.append(k)
        x[k], y[k] = x[k - 1] + step[0], y[k - 1] + step[1]
    rotation = np.radians(o.rotation_deg) * np.cumsum(rng.normal(0, 1, n_frames)) / np.sqrt(max(n_frames, 1))
    focus = o.focus_depth_um + o.focus_wander_um * _smooth(rng, n_frames, 12.0)
    return Motion(x, y, rotation, focus, np.array(saccades, int))


def _smooth(rng, n, width):
    raw = rng.standard_normal(n + 6 * int(width) + 2)
    kernel = np.exp(-0.5 * (np.arange(-3 * int(width), 3 * int(width) + 1) / width) ** 2)
    out = np.convolve(raw, kernel, mode="same")[3 * int(width) + 1:3 * int(width) + 1 + n]
    # unit SD in expectation (the filter's gain), not the sample's: a short burst would otherwise inflate it
    return out / np.sqrt((kernel ** 2).sum())


TEXTURE_COMPONENT_SIGMAS = (0.0,) + tuple(TEXTURE_SIGMAS) + (30.0,)


def texture_field(shape, scene: Scene, rng, amplitudes=None):
    """A static scleral texture in log units: the sum of independent band-limited Gaussian fields, one per band
    between successive TEXTURE_COMPONENT_SIGMAS (the bands synth.targets measures, plus a 11-30 px mottle), each
    scaled to unit SD and then by its amplitude (x1e-3; Optics.texture_amplitudes, fitted by calibrate_texture)."""
    amplitudes = np.asarray(amplitudes if amplitudes is not None else scene.optics.texture_amplitudes, float) * 1e-3
    field = np.zeros(shape, np.float32)
    sigmas = TEXTURE_COMPONENT_SIGMAS
    for k in range(1, len(sigmas)):
        if amplitudes[k - 1] == 0:
            continue
        noise = rng.standard_normal(shape).astype(np.float32)
        lo = cv2.GaussianBlur(noise, (0, 0), sigmas[k - 1]) if sigmas[k - 1] > 0 else noise
        band = lo - cv2.GaussianBlur(noise, (0, 0), sigmas[k])
        field += amplitudes[k - 1] * band / max(float(band.std()), 1e-9)
    return field


def illumination(shape, scene: Scene, rng):
    """Smooth illumination falling off from a centre near the middle of the field, scaled so that its 5th to 95th
    percentile ratio matches the measured value."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w * rng.uniform(0.35, 0.65), h * rng.uniform(0.35, 0.65)
    r2 = ((xx - cx) / w) ** 2 + ((yy - cy) / h) ** 2
    target = scene.optics.illumination_p5_p95
    low, high = 0.0, 5.0
    for _ in range(40):
        a = 0.5 * (low + high)
        field = np.exp(-a * r2)
        ratio = np.percentile(field, 5) / np.percentile(field, 95)
        if ratio > target:
            low = a
        else:
            high = a
    return (field / field.max()).astype(np.float32)


class Renderer:
    """Renders frames of one scene from its cells, with the motion, texture and illumination fixed per burst."""

    def __init__(self, scene: Scene, network, rng, motion: Motion, supersample=2):
        self.scene = scene
        self.network = network
        self.rng = rng
        self.motion = motion
        self.ss = supersample
        o = scene.optics
        a = scene.anatomy
        self.shape = (scene.height_px, scene.width_px)
        self.fine = (scene.height_px * supersample, scene.width_px * supersample)
        self.fine_um = a.um_per_px / supersample
        # the texture lives on the tissue: a canvas larger than the field by the motion bound
        pad = int(o.motion_bound_px * 2 + 40)
        self.pad = pad
        self.texture = texture_field((self.shape[0] + 2 * pad, self.shape[1] + 2 * pad), scene, rng)
        self.light = illumination(self.shape, scene, rng)
        self.flicker_phase = rng.uniform(0, 2 * np.pi)
        weights = np.array([w for w, _ in o.spectrum], float)
        self.weights = weights / weights.sum()
        mu = np.array([m for _, m in o.spectrum], float) * 1e-3                     # 1/mm -> 1/µm
        # optical depth one cell adds to one fine pixel, per band (µm^2 / µm^2)
        self.tau_per_cell = mu * scene.physiology.rbc_volume_um3 / HAEMATOCRIT_REF / self.fine_um ** 2
        self.footprint_sigma = 0.5 * scene.physiology.rbc_diameter_um / 2.0 / self.fine_um   # fine px
        edges = np.asarray(SLAB_EDGES)
        self.slab_edges = edges
        self.slab_depth = 0.5 * (edges[:-1] + edges[1:])
        self.returning = np.exp(-edges / o.scatter_veil_length_um)       # share of light returning from below
        x0, x1, y0, y1 = network.field_of_view
        self.origin = np.array([x0, y0])
        self.centre_um = np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1)])
        # the globe's sagitta over the field, for the focus that varies across it: the camera depth of the surface
        # at every pixel (fine grid), split into a few levels for the slab blur
        hf, wf = self.fine
        yy, xx = np.mgrid[0:hf, 0:wf].astype(np.float32)
        surface = np.column_stack([self.origin[0] + (xx.ravel() + 0.5) * self.fine_um,
                                   self.origin[1] + (yy.ravel() + 0.5) * self.fine_um, np.zeros(xx.size)])
        self.surface_depth = self.camera_depth(surface).reshape(hf, wf).astype(np.float32)
        self.sag_levels = np.linspace(float(self.surface_depth.min()), float(self.surface_depth.max()), 4)
        self.glints = self._glints(rng)

    def _glints(self, rng):
        """Specular reflections from the tear film, fixed to the tissue: (x, y, sigma px, peak DN-fraction)."""
        n = self.scene.optics.glints
        h, w = self.shape
        return [(rng.uniform(0, w), rng.uniform(0, h), rng.uniform(1.2, 3.0), rng.uniform(0.4, 0.9)) for _ in range(n)]

    def _globe(self, xyz):
        """Tissue points (x, y along the surface, z depth below it) on the curved globe, seen by the camera:
        returns (X, Y) in µm in the camera's image plane, relative to the field's centre, and the depth along the
        camera axis below the plane tangent to the globe there (µm).

        The surface coordinates are arc lengths from the field's centre (an azimuthal equidistant map of the
        sphere), so a point at arc distance r and depth z lies at polar angle theta = r / R from the camera axis,
        at radius R - z from the eye's centre."""
        R = self.scene.optics.eye_radius_um
        u = xyz[:, 0] - self.centre_um[0]
        v = xyz[:, 1] - self.centre_um[1]
        r = np.hypot(u, v)
        theta = r / R
        rho = R - xyz[:, 2]
        scale = np.where(r > 1e-9, rho * np.sin(theta) / np.maximum(r, 1e-9), 1.0)
        X, Y = u * scale, v * scale
        depth = R - rho * np.cos(theta)
        tx, ty = np.radians(self.scene.optics.camera_tilt_deg)
        depth = depth + X * np.tan(tx) + Y * np.tan(ty)
        return X, Y, depth

    def camera_depth(self, xyz):
        """Depth of tissue points along the camera axis (µm below the tangent plane at the field's centre)."""
        return self._globe(xyz)[2]

    def project(self, xyz, frame_time):
        """Camera pixel coordinates (fine grid) of world points at a fractional frame index."""
        dx, dy, rotation, _ = self.motion.at(frame_time)
        X, Y, _ = self._globe(xyz)
        h0, w0 = self.shape
        p = np.stack([X, Y], 1) / self.scene.anatomy.um_per_px + np.array([w0 / 2, h0 / 2])   # frame-0 pixels
        h, w = self.shape
        c = np.array([w / 2, h / 2])
        cos, sin = np.cos(rotation), np.sin(rotation)
        q = p - c
        q = np.stack([cos * q[:, 0] - sin * q[:, 1], sin * q[:, 0] + cos * q[:, 1]], 1) + c + np.array([dx, dy])
        return q * self.ss + (self.ss - 1) / 2.0

    def _blur_across_field(self, t, depth, focus):
        """Blur a slab's transmission with the blur of its depth at each place in the field: the globe curves away
        from the camera, so the distance from the focal plane grows towards the field's edges. The slab is blurred
        at a few levels of the surface's sagitta and the results blended by each pixel's sagitta."""
        levels = self.sag_levels
        if levels[-1] - levels[0] < 2.0:
            sigma = self.blur_sigma(depth + levels[0], focus)
            return cv2.GaussianBlur(t, (0, 0), sigma) if sigma > 0.3 else t
        blurred = []
        for sag in levels:
            sigma = self.blur_sigma(depth + sag, focus, scatter_depth=depth)
            blurred.append(cv2.GaussianBlur(t, (0, 0), sigma) if sigma > 0.3 else t)
        position = np.clip((self.surface_depth - levels[0]) / (levels[1] - levels[0]), 0, len(levels) - 1 - 1e-6)
        k = position.astype(int)
        w = (position - k).astype(np.float32)
        out = np.zeros_like(t)
        for i in range(len(levels) - 1):
            sel = k == i
            out[sel] = (1 - w[sel]) * blurred[i][sel] + w[sel] * blurred[i + 1][sel]
        return out

    def blur_sigma(self, depth, focus, scatter_depth=None):
        """Blur (fine px) of a slab at a depth with the focal plane at focus."""
        o = self.scene.optics
        psf = o.psf_sigma_px * self.ss
        defocus = o.defocus_um_per_um * abs(depth - focus) / self.fine_um
        tissue = depth if scatter_depth is None else scatter_depth          # scattering follows the tissue depth
        scatter = o.scatter_blur_um_at_500 * (max(tissue, 0.0) / 500.0) ** 1.5 / self.fine_um
        return float(np.sqrt(psf ** 2 + defocus ** 2 + scatter ** 2))

    def transmission(self, xyz, frame_time, weights=None):
        """The fraction of light detected at every fine pixel after the cells' absorption (1 with no cells).
        weights: cells per point (tubes); None is one cell per point."""
        h, w = self.fine
        q = self.project(xyz, frame_time)
        inside = (q[:, 0] >= 0) & (q[:, 0] < w - 1) & (q[:, 1] >= 0) & (q[:, 1] < h - 1)
        q, z = q[inside], xyz[inside, 2]
        weights = None if weights is None else weights[inside]
        slab = np.clip(np.searchsorted(self.slab_edges, z) - 1, 0, len(self.slab_depth) - 1)
        _, _, _, focus = self.motion.at(frame_time)
        detected = np.zeros((h, w), np.float32)
        through = np.ones((h, w), np.float32)                 # product of the slabs above
        all_counts = _splat(q, (h, w), slab, len(self.slab_depth), weights).reshape(len(self.slab_depth), h, w)
        occupied = np.bincount(slab, minlength=len(self.slab_depth)) > 0
        for b in range(len(self.slab_depth)):
            share = self.returning[b] - self.returning[b + 1]
            if b == 0:
                detected += (1 - self.returning[0])
            if occupied[b]:
                counts = cv2.GaussianBlur(all_counts[b], (0, 0), self.footprint_sigma)
                # the exponentials only where there are cells (elsewhere the transmission is 1)
                support = counts > 1e-4
                values = counts[support]
                absorbed = np.zeros(values.shape, np.float32)
                for weight, tau in zip(self.weights, self.tau_per_cell):
                    absorbed += np.float32(weight) * (1.0 - np.exp(np.float32(-self.scene.optics.double_pass * tau) * values))
                t = np.ones((h, w), np.float32)
                t[support] = 1.0 - absorbed
                t = self._blur_across_field(t, self.slab_depth[b], focus)
                through *= t
            detected += share * through
        detected += self.returning[-1] * through
        return detected

    def tissue_maps(self, frame_time):
        """Texture (multiplicative, tissue-fixed) and glints, warped to the camera at a fractional frame index."""
        dx, dy, rotation, _ = self.motion.at(frame_time)
        h, w = self.shape
        c = np.array([w / 2, h / 2])
        cos, sin = np.cos(rotation), np.sin(rotation)
        # camera = R (tissue - c) + c + d; tissue canvas = camera-frame-0 + pad
        matrix = np.array([[cos, -sin, 0.0], [sin, cos, 0.0]])
        matrix[:, 2] = c + np.array([dx, dy]) - matrix[:, :2] @ (c + self.pad)
        texture = cv2.warpAffine(self.texture, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        glint = np.zeros((h, w), np.float32)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        for gx, gy, gs, peak in self.glints:
            px = cos * (gx - c[0]) - sin * (gy - c[1]) + c[0] + dx
            py = sin * (gx - c[0]) + cos * (gy - c[1]) + c[1] + dy
            if -10 < px < w + 10 and -10 < py < h + 10:
                glint += peak * np.exp(-0.5 * ((xx - px) ** 2 + (yy - py) ** 2) / gs ** 2)
        return np.exp(texture), glint

    def expose(self, frame_index, cells_at):
        """Linear irradiance (relative) of one frame: sub-exposures averaged over the exposure time.

        cells_at(t_seconds) must return the cells' world positions at that time."""
        o = self.scene.optics
        period = 1.0 / o.fps
        n_sub = 3
        h, w = self.shape
        total = np.zeros(self.fine, np.float32)
        for j in range(n_sub):
            t_rel = (j + 0.5) / n_sub * o.exposure_s
            got = cells_at(frame_index * period + t_rel)
            xyz, weights = got if isinstance(got, tuple) else (got, None)
            total += self.transmission(xyz, frame_index + t_rel / period, weights)
        total /= n_sub
        detected = cv2.resize(total, (w, h), interpolation=cv2.INTER_AREA)
        texture, glint = self.tissue_maps(frame_index + 0.5 * o.exposure_s / period)
        t = frame_index * period
        flicker = 1.0 + o.flicker_amplitude * np.sin(2 * np.pi * o.flicker_hz * t + self.flicker_phase) \
            + o.flicker_white * self.rng.standard_normal()
        return detected * texture * self.light * flicker, glint * flicker

    def sensor(self, irradiance, glint, background_scale):
        """Electrons, noise, gain, gamma LUT and quantisation: 12-bit DN as the recorder stores them."""
        o = self.scene.optics
        linear = irradiance * background_scale + glint * 4095.0
        electrons = np.maximum(linear, 0) / o.gain_dn_per_e
        electrons = self.rng.poisson(electrons).astype(np.float32) + self.rng.normal(0, o.read_noise_e, electrons.shape)
        linear = np.clip(electrons * o.gain_dn_per_e, 0, 4095.0)
        dn = 4095.0 * (linear / 4095.0) ** o.gamma
        dn[1::2] += o.row_offset_dn
        return np.clip(np.round(dn), 0, o.ceiling_dn).astype(np.uint16)


def _splat(points, shape, layer=None, layers=1, weights=None):
    """Bilinear splat of unit weights at fractional (x, y) positions onto `layers` grids at once (layer gives
    each point's grid). Returns (layers, h, w) float32, or (h, w) when layers is 1."""
    h, w = shape
    x, y = points[:, 0], points[:, 1]
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx, fy = (x - x0).astype(np.float32), (y - y0).astype(np.float32)
    base = (np.zeros(len(x), np.int64) if layer is None else layer.astype(np.int64)) * (h * w)
    index = np.concatenate([base + y0 * w + x0, base + y0 * w + x0 + 1, base + (y0 + 1) * w + x0,
                            base + (y0 + 1) * w + x0 + 1])
    unit = np.ones(len(x), np.float32) if weights is None else np.asarray(weights, np.float32)
    weight = np.concatenate([(1 - fx) * (1 - fy) * unit, fx * (1 - fy) * unit, (1 - fx) * fy * unit, fx * fy * unit])
    ok = np.concatenate([(x0 >= 0) & (x0 < w) & (y0 >= 0) & (y0 < h), (x0 + 1 < w) & (x0 >= -1) & (y0 >= 0) & (y0 < h),
                         (x0 >= 0) & (x0 < w) & (y0 + 1 < h) & (y0 >= -1), (x0 + 1 < w) & (x0 >= -1) & (y0 + 1 < h) & (y0 >= -1)])
    grid = np.bincount(index[ok], weights=weight[ok], minlength=layers * h * w).astype(np.float32)
    grid = grid.reshape(layers, h, w)
    return grid[0] if layers == 1 else grid


def background_scale(scene: Scene, typical_transmission=0.88):
    """The linear DN of unobstructed, average-lit sclera, set so the background matches the measured level."""
    o = scene.optics
    linear = 4095.0 * (o.background_dn / 4095.0) ** (1.0 / o.gamma)
    return linear / typical_transmission


def measured_texture(field, n_frames, noise_sd_dn, level_dn, rng, step_px=3.0):
    """The band SDs synth.targets would report for a texture field (log units) seen through n_frames noisy,
    randomly shifted frames that are registered back with linear interpolation and averaged, as the estimator
    does on a burst."""
    from .targets import texture_stats
    h, w = field.shape
    image = level_dn * np.exp(field)
    total = np.zeros_like(image)
    for _ in range(n_frames):
        dx, dy = rng.normal(0, step_px, 2)
        shifted = cv2.warpAffine(image, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h), borderMode=cv2.BORDER_REFLECT)
        noisy = shifted + rng.normal(0, noise_sd_dn, image.shape).astype(np.float32)
        total += cv2.warpAffine(noisy, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h),
                                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
    mean = total / n_frames
    valid = np.zeros((h, w), bool)
    valid[24:-24, 24:-24] = True
    stats = texture_stats(mean, np.zeros((h, w), bool), valid)
    return np.array([stats[f"texture band {s:g} px (SD x1000)"] for s in TEXTURE_SIGMAS])


def calibrate_texture(scene: Scene, target, n_frames=90, noise_sd_dn=38.0, level_dn=2650.0, seed=0):
    """Amplitudes for texture_field whose measured band SDs (measured_texture) best match target (x1e-3).

    Band energies of independent components add, so measured^2 = noise^2 + M amplitudes^2, with M measured
    component by component; the squared amplitudes are the non-negative least-squares solution."""
    from scipy.optimize import nnls
    target = np.asarray(target, float)
    shape = (scene.height_px, scene.width_px)
    n_components = len(TEXTURE_COMPONENT_SIGMAS) - 1
    rng = np.random.default_rng(seed)
    floor = measured_texture(np.zeros(shape, np.float32), n_frames, noise_sd_dn, level_dn, rng) ** 2
    columns = []
    for k in range(n_components):
        unit = np.zeros(n_components)
        unit[k] = 1.0
        field = texture_field(shape, scene, np.random.default_rng(seed + k + 1), unit)
        columns.append(measured_texture(field, n_frames, 0.0, level_dn, np.random.default_rng(seed)) ** 2)
    matrix = np.stack(columns, 1)
    squared, _ = nnls(matrix, np.clip(target ** 2 - floor, 0, None))
    amplitudes = np.sqrt(squared)
    field = texture_field(shape, scene, np.random.default_rng(seed), amplitudes)
    return amplitudes, measured_texture(field, n_frames, noise_sd_dn, level_dn, np.random.default_rng(seed))
