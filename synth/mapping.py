"""Building a patient's vascular map from live slit-lamp imaging, with no stored recordings: a proof of concept.

    from synth.gaze import patient, SlitLamp
    from synth.mapping import plan, Mapper
    eye = patient(7)
    lamp = SlitLamp(eye, aim=(40.0, 0.0))
    targets = plan(lamp, clocks=(60, 120), s_mm=(1.0, 6.0))          # fixation targets that tile a region
    mapper = Mapper(eye.globe, region=((60, 120), (0.5, 6.5)))
    previous = None
    for gaze in targets:
        burst = lamp.record(gaze, n_frames=16, previous=previous)      # the camera sees what the eye shows it now
        mapper.add(burst.result.frames, gaze, lamp)                    # the map grows from the frames alone
        previous = gaze
    report = mapper.evaluate(eye, bursts=...)                          # coverage and accuracy against the truth

The mapper never sees the truth: it is given each burst's frames as they come, the gaze that was asked of the
patient and the camera's fixed geometry. For each burst it

1. drops the frames a lid or a blink covers (they are brighter than sclera and lack its vessels), and registers the
   rest to one another by phase correlation (the eye's fixational movements, the head on the chin rest);
2. averages them into an optical-density image, and measures the vessels in it (a multi-scale Hessian
   vesselness) and the red cells' flicker (the variance of each pixel about the registered mean, high where blood
   flows);
3. predicts where the field lies on the globe from the commanded gaze and the camera's aim (Listing's law), then
   refines that by registering the burst against the map built so far, rendered at the predicted place (the
   patient's actual gaze differs from the commanded one by the fixation error);
4. adds the burst into the map: a chart of the globe's surface (polar angle and clock angle about the corneal
   axis), weighted towards each field's centre.

The map's products are the mean optical density, the vessel map and the perfusion map. `evaluate` compares them
with the eye's network: the share of the region covered, the error of every burst's placement, and the precision
and recall of the vessel map (with a tolerance of a few micrometres).
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from scipy.optimize import least_squares
from skimage.registration import phase_cross_correlation

from . import media
from .gaze import direction_of, listing


# ----------------------------------------------------------------------------------------------- planning

def gaze_for(lamp, s_um, clock_deg):
    """The gaze (h, v) degrees that brings a surface point (s, clock) of the eye under the camera's axis."""
    g = lamp.eye.globe
    p = g.point(s_um, clock_deg, 0.0)
    p = p / np.linalg.norm(p)
    a = direction_of(*lamp.aim)

    def residual(x):
        return listing(*x) @ p - a

    best = None
    for start in ((0.0, 0.0), (-20.0, 0.0), (20.0, 0.0), (0.0, 20.0), (0.0, -20.0)):
        fit = least_squares(residual, start, bounds=([-45, -40], [45, 40]))
        if best is None or fit.cost < best.cost:
            best = fit
    return tuple(best.x), float(np.sqrt(2 * best.cost))


def plan(lamp, clocks, s_mm, overlap=0.2, max_gaze_deg=35.0):
    """Fixation targets whose fields tile the region of clock angles clocks = (from, to) and distances from the
    limbus s_mm = (from, to) with the given overlap, in a serpentine order; targets beyond max_gaze_deg or that the
    camera cannot reach are left out. Returns [(h, v)]."""
    g = lamp.eye.globe
    sc = lamp.scene
    fw = sc.width_px * sc.anatomy.um_per_px * (1 - overlap)
    fh = sc.height_px * sc.anatomy.um_per_px * (1 - overlap)
    targets = []
    rows = np.arange(s_mm[0] * 1000.0, s_mm[1] * 1000.0 + 1, fh)
    for r, s in enumerate(rows):
        radius = g.R * math.sin(float(g.theta(s)))
        span = math.radians((clocks[1] - clocks[0]) % 360.0) * radius
        n = max(1, int(math.ceil(span / fw)))
        cs = clocks[0] + (np.arange(n) + 0.5) / n * ((clocks[1] - clocks[0]) % 360.0)
        if r % 2:
            cs = cs[::-1]
        for c in cs:
            gaze, miss = gaze_for(lamp, s, c)
            if miss < 1e-3 and max(abs(gaze[0]), abs(gaze[1])) <= max_gaze_deg:
                targets.append(gaze)
    return targets


# ----------------------------------------------------------------------------------------------- the map

def vesselness(od, scales_px=(1.5, 3.0, 6.0)):
    """A multi-scale Hessian vesselness of a dark-on-bright vessel image given in optical density (vessels bright):
    the largest negative principal curvature, scale-normalised, over the scales."""
    out = np.zeros_like(od, np.float32)
    for s in scales_px:
        f = cv2.GaussianBlur(od.astype(np.float32), (0, 0), s)
        dxx = cv2.Sobel(f, cv2.CV_32F, 2, 0, ksize=3)
        dyy = cv2.Sobel(f, cv2.CV_32F, 0, 2, ksize=3)
        dxy = cv2.Sobel(f, cv2.CV_32F, 1, 1, ksize=3)
        tmp = np.sqrt((dxx - dyy) ** 2 + 4 * dxy ** 2)
        low = 0.5 * (dxx + dyy - tmp)                       # the more negative eigenvalue: across a bright ridge
        out = np.maximum(out, np.clip(-low, 0, None) * s ** 2)
    return out


def _illumination(image, grid=(32, 24), rounds=4):
    """The smooth illumination across a field: a quadratic surface fitted to the image's coarse brightness, again and
    again without the points far from it (vessels, lids), normalised to the sclera's level."""
    small = cv2.resize(image.astype(np.float32), grid, interpolation=cv2.INTER_AREA)
    gy, gx = np.mgrid[0:grid[1], 0:grid[0]].astype(np.float64)
    gx, gy = gx / grid[0] - 0.5, gy / grid[1] - 0.5
    A = np.stack([np.ones_like(gx), gx, gy, gx * gx, gx * gy, gy * gy], -1).reshape(-1, 6)
    b = small.ravel().astype(np.float64)
    keep = np.isfinite(b)
    for _ in range(rounds):
        coef, *_ = np.linalg.lstsq(A[keep], b[keep], rcond=None)
        resid = b - A @ coef
        mad = np.median(np.abs(resid[keep])) + 1e-9
        keep = np.isfinite(b) & (np.abs(resid) < 2.5 * 1.4826 * mad)
    h, w = image.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    xx, yy = xx / w - 0.5, yy / h - 0.5
    surface = coef[0] + coef[1] * xx + coef[2] * yy + coef[3] * xx * xx + coef[4] * xx * yy + coef[5] * yy * yy
    return np.maximum(surface, 1e-6).astype(np.float32)


class Mapper:
    """The map of one eye's surface over a region ((clock from, clock to), (s from, s to) mm), at um_per_px."""

    def __init__(self, globe, region, um_per_px=4.0):
        self.g = globe
        (c0, c1), (s0, s1) = region
        self.c0, self.span = c0, (c1 - c0) % 360.0
        self.s0, self.s1 = s0 * 1000.0, s1 * 1000.0
        self.um = um_per_px
        theta_max = float(globe.theta(self.s1))
        self.r_ref = globe.R * max(math.sin(theta_max), math.sin(float(globe.theta(self.s0))))
        self.H = int(math.ceil((self.s1 - self.s0) / um_per_px))
        self.W = int(math.ceil(math.radians(self.span) * self.r_ref / um_per_px))
        shape = (self.H, self.W)
        self.od = np.zeros(shape, np.float32)
        self.vessel = np.zeros(shape, np.float32)
        self.flicker = np.zeros(shape, np.float32)
        self.weight = np.zeros(shape, np.float32)
        self.placements = []                      # per burst: dict(predicted, refined (s_um, clock, roll), shift_um)
        rows, cols = np.mgrid[0:self.H, 0:self.W].astype(np.float64)
        self._s = self.s0 + (rows + 0.5) * um_per_px
        self._clock = self.c0 + np.degrees((cols + 0.5) * um_per_px / self.r_ref)
        self._xyz = None

    # chart <-> globe
    def chart_xyz(self):
        if self._xyz is None:
            self._xyz = self.g.point(self._s, self._clock, 0.0)
        return self._xyz

    def to_chart(self, s_um, clock):
        return ((clock - self.c0) % 360.0) * math.pi / 180 * self.r_ref / self.um - 0.5, (s_um - self.s0) / self.um - 0.5

    def _field_lookup(self, pose_s, pose_clock, roll_deg, um_cam, w, h):
        """For every chart pixel, its pixel in a camera field at that pose (float maps), and which fall inside."""
        frame = self.g.frame(pose_s, pose_clock)
        local = self.g.to_local(self.chart_xyz(), frame)
        r = math.radians(roll_deg)
        X = math.cos(r) * local[..., 0] + math.sin(r) * local[..., 1]
        Y = -math.sin(r) * local[..., 0] + math.cos(r) * local[..., 1]
        px = X / um_cam + w / 2                   # the renderer's convention: the field's centre at pixel w / 2
        py = Y / um_cam + h / 2
        inside = (px >= 0) & (px <= w - 1) & (py >= 0) & (py <= h - 1) & (local[..., 2] > -1e3)
        return px.astype(np.float32), py.astype(np.float32), inside

    def _map_in_field(self, layer, pose_s, pose_clock, roll_deg, um_cam, w, h):
        """The map layer rendered into a camera field at a pose (NaN where the map holds nothing)."""
        frame = self.g.frame(pose_s, pose_clock)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        X, Y = (xx - w / 2) * um_cam, (yy - h / 2) * um_cam
        r = math.radians(roll_deg)
        lx, ly = math.cos(r) * X - math.sin(r) * Y, math.sin(r) * X + math.cos(r) * Y
        xyz = self.g.from_local(np.stack([lx, ly, np.zeros_like(lx)], -1), frame)
        s, clock, _ = self.g.surface_coordinates(xyz)
        cx, cy = self.to_chart(s, clock)
        filled = np.where(self.weight > 0, layer / np.maximum(self.weight, 1e-9), np.nan).astype(np.float32)
        return cv2.remap(filled, cx.astype(np.float32), cy.astype(np.float32), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)

    # one burst
    @staticmethod
    def stabilise(frames):
        """Drop covered frames and register the rest by phase correlation. Returns (od stack registered to the first
        kept frame, kept indices, shifts px, valid mask of the field)."""
        f = frames.astype(np.float32)
        level = np.array([np.median(x) for x in f])
        background = np.median(level)
        texture = np.array([np.std(cv2.GaussianBlur(x, (0, 0), 2) - cv2.GaussianBlur(x, (0, 0), 8)) for x in f])
        # a lid or blink brightens the frame and wipes out the vessels' fine structure
        kept = np.flatnonzero((level < 1.06 * background) & (texture > 0.6 * np.median(texture)))
        if len(kept) == 0:
            return None
        od = media.optical_density(f[kept])
        ref = od[0]
        hp = lambda x: (cv2.GaussianBlur(x, (0, 0), 1.0) - cv2.GaussianBlur(x, (0, 0), 12)).astype(np.float32)
        window = cv2.createHanningWindow(ref.shape[::-1], cv2.CV_32F)
        h, w = ref.shape
        out, shifts = [], []
        valid = np.ones((h, w), np.float32)
        for x in od:
            (sx, sy), _ = cv2.phaseCorrelate(hp(ref), hp(x), window)
            M = np.float32([[1, 0, -sx], [0, 1, -sy]])
            out.append(cv2.warpAffine(x, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=np.nan))
            shifts.append((sx, sy))
        stack = np.stack(out)
        valid = np.isfinite(stack).mean(0) >= 0.5          # seen in at least half the kept frames
        # pixels a lid covers in the kept frames: brighter (skin) or darker (its margin) than the sclera, once the
        # illumination's smooth fall-off across the field is divided out (a robust quadratic fit)
        mean_raw = np.nanmean(np.stack([cv2.warpAffine(f[k], np.float32([[1, 0, -s[0]], [0, 1, -s[1]]]), (w, h))
                                        for k, s in zip(kept, shifts)]), 0)
        smooth = cv2.GaussianBlur(mean_raw, (0, 0), 6) / _illumination(mean_raw)
        lid = (smooth > 1.09) | (smooth < 0.6)
        lid = cv2.dilate(lid.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        valid &= ~lid
        return stack, kept, np.array(shifts), valid

    def add(self, frames, gaze, lamp, refine=True):
        """Add one burst, given only its frames, the commanded gaze and the camera. Returns its placement."""
        sc = lamp.scene
        um_cam, h, w = sc.anatomy.um_per_px, sc.height_px, sc.width_px
        got = self.stabilise(frames)
        if got is None:
            self.placements.append(None)
            return None
        stack, kept, shifts, valid = got
        mean = np.nanmean(stack, 0)
        flick = np.nanvar(stack - mean[None], 0) if len(stack) > 2 else np.zeros_like(mean)
        mean = np.nan_to_num(mean)
        flick = np.nan_to_num(flick)
        ves = vesselness(mean)
        # where the field should be: the commanded gaze through the fixed camera (the first kept frame)
        pose, roll = lamp.field_pose(gaze)
        s_p, c_p = pose.s_mm * 1000.0, pose.clock_deg
        shift_um = np.zeros(2)
        diagnostics = dict(overlap=0.0, response=0.0, valid=float(valid.mean()))
        if refine and (self.weight > 0).any():
            prior = self._map_in_field(self.od, s_p, c_p, roll, um_cam, w, h)
            overlap = np.isfinite(prior) & valid
            diagnostics["overlap"] = float(overlap.mean())
            if overlap.mean() > 0.08:
                band = lambda x: (cv2.GaussianBlur(x, (0, 0), 1.5) - cv2.GaussianBlur(x, (0, 0), 12)).astype(np.float32)
                a = band(np.where(np.isfinite(prior), prior, np.nanmean(prior)).astype(np.float32))
                b = band(mean.astype(np.float32))
                # masked normalised cross-correlation over the shifts (Padfield's masked FFT registration): the
                # shift that brings the burst onto the map where both hold data
                shift = phase_cross_correlation(a, b, reference_mask=np.isfinite(prior), moving_mask=valid,
                                                overlap_ratio=0.2)
                sy, sx = -float(shift[0][0]), -float(shift[0][1])
                M = np.float32([[1, 0, -sx], [0, 1, -sy]])
                moved = cv2.warpAffine(b, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=np.nan)
                both = np.isfinite(prior) & np.isfinite(moved) & (cv2.warpAffine(valid.astype(np.uint8), M, (w, h)) > 0)
                if both.sum() > 500:
                    x, y = a[both], moved[both]
                    diagnostics["response"] = float(np.corrcoef(x, y)[0, 1])
                if diagnostics["response"] > 0.3 and math.hypot(sx, sy) < 0.25 * w:
                    # the burst's content sits (sx, sy) px from where the map has it: move the field the other way
                    r = math.radians(roll)
                    dxl, dyl = -sx * um_cam, -sy * um_cam
                    shift_um = np.array([math.cos(r) * dxl - math.sin(r) * dyl, math.sin(r) * dxl + math.cos(r) * dyl])
                    frame = self.g.frame(s_p, c_p)
                    moved_c = self.g.from_local(np.array([[shift_um[0], shift_um[1], 0.0]]), frame)
                    s_n, c_n, _ = self.g.surface_coordinates(moved_c)
                    s_p, c_p = float(s_n[0]), float(c_n[0])
        px, py, inside = self._field_lookup(s_p, c_p, roll, um_cam, w, h)
        if not inside.any():
            self.placements.append(dict(pose=(s_p, c_p, roll), shift_um=shift_um, kept=len(kept), first=int(kept[0]), **diagnostics))
            return self.placements[-1]
        # weight: towards the field's centre, zero where the burst is invalid
        yy, xx = np.mgrid[0:h, 0:w]
        centre = (np.minimum.reduce([xx, w - 1 - xx, yy, h - 1 - yy]) / (0.15 * min(w, h))).clip(0, 1)
        weight_img = (centre * valid).astype(np.float32) * len(kept)
        sample = lambda img: cv2.remap(img.astype(np.float32), px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=0)
        wgt = sample(weight_img) * inside
        self.od += wgt * sample(mean)
        self.vessel += wgt * sample(ves)
        self.flicker += wgt * sample(flick)
        self.weight += wgt
        self.placements.append(dict(pose=(s_p, c_p, roll), shift_um=shift_um, kept=len(kept), first=int(kept[0]), **diagnostics))
        return self.placements[-1]

    # products
    def layer(self, name):
        data = getattr(self, name)
        return np.where(self.weight > 0, data / np.maximum(self.weight, 1e-9), np.nan)

    def vessel_map(self, threshold=None):
        """Binary vessel map from the accumulated vesselness (Otsu's threshold over the covered area by default)."""
        v = np.nan_to_num(self.layer("vessel"))
        covered = self.weight > 0
        if threshold is None:
            vals = v[covered]
            if vals.size == 0:
                return np.zeros_like(covered)
            scaled = np.clip(vals / max(np.percentile(vals, 99.5), 1e-9) * 255, 0, 255).astype(np.uint8)
            t, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            threshold = t / 255 * max(np.percentile(vals, 99.5), 1e-9)
        return (v > threshold) & covered

    # truth
    def truth_vessels(self, eye, min_diameter_um=0.0, max_depth_um=1e9):
        """The eye's vessels drawn into the chart at their local calibre (a mask), for the evaluation."""
        mask = np.zeros((self.H, self.W), np.uint8)
        for e in eye.network.edges:
            if e.diameter < min_diameter_um or float(np.mean(eye.globe.R - np.linalg.norm(e.points, axis=1))) > max_depth_um:
                continue
            s, clock, _ = self.g.surface_coordinates(e.points)
            x, y = self.to_chart(s, clock)
            if x.max() < -5 or x.min() > self.W + 5 or y.max() < -5 or y.min() > self.H + 5:
                continue
            if np.abs(np.diff(x)).max(initial=0) > self.W / 2:
                continue
            width = max(1, int(round(float(np.median(e.local_diameter)) / self.um)))
            pts = np.round(np.c_[x, y] * 4).astype(np.int32)
            cv2.polylines(mask, [pts], False, 1, width, cv2.LINE_8, shift=2)
        return mask > 0

    def evaluate(self, eye, true_poses=None, exposed=None, tolerance_um=8.0):
        """Coverage, placement error and vessel precision / recall against the eye's network.

        A map built without an absolute reference sits on the globe as a whole within the first burst's placement
        error; as is usual for mosaics, it is scored after one global shift that best aligns it with the truth
        (reported as global_offset_um), and the placement errors are given both as they are and relative to that
        common offset."""
        covered = self.weight > 0
        region = np.ones_like(covered) if exposed is None else exposed
        out = dict(coverage=float((covered & region).sum() / max(region.sum(), 1)))
        truth = self.truth_vessels(eye, max_depth_um=500.0)
        found = self.vessel_map()
        # the global shift of the map against the truth
        a = cv2.GaussianBlur((truth & covered).astype(np.float32), (0, 0), 2)
        b = cv2.GaussianBlur(found.astype(np.float32), (0, 0), 2)
        (sx, sy), _ = cv2.phaseCorrelate(a, b)
        out["global_offset_um"] = [float(sx * self.um), float(sy * self.um)]
        M = np.float32([[1, 0, -sx], [0, 1, -sy]])
        found = cv2.warpAffine(found.astype(np.uint8), M, found.shape[::-1], flags=cv2.INTER_NEAREST) > 0
        cov = cv2.warpAffine(covered.astype(np.uint8), M, covered.shape[::-1], flags=cv2.INTER_NEAREST) > 0
        if true_poses is not None:
            errs, rel = [], []
            for placed, t in zip(self.placements, true_poses):
                if placed is None or t is None:
                    continue
                px, py = self.to_chart(placed["pose"][0], placed["pose"][1])
                tx, ty = self.to_chart(t[0], t[1])
                errs.append(np.array([px - tx, py - ty]) * self.um)
            if errs:
                errs = np.array(errs)
                out["placement_error_um_median"] = float(np.median(np.linalg.norm(errs, axis=1)))
                rel = np.linalg.norm(errs - np.median(errs, 0), axis=1)
                out["relative_placement_error_um_median"] = float(np.median(rel))
                out["relative_placement_error_um_p90"] = float(np.percentile(rel, 90))
        truth = truth & cov
        k = max(1, int(round(tolerance_um / self.um)))
        kernel = np.ones((2 * k + 1, 2 * k + 1), np.uint8)
        truth_d = cv2.dilate(truth.astype(np.uint8), kernel) > 0
        found_d = cv2.dilate(found.astype(np.uint8), kernel) > 0
        found = found & cov
        out["precision"] = float((found & truth_d).sum() / max(found.sum(), 1))
        out["recall"] = float((truth & found_d).sum() / max(truth.sum(), 1))
        large = self.truth_vessels(eye, min_diameter_um=15.0, max_depth_um=500.0) & cov
        out["recall_15um_and_wider"] = float((large & found_d).sum() / max(large.sum(), 1))
        return out
