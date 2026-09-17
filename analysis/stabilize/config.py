"""Every tunable parameter, each citing the METHODS.md section that justifies
it. Change the reasoning there first, then the number here, so the two never
disagree. The full set is saved into every metrics.json, so a result can
always be traced to the settings that produced it.
"""
import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Params:
    # §2 Working scale — vessels span many pixels, so half resolution keeps
    # the structure while cutting computation 4x. Short strips are processed
    # at full resolution, or the large filter scales would exceed the image.
    working_scale: float = 0.5
    min_height_to_downscale: int = 400

    # §3 Frame quality gate — robust z-scores; flag, never delete.
    sharpness_blur_sigma: float = 1.0
    sharpness_min_z: float = -3.5
    brightness_max_abs_z: float = 3.5
    # physical floors: a statistically unusual frame is only rejected if it is
    # also meaningfully different, or a very steady burst (tiny MAD) would
    # reject frames over trivial wobbles
    sharpness_min_ratio: float = 0.8          # vs the burst median
    brightness_max_rel_change: float = 0.10   # vs the burst median
    min_frames: int = 20

    # §4 Glare — clipped pixels carry no information and are fixed to the
    # camera, not the tissue.
    glare_fraction_of_full_scale: float = 0.98
    glare_dilate_px: int = 30                 # full-resolution pixels

    # §5 Vesselness — vessels here are wide (response still rising at 16 px).
    sigmas_px: tuple = (4.0, 8.0, 16.0, 32.0)  # full-resolution pixels
    frangi_beta: float = 0.5
    contrast_percentile: float = 99.0

    # §6 Vessel envelope mask
    envelope_percentile: float = 90.0
    threshold_sample_frames: int = 24

    # §8 Groupwise registration
    # clean bursts stop early on converge_rms_px; hard bursts were still
    # improving at 4 (registered frames 430 -> 654 and climbing)
    max_iterations: int = 8
    converge_rms_px: float = 0.1              # full-resolution pixels
    coarse_min_response: float = 0.05
    fine_min_response: float = 0.10
    fine_max_step_px: float = 6.0             # full-resolution pixels

    # §9 Stability metrics
    coverage_min_fraction: float = 0.5
    saccade_mad_k: float = 6.0
    diagnostic_frames: int = 40

    def to_dict(self):
        return asdict(self)

    def params_hash(self):
        """Short fingerprint of every parameter. Recorded in each result and
        used to decide whether a result is still current: a changed setting
        must invalidate old results, which file times alone can't detect."""
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class NonrigidParams:
    """§13 Non-rigid refinement (experimental). Kept apart from Params so that
    tuning it doesn't invalidate translation results."""
    # patches: large enough to hold several vessels (phase correlation needs
    # structure in two directions), small enough to follow rotation and
    # magnification across the field. Full-resolution pixels.
    patch_px: int = 320
    stride_px: int = 160
    patch_min_response: float = 0.05
    min_patches: int = 6
    # local deviation from the frame's affine: bounded, so a wrong patch
    # match can't tear the image. Full-resolution pixels.
    max_local_px: float = 8.0
    # single-pose reference: the most self-consistent run of this many frames
    reference_frames: int = 25
    reference_iterations: int = 2
    max_iterations: int = 6
    # converge on the 90th percentile of per-frame updates, not the median:
    # the frames that cause doubling are the minority still moving
    converge_p90_px: float = 0.25
    # frames that still disagree with the template after refinement
    reject_z: float = -3.5
    reject_ncc_drop: float = 0.05
    # tile grid for the residual-spread diagnostic (rows, cols)
    tile_grid: tuple = (3, 4)

    def to_dict(self):
        return asdict(self)

    def params_hash(self, base):
        """Fingerprint of the translation stage's parameters and these."""
        blob = json.dumps({"base": base.to_dict(), "nonrigid": self.to_dict()},
                          sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
