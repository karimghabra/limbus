"""Every tunable parameter of the lucky stage, each citing the METHODS.md
section that justifies it. Change the reasoning there first, then the
number here. The full set is saved into every result's lucky.json."""
import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class LuckyParams:
    # §L2 bands: differences of Gaussians between these scales (full-res
    # px). 0 = the image itself, so the finest band holds everything down to
    # the pixel. The last scale is the lowpass, always a plain mean.
    band_sigmas_px: tuple = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0)
    # §L2 powers fused side by side. 0 is the ordinary mean (the baseline
    # every comparison is made against); `output` is the lucky image kept.
    # p = 1 is the matched filter, and the only power that never lost to the
    # mean on the ground-truth benchmark (§L5)
    powers: tuple = (0.0, 1.0, 2.0)
    output: float = 1.0
    # §L2 relative amplitude: floor added to the projection, and its window
    # r_k = max(min, factor * s_k+1)
    template_floor: float = 0.25
    energy_radius_factor: float = 4.0
    energy_min_radius_px: float = 12.0
    fill_sigma_px: float = 16.0         # normalised-convolution fill of holes
    min_coverage: float = 0.5           # as stabilize §9

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class AnnotateParams:
    # §L3 background: a closing removes dark structures up to ~2x radius wide
    background_radius_px: float = 40.0
    background_downscale: int = 4
    fill_sigma_px: float = 16.0
    # §L3 vesselness scales, full-resolution px: capillaries to venules
    sigmas_px: tuple = (1.5, 3.0, 6.0, 12.0, 24.0)
    # Lindeberg's ridge normalisation (sigma^1.5) rather than Frangi's
    # sigma^2: wide scales otherwise win on the halo around thin vessels
    scale_gamma: float = 1.5
    frangi_beta: float = 0.5
    contrast_percentile: float = 95.0
    # §L3 centrelines: 'ridge' (non-maximum suppression of vesselness),
    # 'steger' (line points at every scale) or 'hybrid' (ridges, with vessels
    # at least 2x wide_radius_px wide re-traced by line points at scales >=
    # wide_sigma_px, or their medial axis). Wide = dark core: contrast below
    # wide_depth_fraction x its wide_depth_percentile, surviving the opening
    centreline: str = "hybrid"
    wide_sigma_px: float = 12.0
    wide_radius_px: float = 10.0
    wide_centre: str = "steger"
    wide_depth_percentile: float = 0.5
    wide_depth_fraction: float = 0.6
    # §L3 hysteresis along the ridges, on vesselness (0..1); tuned for the
    # best centreline F1 of the PLAIN MEAN on the benchmark (§L5), so the
    # lucky image gets no home advantage
    low_threshold: float = 0.06
    high_threshold: float = 0.25
    max_hole_px: int = 40               # holes filled before thinning (no loops)
    spur_px: int = 12
    min_centreline_px: int = 25

    def to_dict(self):
        return asdict(self)


def params_hash(*ps):
    blob = json.dumps([p.to_dict() for p in ps], sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
