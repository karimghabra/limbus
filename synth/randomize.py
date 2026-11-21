"""Random scenes for training data: every parameter drawn from a seed and a configuration.

    from synth.randomize import random_scene
    scene = random_scene(seed=17)                                  # the default ranges
    scene = random_scene(seed=17, config={"optics.psf_sigma_px": (0.6, 2.5)})   # one range overridden

Anatomy and physiology are drawn within their cited ranges (docs/synthetic_conjunctiva_anatomy.md), so every
random network still passes the anatomy check; the appearance - blur, focus, motion, noise, gain, texture,
illumination, flicker - is drawn from wider domain-randomisation ranges around the values measured on the real
bursts, so a network trained on the output does not learn one camera's settings. Most draws pass the anatomy check
(docs/synthetic_conjunctiva_validation.md gives the rate); random_scene(seed, anatomical=True) keeps only those
that do. A range is (low, high), drawn
uniformly; ("log", low, high) draws log-uniformly; a scalar fixes the value. Parameters not listed keep their
defaults.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .anatomy import Scene

DEFAULT_RANGES = {
    # anatomy, within the cited ranges
    "anatomy.murray_exponent_arterial": (2.59, 3.0),                  # [Luo2017, MayrovitzRoy1983]
    "anatomy.murray_exponent_venous": (2.5, 3.0),                     # within 1.95-3.14 [Luo2017, Murray1926a]
    "anatomy.capillary_mesh_spacing": (210.0, 260.0),                 # capillary length density 4.4-12 mm/mm2 [Kvernebo2022]
    "anatomy.episcleral_vessels_per_mm": (1.5, 4.0),
    "anatomy.episcleral_avas_per_mm2": (0.5, 1.0),                    # [Selbach2005]
    "anatomy.communicating_per_mm2": (0.6, 2.0),
    "anatomy.arterial_trees_per_mm2": (0.35, 0.7),
    "anatomy.venous_trees_per_mm2": (0.45, 0.8),
    "anatomy.radial_bias": (0.1, 0.4),
    # physiology, within the cited ranges
    "physiology.arterial_pressure_at_100um": (45.0, 55.0),            # within 45-65 mmHg [Zweifach1974]
    "physiology.arterial_pressure_at_10um": (30.0, 33.0),             # within 30-35 mmHg [Zweifach1974]
    "physiology.venous_pressure_at_100um": (7.0, 9.6),                # within 6.3-9.6 mmHg [Zeimer1983, Sit2011]
    "physiology.heart_rate_hz": (0.9, 1.4),
    "physiology.arterial_pulsatility": (0.15, 0.3),                   # pulsatility index about 0.8 [Koutsiaris2010]                           # [Neriyanuri2023], 54-84 bpm
    "physiology.inflow_discharge_haematocrit": (0.35, 0.45),          # [SecombPries2013]
    "physiology.clump_amplitude": (0.3, 0.9),
    # appearance: domain randomisation around the measured values
    "optics.psf_sigma_px": (0.6, 1.8),
    "optics.focus_depth_um": (80.0, 300.0),
    "optics.focus_wander_um": (5.0, 60.0),
    "optics.defocus_um_per_um": (0.02, 0.06),
    "optics.scatter_veil_length_um": (450.0, 1000.0),
    "optics.scatter_blur_um_at_500": (8.0, 22.0),
    "optics.illumination_p5_p95": (0.45, 0.9),
    "optics.background_dn": (1500.0, 3300.0),
    "optics.flicker_amplitude": (0.0, 0.02),
    "optics.flicker_white": (0.0, 0.006),
    "optics.motion_step_px": ("log", 1.0, 8.0),
    "optics.motion_bound_px": (20.0, 100.0),
    "optics.saccade_rate_hz": (0.0, 1.0),
    "optics.rotation_deg": (0.0, 0.3),
    "optics.read_noise_e": (4.0, 12.0),
    "optics.gain_dn_per_e": ("log", 0.5, 3.0),
    "optics.glints": (0, 10),
    "optics.texture_scale": ("log", 0.5, 2.0),
}


def _draw(rng, spec):
    if isinstance(spec, tuple) and len(spec) == 3 and spec[0] == "log":
        return float(np.exp(rng.uniform(np.log(spec[1]), np.log(spec[2]))))
    if isinstance(spec, tuple) and len(spec) == 2:
        low, high = spec
        if isinstance(low, int) and isinstance(high, int):
            return int(rng.integers(low, high + 1))
        return float(rng.uniform(low, high))
    return spec


def random_scene(seed, config=None, base: Scene | None = None, anatomical=False, tries=20) -> Scene:
    """A Scene with every parameter in the ranges drawn from seed (config overrides or adds ranges).

    With anatomical=True the network and its flow are generated and checked (synth.check), and the draw is repeated
    with derived seeds until every checked quantity is within its cited range; the returned scene's seed is the one
    that passed."""
    if anatomical:
        from . import check, haemodynamics, network
        for attempt in range(tries):
            scene = random_scene(seed * 1000 + attempt if attempt else seed, config, base)
            net = network.generate(scene)
            flow = haemodynamics.solve(net, scene.physiology)
            if all(row["status"] != "outside" for row in check.anatomy_report(scene, net, flow)):
                return scene
        raise RuntimeError(f"no anatomical scene in {tries} draws from seed {seed}: the ranges are too wide")
    ranges = dict(DEFAULT_RANGES)
    ranges.update(config or {})
    rng = np.random.default_rng([seed, 7])
    base = base or Scene()
    groups = {"anatomy": {}, "physiology": {}, "optics": {}, "scene": {}}
    texture_scale = 1.0
    for key in sorted(ranges):
        group, name = key.split(".", 1)
        value = _draw(rng, ranges[key])
        if key == "optics.texture_scale":
            texture_scale = value
            continue
        groups[group][name] = value
    optics = replace(base.optics, **groups["optics"])
    optics = replace(optics, texture_amplitudes=tuple(a * texture_scale for a in optics.texture_amplitudes))
    return replace(base, anatomy=replace(base.anatomy, **groups["anatomy"]),
                   physiology=replace(base.physiology, **groups["physiology"]), optics=optics, seed=int(seed),
                   **groups["scene"])
