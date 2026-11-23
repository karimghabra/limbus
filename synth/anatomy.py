"""The anatomical, physiological and optical parameters of the synthetic conjunctiva, each with its source.

Every number the generator, the flow model, the red-cell model or the renderer uses is a field here, and the comment
beside it names where it comes from: a reference key of docs/synthetic_conjunctiva_anatomy.md (written [Key]), a
measurement of the real bursts ("measured", synth/targets.py), or "fitted" / "assumed" (docs §9 and §10 list
every such case and what it was fitted to). The constants left in the code are geometric details of how the
generator draws (curve smoothness, depth undulation, lattice jitter, slab boundaries); each is commented where it is
used and docs §9 lists them. The cited ranges the generated networks are held to are in synth/check.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, replace


@dataclass
class Anatomy:
    """Structure: layers, vessel calibres, branching, density, tortuosity, perforators. Lengths in µm.

    The comment after each value names its source; docs/synthetic_conjunctiva_anatomy.md gives the numbers."""

    # imaging scale: measured, the millimetre ruler imaged through the camera (docs §1.3)
    um_per_px: float = 3.26
    # the field: x along the limbus, y away from it (radial), z depth below the epithelial surface
    limbus_distance_mm: float = 3.0
    # layer depths below the epithelial surface, µm (docs §2)
    epithelium: float = 45.0                         # 42 ± 7 µm [Zhang2013]; 45-52 µm [Zhang2011, FengSimpson2008]
    conjunctival_plexus: tuple = (50.0, 230.0)       # in the 198 µm stroma under the epithelium [Zhang2013, Akagi2018]
    episcleral_plexus: tuple = (260.0, 440.0)        # beneath conjunctiva and Tenon's [FernandezVigo2021, Kumar2013, Kemenes2025]
    deep_plexus: tuple = (520.0, 900.0)              # intrascleral, 200-1000 µm slab, sclera 0.5 mm [Akagi2018, Olsen1998]
    # calibres, µm (docs §3)
    capillary_diameter: tuple = (4.5, 9.0)           # capillaries < 9 µm [Moka2020, Koutsiaris2015]
    terminal_arteriole_diameter: tuple = (9.0, 12.0)     # precapillary arterioles 6-12 µm [Koutsiaris2010]
    postcapillary_venule_diameter: tuple = (10.0, 16.0)  # post-capillary venules 9-14 and 14-24 µm classes [Moka2020, Koutsiaris2015]
    episcleral_artery_diameter: tuple = (40.0, 80.0)     # episcleral vessels 79 ± 9 µm [MacKenzie2016]
    episcleral_vein_diameter: tuple = (50.0, 100.0)      # episcleral veins 50-100 µm, 61 ± 5 µm [Asanad2021, Kim2024]
    episcleral_vein_mean_sd: tuple = (61.0, 8.0)         # drawn about the measured mean [Kim2024], within the range above
    deep_vessel_diameter: tuple = (20.0, 50.0)           # deep scleral vessels 20-50 µm at ~500 µm [Bizheva2011]
    communicating_diameter: tuple = (10.0, 20.0)         # radial channels 10-20 µm wide [Bizheva2011]
    perforating_artery_diameter: tuple = (20.0, 50.0)    # as the deep vessels they join [Bizheva2011]
    max_arteriole_diameter: float = 58.0             # inside the largest conjunctival arteriole, 61 µm [Khansari2017]
    max_venule_diameter: float = 72.0                # inside the largest conjunctival venule, 75 µm [Khansari2017]
    # branching (docs §4): junction exponents, Murray's 3 [Murray1926a, MayrovitzRoy1983, KassabFung1995];
    # retinal arteries 2.59, veins 1.95 [Luo2017]
    murray_exponent_arterial: float = 2.6
    murray_exponent_venous: float = 2.7
    # density and mesh (docs §5)
    capillary_mesh_spacing: float = 285.0            # µm between mesh cell centres (fitted to ~8 mm/mm2 of capillary, docs §5)
    capillary_edge_drop: float = 0.18                # share of mesh ridges absent (open polygons)
    radial_bias: float = 0.15                        # conjunctival vessels run radially from the limbus [Akagi2018]
    arterial_trees_per_mm2: float = 0.4              # arterioles fewer than venules, about 1:2 to 1:4 [Cheung2001, Khansari2017]
    venous_trees_per_mm2: float = 0.4 
    av_pair_offset: tuple = (50.0, 110.0)            # lateral distance of an accompanying venule from its arteriole (about the measured pair spacing, 85 µm, docs §4)
    av_pair_fraction: float = 0.2                    # share of venular attractors placed along arterioles (assumed)
    anastomoses_per_mm2: float = 2.0                 # arteriolar and venular loops between trees (fitted, docs §5)
    episcleral_curvature: float = 0.0008             # rad/µm SD of their slow bend, correlation 250 µm: long straight runs
    episcleral_heading_spread: float = 0.8           # rad about the radial direction: parallel radial trunks [Akagi2018, Kemenes2025]
    episcleral_vessels_per_mm: float = 2.2           # episcleral vessels per mm along the limbus (fitted to the measured diameters, docs §5)
    episcleral_avas_per_mm2: float = 0.75            # arteriovenous anastomoses, 0.5-1 per mm2 [Selbach2005]
    ava_diameter: tuple = (5.0, 8.0)                 # their smooth-muscle arteriolar segment [RohenFunk1994], at rest tone (assumed)
    deep_vessels_per_mm: float = 0.6
    communicating_per_mm2: float = 1.2               # conjunctival-episcleral connections (fitted, docs §2.4)
    communicating_angle_deg: tuple = (20.0, 60.0)    # oblique, no measured angle exists (docs §2.4) [Asanad2021, Li2011]
    perforator_per_mm2: float = 0.25                 # deep-to-episcleral perforations in the field (docs §2.4)
    perforator_angle_deg: tuple = (30.0, 70.0)
    # tortuosity: lateral wiggle of the centreline, (amplitude µm, correlation length µm), per kind (fitted to the
    # measured tortuosity, docs §5)
    wiggle: dict = field(default_factory=lambda: {
        "arteriole": (36.0, 220.0), "venule": (40.0, 220.0), "capillary": (4.0, 110.0),
        "episcleral": (8.0, 220.0), "deep": (8.0, 260.0), "communicating": (2.0, 40.0),
    })
    # multi-scale irregularity on top of the meander (fitted to the measured curvature, jitter and width variation,
    # docs §5): fine jitter (amplitude µm, correlation µm), kinks (per mm, amplitude µm, width µm), and the
    # variation of the calibre along a vessel (coefficient of variation, correlation µm)
    bend_wiggle: tuple = (5.0, 70.0)                 # gentle bends of 100-300 µm on vessels above capillary size (fitted to the curvature spectrum)
    fine_jitter: tuple = (0.3, 6.0)
    large_jitter: tuple = (0.03, 0.5)                # extra jitter per µm of calibre: (amplitude, correlation) (fitted by eye, docs §5)
    kinks: tuple = (0.0, 2.0, 8.0)
    taper: tuple = (0.05, 0.25)                      # smooth change of calibre end to end along a whole vessel, relative
    # the calibre along a vessel varies smoothly over the whole vessel, continuous through every junction it passes:
    # (coefficient of variation, correlation µm) for small (<= 12 µm) and large (>= 30 µm) vessels, interpolated
    # between; CV about 0.1 above 15 µm and up to 0.4 below [Kiani1993], ~0.19 in 9-24 µm conjunctival vessels
    # [Shahidi2010]; correlation lengths fitted to the real bursts (docs §5)
    width_variation: tuple = (0.18, 100.0)
    width_variation_large: tuple = (0.08, 300.0)
    wall_irregularity: tuple = (0.03, 14.0)          # fine relative unevenness of the lumen (fitted by eye, docs §5)
    # companion venules beside arterioles (docs §4)
    pair_fraction: float = 1.0                       # share of the larger arterioles that have one (fitted to the measured paired share, docs §4)
    pair_min_diameter: float = 25.0                  # µm: only arterioles reaching this calibre are accompanied
    pair_gap_um: tuple = (20.0, 60.0)                # edge-to-edge gap, fitted to the measured pair spacing (85 µm centre to centre); the mesentery's within 15 µm [NelloreHarris2002] drew them touching
    pair_max_venule_diameter: float = 40.0           # a companion drains its arteriole's territory only (fitted: wider ones stagnate)
    pair_calibre_ratio: tuple = (1.3, 1.5)           # artery:vein calibre 0.653 ± 0.068 [Yusufu2025], ± 2 SD
    # growth (space colonisation [Runions2007]) step, reach and kill distance, µm, and attractor density
    growth_step: float = 18.0
    growth_influence: float = 400.0
    growth_kill: float = 210.0                       # sets the spacing of branches (fitted, docs §5)
    venous_kill_ratio: float = 1.0                   # venules are more numerous than arterioles [Khansari2017]
    attractors_per_mm2: float = 120.0
    growth_persistence: float = 0.8
    branch_spacing_um: float = 160.0                 # least distance between branch points of a tree (fitted to segment lengths, docs §5)
    max_terminal_branch_um: float = 450.0            # unbranched terminal branches are cut to this (fitted by eye, docs §5)
    path_straightening: float = 0.4                  # share of a grown path's deviation from its chord kept (fitted by eye and to the measured straightness)


@dataclass
class Physiology:
    """Blood and haemodynamics (docs §6)."""
    inflow_discharge_haematocrit: float = 0.40       # discharge haematocrit entering the field (docs §6.3)
    plasma_viscosity_mpas: float = 1.25              # plasma viscosity at 37 °C [Pries1992]
    rbc_volume_um3: float = 90.0                     # 90 µm^3 [Bosschaart2014]; 75 ± 25 fL in vivo [Ding2025]
    rbc_diameter_um: float = 7.8                     # 7.82 µm [Fedosov2010, from Evans & Fung]
    rbc_thickness_um: float = 2.2
    # boundary pressures (mmHg), log-linear in calibre between 10 and 100 µm (docs §6.2):
    arterial_pressure_at_100um: float = 50.0         # 45-65 mmHg in 40-50 µm arterioles [Zweifach1974]
    arterial_pressure_at_10um: float = 30.0          # 30-35 mmHg after the precapillary branches [Zweifach1974]
    venous_pressure_at_10um: float = 14.0            # just above capillary pressure, 13-16 mmHg [Williams1990, Hahn2004]
    venous_pressure_at_100um: float = 8.0            # episcleral venous pressure 6.3-9.6 mmHg [Zeimer1983, Sit2011, Arora2017]
    capillary_boundary_pressure: float = 15.0        # 13-16 mmHg [Williams1990, Hahn2004]; unused while the mesh is closed
    radial_pressure_gradient_mmhg_per_mm: float = 0.15   # along episcleral and deep vessels (fitted, docs §6.2)
    heart_rate_hz: float = 1.1                       # cardiac period 909 ± 86 ms [Neriyanuri2023]
    arterial_pulsatility: float = 0.12               # half the pulse pressure over the mean at the arterial inlets (fitted, docs §6.3)
    # the shape of the arterial pressure wave over one cycle: a systolic peak and a dicrotic wave, as (time, width,
    # height) of two Gaussian waves in fractions of the period (a generic arterial wave, docs §6.3)
    pulse_shape: tuple = ((0.17, 0.07, 1.0), (0.42, 0.10, 0.32))
    pulse_harmonics: int = 4
    # volume distensibility of the vessels, 1/mmHg (relative change of volume per mmHg), on the arterial side,
    # in the capillaries and on the venous side: it damps the pulse and delays it downstream (fitted to the
    # arteriolar pulsatility [Koutsiaris2010], the capillary pulse pressure [Zweifach1974] and the far smaller
    # venous pulsatility [Puyo2019]; veins several times more distensible than arteries, docs §6.3)
    distensibility_per_mmhg: tuple = (0.002, 0.03, 0.3)
    cell_free_layer_um: tuple = (0.8, 3.1)           # 0.8 µm at 10 µm to 3.1 µm at 50 µm [Kim2007]
    profile_bluntness: float = 4.0                   # v_centre / v_mean = (K + 2) / K = 1.5 [Tangelder1986]
    aggregation_cv: float = 0.42                     # inter-cell distance 20.5 ± 8.6 µm [Ding2025]
    clump_length_um: float = 25.0                    # axial correlation of red-cell structure 18-28 µm [Fedosov2010], ~30 µm [Kim2007]
    clump_amplitude: float = 0.25                     # relative SD of the axial cell density (fitted to the measured flicker, docs §6.4)


@dataclass
class Optics:
    """Light, tissue and camera (docs §7 and the measured targets)."""
    # spectral bands of the detected light: (weight, whole-blood absorption coefficient at Hct 0.45 in 1/mm, the
    # mean of oxy and deoxy at 450, 470, 500, 520, 545, 575, 600, 625 and 660 nm [Bosschaart2014]); the weights,
    # the source spectrum times the sensor's response, are fitted to the measured contrast by calibre (docs §7.1)
    spectrum: tuple = ((0.022, 32.6), (0.03, 12.7), (0.068, 10.3), (0.068, 13.6), (0.038, 25.4), (0.038, 23.8),
                       (0.22, 5.1), (0.258, 1.4), (0.258, 0.7))
    double_pass: float = 2.0
    scatter_veil_length_um: float = 650.0            # depth at which 1 - 1/e of the detected light has turned back
    scatter_blur_um_at_500: float = 16.0             # Gaussian sigma of the image of a point 500 µm deep
    # the camera's pose on the eye: the field's centre is where the camera axis meets the globe; the globe is a
    # sphere of the anterior scleral radius (docs §1.4), and the camera axis may tilt from the surface normal there
    eye_radius_um: float = 12500.0                   # anterior sclera 13.1 ± 0.8 mm [Choi2014], 11.2 ± 0.3 mm [Jesus2017]
    camera_tilt_deg: tuple = (0.0, 0.0)              # tilt of the camera axis from the normal, about the y and x axes
    psf_sigma_px: float = 0.9
    defocus_um_per_um: float = 0.06                 # blur sigma per µm of distance from the focal plane
    focus_depth_um: float = 250.0
    focus_wander_um: float = 25.0
    illumination_p5_p95: float = 0.64
    background_dn: float = 2650.0
    flicker_hz: float = 8.77                        # |120 Hz mains - 4 x 32.19 fps|; measured 8.78 Hz
    flicker_amplitude: float = 0.0111               # 0.89 % measured after the gamma 0.8 LUT, so 0.89 / 0.8 % in linear light
    flicker_white: float = 0.0012                   # fitted: the measured 0.21 % residual also holds motion
    texture_band_sd: tuple = (0.94, 0.91, 1.58, 2.70, 4.55)   # measured, x1e-3 ln units, at the TEXTURE_SIGMAS bands
    texture_amplitudes: tuple = (2.32, 0.0, 2.15, 3.21, 4.98, 12.89)   # generator components, fitted (camera.calibrate_texture)
    exposure_s: float = 0.012838
    fps: float = 32.1926
    gain_dn_per_e: float = 1.344
    read_noise_e: float = 6.8
    gamma: float = 0.8
    ceiling_dn: int = 4091
    row_offset_dn: float = 3.56
    motion_step_px: float = 4.1                     # measured 3.46 px; the estimator under-reads steps by ~15 % (docs §7.6)
    motion_bound_px: float = 60.0
    saccade_rate_hz: float = 0.46
    saccade_px: tuple = (40.0, 130.0)
    rotation_deg: float = 0.05
    glints: int = 4
    tube_um_per_px: float = 6.0                     # at this magnification or coarser, vessels are drawn as tubes of mean density


@dataclass
class Scene:
    anatomy: Anatomy = field(default_factory=Anatomy)
    physiology: Physiology = field(default_factory=Physiology)
    optics: Optics = field(default_factory=Optics)
    width_px: int = 768
    height_px: int = 512
    margin_um: float = 350.0
    seed: int = 0

    def with_(self, **changes):
        return replace(self, **changes)


def describe(obj):
    """(name, value) for every field of a parameter dataclass."""
    return [(f.name, getattr(obj, f.name)) for f in fields(obj)]
