"""The whole anterior eye, built from where the blood comes from and where it goes, with levels of detail.

    from synth.eye import Eye, Pose, simulate_view
    eye = Eye(seed=4)                                   # the whole eye's network at every level, solved once
    result = simulate_view(eye, Pose(s_mm=3.0, clock_deg=90), n_frames=60)   # a burst of one region

Levels of detail (docs/synthetic_conjunctiva_anatomy.md §11), all parts of one network that is generated and solved
once per eye:

- Coarse (the whole globe): the anterior ciliary arteries leaving the four rectus
  tendons at their insertions (the spiral of Tillaux), two per muscle and one on the lateral rectus; the episcleral
  arterial circle they form near the limbus; the anterior conjunctival arteries it gives off towards the limbal
  arcades and back into the bulbar conjunctiva; the posterior conjunctival arteries entering from the fornices (the
  palpebral arcades), meeting the anterior ones at the watershed; companion conjunctival veins draining to the
  fornices; the limbal venous ring and the episcleral veins draining away from the limbus; episcleral arteriovenous
  anastomoses. Where no fine vessels are grown (outside Eye(fine_region=...)), the fine vessels each trunk feeds
  are represented by lumped beds (one resistance every 0.5 mm between a trunk and its companion vein), so the
  solve sees the load they put on the trunks. Pressures are set at the true inlets (the anterior ciliary arteries,
  the palpebral arcades) and outlets (the episcleral and palpebral veins at the episcleral venous pressure), and
  every flow and velocity follows.
- Fine (grown tile by tile of about 1 mm over the fine region, the whole conjunctiva by default): the conjunctival
  arteriolar and venular trees (synth.network.generate_fine), rooted on the coarse trunks that cross each tile,
  from a seed that depends only on the eye's seed and the tile's position; and one capillary plexus over the whole
  region that joins the trees' terminals across the tiles' borders (build_whole).
- The view (the camera's field only): the pieces of the eye's network that fall in it, in the camera's
  coordinates, each carrying the flow of the eye's edge it came from - so zooming from an overview to high
  magnification never changes a vessel's path or flow - and rendered with discrete red cells at high magnification,
  or as tubes carrying their mean red-cell density at low magnification (synth.camera).

Eye coordinates: µm, the eye's centre at the origin, +z out of the cornea along the optical axis, +y superior, +x
temporal (a right eye seen from in front: the clock angle runs superior 0°, temporal 90°, inferior 180°, nasal
270°). A point of the globe is given by its distance s from the limbus along the surface (positive towards the
fornix), its clock angle, and its depth below the surface.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.spatial import SphericalVoronoi, cKDTree

from .anatomy import Scene
from .haemodynamics import Flow, solve
from .network import (Edge, Network, _Builder, _noise_at, _prune_terminals, _segment, _smooth_noise, assign_calibres,
                      generate_fine, resample, wiggle)

MUSCLES = {"superior": 0.0, "temporal": 90.0, "inferior": 180.0, "nasal": 270.0}   # clock angle of each rectus


@dataclass
class EyeAnatomy:
    """The whole-eye geometry and supply; each value's source is in docs §11."""
    radius_um: float = 12500.0                       # anterior sclera 13.1 ± 0.8 [Choi2014], 11.2 ± 0.3 mm [Jesus2017]
    limbus_radius_um: float = 5900.0                 # white-to-white 11.80 ± 0.42 mm [GaneshLin2023]
    insertion_mm: dict = field(default_factory=lambda: {"nasal": 5.7, "inferior": 6.0, "temporal": 6.8,
                                                        "superior": 6.8})   # AS-OCT in adults [Pihlblad2016]
    insertion_width_mm: dict = field(default_factory=lambda: {"nasal": 8.5, "inferior": 6.4, "temporal": 8.5,
                                                              "superior": 10.3})   # [Mistry2026]
    # major anterior ciliary arteries per rectus, drawn per eye from the measured shares of 1, 2 and 3 bundles on
    # the horizontal recti [PanYang2020]; two on each vertical rectus, the textbook count [AshtonSmith1953]
    arteries_per_muscle: dict = field(default_factory=lambda: {
        "nasal": ((1, 2, 3), (0.068, 0.464, 0.468)), "temporal": ((1, 2, 3), (0.412, 0.399, 0.189)),
        "superior": ((2,), (1.0,)), "inferior": ((2,), (1.0,))})
    ciliary_artery_diameter: tuple = (70.0, 110.0)   # not measured; larger than the 40-80 µm episcleral arteries (assumed)
    circle_mm: float = 2.0                           # episcleral arcades ~2-3 mm behind the limbus [Kemenes2025]
    circle_gap_fraction: float = 0.15                # parts of the superficial circle missing [MeyerWatson1987] (share assumed)
    circle_diameter: tuple = (40.0, 60.0)            # as the episcleral arteries [MacKenzie2016]
    limbal_arcade_mm: float = 0.4                    # inside the palisade zone of 1.5-2.1 mm [Kemenes2025]
    limbal_vein_mm: float = 0.25
    anterior_artery_spacing_mm: float = 1.8          # anterior conjunctival arteries along the circle (fitted to density)
    posterior_artery_spacing_mm: float = 2.5         # posterior conjunctival arteries along the fornix (fitted)
    fornix_vein_spacing_mm: float = 1.5              # conjunctival veins leaving at the fornix (fitted)
    coarse_attractors_per_mm2: float = 20.0          # coarse growth (space colonisation [Runions2007]) settings: the
    coarse_growth_step_um: float = 40.0              # coarse level stops at trunk scale and the tiles add the rest
    coarse_growth_influence_um: float = 1500.0
    coarse_growth_kill_um: float = 450.0
    coarse_leaf_diameter: tuple = (20.0, 28.0)       # where the coarse level hands over to the tiles (µm)
    # the widest conjunctival trunks: arterioles to 61 µm and venules to 75 µm are reported [Khansari2017], but the
    # widest sharp conjunctival vessels of the recordings are about 40-50 µm, and wider trunks at conjunctival depth
    # rendered far darker than any real vessel (docs §11.4)
    coarse_artery_max_um: float = 40.0
    coarse_vein_max_um: float = 48.0
    coarse_radial_bias: float = 0.15                 # conjunctival vessels run radially [Akagi2018], as at the patch level
    coarse_growth_persistence: float = 0.6           # how much a growing coarse vessel keeps its heading (fitted to
                                                     # the orientation spread and junction density of the recordings)
    coarse_branch_spacing_um: float = 400.0          # least distance between branch points of a coarse tree (as at
    coarse_max_terminal_um: float = 1500.0           # the patch level, docs §5, at trunk scale)
    conjunctival_artery_diameter: tuple = (25.0, 40.0)   # within the 5.9-61 µm of conjunctival arterioles [Khansari2017]
    companion_offset_um: tuple = (75.0, 120.0)       # vein beside artery, centre to centre (measured 85 µm, docs §4)
    fornix_mm: dict = field(default_factory=lambda: {"superior": 15.0, "temporal": 12.0, "inferior": 10.5,
                                                     "nasal": 8.0})   # fornix depths 15.3-16.1 / 10.1-10.9 mm [Jutley2016, Khan2014]; horizontal assumed
    episcleral_veins_per_mm: float = 0.6             # radial episcleral veins per mm of limbus (fitted, docs §5)
    episcleral_vein_diameter: tuple = (50.0, 100.0)  # [Asanad2021, Kim2024]
    episcleral_heading_spread: float = 0.8           # rad about the meridian, as at the patch level (docs §5)
    episcleral_curvature: float = 0.0008             # rad/µm SD of the slow bend, as at the patch level (docs §5)
    episcleral_avas_per_mm2: float = 0.75            # arteriovenous anastomoses, 0.5-1 per mm2 [Selbach2005]
    episcleral_depth_um: float = 350.0               # docs §2.2
    conjunctival_trunk_depth_um: float = 160.0       # docs §2.2
    # pressures at the true inlets and outlets, mmHg (docs §11.6)
    # ophthalmic artery ~0.7 x brachial [StraussKedra1987, daSilvaKrieglstein1980]; the anterior ciliary arteries lie
    # downstream of it through the muscular arteries and were never measured: 55 mmHg (0.6 x a mean brachial 93 mmHg),
    # fitted so that the conjunctival red-cell velocities land within the cited ranges (docs §11.6)
    ciliary_artery_pressure: float = 55.0
    palpebral_artery_pressure: float = 50.0          # lid arcades, a little further down the ophthalmic tree (assumed)
    ciliary_body_pressure: float = 45.0              # where perforating branches leave for the intramuscular circle (assumed)
    venous_outlet_pressure: float = 9.0              # episcleral venous pressure 7.6-9.5 mmHg [Zeimer1983, Selbach2005b]
    bed_conductance_scale: float = 0.08              # the lumped fine beds (fitted to the cited trunk velocities)
    # regional variation: trunk density by quadrant, relative; deep vessel density ~25 % nasal and temporal against
    # ~21 % superior and inferior [Akagi2018]
    quadrant_density: dict = field(default_factory=lambda: {"superior": 0.92, "temporal": 1.08, "inferior": 0.92,
                                                            "nasal": 1.08})
    # the palpebral fissure seen from in front, mm: 25.6 mm wide; upper lid ~1.8 mm below the superior limbus, lower
    # lid at the inferior limbus [Rana2023]
    fissure_half_width_mm: float = 12.8
    upper_lid_mm: float = 4.1
    lower_lid_mm: float = -5.9
    tile_mm: float = 1.0


# ----------------------------------------------------------------------------------------------- the globe

def direction(theta, clock_deg):
    """Unit vectors from the eye's centre at polar angle theta (radians, from the corneal axis) and clock angle."""
    psi = np.radians(clock_deg)
    st = np.sin(theta)
    return np.stack([st * np.sin(psi), st * np.cos(psi), np.cos(theta) * np.ones_like(psi)], -1)


class Globe:
    def __init__(self, a: EyeAnatomy):
        self.a = a
        self.R = a.radius_um
        self.theta_limbus = math.asin(a.limbus_radius_um / a.radius_um)

    def theta(self, s_um):
        return self.theta_limbus + np.asarray(s_um, float) / self.R

    def point(self, s_um, clock_deg, depth_um):
        """3D eye coordinates of points given by (s from the limbus, clock angle, depth), µm."""
        s_um, clock_deg, depth_um = np.broadcast_arrays(np.asarray(s_um, float), np.asarray(clock_deg, float),
                                                        np.asarray(depth_um, float))
        return (self.R - depth_um)[..., None] * direction(self.theta(s_um), clock_deg)

    def surface_coordinates(self, xyz):
        """(s µm, clock °, depth µm) of 3D points."""
        r = np.linalg.norm(xyz, axis=-1)
        n = xyz / r[..., None]
        theta = np.arccos(np.clip(n[..., 2], -1, 1))
        clock = np.degrees(np.arctan2(n[..., 0], n[..., 1])) % 360.0
        return (theta - self.theta_limbus) * self.R, clock, self.R - r

    def frame(self, s_um, clock_deg):
        """The local frame at a surface point: centre (unit), e_circ (towards larger clock angle), e_mer (away from
        the limbus)."""
        theta = float(self.theta(s_um))
        c = direction(theta, clock_deg)
        e_mer = direction(theta + 1e-4, clock_deg) - direction(theta - 1e-4, clock_deg)
        e_mer /= np.linalg.norm(e_mer)
        e_circ = np.cross(e_mer, c)
        e_circ /= np.linalg.norm(e_circ)
        if np.dot(e_circ, direction(theta, clock_deg + 1e-3) - c) < 0:
            e_circ = -e_circ
        return c, e_circ, e_mer

    def to_local(self, xyz, frame):
        """Azimuthal equidistant coordinates about a frame's centre: (x along e_circ, y along e_mer, depth), µm."""
        c, e_circ, e_mer = frame
        r = np.linalg.norm(xyz, axis=-1)
        n = xyz / r[..., None]
        cosa = np.clip(n @ c, -1, 1)
        alpha = np.arccos(cosa)
        t = n - cosa[..., None] * c
        tn = np.linalg.norm(t, axis=-1)
        t = t / np.maximum(tn, 1e-12)[..., None]
        rho = self.R * alpha
        return np.stack([rho * (t @ e_circ), rho * (t @ e_mer), self.R - r], -1)

    def from_local(self, local, frame):
        """Inverse of to_local."""
        c, e_circ, e_mer = frame
        x, y, z = local[..., 0], local[..., 1], local[..., 2]
        rho = np.hypot(x, y)
        alpha = rho / self.R
        t = (x[..., None] * e_circ + y[..., None] * e_mer) / np.maximum(rho, 1e-12)[..., None]
        n = np.cos(alpha)[..., None] * c + np.sin(alpha)[..., None] * t
        return (self.R - z)[..., None] * n

    def quadrant(self, clock_deg):
        names = ["superior", "temporal", "inferior", "nasal"]
        return names[int(((clock_deg + 45.0) % 360.0) // 90.0)]

    def interpolate_by_clock(self, values, clock_deg):
        """A per-quadrant value interpolated around the clock (quadrant centres at 0, 90, 180, 270°)."""
        order = ["superior", "temporal", "inferior", "nasal", "superior"]
        position = (clock_deg % 360.0) / 90.0
        k = int(position) % 4
        w = position - int(position)
        return (1 - w) * values[order[k]] + w * values[order[k + 1]]


# ----------------------------------------------------------------------------------------------- the coarse level

def _meander(rng, n, step_um, amplitude_um, correlation_um):
    return amplitude_um * _smooth_noise(rng, n, correlation_um / step_um)


def _radial_path(globe, rng, s0, s1, clock, depth0, depth1, amplitude=40.0, correlation=900.0, step=10.0):
    """A nearly straight path along a meridian from s0 to s1 (µm), with a slow lateral meander."""
    n = max(3, int(abs(s1 - s0) / step) + 1)
    s = np.linspace(s0, s1, n)
    lateral = _meander(rng, n, step, amplitude, correlation)
    lateral *= np.sin(np.linspace(0, np.pi, n)) ** 0.3          # the ends stay where they are
    theta = globe.theta(s)
    clock_path = clock + np.degrees(lateral / (globe.R * np.sin(theta)))
    depth = np.linspace(depth0, depth1, n)
    return globe.point(s, clock_path, depth)


def _ring_path(globe, s, clock0, clock1, depth, step=10.0):
    theta = float(globe.theta(s))
    arc = np.radians((clock1 - clock0) % 360.0) * globe.R * math.sin(theta)
    n = max(3, int(arc / step) + 1)
    clocks = clock0 + np.linspace(0, (clock1 - clock0) % 360.0, n)
    return globe.point(np.full(n, s), clocks, np.full(n, depth))


def _shape(rng, xy, anatomy, kind):
    """A coarse vessel's path in the unrolled band with the patch level's shape: its meander and its gentle bends of
    100-300 µm (docs §5); both taper to nothing at the ends, so junctions stay put."""
    xy = wiggle(xy, rng, *anatomy.wiggle[kind], step=8.0)
    return wiggle(xy, rng, *anatomy.bend_wiggle, step=8.0)


def _plane_vessel(globe, rng, s0, clock0, s_end, depth0, depth1, heading_spread, curvature_sd, heading_offset=0.0,
                  step=12.0):
    """A long vessel that runs away from the limbus, leaving at a heading drawn about the meridian (plus
    heading_offset for a side branch) and bending slowly at random about that heading (a curvature pulled back
    towards it, as at the patch level), until it reaches s_end; returned in eye coordinates. Episcleral veins drain
    away from the limbus [Meyer1988]: the heading stays within 1.3 rad of the meridian, so none spirals round the
    globe."""
    heading = float(np.clip(rng.normal(0.0, heading_spread), -1.2, 1.2)) + heading_offset   # 0: away from the limbus
    s, c = float(s0), float(clock0)
    pts = []
    n_max = int(3 * abs(s_end - s0) / step) + 10
    curvature = curvature_sd * _smooth_noise(rng, n_max, 250.0 / step)
    # pulled back towards its own initial heading, as at the patch level, so veins keep the spread of courses the
    # recordings show instead of all turning onto the meridian
    target = float(np.clip(heading, -1.3, 1.3))
    for k in range(n_max):
        theta = float(globe.theta(s))
        pts.append((s, c))
        heading += curvature[k] * step + (target - heading) * step / 1000.0
        s += step * math.cos(heading)
        c += math.degrees(step * math.sin(heading) / (globe.R * math.sin(theta)))
        if s >= s_end or s <= 0:
            break
    pts = np.array(pts)
    depth = np.linspace(depth0, depth1, len(pts))
    return globe.point(pts[:, 0], pts[:, 1], depth)


@dataclass
class Eye:
    """One synthetic eye: its anatomy, the scene settings it shares with its views, and its network at every level of
    detail, solved once. The same seed always gives the same eye.

    fine_region: (clock from, clock to (degrees, clockwise), s from, s to (µm from the limbus)) to grow the fine
    level over part of the conjunctiva only (the coarse level's lumped beds stand for it elsewhere); None (the
    default) grows it over the whole conjunctiva."""
    seed: int = 0
    anatomy: EyeAnatomy = field(default_factory=EyeAnatomy)
    scene: Scene = field(default_factory=Scene)
    fine_region: tuple | None = None

    def __post_init__(self):
        import time as _time
        self.globe = Globe(self.anatomy)
        self.scene = replace(self.scene, optics=replace(self.scene.optics, eye_radius_um=self.anatomy.radius_um))
        self.seconds = {}
        start = _time.perf_counter()
        self.coarse = build_coarse(self)
        self.seconds["coarse"] = _time.perf_counter() - start
        start = _time.perf_counter()
        self.network = build_whole(self)
        self.seconds["fine"] = _time.perf_counter() - start
        start = _time.perf_counter()
        # the haematocrit iteration is stopped at 100 rounds: a few near-stagnant loops keep flipping, while the
        # flows of the rest differ from 400 rounds by 0.04 % (median) and 0.7 % (99th percentile); the final pass
        # conserves flow and red cells exactly either way
        self.flow = solve(self.network, self.scene.physiology, max_iterations=100)
        self.seconds["flow"] = _time.perf_counter() - start
        self._bounds = None

    def in_fine_region(self, s_um, clock_deg, margin=0.0):
        """Whether a surface point (s µm from the limbus, clock °) lies in the conjunctiva of the fine region."""
        fornix = self.globe.interpolate_by_clock(self.anatomy.fornix_mm, clock_deg % 360.0) * 1000.0
        if not (-margin <= s_um <= fornix + margin):
            return False
        if self.fine_region is None:
            return True
        c0, c1, s0, s1 = self.fine_region
        return (clock_deg - c0) % 360.0 <= (c1 - c0) % 360.0 and s0 - margin <= s_um <= s1 + margin

    def edge_bounds(self):
        """Per edge of the network: the centre and radius of a sphere holding it, and its level of detail."""
        if self._bounds is None:
            centres = np.array([0.5 * (e.points.min(0) + e.points.max(0)) for e in self.network.edges])
            radii = np.array([0.5 * np.linalg.norm(e.points.max(0) - e.points.min(0)) for e in self.network.edges])
            levels = np.array([e.level for e in self.network.edges])
            self._bounds = (centres, radii, levels)
        return self._bounds

    # the tiles
    def tile_key(self, s_um, clock_deg):
        t = self.anatomy.tile_mm * 1000.0
        i = int(math.floor(s_um / t))
        n = self.tiles_around(i)
        j = int(math.floor((clock_deg % 360.0) / 360.0 * n))
        return i, j

    def tiles_around(self, i):
        t = self.anatomy.tile_mm * 1000.0
        theta = float(self.globe.theta((i + 0.5) * t))
        return max(4, int(round(2 * math.pi * self.globe.R * math.sin(theta) / t)))

    def tile_centre(self, key):
        i, j = key
        t = self.anatomy.tile_mm * 1000.0
        n = self.tiles_around(i)
        return (i + 0.5) * t, (j + 0.5) * 360.0 / n


def build_coarse(eye: Eye):
    """The coarse network of the whole globe, in eye coordinates."""
    a, g = eye.anatomy, eye.globe
    A = eye.scene.anatomy                            # the patch level's shape parameters
    rng = np.random.default_rng([eye.seed, 11])
    b = _Builder()
    pressure = {}
    epi = a.episcleral_depth_um
    conj = a.conjunctival_trunk_depth_um

    def edge(points, d, kind, layer, u=None, v=None):
        u = b.node(points[0]) if u is None else u
        v = b.node(points[-1]) if v is None else v
        b.edge(u, v, points, d, kind, layer, b.vessel(kind))
        return len(b.edges) - 1, u, v

    # 1. anterior ciliary arteries: from the rectus insertions forward to the episcleral circle
    circle_nodes = []
    for muscle, clock in MUSCLES.items():
        s_ins = a.insertion_mm[muscle] * 1000.0
        counts, shares = a.arteries_per_muscle[muscle]
        n = int(rng.choice(counts, p=np.asarray(shares) / np.sum(shares)))
        theta = float(g.theta(s_ins))
        half = 0.5 * a.insertion_width_mm[muscle] * 1000.0 / (g.R * math.sin(theta))
        offsets = [0.0] if n == 1 else list(np.degrees(np.linspace(-half, half, n + 2)[1:-1]))
        for off in offsets:
            c = clock + off
            d = rng.uniform(*a.ciliary_artery_diameter)
            pts = _radial_path(g, rng, s_ins, a.circle_mm * 1000.0, c, epi, epi, amplitude=60.0)
            i, u, v = edge(pts, d, "ciliary artery", "episcleral")
            b.boundary[u] = "arterial"
            pressure[u] = a.ciliary_artery_pressure
            circle_nodes.append((c % 360.0, v, d))
            # a perforating branch into the sclera, about halfway: it leaves the field for the ciliary body
            mid = len(pts) // 2
            node = b.split(i, mid)
            p = b.nodes[node]
            s_p, c_p, _ = g.surface_coordinates(p)
            deep = g.point(s_p + 800.0, c_p, epi + 450.0)            # oblique through the sclera (docs §2.4)
            path = np.vstack([p, g.point(s_p + 300.0, c_p, epi + 200.0), deep])
            j, _, w = edge(resample(path, 10.0), 25.0, "perforating artery", "perforating", u=node)
            b.boundary[w] = "arterial"
            pressure[w] = a.ciliary_body_pressure
    # 2. the episcleral arterial circle joining the ciliary arteries around the limbus
    circle_nodes.sort()
    circle_edges = []
    for k in range(len(circle_nodes)):
        c0, n0, _ = circle_nodes[k]
        c1, n1, _ = circle_nodes[(k + 1) % len(circle_nodes)]
        if rng.random() < a.circle_gap_fraction:
            continue
        pts = _ring_path(g, a.circle_mm * 1000.0, c0, c1, epi)
        i, _, _ = edge(pts, rng.uniform(*a.circle_diameter), "episcleral artery", "episcleral", u=n0, v=n1)
        circle_edges.append(i)
    # 3. the limbal venous ring and the arterial limbal arcade
    lv = []
    lv_clocks = np.arange(0, 360, 6.0)
    for c in lv_clocks:
        lv.append(b.node(g.point(a.limbal_vein_mm * 1000.0, c, conj)))
    for k in range(len(lv)):
        pts = _ring_path(g, a.limbal_vein_mm * 1000.0, lv_clocks[k], lv_clocks[(k + 1) % len(lv)], conj)
        edge(pts, 35.0, "limbal vein", "conjunctival", u=lv[k], v=lv[(k + 1) % len(lv)])
    # 4. the coarse conjunctival network, grown over the conjunctival band unrolled around the clock: anterior
    #    conjunctival arteries rise from the episcleral circle and branch back towards the watershed, posterior
    #    conjunctival arteries enter from the fornix and branch forward to meet them; venular trees drain to the
    #    fornix and to the limbal venous ring, a share of their growth following the arteries so that veins accompany
    #    them. Space colonisation [Runions2007] at trunk scale; the tiles add everything finer.
    from .network import _chaikin, _murray_diameters, _tree_paths, colonise
    s_ref = 6000.0
    C = 2 * math.pi * g.R * math.sin(float(g.theta(s_ref)))         # µm of x per full turn
    s_c = a.circle_mm * 1000.0

    def to_plane(s_um, clock):
        return np.column_stack([(np.asarray(clock) % 360.0) / 360.0 * C, np.asarray(s_um)])

    def from_plane(xy, depth):
        clock = (xy[:, 0] / C * 360.0) % 360.0
        return g.point(xy[:, 1], clock, np.full(len(xy), depth))

    fornix_of = lambda c: g.interpolate_by_clock(a.fornix_mm, c % 360.0) * 1000.0
    density = lambda c: g.interpolate_by_clock(a.quadrant_density, c % 360.0)
    # attractors over the band, by true area (more towards the equator, where the circles are longer)
    n_att = int(a.coarse_attractors_per_mm2 * C * 1e-3 * 14.0)
    clocks = rng.uniform(0, 360, n_att * 2)
    ss = rng.uniform(s_c + 200.0, 16000.0, n_att * 2)
    keep = (ss < np.array([fornix_of(c) for c in clocks])) & (rng.random(n_att * 2) < np.sin(g.theta(ss)) * np.array([density(c) for c in clocks]) / 1.2)
    attractors = to_plane(ss[keep], clocks[keep])
    roots, headings, sources = [], [], []
    arcade_ends = []
    clock = rng.uniform(0, 360)
    end = clock + 360.0
    while clock < end:                                   # anterior conjunctival arteries from the circle
        roots.append(to_plane([s_c + 150.0], [clock])[0])
        headings.append(math.pi / 2)
        sources.append(("anterior", clock % 360.0))
        theta_c = float(g.theta(s_c))
        clock += np.degrees(a.anterior_artery_spacing_mm * 1000.0 / density(clock) / (g.R * math.sin(theta_c))) * rng.uniform(0.8, 1.2)
    clock = rng.uniform(0, 360)
    end = clock + 360.0
    while clock < end:                                   # posterior conjunctival arteries from the fornix
        f = fornix_of(clock)
        roots.append(to_plane([f], [clock])[0])
        headings.append(-math.pi / 2)
        sources.append(("posterior", clock % 360.0))
        clock += np.degrees(a.posterior_artery_spacing_mm * 1000.0 / (g.R * math.sin(float(g.theta(f))))) * rng.uniform(0.8, 1.2)
    positions, parent, tree = colonise(rng, roots, headings, attractors, a.coarse_growth_step_um,
                                       a.coarse_growth_influence_um, a.coarse_growth_kill_um, persistence=a.coarse_growth_persistence,
                                       radial_bias=a.coarse_radial_bias, branch_spacing=a.coarse_branch_spacing_um)
    positions, parent, tree = _prune_terminals(positions, parent, tree, a.coarse_max_terminal_um)
    paths, children = _tree_paths(parent)
    leaf_d = rng.uniform(*a.coarse_leaf_diameter, len(parent))
    d = np.minimum(_murray_diameters(parent, children, leaf_d, eye.scene.anatomy.murray_exponent_arterial),
                   a.coarse_artery_max_um)
    node_of = {}

    def node_for(i, depth):
        if i not in node_of:
            node_of[i] = b.node(from_plane(positions[i][None], depth)[0])
        return node_of[i]

    artery_edges = []
    for path in paths:
        xy = _chaikin(positions[path], rounds=3)
        chord = xy[0] + np.linspace(0, 1, len(xy))[:, None] * (xy[-1] - xy[0])
        xy = chord + 0.6 * (xy - chord)
        xy = _shape(rng, xy, A, "arteriole")
        pts = from_plane(resample(xy, 10.0), conj)
        u, v = node_for(path[0], conj), node_for(path[-1], conj)
        b.edge(u, v, pts, float(np.median(d[path[1:]])), "conjunctival artery", "conjunctival", b.vessel("conjunctival artery"))
        artery_edges.append(len(b.edges) - 1)
    for k, (kind, c) in enumerate(sources):
        if k not in node_of:
            continue
        node = node_of[k]
        if kind == "posterior":
            b.boundary[node] = "arterial"
            pressure[node] = a.palpebral_artery_pressure
            continue
        # the anterior artery rises obliquely from the circle, and sends a short branch forward to the limbal arcade
        start = g.point(s_c, c, epi)
        nearest = min(circle_edges, key=lambda i: np.min(np.linalg.norm(b.edges[i].points - start, axis=1))) if circle_edges else None
        if nearest is None:
            b.boundary[node] = "arterial"
            pressure[node] = a.ciliary_artery_pressure
            continue
        pts_c = b.edges[nearest].points
        kk = int(np.argmin(np.linalg.norm(pts_c - start, axis=1)))
        if 2 <= kk <= len(pts_c) - 3:
            root = b.split(nearest, kk)
            circle_edges.append(len(b.edges) - 1)
        else:
            root = b.edges[nearest].u if kk < len(pts_c) / 2 else b.edges[nearest].v
        d0 = float(d[k]) if children[k] else float(leaf_d[k])
        # the anterior conjunctival artery's origin from the circle, at the calibre of these arteries
        d_root = float(np.clip(d0, *a.conjunctival_artery_diameter))
        edge(np.vstack([b.nodes[root], b.nodes[node]]), d_root, "communicating artery", "communicating", u=root, v=node)
        forward = _radial_path(g, rng, s_c + 150.0, a.limbal_arcade_mm * 1000.0, c, conj, conj, amplitude=25.0)
        forward[0] = b.nodes[node]
        fi, _, f_end = edge(forward, 0.7 * d0, "conjunctival artery", "conjunctival", u=node)
        artery_edges.append(fi)
        lv_k = int(np.argmin(np.abs(((lv_clocks - c + 180) % 360) - 180)))
        loop = np.vstack([b.nodes[f_end], g.point(0.5 * (a.limbal_arcade_mm + a.limbal_vein_mm) * 1000.0, c + 0.3, conj),
                          b.nodes[lv[lv_k]]])
        li, _, _ = edge(resample(loop, 10.0), 8.0, "capillary", "conjunctival", u=f_end, v=lv[lv_k])
        b.edges[li].length_override = rng.uniform(1500.0, 2100.0)     # a palisade hairpin, 1.5-2.1 mm [Kemenes2025]
        arcade_ends.append((c % 360.0, f_end))
    # the arterial limbal arcade: the forward branches' ends joined around the limbus [VanBuskirk1989, Kemenes2025]
    arcade_ends.sort()
    for k in range(len(arcade_ends)):
        c0, n0 = arcade_ends[k]
        c1, n1 = arcade_ends[(k + 1) % len(arcade_ends)]
        if n0 == n1:
            continue
        pts = _ring_path(g, a.limbal_arcade_mm * 1000.0, c0, c1, conj)
        pts[0], pts[-1] = b.nodes[n0], b.nodes[n1]
        edge(pts, rng.uniform(18.0, 28.0), "conjunctival artery", "conjunctival", u=n0, v=n1)
    # the venular trees: roots at the fornix (outlets) and on the limbal venous ring; half of their attractors lie
    # beside the arteries just grown, so veins accompany arteries
    # the attractors beside an artery lie to one side of it, at about the recorded spacing of an artery and its
    # companion vein (85 µm centre to centre, docs §4), so the veins run alongside the arteries, not on top of them
    beside_parts = []
    for i in artery_edges:
        xy = to_plane(*g.surface_coordinates(b.edges[i].points)[:2])
        if len(xy) < 3:
            continue
        tangent = np.gradient(xy, axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
        normal = np.stack([-tangent[:, 1], tangent[:, 0]], 1)
        offset = rng.choice([-1.0, 1.0]) * rng.uniform(*a.companion_offset_um, len(xy))
        beside_parts.append(xy + normal * offset[:, None])
    art_pts = np.vstack(beside_parts)
    pick = rng.choice(len(art_pts), min(len(art_pts) // 3, len(attractors)), replace=False)
    beside = art_pts[pick]
    v_attractors = np.vstack([attractors[rng.random(len(attractors)) < 0.5], beside])
    v_roots, v_headings, v_sources = [], [], []
    clock = rng.uniform(0, 360)
    end = clock + 360.0
    while clock < end:
        f = fornix_of(clock)
        v_roots.append(to_plane([f], [clock])[0])
        v_headings.append(-math.pi / 2)
        v_sources.append(("fornix", clock % 360.0))
        clock += np.degrees(a.fornix_vein_spacing_mm * 1000.0 / (g.R * math.sin(float(g.theta(f))))) * rng.uniform(0.8, 1.2)
    for k, c in enumerate(lv_clocks[::2]):
        v_roots.append(to_plane([a.limbal_vein_mm * 1000.0 + 50.0], [c])[0])
        v_headings.append(math.pi / 2)
        v_sources.append(("limbal", 2 * k))
    vpos, vpar, vtree = colonise(rng, v_roots, v_headings, v_attractors, a.coarse_growth_step_um,
                                 a.coarse_growth_influence_um, a.coarse_growth_kill_um, persistence=a.coarse_growth_persistence,
                                 radial_bias=a.coarse_radial_bias, branch_spacing=a.coarse_branch_spacing_um)
    vpos, vpar, vtree = _prune_terminals(vpos, vpar, vtree, a.coarse_max_terminal_um)
    vpaths, vchildren = _tree_paths(vpar)
    vleaf = rng.uniform(*a.coarse_leaf_diameter, len(vpar)) * 1.3
    vd = np.minimum(_murray_diameters(vpar, vchildren, vleaf, eye.scene.anatomy.murray_exponent_venous),
                    a.coarse_vein_max_um)
    vnode_of = {}
    for k, (kind, ref) in enumerate(v_sources):
        if kind == "limbal":
            vnode_of[k] = lv[ref]
    for path in vpaths:
        xy = _chaikin(vpos[path], rounds=3)
        chord = xy[0] + np.linspace(0, 1, len(xy))[:, None] * (xy[-1] - xy[0])
        xy = chord + 0.6 * (xy - chord)
        xy = _shape(rng, xy, A, "venule")
        pts = from_plane(resample(xy, 10.0), conj + 10.0)
        ends = []
        for i in (path[0], path[-1]):
            if i not in vnode_of:
                vnode_of[i] = b.node(from_plane(vpos[i][None], conj + 10.0)[0])
            ends.append(vnode_of[i])
        pts[0], pts[-1] = b.nodes[ends[0]], b.nodes[ends[1]]
        b.edge(ends[0], ends[1], pts, float(np.median(vd[path[1:]])), "conjunctival vein", "conjunctival", b.vessel("conjunctival vein"))
    for k, (kind, ref) in enumerate(v_sources):
        if kind == "fornix" and k in vnode_of:
            b.boundary[vnode_of[k]] = "venous"
            pressure[vnode_of[k]] = a.venous_outlet_pressure
    # 5. episcleral veins: from the limbal venous ring away from the limbus, past the insertions, in long gently
    #    bending runs whose heading varies about the meridian, with side branches (docs §5)
    n_veins = int(round(a.episcleral_veins_per_mm * 2 * math.pi * a.limbus_radius_um * 1e-3))
    for c in np.sort(rng.uniform(0, 360, n_veins)):
        quad = g.quadrant(c)
        lv_k = int(np.argmin(np.abs(((lv_clocks - c + 180) % 360) - 180)))
        s_end = (a.insertion_mm[quad] + 2.0) * 1000.0
        pts = _plane_vessel(g, rng, a.limbal_vein_mm * 1000.0, c, s_end, conj, epi, a.episcleral_heading_spread,
                            a.episcleral_curvature)
        pts[0] = b.nodes[lv[lv_k]]
        d = float(np.clip(rng.normal(*eye.scene.anatomy.episcleral_vein_mean_sd), *a.episcleral_vein_diameter))
        i, _, end = edge(pts, d, "episcleral vein", "episcleral", u=lv[lv_k])
        b.boundary[end] = "venous"
        pressure[end] = a.venous_outlet_pressure
        for _ in range(rng.poisson(0.4 * b.edges[i].length * 1e-3)):
            e = b.edges[i]
            if len(e.points) < 40:
                break
            k = int(rng.integers(15, len(e.points) - 15))
            node = b.split(i, k)
            s_k, c_k, z_k = g.surface_coordinates(b.nodes[node])
            side = rng.choice([-1.0, 1.0])
            branch = _plane_vessel(g, rng, s_k, c_k, s_end, z_k, epi, a.episcleral_heading_spread, a.episcleral_curvature,
                                   heading_offset=side * math.radians(rng.uniform(25, 50)))
            branch[0] = b.nodes[node]
            gamma = eye.scene.anatomy.murray_exponent_venous
            ratio = rng.uniform(0.5, 0.8)
            j, _, w = edge(branch, max(a.episcleral_vein_diameter[0], d * ratio / (1 + ratio ** gamma) ** (1 / gamma)),
                           "episcleral vein", "episcleral", u=node)
            b.boundary[w] = "venous"
            pressure[w] = a.venous_outlet_pressure
    # 5b. episcleral arteriovenous anastomoses: the episclera has no capillary bed; its arteries open into the
    #     venous network through anastomoses, 0.5-1 per mm2 [Selbach2005], at resting tone (synth.network)
    epi_art = [i for i, e in enumerate(b.edges) if e.kind in ("episcleral artery", "ciliary artery")]
    epi_vein = [i for i, e in enumerate(b.edges) if e.kind == "episcleral vein"]
    if epi_art and epi_vein:
        vp = np.vstack([b.edges[i].points for i in epi_vein])
        vid = np.concatenate([np.full(len(b.edges[i].points), i) for i in epi_vein])
        vk = np.concatenate([np.arange(len(b.edges[i].points)) for i in epi_vein])
        vtree_e = cKDTree(vp)
        band_mm2 = 2 * math.pi * g.R * math.sin(float(g.theta(4000.0))) * 1e-3 * 6.0
        n_ava = rng.poisson(a.episcleral_avas_per_mm2 * band_mm2)
        lengths = np.array([b.edges[i].length for i in epi_art])
        for _ in range(n_ava):
            ai = epi_art[int(rng.choice(len(epi_art), p=lengths / lengths.sum()))]
            pts = b.edges[ai].points
            if len(pts) < 6:
                continue
            idx = int(rng.integers(2, len(pts) - 2))
            dist, j = vtree_e.query(pts[idx])
            if dist > 600.0 or vk[j] <= 1 or vk[j] >= len(b.edges[vid[j]].points) - 2:
                continue
            ava = Edge(-1, -1, np.vstack([pts[idx], vp[j]]), 0.0, "bed", "bed", -1)
            ava.bed = (ai, idx, int(vid[j]), int(vk[j]))
            ava.ava = True
            b.edges.append(ava)
    # 6. lumped beds: every 0.5 mm along a coarse conjunctival artery, the fine vessels the tiles will draw carry its
    #    blood to the nearest conjunctival vein; one equivalent resistance stands for them at the coarse level
    veins = [i for i, e in enumerate(b.edges) if e.kind == "conjunctival vein"]
    vein_pts = np.vstack([b.edges[i].points for i in veins])
    vein_id = np.concatenate([np.full(len(b.edges[i].points), i) for i in veins])
    vein_k = np.concatenate([np.arange(len(b.edges[i].points)) for i in veins])
    vtree = cKDTree(vein_pts)
    for ai in [i for i, e in enumerate(b.edges) if e.kind == "conjunctival artery"]:
        art = b.edges[ai]
        n_beds = max(1, int(art.length / 500.0))
        for k in range(1, n_beds + 1):
            idx = int(len(art.points) * k / (n_beds + 1))
            if idx <= 1 or idx >= len(art.points) - 2:
                continue
            dist, j = vtree.query(art.points[idx])
            if dist > 900.0 or vein_k[j] <= 1 or vein_k[j] >= len(b.edges[vein_id[j]].points) - 2:
                continue
            bed = Edge(-1, -1, np.vstack([art.points[idx], vein_pts[j]]), 0.0, "bed", "bed", -1)
            bed.bed = (ai, idx, int(vein_id[j]), int(vein_k[j]))
            b.edges.append(bed)
    # every trunk tip drains through a bed of its own: a tip beyond its trunk's last bed would carry no flow, yet the
    # tiles root fine trees on it, and a fine tree draining into a vessel without flow ends in a stagnant column
    arteries = [i for i, e in enumerate(b.edges) if e.kind == "conjunctival artery"]
    art_pts = np.vstack([b.edges[i].points for i in arteries])
    art_id = np.concatenate([np.full(len(b.edges[i].points), i) for i in arteries])
    art_k = np.concatenate([np.arange(len(b.edges[i].points)) for i in arteries])
    atree = cKDTree(art_pts)
    degree = {}
    for e in b.edges:
        if e.kind != "bed":
            degree[e.u] = degree.get(e.u, 0) + 1
            degree[e.v] = degree.get(e.v, 0) + 1
    for i, e in enumerate(list(b.edges)):
        if e.kind not in ("conjunctival artery", "conjunctival vein") or len(e.points) < 5:
            continue
        for node, idx in ((e.u, 1), (e.v, len(e.points) - 2)):
            if degree.get(node, 0) != 1 or node in b.boundary:
                continue
            if e.kind == "conjunctival artery":
                dist, j = vtree.query(e.points[idx])
                other, ok = (int(vein_id[j]), int(vein_k[j])), 1 <= vein_k[j] <= len(b.edges[vein_id[j]].points) - 2
                pair = (i, idx) + other
            else:
                dist, j = atree.query(e.points[idx])
                other, ok = (int(art_id[j]), int(art_k[j])), 1 <= art_k[j] <= len(b.edges[art_id[j]].points) - 2
                pair = other + (i, idx)
            if dist > 1500.0 or not ok:
                continue
            bed = Edge(-1, -1, np.vstack([b.edges[pair[0]].points[pair[1]], b.edges[pair[2]].points[pair[3]]]), 0.0,
                       "bed", "bed", -1)
            bed.bed = pair
            b.edges.append(bed)
    # resolve the beds into real nodes (split the artery and vein edges at the bed points)
    beds = [e for e in b.edges if e.kind == "bed"]
    b.edges = [e for e in b.edges if e.kind != "bed"]
    _attach_beds(b, beds, a)
    # their course's unevenness and their calibre along their length are drawn once the network is whole
    # (build_whole), so they run on continuously through every split
    net = Network(np.asarray(b.nodes), b.edges, dict(b.boundary), (0, 0, 0, 0), (0, 0, 0, 0), dict(b.vessel_kind))
    net.boundary_pressure = {k: v for k, v in pressure.items()}
    return net


def _attach_beds(b, beds, a):
    """Split trunks at their bed points and add the lumped bed edges (a short tube of equal conductance)."""
    by_edge = {}
    for bed in beds:
        ai, idx, vi, vk = bed.bed
        by_edge.setdefault(ai, []).append(idx)
        by_edge.setdefault(vi, []).append(vk)
    node_at = {}
    for ei in list(by_edge):
        points = sorted(set(by_edge[ei]), reverse=True)
        for idx in points:
            e = b.edges[ei]
            if idx <= 0 or idx >= len(e.points) - 1:
                continue
            key = (ei, idx)
            node = b.split(ei, idx)            # the head part keeps index ei, the tail is new; idx descends so earlier
            node_at[key] = node                # indices on ei stay valid
    for bed in beds:
        ai, idx, vi, vk = bed.bed
        u, v = node_at.get((ai, idx)), node_at.get((vi, vk))
        if u is None or v is None:
            continue
        p, q = b.nodes[u], b.nodes[v]
        if getattr(bed, "ava", False):
            # an episcleral arteriovenous anastomosis: a short vessel of 5-8 µm at resting tone (docs §2.4)
            pts = resample(np.vstack([p, q]), 10.0)
            b.edges.append(Edge(u, v, pts, float(np.random.default_rng(u * 7919 + v).uniform(5.0, 8.0)),
                                "arteriovenous anastomosis", "episcleral", b.vessel("arteriovenous anastomosis")))
            continue
        # an equivalent tube: 12 µm wide, as long as needed for the bed's conductance (fitted, docs §11.6)
        length = max(np.linalg.norm(q - p), 1.0) * 4.0 / max(a.bed_conductance_scale, 1e-6)
        pts = np.vstack([p, p + (q - p) * 0.5, q])
        b.edges.append(Edge(u, v, pts, 12.0, "bed", "bed", -1))
        b.edges[-1].length_override = length


# ----------------------------------------------------------------------------------------------- the fine level

TRUNK_KINDS = ("conjunctival artery", "conjunctival vein", "episcleral vein", "episcleral artery", "ciliary artery",
               "limbal vein")
LEVEL_NAMES = {0: "coarse", 1: "fine trees", 2: "capillary plexus"}


def _tile_seed(eye, key, salt=""):
    digest = hashlib.sha256(f"{eye.seed}:{key[0]}:{key[1]}{salt}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def fine_tile_keys(eye):
    """The tiles of the fine level: every tile of the conjunctiva, from the limbus to the fornix, whose centre lies
    in the eye's fine region."""
    t = eye.anatomy.tile_mm * 1000.0
    s_max = max(eye.anatomy.fornix_mm.values()) * 1000.0
    keys = []
    for i in range(int(math.ceil(s_max / t))):
        for j in range(eye.tiles_around(i)):
            s, c = eye.tile_centre((i, j))
            if eye.in_fine_region(s, c, margin=0.5 * t):
                keys.append((i, j))
    return keys


def build_tile(eye: Eye, key):
    """The conjunctival arteriolar and venular trees of one tile, in eye coordinates, grown from roots on the coarse
    trunks that cross it and on the episcleral vessels beneath it, from a seed that depends only on the eye's seed
    and the tile's position. No capillaries: the eye's one capillary plexus joins every tile's terminals.
    Returns dict(network, roots: [(coarse edge, point index, node)], leaves, frame, domain, key)."""
    a, g = eye.anatomy, eye.globe
    s_c, clock_c = eye.tile_centre(key)
    frame = g.frame(s_c, clock_c)
    t = a.tile_mm * 1000.0
    n = eye.tiles_around(key[0])
    width = 2 * math.pi * g.R * math.sin(float(g.theta(s_c))) / n
    domain = (-width / 2, width / 2, -t / 2, t / 2)
    coarse = eye.coarse
    rng = np.random.default_rng(_tile_seed(eye, key))
    roots, attach = [], []
    spacing = 1000.0 / max(eye.scene.anatomy.arterial_trees_per_mm2 * 4.0, 1e-3)    # roots per mm of trunk
    centres, radii = _coarse_bounds(eye)
    near = set(np.flatnonzero(np.linalg.norm(centres - frame[0] * g.R, axis=1) < radii + math.hypot(width, t)).tolist())
    for ei, e in enumerate(coarse.edges):
        if ei not in near or e.kind not in ("conjunctival artery", "conjunctival vein"):
            continue
        local = g.to_local(e.points, frame)
        inside = (local[:, 0] > domain[0]) & (local[:, 0] < domain[1]) & (local[:, 1] > domain[2]) & (local[:, 1] < domain[3])
        if not inside.any():
            continue
        idx = np.flatnonzero(inside)
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(local[idx, :2], axis=0), axis=1))]
        marks = np.arange(rng.uniform(0, spacing), arc[-1], spacing * rng.uniform(0.8, 1.2))
        for m in marks:
            k = idx[int(np.searchsorted(arc, m).clip(0, len(idx) - 1))]
            if k <= 1 or k >= len(e.points) - 2:
                continue
            tangent = local[min(k + 3, len(local) - 1), :2] - local[max(k - 3, 0), :2]
            heading = math.atan2(tangent[1], tangent[0]) + rng.choice([-1, 1]) * math.radians(rng.uniform(30, 85))
            roots.append(dict(venous=e.kind == "conjunctival vein", top=local[k].copy(), heading=heading))
            attach.append((ei, k))
    # communicating vessels, as at the patch level: they rise obliquely from the episcleral vessels beneath the tile
    # and root conjunctival trees that grow in every direction, not only off the radial trunks (docs §2.4)
    A = eye.scene.anatomy
    under = {False: [], True: []}
    for ei, e in enumerate(coarse.edges):
        venous = e.kind == "episcleral vein"
        if ei not in near or (not venous and e.kind not in ("episcleral artery", "ciliary artery")):
            continue
        local = g.to_local(e.points, frame)
        inside = (local[:, 0] > domain[0] + 100) & (local[:, 0] < domain[1] - 100) & \
                 (local[:, 1] > domain[2] + 100) & (local[:, 1] < domain[3] - 100)
        inside[:2] = inside[-2:] = False
        under[venous] += [(ei, int(k), local[k]) for k in np.flatnonzero(inside)]
    top, bottom = A.conjunctival_plexus
    have = {v: any(r["venous"] == v for r in roots) for v in (False, True)}
    n_comm = rng.poisson(A.communicating_per_mm2 * width * t * 1e-6)
    n_comm = max(n_comm, sum(1 for v in (False, True) if not have[v] and under[v]))
    for k in range(n_comm):
        missing = [v for v in (False, True) if not have[v] and under[v]]
        venous = missing[0] if missing else ((k % 2 == 1) if under[False] and under[True] else bool(under[True]))
        if not under[venous]:
            continue
        for _ in range(6):
            ei, kk, point = under[venous][int(rng.integers(len(under[venous])))]
            angle = math.radians(rng.uniform(*A.communicating_angle_deg))
            z_top = rng.uniform(top + 0.5 * (bottom - top), bottom)
            run = max(point[2] - z_top, 20.0) / math.tan(angle)
            direction_ = rng.uniform(0, 2 * math.pi)
            top_xyz = np.array([point[0] + run * math.cos(direction_), point[1] + run * math.sin(direction_), z_top])
            if domain[0] + 60 < top_xyz[0] < domain[1] - 60 and domain[2] + 60 < top_xyz[1] < domain[3] - 60:
                roots.append(dict(venous=venous, top=top_xyz, heading=rng.uniform(0, 2 * math.pi), bottom=point.copy()))
                attach.append((ei, kk))
                have[venous] = True
                break
    net, root_nodes = generate_fine(eye.scene, domain, roots, _tile_seed(eye, key), mesh=False)
    root_list = [(ei, k, node) for (ei, k), node in zip(attach, root_nodes) if node is not None]
    for e in net.edges:
        e.points = g.from_local(e.points, frame)
    net.nodes = g.from_local(net.nodes, frame) if len(net.nodes) else net.nodes
    return dict(network=net, roots=root_list, leaves=list(net.leaves), frame=frame, domain=domain, key=key)


def _coarse_bounds(eye):
    """Centre and radius of a sphere holding each coarse edge (cached)."""
    if getattr(eye, "_coarse_bounds", None) is None:
        edges = eye.coarse.edges
        lo = np.array([e.points.min(0) for e in edges])
        hi = np.array([e.points.max(0) for e in edges])
        eye._coarse_bounds = (0.5 * (lo + hi), 0.5 * np.linalg.norm(hi - lo, axis=1))
    return eye._coarse_bounds


def _surface_path(g, p, q, rng, amplitude, correlation):
    """A slightly bowed vessel between two points of the eye (µm), drawn in the local frame at its middle."""
    s, c, _ = g.surface_coordinates(0.5 * (p + q))
    frame = g.frame(float(s), float(c))
    ends = g.to_local(np.vstack([p, q]), frame)
    path = _segment(ends[0], ends[1], rng, amplitude, correlation)
    out = g.from_local(path, frame)
    out[0], out[-1] = p, q
    return out


def build_plexus(eye: Eye, keys):
    """One capillary plexus over the whole fine region: the Voronoi tessellation, on the sphere, of a jittered
    lattice of sites laid tile by tile (vertices of degree three, like capillary junctions; docs §5), so the mesh runs
    on across tile borders without a seam. Returns (nodes (n, 3) in eye coordinates, edges [(u, v, points,
    diameter)])."""
    a, g, A = eye.anatomy, eye.globe, eye.scene.anatomy
    spacing = A.capillary_mesh_spacing
    stretch = np.random.default_rng([eye.seed, 23]).uniform(1.0, 1.3)       # mesh cells longer radially
    sites = []
    for key in keys:
        r = np.random.default_rng(_tile_seed(eye, key, ":mesh"))
        s_c, c_c = eye.tile_centre(key)
        frame = g.frame(s_c, c_c)
        t = a.tile_mm * 1000.0
        width = 2 * math.pi * g.R * math.sin(float(g.theta(s_c))) / eye.tiles_around(key[0])
        dy = spacing * 0.866 * stretch
        ys = np.arange(-t / 2 + r.uniform(0, dy), t / 2, dy)
        for row, y in enumerate(ys):
            xs = np.arange(-width / 2 + r.uniform(0, spacing), width / 2, spacing)
            local = np.column_stack([xs, np.full(len(xs), y), np.zeros(len(xs))])
            local[:, :2] += r.normal(0, 0.3 * spacing, (len(xs), 2))
            sites.append(g.from_local(local, frame))
    P = np.vstack(sites)
    # sites laid by neighbouring tiles may fall too close together at their shared border: thin them
    tree = cKDTree(P)
    drop = set()
    for i, j in sorted(tree.query_pairs(0.45 * spacing)):
        if i not in drop and j not in drop:
            drop.add(j)
    P = P[[k for k in range(len(P)) if k not in drop]]
    unit = P / np.linalg.norm(P, axis=1, keepdims=True)
    sv = SphericalVoronoi(unit, radius=1.0)
    sv.sort_vertices_of_regions()
    V = sv.vertices * g.R
    s_v, c_v, _ = g.surface_coordinates(V)
    fornix = np.array([g.interpolate_by_clock(a.fornix_mm, c) * 1000.0 for c in c_v])
    keep = (s_v > a.limbal_vein_mm * 1000.0 + 150.0) & (s_v < fornix)
    keep &= np.array([eye.in_fine_region(s, c) for s, c in zip(s_v, c_v)])
    ridges = set()
    for region in sv.regions:
        for k in range(len(region)):
            i, j = region[k], region[(k + 1) % len(region)]
            if keep[i] and keep[j]:
                ridges.add((min(i, j), max(i, j)))
    ridges = sorted(r for r in ridges if np.linalg.norm(V[r[0]] - V[r[1]]) < 2.5 * spacing)
    rng = np.random.default_rng([eye.seed, 24])
    ridges = [r for r in ridges if rng.random() > A.capillary_edge_drop]
    while True:                         # a dead end carries no flow: prune dangling ridges, again and again
        degree = np.bincount(np.array(ridges).ravel(), minlength=len(V)) if ridges else np.zeros(len(V), int)
        dangling = degree == 1
        if not dangling.any():
            break
        ridges = [r for r in ridges if not (dangling[r[0]] or dangling[r[1]])]
    used = sorted({k for r in ridges for k in r})
    index = {k: n for n, k in enumerate(used)}
    top = A.conjunctival_plexus[0]
    depth = top + np.abs(rng.normal(0, 12.0, len(used)))
    unitv = V[used] / g.R
    nodes = unitv * (g.R - depth)[:, None]
    edges = []
    for i, j in ridges:
        u, v = index[i], index[j]
        pts = _surface_path(g, nodes[u], nodes[v], rng, *A.wiggle["capillary"])
        edges.append((u, v, pts, float(rng.uniform(*A.capillary_diameter))))
    return nodes, edges


def _roughen_trunks(edges, rng, anatomy):
    """A mild unevenness of the course of every coarse vessel, growing with its calibre (anatomy.fine_jitter and
    large_jitter, as at the patch level), across the vessel in the surface's plane and tapered to nothing at its
    ends, so junctions stay put."""
    amp0, ell0 = anatomy.fine_jitter
    for e in edges:
        if e.kind not in TRUNK_KINDS or len(e.points) < 6:
            continue
        pts = e.points
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        if s[-1] <= 0:
            continue
        amplitude = amp0 + anatomy.large_jitter[0] * e.diameter
        ell = ell0 + anatomy.large_jitter[1] * e.diameter
        offset = amplitude * _noise_at(rng, s, ell)
        taper = np.clip(np.minimum(s, s[-1] - s) / (2 * ell), 0, 1)
        tangent = np.gradient(pts, axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
        normal = np.cross(tangent, pts / np.linalg.norm(pts, axis=1, keepdims=True))
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
        e.points = pts + (offset * taper)[:, None] * normal


def build_whole(eye: Eye):
    """The eye's network at every level of detail in one piece: the coarse vessels, the trees of every tile of the
    fine region rooted on them, and the capillary plexus joining the trees' terminals; the coarse level's lumped beds
    remain only where no fine vessels are drawn. Returns the network (eye coordinates; every edge has .level 0, 1 or
    2)."""
    g, A = eye.globe, eye.scene.anatomy
    coarse = eye.coarse
    keys = fine_tile_keys(eye)
    tiles = [build_tile(eye, key) for key in keys]
    b = _Builder()
    b.nodes = [p.copy() for p in coarse.nodes]
    b.vessel_kind = dict(coarse.vessel_names)
    b.next_vessel = max(b.vessel_kind, default=0)
    b.boundary = dict(coarse.boundary)
    for e in coarse.edges:
        c = Edge(e.u, e.v, e.points.copy(), e.diameter, e.kind, e.layer, e.vessel, paired_with=e.paired_with)
        if getattr(e, "length_override", None):
            c.length_override = e.length_override
        c.level = 0
        b.edges.append(c)
    # the lumped beds stand for fine vessels: where the fine level draws them, the beds go
    drop = set()
    for i, e in enumerate(coarse.edges):
        if e.kind == "bed":
            s, c, _ = g.surface_coordinates(e.points[[0, -1]])
            if any(eye.in_fine_region(si, ci) for si, ci in zip(s, c)):
                drop.add(i)
    # split the coarse vessels where trees are rooted (descending, so earlier indices on an edge stay valid)
    at = {}
    for tile in tiles:
        for ei, k, _ in tile["roots"]:
            at.setdefault(ei, set()).add(k)
    split_node = {}
    for ei in sorted(at):
        for k in sorted(at[ei], reverse=True):
            e = b.edges[ei]
            if k <= 0:
                split_node[(ei, k)] = e.u
            elif k >= len(e.points) - 1:
                split_node[(ei, k)] = e.v
            else:
                split_node[(ei, k)] = b.split(ei, k)
                b.edges[-1].level = 0
    # the tiles' trees, their roots merged into the coarse nodes
    leaves = []
    for tile in tiles:
        net = tile["network"]
        remap = {node: split_node[(ei, k)] for ei, k, node in tile["roots"]}
        index = {}
        for k in range(len(net.nodes)):
            if k in remap:
                index[k] = remap[k]
            else:
                index[k] = len(b.nodes)
                b.nodes.append(net.nodes[k].copy())
        vessel_base = b.next_vessel
        for vid, kind in net.vessel_names.items():
            b.vessel_kind[vessel_base + vid] = kind
            b.next_vessel = max(b.next_vessel, vessel_base + vid)
        for e in net.edges:
            pts = e.points.copy()
            pts[0], pts[-1] = b.nodes[index[e.u]], b.nodes[index[e.v]]
            c = Edge(index[e.u], index[e.v], pts, e.diameter, e.kind, e.layer,
                     vessel_base + e.vessel if e.vessel >= 0 else -1,
                     paired_with=vessel_base + e.paired_with if e.paired_with >= 0 else -1)
            c.width = e.width
            c.level = 1
            b.edges.append(c)
        leaves += [index[k] for k in tile["leaves"]]
    # the capillary plexus
    plexus_nodes, plexus_edges = build_plexus(eye, keys)
    base = len(b.nodes)
    b.nodes += [p for p in plexus_nodes]
    rng = np.random.default_rng([eye.seed, 25])
    for u, v, pts, d in plexus_edges:
        c = Edge(base + u, base + v, pts, d, "capillary", "conjunctival", b.vessel("capillary"))
        c.level = 2
        b.edges.append(c)
    # the terminal arterioles and post-capillary venules each open into one or two plexus junctions, and so does the
    # tip of every coarse trunk that ends in the fine region (a trunk branches on into the fine vessels there)
    spacing = A.capillary_mesh_spacing
    degree = np.zeros(len(b.nodes), int)
    for i, e in enumerate(b.edges):
        if i not in drop:
            degree[e.u] += 1
            degree[e.v] += 1
    tips = []
    for i, e in enumerate(b.edges):
        if e.level == 0 and i not in drop and e.kind in ("conjunctival artery", "conjunctival vein"):
            for node in (e.u, e.v):
                if degree[node] == 1 and node not in b.boundary:
                    s, c, _ = g.surface_coordinates(b.nodes[node][None])
                    if eye.in_fine_region(float(s[0]), float(c[0])):
                        tips.append(node)
    if len(plexus_nodes):
        tree = cKDTree(plexus_nodes)
        for node, n_links, d_link in [(k, 1 + int(rng.random() < 0.35), None) for k in leaves] + \
                                     [(k, 3, 9.0) for k in tips]:
            p = b.nodes[node]
            distance, nearest = tree.query(p, k=n_links)
            for dist, near in zip(np.atleast_1d(distance), np.atleast_1d(nearest)):
                if dist > 2.5 * spacing:
                    continue
                pts = _surface_path(g, p, plexus_nodes[near], rng, *A.wiggle["capillary"])
                d = d_link or float(rng.uniform(*A.capillary_diameter))
                c = Edge(node, base + int(near), pts, d, "capillary", "conjunctival", b.vessel("capillary"))
                c.level = 2
                b.edges.append(c)
    # the trunks' course and calibre (continuous along each vessel through every split)
    trunks = [e for i, e in enumerate(b.edges) if getattr(e, "level", 0) == 0 and i not in drop]
    _roughen_trunks(trunks, np.random.default_rng([eye.seed, 26]), A)
    for e in trunks:
        e.width = None
    assign_calibres([e for e in trunks if e.kind in TRUNK_KINDS], np.random.default_rng([eye.seed, 27]), A)
    # and the calibre along every other drawn vessel: the remaining coarse kinds and the capillary plexus
    assign_calibres([e for e in trunks if e.kind not in TRUNK_KINDS and e.kind != "bed"] +
                    [e for e in b.edges if e.level == 2], np.random.default_rng([eye.seed, 28]), A)
    edges = [e for i, e in enumerate(b.edges) if i not in drop]
    net = Network(np.asarray(b.nodes), edges, dict(b.boundary), (0, 0, 0, 0), (0, 0, 0, 0), dict(b.vessel_kind))
    net.boundary_pressure = dict(coarse.boundary_pressure)
    return _keep_perfusable(net)


def _keep_perfusable(net):
    """Drop zero-length edges and the components that reach no boundary node (they could carry no flow)."""
    n = len(net.nodes)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    edges = [e for e in net.edges if e.u != e.v and e.length > 1.0]
    for e in edges:
        ru, rv = find(e.u), find(e.v)
        if ru != rv:
            parent[ru] = rv
    rooted = {find(k) for k in net.boundary}
    edges = [e for e in edges if find(e.u) in rooted]
    used = sorted({k for e in edges for k in (e.u, e.v)})
    remap = {old: new for new, old in enumerate(used)}
    for e in edges:
        e.u, e.v = remap[e.u], remap[e.v]
    out = Network(net.nodes[used], edges, {remap[k]: r for k, r in net.boundary.items() if k in remap}, net.domain,
                  net.field_of_view, net.vessel_names)
    out.boundary_pressure = {remap[k]: p for k, p in net.boundary_pressure.items() if k in remap}
    return out


def perfusion(eye: Eye, threshold_mm_s=0.005):
    """Share of the length of each level that carries red cells moving at least threshold_mm_s, and the number of
    edges that do not (dead ends excepted, which carry no flow by construction and are reported apart)."""
    v = np.abs(eye.flow.rbc_velocity)
    hd = eye.flow.hd
    out = {}
    for level, name in LEVEL_NAMES.items():
        sel = np.array([e.level == level for e in eye.network.edges])
        if not sel.any():
            continue
        length = eye.flow.length[sel]
        moving = (v[sel] >= threshold_mm_s) & (hd[sel] > 0)
        out[name] = dict(edges=int(sel.sum()), still=int((~moving).sum()),
                         perfused_length_share=float(length[moving].sum() / length.sum()))
    return out


# ----------------------------------------------------------------------------------------------- the view

@dataclass
class Pose:
    """Where the camera looks: the field's centre at s_mm from the limbus and clock_deg, plus the scene's
    magnification (um_per_px), field size, focus and tilt, which live in the Scene."""
    s_mm: float = 3.0
    clock_deg: float = 90.0


def assemble_view(eye: Eye, pose: Pose, scene: Scene, fine=True, roll_deg=0.0):
    """The network of the camera's field in the camera's own coordinates (x, y along the surface about the field's
    centre, depth). Every level of detail is a selection of the one network of the eye, solved once, so every piece
    carries exactly the flow of the eye's edge it came from at every magnification. fine: draw the fine trees and the
    capillary plexus too (else only the coarse level). roll_deg turns the camera's image about the field's centre
    (0: image x along the limbus, towards larger clock angles). Returns (network, flow)."""
    g = eye.globe
    s0 = pose.s_mm * 1000.0
    frame = g.frame(s0, pose.clock_deg)
    if roll_deg:
        c, e1, e2 = frame
        r = math.radians(roll_deg)
        frame = (c, math.cos(r) * e1 + math.sin(r) * e2, -math.sin(r) * e1 + math.cos(r) * e2)
    um = scene.anatomy.um_per_px
    half_w = scene.width_px * um / 2 + scene.margin_um
    half_h = scene.height_px * um / 2 + scene.margin_um
    domain = (-half_w, half_w, -half_h, half_h)
    b = _Builder()
    index = []                         # per view edge: the eye's edge it is a piece of
    anchors = []                       # per view node: the eye's two nodes it lies between, and how far along
    boundary = {}
    node_map = {}
    full = eye.network
    flow = eye.flow
    centres, radii, levels = eye.edge_bounds()
    reach = math.hypot(half_w, half_h)
    near = np.linalg.norm(centres - frame[0] * g.R, axis=1) < radii + reach + 50.0
    if not fine:
        near &= levels == 0

    def share(ei, k):
        seg = np.linalg.norm(np.diff(full.edges[ei].points, axis=0), axis=1)
        s = np.r_[0, np.cumsum(seg)]
        return float(s[k] / max(s[-1], 1e-9))

    for ei in np.flatnonzero(near):
        e = full.edges[ei]
        local = g.to_local(e.points, frame)
        keep = (local[:, 0] > domain[0]) & (local[:, 0] < domain[1]) & (local[:, 1] > domain[2]) & (local[:, 1] < domain[3])
        if not keep.any():
            continue
        runs = np.split(np.arange(len(keep)), np.flatnonzero(np.diff(keep.astype(int))) + 1)
        for run in runs:
            if not keep[run[0]] or len(run) < 2:
                continue
            ends = []
            for end, node in ((run[0], e.u if run[0] == 0 else None), (run[-1], e.v if run[-1] == len(local) - 1 else None)):
                if node is not None:
                    if node not in node_map:
                        node_map[node] = b.node(local[end])
                        anchors.append((node, node, 0.0))
                        if node in full.boundary:
                            boundary[node_map[node]] = full.boundary[node]
                    ends.append(node_map[node])
                else:
                    nn = b.node(local[end])
                    boundary[nn] = "arterial" if "arter" in e.kind else ("venous" if "vein" in e.kind or "venule" in e.kind else "capillary")
                    anchors.append((e.u, e.v, share(ei, end)))
                    ends.append(nn)
            b.edge(ends[0], ends[1], local[run].copy(), e.diameter, e.kind, e.layer, e.vessel)
            b.edges[-1].width = e.width[run] if e.width is not None and len(e.width) == len(local) else None
            b.edges[-1].paired_with = e.paired_with
            index.append(int(ei))
    # where only part of a node's vessels are drawn (the coarse level alone), the node is where blood leaves or joins
    # through vessels the view leaves out; the cells handle that (synth.cells keep_share)
    net = Network(np.asarray(b.nodes) if b.nodes else np.zeros((0, 3)), b.edges, boundary, domain,
                  (-scene.width_px * um / 2, scene.width_px * um / 2, -scene.height_px * um / 2, scene.height_px * um / 2),
                  dict(b.vessel_kind))
    net.origins = [("eye", ei) for ei in index]                # where every view edge came from
    view_flow = _inherit_flow(net, flow, np.array(index, int), anchors, scene)
    net.boundary_pressure = {node: float(view_flow.pressure[node]) for node in boundary}
    return net, view_flow


def _inherit_flow(net, source, index, anchors, scene):
    """The view's Flow: every piece takes the flow, haematocrit, velocity and pulse of the eye's edge it came from
    (index); every node takes the pressure and its pulse of the eye's node it is, or, at a cut end, interpolated
    along the cut edge (anchors). The volume the distending vessels store stays with the eye's own nodes: a cut end
    stores none."""
    a = np.array([k[0] for k in anchors], int)
    c = np.array([k[1] for k in anchors], int)
    f = np.array([k[2] for k in anchors], float)
    pressure = (1 - f) * source.pressure[a] + f * source.pressure[c]
    pulse_pressure = (1 - f) * source.pulse_pressure[:, a] + f * source.pulse_pressure[:, c]
    compliance = np.where(a == c, source.compliance[a], 0.0)
    fields = [getattr(source, k)[index] for k in ("flow", "hd", "ht", "viscosity", "velocity", "rbc_velocity",
                                                  "shear_rate")]
    length = np.array([e.length for e in net.edges])
    diameter = np.array([e.diameter for e in net.edges])
    return Flow(pressure, *fields, length, diameter, 0, scene.physiology, source.pulse[:, index], pulse_pressure,
                compliance)


def view_scene(eye: Eye, pose: Pose, scene: Scene | None = None) -> Scene:
    """The camera scene for a view: the eye's settings with the given field size and magnification."""
    scene = scene or eye.scene
    return replace(scene, optics=replace(scene.optics, eye_radius_um=eye.anatomy.radius_um), seed=eye.seed)


def summary(eye: Eye) -> dict:
    """Counts, lengths and flows of the coarse level by kind."""
    out = {}
    for k in sorted({e.kind for e in eye.network.edges}):
        sel = [i for i, e in enumerate(eye.network.edges) if e.kind == k]
        out[k] = dict(edges=len(sel), length_mm=round(sum(eye.network.edges[i].length for i in sel) * 1e-3, 1),
                      median_speed_mm_s=round(float(np.median(np.abs(eye.flow.rbc_velocity[sel]))), 3),
                      median_flow_pl_s=round(float(np.median(np.abs(eye.flow.flow[sel]))), 1))
    return out


def simulate_view(eye: Eye, pose: Pose, n_frames=60, scene: Scene | None = None, fine=None, **kw):
    """A burst of one region of the eye: the view's network with inherited flows, its cells (or tubes at low
    magnification), rendered by the camera at that pose. fine: include the tiles (default: below 8 µm/px)."""
    from .simulate import render_network
    import time as _time

    scene = view_scene(eye, pose, scene)
    if fine is None:
        fine = scene.anatomy.um_per_px < 8.0
    start = _time.perf_counter()
    net, flow = assemble_view(eye, pose, scene, fine=fine)
    timings = {"view assembly": _time.perf_counter() - start}
    return render_network(scene, net, flow, n_frames, timings=timings, **kw)


def overview(eye: Eye, um_per_px=20.0, size_px=(1300, 1000)):
    """The whole visible anterior eye at low magnification, seen from in front: the coarse network as tubes of mean
    red-cell density, one frame. Returns (frame uint16, extent in mm (left, right, bottom, top), outlines dict with
    the limbus and the lid margins in mm)."""
    from .simulate import render_network
    a = eye.anatomy
    apex = -eye.globe.theta_limbus * eye.globe.R / 1000.0           # s of the corneal apex, mm
    scene = replace(eye.scene, anatomy=replace(eye.scene.anatomy, um_per_px=um_per_px),
                    optics=replace(eye.scene.optics, glints=0, motion_step_px=0.0, saccade_rate_hz=0.0,
                                   focus_depth_um=eye.globe.R * (1 - math.cos(0.6)), focus_wander_um=0.0),
                    width_px=size_px[0], height_px=size_px[1], margin_um=500.0)
    pose = Pose(s_mm=apex, clock_deg=0.0)
    net, flow = assemble_view(eye, pose, scene, fine=False)
    result = render_network(scene, net, flow, n_frames=1, record_cells=False, record_masks=False, tubes=True)
    w, h = size_px[0] * um_per_px / 2000.0, size_px[1] * um_per_px / 2000.0
    t = np.linspace(0, 2 * np.pi, 361)
    xs = np.linspace(-a.fissure_half_width_mm, a.fissure_half_width_mm, 200)
    shape = np.sqrt(np.clip(1 - (xs / a.fissure_half_width_mm) ** 2, 0, None))
    outlines = {"limbus": (a.limbus_radius_um * 1e-3 * np.cos(t), a.limbus_radius_um * 1e-3 * np.sin(t)),
                "upper lid": (xs, a.upper_lid_mm * shape), "lower lid": (xs, a.lower_lid_mm * shape)}
    return result.frames[0], (-w, w, -h, h), outlines
