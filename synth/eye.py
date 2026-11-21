"""The whole anterior eye, built from where the blood comes from and where it goes, with levels of detail.

    from synth.eye import Eye, Pose, simulate_view
    eye = Eye(seed=4)                                   # the coarse network of the whole globe, solved once
    result = simulate_view(eye, Pose(s_mm=3.0, clock_deg=90), n_frames=60)   # a burst of one region

Levels of detail (docs/synthetic_conjunctiva_anatomy.md §11):

- Coarse (the whole globe, generated and solved once per eye): the anterior ciliary arteries leaving the four rectus
  tendons at their insertions (the spiral of Tillaux), two per muscle and one on the lateral rectus; the episcleral
  arterial circle they form near the limbus; the anterior conjunctival arteries it gives off towards the limbal
  arcades and back into the bulbar conjunctiva; the posterior conjunctival arteries entering from the fornices (the
  palpebral arcades), meeting the anterior ones at the watershed; companion conjunctival veins draining to the
  fornices; the limbal venous ring and the episcleral veins draining away from the limbus; episcleral arteriovenous
  anastomoses. The fine vessels each trunk feeds are represented at this level by lumped beds (one resistance every
  0.5 mm between a trunk and its companion vein), so the coarse solve sees the load they put on the trunks. Pressures
  are set at the true inlets (the anterior ciliary arteries, the palpebral arcades) and outlets (the episcleral and
  palpebral veins at the episcleral venous pressure), and every flow and velocity of the coarse level follows.
- Fine (per tile of about 1 mm, generated on demand): the conjunctival arteriolar and venular trees, the capillary
  mesh and their anastomoses (synth.network.generate_fine), grown from roots on the coarse trunks that cross the
  tile, from a seed that depends only on the eye's seed and the tile's position. A tile's flows are solved with the
  coarse pressures at its roots, so a tile is the same whenever and from wherever it is generated.
- The view (the camera's field only): the coarse pieces and the tiles that fall in it, assembled into one network
  in the camera's coordinates with the flows inherited from their own level - so zooming from an overview to high
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
from scipy.spatial import cKDTree

from .anatomy import Scene
from .haemodynamics import Flow, solve
from .network import _prune_terminals, width_profile, wiggle, Edge, Network, _Builder, _smooth_noise, generate_fine, resample

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
    coarse_radial_bias: float = 0.5                  # conjunctival vessels converge radially on the limbus [Akagi2018]
    coarse_branch_spacing_um: float = 600.0          # least distance between branch points of a coarse tree (as at
    coarse_max_terminal_um: float = 1500.0           # the patch level, docs §5, at trunk scale)
    conjunctival_artery_diameter: tuple = (25.0, 40.0)   # within the 5.9-61 µm of conjunctival arterioles [Khansari2017]
    watershed_mm: float = 4.0                        # lid branches supply the conjunctiva from 4 mm on [Sun2022]
    fornix_mm: dict = field(default_factory=lambda: {"superior": 15.0, "temporal": 12.0, "inferior": 10.5,
                                                     "nasal": 8.0})   # fornix depths 15.3-16.1 / 10.1-10.9 mm [Jutley2016, Khan2014]; horizontal assumed
    episcleral_veins_per_mm: float = 0.6             # radial episcleral veins per mm of limbus (fitted, docs §5)
    episcleral_vein_diameter: tuple = (50.0, 100.0)  # [Asanad2021, Kim2024]
    episcleral_heading_spread: float = 0.5          # rad about the meridian (fitted, as at the patch level, docs §5)
    episcleral_curvature: float = 0.0004             # rad/µm, gentle bends over millimetres (docs §5)
    episcleral_avas_per_mm2: float = 0.75            # arteriovenous anastomoses, 0.5-1 per mm2 [Selbach2005]
    episcleral_depth_um: float = 350.0               # docs §2.2
    conjunctival_trunk_depth_um: float = 160.0       # docs §2.2
    # pressures at the true inlets and outlets, mmHg (docs §11.6)
    ciliary_artery_pressure: float = 65.0            # ophthalmic artery ~0.7 x brachial [StraussKedra1987, daSilvaKrieglstein1980]
    palpebral_artery_pressure: float = 60.0          # lid arcades, a little further down the ophthalmic tree (assumed)
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

    def exposed(self, xyz):
        """Whether points lie in the palpebral fissure (not under the lids), seen from in front."""
        a = self.a
        x, y = xyz[..., 0] * 1e-3, xyz[..., 1] * 1e-3
        half = a.fissure_half_width_mm
        shape = np.clip(1 - (x / half) ** 2, 0, None)
        return (np.abs(x) <= half) & (y <= a.upper_lid_mm * shape ** 0.5) & (y >= a.lower_lid_mm * shape ** 0.5) & (xyz[..., 2] > 0)


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
    heading_offset for a side branch) and turning back towards the meridian with a slow random bend (a
    mean-reverting curvature), until it reaches s_end; returned in eye coordinates. Episcleral veins run radially
    from the limbus [Meyer1988], so none spirals round the globe."""
    heading = rng.normal(0.0, heading_spread) + heading_offset          # 0 is straight away from the limbus
    s, c = float(s0), float(clock0)
    pts = []
    n_max = int(3 * abs(s_end - s0) / step) + 10
    curvature = curvature_sd * _smooth_noise(rng, n_max, 250.0 / step)
    target = rng.normal(0.0, 0.12)
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
    """One synthetic eye: its anatomy, the scene settings it shares with its views, and its coarse network."""
    seed: int = 0
    anatomy: EyeAnatomy = field(default_factory=EyeAnatomy)
    scene: Scene = field(default_factory=Scene)

    def __post_init__(self):
        self.globe = Globe(self.anatomy)
        self.scene = replace(self.scene, optics=replace(self.scene.optics, eye_radius_um=self.anatomy.radius_um))
        self.network, self.trunks = build_coarse(self)
        self.flow = solve(self.network, self.scene.physiology)
        self._tiles = {}

    # the fine level
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

    def tile(self, key):
        """The fine vessels of a tile (cached): network in eye coordinates, flow, and its root attachments."""
        if key not in self._tiles:
            self._tiles[key] = build_tile(self, key)
        return self._tiles[key]


def build_coarse(eye: Eye):
    """The coarse network of the whole globe. Returns (network in eye coordinates, trunk registry)."""
    a, g = eye.anatomy, eye.globe
    A = eye.scene.anatomy                            # the patch level's shape parameters
    rng = np.random.default_rng([eye.seed, 11])
    b = _Builder()
    pressure = {}
    trunks = []
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
                                       a.coarse_growth_influence_um, a.coarse_growth_kill_um, persistence=0.85,
                                       radial_bias=a.coarse_radial_bias, branch_spacing=a.coarse_branch_spacing_um)
    positions, parent, tree = _prune_terminals(positions, parent, tree, a.coarse_max_terminal_um)
    paths, children = _tree_paths(parent)
    leaf_d = rng.uniform(*a.coarse_leaf_diameter, len(parent))
    d = np.minimum(_murray_diameters(parent, children, leaf_d, eye.scene.anatomy.murray_exponent_arterial), 58.0)
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
        edge(np.vstack([b.nodes[root], b.nodes[node]]), d0, "communicating artery", "communicating", u=root, v=node)
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
    art_pts = np.vstack([to_plane(*g.surface_coordinates(b.edges[i].points)[:2]) for i in artery_edges])
    pick = rng.choice(len(art_pts), min(len(art_pts) // 3, len(attractors)), replace=False)
    beside = art_pts[pick] + rng.normal(0, 1, (len(pick), 2)) * 60.0
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
                                 a.coarse_growth_influence_um, a.coarse_growth_kill_um, persistence=0.85,
                                 radial_bias=a.coarse_radial_bias, branch_spacing=a.coarse_branch_spacing_um)
    vpos, vpar, vtree = _prune_terminals(vpos, vpar, vtree, a.coarse_max_terminal_um)
    vpaths, vchildren = _tree_paths(vpar)
    vleaf = rng.uniform(*a.coarse_leaf_diameter, len(vpar)) * 1.3
    vd = np.minimum(_murray_diameters(vpar, vchildren, vleaf, eye.scene.anatomy.murray_exponent_venous), 72.0)
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
        d = rng.uniform(*a.episcleral_vein_diameter)
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
    # resolve the beds into real nodes (split the artery and vein edges at the bed points)
    beds = [e for e in b.edges if e.kind == "bed"]
    b.edges = [e for e in b.edges if e.kind != "bed"]
    _attach_beds(b, beds, a)
    # the calibre varies and tapers along every vessel, as at the patch level (docs §5)
    for e in b.edges:
        if e.kind in ("conjunctival artery", "conjunctival vein", "episcleral vein", "episcleral artery",
                      "ciliary artery", "limbal vein") and len(e.points) >= 4:
            seg = np.linalg.norm(np.diff(e.points, axis=0), axis=1)
            e.width = width_profile(rng, np.r_[0.0, np.cumsum(seg)], A, step=max(float(np.median(seg)), 1.0))
    net = Network(np.asarray(b.nodes), b.edges, dict(b.boundary), (0, 0, 0, 0), (0, 0, 0, 0), dict(b.vessel_kind))
    net.boundary_pressure = {k: v for k, v in pressure.items()}
    return net, trunks


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

def _tile_seed(eye, key):
    digest = hashlib.sha256(f"{eye.seed}:{key[0]}:{key[1]}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def build_tile(eye: Eye, key):
    """The fine vessels of one tile, in eye coordinates, with their flow solved from the coarse pressures at their
    roots. Returns dict(network, flow, roots: [(coarse edge, point index, node)], frame, domain)."""
    a, g = eye.anatomy, eye.globe
    s_c, clock_c = eye.tile_centre(key)
    frame = g.frame(s_c, clock_c)
    t = a.tile_mm * 1000.0
    n = eye.tiles_around(key[0])
    width = 2 * math.pi * g.R * math.sin(float(g.theta(s_c))) / n
    domain = (-width / 2, width / 2, -t / 2, t / 2)
    coarse = eye.network
    rng = np.random.default_rng(_tile_seed(eye, key))
    roots, attach = [], []
    spacing = 1000.0 / max(eye.scene.anatomy.arterial_trees_per_mm2 * 4.0, 1e-3)    # roots per mm of trunk
    for ei, e in enumerate(coarse.edges):
        if e.kind not in ("conjunctival artery", "conjunctival vein"):
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
            if k <= 0 or k >= len(e.points) - 1:
                continue
            tangent = local[min(k + 3, len(local) - 1), :2] - local[max(k - 3, 0), :2]
            heading = math.atan2(tangent[1], tangent[0]) + rng.choice([-1, 1]) * math.radians(rng.uniform(35, 60))
            roots.append(dict(venous=e.kind == "conjunctival vein", top=local[k].copy(), heading=heading))
            attach.append((ei, k))
    scene = eye.scene
    net, root_nodes = generate_fine(scene, domain, roots, _tile_seed(eye, key))
    # keep the roots whose trees grew; they are the tile's boundary, at the coarse pressure there
    boundary, bpressure, root_list = {}, {}, []
    for (ei, k), node, r in zip(attach, root_nodes, roots):
        if node is None:
            continue
        boundary[node] = "venous" if r["venous"] else "arterial"
        bpressure[node] = coarse_pressure_at(eye, ei, k)
        root_list.append((ei, k, node))
    net.boundary = boundary
    net.boundary_pressure = bpressure
    net = _drop_unrooted(net)
    root_list = [(ei, k, node) for (ei, k, node) in root_list if node in net.boundary]
    flow = solve(net, scene.physiology) if net.edges else None
    # into eye coordinates
    for e in net.edges:
        e.points = g.from_local(e.points, frame)
    net.nodes = g.from_local(net.nodes, frame) if len(net.nodes) else net.nodes
    return dict(network=net, flow=flow, roots=root_list, frame=frame, domain=domain, key=key)


def _drop_unrooted(net):
    """Remove fine components that reach no root (they could carry no flow); renumber the boundary maps."""
    n = len(net.nodes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for e in net.edges:
        parent[find(e.u)] = find(e.v)
    rooted = {find(k) for k in net.boundary}
    edges = [e for e in net.edges if find(e.u) in rooted and e.length > 1.0 and e.u != e.v]
    used = sorted({k for e in edges for k in (e.u, e.v)})
    remap = {old: new for new, old in enumerate(used)}
    for e in edges:
        e.u, e.v = remap[e.u], remap[e.v]
    out = Network(net.nodes[used] if used else np.zeros((0, 3)), edges,
                  {remap[k]: r for k, r in net.boundary.items() if k in remap}, net.domain, net.field_of_view,
                  net.vessel_names)
    out.boundary_pressure = {remap[k]: p for k, p in net.boundary_pressure.items() if k in remap}
    return out


def coarse_pressure_at(eye, edge_index, point_index):
    """The coarse solution's pressure at a point of a coarse edge (linear in arclength between its nodes)."""
    e = eye.network.edges[edge_index]
    seg = np.linalg.norm(np.diff(e.points, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    f = s[point_index] / max(s[-1], 1e-9)
    p = eye.flow.pressure
    return float((1 - f) * p[e.u] + f * p[e.v])


# ----------------------------------------------------------------------------------------------- the view

@dataclass
class Pose:
    """Where the camera looks: the field's centre at s_mm from the limbus and clock_deg, plus the scene's
    magnification (um_per_px), field size, focus and tilt, which live in the Scene."""
    s_mm: float = 3.0
    clock_deg: float = 90.0


def assemble_view(eye: Eye, pose: Pose, scene: Scene, fine=True):
    """The network of the camera's field in the camera's own coordinates (x, y along the surface about the field's
    centre, depth), with the flow of every piece inherited from its own level. Returns (network, flow)."""
    g = eye.globe
    s0 = pose.s_mm * 1000.0
    frame = g.frame(s0, pose.clock_deg)
    um = scene.anatomy.um_per_px
    half_w = scene.width_px * um / 2 + scene.margin_um
    half_h = scene.height_px * um / 2 + scene.margin_um
    domain = (-half_w, half_w, -half_h, half_h)
    b = _Builder()
    props = []                         # per view edge: (source flow object, source edge index)
    boundary, bpressure = {}, {}
    node_map = {}

    def inside(local):
        return (local[:, 0] > domain[0]) & (local[:, 0] < domain[1]) & (local[:, 1] > domain[2]) & (local[:, 1] < domain[3])

    def add_pieces(net, flow, source, ei, local, pressure_at):
        e = net.edges[ei]
        keep = inside(local)
        if not keep.any():
            return
        runs = np.split(np.arange(len(keep)), np.flatnonzero(np.diff(keep.astype(int))) + 1)
        for run in runs:
            if not keep[run[0]] or len(run) < 2:
                continue
            ends = []
            for end, node in ((run[0], e.u if run[0] == 0 else None), (run[-1], e.v if run[-1] == len(local) - 1 else None)):
                if node is not None:
                    keyn = (source, node)
                    if keyn not in node_map:
                        node_map[keyn] = b.node(local[end])
                    ends.append(node_map[keyn])
                else:
                    nn = b.node(local[end])
                    boundary[nn] = "arterial" if "arter" in e.kind else ("venous" if "vein" in e.kind or "venule" in e.kind else "capillary")
                    bpressure[nn] = pressure_at(ei, end)
                    ends.append(nn)
            b.edge(ends[0], ends[1], local[run].copy(), e.diameter, e.kind, e.layer, e.vessel)
            b.edges[-1].width = e.width[run] if e.width is not None and len(e.width) == len(local) else None
            props.append((flow, ei, source))

    coarse = eye.network
    for ei, e in enumerate(coarse.edges):
        if e.kind == "bed":
            continue
        add_pieces(coarse, eye.flow, "coarse", ei, g.to_local(e.points, frame),
                   lambda i, k: coarse_pressure_at(eye, i, k))
    if fine:
        for key in tiles_in_view(eye, frame, domain):
            tile = eye.tile(key)
            net, flow = tile["network"], tile["flow"]
            if flow is None:
                continue
            for ei, e in enumerate(net.edges):
                local = g.to_local(e.points, frame)

                def tile_pressure(i, k, net=net, flow=flow):
                    ee = net.edges[i]
                    seg = np.linalg.norm(np.diff(ee.points, axis=0), axis=1)
                    s = np.r_[0, np.cumsum(seg)]
                    f = s[k] / max(s[-1], 1e-9)
                    return float((1 - f) * flow.pressure[ee.u] + f * flow.pressure[ee.v])
                add_pieces(net, flow, key, ei, local, tile_pressure)
            # a root sits on a coarse trunk: the trunk and the tree share that node in the view
            for ci, k, node in tile["roots"]:
                keyn = (key, node)
                if keyn in node_map:
                    view_node = node_map[keyn]
                    boundary.pop(view_node, None)
                    bpressure.pop(view_node, None)
                    _join_to_coarse(b, props, eye, ci, k, view_node, g, frame)
    net = Network(np.asarray(b.nodes) if b.nodes else np.zeros((0, 3)), b.edges, boundary, domain,
                  (-scene.width_px * um / 2, scene.width_px * um / 2, -scene.height_px * um / 2, scene.height_px * um / 2),
                  dict(b.vessel_kind))
    net.boundary_pressure = bpressure
    net.origins = [(source, ei) for (_, ei, source) in props[:len(net.edges)]]   # where every view edge came from
    flow = _inherit_flow(net, props, scene)
    return net, flow


def _join_to_coarse(b, props, eye, coarse_edge, point_index, root_node, g, frame):
    """Move a fine tree's root onto the view piece of its coarse trunk: split the piece at the nearest point and
    merge the root into that node."""
    target = eye.network.edges[coarse_edge].points[point_index]
    local = g.to_local(target[None], frame)[0]
    best = None
    for vi, (flow, ei, source) in enumerate(props):
        if source != "coarse" or ei != coarse_edge:
            continue
        pts = b.edges[vi].points
        d = np.linalg.norm(pts - local, axis=1)
        k = int(np.argmin(d))
        if best is None or d[k] < best[0]:
            best = (d[k], vi, k)
    if best is None or best[0] > 5.0:
        return
    _, vi, k = best
    e = b.edges[vi]
    if k == 0:
        node = e.u
    elif k == len(e.points) - 1:
        node = e.v
    else:
        node = b.split(vi, k)
        props.append(props[vi])
        if e.width is not None:
            b.edges[-1].width, e.width = e.width[k:], e.width[:k + 1]
    for edge in b.edges:
        if edge.u == root_node:
            edge.u = node
            edge.points[0] = b.nodes[node]
        if edge.v == root_node:
            edge.v = node
            edge.points[-1] = b.nodes[node]


def tiles_in_view(eye, frame, domain):
    """The keys of the tiles whose area meets the view's domain."""
    g = eye.globe
    xs = np.linspace(domain[0], domain[1], 9)
    ys = np.linspace(domain[2], domain[3], 9)
    grid = np.array([[x, y, 0.0] for x in xs for y in ys])
    s, clock, _ = g.surface_coordinates(g.from_local(grid, frame))
    keys = set()
    for si, ci in zip(s, clock):
        if si <= 0:
            continue
        i, j = eye.tile_key(si, ci)
        n = eye.tiles_around(i)
        for di in (-1, 0, 1):
            ii = i + di
            if ii < 0:
                continue
            nn = eye.tiles_around(ii)
            jj = int(round(j * nn / n))
            for dj in (-1, 0, 1):
                keys.add((ii, (jj + dj) % nn))
    return sorted(keys)


def _inherit_flow(net, props, scene):
    """The view's Flow: every piece takes the flow, haematocrit and velocity of the edge it came from; node
    pressures are interpolated likewise."""
    E = len(net.edges)
    fields = {k: np.zeros(E) for k in ("flow", "flow_arterial", "flow_rest", "hd", "ht", "viscosity", "velocity",
                                        "rbc_velocity", "shear_rate")}
    for vi, (flow, ei, _) in enumerate(props[:E]):
        for k in fields:
            fields[k][vi] = getattr(flow, k)[ei]
    length = np.array([e.length for e in net.edges])
    diameter = np.array([e.diameter for e in net.edges])
    pressure = np.zeros(len(net.nodes))
    for node, p in net.boundary_pressure.items():
        pressure[node] = p
    return Flow(pressure, fields["flow"], fields["flow_arterial"], fields["flow_rest"], fields["hd"], fields["ht"],
                fields["viscosity"], fields["velocity"], fields["rbc_velocity"], fields["shear_rate"], length, diameter,
                0, scene.physiology)


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
    timings = {"view assembly": _time.perf_counter() - start, "tiles cached": len(eye._tiles)}
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
