"""An anatomically constrained 3D microvessel network of the bulbar conjunctiva, episclera and sclera.

Coordinates are micrometres: x along the limbus, y radially away from it, z depth below the epithelial surface
(positive down). The network is generated in a domain larger than the camera's field by Scene.margin_um, so the
vessels at the image border continue naturally and eye motion never shows an edge; where a vessel leaves the
domain its end is a boundary node with a pressure set by its role and calibre (synth.haemodynamics).

The layers, from the surface down (docs/synthetic_conjunctiva_anatomy.md §2):

1. The conjunctival plexus, in the substantia propria. Arteriolar trees and venular trees are grown by space
   colonisation [Runions2007] from roots at the domain border (anterior and posterior conjunctival vessels
   entering the field) and from communicating vessels that rise from the episcleral plexus. A share of the
   venular attractors lies beside the arterioles, so venules accompany arterioles as they do in vivo. Diameters
   follow Murray's law from the terminal arterioles and post-capillary venules upwards. The arteriolar and
   venular terminals meet in a polygonal capillary mesh (a Voronoi tessellation of a jittered lattice, whose
   vertices have degree three like real capillary junctions), just below the epithelium. Arteriolar-arteriolar
   and venular-venular anastomoses close loops between trees.
2. The episcleral plexus: long, straight, mostly radial arteries (anterior ciliary branches, running towards
   the limbus) and veins (draining away from it), with side branches.
3. The deep (intrascleral) plexus: a few deep, straight vessels.

Communicating vessels join the conjunctival roots to episcleral vessels, and perforating vessels join episcleral
vessels to the deep plexus, each running obliquely at an angle to the scleral surface drawn from the cited range.

Every edge carries its 3D centreline (sampled every few µm), its diameter, its kind (arteriole, venule,
capillary, episcleral artery or vein, deep, communicating, perforating) and its layer, and belongs to a named
vessel: a vessel runs on through a branch point along its larger daughter, as anatomists name vessels.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import Voronoi, cKDTree

from .anatomy import Scene

ARTERIAL_KINDS = ("arteriole", "episcleral artery", "perforating artery", "communicating artery", "deep artery")
VENOUS_KINDS = ("venule", "episcleral vein", "perforating vein", "communicating vein", "deep vein")


@dataclass
class Edge:
    u: int
    v: int
    points: np.ndarray            # (n, 3) µm, from node u to node v
    diameter: float               # µm
    kind: str
    layer: str
    vessel: int = -1
    width: np.ndarray | None = None   # (n,) the local diameter over the nominal one at each point (mean 1)
    paired_with: int = -1             # for a companion venule: the vessel id of the arteriole it accompanies

    @property
    def local_diameter(self):
        return self.diameter * (self.width if self.width is not None else np.ones(len(self.points)))

    @property
    def length(self):
        return float(np.linalg.norm(np.diff(self.points, axis=0), axis=1).sum())


@dataclass
class Network:
    nodes: np.ndarray                                  # (N, 3) µm
    edges: list
    boundary: dict = field(default_factory=dict)       # node -> role: 'arterial', 'venous' or 'capillary'
    domain: tuple = (0.0, 0.0, 0.0, 0.0)               # x0, x1, y0, y1 of the generation domain, µm
    field_of_view: tuple = (0.0, 0.0, 0.0, 0.0)        # x0, x1, y0, y1 of the camera's field (frame 0), µm
    vessel_names: dict = field(default_factory=dict)   # vessel id -> kind of its first edge
    boundary_offset: dict = field(default_factory=dict)  # node -> mmHg added to its role's boundary pressure
    boundary_pressure: dict = field(default_factory=dict)  # node -> mmHg, overriding the role's pressure (the eye)

    def degree(self):
        degree = np.zeros(len(self.nodes), int)
        for e in self.edges:
            degree[e.u] += 1
            degree[e.v] += 1
        return degree

    def incident(self):
        lists = [[] for _ in range(len(self.nodes))]
        for index, e in enumerate(self.edges):
            lists[e.u].append(index)
            lists[e.v].append(index)
        return lists


# ----------------------------------------------------------------------------------------------- geometry helpers

def _smooth_noise(rng, n, correlation_samples):
    """Zero-mean, unit-SD smooth noise of length n with a Gaussian correlation of the given samples."""
    if n <= 1:
        return np.zeros(n)
    pad = int(3 * correlation_samples) + 1
    raw = rng.standard_normal(n + 2 * pad)
    width = max(correlation_samples, 0.5)
    kernel = np.exp(-0.5 * (np.arange(-pad, pad + 1) / width) ** 2)
    smooth = np.convolve(raw, kernel / np.sqrt((kernel ** 2).sum()), mode="same")[pad:pad + n]
    return smooth


def resample(points, step):
    """Resample a polyline at a fixed arclength step, keeping both ends."""
    points = np.asarray(points, float)
    if len(points) < 2:
        return points
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(lengths)]
    if s[-1] == 0:
        return points[[0, -1]]
    n = max(2, int(math.ceil(s[-1] / step)) + 1)
    q = np.linspace(0, s[-1], n)
    return np.stack([np.interp(q, s, points[:, k]) for k in range(points.shape[1])], 1)


def _chaikin(points, rounds=2):
    """Corner-cutting smoothing that keeps the end points."""
    for _ in range(rounds):
        if len(points) < 3:
            return points
        a, b = points[:-1], points[1:]
        q = 0.75 * a + 0.25 * b
        r = 0.25 * a + 0.75 * b
        middle = np.empty((2 * len(a), points.shape[1]))
        middle[0::2], middle[1::2] = q, r
        points = np.vstack([points[:1], middle[1:-1], points[-1:]])
    return points


def wiggle(points, rng, amplitude, correlation, step=4.0):
    """Add a smooth lateral displacement in the xy plane, tapered to zero at both ends (junctions stay put)."""
    points = resample(points, step)
    n = len(points)
    if n < 5 or amplitude <= 0:
        return points
    tangent = np.gradient(points[:, :2], axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], 1)
    offset = amplitude * _smooth_noise(rng, n, correlation / step)
    s = np.linspace(0, 1, n)
    taper = np.minimum(1.0, np.minimum(s, 1 - s) * n * step / max(correlation, step))
    out = points.copy()
    out[:, :2] += (offset * taper)[:, None] * normal
    return out


def _segment(p, q, rng, amplitude, correlation, step=4.0):
    """A wiggly 3D segment from p to q: a lateral wiggle in xy, depth linear between the ends."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    flat = wiggle(np.vstack([p[:2], q[:2]]), rng, amplitude, correlation, step)
    flat = resample(flat, step)
    t = np.linspace(0, 1, len(flat))
    return np.column_stack([flat, p[2] + t * (q[2] - p[2])])


def _random_curve(rng, start, heading, length, curvature_sd, correlation, step=10.0, reversion=0.0, target=None):
    """A smooth 2D curve: heading follows a mean-reverting random curvature (rad/µm)."""
    n = max(2, int(length / step))
    curvature = curvature_sd * _smooth_noise(rng, n, correlation / step)
    points = [np.asarray(start, float)]
    h = heading
    for k in range(n):
        if target is not None:
            h += reversion * step * math.sin(target - h)
        h += curvature[k] * step
        points.append(points[-1] + step * np.array([math.cos(h), math.sin(h)]))
    return np.array(points)


# ----------------------------------------------------------------------------------------------- builder

class _Builder:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.boundary = {}
        self.next_vessel = 0
        self.vessel_kind = {}

    def node(self, xyz):
        self.nodes.append(np.asarray(xyz, float))
        return len(self.nodes) - 1

    def vessel(self, kind):
        self.next_vessel += 1
        self.vessel_kind[self.next_vessel] = kind
        return self.next_vessel

    def edge(self, u, v, points, diameter, kind, layer, vessel):
        points = np.asarray(points, float)
        points[0] = self.nodes[u]
        points[-1] = self.nodes[v]
        self.edges.append(Edge(u, v, points, float(diameter), kind, layer, vessel))
        return len(self.edges) - 1

    def split(self, edge_index, at_index):
        """Split an edge at one of its interior points; return the new node."""
        e = self.edges[edge_index]
        at_index = int(np.clip(at_index, 1, len(e.points) - 2))
        node = self.node(e.points[at_index])
        tail = Edge(node, e.v, e.points[at_index:].copy(), e.diameter, e.kind, e.layer, e.vessel,
                    paired_with=e.paired_with)
        e.points = e.points[:at_index + 1].copy()
        e.v = node
        self.edges.append(tail)
        return node


# ----------------------------------------------------------------------------------------------- space colonisation

def colonise(rng, roots, root_headings, attractors, step, influence, kill, persistence=0.5, max_iterations=3000,
             branch_angle=(20.0, 70.0), radial_bias=0.0):
    """Grow trees towards attractors (space colonisation [Runions2007]).

    A tip grows on towards the attractors nearest to it; a node that already has a child may sprout one side
    branch, whose first step makes an angle with the continuation within branch_angle (degrees), and never more
    than one. Candidates
    closer than half a step to an existing node are dropped, so trees neither double back nor overlap.
    radial_bias pulls every tree along the y axis in the direction its root heads (radial vessels).
    Returns (positions (n, 2), parent (n,), tree (n,))."""
    capacity = len(roots) + 20 * len(attractors) + 1000
    P = np.zeros((capacity, 2))
    H = np.zeros((capacity, 2))
    parent = np.full(capacity, -1)
    tree = np.zeros(capacity, int)
    n_children = np.zeros(capacity, int)
    first_child_dir = np.zeros((capacity, 2))
    n = len(roots)
    P[:n] = np.asarray(roots, float)
    H[:n] = np.column_stack([np.cos(root_headings), np.sin(root_headings)])
    tree[:n] = np.arange(n)
    radial = np.sign(H[:n, 1])
    alive = np.asarray(attractors, float)
    cos_low, cos_high = np.cos(np.radians(branch_angle[1])), np.cos(np.radians(branch_angle[0]))
    index_tree = cKDTree(P[:n])
    indexed = n
    for _ in range(max_iterations):
        if len(alive) == 0 or n + len(roots) * 4 >= capacity:
            break
        if n - indexed > 200:
            index_tree, indexed = cKDTree(P[:n]), n
        # nearest node of every attractor that can still grow (fewer than two children); the tree index is
        # refreshed lazily and the most recent nodes are checked directly
        k = min(8, indexed)
        dists, idxs = index_tree.query(alive, k=k, distance_upper_bound=influence)
        dists, idxs = dists.reshape(len(alive), k), idxs.reshape(len(alive), k)
        valid = np.isfinite(dists)
        eligible = np.zeros_like(valid)
        eligible[valid] = n_children[idxs[valid]] < 2
        dists = np.where(eligible, dists, np.inf)
        best = np.argmin(dists, axis=1)
        distance = dists[np.arange(len(alive)), best]
        nearest = np.where(np.isfinite(distance), idxs[np.arange(len(alive)), best], 0)
        if n > indexed:
            recent = P[indexed:n]
            d_recent = np.linalg.norm(alive[:, None, :] - recent[None], axis=2)
            d_recent = np.where((n_children[indexed:n] < 2)[None], d_recent, np.inf)
            j = np.argmin(d_recent, axis=1)
            dj = d_recent[np.arange(len(alive)), j]
            better = dj < distance
            distance = np.where(better, dj, distance)
            nearest = np.where(better, indexed + j, nearest)
        has = np.isfinite(distance) & (distance <= influence)
        if not has.any():
            break
        vectors = alive[has] - P[nearest[has]]
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)
        pull = np.zeros((n, 2))
        np.add.at(pull, nearest[has], vectors)
        growing = np.flatnonzero((np.linalg.norm(pull, axis=1) > 0) & (n_children[:n] < 2))
        added = []
        for index in growing:
            direction = pull[index] / np.linalg.norm(pull[index])
            if n_children[index] == 0:
                direction = (1 - persistence) * direction + persistence * H[index]
                direction = direction + radial_bias * np.array([0.0, radial[tree[index]]])
            else:
                # a side branch leaves at an acute angle to the parent's continuation: its first step is turned
                # into the allowed window on the side it is pulled towards, and it bends towards its attractors
                # as it grows on (the larger daughter makes the smaller angle with the parent [Zamir1978])
                cosine = float(direction @ first_child_dir[index])
                if not cos_low <= cosine <= cos_high:
                    base = first_child_dir[index]
                    side = np.sign(base[0] * direction[1] - base[1] * direction[0]) or 1.0
                    limit = branch_angle[1] if cosine < cos_low else branch_angle[0]
                    angle = np.radians(limit) * side
                    direction = np.array([base[0] * np.cos(angle) - base[1] * np.sin(angle),
                                          base[0] * np.sin(angle) + base[1] * np.cos(angle)])
            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                continue
            direction = direction / norm
            candidate = P[index] + step * direction
            near, _ = index_tree.query(candidate)
            if n > indexed:
                near = min(near, float(np.min(np.linalg.norm(P[indexed:n] - candidate, axis=1))))
            if near < 0.5 * step:
                continue
            P[n], H[n], parent[n], tree[n] = candidate, direction, index, tree[index]
            if n_children[index] == 0:
                first_child_dir[index] = direction
            n_children[index] += 1
            added.append(n)
            n += 1
        if not added:
            # every node that was pulled was blocked (too close to another node, or a side branch outside the
            # angle range): the attractors pulling them cannot be reached, so drop those and carry on
            blocked = np.isin(nearest, growing) & has
            if not blocked.any():
                break
            alive = alive[~blocked]
            continue
        d_kill, _ = cKDTree(P[added]).query(alive)
        alive = alive[d_kill > kill]
    return P[:n].copy(), parent[:n].copy(), tree[:n].copy()


def _tree_paths(parent):
    """Split a parent-pointer tree into paths between branch points, leaves and roots.

    Returns (paths: list of node-index lists from proximal to distal, children count)."""
    n = len(parent)
    children = [[] for _ in range(n)]
    for index, p in enumerate(parent):
        if p >= 0:
            children[p].append(index)
    paths = []
    stack = [index for index in range(n) if parent[index] < 0]
    while stack:
        start = stack.pop()
        for child in children[start]:
            path = [start, child]
            while len(children[path[-1]]) == 1:
                path.append(children[path[-1]][0])
            paths.append(path)
            if len(children[path[-1]]) > 1:
                stack.append(path[-1])
    return paths, children


def _murray_diameters(parent, children, leaf_diameter, exponent):
    """Diameter of the segment above every node, from the leaves up: d^g = sum of the children's d^g."""
    n = len(parent)
    order = np.argsort(-_depths(parent))           # deepest first
    d = np.zeros(n)
    for index in order:
        if not children[index]:
            d[index] = leaf_diameter[index]
        else:
            d[index] = float(np.sum(d[children[index]] ** exponent) ** (1.0 / exponent))
    return d


def _depths(parent):
    """Number of steps from every node to its root (parents are always created before their children)."""
    depth = np.zeros(len(parent), int)
    for index, p in enumerate(parent):
        if p >= 0:
            depth[index] = depth[p] + 1
    return depth


# ----------------------------------------------------------------------------------------------- the layers

def _conjunctival_depth(rng, diameter, scene):
    """Depth of a conjunctival vessel: capillaries just under the epithelium, larger vessels deeper in the
    substantia propria (docs §2.2)."""
    a = scene.anatomy
    top, bottom = a.conjunctival_plexus
    small = a.capillary_diameter[1]
    fraction = np.clip((diameter - small) / (60.0 - small), 0, 1) ** 0.7
    return top + (bottom - top) * (0.15 + 0.75 * fraction) + rng.normal(0, 8.0)


def _add_depth(points2d, z_start, z_end, rng, undulation=6.0, correlation=80.0, step=4.0):
    points2d = resample(points2d, step)
    n = len(points2d)
    z = np.linspace(z_start, z_end, n)
    if n > 4:
        s = np.linspace(0, 1, n)
        z = z + undulation * _smooth_noise(rng, n, correlation / step) * np.sin(np.pi * s)
    return np.column_stack([points2d, z])


def _inside(p, domain, pad=0.0):
    x0, x1, y0, y1 = domain
    return (x0 + pad <= p[0] <= x1 - pad) and (y0 + pad <= p[1] <= y1 - pad)


def _clip_to_domain(points, domain):
    """Keep the leading part of a 2D curve that lies inside the domain; the last point lands on the border."""
    inside = np.array([_inside(p, domain) for p in points])
    if inside.all():
        return points, False
    first_out = int(np.argmin(inside))
    if first_out == 0:
        return points[:1], True
    a, b = points[first_out - 1], points[first_out]
    x0, x1, y0, y1 = domain
    t = 1.0
    for axis, low, high in ((0, x0, x1), (1, y0, y1)):
        if b[axis] < low:
            t = min(t, (low - a[axis]) / (b[axis] - a[axis]))
        elif b[axis] > high:
            t = min(t, (high - a[axis]) / (b[axis] - a[axis]))
    return np.vstack([points[:first_out], a + t * (b - a)]), True


def _straight_layer(builder, rng, scene, domain, layer, count_per_mm, depth_range, artery_diameter, vein_diameter,
                    curvature_sd, branch_rate_per_mm):
    """Long, mostly radial vessels crossing the domain (episcleral or deep plexus), with side branches."""
    x0, x1, y0, y1 = domain
    width_mm = (x1 - x0) * 1e-3
    count = max(1, rng.poisson(count_per_mm * width_mm))
    made = []
    for index in range(count):
        venous = index % 2 == 1 if count > 1 else rng.random() < 0.5
        kind = f"{layer} vein" if venous else f"{layer} artery"
        diameter = rng.uniform(*(vein_diameter if venous else artery_diameter))
        # start on the posterior border (y1) or on a side, heading towards the limbus (decreasing y)
        start = np.array([rng.uniform(x0, x1), y1])
        heading = -math.pi / 2 + rng.normal(0, scene.anatomy.episcleral_heading_spread)
        curve = _random_curve(rng, start, heading, 3 * (y1 - y0), curvature_sd, 800.0, step=12.0,
                              reversion=1 / 900.0, target=heading)
        curve, _ = _clip_to_domain(curve, domain)
        if len(curve) < 3:
            continue
        z = rng.uniform(*depth_range)
        z_end = np.clip(z + rng.normal(0, 30), *depth_range)
        points = _add_depth(curve, z, z_end, rng, undulation=10.0, correlation=200.0, step=6.0)
        # flow: arteries run towards the limbus, so the posterior end is upstream; veins drain away from it
        u, v = builder.node(points[0]), builder.node(points[-1])
        vessel = builder.vessel(kind)
        builder.boundary[u] = "venous" if venous else "arterial"
        builder.boundary[v] = "venous" if venous else "arterial"
        made.append(builder.edge(u, v, points, diameter, kind, layer, vessel))
        # side branches leave at 30-70 degrees and run to the border
        n_branches = rng.poisson(branch_rate_per_mm * (len(curve) * 12e-3))
        for _ in range(n_branches):
            parent_index = max((i for i, ed in enumerate(builder.edges) if ed.vessel == vessel),
                               key=lambda i: builder.edges[i].length)
            e = builder.edges[parent_index]
            if len(e.points) < 20:
                continue
            at = int(rng.integers(5, len(e.points) - 5))
            tangent = e.points[min(at + 2, len(e.points) - 1), :2] - e.points[at - 2, :2]
            base = math.atan2(tangent[1], tangent[0])
            angle = base + rng.choice([-1, 1]) * math.radians(rng.uniform(20, 45))
            branch = _random_curve(rng, e.points[at, :2], angle, 1.5 * (y1 - y0), curvature_sd * 1.5, 300.0,
                                   step=10.0)
            branch, _ = _clip_to_domain(branch, domain)
            if len(branch) < 6:
                continue
            node = builder.split(parent_index, at)
            gamma = scene.anatomy.murray_exponent_venous if venous else scene.anatomy.murray_exponent_arterial
            ratio = rng.uniform(0.45, 0.8)
            d_branch = diameter * ratio / (1 + ratio ** gamma) ** (1 / gamma)
            # a side branch of a named episcleral or deep vessel is still a vessel of that class: not below its
            # cited calibre range
            d_branch = max(d_branch, (vein_diameter if venous else artery_diameter)[0])
            z_b = builder.nodes[node][2]
            bpoints = _add_depth(branch, z_b, np.clip(z_b + rng.normal(0, 25), *depth_range), rng,
                                 undulation=8.0, correlation=150.0, step=6.0)
            end = builder.node(bpoints[-1])
            builder.boundary[end] = "venous" if venous else "arterial"
            made.append(builder.edge(node, end, bpoints, d_branch, kind, layer, builder.vessel(kind)))
    return made


def _point_on_kind(builder, rng, kinds, region=None):
    """A random interior point of a random edge of the given kinds: (edge index, point index)."""
    candidates = [i for i, e in enumerate(builder.edges) if e.kind in kinds and len(e.points) > 12]
    if region is not None:
        candidates = [i for i in candidates if _inside(builder.edges[i].points[len(builder.edges[i].points) // 2], region)]
    if not candidates:
        return None
    weights = np.array([builder.edges[i].length for i in candidates])
    i = candidates[int(rng.choice(len(candidates), p=weights / weights.sum()))]
    return i, int(rng.integers(6, len(builder.edges[i].points) - 6))


def _oblique_path(rng, top, bottom, step=4.0):
    """A gently curved 3D path from a point in an upper layer down to a point in a lower one."""
    n = max(4, int(np.linalg.norm(bottom - top) / step))
    s = np.linspace(0, 1, n)
    path = top[None] + s[:, None] * (bottom - top)[None]
    bend = rng.normal(0, 0.08) * np.linalg.norm(bottom[:2] - top[:2])
    direction = bottom[:2] - top[:2]
    if np.linalg.norm(direction) > 0:
        normal = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
        path[:, :2] += (bend * np.sin(np.pi * s))[:, None] * normal
    return path


def _conjunctival_trees(builder, rng, scene, domain, communicating, border_roots=True):
    """Arteriolar and venular trees in the conjunctival plexus, the capillary mesh between them, anastomoses.

    communicating lists roots given from outside (dicts with venous, top (x, y, z), optionally heading); with
    border_roots False every tree grows from one of them (the eye's tiles root their fine trees on coarse trunks)."""
    a = scene.anatomy
    x0, x1, y0, y1 = domain
    area_mm2 = (x1 - x0) * (y1 - y0) * 1e-6
    top, bottom = a.conjunctival_plexus
    trees = {}
    for venous in (False, True):
        kind = "venule" if venous else "arteriole"
        per_mm2 = a.venous_trees_per_mm2 if venous else a.arterial_trees_per_mm2
        given = [c for c in communicating if c["venous"] == venous]
        n_roots = max(2, rng.poisson(per_mm2 * area_mm2)) if border_roots else len(given)
        roots, headings, sources = [], [], []
        # communicating roots: rise obliquely from an episcleral vessel of the same side of the circulation
        for root in given[:n_roots // 2] if border_roots else given:
            roots.append(np.asarray(root["top"][:2], float))
            headings.append(root.get("heading", rng.uniform(0, 2 * math.pi)))
            sources.append(root)
        # border roots: anterior conjunctival vessels enter from the limbal side, posterior ones from the fornix side
        while border_roots and len(roots) < n_roots:
            side = rng.random()
            if side < 0.4:
                p, h = (rng.uniform(x0, x1), y0), math.pi / 2
            elif side < 0.8:
                p, h = (rng.uniform(x0, x1), y1), -math.pi / 2
            elif side < 0.9:
                p, h = (x0, rng.uniform(y0, y1)), 0.0
            else:
                p, h = (x1, rng.uniform(y0, y1)), math.pi
            roots.append(np.array(p, float))
            headings.append(h + rng.normal(0, 0.3))
            sources.append(None)
        n_attractors = rng.poisson(a.attractors_per_mm2 * area_mm2)
        attractors = np.column_stack([rng.uniform(x0, x1, n_attractors), rng.uniform(y0, y1, n_attractors)])
        if venous and trees.get("arteriole") is not None:
            # venules accompany arterioles: part of the attractors lie beside the grown arterioles
            P = trees["arteriole"]["positions"]
            pick = rng.choice(len(P), int(a.av_pair_fraction * n_attractors), replace=True)
            par = trees["arteriole"]["parent"][pick]
            ok = par >= 0
            pick, par = pick[ok], par[ok]
            tangent = P[pick] - P[par]
            tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
            normal = np.stack([-tangent[:, 1], tangent[:, 0]], 1)
            offset = rng.uniform(*a.av_pair_offset, len(pick)) * rng.choice([-1, 1], len(pick))
            beside = P[pick] + normal * offset[:, None]
            keep = int((1 - a.av_pair_fraction) * n_attractors)
            attractors = np.vstack([attractors[:keep], beside])
            inside = (attractors[:, 0] > x0) & (attractors[:, 0] < x1) & (attractors[:, 1] > y0) & (attractors[:, 1] < y1)
            attractors = attractors[inside]
        kill = a.growth_kill * (a.venous_kill_ratio if venous else 1.0)
        if not roots:
            trees[kind] = None
            continue
        positions, parent, tree = colonise(rng, roots, headings, attractors, a.growth_step, a.growth_influence,
                                           kill, persistence=a.growth_persistence, radial_bias=a.radial_bias)
        trees[kind] = dict(positions=positions, parent=parent, tree=tree, roots=roots, sources=sources)
    made = {"arteriole": dict(leaves=[]), "venule": dict(leaves=[])}
    for kind, data in trees.items():
        if data is None:
            continue
        venous = kind == "venule"
        positions, parent, tree = data["positions"], data["parent"], data["tree"]
        paths, children = _tree_paths(parent)
        leaf_range = a.postcapillary_venule_diameter if venous else a.terminal_arteriole_diameter
        leaf_d = rng.uniform(*leaf_range, len(parent))
        gamma = a.murray_exponent_venous if venous else a.murray_exponent_arterial
        d = _murray_diameters(parent, children, leaf_d, gamma)
        # a tree rooted outside the field keeps growing there; inside it no conjunctival vessel exceeds the largest
        # measured (arterioles 61 µm, venules 75 µm [Khansari2017])
        d = np.minimum(d, a.max_venule_diameter if venous else a.max_arteriole_diameter)
        # nodes: tree roots, branch points and leaves become network nodes
        node_of = {}
        z_of = {}
        for path in paths:
            for end in (path[0], path[-1]):
                if end not in node_of:
                    z_of[end] = _conjunctival_depth(rng, d[end] if parent[end] >= 0 else d[path[1]], scene)
        # roots whose tree never grew (no attractor came within reach) are left out
        roots_index = [i for i in range(len(parent)) if parent[i] < 0 and children[i]]
        for index in roots_index:
            source = data["sources"][index] if index < len(data["sources"]) else None
            if source is not None:
                z_of[index] = source["top"][2]
        if not border_roots:
            # trees rooted on a trunk leave it gently: a node's depth differs from its root's by at most a quarter of
            # its distance from it, so no first segment plunges steeply (seen end-on it would show as a dark dot)
            for end in list(z_of):
                r = int(tree[end])
                source = data["sources"][r] if r < len(data["sources"]) else None
                if source is None or end == r:
                    continue
                z_root = float(source["top"][2])
                reach = 0.25 * float(np.hypot(*(positions[end] - positions[r])))
                z_of[end] = float(np.clip(z_of[end], z_root - reach, z_root + reach))
        for end, z in z_of.items():
            node_of[end] = builder.node((positions[end][0], positions[end][1], z))
        # vessel identity: a vessel continues into the child with the larger diameter
        vessel_at = {}
        leaves = []
        depth = _depths(parent)
        for path in sorted(paths, key=lambda p: depth[p[0]]):
            start = path[0]
            first = path[1]
            if parent[start] < 0 or start not in vessel_at:
                vessel_id = builder.vessel(kind)
            else:
                siblings = children[start]
                biggest = siblings[int(np.argmax(d[siblings]))]
                vessel_id = vessel_at[start] if first == biggest else builder.vessel(kind)
            vessel_at[path[-1]] = vessel_id
            pts2d = _chaikin(positions[path], rounds=4)
            # straighten towards the chord: growth bends a branch round towards its attractors in an arc, whereas
            # conjunctival vessels run in nearly straight lines between their branch points
            chord = pts2d[0] + np.linspace(0, 1, len(pts2d))[:, None] * (pts2d[-1] - pts2d[0])
            pts2d = chord + a.path_straightening * (pts2d - chord)
            amplitude, correlation = a.wiggle[kind]
            pts2d = wiggle(pts2d, rng, amplitude, correlation)
            points = _add_depth(pts2d[:, :2], z_of[start], z_of[path[-1]], rng, undulation=5.0)
            diameter = float(np.median(d[path[1:]]))
            builder.edge(node_of[start], node_of[path[-1]], points, diameter, kind, "conjunctival", vessel_id)
            if not children[path[-1]]:
                leaves.append(node_of[path[-1]])
        for index in roots_index:
            source = data["sources"][index] if index < len(data["sources"]) else None
            if source is None:
                builder.boundary[node_of[index]] = "venous" if venous else "arterial"
            else:
                source["conjunctival_node"] = node_of[index]
                source["diameter"] = float(d[index]) if children[index] else float(leaf_d[index])
        made[kind] = dict(leaves=leaves)
    return made


def _capillary_mesh(builder, rng, scene, domain, leaves):
    """A polygonal capillary mesh (Voronoi ridges of a jittered hexagonal lattice) joined to the terminals."""
    a = scene.anatomy
    x0, x1, y0, y1 = domain
    spacing = a.capillary_mesh_spacing
    rows = int((y1 - y0) / (spacing * 0.866)) + 3
    cols = int((x1 - x0) / spacing) + 3
    jy, jx = np.mgrid[0:rows, 0:cols].astype(float)
    sites = np.column_stack([(x0 - spacing + jx * spacing + 0.5 * spacing * (jy % 2)).ravel(),
                             (y0 - spacing + jy * spacing * 0.866).ravel()])
    sites += rng.normal(0, 0.3 * spacing, sites.shape)
    stretch = rng.uniform(1.0, 1.3)
    sites[:, 1] = y0 + (sites[:, 1] - y0) * stretch
    vor = Voronoi(sites)
    vertices = vor.vertices
    keep = np.array([_inside(v, domain) for v in vertices])
    ridges = [tuple(r) for r in vor.ridge_vertices if -1 not in r and keep[r[0]] and keep[r[1]]]
    ridges = [r for r in ridges if rng.random() > a.capillary_edge_drop]
    # prune dangling ridges repeatedly (a dead end carries no flow); the mesh is closed at the domain's edge, which
    # lies outside the camera's field, so capillary flow enters and leaves only through arterioles and venules
    while True:
        degree = np.zeros(len(vertices), int)
        for i, j in ridges:
            degree[i] += 1
            degree[j] += 1
        dangling = set(np.flatnonzero(degree == 1).tolist())
        if not dangling:
            break
        ridges = [r for r in ridges if r[0] not in dangling and r[1] not in dangling]
    used = sorted({k for r in ridges for k in r})
    if not used:
        return []
    top = scene.anatomy.conjunctival_plexus[0]
    z = top + np.abs(rng.normal(0, 12.0, len(vertices)))
    node_of = {k: builder.node((vertices[k][0], vertices[k][1], z[k])) for k in used}
    made = []
    for i, j in ridges:
        p, q = builder.nodes[node_of[i]], builder.nodes[node_of[j]]
        pts = _segment(p, q, rng, *a.wiggle["capillary"])
        made.append(builder.edge(node_of[i], node_of[j], pts, rng.uniform(*a.capillary_diameter), "capillary",
                                 "conjunctival", builder.vessel("capillary")))
    degree = np.zeros(len(vertices), int)
    for i, j in ridges:
        degree[i] += 1
        degree[j] += 1
    # the terminal arterioles and post-capillary venules each feed one or two mesh vertices
    mesh_nodes = np.array(used)
    tree = cKDTree(vertices[mesh_nodes])
    for leaf in leaves:
        p = builder.nodes[leaf]
        n_links = 1 + int(rng.random() < 0.35)
        distance, nearest = tree.query(p[:2], k=n_links)
        for dist, near in zip(np.atleast_1d(distance), np.atleast_1d(nearest)):
            if dist > 2.5 * spacing:
                continue
            target = node_of[mesh_nodes[near]]
            pts = _segment(p, builder.nodes[target], rng, *a.wiggle["capillary"])
            made.append(builder.edge(leaf, target, pts, rng.uniform(*a.capillary_diameter), "capillary",
                                     "conjunctival", builder.vessel("capillary")))
    return made


def _anastomoses(builder, rng, scene, domain, kind, count):
    """Links between two different vessels of the same kind (closing arcades), 60-350 µm long."""
    made = 0
    candidates = [i for i, e in enumerate(builder.edges) if e.kind == kind and len(e.points) > 16]
    tries = 0
    while made < count and tries < 40 * max(count, 1) and len(candidates) > 1:
        tries += 1
        i, j = rng.choice(candidates, 2, replace=False)
        ei, ej = builder.edges[i], builder.edges[j]
        if ei.vessel == ej.vessel:
            continue
        a_index = int(rng.integers(6, len(ei.points) - 6))
        tree = cKDTree(ej.points[6:-6, :2])
        distance, near = tree.query(ei.points[a_index, :2])
        if not 60.0 <= distance <= 350.0:
            continue
        b_index = int(near) + 6
        node_a = builder.split(i, a_index)
        # the split may have moved part of ej into a new edge if i == j's vessel; recompute from node positions
        node_b = builder.split(j, b_index)
        p, q = builder.nodes[node_a], builder.nodes[node_b]
        pts = _segment(p, q, rng, *scene.anatomy.wiggle[kind])
        diameter = rng.uniform(*(scene.anatomy.terminal_arteriole_diameter if kind == "arteriole"
                                 else scene.anatomy.postcapillary_venule_diameter))
        builder.edge(node_a, node_b, pts, diameter, kind, "conjunctival", builder.vessel(kind))
        candidates = [k for k, e in enumerate(builder.edges) if e.kind == kind and len(e.points) > 16]
        made += 1
    return made


def _bumps(rng, s, rate_per_um, amplitude, width):
    """A sum of Gaussian bumps of random sign at Poisson positions along arclength s: local kinks."""
    out = np.zeros_like(s)
    if len(s) < 3 or rate_per_um <= 0:
        return out
    n = rng.poisson(rate_per_um * s[-1])
    for centre in rng.uniform(0, s[-1], n):
        out += rng.choice([-1.0, 1.0]) * amplitude * rng.uniform(0.5, 1.5) * np.exp(-0.5 * ((s - centre) / width) ** 2)
    return out


def _roughen(network_edges, rng, scene):
    """Multi-scale irregularity on every segment, after the topology is final: a fine lateral jitter and sparse
    kinks on top of each vessel's smooth meander, and a smooth random variation of the calibre along its length.
    Both taper to nothing at the segment ends, so junctions stay where they are."""
    a = scene.anatomy
    jitter_amplitude, jitter_length = a.fine_jitter
    kink_rate, kink_amplitude, kink_width = a.kinks
    cv, width_length = a.width_variation
    for e in network_edges:
        if e.layer == "communicating" or e.kind == "arteriovenous anastomosis":
            continue
        pts = resample(e.points, 2.0)
        n = len(pts)
        if n < 6:
            continue
        seg = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)
        s = np.r_[0.0, np.cumsum(seg)]
        taper = np.clip(np.minimum(s, s[-1] - s) / 12.0, 0, 1)
        tangent = np.gradient(pts[:, :2], axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
        normal = np.stack([-tangent[:, 1], tangent[:, 0]], 1)
        offset = jitter_amplitude * _smooth_noise(rng, n, jitter_length / 2.0)
        if e.kind != "capillary":                      # capillaries stay straight between their nodes
            offset += _bumps(rng, s, kink_rate * 1e-3, kink_amplitude, kink_width)
        scale = 0.5 if e.kind == "capillary" else 1.0
        pts[:, :2] += (scale * offset * taper)[:, None] * normal
        e.points = pts
        e.points[0], e.points[-1] = pts[0], pts[-1]
        variation = 1.0 + cv * _smooth_noise(rng, n, width_length / 2.0)
        variation = np.clip(variation, 0.55, 1.45)
        e.width = variation / variation.mean()


def _vessel_chains(builder, kind):
    """The edges of every named vessel of a kind, in order from its first edge to its last."""
    by_vessel = {}
    for i, e in enumerate(builder.edges):
        if e.kind == kind:
            by_vessel.setdefault(e.vessel, []).append(i)
    chains = {}
    for vessel, members in by_vessel.items():
        starts = {builder.edges[i].u: i for i in members}
        ends = {builder.edges[i].v for i in members}
        first = [i for i in members if builder.edges[i].u not in ends]
        if len(first) != 1:
            continue
        chain = [first[0]]
        while builder.edges[chain[-1]].v in starts and len(chain) < len(members):
            chain.append(starts[builder.edges[chain[-1]].v])
        chains[vessel] = chain
    return chains


def _paired_venules(builder, rng, scene, domain):
    """Companion venules running beside the larger arterioles (docs §4): an offset copy of the arteriole's course,
    on one side, at a slowly varying edge-to-edge gap, a little deeper, wider than the arteriole by the cited calibre
    ratio, draining the capillary mesh along its length and leaving into the venular network or the domain border
    at the arteriole's proximal end, so that it carries blood back alongside the arteriole that brought it."""
    a = scene.anatomy
    mesh_nodes = sorted({k for e in builder.edges if e.kind == "capillary" for k in (e.u, e.v)})
    if not mesh_nodes:
        return 0
    mesh_tree = cKDTree(np.asarray([builder.nodes[k][:2] for k in mesh_nodes]))
    venule_nodes = sorted({k for e in builder.edges if e.kind == "venule" for k in (e.u, e.v)})
    venule_tree = cKDTree(np.asarray([builder.nodes[k][:2] for k in venule_nodes])) if venule_nodes else None
    made = 0
    taken_xy, taken_r = [], []                  # companions drawn so far: no two may share lumen space
    arteriole_points = {}
    for i, e in enumerate(builder.edges):
        if e.kind == "arteriole":
            arteriole_points.setdefault(e.vessel, []).append(np.column_stack([e.points[:, :2], np.full(len(e.points), e.diameter / 2)]))
    for vessel, chain in _vessel_chains(builder, "arteriole").items():
        edges = [builder.edges[i] for i in chain]
        if max(e.diameter for e in edges) < a.pair_min_diameter or rng.random() > a.pair_fraction:
            continue
        points = np.vstack([e.points if k == 0 else e.points[1:] for k, e in enumerate(edges)])
        d_art = np.concatenate([np.full(len(e.points) if k == 0 else len(e.points) - 1, e.diameter) for k, e in enumerate(edges)])
        points = resample(np.column_stack([points, d_art]), 4.0)
        d_art = points[:, 3]
        points = points[:, :3]
        seg = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
        s = np.r_[0.0, np.cumsum(seg)]
        if s[-1] < 250.0:
            continue
        smooth = np.stack([np.convolve(np.pad(points[:, k], 12, mode="edge"), np.ones(25) / 25, mode="valid") for k in (0, 1)], 1)
        tangent = np.gradient(smooth, axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
        normal = np.stack([-tangent[:, 1], tangent[:, 0]], 1)
        side = rng.choice([-1.0, 1.0])
        ratio = rng.uniform(*a.pair_calibre_ratio)
        d_ven = np.minimum(ratio * d_art, a.pair_max_venule_diameter)
        gap = np.clip(rng.uniform(*a.pair_gap_um) * (1 + 0.3 * _smooth_noise(rng, len(s), 30.0)), 2.0, None)
        offset = d_art / 2 + gap + d_ven / 2
        companion = points.copy()
        companion[:, :2] += side * offset[:, None] * normal
        companion[:, 2] += rng.uniform(5.0, 15.0)
        keep = s <= s[-1] * rng.uniform(0.7, 0.95)           # it stops short of the arteriole's terminal end
        keep &= np.array([_inside(p, domain) for p in companion])
        # stay clear of the companions drawn already and of every other arteriole: keep the longest run of the
        # companion whose lumen touches neither
        others = [np.vstack(v) for k, v in arteriole_points.items() if k != vessel]
        obstacles = np.vstack(others + ([np.column_stack([np.vstack(taken_xy), np.concatenate(taken_r)])] if taken_xy else []))
        if len(obstacles):
            tree_o = cKDTree(obstacles[:, :2])
            reach = d_ven / 2 + obstacles[:, 2].max() + 2.0
            for k in np.flatnonzero(keep):
                for j in tree_o.query_ball_point(companion[k, :2], reach[k]):
                    if np.linalg.norm(obstacles[j, :2] - companion[k, :2]) < d_ven[k] / 2 + obstacles[j, 2] + 2.0:
                        keep[k] = False
                        break
        runs = np.split(np.arange(len(keep)), np.flatnonzero(np.diff(keep.astype(int))) + 1)
        runs = [r for r in runs if keep[r[0]]]
        if not runs:
            continue
        best = max(runs, key=len)
        companion, d_ven, s_c = companion[best], d_ven[best], s[best] - s[best[0]]
        if len(companion) < 40:                       # under ~160 µm: not worth a vessel
            continue
        taken_xy.append(companion[:, :2])
        taken_r.append(d_ven / 2)
        # nodes every 150-250 µm, each draining a capillary of the mesh
        cuts = [0]
        while True:
            nxt = np.searchsorted(s_c, s_c[cuts[-1]] + rng.uniform(100.0, 160.0))
            if nxt >= len(s_c) - 3:
                break
            cuts.append(int(nxt))
        cuts.append(len(s_c) - 1)
        nodes = [builder.node(companion[c]) for c in cuts]
        vessel_id = builder.vessel("venule")
        for k in range(len(cuts) - 1):
            piece = companion[cuts[k]:cuts[k + 1] + 1]
            diameter = float(np.median(d_ven[cuts[k]:cuts[k + 1] + 1]))
            # the companion's flow runs from its distal end (high index) back to the arteriole's root (index 0)
            index = builder.edge(nodes[k + 1], nodes[k], piece[::-1].copy(), diameter, "venule", "conjunctival", vessel_id)
            builder.edges[index].paired_with = vessel
        for node in nodes[1:]:
            dists, nears = mesh_tree.query(builder.nodes[node][:2], k=1)
            for dist, near in zip(np.atleast_1d(dists), np.atleast_1d(nears)):
                if dist >= 300.0:
                    continue
                target = mesh_nodes[int(near)]
                pts = _segment(builder.nodes[target], builder.nodes[node], rng, *a.wiggle["capillary"])
                builder.edge(target, node, pts, rng.uniform(*a.capillary_diameter), "capillary", "conjunctival",
                             builder.vessel("capillary"))
        # the outlet: the domain border if the arteriole entered from it, else the nearest venule
        outlet = nodes[0]
        x0, x1, y0, y1 = domain
        p = builder.nodes[outlet]
        if min(p[0] - x0, x1 - p[0], p[1] - y0, y1 - p[1]) < 80.0 or venule_tree is None:
            builder.boundary[outlet] = "venous"
        else:
            dist, near = venule_tree.query(p[:2])
            target = venule_nodes[int(near)]
            pts = _segment(builder.nodes[outlet], builder.nodes[target], rng, *a.wiggle["venule"])
            builder.edge(outlet, target, pts, float(d_ven[0]), "venule", "conjunctival", vessel_id)
        made += 1
    return made


def generate(scene: Scene) -> Network:
    """Generate the 3D network of one scene."""
    rng = np.random.default_rng(scene.seed)
    a = scene.anatomy
    width_um = scene.width_px * a.um_per_px
    height_um = scene.height_px * a.um_per_px
    m = scene.margin_um
    domain = (0.0, width_um + 2 * m, 0.0, height_um + 2 * m)
    field_of_view = (m, m + width_um, m, m + height_um)
    area_mm2 = (domain[1] - domain[0]) * (domain[3] - domain[2]) * 1e-6
    builder = _Builder()

    # 2. and 3. the episcleral and deep plexuses
    _straight_layer(builder, rng, scene, domain, "episcleral", a.episcleral_vessels_per_mm, a.episcleral_plexus,
                    a.episcleral_artery_diameter, a.episcleral_vein_diameter, a.episcleral_curvature, 0.9)
    _straight_layer(builder, rng, scene, domain, "deep", a.deep_vessels_per_mm, a.deep_plexus,
                    a.deep_vessel_diameter, a.deep_vessel_diameter, a.episcleral_curvature, 0.2)

    # episcleral arteriovenous anastomoses: the episclera has no capillary bed; its arteries open directly into
    # the venous network through short anastomoses about the size of small arterioles [Selbach2005, FunkRohen1996]
    n_ava = rng.poisson(a.episcleral_avas_per_mm2 * area_mm2)
    for _ in range(n_ava):
        upper = _point_on_kind(builder, rng, ("episcleral artery",))
        if upper is None:
            break
        p = builder.edges[upper[0]].points[upper[1]]
        best = None
        for i, e in enumerate(builder.edges):
            if e.kind == "episcleral vein" and len(e.points) > 12:
                d = np.linalg.norm(e.points[6:-6, :2] - p[:2], axis=1)
                j = int(np.argmin(d))
                if 40.0 <= d[j] <= 400.0 and (best is None or d[j] < best[0]):
                    best = (d[j], i, j + 6)
        if best is None:
            continue
        node_a = builder.split(upper[0], upper[1])
        node_v = builder.split(best[1], best[2])
        pts = _segment(builder.nodes[node_a], builder.nodes[node_v], rng, *a.wiggle["communicating"])
        builder.edge(node_a, node_v, pts, rng.uniform(*a.ava_diameter), "arteriovenous anastomosis",
                     "episcleral", builder.vessel("arteriovenous anastomosis"))

    # communicating vessels: their lower ends on episcleral vessels, their tops become conjunctival roots
    communicating = []
    n_comm = rng.poisson(a.communicating_per_mm2 * area_mm2)
    for k in range(n_comm):
        venous = k % 2 == 1
        picked = _point_on_kind(builder, rng, ("episcleral vein",) if venous else ("episcleral artery",),
                                region=(domain[0] + 100, domain[1] - 100, domain[2] + 100, domain[3] - 100))
        if picked is None:
            continue
        edge_index, point_index = picked
        bottom = builder.edges[edge_index].points[point_index].copy()
        angle = math.radians(rng.uniform(*a.communicating_angle_deg))
        z_top = rng.uniform(a.conjunctival_plexus[0] + 0.5 * (a.conjunctival_plexus[1] - a.conjunctival_plexus[0]),
                            a.conjunctival_plexus[1])
        run = (bottom[2] - z_top) / math.tan(angle)
        direction = rng.uniform(0, 2 * math.pi)
        top = np.array([bottom[0] + run * math.cos(direction), bottom[1] + run * math.sin(direction), z_top])
        if not _inside(top, domain, pad=60):
            continue
        communicating.append(dict(venous=venous, top=top, bottom_edge=edge_index, bottom_point=point_index,
                                  bottom=bottom, angle=math.degrees(angle)))

    # 1. the conjunctival plexus
    made = _conjunctival_trees(builder, rng, scene, domain, communicating)
    leaves = made["arteriole"]["leaves"] + made["venule"]["leaves"]
    _capillary_mesh(builder, rng, scene, domain, leaves)
    _paired_venules(builder, rng, scene, domain)
    n_anastomoses = rng.poisson(a.anastomoses_per_mm2 * area_mm2)
    _anastomoses(builder, rng, scene, domain, "arteriole", n_anastomoses // 2)
    _anastomoses(builder, rng, scene, domain, "venule", n_anastomoses - n_anastomoses // 2)

    # join the communicating vessels: split the episcleral edge at the point nearest to the planned bottom
    for c in communicating:
        if "conjunctival_node" not in c:
            continue
        kinds = ("episcleral vein",) if c["venous"] else ("episcleral artery",)
        best = None
        for i, e in enumerate(builder.edges):
            if e.kind in kinds and len(e.points) > 6:
                d = np.linalg.norm(e.points[3:-3] - c["bottom"], axis=1)
                j = int(np.argmin(d))
                if best is None or d[j] < best[0]:
                    best = (d[j], i, j + 3)
        if best is None:
            continue
        node_bottom = builder.split(best[1], best[2])
        node_top = c["conjunctival_node"]
        path = _oblique_path(rng, builder.nodes[node_top], builder.nodes[node_bottom])
        kind = "communicating vein" if c["venous"] else "communicating artery"
        diameter = float(np.clip(c.get("diameter", 15.0), *a.communicating_diameter))
        builder.edge(node_top, node_bottom, path, diameter, kind, "communicating", builder.vessel(kind))

    # perforating vessels between the deep and episcleral plexuses
    n_perf = rng.poisson(a.perforator_per_mm2 * area_mm2)
    for k in range(n_perf):
        venous = k % 2 == 1
        upper = _point_on_kind(builder, rng, ("episcleral vein",) if venous else ("episcleral artery",))
        lower = _point_on_kind(builder, rng, ("deep vein",) if venous else ("deep artery",))
        if upper is None or lower is None:
            continue
        top_point = builder.edges[upper[0]].points[upper[1]]
        lower_edge = builder.edges[lower[0]]
        angle = math.radians(rng.uniform(*a.perforator_angle_deg))
        run = (lower_edge.points[:, 2].mean() - top_point[2]) / math.tan(angle)
        distances = np.abs(np.linalg.norm(lower_edge.points[:, :2] - top_point[:2], axis=1) - run)
        j = int(np.clip(np.argmin(distances), 4, len(lower_edge.points) - 5))
        if distances[j] > 150:
            continue
        node_top = builder.split(upper[0], upper[1])
        node_bottom = builder.split(lower[0], j)
        path = _oblique_path(rng, builder.nodes[node_top], builder.nodes[node_bottom])
        kind = "perforating vein" if venous else "perforating artery"
        diameter = rng.uniform(*a.perforating_artery_diameter) * (1.3 if venous else 1.0)
        builder.edge(node_top, node_bottom, path, diameter, kind, "perforating", builder.vessel(kind))

    _roughen(builder.edges, np.random.default_rng([scene.seed, 2]), scene)
    network = Network(np.asarray(builder.nodes), builder.edges, dict(builder.boundary), domain, field_of_view,
                      dict(builder.vessel_kind))
    network = _clean(network)
    # episcleral and deep vessels carry blood along the radius: arteries towards the limbus, veins away from it
    # [Meyer1988], so their boundary pressure falls towards the limbus for arteries and away from it for veins
    incident = network.incident()
    y_mid = 0.5 * (domain[2] + domain[3])
    gradient = scene.physiology.radial_pressure_gradient_mmhg_per_mm
    for node, role in network.boundary.items():
        kinds = {network.edges[i].kind for i in incident[node]}
        if any(k.startswith(("episcleral", "deep")) for k in kinds):
            sign = 1.0 if role == "arterial" else -1.0
            network.boundary_offset[node] = sign * gradient * (network.nodes[node][1] - y_mid) * 1e-3
    return network


def generate_fine(scene: Scene, domain, roots, seed) -> tuple:
    """The fine vessels of one tile of the eye: conjunctival arteriolar and venular trees grown from the given roots
    (points on the coarse trunks, see synth.eye), the capillary mesh between them, anastomoses, and the multi-scale
    irregularity, all from one seed (the tile's), so a tile generated again is the same tile.

    roots: dicts with venous, top (x, y, z µm in the tile's own coordinates) and heading (radians). Returns the
    network (no boundary nodes) and, per root, its node (or None when its tree did not grow)."""
    rng = np.random.default_rng(seed)
    builder = _Builder()
    roots = [dict(r) for r in roots]
    made = _conjunctival_trees(builder, rng, scene, domain, roots, border_roots=False)
    leaves = made["arteriole"]["leaves"] + made["venule"]["leaves"]
    _capillary_mesh(builder, rng, scene, domain, leaves)
    x0, x1, y0, y1 = domain
    n_anastomoses = rng.poisson(scene.anatomy.anastomoses_per_mm2 * (x1 - x0) * (y1 - y0) * 1e-6)
    _anastomoses(builder, rng, scene, domain, "arteriole", n_anastomoses // 2)
    _anastomoses(builder, rng, scene, domain, "venule", n_anastomoses - n_anastomoses // 2)
    _roughen(builder.edges, np.random.default_rng([int(seed) % (2 ** 32), 2]), scene)
    root_nodes = [r.get("conjunctival_node") for r in roots]
    network = Network(np.asarray(builder.nodes) if builder.nodes else np.zeros((0, 3)), builder.edges, {}, domain,
                      domain, dict(builder.vessel_kind))
    return network, root_nodes


def _clean(network: Network) -> Network:
    """Drop zero-length edges and components that touch no boundary node (they could carry no flow)."""
    edges = [e for e in network.edges if e.length > 1.0 and e.u != e.v]
    n = len(network.nodes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for e in edges:
        parent[find(e.u)] = find(e.v)
    roots_with_boundary = {find(k) for k in network.boundary}
    edges = [e for e in edges if find(e.u) in roots_with_boundary]
    used = sorted({k for e in edges for k in (e.u, e.v)})
    remap = {old: new for new, old in enumerate(used)}
    for e in edges:
        e.u, e.v = remap[e.u], remap[e.v]
    boundary = {remap[k]: role for k, role in network.boundary.items() if k in remap}
    return Network(network.nodes[used], edges, boundary, network.domain, network.field_of_view, network.vessel_names)



def summary(network: Network) -> dict:
    """Counts and lengths by kind, for the notebook and the anatomy check."""
    out = {"nodes": len(network.nodes), "edges": len(network.edges), "boundary nodes": len(network.boundary)}
    by_kind = {}
    for e in network.edges:
        k = by_kind.setdefault(e.kind, {"edges": 0, "length_mm": 0.0, "vessels": set()})
        k["edges"] += 1
        k["length_mm"] += e.length * 1e-3
        k["vessels"].add(e.vessel)
    out["by kind"] = {kind: {"edges": v["edges"], "length (mm)": round(v["length_mm"], 2), "vessels": len(v["vessels"])}
                      for kind, v in sorted(by_kind.items())}
    return out
