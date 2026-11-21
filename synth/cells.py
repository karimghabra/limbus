"""Discrete red blood cells carried through the network.

Every segment holds the number of cells its tube haematocrit implies, N = H_T A L / V_rbc (docs §6.4). A cell has
an axial position s along its segment's centreline and a position (r, theta) in the circular cross-section; r is
drawn uniformly by area inside the red-cell core, which is narrower than the lumen by the cell-free layer, whose
width grows with calibre [Kim2007, Tateishi1994]. In capillaries narrower than a red cell the cells run in single
file on the axis.

Axial spacing: the gaps between successive cells follow a gamma distribution with the coefficient of variation of
measured capillary inter-cell distances [Ding2025], and the axial density is modulated by a log-normal random field
with the axial correlation length of red-cell structure in microvessels [Fedosov2010, Kim2007], so that cells
gather in aggregates separated by plasma gaps [Fahraeus1929] - the moving texture the camera sees as flicker. Its
amplitude is fitted to the measured flicker of the real bursts (docs §6.4). Velocity follows a blunted profile u(rho) = u_max (1 - rho^K) (K > 2 is blunter than Poiseuille
[Tangelder1986]), scaled so that the mean over the cells equals the segment's mean red-cell velocity from the flow
solution at that instant, with the cardiac pulse. A cell that reaches the downstream end of its segment passes to
an outgoing segment chosen so that the red-cell flux leaving each junction matches the flow solution's
phase-separated split (a deterministic quota per junction, so the split is exact rather than only on average);
cells leave the domain at outflow boundaries and enter at inflow boundaries as a gamma renewal process at the
segment's red-cell flux.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .anatomy import Physiology
from .haemodynamics import Flow
from .network import Network, resample

SAMPLE_UM = 1.0             # centreline sampling for the cell lookup


@dataclass
class Geometry:
    """Every segment's centreline sampled every SAMPLE_UM, with an orthonormal frame, in one flat array."""
    start: np.ndarray        # (E,) index of each segment's first sample
    count: np.ndarray        # (E,)
    length: np.ndarray       # (E,) µm
    centre: np.ndarray       # (M, 3)
    normal: np.ndarray       # (M, 3)
    binormal: np.ndarray     # (M, 3)
    tangent: np.ndarray      # (M, 3)
    width: np.ndarray        # (M,) local diameter over the segment's nominal diameter


def geometry(network: Network) -> Geometry:
    starts, counts, lengths, centres, normals, binormals, tangents, widths = [], [], [], [], [], [], [], []
    offset = 0
    for e in network.edges:
        if e.width is not None:
            points = resample(np.column_stack([e.points, e.width]), SAMPLE_UM)
            width = points[:, 3]
            points = points[:, :3]
        else:
            points = resample(e.points, SAMPLE_UM)
            width = np.ones(len(points))
        tangent = np.gradient(points, axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
        reference = np.where(np.abs(tangent[:, 2:3]) < 0.9, np.array([[0.0, 0.0, 1.0]]), np.array([[1.0, 0.0, 0.0]]))
        normal = np.cross(tangent, reference)
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
        binormal = np.cross(tangent, normal)
        starts.append(offset)
        counts.append(len(points))
        lengths.append(float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()))
        centres.append(points)
        normals.append(normal)
        binormals.append(binormal)
        tangents.append(tangent)
        widths.append(width)
        offset += len(points)
    return Geometry(np.array(starts), np.array(counts), np.array(lengths), np.vstack(centres), np.vstack(normals),
                    np.vstack(binormals), np.vstack(tangents), np.concatenate(widths))


def cell_free_layer(diameter, physiology: Physiology):
    """Width of the plasma layer at the wall (µm): linear in diameter from the low end of the cited range at 10 µm
    to the high end at 50 µm [Kim2007], held constant outside that range."""
    low, high = physiology.cell_free_layer_um
    t = np.clip((np.asarray(diameter, float) - 10.0) / (50.0 - 10.0), 0, 1)
    return low + (high - low) * t


class Cells:
    """The red cells of one network, advanced in time."""

    def __init__(self, network: Network, flow: Flow, physiology: Physiology, rng, geometry_=None):
        self.network = network
        self.flow = flow
        self.p = physiology
        self.rng = rng
        self.geo = geometry_ or geometry(network)
        E = len(network.edges)
        d = flow.diameter
        self.radius = d / 2
        cfl = cell_free_layer(d, physiology)
        # radius available to cell centres: the core inside the cell-free layer; zero in single-file capillaries
        self.core = np.clip(self.radius - cfl - 0.5 * physiology.rbc_thickness_um, 0.0, None)
        self.core[d < physiology.rbc_diameter_um + 1.0] = 0.0
        self.single_file = self.core <= 0.0
        K = physiology.profile_bluntness
        rho_max = np.where(self.radius > 0, self.core / self.radius, 0.0)
        # mean of (1 - rho^K) over cells spread uniformly by area within rho_max
        self.profile_mean = 1.0 - 2.0 * rho_max ** K / (K + 2.0)
        self.length = self.geo.length
        area = np.pi * d ** 2 / 4
        expected = flow.ht * area * self.length / physiology.rbc_volume_um3
        self.expected = expected
        # ends of every edge and the node topology
        self.u = np.array([e.u for e in network.edges])
        self.v = np.array([e.v for e in network.edges])
        self.incident = network.incident()
        self.boundary = network.boundary
        # at a junction the lumens merge: the last stretch of every narrower vessel lies inside the widest one, so
        # its cells there are not drawn a second time (otherwise the cell clouds overlap and the junction shows as
        # a dark blob); trim_u / trim_v are the hidden lengths at each end
        self.trim_u = np.zeros(E)
        self.trim_v = np.zeros(E)
        geo = self.geo

        def direction_at(i, node):
            """Unit direction of edge i leaving node (over its first 10 µm)."""
            start, count = geo.start[i], geo.count[i]
            pts = geo.centre[start:start + count]
            a, b = (pts[0], pts[min(10, count - 1)]) if self.u[i] == node else (pts[-1], pts[max(count - 11, 0)])
            v = b - a
            return v / max(np.linalg.norm(v), 1e-9)

        for node, members in enumerate(self.incident):
            if len(members) < 3:
                continue                                  # a plain continuation: nothing overlaps
            widest = max(members, key=lambda i: d[i])
            w_dir = direction_at(widest, node)
            others = [i for i in members if i != widest]
            continuation = min(others, key=lambda i: float(direction_at(i, node) @ w_dir))   # most opposite
            for i in others:
                if i == continuation:
                    continue
                if self.u[i] == node:
                    self.trim_u[i] = max(self.trim_u[i], d[widest] / 2)
                if self.v[i] == node:
                    self.trim_v[i] = max(self.trim_v[i], d[widest] / 2)
        self.quota = {}
        # initial cells: a clumped renewal sequence along every segment
        edge_ids, s_values = [], []
        for e in range(E):
            n = expected[e]
            if n < 1e-3 or self.length[e] <= 0:
                continue
            s = self._renewal_positions(self.length[e], n)
            edge_ids.append(np.full(len(s), e))
            s_values.append(s)
        self.edge = np.concatenate(edge_ids) if edge_ids else np.zeros(0, int)
        self.s = np.concatenate(s_values) if s_values else np.zeros(0)
        self.rfrac, self.theta = self._cross_section(len(self.edge))
        self.ids = np.arange(len(self.edge))
        self.next_id = len(self.edge)
        self.t = 0.0
        # inflow boundaries: (edge, entering end) and the time to the next arrival
        self.inflow = []
        for node in network.boundary:
            for e in self.incident[node]:
                self.inflow.append((e, node))
        self.inflow_scale = {key: 1.0 for key in self.inflow}
        # junctions whose drawn vessels do not conserve red-cell flux: in an eye's view, blood also leaves or joins
        # through vessels the view does not draw (the coarse level's lumped beds, fine trees whose flow was solved
        # at their own level). Where more red-cell flux arrives than leaves, the excess cells leave with that blood
        # (keep_share); where more leaves than arrives, the missing cells enter as at an inflow boundary. Without
        # this the cells of a vessel pile up past such a junction, or thin out, in blocks along the vessel.
        q, hd = flow.flow, flow.hd
        self.keep_share = {}
        self.keep_quota = {}
        for node, members in enumerate(self.incident):
            if node in network.boundary or len(members) < 2:
                continue
            f_in = f_out = 0.0
            for e in members:
                into = (self.v[e] == node and q[e] > 0) or (self.u[e] == node and q[e] < 0)
                f_in += abs(q[e]) * hd[e] if into else 0.0
                f_out += 0.0 if into else abs(q[e]) * hd[e]
            if f_in > 1.01 * f_out:
                self.keep_share[node] = f_out / f_in
            elif f_out > 1.01 * f_in and f_out > 0:
                for e in members:
                    leaving = (self.u[e] == node and q[e] > 0) or (self.v[e] == node and q[e] < 0)
                    if leaving and abs(q[e]) * hd[e] > 0:
                        key = (e, node)
                        self.inflow.append(key)
                        self.inflow_scale[key] = (f_out - f_in) / f_out
        self.next_arrival = {key: self.rng.exponential(1.0) for key in self.inflow}
        self.clump_state = {}

    # ------------------------------------------------------------------------------------------- sampling

    def _gamma_gaps(self, n, mean):
        cv = self.p.aggregation_cv
        shape = 1.0 / cv ** 2
        return self.rng.gamma(shape, mean / shape, n)

    def _density_field(self, n):
        """A stationary log-normal density modulation along a vessel, one value per µm, with the cited axial
        correlation length of red-cell structure and the fitted relative amplitude (mean 1)."""
        ell = self.p.clump_length_um
        sigma = self.p.clump_amplitude
        pad = int(3 * ell) + 1
        raw = self.rng.standard_normal(n + 2 * pad)
        kernel = np.exp(-0.5 * (np.arange(-pad, pad + 1) / ell) ** 2)
        g = np.convolve(raw, kernel / np.sqrt((kernel ** 2).sum()), mode="same")[pad:pad + n]
        return np.exp(sigma * g - 0.5 * sigma ** 2)

    def _renewal_positions(self, length, expected):
        """Cell positions along a segment: a gamma renewal sequence (the spacing statistics of cells) thinned and
        thickened by the clump field, so that cells gather in aggregates separated by plasma gaps."""
        positions = self._plain_renewal(length * 1.0, expected)
        if self.p.clump_amplitude <= 0 or len(positions) == 0:
            return positions
        field = self._density_field(int(length) + 2)
        weight = field[np.clip(positions.astype(int), 0, len(field) - 1)]
        keep = self.rng.random(len(positions)) < weight / weight.max()
        positions = positions[keep]
        # top the count back up to the expected number by drawing extra cells where the field is dense
        missing = self.rng.poisson(max(expected - len(positions), 0.0)) if expected > len(positions) else 0
        if missing:
            cdf = np.cumsum(field)
            extra = np.searchsorted(cdf, self.rng.uniform(0, cdf[-1], missing)) + self.rng.random(missing)
            positions = np.sort(np.r_[positions, np.clip(extra, 0, length - 1e-6)])
        return positions

    def _plain_renewal(self, length, expected):
        mean_gap = length / max(expected, 1e-9)
        n_draw = int(expected * 1.5 + 20)
        gaps = self._gamma_gaps(n_draw, mean_gap)
        positions = np.cumsum(gaps) - self.rng.uniform(0, gaps[0])
        while positions[-1] < length:
            more = np.cumsum(self._gamma_gaps(n_draw, mean_gap)) + positions[-1]
            positions = np.r_[positions, more]
        return positions[(positions >= 0) & (positions < length)]

    def _cross_section(self, n):
        return np.sqrt(self.rng.uniform(0, 1, n)), self.rng.uniform(0, 2 * np.pi, n)

    # ------------------------------------------------------------------------------------------- motion

    def _rbc_velocity(self, t):
        """Signed mean red-cell velocity of every segment at time t, µm/s."""
        q_now = self.flow.flow_at(t)
        q_mean = self.flow.flow
        scale = np.where(np.abs(q_mean) > 1e-12, q_now / np.where(np.abs(q_mean) > 1e-12, q_mean, 1.0), 0.0)
        return self.flow.rbc_velocity * 1e3 * scale, q_now

    def _sample(self):
        g = self.geo
        return g.start[self.edge] + np.clip(np.round(self.s / SAMPLE_UM).astype(int), 0, g.count[self.edge] - 1)

    def cell_speed(self, v_edge):
        """Speed of every cell along its segment (signed, µm/s) from its radial position and the profile; where the
        lumen narrows the blood speeds up as the square of the width, so that the flux is the same along it."""
        e = self.edge
        K = self.p.profile_bluntness
        rho = np.where(self.radius[e] > 0, self.rfrac * self.core[e] / self.radius[e], 0.0)
        shape = (1.0 - rho ** K) / self.profile_mean[e]
        return v_edge[e] * shape / self.geo.width[self._sample()] ** 2

    def step(self, dt):
        """Advance every cell by dt seconds (velocities evaluated at the middle of the step)."""
        v_edge, q_now = self._rbc_velocity(self.t + 0.5 * dt)
        self.s = self.s + self.cell_speed(v_edge) * dt
        self._inject(dt, q_now, v_edge)
        for _ in range(12):
            out = (self.s >= self.length[self.edge]) | (self.s < 0)
            if not out.any():
                break
            self._transfer(np.flatnonzero(out), q_now)
        self.t += dt

    def _inject(self, dt, q_now, v_edge):
        """New cells at inflow boundaries: a clumped renewal process at the red-cell flux."""
        new_edge, new_s = [], []
        for key in self.inflow:
            e, node = key
            q = q_now[e]
            entering = (node == self.u[e] and q > 0) or (node == self.v[e] and q < 0)
            if not entering:
                continue
            rate = abs(q) * 1e3 * self.flow.hd[e] / self.p.rbc_volume_um3 * self.inflow_scale[key]   # cells per second
            if rate <= 0:
                continue
            # next_arrival is kept in units of mean gaps, so that a changing rate keeps the process continuous
            # the clump field travels in with the blood: an Ornstein-Uhlenbeck value per inflow, correlated over
            # the time the blood takes to cover one clump length
            speed = max(abs(v_edge[e]), 1e-6)
            g = self.clump_state.get(key, self.rng.standard_normal())
            rho = np.exp(-speed * dt / max(self.p.clump_length_um, 1e-6))
            g = rho * g + np.sqrt(1 - rho ** 2) * self.rng.standard_normal()
            self.clump_state[key] = g
            sigma = self.p.clump_amplitude
            rate *= np.exp(sigma * g - 0.5 * sigma ** 2)
            budget = rate * dt
            remaining = self.next_arrival[key]
            times = []
            elapsed = 0.0
            while remaining <= budget - elapsed:
                elapsed += remaining
                times.append(elapsed / rate)
                remaining = float(self._gamma_gaps(1, 1.0)[0])
            self.next_arrival[key] = remaining - (budget - elapsed)
            speed = abs(v_edge[e])
            for t_in in times:
                travelled = speed * (dt - t_in)
                s = travelled if node == self.u[e] else self.length[e] - travelled
                new_edge.append(e)
                new_s.append(float(np.clip(s, 0.0, self.length[e] - 1e-6)))
        if new_edge:
            n = len(new_edge)
            r, th = self._cross_section(n)
            self.edge = np.r_[self.edge, new_edge]
            self.s = np.r_[self.s, new_s]
            self.rfrac = np.r_[self.rfrac, r]
            self.theta = np.r_[self.theta, th]
            self.ids = np.r_[self.ids, np.arange(self.next_id, self.next_id + n)]
            self.next_id += n

    def _outgoing(self, node, q_now):
        """Segments leaving a node under the current flow, and their red-cell flux shares."""
        out, flux = [], []
        for e in self.incident[node]:
            q = q_now[e]
            # only a segment that carries red cells away can take them: one without red-cell flux (no flow, or
            # plasma only) would hold arriving cells still and pile them into a dark spot
            if ((self.u[e] == node and q > 0) or (self.v[e] == node and q < 0)) and abs(q) * self.flow.hd[e] > 0:
                out.append(e)
                flux.append(abs(q) * self.flow.hd[e])
        flux = np.array(flux)
        if len(out) and flux.sum() > 0:
            return out, flux / flux.sum()
        return out, flux

    def _transfer(self, indices, q_now):
        """Move cells that ran past the end of their segment onto the next one, or out of the domain.

        At each junction the arriving cells are shared among the outgoing segments by a largest-remainder quota
        on their red-cell flux shares, carried over from step to step, so the split of red-cell flux is exact."""
        keep = np.ones(len(self.edge), bool)
        e = self.edge[indices]
        L = self.length[e]
        forward = self.s[indices] >= L
        node = np.where(forward, self.v[e], self.u[e])
        overshoot = np.where(forward, self.s[indices] - L, -self.s[indices])
        for n in np.unique(node):
            members = np.flatnonzero(node == n)
            cells_here = indices[members]
            if int(n) in self.boundary:
                keep[cells_here] = False
                continue
            if int(n) in self.keep_share:          # the share of arriving cells that stays in drawn vessels
                quota = self.keep_quota.get(int(n), 0.0) + len(members) * self.keep_share[int(n)]
                stay = int(np.floor(quota))
                self.keep_quota[int(n)] = quota - stay
                order = self.rng.permutation(len(members))
                keep[cells_here[order[stay:]]] = False
                members, cells_here = members[order[:stay]], cells_here[order[:stay]]
                if not len(members):
                    continue
            out, share = self._outgoing(int(n), q_now)
            if len(out) == 0:
                # no drawn vessel carries blood away from here: the blood leaves through vessels this network does
                # not draw (in an eye's view, the coarse level's lumped beds or a region without tiles), and so do
                # its cells; waiting would pile them into a dark spot
                keep[cells_here] = False
                continue
            share = share if share.sum() > 0 else np.full(len(out), 1.0 / len(out))
            residual = self.quota.setdefault(int(n), {})
            r = np.array([residual.get(o, 0.0) for o in out]) + len(members) * share
            counts = np.clip(np.floor(r), 0, None).astype(int)
            while counts.sum() < len(members):
                counts[int(np.argmax(r - counts))] += 1
            while counts.sum() > len(members):
                spare = np.where(counts > 0, r - counts, np.inf)
                counts[int(np.argmin(spare))] -= 1
            r -= counts
            for o, value in zip(out, r):
                residual[o] = float(value)
            chosen = np.repeat(np.array(out), counts)
            self.rng.shuffle(chosen)
            self.edge[cells_here] = chosen
            starts_here = self.u[chosen] == n
            self.s[cells_here] = np.where(starts_here, overshoot[members], self.length[chosen] - overshoot[members])
            self.rfrac[cells_here] = np.sqrt(self.rng.random(len(members)))
            self.theta[cells_here] = self.rng.uniform(0, 2 * np.pi, len(members))
        if not keep.all():
            self.edge, self.s = self.edge[keep], self.s[keep]
            self.rfrac, self.theta, self.ids = self.rfrac[keep], self.theta[keep], self.ids[keep]

    # ------------------------------------------------------------------------------------------- positions

    def positions(self):
        """World coordinates (µm) of every cell's centre, (n, 3)."""
        g = self.geo
        e = self.edge
        index = self._sample()
        r = self.rfrac * self.core[e] * g.width[index]
        offset = (r * np.cos(self.theta))[:, None] * g.normal[index] + (r * np.sin(self.theta))[:, None] * g.binormal[index]
        return g.centre[index] + offset

    def visible(self):
        """The cells to draw: all but those in the stretch of a narrower vessel that lies inside the widest vessel
        at a junction."""
        e = self.edge
        return (self.s >= self.trim_u[e]) & (self.s <= self.length[e] - self.trim_v[e])

    def warm_up(self, seconds, dt):
        for _ in range(int(round(seconds / dt))):
            self.step(dt)
