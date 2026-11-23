"""Steady and pulsatile blood flow on the 3D network: mass conservation at every junction, diameter- and
haematocrit-dependent viscosity, and red-cell phase separation at diverging bifurcations.

Each vessel segment is a Poiseuille tube with conductance g = pi D^4 / (128 mu L); the apparent viscosity mu is
the plasma viscosity times the relative viscosity of the in vivo law [Pries1994], which includes the
Fahraeus-Lindqvist effect and the endothelial surface layer [PriesSecomb2005]. Pressures at the boundary nodes (where vessels leave
the domain) are set from the vessel's role and calibre (docs §6.2), and the interior pressures solve the linear
system sum_j g_ij (p_i - p_j) = 0, which conserves volume flow exactly at every interior node.

The discharge haematocrit H_D is then propagated downstream node by node in order of falling pressure (the flow
graph of a potential flow has no cycles): at a converging junction the red-cell fluxes add; at a diverging
bifurcation the red-cell flux splits by the phase-separation law [Pries1990, PriesSecomb2005], so the daughter
with the larger share of the flow takes a disproportionately larger share of the cells. Red-cell flux is therefore
conserved at every node as well. Viscosity depends on H_D, so flow and haematocrit are iterated to a fixed point.

The tube haematocrit H_T (the cells actually present in a segment, lower than H_D: the Fahraeus effect) follows
[Pries1992]; the mean red-cell velocity is v_rbc = Q H_D / (A H_T) and the mean blood velocity Q / A.

Pulsation: with the viscosities frozen at their mean values the system is linear, so the pulse is solved harmonic
by harmonic about the mean flow. The arterial inlets carry an arterial pressure wave (a systolic peak and a dicrotic
wave, half its pulse pressure m times the mean; docs §6.3) at the cardiac frequency f; the vessels are distensible,
so every node stores a volume C dp/dt (C from the volume of the vessels meeting there and their distensibility,
larger on the venous side), and each harmonic solves sum_j g_ij (P_i - P_j) + i k w C_i P_i = 0. The pulse is
therefore largest in the arterioles, damped and delayed through the capillaries, and small in the veins. Volume is
conserved at every node with the stored volume included, and exactly on average over the cycle. Flow may reverse
in a segment during the cycle, as it does in some capillaries.

Units: µm, s, Pa internally; pressures reported in mmHg, flows in pl/s (1 pl = 1000 µm^3), velocities in mm/s.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from .anatomy import Physiology
from .network import ARTERIAL_KINDS, VENOUS_KINDS, Network

MMHG = 133.322                     # Pa per mmHg


# ----------------------------------------------------------------------------------------------- blood rheology

def _shape_c(d):
    """The haematocrit-shape exponent C(D) of the viscosity laws [Pries1992, Pries1994]."""
    s = 1.0 / (1.0 + 1e-11 * d ** 12)
    return (0.8 + np.exp(-0.075 * d)) * (-1.0 + s) + s


def viscosity_in_vivo(d, hd):
    """Relative apparent viscosity in microvessels in vivo [Pries1994], with the endothelial surface layer
    represented by the effective diameter d - 1.1 µm."""
    d = np.maximum(np.asarray(d, float), 2.5)
    hd = np.clip(np.asarray(hd, float), 0.0, 0.95)
    mu45 = 6.0 * np.exp(-0.085 * d) + 3.2 - 2.44 * np.exp(-0.06 * d ** 0.645)
    c = _shape_c(d)
    ratio = (d / (d - 1.1)) ** 2
    return (1.0 + (mu45 - 1.0) * ((1.0 - hd) ** c - 1.0) / ((1.0 - 0.45) ** c - 1.0) * ratio) * ratio


def tube_haematocrit(d, hd):
    """Tube haematocrit from discharge haematocrit in a tube of diameter d µm (Fahraeus effect) [Pries1992]."""
    d = np.asarray(d, float)
    hd = np.asarray(hd, float)
    return hd * (hd + (1.0 - hd) * (1.0 + 1.7 * np.exp(-0.35 * d) - 0.6 * np.exp(-0.01 * d)))


def phase_separation(fq_blood, d_parent, d_alpha, d_beta, hd_parent):
    """Fraction of the parent's red-cell flux entering daughter alpha, given its fraction of the blood flow
    [Pries1990, with the parameters of PriesSecomb2005]. d in µm."""
    d_parent = max(d_parent, 1e-3)
    x0 = 0.964 * (1.0 - hd_parent) / d_parent
    ratio = (d_alpha / d_beta) ** 2
    a = -13.29 * ((ratio - 1.0) / (ratio + 1.0)) * (1.0 - hd_parent) / d_parent
    b = 1.0 + 6.98 * (1.0 - hd_parent) / d_parent
    if fq_blood <= x0:
        return 0.0
    if fq_blood >= 1.0 - x0:
        return 1.0
    x = (fq_blood - x0) / (1.0 - 2.0 * x0)
    logit = np.log(x / (1.0 - x))
    return float(1.0 / (1.0 + np.exp(-(a + b * logit))))


# ----------------------------------------------------------------------------------------------- boundary conditions

def boundary_pressure(role, diameter, physiology: Physiology):
    """Pressure (mmHg) at a boundary node, from the role and calibre of the vessel leaving the domain there:
    log-linear in diameter between the two calibres given for each side of the circulation (docs §6.2)."""
    p = physiology
    t = np.clip(np.log(diameter / 10.0) / np.log(10.0), -0.5, 1.5)    # 0 at 10 µm, 1 at 100 µm
    at_10um, at_100um = {"arterial": (p.arterial_pressure_at_10um, p.arterial_pressure_at_100um),
                         "venous": (p.venous_pressure_at_10um, p.venous_pressure_at_100um)}[role]
    return at_10um + t * (at_100um - at_10um)


@dataclass
class Flow:
    pressure: np.ndarray            # (N,) mmHg at the nodes
    flow: np.ndarray                # (E,) pl/s, positive from edge.u to edge.v
    hd: np.ndarray                  # (E,) discharge haematocrit
    ht: np.ndarray                  # (E,) tube haematocrit
    viscosity: np.ndarray           # (E,) relative apparent viscosity
    velocity: np.ndarray            # (E,) mean blood velocity Q / A, mm/s (signed like flow)
    rbc_velocity: np.ndarray        # (E,) mean red-cell velocity, mm/s (signed)
    shear_rate: np.ndarray          # (E,) wall shear rate 8 |v| / D, 1/s
    length: np.ndarray              # (E,) µm
    diameter: np.ndarray            # (E,) µm
    iterations: int
    physiology: Physiology
    pulse: np.ndarray               # (K, E) complex pl/s: harmonic k + 1 of every edge's flow
    pulse_pressure: np.ndarray      # (K, N) complex mmHg: of every node's pressure
    compliance: np.ndarray          # (N,) µm^3 / mmHg stored at every node

    def _phasors(self, t):
        k = np.arange(1, len(self.pulse) + 1)
        return np.exp(2j * np.pi * self.physiology.heart_rate_hz * k * t)

    def flow_at(self, t):
        """Flow (pl/s) of every edge at time t (s), with the cardiac pulse."""
        return self.flow + np.real(self._phasors(t) @ self.pulse)

    def pressure_at(self, t):
        """Pressure (mmHg) of every node at time t (s)."""
        return self.pressure + np.real(self._phasors(t) @ self.pulse_pressure)

    def storage_at(self, t):
        """Rate (pl/s) at which every node's distending vessels take up volume at time t (s)."""
        k = np.arange(1, len(self.pulse) + 1)[:, None]
        w = 2 * np.pi * self.physiology.heart_rate_hz
        rate = 1j * k * w * self.pulse_pressure * self.compliance[None, :]          # µm^3/s
        return np.real(self._phasors(t) @ rate) * 1e-3

    def conservation(self, network: Network):
        """Largest relative imbalance of volume flow and of red-cell flux over the interior nodes."""
        n = len(network.nodes)
        q_net = np.zeros(n)
        q_abs = np.zeros(n)
        c_net = np.zeros(n)
        c_abs = np.zeros(n)
        for index, e in enumerate(network.edges):
            q = self.flow[index]
            c = q * self.hd[index]
            q_net[e.u] -= q
            q_net[e.v] += q
            q_abs[e.u] += abs(q)
            q_abs[e.v] += abs(q)
            c_net[e.u] -= c
            c_net[e.v] += c
            c_abs[e.u] += abs(c)
            c_abs[e.v] += abs(c)
        interior = np.array([k not in network.boundary for k in range(n)])
        # relative to the node's own throughput, or to a millionth of the largest throughput for stagnant nodes
        q_floor = 1e-6 * max(q_abs.max(initial=0.0), 1e-30)
        c_floor = 1e-6 * max(c_abs.max(initial=0.0), 1e-30)
        flow_error = np.abs(q_net[interior]) / np.maximum(q_abs[interior], q_floor)
        cell_error = np.abs(c_net[interior]) / np.maximum(c_abs[interior], c_floor)
        return float(flow_error.max(initial=0.0)), float(cell_error.max(initial=0.0))


def _solve(network: Network, conductance, boundary_values):
    """Node pressures with fixed values at the boundary nodes and conservation everywhere else."""
    n = len(network.nodes)
    u = np.array([e.u for e in network.edges])
    v = np.array([e.v for e in network.edges])
    rows = np.r_[u, v, u, v]
    cols = np.r_[u, v, v, u]
    vals = np.r_[conductance, conductance, -conductance, -conductance]
    matrix = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    fixed = np.zeros(n, bool)
    value = np.zeros(n)
    for node, p in boundary_values.items():
        fixed[node] = True
        value[node] = p
    free = ~fixed
    rhs = -matrix[free][:, fixed] @ value[fixed]
    pressure = value.copy()
    pressure[free] = spsolve(matrix[free][:, free].tocsc(), rhs)
    return pressure


def solve(network: Network, physiology: Physiology, max_iterations=400, tolerance=1e-5, relax=0.5) -> Flow:
    edges = network.edges
    E = len(edges)
    # an edge may stand for more vessel than its centreline shows (the eye's lumped beds): length_override
    length = np.array([getattr(e, "length_override", None) or e.length for e in edges])
    diameter = np.array([e.diameter for e in edges])
    incident = network.incident()
    # boundary pressures from the role and the calibre of the vessel at each boundary node
    p_boundary = {}
    for node, role in network.boundary.items():
        d = max(edges[i].diameter for i in incident[node])
        if node in network.boundary_pressure:
            p_boundary[node] = network.boundary_pressure[node] * MMHG
            continue
        offset = network.boundary_offset.get(node, 0.0)
        p_boundary[node] = (boundary_pressure(role, d, physiology) + offset) * MMHG
    arterial_nodes = {k for k, r in network.boundary.items() if r == "arterial"}
    # dangling branches (ending in a node that is neither a junction nor a boundary) carry no flow; the linear
    # solve leaves them a residue of numerical noise, enough for the haematocrit split to fill them with cells that
    # never move, so their flow is set to exactly zero
    dangling = _dangling(network, incident)
    hd = np.full(E, physiology.inflow_discharge_haematocrit)
    mu_plasma = physiology.plasma_viscosity_mpas * 1e-3
    iterations = 0
    previous_change = np.inf
    for iterations in range(1, max_iterations + 1):
        mu_rel = viscosity_in_vivo(diameter, hd)
        g = np.pi * diameter ** 4 / (128.0 * mu_plasma * mu_rel * length)      # µm^3 / (Pa s)
        pressure = _solve(network, g, p_boundary)
        q = g * np.array([pressure[e.u] - pressure[e.v] for e in edges])        # µm^3 / s
        q[dangling] = 0.0
        new_hd = _propagate_haematocrit(network, q, diameter, pressure, physiology.inflow_discharge_haematocrit,
                                        incident, hd)
        change = float(np.max(np.abs(new_hd - hd))) if E else 0.0
        if change < tolerance:
            hd = new_hd
            break
        if iterations > 1 and change > previous_change:
            relax = max(0.05, 0.5 * relax)          # oscillating: damp harder
        previous_change = change
        hd = (1 - relax) * hd + relax * new_hd
    # final consistent pass: flows from the final viscosities, haematocrit propagated on exactly these flows
    mu_rel = viscosity_in_vivo(diameter, hd)
    g = np.pi * diameter ** 4 / (128.0 * mu_plasma * mu_rel * length)
    pressure = _solve(network, g, p_boundary)
    q = g * np.array([pressure[e.u] - pressure[e.v] for e in edges])
    q[dangling] = 0.0
    hd = _propagate_haematocrit(network, q, diameter, pressure, physiology.inflow_discharge_haematocrit, incident, hd)
    ht = tube_haematocrit(diameter, hd)
    area = np.pi * diameter ** 2 / 4
    velocity = q / area * 1e-3                                   # µm/s -> mm/s
    rbc_velocity = np.where(ht > 1e-6, velocity * hd / np.maximum(ht, 1e-6), velocity)
    shear = 8 * np.abs(velocity * 1e3) / diameter
    return Flow(pressure / MMHG, q * 1e-3, hd, ht, mu_rel, velocity, rbc_velocity, shear, length, diameter, iterations,
                physiology, *_pulse(network, g, p_boundary, arterial_nodes, length, diameter, dangling, physiology))


def pulse_wave(physiology: Physiology):
    """Complex amplitudes of harmonics 1..K of the arterial pressure wave, relative to the mean pressure: the
    generic wave of physiology.pulse_shape scaled so that half its pulse pressure is arterial_pulsatility."""
    t = np.linspace(0, 1, 512, endpoint=False)
    wave = np.zeros_like(t)
    for centre, width, height in physiology.pulse_shape:
        for shift in (-1, 0, 1):                            # periodic
            wave += height * np.exp(-0.5 * ((t - centre - shift) / width) ** 2)
    spectrum = np.fft.rfft(wave) / len(t) * 2
    harmonics = spectrum[1:physiology.pulse_harmonics + 1]
    k = np.arange(1, len(harmonics) + 1)
    rebuilt = np.real(np.exp(2j * np.pi * np.outer(t, k)) @ harmonics)
    return harmonics * physiology.arterial_pulsatility / max(0.5 * (rebuilt.max() - rebuilt.min()), 1e-12)


def _pulse(network, g, p_boundary, arterial_nodes, length, diameter, dangling, physiology):
    """The pulse about the mean flow, harmonic by harmonic, with distensible vessels (see the module docstring).
    Returns (flow harmonics (K, E) pl/s, pressure harmonics (K, N) mmHg, compliance (N,) µm^3/mmHg)."""
    n = len(network.nodes)
    edges = network.edges
    u = np.array([e.u for e in edges])
    v = np.array([e.v for e in edges])
    side = np.array([0 if (e.kind in ARTERIAL_KINDS or "arter" in e.kind) else
                     2 if (e.kind in VENOUS_KINDS or "vein" in e.kind or "venul" in e.kind) else 1 for e in edges])
    kappa = np.asarray(physiology.distensibility_per_mmhg)[side]              # 1/mmHg
    volume = np.pi * diameter ** 2 / 4 * length                               # µm^3
    compliance = np.zeros(n)
    np.add.at(compliance, u, 0.5 * volume * kappa)
    np.add.at(compliance, v, 0.5 * volume * kappa)
    rows = np.r_[u, v, u, v]
    cols = np.r_[u, v, v, u]
    vals = np.r_[g, g, -g, -g]
    laplacian = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()          # µm^3 / (Pa s)
    fixed = np.zeros(n, bool)
    fixed[list(p_boundary)] = True
    free = ~fixed
    harmonics = pulse_wave(physiology)
    w = 2 * np.pi * physiology.heart_rate_hz
    flows = np.zeros((len(harmonics), len(edges)), complex)
    pressures = np.zeros((len(harmonics), n), complex)
    from scipy.sparse import diags
    A_ff = laplacian[free][:, free]
    A_fx = laplacian[free][:, fixed]
    C_f = compliance[free] / MMHG                                              # µm^3 / Pa
    for k, amplitude in enumerate(harmonics, start=1):
        value = np.zeros(n, complex)
        for node in arterial_nodes:
            value[node] = amplitude * p_boundary[node]                         # Pa
        system = (A_ff + diags(1j * k * w * C_f)).tocsc()
        P = value.copy()
        if free.any():
            P[free] = spsolve(system, -A_fx @ value[fixed])
        q = g * (P[u] - P[v])
        q[dangling] = 0.0
        flows[k - 1] = q * 1e-3
        pressures[k - 1] = P / MMHG
    return flows, pressures, compliance


def _dangling(network, incident):
    """Edges of the branches that dangle from the network: stripping, again and again, every edge that ends in a
    node of degree one that is not a boundary node."""
    degree = np.array([len(m) for m in incident])
    alive = np.ones(len(network.edges), bool)
    stack = [n for n in range(len(incident)) if degree[n] == 1 and n not in network.boundary]
    while stack:
        n = stack.pop()
        for i in incident[n]:
            if not alive[i]:
                continue
            alive[i] = False
            e = network.edges[i]
            for m in (e.u, e.v):
                degree[m] -= 1
                if degree[m] == 1 and m not in network.boundary:
                    stack.append(m)
    return ~alive


def _propagate_haematocrit(network, q, diameter, pressure, hd_inflow, incident, previous):
    """Discharge haematocrit of every edge from the inflow boundaries downstream, conserving red-cell flux."""
    edges = network.edges
    E = len(edges)
    hd = np.zeros(E)
    done = np.zeros(E, bool)
    order = np.argsort(-pressure)
    eps = 1e-9 * max(np.abs(q).max(initial=0.0), 1e-12)
    for node in order:
        inflow, outflow = [], []
        for i in incident[node]:
            e = edges[i]
            into = (e.v == node and q[i] > eps) or (e.u == node and q[i] < -eps)
            out = (e.u == node and q[i] > eps) or (e.v == node and q[i] < -eps)
            if into:
                inflow.append(i)
            elif out:
                outflow.append(i)
        if not outflow:
            continue
        if node in network.boundary or not inflow:
            for i in outflow:
                hd[i] = hd_inflow
                done[i] = True
            continue
        q_in = sum(abs(q[i]) for i in inflow)
        cells_in = sum(abs(q[i]) * (hd[i] if done[i] else previous[i]) for i in inflow)
        hd_parent = cells_in / max(q_in, 1e-30)
        d_parent = float(np.sqrt(sum(diameter[i] ** 2 for i in inflow)))
        if len(outflow) == 1:
            share = [1.0]
        elif len(outflow) == 2:
            a, b = outflow
            fq = abs(q[a]) / (abs(q[a]) + abs(q[b]))
            fa = phase_separation(fq, d_parent, diameter[a], diameter[b], hd_parent)
            share = [fa, 1.0 - fa]
        else:
            # more than two daughters: split off the largest-flow daughter against the rest, repeatedly
            share = [0.0] * len(outflow)
            remaining = list(range(len(outflow)))
            left = 1.0
            while len(remaining) > 1:
                flows = np.array([abs(q[outflow[k]]) for k in remaining])
                top = remaining[int(np.argmax(flows))]
                fq = abs(q[outflow[top]]) / flows.sum()
                d_rest = float(np.sqrt(sum(diameter[outflow[k]] ** 2 for k in remaining if k != top)))
                f = phase_separation(fq, d_parent, diameter[outflow[top]], d_rest, hd_parent)
                share[top] = left * f
                left *= 1.0 - f
                remaining.remove(top)
            share[remaining[0]] = left
        for i, s in zip(outflow, share):
            hd[i] = s * cells_in / max(abs(q[i]), 1e-30)
            done[i] = True
    hd = np.clip(hd, 0.0, 0.95)
    return hd


def summary(network: Network, flow: Flow) -> dict:
    """Velocities, flows and haematocrits by vessel kind."""
    out = {}
    kinds = sorted({e.kind for e in network.edges})
    for kind in kinds:
        select = np.array([e.kind == kind for e in network.edges])
        weights = flow.length[select]
        out[kind] = {
            "diameter (um)": float(np.average(flow.diameter[select], weights=weights)),
            "|v_rbc| median (mm/s)": float(np.median(np.abs(flow.rbc_velocity[select]))),
            "|Q| median (pl/s)": float(np.median(np.abs(flow.flow[select]))),
            "H_D mean": float(np.average(flow.hd[select], weights=weights)),
            "H_T mean": float(np.average(flow.ht[select], weights=weights)),
            "wall shear rate median (1/s)": float(np.median(flow.shear_rate[select])),
        }
    out["_arterial"] = ARTERIAL_KINDS
    out["_venous"] = VENOUS_KINDS
    return out
