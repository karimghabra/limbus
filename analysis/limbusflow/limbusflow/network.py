"""End-to-end assembly: segmentation -> graph -> splines -> morphometry ->
velocity -> directed flow graph.

Directed flow graph
-------------------
Each vessel segment is an edge between two nodes. Its measured velocity sign
(relative to the spline's parameter direction u -> v) orients the edge along
the flow. Volumetric flow is estimated as

    Q = v_mean * pi d^2 / 4,      v_mean = v_RBC / 1.6

(the factor 1.6 converts the centre-line red-cell velocity to the mean
velocity over the lumen, Baker & Wayland 1974 / Pries et al.). At a true
junction blood is conserved (Kirchhoff's current law):

    sum_{in} Q  -  sum_{out} Q  =  0

which is used (a) to infer the direction of vessels whose velocity could not
be measured and (b) as a physical consistency check (the residual).

Node types after orientation:
    divergent   1 in -> 2+ out   (arteriolar branching)
    convergent  2+ in -> 1 out   (venular confluence)
    inlet / outlet     border nodes where blood enters / leaves the field of view
    source / sink      interior nodes with only outflow / only inflow (inconsistent
                       or vessel leaves the focal plane)
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field

import cv2
import networkx as nx
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from . import graph as gr
from . import morphometry as mo
from . import perfusion as pf
from . import spline as spl
from . import velocity as ve
from . import vesselness as vs


@dataclass
class Params:
    # enhancement / segmentation
    flatfield_sigma: float = 30.0
    sigmas: tuple = (1.5, 2, 3, 4, 6, 8)          # structural-only mode
    frangi_beta: float = 0.5
    frangi_c: float = 0.02
    seg_low_pct: float = 80
    seg_high_pct: float = 94
    seg_min_size: int = 300
    valid_erode: int = 12
    # functional (flicker) detection - finds thin perfused vessels
    use_flicker: bool = True
    flicker_t_window: float = 10.0
    flicker_min_cover: float = 0.3
    sigmas_struct: tuple = (1, 1.5, 2, 3, 4, 6, 8)
    sigmas_func: tuple = (1, 1.5, 2, 3, 4, 6)
    comb_low_pct: float = 75
    comb_high_pct: float = 92
    comb_min_size: int = 60
    spur_radius_factor: float = 2.0
    # graph
    spur_len: float = 15.0
    junction_merge: float = 10.0
    crossing_dev: float = 35.0
    min_vessel_len: float = 20.0
    border_margin: int = 25
    # spline / diameter
    spline_sigma: float = 0.8
    profile_avg: int = 3
    min_len_measure: float = 15.0
    # velocity
    lumen_frac: float = 0.5
    t_window: float = 10.0
    highpass_s: float = 10.0
    max_lag: int = 15
    min_quality: float = 0.10
    min_r2: float = 0.8
    agree_tol: float = 0.35
    vt_window: int = 24
    rbc_to_mean: float = 1.6
    time_scales: tuple = (1, 3, 9)            # temporal binning factors (slow flow)
    slow_min_quality: float = 0.15


@dataclass
class Vessel:
    id: str
    edge: int
    u: int
    v: int
    spline: spl.VesselSpline
    d_guess: float
    diam: dict = field(default_factory=dict)
    d_smooth: np.ndarray | None = None
    polygon: np.ndarray | None = None
    tort: dict = field(default_factory=dict)
    crossings: list = field(default_factory=list)
    K: np.ndarray | None = None
    vel: dict = field(default_factory=dict)
    flow_sign: int = 0              # +1 u->v, -1 v->u, 0 unknown
    direction_source: str = "unknown"

    # --- convenience -------------------------------------------------------
    @property
    def length(self):
        return self.spline.length

    @property
    def d_median(self):
        m = self.diam.get("valid")
        if m is None or not m.any():
            return float(np.nanmedian(self.d_smooth)) if self.d_smooth is not None else np.nan
        return float(np.nanmedian(self.diam["d"][m]))

    @property
    def v_rbc(self):
        """Signed RBC velocity along u->v (px / frame), best estimate."""
        return self.vel.get("v", np.nan)

    @property
    def speed(self):
        return abs(self.v_rbc) if np.isfinite(self.v_rbc) else np.nan

    def flow_Q(self, rbc_to_mean=1.6):
        """Volumetric flow magnitude in px^3 / frame."""
        return self.speed / rbc_to_mean * np.pi * self.d_median**2 / 4


@dataclass
class Network:
    ref: np.ndarray
    valid: np.ndarray
    params: Params
    ff: np.ndarray = None
    V: np.ndarray = None
    best_sigma: np.ndarray = None
    mask: np.ndarray = None
    skel: np.ndarray = None
    graph: gr.VesselGraph = None
    vessels: dict = field(default_factory=dict)
    fps: float = 1.0
    um_per_px: float | None = None
    frame_idx: np.ndarray | None = None
    dg: nx.MultiDiGraph | None = None
    frangi_stack: dict | None = None

    # ------------------------------------------------------------------ steps
    def compute_flicker(self, frames, reg, idx, progress=True):
        """Temporal-flicker (perfusion) map of the registered video window."""
        p = self.params
        t0 = __import__("time").time()
        stack = pf.registered_stack(frames, reg, idx)
        self.fl, self.fl_parts = pf.flicker_map(stack, p.flicker_t_window, min_cover=p.flicker_min_cover,
                                                return_parts=True)
        del stack
        if progress:
            print(f"flicker map from {len(list(idx))} frames: {__import__('time').time() - t0:.0f} s; "
                  f"residual jitter sigma = {np.sqrt(max(self.fl_parts['coef']['sigma_delta2'], 0)):.2f} px")
        return self

    def enhance(self, keep_stack=False):
        p = self.params
        self.ff = vs.flatfield(self.ref, p.flatfield_sigma)
        self.valid_e = ndi.binary_erosion(self.valid, iterations=p.valid_erode)
        if p.use_flicker and getattr(self, "fl", None) is not None:
            cover_ok = ndi.binary_erosion(self.fl_parts["cover"] >= p.flicker_min_cover, iterations=6)
            self.valid_e = self.valid_e & cover_ok
            self.V, self.V_struct, self.V_func, self.thick = pf.combined_vesselness(
                self.ff, self.fl, self.valid_e, p.sigmas_struct, p.sigmas_func, p.frangi_beta)
            self.best_sigma = None
            if keep_stack:
                _, _, self.frangi_stack = vs.frangi(self.ff, p.sigmas, p.frangi_beta, p.frangi_c, return_all=True)
            return self
        if keep_stack:
            self.V, self.best_sigma, self.frangi_stack = vs.frangi(self.ff, p.sigmas, p.frangi_beta, p.frangi_c,
                                                                   return_all=True)
        else:
            self.V, self.best_sigma = vs.frangi(self.ff, p.sigmas, p.frangi_beta, p.frangi_c)
        return self

    def segment(self):
        p = self.params
        if not hasattr(self, "valid_e"):
            self.valid_e = ndi.binary_erosion(self.valid, iterations=p.valid_erode)
        comb = p.use_flicker and getattr(self, "fl", None) is not None
        self.mask, self.seg_thresholds = vs.segment(self.V, self.valid_e,
                                                    p.comb_low_pct if comb else p.seg_low_pct,
                                                    p.comb_high_pct if comb else p.seg_high_pct,
                                                    p.comb_min_size if comb else p.seg_min_size)
        self.skel = vs.skeletonize(self.mask)
        return self

    def build_graph(self):
        p = self.params
        self.graph = gr.build_graph(self.skel, self.valid, p.spur_len, p.junction_merge, p.crossing_dev,
                                    p.min_vessel_len, p.border_margin, radius=ndi.distance_transform_edt(self.mask),
                                    spur_radius_factor=p.spur_radius_factor)
        return self

    def fit_splines(self):
        p = self.params
        dist = ndi.distance_transform_edt(self.mask)
        H, W = self.mask.shape
        items = sorted(self.graph.edges.items(), key=lambda kv: -gr.VesselGraph.length(kv[1]["path"]))
        self.vessels = {}
        k = 0
        for e, d in items:
            if gr.VesselGraph.length(d["path"]) < p.min_len_measure:
                continue
            sp = spl.fit_spline(d["path"], sigma=p.spline_sigma)
            xi = np.clip(np.round(sp.xy[:, 0]).astype(int), 0, W - 1)
            yi = np.clip(np.round(sp.xy[:, 1]).astype(int), 0, H - 1)
            dgs = 2 * float(np.median(dist[yi, xi]))
            k += 1
            vid = f"V{k:02d}"
            self.vessels[vid] = Vessel(id=vid, edge=e, u=d["u"], v=d["v"], spline=sp, d_guess=max(dgs, 3.0),
                                       crossings=list(d.get("crossings", [])))
        return self

    def measure_morphology(self, progress=True):
        p = self.params
        junc = [tuple(n["pos"]) for n in self.graph.nodes.values() if n["kind"] == "junction"]
        for n, (vid, ves) in enumerate(self.vessels.items()):
            excl = [pt for pt in junc] + list(ves.crossings)
            ves.diam = mo.measure_diameter(self.ff, ves.spline, ves.d_guess, avg=p.profile_avg, exclude_pts=excl,
                                           exclude_r=0.8 * ves.d_guess + 3)
            ves.d_smooth = mo.smooth_diameter(ves.diam["d"], ves.diam["valid"])
            off = np.where(ves.diam["valid"], ves.diam["center_offset"], 0.0)
            off = ndi.uniform_filter1d(ndi.median_filter(off, 9, mode="nearest"), 9, mode="nearest")
            ves.polygon = mo.vessel_polygon(ves.spline, ves.d_smooth, off)
            ves.tort = mo.tortuosity(ves.spline)
            if progress:
                print(f"\rmorphology {n + 1}/{len(self.vessels)}", end="", flush=True)
        if progress:
            print()
        return self

    def measure_velocity(self, frames, reg, idx, fps, progress=True):
        p = self.params
        self.fps = fps
        self.frame_idx = np.asarray(idx)
        pts = [ve.lumen_points(v.spline, v.d_median if np.isfinite(v.d_median) else v.d_guess, p.lumen_frac)
               for v in self.vessels.values()]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            Ks = ve.kymographs(frames, reg, idx, pts, progress=progress)
        for (vid, ves), K in zip(self.vessels.items(), Ks):
            ves.K = K
            ves.vel = self.velocity_for(K)
        return self

    def velocity_for(self, K):
        """Multi-scale velocity (see velocity.estimate)."""
        p = self.params
        return ve.estimate(K, None, self.fps, scales=p.time_scales, t_window=p.t_window, highpass_s=p.highpass_s,
                           max_lag=p.max_lag, vt_window=p.vt_window, min_quality=p.min_quality, min_r2=p.min_r2,
                           agree_tol=p.agree_tol, slow_min_quality=p.slow_min_quality)

    def velocity_for_single_scale(self, K):
        """Original single-scale estimator (kept for comparison in the notebook)."""
        p = self.params
        K2, W = ve.preprocess(K, p.t_window, p.highpass_s)
        Cm, shifts = ve.correlation_map(K2, p.max_lag)
        rid = ve.ridge_velocity(Cm, shifts)
        st = ve.structure_tensor_velocity(K2, rid["v"] if np.isfinite(rid["v"]) else None)
        tof = ve.time_of_flight(K2)
        v = rid["v"]
        # primary estimate: robust time average = median of the sliding-window
        # LSPIV trace, at a lag chosen so the displacement is ~8 px. (The
        # whole-window ridge fit is biased towards slow periods at long lags,
        # because fast patterns decorrelate sooner.)
        lag = ve.best_lag(v)
        ts = ve.lspiv(K2, lag, window=p.vt_window, step=2)
        t_s = ts["t_center"] / self.fps
        v_t = np.where(ts["peak_t"] > 0.05, ts["v_t"], np.nan)
        pul = ve.pulsatility(t_s, v_t)
        vt_ok = np.isfinite(v_t) & (np.sign(v_t) == np.sign(v)) if np.isfinite(v) else np.zeros(len(v_t), bool)
        v_time = float(np.median(v_t[vt_ok])) if vt_ok.sum() >= 5 else v
        # reliability: the ridge must be clear, and at least two of the three
        # independent estimators (ridge fit, structure tensor, time of flight)
        # must agree with the primary value in sign and within agree_tol
        agree = [m for m in (v, st["v"], tof["v"]) if np.isfinite(m) and np.isfinite(v_time)
                 and np.sign(m) == np.sign(v_time) and abs(m - v_time) <= p.agree_tol * abs(v_time) + 0.5]
        reliable = bool(np.isfinite(v_time) and rid["quality"] >= p.min_quality and rid["r2"] >= p.min_r2
                        and len(agree) >= 2)
        return dict(v=v_time if reliable else np.nan, v_time=v_time, v_ridge=v, v_st=st["v"], v_tof=tof["v"], quality=rid["quality"],
                    r2=rid["r2"], reliable=reliable, n_agree=len(agree), lag=lag, t_s=t_s, v_t=v_t,
                    hr_bpm=pul["hr_bpm"], PI=pul["PI"], ridge=rid, Cmap=Cm, shifts=shifts, tof=tof,
                    st_map=st["v_map"], coherence=st["coherence"], K2=K2)

    # ------------------------------------------------------------------ flow graph
    def orient(self, max_iter=20):
        """Measured directions first, then fill gaps with mass conservation."""
        p = self.params
        for ves in self.vessels.values():
            if ves.vel.get("reliable"):
                ves.flow_sign = int(np.sign(ves.vel["v"]))
                ves.direction_source = "measured"
            else:
                ves.flow_sign = 0
                ves.direction_source = "unknown"
        junctions = [n for n, d in self.graph.nodes.items() if d["kind"] == "junction"]
        inc = {n: [v for v in self.vessels.values() if n in (v.u, v.v) and v.u != v.v] for n in junctions}
        for _ in range(max_iter):
            changed = False
            for n in junctions:
                vs_ = inc[n]
                unknown = [v for v in vs_ if v.flow_sign == 0]
                if len(unknown) != 1 or len(vs_) < 3:
                    continue
                net_in = 0.0
                for v in vs_:
                    if v.flow_sign == 0:
                        continue
                    into_n = (v.v == n and v.flow_sign > 0) or (v.u == n and v.flow_sign < 0)
                    net_in += v.flow_Q(p.rbc_to_mean) * (1 if into_n else -1)
                if abs(net_in) < 1e-9:
                    continue
                x = unknown[0]
                # if more flows in than out, the unknown vessel must carry flow away from n
                away = net_in > 0
                x.flow_sign = (1 if x.u == n else -1) if away else (1 if x.v == n else -1)
                x.direction_source = "inferred (conservation)"
                x.vel["Q_inferred"] = abs(net_in)
                changed = True
            if not changed:
                break
        self._build_digraph()
        return self

    def _build_digraph(self):
        p = self.params
        D = nx.MultiDiGraph()
        for n, d in self.graph.nodes.items():
            D.add_node(n, x=float(d["pos"][0]), y=float(d["pos"][1]), kind=d["kind"])
        for vid, ves in self.vessels.items():
            if ves.flow_sign >= 0:
                a, b, sp = ves.u, ves.v, ves.spline
            else:
                a, b, sp = ves.v, ves.u, ves.spline.reversed()
            Q = ves.flow_Q(p.rbc_to_mean)
            if not np.isfinite(Q):
                Q = ves.vel.get("Q_inferred", np.nan)
            D.add_edge(a, b, key=vid, vessel=vid, direction=ves.direction_source, known=ves.flow_sign != 0,
                       length_px=ves.length, d_px=ves.d_median, speed_px_per_frame=ves.speed,
                       speed_px_per_s=ves.speed * self.fps, Q_px3_per_s=Q * self.fps,
                       DM=ves.tort.get("DM"), SOAM=ves.tort.get("SOAM"),
                       xy=sp.xy[::max(1, len(sp.xy) // 60)].round(2).tolist())
        # node classification
        for n in D.nodes:
            ins = [e for e in D.in_edges(n, keys=True, data=True) if e[3]["known"]]
            outs = [e for e in D.out_edges(n, keys=True, data=True) if e[3]["known"]]
            deg = D.degree(n)
            kind = D.nodes[n]["kind"]
            unknown = deg - len(ins) - len(outs)
            if kind == "border" or (kind == "end" and deg == 1):
                label = ("inlet" if outs else "outlet" if ins else "terminal (unknown)") + ("" if kind == "border" else " (in-plane end)")
            elif unknown:
                label = "partially resolved"
            elif len(ins) == 1 and len(outs) >= 2:
                label = "divergent"
            elif len(ins) >= 2 and len(outs) == 1:
                label = "convergent"
            elif ins and not outs:
                label = "sink"
            elif outs and not ins:
                label = "source"
            else:
                label = "through"
            qin = np.nansum([e[3]["Q_px3_per_s"] for e in ins]); qout = np.nansum([e[3]["Q_px3_per_s"] for e in outs])
            D.nodes[n]["flow_type"] = label
            D.nodes[n]["Q_in"] = float(qin); D.nodes[n]["Q_out"] = float(qout)
            all_measured = all(e[3]["direction"] == "measured" for e in ins + outs)
            D.nodes[n]["conservation_residual"] = (float((qin - qout) / max(qin, qout))
                                                   if kind == "junction" and not unknown and all_measured
                                                   and max(qin, qout) > 0 else np.nan)
        self.dg = D

    # ------------------------------------------------------------------ outputs
    def label_image(self):
        """int image: 0 background, k = k-th vessel (order of self.vessels)."""
        lab = np.zeros(self.ref.shape, np.int32)
        for k, ves in enumerate(self.vessels.values(), start=1):
            m = mo.rasterize(ves.polygon, self.ref.shape, sp=ves.spline, d_smooth=ves.d_smooth)
            lab[m & (lab == 0)] = k
        return lab

    def vessel_mask(self, vid):
        ves = self.vessels[vid]
        return mo.rasterize(ves.polygon, self.ref.shape, sp=ves.spline, d_smooth=ves.d_smooth)

    def table(self):
        rows = []
        f = self.um_per_px
        for vid, v in self.vessels.items():
            r = dict(vessel=vid, u=v.u, v=v.v, length_px=v.length, d_median_px=v.d_median,
                     d_boxfit_px=float(np.nanmedian(v.diam["d_fit"][v.diam["valid"]])) if v.diam["valid"].any() else np.nan,
                     subresolved_frac=float(np.mean(v.diam["subres"][v.diam["valid"]])) if v.diam["valid"].any() else np.nan,
                     d_min_px=float(np.nanpercentile(v.diam["d"][v.diam["valid"]], 5)) if v.diam["valid"].any() else np.nan,
                     d_max_px=float(np.nanpercentile(v.diam["d"][v.diam["valid"]], 95)) if v.diam["valid"].any() else np.nan,
                     DM=v.tort["DM"], SOAM=v.tort["SOAM"], KSQ=v.tort["KSQ"], n_inflections=v.tort["n_inflections"],
                     ICM=v.tort["ICM"], v_ridge_px_f=v.vel.get("v_ridge"), v_st_px_f=v.vel.get("v_st"),
                     v_tof_px_f=v.vel.get("v_tof"), velocity_quality=v.vel.get("quality"),
                     velocity_reliable=v.vel.get("reliable"), time_scale=v.vel.get("scale", 1), v_time_px_f=v.vel.get("v_time"), speed_px_s=v.speed * self.fps,
                     Q_px3_s=v.flow_Q(self.params.rbc_to_mean) * self.fps, dominant_freq_bpm=v.vel.get("hr_bpm"),
                     PI=v.vel.get("PI"), direction=v.direction_source, n_crossings=len(v.crossings))
            if f:
                r.update(length_um=v.length * f, d_median_um=v.d_median * f, speed_mm_s=v.speed * self.fps * f / 1000,
                         Q_nl_s=r["Q_px3_s"] * f**3 * 1e-6)
            rows.append(r)
        return pd.DataFrame(rows).set_index("vessel")

    def export(self, prefix):
        """Write <prefix>_graph.json (node-link, with splines), .graphml and _vessels.csv."""
        D = self.dg
        data = nx.node_link_data(D, edges="links")
        for vid, ves in self.vessels.items():
            for l in data["links"]:
                if l["key"] == vid:
                    t, c, k = ves.spline.tck
                    l["spline"] = dict(knots=np.asarray(t).tolist(), coeffs=[np.asarray(ci).tolist() for ci in c],
                                       degree=int(k), orientation="flow" if ves.flow_sign >= 0 else "reversed")
                    l["diameter_profile_px"] = dict(s=ves.spline.s[::2].round(2).tolist(),
                                                    d=np.round(ves.d_smooth[::2], 2).tolist())
                    l["outline_px"] = ves.polygon[::2].round(1).tolist()
        meta = dict(fps=self.fps, um_per_px=self.um_per_px, frames=self.frame_idx.tolist() if self.frame_idx is not None else None,
                    rbc_to_mean=self.params.rbc_to_mean,
                    units=dict(length="px", velocity="px/s", flow="px^3/s"))
        data["graph"] = meta

        def clean(o):
            if isinstance(o, float) and not np.isfinite(o):
                return None
            if isinstance(o, (np.floating,)):
                return None if not np.isfinite(o) else float(o)
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, np.bool_):
                return bool(o)
            if isinstance(o, dict):
                return {k: clean(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [clean(v) for v in o]
            return o
        with open(prefix + "_graph.json", "w") as fh:
            json.dump(clean(data), fh)
        G2 = nx.MultiDiGraph()
        for n, d in D.nodes(data=True):
            G2.add_node(n, **{k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in d.items()})
        for a, b, k, d in D.edges(keys=True, data=True):
            G2.add_edge(a, b, key=k, **{kk: (json.dumps(vv) if isinstance(vv, list) else
                                            (float(vv) if isinstance(vv, (float, np.floating)) else vv))
                                        for kk, vv in d.items()})
        nx.write_graphml(G2, prefix + ".graphml")
        self.table().to_csv(prefix + "_vessels.csv")
        return [prefix + "_graph.json", prefix + ".graphml", prefix + "_vessels.csv"]


def run_all(ref, valid, frames, reg, idx, fps, params=None, um_per_px=None, progress=True):
    net = Network(ref=ref, valid=valid, params=params or Params(), um_per_px=um_per_px)
    if net.params.use_flicker:
        net.compute_flicker(frames, reg, idx, progress)
    net.enhance().segment().build_graph().fit_splines().measure_morphology(progress)
    net.measure_velocity(frames, reg, idx, fps, progress)
    net.orient()
    return net
