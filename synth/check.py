"""The anatomy check: every structure and haemodynamic quantity of a generated network against its cited range.

anatomy_report(scene, network, flow) measures each quantity on the ground truth (not on the images) and compares
it with the range given in docs/synthetic_conjunctiva_anatomy.md, whose reference keys are quoted in "source".
status is "within" or "outside"; quantities the literature reports without a usable range are listed as
"reference" with the published value, for the reader to compare.

Ranges are the published range where one is given, and otherwise the published mean plus or minus two standard
deviations (so a typical healthy subject lies inside); the docs say which, row by row.
"""
from __future__ import annotations

import numpy as np

from .network import Network, _vessel_runs

# name: (low, high, unit, source)
RANGES = {
    "conjunctival vessel depth": (42.0, 240.0, "µm", "epithelium 42 µm, conjunctiva 240 µm thick [Zhang2013]; 0-200 µm slab [Akagi2018]"),
    "episcleral vessel depth": (200.0, 470.0, "µm", "under conjunctiva and Tenon's [Zhang2013, FernandezVigo2021, Kumar2013]; 200-1000 µm slab [Akagi2018]"),
    "deep vessel depth": (450.0, 1000.0, "µm", "intrascleral, 200-1000 µm slab [Akagi2018]; deep scleral vessels at ~500 µm [Bizheva2011]"),
    "capillary diameter": (4.0, 9.0, "µm", "capillaries < 9 µm [Moka2020, Koutsiaris2015]; 4.3 ± 0.9 µm retina [Neriyanuri2023]"),
    "arteriole diameter": (5.9, 61.0, "µm", "range 5.9-42.9 [Khansari2016], 6-61 µm [Khansari2017]"),
    "venule diameter": (6.0, 75.0, "µm", "range 6.0-51.6 [Khansari2016], 6-75 µm [Khansari2017]"),
    "arteriole mean diameter (segments >= 50 µm)": (9.0, 28.0, "µm", "15 ± 3 [Khansari2016], 18 ± 5 µm [Khansari2017], mean ± 2 SD"),
    "venule mean diameter (segments >= 50 µm)": (14.0, 47.0, "µm", "18 ± 2 [Khansari2016], 20 ± 2 [Khansari2017], 30.4 ± 8.4 µm near the limbus [Wang2016]"),
    "episcleral vein diameter": (50.0, 100.0, "µm", "50-100 µm [Asanad2021]; 61 ± 5 µm [Kim2024]"),
    "episcleral artery diameter": (40.0, 97.0, "µm", "episcleral vessels 79 ± 9 µm [MacKenzie2016], mean ± 2 SD, lower end at conjunctival arteriole maxima"),
    "deep vessel diameter": (20.0, 50.0, "µm", "deep scleral vessels 20-50 µm [Bizheva2011]"),
    "communicating vessel diameter": (10.0, 20.0, "µm", "radial channels 10-20 µm [Bizheva2011]"),
    "arteriolar junction exponent (median)": (2.59, 3.14, "", "retinal arteries 2.59 [Luo2017]; Q ~ D^3.01 (2.86-3.14) [MayrovitzRoy1983]; Murray 3 [Murray1926a]"),
    "venular junction exponent (median)": (1.95, 3.14, "", "retinal veins 1.95 [Luo2017] to Murray's 3 [Murray1926a, MayrovitzRoy1983]"),
    "capillary length density": (4.4, 12.0, "mm/mm2", "5.2 ± 1.2 capillaries per mm [Kvernebo2022], x pi/2 for length per area, mean ± 2 SD"),
    "episcleral arteriovenous anastomoses": (0.0, 1.5, "/mm2", "0.5-1 per mm2 in human episclera [Selbach2005]; a field may hold none"),
    "capillary red-cell velocity (median)": (0.29, 1.5, "mm/s", "conjunctival capillaries 0.51 mm/s [Moka2020]; nailfold 0.29-1.09 [Hahn2004]"),
    "arteriolar red-cell velocity (median, >= 6 µm, >= 50 µm long)": (0.4, 1.7, "mm/s", "0.63 ± 0.17 [Khansari2016], 0.70 ± 0.23 [Khansari2017]; precapillary 1.66 [Koutsiaris2010]"),
    "venular red-cell velocity (median, >= 6 µm, >= 50 µm long)": (0.25, 1.0, "mm/s", "0.54 ± 0.13 [Khansari2016], 0.59 ± 0.17 [Khansari2017], 0.50 ± 0.14 [Hu2018]"),
    "arteriolar flow (median, >= 11 µm)": (20.0, 300.0, "pl/s", "86 ± 33 [Khansari2016], 144 pl/s [Khansari2017]; 40 pl/s below 11 µm, 299 above 22 µm [Khansari2016]"),
    "venular flow (median, >= 11 µm)": (5.0, 462.0, "pl/s", "140 ± 55 [Khansari2016], 175 [Khansari2017]; 5-462 pl/s [Koutsiaris2007]"),
    "arteriolar wall shear rate (median)": (150.0, 900.0, "1/s", "320 ± 132 [Khansari2016]; 488 1/s below 11 µm [Khansari2016]"),
    "venular wall shear rate (median)": (30.0, 400.0, "1/s", "190 ± 46 1/s [Khansari2016]; lower in small venules [Koutsiaris2007]"),
    "capillary tube haematocrit (median)": (0.10, 0.45, "", "10-18 % [Klitzman1979]; length fraction 0.41 in retina [Neriyanuri2023]"),
    "capillary pressure (median)": (12.0, 35.0, "mmHg", "13-16 mmHg nailfold [Williams1990, Hahn2004]; 30-35 mmHg after precapillary branches [Zweifach1974]"),
    "capillary linear cell density (median, <= 6 µm)": (23.7, 93.7, "cells/mm", "48 ± 15, range 23.7-93.7 cells/mm in 4.3 ± 0.9 µm retinal capillaries [Neriyanuri2023]; 55.9 ± 19.3 [Ding2025]"),
    "precapillary arteriolar pulsatility index": (0.3, 1.3, "", "PI about 0.8, RI 53 % in 6-12 µm arterioles [Koutsiaris2010]"),
    "venular pulsatility index (>= 15 µm)": (0.0, 0.4, "", "venous flow pulses far less than arterial [Puyo2019]"),
    "capillary pulse pressure (median)": (0.75, 4.5, "mmHg", "1.5-4 mmHg [Zweifach1974, Hahn2004]; the bounds allow for the spread between single networks"),
    "episcleral vein flow (median)": (0.0, 0.84, "µl/min", "0.40 ± 0.22 µl/min in 61 µm episcleral veins [Kim2024], mean ± 2 SD"),
    "companion venule calibre over its arteriole's (median)": (1.27, 1.93, "", "artery:vein calibre 0.653 ± 0.068 [Yusufu2025], mean ± 2 SD"),
    "calibre variation along vessels >= 15 µm (CV, median)": (0.05, 0.4, "", "about 0.1 above 15 µm, up to 0.4 below [Kiani1993]; ~0.19 in conjunctiva [Shahidi2010]"),
    "communicating vessel angle to the surface": (15.0, 75.0, "deg", "oblique course, no measured angle [Asanad2021, Li2011]; docs §2.4"),
}

REFERENCE_ONLY = {
    "red-cell velocity in arteriovenous anastomoses and communicating vessels (median)": ("no published value", "docs §2.4"),
    "venular velocity by diameter, 4-9 / 9-14 / 14-19 / 19-24 µm": ("0.48 / 0.64 / 0.82 / 0.88 mm/s", "[Koutsiaris2015]"),
    "fractal dimension of the projected network": ("1.59-1.63 (slit lamp), 1.28-1.33 (OCTA)", "[Jiang2014, Shi2018, Akagi2018, Liu2019]"),
    "bifurcation angle (daughter to daughter)": ("78° in retinal arteries", "[Luo2017]"),
}


def _junction_exponents(network: Network, kinds):
    """Exponent g solving d0^g = d1^g + d2^g at every three-way junction of the given kinds."""
    incident = network.incident()
    out = []
    for node, edges in enumerate(incident):
        if len(edges) != 3 or node in network.boundary:
            continue
        if not all(network.edges[i].kind in kinds for i in edges):
            continue
        d = sorted((network.edges[i].diameter for i in edges), reverse=True)
        d0, d1, d2 = d
        if d0 >= d1 + d2 or d0 <= d1:
            continue
        low, high = 1.0, 8.0
        for _ in range(50):
            mid = 0.5 * (low + high)
            if d1 ** mid + d2 ** mid > d0 ** mid:
                low = mid
            else:
                high = mid
        out.append(0.5 * (low + high))
    return np.array(out)


def _angles(network: Network, kind_prefix):
    out = []
    for e in network.edges:
        if e.kind.startswith(kind_prefix):
            horizontal = np.linalg.norm(e.points[-1, :2] - e.points[0, :2])
            vertical = abs(e.points[-1, 2] - e.points[0, 2])
            out.append(np.degrees(np.arctan2(vertical, max(horizontal, 1e-9))))
    return np.array(out)


def measure(scene, network: Network, flow) -> dict:
    """The checked quantities of one network (ground truth)."""
    kind = np.array([e.kind for e in network.edges])
    layer = np.array([e.layer for e in network.edges])
    d = flow.diameter
    L = flow.length
    v = np.abs(flow.rbc_velocity)
    q = np.abs(flow.flow)
    x0, x1, y0, y1 = network.domain
    area_mm2 = (x1 - x0) * (y1 - y0) * 1e-6

    def depths(select):
        z = np.concatenate([network.edges[i].points[:, 2] for i in np.flatnonzero(select)]) if select.any() else np.array([np.nan])
        return float(np.percentile(z, 1)), float(np.percentile(z, 99))

    def wmean(values, select):
        return float(np.average(values[select], weights=L[select])) if select.any() else float("nan")

    art = kind == "arteriole"
    ven = kind == "venule"
    cap = kind == "capillary"
    long_ = L >= 50.0
    m = {}
    m["conjunctival vessel depth"] = depths(layer == "conjunctival")
    m["episcleral vessel depth"] = depths(layer == "episcleral")
    m["deep vessel depth"] = depths(layer == "deep")
    m["capillary diameter"] = (float(d[cap].min()), float(d[cap].max())) if cap.any() else (np.nan, np.nan)
    m["arteriole diameter"] = (float(d[art].min()), float(d[art].max()))
    m["venule diameter"] = (float(d[ven].min()), float(d[ven].max()))
    m["arteriole mean diameter (segments >= 50 µm)"] = wmean(d, art & long_)
    m["venule mean diameter (segments >= 50 µm)"] = wmean(d, ven & long_)
    for name, k in (("episcleral vein diameter", "episcleral vein"), ("episcleral artery diameter", "episcleral artery")):
        sel = kind == k
        m[name] = (float(d[sel].min()), float(d[sel].max())) if sel.any() else (np.nan, np.nan)
    sel = layer == "deep"
    m["deep vessel diameter"] = (float(d[sel].min()), float(d[sel].max())) if sel.any() else (np.nan, np.nan)
    sel = layer == "communicating"
    m["communicating vessel diameter"] = (float(d[sel].min()), float(d[sel].max())) if sel.any() else (np.nan, np.nan)
    m["arteriolar junction exponent (median)"] = float(np.median(_junction_exponents(network, ("arteriole",))))
    m["venular junction exponent (median)"] = float(np.median(_junction_exponents(network, ("venule",))))
    m["capillary length density"] = float(L[cap].sum() * 1e-3 / area_mm2)
    m["episcleral arteriovenous anastomoses"] = float((kind == "arteriovenous anastomosis").sum() / area_mm2)
    m["capillary red-cell velocity (median)"] = float(np.median(v[cap]))
    m["arteriolar red-cell velocity (median, >= 6 µm, >= 50 µm long)"] = float(np.median(v[art & long_ & (d >= 6)]))
    m["venular red-cell velocity (median, >= 6 µm, >= 50 µm long)"] = float(np.median(v[ven & long_ & (d >= 6)]))
    m["arteriolar flow (median, >= 11 µm)"] = float(np.median(q[art & (d >= 11)]))
    m["venular flow (median, >= 11 µm)"] = float(np.median(q[ven & (d >= 11)]))
    m["arteriolar wall shear rate (median)"] = float(np.median(flow.shear_rate[art]))
    m["venular wall shear rate (median)"] = float(np.median(flow.shear_rate[ven]))
    m["capillary tube haematocrit (median)"] = float(np.median(flow.ht[cap]))
    cap_nodes = sorted({k for i in np.flatnonzero(cap) for k in (network.edges[i].u, network.edges[i].v)})
    m["capillary pressure (median)"] = float(np.median(flow.pressure[cap_nodes]))
    # single-file capillaries: cells per mm from the tube haematocrit, N / L = H_T A / V_rbc, in capillaries of the
    # calibre of the retinal capillaries the cited counts come from (4.3 ± 0.9 µm, so up to 6 µm)
    area = np.pi * d ** 2 / 4
    narrow = cap & (d <= 6.0)
    m["capillary linear cell density (median, <= 6 µm)"] = float(np.median(flow.ht[narrow] * area[narrow] / scene.physiology.rbc_volume_um3 * 1e3)) if narrow.any() else float("nan")
    # pulsatility index (v_max - v_min) / v_mean over the cardiac cycle in 6-12 µm arterioles
    t = np.linspace(0, 1.0 / scene.physiology.heart_rate_hz, 24, endpoint=False)
    pre = art & (d >= 6) & (d <= 12) & (np.abs(flow.flow) > 0)
    if pre.any():
        series = np.abs(np.stack([flow.flow_at(tt)[pre] for tt in t]))
        pi = (series.max(0) - series.min(0)) / np.maximum(series.mean(0), 1e-12)
        m["precapillary arteriolar pulsatility index"] = float(np.median(pi))
    else:
        m["precapillary arteriolar pulsatility index"] = float("nan")
    post = ven & (d >= 15) & (np.abs(flow.flow) > 0)
    if post.any():
        series = np.abs(np.stack([flow.flow_at(tt)[post] for tt in t]))
        m["venular pulsatility index (>= 15 µm)"] = float(np.median((series.max(0) - series.min(0)) / np.maximum(series.mean(0), 1e-12)))
    else:
        m["venular pulsatility index (>= 15 µm)"] = float("nan")
    pressures = np.stack([flow.pressure_at(tt)[cap_nodes] for tt in t]) if cap_nodes else np.zeros((1, 0))
    m["capillary pulse pressure (median)"] = float(np.median(pressures.max(0) - pressures.min(0))) if cap_nodes else float("nan")
    angles = _angles(network, "communicating")
    m["communicating vessel angle to the surface"] = (float(angles.min()), float(angles.max())) if len(angles) else (np.nan, np.nan)
    sel = kind == "episcleral vein"
    m["episcleral vein flow (median)"] = float(np.median(q[sel]) * 60e-6) if sel.any() else float("nan")     # pl/s -> µl/min
    shunt = (kind == "arteriovenous anastomosis") | (layer == "communicating")
    m["red-cell velocity in arteriovenous anastomoses and communicating vessels (median)"] = (
        f"{np.median(v[shunt]):.2f} mm/s" if shunt.any() else "-")
    ratios = []
    arteriole_d = {}
    for e in network.edges:
        if e.kind == "arteriole":
            arteriole_d.setdefault(e.vessel, []).append(e.diameter)
    for e in network.edges:
        if e.paired_with >= 0 and e.paired_with in arteriole_d:
            ratios.append(e.diameter / np.median(arteriole_d[e.paired_with]))
    m["companion venule calibre over its arteriole's (median)"] = float(np.median(ratios)) if ratios else float("nan")
    cvs = []
    for run in _vessel_runs(network.edges):            # along each named vessel, through the junctions it passes
        local = np.concatenate([network.edges[i].local_diameter for i, _ in run])
        if max(network.edges[i].diameter for i, _ in run) >= 15.0 and len(local) > 3:
            cvs.append(float(np.std(local) / np.mean(local)))
    m["calibre variation along vessels >= 15 µm (CV, median)"] = float(np.median(cvs)) if cvs else float("nan")
    by_size = []
    for lo, hi in ((4, 9), (9, 14), (14, 19), (19, 24)):
        sel = ven & (d >= lo) & (d < hi)
        by_size.append(f"{np.median(v[sel]):.2f}" if sel.any() else "-")
    m["venular velocity by diameter, 4-9 / 9-14 / 14-19 / 19-24 µm"] = " / ".join(by_size) + " mm/s"
    m["bifurcation angle (daughter to daughter)"] = "measured on the images (synth.targets)"
    m["fractal dimension of the projected network"] = "not computed here"
    return m


def anatomy_report(scene, network: Network, flow) -> list:
    """Rows: quantity, value, range, unit, source, status."""
    values = measure(scene, network, flow)
    rows = []
    for name, (low, high, unit, source) in RANGES.items():
        value = values.get(name)
        if isinstance(value, tuple):
            ok = all(np.isfinite(value)) and low <= value[0] and value[1] <= high
            missing = not all(np.isfinite(value))
            shown = f"{value[0]:.3g}-{value[1]:.3g}"
        else:
            ok = value is not None and np.isfinite(value) and low <= value <= high
            missing = value is None or not np.isfinite(value)
            shown = f"{value:.3g}" if not missing else "-"
        status = "absent" if missing else ("within" if ok else "outside")
        rows.append(dict(quantity=name, value=shown, range=f"{low:g}-{high:g}", unit=unit, source=source, status=status))
    for name, (published, source) in REFERENCE_ONLY.items():
        rows.append(dict(quantity=name, value=str(values.get(name, "-")), range=published, unit="", source=source,
                         status="reference"))
    return rows


def as_markdown(rows) -> str:
    lines = ["| quantity | synthetic | cited range | unit | status | source |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['quantity']} | {r['value']} | {r['range']} | {r['unit']} | {r['status']} | {r['source']} |")
    return "\n".join(lines)
