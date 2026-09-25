"""A self-contained HTML report of one pipeline run.

    python -m vesselmap report OUT/faint --image frame.tif [--frames OUT/frames] -o OUT/report

Reads the files a `map` / `refine` / `faint` / `consolidate` run wrote (and,
optionally, a `fit-frames` folder) and writes `report.html` with its images
in `img/` next to it, plus a copy of the interactive graph.  Nothing is
recomputed except the drawings.
"""
from __future__ import annotations

import csv
import html
import json
import os
import shutil

import cv2
import numpy as np

from .draw import to_u8
from .image import load_image
from .network import VesselNetwork

_BLUE, _ORANGE = (214, 149, 43), (14, 132, 232)     # BGR, matching the page's colours
_REF_IMAGES = ("overlay_tier", "overlay", "overlay_blur", "digraph_on_image", "digraph",
               "model_residual")


def _faint(ed) -> bool:
    return ed.info.get("tier") == "faint"


def _draw_tiers(net: VesselNetwork, intensity, dash_invisible=True):
    g = cv2.cvtColor((to_u8(intensity) * 0.8).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for eid, ed in net.edges.items():
        xy = np.round(net.sample(eid, 1.0)["xy"] * 8).astype(np.int32)
        c = _ORANGE if _faint(ed) else _BLUE
        if dash_invisible and not ed.info.get("visible", True):
            for i in range(0, len(xy) - 1, 8):
                cv2.polylines(g, [xy[i:i + 5]], False, c, 2, cv2.LINE_AA, shift=3)
        else:
            cv2.polylines(g, [xy], False, c, 2 if _faint(ed) else 1, cv2.LINE_AA, shift=3)
    return g


def _jpg(path, im):
    if im.ndim == 3 and im.shape[2] == 4:
        a = im[..., 3:4] / 255.0
        im = (im[..., :3] * a + 255 * (1 - a)).astype(np.uint8)
    cv2.imwrite(path, im, [cv2.IMWRITE_JPEG_QUALITY, 85])


def _mask_view(mask, intensity):
    base = cv2.cvtColor((to_u8(intensity) * 0.85).astype(np.uint8), cv2.COLOR_GRAY2BGR).astype(float)
    for k, c in ((1, (230, 160, 40)), (2, (20, 140, 255))):
        base[mask == k] = 0.55 * base[mask == k] + 0.45 * np.array(c, float)
    return base.astype(np.uint8)


def _step(title, cmd, text, facts):
    return (f"<li><h3>{html.escape(title)}</h3><pre>{html.escape(cmd)}</pre>"
            f"<p class=\"muted\">{html.escape(text)}</p><span class=\"t\">{html.escape(facts)}</span></li>")


def _minutes(sec):
    return f"{sec / 60:.1f} min" if sec < 3600 else f"{sec / 3600:.1f} h"


def build_report(run_dir: str, image: str, out: str, frames_dir: str | None = None,
                 stem: str = "map") -> str:
    """Write out/report.html (+ img/, map_digraph.html).  Returns its path."""
    os.makedirs(os.path.join(out, "img"), exist_ok=True)
    p = lambda name: os.path.join(run_dir, f"{stem}{name}")
    net = VesselNetwork.load(p(".json"))
    summ = json.load(open(p("_summary.json")))["summary"]
    I = load_image(image)
    img_name = os.path.basename(image)
    t = open(os.path.join(os.path.dirname(__file__), "report_template.html")).read()
    rep = {}

    # ---------------------------------------------------------------- header
    rep["__EYEBROW__"] = html.escape(
        f"vesselmap · {img_name} · {net.shape[1]} × {net.shape[0]} px")
    rep["__REFNAME__"] = html.escape(img_name)
    rep["__REFFILE__"] = html.escape(img_name).replace("'", "\\'")
    n_fr = 0
    if frames_dir and os.path.exists(os.path.join(frames_dir, "frames.csv")):
        n_fr = len(list(csv.DictReader(open(os.path.join(frames_dir, "frames.csv")))))
    rep["__LEDE__"] = html.escape(
        "Everything the pipeline wrote for one reference frame"
        + (f", plus the same map fitted to {n_fr} other frame{'s' if n_fr != 1 else ''}" if n_fr else "")
        + ". Only still images are used: no velocities or kymographs.")

    # ---------------------------------------------------------------- steps
    meta = net.meta
    steps = []
    faint_meta = meta.get("faint", [])
    lf = sum(net.length(e) for e, ed in net.edges.items() if _faint(ed))
    n_faint = sum(_faint(ed) for ed in net.edges.values())
    if meta.get("builder"):
        steps.append(_step("Map the reference frame", f"python -m vesselmap map {img_name} -o out/map",
                           "Fits a network of splines to the image, coarse to fine; each spline is "
                           "kept only if it explains more of the image than its parameters cost (MDL).",
                           f"{_minutes(meta.get('seconds', 0))}"))
    for r in meta.get("refinements", []):
        steps.append(_step("Refine", f"python -m vesselmap refine out/map/map.json {img_name} -o out/refined",
                           "Adds fine vessels and splits forks whose branches run side by side.",
                           f"{_minutes(r['seconds'])} · {r['length_before']:.0f} → {r['length_after']:.0f} px"))
    for f in faint_meta:
        steps.append(_step("Add the faint tier", f"python -m vesselmap faint out/map/map.json {img_name} -o out/faint",
                           "Traces faint vessels along their length in what the map leaves "
                           "unexplained, and writes the search mask.",
                           f"{_minutes(f['seconds'])} · +{f['edges']} edges · +{f['length'] / 1000:.1f}k px"))
    if "consolidation" in meta:
        steps.append(_step("Consolidate", f"python -m vesselmap consolidate MAP.json {img_name} -o out/vessels",
                           "Merges the segments of each vessel into one spline.", ""))
    if n_fr:
        rows = list(csv.DictReader(open(os.path.join(frames_dir, "frames.csv"))))
        mean_s = np.mean([float(r["seconds"]) for r in rows])
        steps.append(_step("Fit the map to other frames",
                           "python -m vesselmap fit-frames out/faint/map.json FRAME ... -o out/frames --chain --overlays",
                           "Aligns the map to each frame, then adjusts positions and profiles slightly. "
                           "Ids and topology stay the same.",
                           f"about {mean_s / 60:.1f} min per frame · {n_fr} frames"))
    rep["__STEPS__"] = "".join(steps)

    # ---------------------------------------------------------------- images
    for name in _REF_IMAGES:
        src = p(f"_{name}.png")
        if name == "overlay_tier" and not os.path.exists(src):
            _jpg(os.path.join(out, "img", f"{name}.jpg"), _draw_tiers(net, I, False))
        elif os.path.exists(src):
            _jpg(os.path.join(out, "img", f"{name}.jpg"), cv2.imread(src, cv2.IMREAD_UNCHANGED))
    _jpg(os.path.join(out, "img", "frame.jpg"), to_u8(I))
    mask = cv2.imread(p("_search_mask.png"), cv2.IMREAD_UNCHANGED)
    if mask is None:
        from .faint import search_mask
        mask = search_mask(net, net.shape)
    _jpg(os.path.join(out, "img", "search_mask.jpg"), _mask_view(mask, I))
    if os.path.exists(p("_digraph.html")):
        shutil.copy(p("_digraph.html"), os.path.join(out, "map_digraph.html"))

    # ---------------------------------------------------------------- numbers
    k = summ["node_kinds"]
    rep.update({
        "__NEDGES__": f"{summ['n_edges']}", "__NNODES__": f"{summ['n_nodes']}",
        "__LMAP__": f"{(summ['total_length_px'] - lf) / 1000:.1f}k",
        "__LFAINT__": f"{lf / 1000:.1f}k",
        "__KINDS__": f"{k.get('bifurcation', 0)} · {k.get('endpoint', 0)} · {k.get('border', 0)}",
        "__MASK__": f"{np.mean(mask > 0) * 100:.0f}% ({np.mean(mask == 1) * 100:.0f} · "
                    f"{np.mean(mask == 2) * 100:.0f})",
    })
    radii = [float(np.median(net.sample(e, 1.0)["r"])) for e, ed in net.edges.items() if not _faint(ed)]
    rep["__CALLOUT__"] = "" if np.mean(mask > 0) < 0.3 else (
        '<div class="callout"><h3>The search mask is wide</h3><p>Mapped tubes are drawn at the '
        'fitted radius plus 3 px, and the fit gives blurred, faint vessels a large radius '
        f'(median {np.median(radii):.1f} px), so {np.mean(mask > 0) * 100:.0f}% of the image is '
        'inside the mask. A tube based on the apparent width, or a fixed width around the '
        'centreline, would be tighter.</p></div>')

    # ---------------------------------------------------------------- files
    d = json.load(open(p(".json")))
    pick = [e for e in d["edges"] if e["info"].get("tier") == "faint"][:1] or d["edges"][:1]
    rnd = lambda x: [rnd(y) for y in x] if isinstance(x, list) else (round(x, 2) if isinstance(x, float) else x)
    e0 = {key: ({a: rnd(b) for a, b in v.items()} if key == "info" else rnd(v)) for key, v in pick[0].items()}
    e0["ctrl"] = e0["ctrl"][:3] + ["…"]
    lines = ["{"] + [f'  "{key}": ' + json.dumps(v, separators=(", ", ": "), ensure_ascii=False) + ","
                     for key, v in e0.items()]
    lines[-1] = lines[-1][:-1]
    rep["__EDGEJSON__"] = html.escape("\n".join(lines + ["}"]).replace('"…"', "…"))
    rows = list(csv.DictReader(open(p("_edges.csv"))))
    tier = lambda r: r.get("tier", "mapped")
    pick = [r for r in rows if tier(r) == "mapped"][:5] + [r for r in rows if tier(r) == "faint"][:5]
    f = lambda x, n=1: f"{float(x):.{n}f}"
    rep["__ROWS__"] = "".join(
        f"<tr><td>{r['eid']}</td><td>{r['u']} → {r['v']}</td><td><span class='tier {tier(r)}'>"
        f"{tier(r)}</span></td><td class='num'>{f(r['length'])}</td><td class='num'>{f(r['diameter'])}"
        f"</td><td class='num'>{f(r['blur'])}</td><td class='num'>{f(r['contrast'], 3)}</td>"
        f"<td class='num'>{f(r['tortuosity'], 2)}</td></tr>" for r in pick)

    # ---------------------------------------------------------------- frames
    frames, rows2 = [], ""
    if n_fr:
        for r in csv.DictReader(open(os.path.join(frames_dir, "frames.csv"))):
            name = r["frame"]
            fn = VesselNetwork.load(os.path.join(frames_dir, f"{name}.json"))
            src = fn.meta.get("source_image")
            if not src or not os.path.exists(src):
                continue
            If = load_image(src)
            _jpg(os.path.join(out, "img", f"{name}_fit.jpg"), _draw_tiers(fn, If))
            _jpg(os.path.join(out, "img", f"{name}_raw.jpg"), _draw_tiers(net, If, False))
            sx, sy = json.loads(r["affine"])["t"] if r.get("affine") not in (None, "", "null") else (0.0, 0.0)
            vis = lambda want: np.mean([fn.edges[e].info.get("visible", True)
                                        for e, ed in net.edges.items() if _faint(ed) == want] or [1.0])
            note = (f"{vis(False) * 100:.0f}% of mapped" + (f" and {vis(True) * 100:.0f}% of faint" if n_faint else "")
                    + " edges are visible here; dashed edges are not.")
            frames.append(dict(name=html.escape(name), fit=f"img/{name}_fit.jpg", raw=f"img/{name}_raw.jpg",
                               shift=f"{sx:.1f}, {sy:.1f}", note=note))
            rows2 += (f"<tr><td>{html.escape(name)}</td><td class='num'>{sx:.1f}, {sy:.1f}</td>"
                      f"<td class='num'>{float(r['node_shift_median']):.1f} / {float(r['node_shift_p95']):.1f}</td>"
                      f"<td class='num'>{float(r['visible_fraction']) * 100:.0f}%</td>"
                      f"<td class='num'>{float(r['seconds']) / 60:.1f} min</td></tr>")
    if not frames:
        a, b = t.index("<!--FRAMES-->"), t.index("<!--/FRAMES-->") + len("<!--/FRAMES-->")
        t = t[:a] + t[b:]
    rep["__FRAMEROWS__"] = rows2
    rep["__FRAMES__"] = json.dumps(frames)
    for key, val in rep.items():
        t = t.replace(key, val)
    path = os.path.join(out, "report.html")
    with open(path, "w") as fh:
        fh.write('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">\n'
                 '<meta name="viewport" content="width=device-width,initial-scale=1"></head><body>\n')
        fh.write(t)
        fh.write("\n</body></html>\n")
    return path
