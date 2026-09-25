"""Pictures of a vessel network: overlays, the directed graph, residuals."""
from __future__ import annotations

import math

import cv2
import numpy as np

from .network import VesselNetwork

NODE_COLORS = {  # BGR for OpenCV, RGB hex for matplotlib
    "bifurcation": ((60, 60, 255), "#ff3c3c"),
    "junction": ((255, 60, 255), "#ff3cff"),
    "endpoint": ((0, 220, 255), "#ffdc00"),
    "border": ((255, 200, 0), "#00c8ff"),
    "joint": ((200, 200, 200), "#c8c8c8"),
}


def to_u8(intensity, lo=0.5, hi=99.5):
    """8-bit display of an image; NaN (no data) shows black."""
    a, b = np.nanpercentile(intensity, [lo, hi])
    return (np.nan_to_num(np.clip((intensity - a) / max(b - a, 1e-9), 0, 1)) * 255).astype(np.uint8)


def _colormap(values, vmin, vmax, cmap=cv2.COLORMAP_TURBO):
    t = np.clip((np.asarray(values) - vmin) / max(vmax - vmin, 1e-9), 0, 1)
    lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[:, None], cmap)[:, 0]
    return lut[(t * 255).astype(int)]


def overlay(net: VesselNetwork, intensity, color_by="diameter", scale=1.0,
            arrows=True, nodes=True, thickness=None, crossings=True):
    """BGR uint8 image with centrelines coloured by diameter / blur / contrast."""
    base = to_u8(intensity)
    if scale != 1.0:
        base = cv2.resize(base, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor((base * 0.75).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    key = {"diameter": "r", "blur": "s", "contrast": "a", "vessel": None, "tier": "tier",
           "flow": "flow"}[color_by]
    vr = {"r": (0.5, 12.0), "s": (0.6, 8.0), "a": (0.0, 0.6), None: None, "tier": None,
          "flow": None}[key]
    if key == "flow":
        sp = [e.info.get("flow", {}).get("speed_px_per_s") for e in net.edges.values()]
        sp = [x for x in sp if x]
        flo, fhi = (np.percentile(sp, 5), np.percentile(sp, 95)) if sp else (1.0, 10.0)
    for eid in net.edges:
        smp = net.sample(eid, 1.0)
        xy = smp["xy"] * scale
        if key == "flow":
            # measured red-cell speed (log scale); grey where flow is unknown
            fl = net.edges[eid].info.get("flow", {})
            spd = fl.get("speed_px_per_s")
            if spd:
                cols = _colormap(np.full(len(xy), math.log(max(spd, 1e-3))), math.log(max(flo, 1e-3)),
                                 math.log(max(fhi, flo * 1.01, 1e-3)))
            else:
                cols = np.repeat(np.array([[110, 110, 110]]), len(xy), 0)
        elif key == "tier":
            # mapped vessels blue, the faint (recall) tier orange
            faint = net.edges[eid].info.get("tier") == "faint"
            cols = np.repeat(np.array([[0, 150, 255]] if faint else [[255, 170, 60]]), len(xy), 0)
        elif key is None:
            # one colour per edge (after consolidation: per vessel)
            hue = np.uint8((eid * 47) % 180)
            c = cv2.cvtColor(np.array([[[hue, 220, 255]]], np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
            cols = np.repeat(c[None], len(xy), 0)
        else:
            val = smp[key] * (2 if key == "r" else 1)
            cols = _colormap(np.log(val) if key != "a" else val,
                             math.log(vr[0] * (2 if key == "r" else 1)) if key != "a" else vr[0],
                             math.log(vr[1] * (2 if key == "r" else 1)) if key != "a" else vr[1])
        th = thickness or max(1, int(round(scale * 1.2)))
        pts = np.round(xy * 8).astype(np.int32)
        for i in range(0, len(pts) - 1):
            cv2.line(img, tuple(pts[i]), tuple(pts[i + 1]), tuple(int(c) for c in cols[i]),
                     th, cv2.LINE_AA, shift=3)
        if arrows and key == "flow" and not net.edges[eid].info.get("flow", {}).get("speed_px_per_s"):
            pass
        elif arrows and len(xy) > 12:
            i = len(xy) // 2
            p, q = xy[i - 3], xy[i + 3]
            cv2.arrowedLine(img, tuple(np.round(p).astype(int)), tuple(np.round(q).astype(int)),
                            (255, 255, 255), 1, cv2.LINE_AA, tipLength=1.0)
    if nodes:
        deg = net.degrees()
        for nid, n in net.nodes.items():
            kind = net.node_kind(nid, deg[nid])
            if kind == "joint":
                continue
            c = NODE_COLORS[kind][0]
            cv2.circle(img, (int(round(n.x * scale)), int(round(n.y * scale))),
                       max(2, int(2 * scale)), c, -1, cv2.LINE_AA)
    if crossings:
        for c in net.crossings():
            x, y = int(c["x"] * scale), int(c["y"] * scale)
            cv2.drawMarker(img, (x, y), (255, 255, 255), cv2.MARKER_TILTED_CROSS,
                           max(5, int(5 * scale)), 1, cv2.LINE_AA)
    return img


def draw_digraph(net: VesselNetwork, path, intensity=None, figsize=None, dpi=150,
                 color_by="blur", title=None):
    """The network as a directed graph laid out at the vessels' image
    positions: line width = vessel diameter, colour = blur (focus) or
    contrast, arrows = edge direction, markers = node kinds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    H, W = net.shape
    figsize = figsize or (W / 160, H / 160 + 0.6)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    if intensity is not None:
        ax.imshow(to_u8(intensity), cmap="gray", alpha=0.35, extent=(0, W, H, 0))
    ax.set_facecolor("#101418" if intensity is None else "white")
    segs, widths, vals = [], [], []
    key = {"blur": "s", "contrast": "a", "diameter": "r"}[color_by]
    for eid in net.edges:
        smp = net.sample(eid, 2.0)
        xy = smp["xy"]
        for i in range(len(xy) - 1):
            segs.append(xy[i:i + 2])
            widths.append(min(0.3 + 0.12 * 2 * smp["r"][i], 4.0))
            vals.append(smp[key][i] * (2 if key == "r" else 1))
    cmap = plt.get_cmap("viridis_r" if key == "s" else "plasma")
    vmin, vmax = {"s": (0.6, 6.0), "a": (0.0, 0.5), "r": (1.0, 20.0)}[key]
    lc = LineCollection(segs, linewidths=widths, cmap=cmap, capstyle="round")
    lc.set_array(np.clip(np.array(vals), vmin, vmax))
    lc.set_clim(vmin, vmax)
    ax.add_collection(lc)
    # direction arrows at edge midpoints
    for eid, e in net.edges.items():
        smp = net.sample(eid, 1.0)
        xy = smp["xy"]
        if len(xy) < 8:
            continue
        i = len(xy) // 2
        d = xy[min(i + 2, len(xy) - 1)] - xy[max(i - 2, 0)]
        ax.annotate("", xy=xy[i] + 0.5 * d, xytext=xy[i] - 0.5 * d,
                    arrowprops=dict(arrowstyle="-|>", color="k" if intensity is not None else "w",
                                    lw=0.4, mutation_scale=5))
    deg = net.degrees()
    for kind, (_, hexc) in NODE_COLORS.items():
        pts = [(n.x, n.y) for k, n in net.nodes.items() if net.node_kind(k, deg[k]) == kind]
        if pts and kind != "joint":
            p = np.array(pts)
            ax.scatter(p[:, 0], p[:, 1], s=6, c=hexc, edgecolors="k", linewidths=0.2,
                       zorder=5, label=f"{kind} ({len(p)})")
    cr = net.crossings()
    if cr:
        p = np.array([[c["x"], c["y"]] for c in cr])
        ax.scatter(p[:, 0], p[:, 1], s=8, marker="x", c="#888888", linewidths=0.5, zorder=4,
                   label=f"crossing ({len(p)})")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    cb = fig.colorbar(lc, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label({"s": "blur σ (px) — focus", "a": "contrast (OD)", "r": "diameter (px)"}[key])
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="gray", lw=2))
    labels.append("line width ∝ diameter")
    ax.legend(handles, labels, loc="lower left", fontsize=6, framealpha=0.8)
    s = net.summary()
    ax.set_title(title or f"vessel digraph: {s['n_edges']} edges, {s['n_nodes']} nodes, "
                          f"{s['total_length_px']:.0f} px of centreline", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def model_panels(net: VesselNetwork, prepared, path, scale=0.5):
    """image | rendered model | residual, side by side (uint8 PNG)."""
    import torch
    from .render import NetworkModel
    m = NetworkModel(net, prepared.logI, prepared.weight, stride=1)
    with torch.no_grad():
        pred = m.predict().numpy()
    res = prepared.logI - pred
    a, b = np.percentile(prepared.logI, [0.5, 99.5])
    f = lambda x: (np.clip((x - a) / (b - a), 0, 1) * 255).astype(np.uint8)
    lim = 4 * float(np.median(prepared.sigma))
    rr = (np.clip(res / lim, -1, 1) * 127 + 128).astype(np.uint8)
    tiles = [f(prepared.logI), f(pred), rr]
    tiles = [cv2.resize(t, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for t in tiles]
    cv2.imwrite(str(path), np.hstack(tiles))
    return res


def export_html(net: VesselNetwork, path, intensity=None, max_width=1400, title="Vessel digraph"):
    """Self-contained interactive page: the digraph drawn over the image,
    hover an edge or node for its attributes, toggle layers, colour by
    diameter / blur / contrast.  No external resources."""
    import base64
    import html
    import json as _json
    H, W = net.shape
    img_uri = ""
    if intensity is not None:
        u8 = to_u8(intensity)
        sc = min(1.0, max_width / W)
        if sc < 1.0:
            u8 = cv2.resize(u8, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", u8, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_uri = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
    deg = net.degrees()
    edges = []
    for eid, e in net.edges.items():
        smp = net.sample(eid, 2.0)
        st = net.edge_stats(eid)
        edges.append(dict(id=eid, u=e.u, v=e.v,
                          d=" ".join(f"{x:.1f},{y:.1f}" for x, y in smp["xy"]),
                          diam=round(st["diameter"], 2), blur=round(st["blur"], 2),
                          contrast=round(st["contrast"], 3), length=round(st["length"], 1),
                          tort=round(min(st["tortuosity"], 99), 2),
                          gain=round(float(e.info.get("gain", float("nan"))), 1)
                          if e.info.get("gain") is not None else None,
                          orient=e.info.get("orientation", "structural"),
                          tier=e.info.get("tier", "mapped"),
                          speed=e.info.get("flow", {}).get("speed_px_per_s"),
                          joined=",".join(sorted({L["evidence"] for L in e.info.get("links", [])})),
                          visible=e.info.get("visible", True)))
    nodes = [dict(id=k, x=round(n.x, 1), y=round(n.y, 1), kind=net.node_kind(k, deg[k]), deg=deg[k])
             for k, n in net.nodes.items()]
    cross = net.crossings()
    data = _json.dumps(dict(W=W, H=H, edges=edges, nodes=nodes, crossings=cross,
                            summary=net.summary()))
    page = _HTML.replace("__TITLE__", html.escape(title)).replace("__IMG__", img_uri) \
        .replace("__DATA__", data)
    with open(path, "w") as f:
        f.write(page)


_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f6f6f4;--fg:#1d1f22;--mut:#666a70;--card:#fff;--line:#d8d8d4}
@media (prefers-color-scheme:dark){:root{--bg:#141619;--fg:#e8e8e6;--mut:#9aa0a6;--card:#1d2024;--line:#33373c}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif}
header{padding:12px 16px;display:flex;flex-wrap:wrap;gap:12px;align-items:center}
h1{font-size:16px;margin:0 12px 0 0}
label{color:var(--mut);display:flex;gap:4px;align-items:center}
#wrap{position:relative;margin:0 16px 16px;overflow:auto;border:1px solid var(--line);background:#000}
svg{display:block;width:100%;height:auto}
#tip{position:fixed;pointer-events:none;background:var(--card);color:var(--fg);border:1px solid var(--line);
 padding:6px 8px;border-radius:6px;font:12px/1.35 ui-monospace,monospace;display:none;white-space:pre;z-index:9}
.e{fill:none;stroke-linecap:round;stroke-linejoin:round;cursor:pointer}
.e:hover{stroke:#fff!important}
#stats{color:var(--mut)}
</style></head><body>
<header><h1>__TITLE__</h1>
<label>colour <select id="cb"><option value="diam">diameter</option><option value="blur">blur (focus)</option>
<option value="contrast">contrast</option><option value="tier">tier (mapped / faint)</option><option value="speed">flow speed</option></select></label>
<label><input type="checkbox" id="img" checked>image</label>
<label><input type="checkbox" id="arr" checked>arrows</label>
<label><input type="checkbox" id="nod" checked>nodes</label>
<label><input type="checkbox" id="crs">crossings</label>
<span id="stats"></span></header>
<div id="wrap"><svg id="s" xmlns="http://www.w3.org/2000/svg"></svg></div><div id="tip"></div>
<script>
const D=__DATA__, IMG="__IMG__", NS="http://www.w3.org/2000/svg";
const s=document.getElementById('s'); s.setAttribute('viewBox',`0 0 ${D.W} ${D.H}`);
const mk=(t,a)=>{const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);return e};
const defs=mk('defs',{});const m=mk('marker',{id:'ah',viewBox:'0 0 10 10',refX:5,refY:5,markerWidth:4,markerHeight:4,orient:'auto-start-reverse'});
m.appendChild(mk('path',{d:'M0,0L10,5L0,10z',fill:'#fff'}));defs.appendChild(m);s.appendChild(defs);
const gi=mk('image',{href:IMG,x:0,y:0,width:D.W,height:D.H,opacity:0.8});if(IMG)s.appendChild(gi);
const ge=mk('g',{}),ga=mk('g',{}),gn=mk('g',{}),gc=mk('g',{display:'none'});[ge,ga,gn,gc].forEach(g=>s.appendChild(g));
const turbo=t=>{t=Math.max(0,Math.min(1,t));const r=Math.round(34.61+t*(1172.33-t*(10793.56-t*(33300.12-t*(38394.49-t*14825.05)))));
 const g=Math.round(23.31+t*(557.33+t*(1225.33-t*(3574.96-t*(1073.77+t*707.56)))));const b=Math.round(27.2+t*(3211.1-t*(15327.97-t*(27814-t*(22569.18-t*6838.66)))));
 return `rgb(${Math.max(0,Math.min(255,r))},${Math.max(0,Math.min(255,g))},${Math.max(0,Math.min(255,b))})`};
const rng={diam:[1,40,true],blur:[0.6,12,true],contrast:[0,0.6,false]};
{const sp=D.edges.map(e=>e.speed).filter(v=>v>0).sort((a,b)=>a-b);
 rng.speed=sp.length?[sp[Math.floor(sp.length*0.05)],Math.max(sp[Math.floor(sp.length*0.95)],sp[0]*1.01),true]:[1,10,true];}
const col=(k,v)=>{if(k==='tier')return v==='faint'?'#ff9f1c':'#4aa8ff';if(k==='speed'&&!v)return '#777';const [a,b,lg]=rng[k];return turbo(lg?Math.log(v/a)/Math.log(b/a):(v-a)/(b-a))};
const tip=document.getElementById('tip');
const show=(ev,txt)=>{tip.style.display='block';tip.textContent=txt;tip.style.left=(ev.clientX+14)+'px';tip.style.top=(ev.clientY+10)+'px'};
const hide=()=>tip.style.display='none';
const paths=[];
for(const e of D.edges){const p=mk('path',{class:'e',d:'M'+e.d.replace(/ /g,' L'),'stroke-width':Math.max(1.0,Math.min(e.diam*0.22,6))});
 if(!e.visible)p.setAttribute('stroke-dasharray','4 3');
 p.addEventListener('mousemove',ev=>show(ev,`edge ${e.id}: ${e.u} → ${e.v}  (${e.orient}, ${e.tier})\nlength ${e.length} px  tortuosity ${e.tort}\ndiameter ${e.diam} px  blur ${e.blur} px\ncontrast ${e.contrast} OD  gain ${e.gain}`+(e.speed?`\nflow ${e.speed} px/s (measured)`:'')+(e.joined?`\njoined by ${e.joined}`:'')));
 p.addEventListener('mouseleave',hide);ge.appendChild(p);paths.push([p,e]);
 const pts=e.d.split(' ').map(q=>q.split(',').map(Number));if(pts.length>6){const i=pts.length>>1,a=pts[i-2],b=pts[i+1];
 ga.appendChild(mk('path',{d:`M${a[0]},${a[1]}L${b[0]},${b[1]}`,stroke:'#fff','stroke-width':1.2,'marker-end':'url(#ah)',fill:'none'}))}}
const nc={bifurcation:'#ff3c3c',junction:'#ff3cff',endpoint:'#ffdc00',border:'#00c8ff',joint:'#c8c8c8'};
for(const n of D.nodes){if(n.kind==='joint')continue;const c=mk('circle',{cx:n.x,cy:n.y,r:4,fill:nc[n.kind],stroke:'#000','stroke-width':0.8});
 c.addEventListener('mousemove',ev=>show(ev,`node ${n.id}: ${n.kind} (degree ${n.deg})\nx ${n.x}  y ${n.y}`));c.addEventListener('mouseleave',hide);gn.appendChild(c)}
for(const c of D.crossings){gc.appendChild(mk('path',{d:`M${c.x-5},${c.y-5}L${c.x+5},${c.y+5}M${c.x-5},${c.y+5}L${c.x+5},${c.y-5}`,stroke:'#fff','stroke-width':1.5}))}
const recolor=()=>{const k=document.getElementById('cb').value;for(const [p,e] of paths)p.setAttribute('stroke',col(k,e[k]))};
recolor();document.getElementById('cb').onchange=recolor;
const tg=(id,g)=>{document.getElementById(id).onchange=ev=>g.setAttribute('display',ev.target.checked?'inline':'none')};
tg('img',gi);tg('arr',ga);tg('nod',gn);tg('crs',gc);
const k=D.summary.node_kinds;document.getElementById('stats').textContent=
 `${D.summary.n_edges} edges · ${D.summary.n_nodes} nodes (${Object.entries(k).map(([a,b])=>b+' '+a).join(', ')}) · ${D.crossings.length} crossings · ${Math.round(D.summary.total_length_px)} px centreline · direction = structural convention (wide→narrow), not flow`;
</script></body></html>
"""
