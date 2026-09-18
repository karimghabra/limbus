"""QC images and the cross-burst summary (METHODS.md §11)."""
import csv
import html
import json
import os

import cv2
import numpy as np

INK = (35, 35, 35)
MUTED = (120, 120, 120)
GRID = (225, 225, 225)
X_COL = (178, 106, 31)     # BGR — blue
Y_COL = (10, 102, 194)     # BGR — amber
BAD = (40, 40, 200)


NO_DATA = (180, 60, 200)    # BGR — magenta: distinct from any grey level


def _to_8bit(img, lo, hi):
    img = np.nan_to_num(img, nan=lo)
    return np.clip((img - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)


def _panel(img, width, lo, hi, label):
    h, w = img.shape
    height = max(1, int(round(h * width / w)))
    small = cv2.resize(_to_8bit(img, lo, hi), (width, height),
                       interpolation=cv2.INTER_AREA)
    panel = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    # regions with no valid data (too few frames observed them — borders, or
    # reflections clipped in every frame) must not render black, where they
    # would be indistinguishable from a vessel
    missing = ~np.isfinite(img)
    if missing.any():
        small_missing = cv2.resize(missing.astype(np.uint8), (width, height),
                                   interpolation=cv2.INTER_NEAREST) > 0
        panel[small_missing] = NO_DATA
    band = np.full((30, width, 3), 255, np.uint8)
    cv2.putText(band, label, (4, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, INK, 1,
                cv2.LINE_AA)
    return np.vstack([band, panel])


def _trajectory_plot(traj, times, registered, good, width, height):
    plot = np.full((height, width, 3), 255, np.uint8)
    left, right, top, bottom = 70, 20, 20, 40
    pw, ph = width - left - right, height - top - bottom
    idx = np.flatnonzero(registered)
    t = times
    tmax = max(float(t[-1]), 1e-6)
    vals = traj[idx] if idx.size else np.zeros((1, 2))
    vmin, vmax = float(vals.min()), float(vals.max())
    pad = max((vmax - vmin) * 0.1, 1.0)
    vmin, vmax = vmin - pad, vmax + pad

    def xy(ti, v):
        return (int(left + pw * ti / tmax),
                int(top + ph * (1 - (v - vmin) / (vmax - vmin))))

    for k in range(5):
        v = vmin + (vmax - vmin) * k / 4
        y = xy(0, v)[1]
        cv2.line(plot, (left, y), (left + pw, y), GRID, 1)
        cv2.putText(plot, f"{v:.0f}", (4, y + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, MUTED, 1, cv2.LINE_AA)
    for k in range(6):
        ti = tmax * k / 5
        x = xy(ti, vmin)[0]
        cv2.putText(plot, f"{ti:.1f}s", (x - 12, height - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, MUTED, 1, cv2.LINE_AA)
    # break the line wherever frames are missing: joining registered frames
    # straight across a gap would draw motion that was never measured
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1) if idx.size else []
    for col, k, name in ((X_COL, 0, "dx"), (Y_COL, 1, "dy")):
        for run in runs:
            pts = [xy(t[i], traj[i, k]) for i in run]
            if len(pts) > 1:
                cv2.polylines(plot, [np.array(pts, np.int32)], False, col, 2,
                              cv2.LINE_AA)
            elif pts:
                cv2.circle(plot, pts[0], 2, col, -1, cv2.LINE_AA)
    # frames the gate or the registration rejected, as ticks along the axis
    for i in np.flatnonzero(~registered):
        x = xy(t[i], vmin)[0]
        cv2.line(plot, (x, top + ph), (x, top + ph - 8), BAD, 1)
    cv2.putText(plot, "displacement (full-res px):", (left, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, INK, 1, cv2.LINE_AA)
    cv2.putText(plot, "dx", (left + 230, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                X_COL, 2, cv2.LINE_AA)
    cv2.putText(plot, "dy", (left + 265, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                Y_COL, 2, cv2.LINE_AA)
    cv2.putText(plot, "red ticks = frames not used (lines break there)", (left + 310, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, BAD, 1, cv2.LINE_AA)
    cv2.putText(plot, "magenta in images = no valid data", (left + 760, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, NO_DATA, 1, cv2.LINE_AA)
    return plot


def _fmt(v, spec=".2f"):
    return format(v, spec) if isinstance(v, (int, float)) else "n/a"


def write_qc_png(path, mean_raw, mean_st, traj, times, registered, good, rec):
    width = 1780
    finite = mean_st[np.isfinite(mean_st)]
    lo, hi = (np.percentile(finite, [1, 99]) if finite.size
              else (float(np.min(mean_raw)), float(np.max(mean_raw))))
    strip = mean_raw.shape[0] / mean_raw.shape[1] < 0.2
    if strip:
        panels = np.vstack([_panel(mean_raw, width, lo, hi, "RAW MEAN (no correction)"),
                            _panel(mean_st, width, lo, hi, "STABILIZED MEAN")])
    else:
        half = (width - 20) // 2
        a = _panel(mean_raw, half, lo, hi, "RAW MEAN (no correction)")
        b = _panel(mean_st, half, lo, hi, "STABILIZED MEAN")
        gap = np.full((a.shape[0], 20, 3), 255, np.uint8)
        panels = np.hstack([a, gap, b])
        if panels.shape[1] < width:
            panels = np.hstack([panels, np.full((panels.shape[0], width - panels.shape[1], 3), 255, np.uint8)])
    plot = _trajectory_plot(traj, times, registered, good, width, 260)

    m, q = rec["motion"], rec["quality"]
    method = rec.get("method", "translation") + (" (EXPERIMENTAL)" if rec.get("experimental") else "")
    lines = [
        f"{rec['burst']['name']}   [{method}]   {rec['burst']['width']}x{rec['burst']['height']} "
        f"{rec['burst']['pixel_format']} @ {rec['burst']['fps']:.1f} fps, {rec['burst']['frames']} frames",
        f"STABILITY INDEX {rec['stability_index']:.3f}   usable {100 * rec['usable_fraction']:.0f}%   "
        f"vessel overlap {q['before']['overlap']:.3f} -> {q['after']['overlap']:.3f}   "
        f"Dice {q['before']['dice']:.3f} -> {q['after']['dice']:.3f}",
        f"motion: displacement median {_fmt(m.get('displacement_median_px'), '.1f')} px, "
        f"max {_fmt(m.get('displacement_max_px'), '.1f')} px   jitter {_fmt(m.get('jitter_rms_px_s'), '.0f')} px/s   "
        f"drift {_fmt(m.get('drift_px_s'), '.1f')} px/s   saccades {m.get('saccades', 'n/a')}",
        f"residual step median {_fmt(q.get('residual_step_median_px'))} px   "
        f"closure median {_fmt(q.get('closure_median_px'))} px   "
        f"rotation check (quadrant disagreement) median "
        f"{_fmt(rec['diagnostics'].get('quadrant_disagreement_median_px'))} px",
    ]
    text = np.full((22 * len(lines) + 16, width, 3), 255, np.uint8)
    for k, line in enumerate(lines):
        cv2.putText(text, line, (8, 24 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, INK if k != 1 else (20, 110, 20), 1, cv2.LINE_AA)
    canvas = np.vstack([text, panels, plot])
    cv2.imwrite(path, canvas)


# ---- summary across bursts ---------------------------------------------------

COLUMNS = [
    ("burst", "Burst"), ("status", "Status"), ("stability_index", "Stability index"),
    ("usable_pct", "Usable %"), ("overlap_before", "Vessel overlap before"),
    ("dice_before", "Dice before"), ("dice_after", "Dice after"),
    ("displacement_median_px", "Displacement median px"),
    ("displacement_max_px", "Displacement max px"), ("jitter_rms_px_s", "Jitter px/s"),
    ("saccades", "Saccades"), ("residual_step_median_px", "Residual step px"),
    ("rotation_px", "Rotation check px"), ("geometry", "Geometry"),
    ("fps", "fps"), ("frames", "Frames"), ("note", "Note"),
]


def _row(rec):
    b = rec.get("burst", {})
    if rec.get("status") != "ok":
        return {"burst": b.get("name"), "status": "skipped", "frames": b.get("frames"),
                "note": rec.get("skip_reason", "")}
    q, m = rec["quality"], rec["motion"]
    return {
        "burst": b["name"], "status": "ok",
        "stability_index": round(rec["stability_index"], 3),
        "usable_pct": round(100 * rec["usable_fraction"], 1),
        "overlap_before": round(q["before"]["overlap"], 3),
        "dice_before": round(q["before"]["dice"], 3),
        "dice_after": round(q["after"]["dice"], 3),
        "displacement_median_px": round(m.get("displacement_median_px", float("nan")), 1),
        "displacement_max_px": round(m.get("displacement_max_px", float("nan")), 1),
        "jitter_rms_px_s": round(m.get("jitter_rms_px_s", float("nan")), 0),
        "saccades": m.get("saccades"),
        "residual_step_median_px": round(q.get("residual_step_median_px", float("nan")), 2),
        "rotation_px": round(rec["diagnostics"].get("quadrant_disagreement_median_px", float("nan")), 2),
        "geometry": f"{b['width']}x{b['height']}", "fps": round(b["fps"], 1),
        "frames": b["frames"], "note": "",
    }


def write_summary(out_root, records):
    rows = [_row(r) for r in records]
    rows.sort(key=lambda r: (r["status"] != "ok", -(r.get("stability_index") or 0)))
    with open(os.path.join(out_root, "summary.csv"), "w", newline="",
              encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=[k for k, _ in COLUMNS])
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k, _ in COLUMNS})
    _write_html(os.path.join(out_root, "summary.html"), rows)
    return rows


def _write_html(path, rows):
    ok = [r for r in rows if r["status"] == "ok"]
    head = "".join(f"<th>{html.escape(label)}</th>" for _, label in COLUMNS)
    body = []
    for r in rows:
        cells = []
        for key, _ in COLUMNS:
            val = r.get(key, "")
            val = "" if val is None else val
            if key == "burst" and r["status"] == "ok":
                cell = (f'<a href="{html.escape(str(val))}/qc.png">'
                        f"{html.escape(str(val))}</a>")
            else:
                cell = html.escape(str(val))
            cls = ' class="num"' if isinstance(val, (int, float)) else ""
            cells.append(f"<td{cls}>{cell}</td>")
        body.append(f'<tr class="{r["status"]}">' + "".join(cells) + "</tr>")
    si = [r["stability_index"] for r in ok]
    stats = (f"{len(ok)} stabilized, {len(rows) - len(ok)} skipped · stability index "
             f"median {np.median(si):.3f}, range {min(si):.3f}–{max(si):.3f}") if si else ""
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Burst stability summary</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root {{ --bg:#f6f7f8; --panel:#fff; --ink:#15191d; --muted:#5d6873; --rule:#dde2e6; --accent:#1f6fb2; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg:#0f1316; --panel:#171c21; --ink:#e6ebef; --muted:#93a1ad; --rule:#28313a; --accent:#6ba6e0; }} }}
  body {{ margin:0; padding-block:28px; padding-inline:max(16px, 3vw); background:var(--bg); color:var(--ink);
         font:14px/1.5 "Segoe UI", system-ui, sans-serif; }}
  h1 {{ font-size:22px; margin:0 0 4px; }}
  p {{ color:var(--muted); max-width:78ch; margin:4px 0; }}
  .wrap {{ overflow-x:auto; background:var(--panel); border:1px solid var(--rule); border-radius:6px; margin-top:18px; }}
  table {{ border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }}
  th, td {{ padding:7px 10px; border-bottom:1px solid var(--rule); text-align:left; white-space:nowrap; }}
  th {{ font-size:12px; color:var(--muted); font-weight:600; position:sticky; top:0; background:var(--panel); }}
  td.num {{ text-align:right; }}
  tr.skipped td {{ color:var(--muted); }}
  a {{ color:var(--accent); }}
</style></head><body>
<h1>Burst stability summary</h1>
<p>{html.escape(stats)}</p>
<p><b>Stability index</b> = vessel overlap after stabilization: the probability that a pixel labelled vessel in one frame is also vessel in another (1 = every frame agrees).
It compares bursts processed with the same settings, not an absolute: red-cell gaps keep it below 1 even when perfectly aligned.
Read it beside <b>Usable %</b>. <b>Before</b> columns show the unstabilized burst; displacement and jitter describe the eye's motion.
<b>Rotation check</b> near 0 px means a translation model was sufficient. Click a burst for its QC image. Methods: <code>stabilize/METHODS.md</code>.</p>
<div class="wrap"><table><thead><tr>{head}</tr></thead><tbody>
{''.join(body)}
</tbody></table></div>
</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
