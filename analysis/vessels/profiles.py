"""Per-vessel space-time profiles (kymographs) and their spectra.

Once a vessel has an identity, every raw frame of the burst can be sampled
along ITS centreline: stabilization maps stabilized(x) = raw(x + d(x)), so a
centreline point p in the averaged frame sits at p + d(p) in raw frame i.
Sampling the raw frames - not the warped ones - keeps every measurement on
the pixels the camera actually recorded, with one interpolation instead of
two.

For each vessel this gives

  kymograph K[i, s]   absorbance at distance s along the vessel in frame i,
                      averaged across the lumen (a half-radius band), with
                      the background just beside the vessel subtracted
  time series m[i]    the kymograph averaged along the vessel: how dark the
                      whole vessel is in frame i

and from m an amplitude spectrum. The camera's own per-frame timestamps set
the frequency axis, so dropped frames do not shift it; frames the
stabilization rejected are left out and the spectrum is computed by least
squares at each frequency (a Lomb-Scargle periodogram) rather than by an FFT
that would assume even spacing.

usage:
  python -m vessels.profiles <burst> [--method nonrigid] [--vessels DIR]
                             [--min-length 80] [--max-vessels 40]
"""
import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "vessels"

from . import image as vimg                      # noqa: E402
from . import model as vmodel                    # noqa: E402


def centreline_of(v, step=1.0, threaded=True):
    """Dense centrelines of one vessel record from vessels.json, longest
    first: (length, points, radius).

    A vessel is threaded from several fitted pieces, and until the export
    carried that threading the only geometry here was the pieces - so a vessel
    that qualified for measurement on its THREADED length was then measured
    along its longest single piece, which on one crop was 60 % of the total
    and on its longest vessel 18 %. When the record carries `centreline`, that
    is what gets sampled, with the per-point radius beside it.
    """
    if threaded and v.get("centreline"):
        C = np.asarray(v["centreline"], float)
        if len(C) >= 2:
            L = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(C, axis=0).T))])
            if L[-1] >= 4:
                t = np.arange(0, L[-1], step)
                pts = np.stack([np.interp(t, L, C[:, 0]), np.interp(t, L, C[:, 1])], 1)
                rp = v.get("radius_profile")
                if rp and len(rp) == len(C):
                    r = float(np.median(np.asarray(rp, float)))
                else:
                    r = float(v.get("radius_px") or 2.0)
                return [(float(L[-1]), pts, r)]
    out = []
    for p in v["pieces"]:
        P = np.asarray(p["points"], float)
        if len(P) < 2:
            continue
        C = vmodel.catmull(P)[0] if len(P) > 2 else P
        L = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(C, axis=0).T))])
        if L[-1] < 4:
            continue
        s = np.arange(0, L[-1], step)
        pts = np.stack([np.interp(s, L, C[:, 0]), np.interp(s, L, C[:, 1])], 1)
        r = float(np.median(p["radius_px"])) if p.get("radius_px") else 2.0
        out.append((float(L[-1]), pts, r))
    out.sort(key=lambda t: -t[0])
    return out


def sample_along(A, C, r, n_in=5, gap=3.0, n_bg=4):
    """Absorbance along C: the mean over a band of half-width r/2 across the
    vessel, minus the median of two background strips just outside it."""
    t = np.gradient(C, axis=0)
    t /= np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-9)[:, None]
    nrm = np.stack([-t[:, 1], t[:, 0]], 1)
    offs_in = np.linspace(-r / 2, r / 2, n_in)
    offs_bg = np.concatenate([-(r + gap + np.arange(n_bg)), r + gap + np.arange(n_bg)])

    def strip(offs):
        px = (C[:, None, 0] + nrm[:, None, 0] * offs[None, :]).astype(np.float32)
        py = (C[:, None, 1] + nrm[:, None, 1] * offs[None, :]).astype(np.float32)
        return cv2.remap(A, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return strip(offs_in).mean(1) - np.median(strip(offs_bg), axis=1)


def periodogram(t, y, freqs):
    """Amplitude at each frequency by least squares - correct for the uneven
    times left by dropped and rejected frames."""
    y = np.asarray(y, float)
    y = y - y.mean()
    amp = np.empty(len(freqs))
    for k, f in enumerate(freqs):
        w = 2 * np.pi * f * np.asarray(t, float)
        G = np.stack([np.cos(w), np.sin(w)], 1)
        c, *_ = np.linalg.lstsq(G, y, rcond=None)
        amp[k] = float(np.hypot(*c))
    return amp


def _prepare_kymo(K, smooth=1.5):
    """A kymograph ready for motion measurement: the static pattern (the
    vessel's own shape, which does not move) removed as the median over time
    at each position, each frame's overall brightness removed, and a little
    smoothing ALONG the vessel only. Filtering across time would correlate
    neighbouring frames and manufacture a zero-shift peak - the very thing
    the measurement is looking for."""
    K = np.asarray(K, np.float32)
    K = K - np.median(K, axis=0, keepdims=True)            # static pattern
    K = K - K.mean(1, keepdims=True)                       # frame brightness
    if smooth:
        K = cv2.GaussianBlur(K, (0, 0), sigmaX=float(smooth), sigmaY=1e-6)
    return K


def streak_velocity_xcorr(K, t, win_s=0.3, lag_frames=(1, 2, 3), max_px_per_frame=30.0,
                          min_peak=0.15, agree_px=2.0, smooth=1.5):
    """Speed along a vessel by cross-correlating the kymograph in time.

    The structure tensor linearises the motion, so it only reads slopes of a
    few pixels per frame. Cross-correlation does not: for each window the
    normalised correlation between rows separated by g = 1, 2 and 3 frames is
    measured over every shift up to g * max_px_per_frame, and the shift where
    it peaks (refined to sub-pixel by a parabola through its neighbours),
    divided by g, is the displacement per frame. The sign says which way
    along the centreline the blood runs.

    Two checks stop a confident wrong answer. The peak correlation must reach
    min_peak, and the three lags must AGREE to within agree_px: a periodic or
    aliased pattern peaks at a different place for each lag, while real
    transport moves g times as far in g frames. Windows failing either, or
    peaking at the edge of the search range, are reported as unknown.

    Returns (times, speed px/s, peak correlation), one entry per window.
    """
    K = _prepare_kymo(K, smooth)
    dt = float(np.median(np.diff(t)))
    n = max(4, int(round(win_s / dt)))
    m = int(max_px_per_frame)
    out_t, out_v, out_p = [], [], []
    for a0 in range(0, len(t) - n + 1, max(1, n // 2)):
        W = K[a0:a0 + n]
        per_lag = []
        for g in lag_frames:
            if n <= g:
                continue
            A_, B_ = W[:-g], W[g:]
            na = float(np.linalg.norm(A_) * np.linalg.norm(B_))
            if na <= 0:
                continue
            shifts = np.arange(-m * g, m * g + 1)
            score = np.full(len(shifts), -np.inf)
            for j, sh in enumerate(shifts):
                if abs(sh) >= W.shape[1] - 4:
                    continue
                if sh >= 0:
                    score[j] = float((A_[:, sh:] * B_[:, :B_.shape[1] - sh]).sum()) / na
                else:
                    score[j] = float((A_[:, :A_.shape[1] + sh] * B_[:, -sh:]).sum()) / na
            j = int(np.argmax(score))
            frac = 0.0
            if 0 < j < len(shifts) - 1 and np.isfinite(score[j - 1:j + 2]).all():
                d = score[j - 1] - 2 * score[j] + score[j + 1]
                frac = 0.5 * (score[j - 1] - score[j + 1]) / d if abs(d) > 1e-12 else 0.0
            per_lag.append((-(shifts[j] + frac) / g, float(score[j]), j in (0, len(shifts) - 1)))
        out_t.append(float(t[a0] + n * dt / 2))
        if not per_lag:
            out_v.append(np.nan)
            out_p.append(0.0)
            continue
        v = np.array([p[0] for p in per_lag])
        peak = float(np.median([p[1] for p in per_lag]))
        bad = any(p[2] for p in per_lag) or peak < min_peak or (v.max() - v.min()) > agree_px
        out_v.append(np.nan if bad else float(np.median(v)) / dt)
        out_p.append(peak)
    return np.array(out_t), np.array(out_v), np.array(out_p)


def build(burst_dir, res_dir, vessels_json, min_length=80.0, max_vessels=40, log=print):
    """Returns (summary, kymographs, times). One pass over the raw frames."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from stabilize.bursts import load_burst, read_frame
    from stabilize.fields import FieldSet

    b = load_burst(burst_dir)
    with open(os.path.join(res_dir, "transforms.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fpath = os.path.join(res_dir, "fields.npz")
    fs = FieldSet(fpath) if os.path.exists(fpath) else None
    with open(vessels_json, encoding="utf-8") as f:
        data = json.load(f)
    ves = sorted((v for v in data["vessels"] if v["length_px"] >= min_length),
                 key=lambda v: -v["length_px"])[:max_vessels]
    lines = {}
    for v in ves:
        pieces = centreline_of(v)
        if pieces:
            lines[v["id"]] = pieces[0]
    log(f"{b.name}: {len(lines)} vessels of {len(data['vessels'])} (length >= {min_length:.0f} px)")

    series = {k: [] for k in lines}
    kymo = {k: [] for k in lines}
    ctrl = {k: [] for k in lines}
    times, used = [], []
    for i in range(len(b)):
        if fs is not None:
            if not (fs.has(i) and fs.is_used(i)):
                continue
            fx, fy = fs.field(i)
        else:
            if rows[i].get("registered") != "1":
                continue
            fx = fy = None
        raw = read_frame(b.files[i]).astype(np.float32)
        A = vimg.flat_absorbance(raw, np.ones(raw.shape, bool))
        for k, (_, C, rad) in lines.items():   # the vessel, then its control
            if fx is None:
                Ci = C + [float(rows[i]["dx_px"]), float(rows[i]["dy_px"])]
            else:
                xi = np.clip(np.rint(C[:, 0]).astype(int), 0, raw.shape[1] - 1)
                yi = np.clip(np.rint(C[:, 1]).astype(int), 0, raw.shape[0] - 1)
                Ci = C + np.stack([fx[yi, xi], fy[yi, xi]], 1)
            p = sample_along(A, Ci, max(rad, 1.5))
            kymo[k].append(p.astype(np.float32))
            series[k].append(float(np.mean(p)))
            # control: the same line pushed sideways into the background. Any
            # speed found there is residual motion or a sampling artefact,
            # not blood: nothing flows along a line beside the vessel.
            tg = np.gradient(Ci, axis=0)
            tg /= np.maximum(np.hypot(tg[:, 0], tg[:, 1]), 1e-9)[:, None]
            off = np.stack([-tg[:, 1], tg[:, 0]], 1) * (max(rad, 1.5) + 12.0)
            ctrl[k].append(sample_along(A, Ci + off, max(rad, 1.5)).astype(np.float32))
        times.append(float(b.times_s[i]))
        used.append(i)
    if len(times) < 4:
        raise SystemExit("too few usable frames")
    t = np.asarray(times) - times[0]
    dt = float(np.median(np.diff(t)))
    summary = {"burst": b.name, "n_frames": len(used), "frames_total": len(b),
               "fps_median": round(1 / dt, 2), "duration_s": round(float(t[-1]), 3), "vessels": {}}
    # Every vessel sees the same slow drift - focus and illumination wander
    # over a burst, and the whole eye moves in and out of the depth of field.
    # The COMMON MODE (the median of the detrended series across vessels,
    # each scaled to unit spread) is that shared signal; regressing it out of
    # each vessel leaves what happened in THAT vessel.
    det = {}
    for k in lines:
        y = np.asarray(series[k], float)
        if len(y) >= 16 and np.isfinite(y).all():
            det[k] = y - np.polyval(np.polyfit(t, y, 2), t)
    if len(det) >= 3:
        stack = np.stack([d / max(d.std(), 1e-12) for d in det.values()])
        common = np.median(stack, 0)
    else:
        common = np.zeros(len(t))
    freqs = np.linspace(0.3, min(0.5 / dt, 12.0), 240)
    card = (freqs >= 0.7) & (freqs <= 3.0)              # plausible heart rates, 42-180 bpm
    spectra, speeds = {}, {}
    for k, d in det.items():
        g = np.dot(d, common) / max(np.dot(common, common), 1e-12)
        resid = d - g * common
        amp = periodogram(t, resid, freqs)
        j = int(np.argmax(amp))
        jc = int(np.flatnonzero(card)[np.argmax(amp[card])])
        spectra[k] = amp
        summary["vessels"][str(k)] = {
            "length_px": next(v["length_px"] for v in ves if v["id"] == k),
            "mean_absorbance": round(float(np.mean(series[k])), 5),
            "rms": round(float(d.std()), 5), "rms_after_common": round(float(resid.std()), 5),
            "common_mode_gain": round(float(g), 4),
            "peak_hz": round(float(freqs[j]), 3), "peak_amp": round(float(amp[j]), 6),
            "peak_over_median": round(float(amp[j] / max(np.median(amp), 1e-12)), 2),
            "cardiac_hz": round(float(freqs[jc]), 3),
            "cardiac_over_median": round(float(amp[jc] / max(np.median(amp), 1e-12)), 2)}
        wt, wv, wp = streak_velocity_xcorr(kymo[k], t)
        ok = np.isfinite(wv)
        summary["vessels"][str(k)].update({
            "speed_windows": int(ok.sum()), "speed_windows_total": int(len(wv)),
            "speed_px_s_median": (round(float(np.median(wv[ok])), 1) if ok.any() else None),
            "speed_px_s_iqr": (round(float(np.subtract(*np.percentile(wv[ok], [75, 25]))), 1)
                               if ok.sum() > 3 else None),
            "match_peak_median": round(float(np.median(wp)), 3) if len(wp) else None})
        ct, cv_, cp = streak_velocity_xcorr(np.asarray(ctrl[k], np.float32), t)
        cok = np.isfinite(cv_)
        summary["vessels"][str(k)].update({
            "control_speed_px_s_median": (round(float(np.median(cv_[cok])), 1) if cok.any() else None),
            "control_windows": int(cok.sum()),
            "control_match_peak_median": round(float(np.median(cp)), 3) if len(cp) else None})
        # A speed is worth using when the vessel clearly beats its own control
        # and the frames actually matched. Vessel 88 of burst 15-22-26 is the
        # case this catches: -47 px/s along the vessel against -76 px/s in the
        # background beside it, which is not flow.
        vs = summary["vessels"][str(k)]
        cs = vs["control_speed_px_s_median"]
        vs["speed_trusted"] = bool(vs["speed_px_s_median"] is not None and cs is not None
                                   and abs(vs["speed_px_s_median"]) > 3 * abs(cs)
                                   and (vs["match_peak_median"] or 0) >= 0.3
                                   and vs["speed_windows"] >= 0.5 * vs["speed_windows_total"])
        speeds[k] = np.stack([wt, wv, wp], 1)
    summary["common_mode_rms"] = round(float(common.std()), 4)
    summary["freqs_hz"] = [round(float(f), 4) for f in freqs]
    return summary, {k: np.asarray(v, np.float32) for k, v in kymo.items()}, t, spectra, common, speeds


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vessels.profiles", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("burst")
    ap.add_argument("--method", default="nonrigid")
    ap.add_argument("--rec-dir", default=os.environ.get("REC_DIR"))
    ap.add_argument("--res-base", default=os.environ.get("RES_BASE"))
    ap.add_argument("--vessels", default=None, help="folder holding vessels.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-length", type=float, default=80.0)
    ap.add_argument("--max-vessels", type=int, default=40)
    a = ap.parse_args(argv)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rec = a.rec_dir or os.path.join(os.path.dirname(repo), "recordings")
    base = a.res_base or os.path.join(os.path.dirname(repo), "stabilization")
    res_dir = os.path.join(base, a.method, a.burst)
    vdir = a.vessels or os.path.join(res_dir, "vessels")
    out = a.out or os.path.join(vdir, "profiles")
    os.makedirs(out, exist_ok=True)
    summary, kymo, t, spectra, common, speeds = build(os.path.join(rec, a.burst), res_dir,
                                              os.path.join(vdir, "vessels.json"),
                                              a.min_length, a.max_vessels)
    np.savez_compressed(os.path.join(out, "kymographs.npz"), times_s=t, common_mode=common,
                        freqs_hz=np.asarray(summary["freqs_hz"]),
                        **{f"v{k}": v for k, v in kymo.items()},
                        **{f"amp{k}": v for k, v in spectra.items()},
                        **{f"speed{k}": v for k, v in speeds.items()})
    with open(os.path.join(out, "profiles.json"), "w") as f:
        json.dump(summary, f, indent=1)
    for k, K in kymo.items():
        g = K - np.median(K, axis=0, keepdims=True)
        lo, hi = np.percentile(g, [1, 99])
        img = np.clip((g - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out, f"kymo_{k:03d}.png"), cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS))
    print(f"{len(kymo)} vessels, {summary['n_frames']} frames, {summary['duration_s']:.2f} s -> {out}")


if __name__ == "__main__":
    main()
