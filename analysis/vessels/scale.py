"""Image scale from a grid-target burst: pixels per grid period, and, given
the target's pitch, micrometres per pixel.

Speeds and radii come out of the other modules in pixels. Turning them into
mm/s and micrometres needs one number, and a photograph of a ruled target is
the honest way to get it: the grid's period is a strong, narrow peak in the
power spectrum of the image, so it can be measured to a fraction of a pixel
even when the target is slightly rotated or out of focus.

The peak is found in the 2D power spectrum of the high-passed, windowed
frame, refined to sub-pixel by a parabola through its neighbours in each
direction, and the period is reported per axis together with the grid's
rotation. A frame with no grid gives a weak, broad peak, which is reported
rather than hidden: the peak-to-background ratio is part of the output.

usage:
  python -m vessels.scale <burst or tiff> [--pitch-um 50] [--rec-dir DIR]
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import tifffile


def grid_period(img, min_period=6.0, max_period=200.0):
    """(period_px, angle_deg, peak_ratio) of the strongest periodic pattern."""
    a = np.asarray(img, np.float32)
    a = a - cv2.GaussianBlur(a, (0, 0), max_period / 4)
    a = a * np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1])).astype(np.float32)
    P = np.abs(np.fft.rfft2(a)) ** 2
    ky = np.fft.fftfreq(a.shape[0])[:, None]
    kx = np.fft.rfftfreq(a.shape[1])[None, :]
    k = np.hypot(kx, ky)
    band = (k >= 1 / max_period) & (k <= 1 / min_period)
    Pm = np.where(band, P, 0.0)
    iy, ix = np.unravel_index(int(np.argmax(Pm)), Pm.shape)

    def refine(vals, i):
        if 0 < i < len(vals) - 1:
            y0, y1, y2 = vals[i - 1], vals[i], vals[i + 1]
            d = y0 - 2 * y1 + y2
            return i + (0.5 * (y0 - y2) / d if abs(d) > 1e-12 else 0.0)
        return float(i)

    fy = np.fft.fftfreq(a.shape[0])
    fx = np.fft.rfftfreq(a.shape[1])
    ix_r = refine(Pm[iy, :], ix)
    iy_r = refine(np.roll(Pm[:, ix], -iy + len(fy) // 2), len(fy) // 2) - len(fy) // 2 + iy
    kxr = float(np.interp(ix_r, np.arange(len(fx)), fx))
    kyr = float(fy[int(round(iy_r)) % len(fy)])
    kk = float(np.hypot(kxr, kyr))
    if kk <= 0:
        return None, None, 0.0
    bg = float(np.median(Pm[band]))
    return 1.0 / kk, float(np.degrees(np.arctan2(kyr, kxr))), float(Pm[iy, ix] / max(bg, 1e-12))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vessels.scale", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="burst folder name, or a path to a TIFF")
    ap.add_argument("--rec-dir", default=os.environ.get("REC_DIR"))
    ap.add_argument("--pitch-um", type=float, default=None,
                    help="the target's grid pitch in micrometres; without it only pixels are reported")
    ap.add_argument("--frames", type=int, default=5, help="frames to measure and take the median of")
    ap.add_argument("--out", default=None, help="write scale.json here")
    a = ap.parse_args(argv)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rec = a.rec_dir or os.path.join(os.path.dirname(repo), "recordings")
    if a.source.lower().endswith((".tif", ".tiff")):
        files = [a.source]
    else:
        files = sorted(glob.glob(os.path.join(rec, a.source, "frame_*.tif")))
        if not files:
            files = sorted(glob.glob(os.path.join(a.source, "frame_*.tif")))
        if not files:
            raise SystemExit(f"no frames found for {a.source}")
        step = max(1, len(files) // max(a.frames, 1))
        files = files[::step][:a.frames]
    rows = []
    for f in files:
        p, ang, ratio = grid_period(tifffile.imread(f).astype(np.float32))
        if p:
            rows.append((p, ang, ratio))
            print(f"  {os.path.basename(f)}: period {p:.2f} px, grid at {ang:+.1f} deg, peak/background {ratio:.0f}")
    if not rows:
        raise SystemExit("no periodic pattern found")
    per = float(np.median([r[0] for r in rows]))
    spread = float(np.std([r[0] for r in rows]))
    out = {"source": a.source, "frames": len(rows), "period_px": round(per, 3),
           "period_px_spread": round(spread, 3),
           "angle_deg": round(float(np.median([r[1] for r in rows])), 2),
           "peak_over_background": round(float(np.median([r[2] for r in rows])), 1)}
    if a.pitch_um:
        out["pitch_um"] = a.pitch_um
        out["um_per_px"] = round(a.pitch_um / per, 4)
        print(f"\n{per:.2f} px per {a.pitch_um:g} um grid period  ->  {a.pitch_um / per:.4f} um/px")
        print(f"a speed of 100 px/s is {100 * a.pitch_um / per / 1000:.3f} mm/s")
    else:
        print(f"\ngrid period {per:.2f} px (spread {spread:.2f} over {len(rows)} frames);"
              f" pass --pitch-um to get um/px")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
    return out


if __name__ == "__main__":
    main()
