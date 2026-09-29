"""The Camera tab's stabilized live view, end to end, without a camera.

A replay camera plays a TIFF burst at its recorded frame rate, as the Basler
delivers it (16-bit, MSB-aligned), through the app's own grab thread and
preview. The test records the preview images the app actually shows: raw for
a few seconds, then with "Stabilize the live view" on. Each shown image is
phase-correlated with the first of its stretch: raw, they wander with the eye;
stabilized, they should hold still. Then Re-lock, switching off (the worker
process must exit), and switching on without a GPU (it must say so and
switch itself off).

usage: python benchmarks/test_live_view.py <shots dir> <burst folder>
"""
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
import tifffile

from bench_common import REPO

sys.path.insert(0, os.path.abspath(REPO))
import camera_recorder as cr  # noqa: E402
from PyQt5 import QtWidgets  # noqa: E402

SHOTS = os.path.abspath(sys.argv[1])
BURST = os.path.abspath(sys.argv[2])
fail = []


def check(ok, what):
    print(("  ok    " if ok else "  FAIL  ") + what, flush=True)
    if not ok:
        fail.append(what)


class ReplayCamera:
    """A camera backend that plays a burst in a loop at its recorded rate."""

    def __init__(self, frames, fps):
        self.frames, self.fps = frames, fps
        self.i = 0
        self.t_next = None

    def open(self):
        h, w = self.frames.shape[1:]
        return {"model": "Replay", "serial": os.path.basename(BURST), "width": w, "height": h,
                "width_max": w, "height_max": h, "is_color": False, "bit16": True,
                "pixel_format": "Mono12", "pixel_formats": ["Mono12"], "bit_depth": 12,
                "can12": True, "exposure_us": 10000, "exposure_min": 20, "exposure_max": 1e6,
                "gain": 10, "gain_min": 0, "gain_max": 36, "fps": self.fps,
                "fps_max": self.fps, "auto_exposure": False}

    def read(self):
        now = time.perf_counter()
        if self.t_next is None:
            self.t_next = now
        if now < self.t_next:
            time.sleep(self.t_next - now)
        self.t_next += 1.0 / self.fps
        frame = self.frames[self.i % len(self.frames)]
        self.i += 1
        return frame

    def set(self, key, value):
        pass

    def resulting_fps(self):
        return self.fps

    def close(self):
        pass


def pump(app, seconds):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.005)


def wait_for(app, cond, seconds):
    end = time.time() + seconds
    while time.time() < end and not cond():
        pump(app, 0.05)
    return cond()


def wander(shown, scale):
    """How far each shown image lies from the first (full-resolution px)."""
    ref = shown[0].astype(np.float32)
    win = cv2.createHanningWindow((ref.shape[1], ref.shape[0]), cv2.CV_32F)
    out = []
    for img in shown[1:]:
        (dx, dy), _ = cv2.phaseCorrelate(ref.copy(), img.astype(np.float32), win)
        out.append(np.hypot(dx, dy) / scale)
    return np.array(out)


def main():
    files = sorted(glob.glob(os.path.join(BURST, "frame_*.tif")))
    frames = np.stack([tifffile.imread(f) for f in files]).astype(np.uint16) << 4
    with open(os.path.join(BURST, "manifest.json"), encoding="utf-8") as f:
        fps = float(json.load(f)["capture"]["effective_fps"])
    cr.make_backend = lambda desc: ReplayCamera(frames, fps)
    os.makedirs(SHOTS, exist_ok=True)
    app = QtWidgets.QApplication(sys.argv)
    win = cr.MainWindow()
    win.resize(1600, 1000)
    win.show()
    shown = []
    display = win._display
    win._display = lambda img: (shown.append(img.copy()), display(img))
    desc = {"type": "replay", "label": "Replay"}
    win.cameras = [desc]
    win._connect_camera(desc)
    check(wait_for(app, lambda: win.cam_thread and win.cam_thread.latest_frame is not None, 10),
          f"the replay camera streams {frames.shape[2]}x{frames.shape[1]} at {fps:.0f} fps")
    scale = None

    print("raw:")
    pump(app, 1.0)
    shown.clear()
    pump(app, 3.0)
    raw = list(shown)
    scale = raw[0].shape[1] / frames.shape[2]
    w_raw = wander(raw, scale)
    win.grab().save(os.path.join(SHOTS, "live_raw.png"))
    print(f"  {len(raw)} previews shown; wander from the first: median {np.median(w_raw):.1f} px, "
          f"max {w_raw.max():.1f} px")

    print("stabilized:")
    t0 = time.time()
    win.live_check.setChecked(True)
    locked = wait_for(app, lambda: win._live_locked, 30)
    check(locked, f"locks on ({time.time() - t0:.1f} s from switching on): "
                  f"{win.live_status.text().splitlines()[0]!r}")
    shown.clear()
    pump(app, 4.0)
    stab = list(shown)
    status = win.live_status.text()
    w_stab = wander(stab, scale)
    win.grab().save(os.path.join(SHOTS, "live_stabilized.png"))
    print(f"  {len(stab)} previews shown; wander from the first: median {np.median(w_stab):.2f} px, "
          f"max {w_stab.max():.2f} px; status {status!r}")
    check(len(stab) >= 0.5 * len(raw), f"the view keeps updating ({len(stab)} vs {len(raw)} previews raw)")
    # still to within a pixel, and, where the eye did move, most of its
    # motion removed (a steady burst leaves little to remove)
    moved = np.median(w_raw) > 2.0
    check(np.median(w_stab) < 1.0 and (not moved or np.median(w_stab) < 0.2 * np.median(w_raw)),
          f"stabilized previews hold still: median wander {np.median(w_stab):.2f} px vs "
          f"{np.median(w_raw):.1f} px raw")
    try:
        rate = int(status.split("·")[1].split("of")[0])
    except (IndexError, ValueError):
        rate = 0
    check(rate >= 0.9 * fps and win.live.skipped == 0,
          f"every frame is stabilized: {rate} fps of {fps:.0f}, {win.live.skipped} skipped")

    print("re-lock:")
    win.relock_btn.click()
    relocking = wait_for(app, lambda: not win._live_locked, 5)
    check(relocking and wait_for(app, lambda: win._live_locked, 20),
          "Re-lock locks on afresh")

    print("off:")
    client = win.live
    win.live_check.setChecked(False)
    check(win.live is None and win.cam_thread.live is None and win.live_status.text() == "Off",
          "switching off detaches the stabilizer")
    check(wait_for(app, lambda: client.proc.poll() is not None, 10),
          f"its worker process exits (code {client.proc.poll()})")
    shown.clear()
    pump(app, 1.0)
    check(len(shown) > 10, "the raw view carries on")

    print("no GPU:")
    os.environ["STABILIZE_DEVICE"] = "cpu"
    win.live_check.setChecked(True)
    gave_up = wait_for(app, lambda: not win.live_check.isChecked(), 30)
    os.environ.pop("STABILIZE_DEVICE")
    check(gave_up and "GPU" in win.live_status.text(),
          f"says it needs a GPU and switches off: {win.live_status.text()!r}")
    win.grab().save(os.path.join(SHOTS, "live_no_gpu.png"))
    win.close()
    print("\nRESULT:", "FAIL " + "; ".join(fail) if fail else "PASS")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
