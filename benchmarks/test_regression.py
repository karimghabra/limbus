"""Regression pass over the paths the burst work touched:
full-frame ROI, H.264 recording, and switching pixel format to Mono8.

usage: python test_regression.py <output_dir>
"""
import glob
import os
import sys
import time

from bench_common import REPO, probe

sys.path.insert(0, os.path.abspath(REPO))
import camera_recorder as cr  # noqa: E402
from PyQt5 import QtCore, QtWidgets  # noqa: E402

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
app = QtWidgets.QApplication(sys.argv)
win = cr.MainWindow()
win.output_dir = OUT
win.show()
fail = []
steps = []


def step(fn):
    steps.append(fn)
    return fn


@step
def full_frame():
    win._roi_full()
    print(f"after Full: roi={win.roi} hint={win.roi_hint.text()}")
    d = win._cam_info
    if win.roi != (d["width_default"], d["height_default"]):
        fail.append(f"Full gave {win.roi}, expected the specified imaging "
                    f"area {(d['width_default'], d['height_default'])}")


@step
def check_mono12_ceiling():
    ceiling = (win.cam_thread.latest_status or {}).get("max_fps")
    print(f"Mono12 full-frame ceiling: {ceiling}")
    if not ceiling or not 30 < ceiling < 34:
        fail.append(f"Mono12 ceiling {ceiling}, expected ~32 fps")


@step
def long_exposure():
    """A long exposure limits the frame rate; the ceiling must follow it."""
    win.auto_exp.setChecked(False)
    win.exp_spin.setValue(60000.0)


@step
def check_exposure_ceiling():
    status = win.cam_thread.latest_status or {}
    print(f"60 ms exposure -> ceiling={status.get('max_fps')} "
          f"hint='{win.exp_hint.text()}'")
    if not status.get("max_fps") or status["max_fps"] > 20:
        fail.append(f"ceiling {status.get('max_fps')} did not follow the "
                    "60 ms exposure (expected ~16 fps)")
    if "limits the camera" not in win.exp_hint.text():
        fail.append("no warning that exposure is limiting the frame rate")


@step
def short_exposure():
    win.exp_spin.setValue(5000.0)


@step
def check_ceiling_recovers():
    status = win.cam_thread.latest_status or {}
    print(f"5 ms exposure -> ceiling={status.get('max_fps')}")
    if not status.get("max_fps") or status["max_fps"] < 30:
        fail.append(f"ceiling {status.get('max_fps')} did not recover after "
                    "shortening the exposure")


@step
def record_video():
    win.fps_spin.setValue(25.0)
    win._toggle_record()          # start
    time.sleep(0.1)


@step
def stop_video():
    win._toggle_record()          # stop -> background finalize


@step
def check_video():
    mp4s = sorted(glob.glob(os.path.join(OUT, "*.mp4")))
    if not mp4s:
        fail.append("no mp4 written")
        return
    p = probe(mp4s[-1])
    print(f"video: {os.path.basename(mp4s[-1])} {p.get('pix_fmt')} "
          f"bits={p.get('bits_per_raw_sample')} frames={p.get('nb_read_frames')} "
          f"{p.get('width')}x{p.get('height')}")
    if int(p.get("nb_read_frames", 0) or 0) < 10:
        fail.append("video has too few frames")


@step
def to_mono8():
    formats = [win.depth_combo.itemText(i)
               for i in range(win.depth_combo.count())]
    idx = formats.index("Mono8")
    win.depth_combo.setCurrentIndex(idx)
    win._depth_changed(idx)


@step
def mono8_fix_exposure():
    """Reconnecting reloads factory defaults, so auto-exposure is back on and
    will have chosen a long exposure in a dark scene. Pin it before asking
    what the pixel format alone allows."""
    win.auto_exp.setChecked(False)
    win.exp_spin.setValue(5000.0)


@step
def check_mono8():
    frame = win.cam_thread.latest_frame
    info = win._cam_info
    print(f"Mono8: dtype={frame.dtype} shape={frame.shape} "
          f"format={info['pixel_format']} hint={win.roi_hint.text()}")
    if info["pixel_format"] != "Mono8":
        fail.append("did not switch to Mono8")
    if frame.dtype.name != "uint8":
        fail.append(f"Mono8 frames should be uint8, got {frame.dtype}")
    ceiling = (win.cam_thread.latest_status or {}).get("max_fps")
    print(f"Mono8 full-frame ceiling: {ceiling}")
    if not ceiling or ceiling < 38:
        fail.append(f"Mono8 ceiling {ceiling}, expected ~41 fps "
                    "(stale from the Mono12 setting?)")


@step
def snapshot():
    win._snapshot()
    pngs = glob.glob(os.path.join(OUT, "*.png"))
    print(f"snapshot: {[os.path.basename(p) for p in pngs]}")
    if not pngs:
        fail.append("snapshot wrote nothing")


def run_next():
    if not steps:
        print("\nRESULT:", "FAIL " + "; ".join(fail) if fail else "PASS")
        if win.cam_thread:
            win.cam_thread.shutdown()
        sys.stdout.flush()
        os._exit(1 if fail else 0)
    fn = steps.pop(0)
    try:
        fn()
    except Exception as exc:
        fail.append(f"{fn.__name__}: {exc}")
    # generous gaps: reconnects restart the stream, recordings finalize
    QtCore.QTimer.singleShot(4000, run_next)


QtCore.QTimer.singleShot(3000, run_next)
QtCore.QTimer.singleShot(120000,
                         lambda: (fail.append("timed out"), os._exit(1)))
app.exec_()
