"""The Record button as a TIFF burst: hold, tap-to-latch, lockout against
the fixed-length Capture burst, video mode still working, and closing the
window mid-burst without losing the metadata.

usage: python test_record_burst.py <out_dir>
"""
import glob
import json
import os
import sys
import time

from bench_common import REPO

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
steps = []   # (function, seconds to wait after it)


def step(wait=3.5):
    def deco(fn):
        steps.append((fn, wait))
        return fn
    return deco


def bursts():
    return sorted(glob.glob(os.path.join(OUT, "burst_*")))


def check_burst(label, expect_secs):
    folder = bursts()[-1]
    tifs = glob.glob(os.path.join(folder, "*.tif"))
    man_path = os.path.join(folder, "manifest.json")
    have_manifest = os.path.exists(man_path)
    rate = 0.0
    if have_manifest:
        with open(man_path, encoding="utf-8") as f:
            rate = json.load(f)["capture"]["effective_fps"]
    span = len(tifs) / rate if rate else 0.0
    print(f"{label}: {len(tifs)} tifs at {rate:.1f} fps = {span:.2f} s "
          f"(expected ~{expect_secs} s), manifest={have_manifest}, "
          f"csv={os.path.exists(os.path.join(folder, 'frames.csv'))} | "
          f"info='{win.record_info.text()[:60]}' button='{win.record_btn.text()}'")
    if not tifs:
        fail.append(f"{label}: no frames saved")
    if not have_manifest:
        fail.append(f"{label}: manifest.json missing")
    if rate and abs(span - expect_secs) > 0.6:
        fail.append(f"{label}: burst lasted {span:.2f} s, expected ~{expect_secs}")
    if win.record_btn.text() != "●  Record burst":
        fail.append(f"{label}: button not restored (reads '{win.record_btn.text()}')")


@step()
def setup():
    print("default mode:", win.record_mode.currentData(),
          "| button:", win.record_btn.text(), "| hint:", win.record_hint.text())
    if win.record_mode.currentData() != "burst":
        fail.append("Record button does not default to TIFF bursts")
    formats = [win.depth_combo.itemText(i)
               for i in range(win.depth_combo.count())]
    win.depth_combo.setCurrentIndex(formats.index("Mono12"))
    win._depth_changed(formats.index("Mono12"))


@step()
def configure():
    win.roi_h.setValue(200)
    win._roi_apply()
    win.auto_exp.setChecked(False)
    win.exp_spin.setValue(2000.0)
    win.fps_spin.setValue(150.0)


# ---- hold: capture runs exactly as long as the button is down ------------
@step(wait=2.0)
def hold_press():
    win._record_pressed()
    if not win.recording or not win.bursting:
        fail.append("pressing Record did not start a burst")
    if win.burst_btn.isEnabled():
        fail.append("Capture burst still enabled during a Record burst")


@step(wait=3.0)
def hold_release():
    win._record_released()


@step(wait=0.5)
def hold_check():
    check_burst("hold 2.0 s", 2.0)


# ---- tap: latches until tapped again ---------------------------------------
@step(wait=0.1)
def tap_down():
    win._record_pressed()


@step(wait=1.5)
def tap_up():
    win._record_released()          # quick tap: should keep capturing
    if not win.recording:
        fail.append("a quick tap stopped the burst instead of latching")


@step(wait=0.1)
def second_tap_down():
    win._record_pressed()


@step(wait=3.0)
def second_tap_up():
    win._record_released()


@step(wait=0.5)
def tap_check():
    check_burst("tap-tap ~1.6 s", 1.6)


# ---- lockout the other way: fixed burst blocks the Record button ----------
@step(wait=0.3)
def panel_burst():
    win.burst_unit.setCurrentIndex(0)
    win.burst_frames.setValue(600)
    win._burst_clicked()
    print(f"during Capture burst: record button enabled="
          f"{win.record_btn.isEnabled()}")
    if win.record_btn.isEnabled():
        fail.append("Record button still enabled during a Capture burst")


@step(wait=6.0)
def panel_wait():
    pass


@step(wait=0.5)
def panel_done():
    if win.bursting:
        fail.append("fixed-length burst still running")
    if not win.record_btn.isEnabled():
        fail.append("Record button not re-enabled after Capture burst")


# ---- video mode still works ------------------------------------------------
@step(wait=2.0)
def video_press():
    win.record_mode.setCurrentIndex(1)
    print("video mode button:", win.record_btn.text())
    win._record_pressed()


@step(wait=5.0)
def video_release():
    win._record_released()


@step(wait=0.5)
def video_check():
    mp4s = glob.glob(os.path.join(OUT, "*.mp4"))
    print(f"video mode: {[os.path.basename(p) for p in mp4s]}")
    if not mp4s:
        fail.append("video mode did not produce a video file")
    win.record_mode.setCurrentIndex(0)


# ---- closing mid-burst keeps the metadata ----------------------------------
@step(wait=1.5)
def close_press():
    win._record_pressed()
    time.sleep(0.05)
    win._record_released()          # latch


@step(wait=0.0)
def close_mid_burst():
    folder = bursts()[-1]
    win.close()                     # blocks in _wait_for_burst until saved
    ok = os.path.exists(os.path.join(folder, "manifest.json"))
    tifs = len(glob.glob(os.path.join(folder, "*.tif")))
    print(f"closed mid-burst: {tifs} tifs, manifest saved={ok}")
    if not ok:
        fail.append("closing mid-burst lost manifest.json")


def run_next():
    if not steps:
        print("\nRESULT:", "FAIL " + "; ".join(fail) if fail else "PASS")
        sys.stdout.flush()
        os._exit(1 if fail else 0)
    fn, wait = steps.pop(0)
    print(f"[{time.strftime('%H:%M:%S')}] step {fn.__name__}", flush=True)
    try:
        fn()
    except Exception as exc:
        fail.append(f"{fn.__name__}: {exc}")
    QtCore.QTimer.singleShot(int(wait * 1000), run_next)


QtCore.QTimer.singleShot(3000, run_next)
def on_timeout():
    print("TIMED OUT - visible top-level windows:", flush=True)
    for w in QtWidgets.QApplication.topLevelWidgets():
        if w.isVisible():
            text = w.text() if hasattr(w, "text") and callable(w.text) else ""
            print(f"  {type(w).__name__}: title={w.windowTitle()!r} text={text!r}", flush=True)
    os._exit(1)


QtCore.QTimer.singleShot(120000, on_timeout)
app.exec_()
