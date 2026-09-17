"""Capture a short burst, then screenshot both tabs for documentation."""
import glob
import os
import sys

from bench_common import REPO

sys.path.insert(0, os.path.abspath(REPO))
import camera_recorder as cr  # noqa: E402
from PyQt5 import QtCore, QtWidgets  # noqa: E402

OUT = sys.argv[1]
SHOTS = sys.argv[2]
os.makedirs(OUT, exist_ok=True)
app = QtWidgets.QApplication(sys.argv)
win = cr.MainWindow()
win.output_dir = OUT
win.resize(1180, 720)
win.show()
steps = []


def step(fn):
    steps.append(fn)
    return fn


@step
def configure():
    formats = [win.depth_combo.itemText(i)
               for i in range(win.depth_combo.count())]
    idx = formats.index("Mono12")
    win.depth_combo.setCurrentIndex(idx)
    win._depth_changed(idx)


@step
def strip():
    i = [win.speed_combo.itemData(n)
         for n in range(win.speed_combo.count())].index(400.0)
    win.speed_combo.setCurrentIndex(i)
    win._speed_preset(i)
    win.off_y.setValue(600)


@step
def burst():
    win.burst_frames.setValue(400)
    win._burst_clicked()


@step
def shoot_camera_tab():
    win.grab().save(os.path.join(SHOTS, "tab_camera.png"))
    print("camera tab:", win.roi_hint.text(), "|", win.burst_info.text())


@step
def open_review():
    win.tabs.setCurrentIndex(1)
    burst_dir = sorted(glob.glob(os.path.join(OUT, "burst_*")))[-1]
    win.review.load(burst_dir)
    win.review.auto_contrast.setChecked(True)
    win.review._step(40)


@step
def shoot_review_tab():
    win.grab().save(os.path.join(SHOTS, "tab_review.png"))
    print("review tab:", win.review.counter.text(),
          win.review.source.summary()[:3])


def run_next():
    if not steps:
        if win.cam_thread:
            win.cam_thread.shutdown()
        sys.stdout.flush()
        os._exit(0)
    fn = steps.pop(0)
    try:
        fn()
    except Exception as exc:
        print(f"{fn.__name__} failed: {exc}")
    QtCore.QTimer.singleShot(3500, run_next)


QtCore.QTimer.singleShot(3000, run_next)
app.exec_()
