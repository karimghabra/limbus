"""How much RAM does a burst actually cost?

The design intent is that a burst streams to disk rather than accumulating in
memory: the queue is bounded, so RAM should stay flat no matter how long the
burst runs. Measure peak process RSS during a long burst to check that.

usage: python test_burst_memory.py <out_dir> [frames] [fps] [roi_h] [exp_us]
"""
import os
import sys
import threading
import time

import psutil

from bench_common import REPO

sys.path.insert(0, os.path.abspath(REPO))
import camera_recorder as cr  # noqa: E402
from PyQt5 import QtCore, QtWidgets  # noqa: E402

OUT = sys.argv[1]
FRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
FPS = float(sys.argv[3]) if len(sys.argv) > 3 else 400.0
ROI_H = int(sys.argv[4]) if len(sys.argv) > 4 else 60
EXP = float(sys.argv[5]) if len(sys.argv) > 5 else 2000.0

os.makedirs(OUT, exist_ok=True)
app = QtWidgets.QApplication(sys.argv)
win = cr.MainWindow()
win.output_dir = OUT
win.show()

proc = psutil.Process()
samples = []
stop = threading.Event()


def sample():
    while not stop.wait(0.25):
        samples.append(proc.memory_info().rss / 1e6)


threading.Thread(target=sample, daemon=True).start()
baseline = {}


def pick():
    formats = [win.depth_combo.itemText(i)
               for i in range(win.depth_combo.count())]
    idx = formats.index("Mono12")
    win.depth_combo.setCurrentIndex(idx)
    win._depth_changed(idx)


def configure():
    win.roi_h.setValue(ROI_H)
    win._roi_apply()
    win.auto_exp.setChecked(False)
    win.exp_spin.setValue(EXP)
    win.fps_spin.setValue(FPS)


def start():
    baseline["rss"] = proc.memory_info().rss / 1e6
    baseline["at"] = len(samples)
    print(f"RSS before burst: {baseline['rss']:.0f} MB")
    win.cam_thread.burst_finished.connect(done)
    win.burst_frames.setValue(FRAMES)
    win._burst_clicked()


def done(result):
    stop.set()
    time.sleep(0.4)
    during = samples[baseline["at"]:] or samples
    frame_bytes = 1936 * ROI_H * 2
    queue_cap = max(8, int(cr.BURST_QUEUE_BYTES // frame_bytes))
    print(f"burst: written={result['written']} dropped={result['dropped']} "
          f"at {result['effective_fps']:.1f} fps")
    print(f"frame = {frame_bytes / 1e6:.2f} MB; queue holds up to "
          f"{queue_cap} frames = {queue_cap * frame_bytes / 1e6:.0f} MB")
    print(f"RSS during burst: min {min(during):.0f} MB, "
          f"peak {max(during):.0f} MB, "
          f"rise over baseline {max(during) - baseline['rss']:.0f} MB")
    print(f"data written to disk: "
          f"{result['written'] * frame_bytes / 1e9:.2f} GB")
    app.quit()


QtCore.QTimer.singleShot(2500, pick)
QtCore.QTimer.singleShot(7000, configure)
QtCore.QTimer.singleShot(10000, start)
app.exec_()
if win.cam_thread:
    win.cam_thread.shutdown()
sys.stdout.flush()
os._exit(0)
