"""Record from the camera with every quality preset and verify what lands on
disk: real bit depth, frame count, dropped frames, and that the frame-rate
restamp works in Matroska as well as MP4.

usage: python test_codecs_camera.py <out_dir> [seconds] [roi_height]
"""
import glob
import os
import subprocess
import sys

from bench_common import REPO

sys.path.insert(0, os.path.abspath(REPO))
import camera_recorder as cr  # noqa: E402
from PyQt5 import QtCore, QtWidgets  # noqa: E402

OUT = sys.argv[1]
SECS = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
ROI_H = int(sys.argv[3]) if len(sys.argv) > 3 else 958
os.makedirs(OUT, exist_ok=True)

app = QtWidgets.QApplication(sys.argv)
win = cr.MainWindow()
win.output_dir = OUT
win.show()
fail = []
state = {}


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=codec_name,pix_fmt,bits_per_raw_sample,"
         "nb_read_frames,avg_frame_rate:format=duration",
         "-of", "default=nw=1", path], capture_output=True, text=True).stdout
    return dict(l.split("=", 1) for l in out.splitlines() if "=" in l)


def setup():
    formats = [win.depth_combo.itemText(i)
               for i in range(win.depth_combo.count())]
    idx = formats.index("Mono12")
    win.depth_combo.setCurrentIndex(idx)
    win._depth_changed(idx)


def configure():
    win.roi_h.setValue(ROI_H)
    win._roi_apply()
    win.auto_exp.setChecked(False)
    win.exp_spin.setValue(5000.0)
    win.fps_spin.setValue(40.0)


def presets():
    return [(i, cr.QUALITY_PRESETS[i][0]) for i in (0, 3, 4)]


queue = list(presets())


def start_one():
    if not queue:
        finish()
        return
    idx, label = queue[0]
    win.record_mode.setCurrentIndex(win.record_mode.findData("video"))   # Record defaults to TIFF bursts
    win.quality_combo.setCurrentIndex(idx)
    win._update_rate_warning()
    print(f"\n=== {label} === warning: "
          f"{win.rate_warning.text() if win.rate_warning.isVisible() else '(none)'}")
    win._toggle_record()
    state["writer"] = win.cam_thread.writer
    QtCore.QTimer.singleShot(int(SECS * 1000), stop_one)


def stop_one():
    win._toggle_record()
    QtCore.QTimer.singleShot(1500, wait_finish)


def wait_finish():
    if win._finishing is not None:
        QtCore.QTimer.singleShot(1000, wait_finish)
        return
    idx, label = queue.pop(0)
    writer = state["writer"]
    p = probe(writer.path)
    dur = float(p.get("duration", 0) or 0)
    frames = int(p.get("nb_read_frames", 0) or 0)
    print(f"{label}: {os.path.basename(writer.path)}")
    print(f"  written={writer.frames_written} dropped={writer.frames_dropped} "
          f"declared_fps={writer.fps:.2f}")
    print(f"  ffprobe codec={p.get('codec_name')} pix_fmt={p.get('pix_fmt')} "
          f"bits={p.get('bits_per_raw_sample')} frames={frames} "
          f"duration={dur:.2f}s -> {frames / dur if dur else 0:.2f} fps")
    codec = cr.codec_for(cr.QUALITY_PRESETS[idx][1])
    if frames != writer.frames_written:
        fail.append(f"{label}: file has {frames} frames, writer wrote "
                    f"{writer.frames_written}")
    depth = p.get("pix_fmt", "")
    if codec["bits"] >= 12 and not ("12" in depth or "16" in depth):
        fail.append(f"{label}: expected >=12-bit, file is {depth}")
    if codec["bits"] == 10 and "10" not in depth:
        fail.append(f"{label}: expected 10-bit, file is {depth}")
    # playback speed must match the rate frames really arrived at
    if dur and writer.frames_written > 8:
        real = writer.frames_written / dur
        span = (writer.t_last - writer.t_first) if writer.t_first else 0
        arrived = (writer.frames_written - 1) / span if span else 0
        print(f"  arrived at {arrived:.2f} fps, file plays at {real:.2f} fps")
        if arrived and abs(real - arrived) / arrived > 0.10:
            fail.append(f"{label}: file plays at {real:.2f} fps but frames "
                        f"arrived at {arrived:.2f} fps (restamp failed)")
    QtCore.QTimer.singleShot(1500, start_one)


def finish():
    print("\nRESULT:", "FAIL " + "; ".join(fail) if fail else "PASS")
    if win.cam_thread:
        win.cam_thread.shutdown()
    sys.stdout.flush()
    os._exit(1 if fail else 0)


QtCore.QTimer.singleShot(2500, setup)
QtCore.QTimer.singleShot(7000, configure)
QtCore.QTimer.singleShot(10000, start_one)
QtCore.QTimer.singleShot(240000, lambda: (fail.append("timed out"), finish()))
app.exec_()
