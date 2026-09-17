"""Drive the real GUI through a Mono12 burst and verify what lands on disk.

usage: python test_burst_gui.py <output_dir> [frames] [fps] [roi_height]
"""
import glob
import json
import os
import sys

from bench_common import REPO

sys.path.insert(0, os.path.abspath(REPO))
import camera_recorder as cr  # noqa: E402
from PyQt5 import QtCore, QtWidgets  # noqa: E402

OUT = sys.argv[1]
FRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 200
FPS = float(sys.argv[3]) if len(sys.argv) > 3 else 40.0
ROI_H = int(sys.argv[4]) if len(sys.argv) > 4 else 958
# exposure must fit inside the frame period, or it caps the rate instead
EXPOSURE_US = float(sys.argv[5]) if len(sys.argv) > 5 else 5000.0

app = QtWidgets.QApplication(sys.argv)
win = cr.MainWindow()
win.output_dir = OUT
os.makedirs(OUT, exist_ok=True)
win.show()
fail = []


def verify(result):
    d = result["directory"]
    tiffs = sorted(glob.glob(os.path.join(d, "*.tif")))
    print(f"\nburst finished: written={result['written']} "
          f"dropped={result['dropped']} "
          f"effective_fps={result['effective_fps']:.2f} errors={result['errors'][:1]}")
    print(f"files on disk: {len(tiffs)} tiff")
    if len(tiffs) != result["written"]:
        fail.append("tiff count != frames written")
    if result["dropped"]:
        fail.append(f"{result['dropped']} frames dropped")

    import tifffile
    arr = tifffile.imread(tiffs[0])
    print(f"first frame: dtype={arr.dtype} shape={arr.shape} "
          f"min={arr.min()} max={arr.max()}")
    if arr.dtype.name != "uint16":
        fail.append("frames are not uint16")
    if int(arr.max()) > 4095:
        fail.append(f"value {arr.max()} exceeds 12-bit range: not right-aligned")
    if arr.shape[0] != ROI_H:
        fail.append(f"height {arr.shape[0]} != requested ROI {ROI_H}")

    with tifffile.TiffFile(tiffs[0]) as tf:
        desc = tf.pages[0].description
    per_frame = json.loads(desc)
    print(f"embedded per-frame tags: {sorted(per_frame)}")
    for key in ("index", "host_time_utc"):
        if key not in per_frame:
            fail.append(f"per-frame metadata missing {key}")

    man_path = os.path.join(d, "manifest.json")
    with open(man_path, encoding="utf-8") as f:
        man = json.load(f)
    print("manifest sections:", sorted(man))
    acq = man.get("acquisition_requested", {})
    start = man.get("camera_state_at_start", {})
    end = man.get("camera_state_at_end", {})
    print(f"  requested: pixel_format={acq.get('pixel_format')} "
          f"{acq.get('width')}x{acq.get('height')} "
          f"bit_depth={acq.get('bit_depth')} fps={acq.get('requested_fps')} "
          f"auto_brightness={acq.get('auto_brightness_enabled')} "
          f"chunks={acq.get('per_frame_chunks')}")
    print(f"  state at start: exposure_us={start.get('exposure_us')} "
          f"gain={start.get('gain')} exposure_auto={start.get('exposure_auto')} "
          f"gain_auto={start.get('gain_auto')} "
          f"readout_us={start.get('sensor_readout_us')} "
          f"max_fps_estimate={start.get('max_fps_estimate')} "
          f"temp={start.get('temperature_c')}")
    print(f"  state at end:   exposure_us={end.get('exposure_us')} "
          f"gain={end.get('gain')} temp={end.get('temperature_c')}")
    print(f"  camera={man.get('camera')}")
    print(f"  capture={man.get('capture')}")
    print(f"  pixel_values={man.get('pixel_values')}")
    print(f"  software={man.get('software')}")
    if acq.get("pixel_format") != "Mono12":
        fail.append("manifest pixel_format is not Mono12")
    if not start:
        fail.append("manifest has no camera_state_at_start")
    if start.get("exposure_auto") != "Off":
        fail.append(f"stale metadata: exposure_auto={start.get('exposure_auto')}"
                    " but auto was switched off before the burst")
    if start.get("exposure_us") not in (None,) and \
            abs(start["exposure_us"] - EXPOSURE_US) > 1.0:
        fail.append(f"stale metadata: exposure_us={start['exposure_us']}, "
                    f"expected {EXPOSURE_US}")
    if not man.get("camera_state_at_end"):
        fail.append("manifest has no camera_state_at_end")
    if any(k.startswith("_") for k in man):
        fail.append("internal bookkeeping keys leaked into the manifest")

    csv_path = os.path.join(d, "frames.csv")
    with open(csv_path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    print(f"frames.csv: {len(lines) - 1} rows, columns: {lines[0]}")
    print(f"  row 1: {lines[1]}")
    print(f"  row {len(lines) - 1}: {lines[-1]}")
    if len(lines) - 1 != result["written"]:
        fail.append("frames.csv rows != frames written")
    # allow 2%: the camera's ceiling is rarely exactly the requested rate
    if result["effective_fps"] < FPS * 0.98:
        fail.append(f"ran at {result['effective_fps']:.2f} fps, wanted {FPS}")


def done(result):
    try:
        verify(result)
    except Exception as exc:
        fail.append(f"verification error: {exc}")
    print("\nRESULT:", "FAIL " + "; ".join(fail) if fail else "PASS")
    app.quit()


def pick_format():
    formats = [win.depth_combo.itemText(i)
               for i in range(win.depth_combo.count())]
    print("pixel formats offered by the GUI:", formats)
    idx = formats.index("Mono12")
    win.depth_combo.setCurrentIndex(idx)
    win._depth_changed(idx)          # reconnects the camera


def set_roi():
    win.roi_w.setValue(win.roi_w.maximum())
    win.roi_h.setValue(ROI_H)
    win._roi_apply()


def start():
    win.cam_thread.burst_finished.connect(done)
    win.auto_exp.setChecked(False)
    win.exp_spin.setValue(EXPOSURE_US)
    win.fps_spin.setValue(FPS)
    print(f"fps spin: value={win.fps_spin.value()} max={win.fps_spin.maximum()}")
    if win.fps_spin.value() < FPS:
        print(f"WARNING: frame rate clamped to {win.fps_spin.value()}")
    win.burst_frames.setValue(FRAMES)
    print(f"roi hint: {win.roi_hint.text()}")
    print(f"burst hint: {win.burst_info.text()}")
    print("live values:",
          {k: v.text() for k, v in win.live_labels.items()})
    win._burst_clicked()


QtCore.QTimer.singleShot(2500, pick_format)
QtCore.QTimer.singleShot(7000, set_roi)
QtCore.QTimer.singleShot(10000, start)
QtCore.QTimer.singleShot(120000, lambda: (fail.append("timed out"), app.quit()))
app.exec_()
if win.cam_thread:
    win.cam_thread.shutdown()
# skip interpreter finalization: pylon teardown at exit can fault after the
# camera is already closed cleanly (the app's main() does the same)
sys.stdout.flush()
os._exit(1 if fail else 0)
