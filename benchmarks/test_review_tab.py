"""Exercise the new ROI positioning controls and the Review tab: capture a
short burst and a short video, then play them back in the app.

usage: python test_review_tab.py <out_dir>
"""
import glob
import os
import sys

from bench_common import REPO

sys.path.insert(0, os.path.abspath(REPO))
import camera_recorder as cr  # noqa: E402
from PyQt5 import QtCore, QtWidgets  # noqa: E402

OUT = sys.argv[1]
ROWS, OFFSET_Y, FRAMES = 200, 400, 60
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
def pick_mono12():
    formats = [win.depth_combo.itemText(i)
               for i in range(win.depth_combo.count())]
    win.depth_combo.setCurrentIndex(formats.index("Mono12"))
    win._depth_changed(formats.index("Mono12"))


@step
def speed_preset():
    """The 200 fps preset should pick a row count, not just guess."""
    idx = [win.speed_combo.itemData(i)
           for i in range(win.speed_combo.count())].index(200.0)
    win.speed_combo.setCurrentIndex(idx)
    win._speed_preset(idx)


@step
def settle_preset():
    """Auto-exposure needs a moment to come down to the capped limit."""
    status = win.cam_thread.latest_status or {}
    print(f"just after preset: exposure={status.get('exposure_us')} µs, "
          f"ceiling={status.get('max_fps', 0):.1f} fps")


@step
def check_preset():
    status = win.cam_thread.latest_status or {}
    print(f"200 fps preset -> ROI {win.roi}, exposure "
          f"{status.get('exposure_us', 0):.0f} µs, camera says "
          f"{status.get('max_fps', 0):.1f} fps  ({win.roi_hint.text()})")
    if not 150 <= (status.get("max_fps") or 0) <= 250:
        fail.append(f"200 fps preset gave a {status.get('max_fps')} fps ceiling")


@step
def place_band():
    win.roi_h.setValue(ROWS)
    win._roi_apply()
    win.off_y.setValue(OFFSET_Y)


@step
def check_band():
    status = win.cam_thread.latest_status or {}
    print(f"placed band: offset_y={status.get('offset_y')} "
          f"(asked {OFFSET_Y}), height={win.roi[1]}")
    if int(status.get("offset_y", -1)) != OFFSET_Y:
        fail.append(f"ROI top row is {status.get('offset_y')}, "
                    f"expected {OFFSET_Y}")


@step
def capture_burst():
    win.auto_exp.setChecked(False)
    win.exp_spin.setValue(3000.0)
    win.burst_frames.setValue(FRAMES)
    win.cam_thread.burst_finished.connect(lambda r: None)
    win._burst_clicked()


@step
def record_video():
    win.record_mode.setCurrentIndex(win.record_mode.findData("video"))   # Record defaults to TIFF bursts
    win.quality_combo.setCurrentIndex(0)       # H.264, decodable by OpenCV
    win._toggle_record()


@step
def stop_video():
    win._toggle_record()


@step
def check_listing():
    win.review.refresh()
    labels = [win.review.listing.item(i).text()
              for i in range(win.review.listing.count())]
    print(f"review listing: {labels}")
    if not any(l.startswith("burst_") for l in labels):
        fail.append("burst missing from the Review listing")
    if not any(l.endswith(".mp4") for l in labels):
        fail.append("video missing from the Review listing")


@step
def play_burst():
    burst = sorted(glob.glob(os.path.join(OUT, "burst_*")))[-1]
    win.review.load(burst)
    src = win.review.source
    print(f"burst source: {type(src).__name__}, {len(src)} frames, "
          f"{src.bit_depth}-bit, {src.fps:.1f} fps")
    print(f"  summary: {src.summary()[:4]}")
    print(f"  frame 0 metadata: {src.meta(0)}")
    if not isinstance(src, cr.BurstSource):
        fail.append("burst did not load as a BurstSource")
    elif len(src) != FRAMES:
        fail.append(f"burst has {len(src)} frames, expected {FRAMES}")
    if not src.meta(0):
        fail.append("no per-frame metadata in the Review tab")
    win.review._step(5)
    print(f"  after stepping: counter={win.review.counter.text()}, "
          f"pixmap={'yes' if win.review.view.pixmap() else 'no'}")
    if win.review.counter.text() != f"6 / {FRAMES}":
        fail.append(f"frame counter reads {win.review.counter.text()}")
    if not win.review.view.pixmap():
        fail.append("nothing rendered in the Review view")
    win.review.auto_contrast.setChecked(True)   # must not crash
    win.review._toggle_play()


@step
def check_playing():
    if not win.review.timer.isActive():
        fail.append("playback did not start")
    win.review._toggle_play()
    if win.review.timer.isActive():
        fail.append("playback did not stop")


@step
def play_video():
    mp4 = sorted(glob.glob(os.path.join(OUT, "*.mp4")))[-1]
    win.review.load(mp4)
    src = win.review.source
    if src is None:
        fail.append(f"could not open {os.path.basename(mp4)}: "
                    f"{win.review.view.text()}")
        return
    print(f"video source: {type(src).__name__}, {len(src)} frames, "
          f"{src.fps:.1f} fps, rendered="
          f"{'yes' if win.review.view.pixmap() else 'no'}")
    if not win.review.view.pixmap():
        fail.append("video frame did not render")


@step
def undecodable_video():
    """An FFV1/HEVC file should explain itself, not crash."""
    quality = cr.QUALITY_PRESETS[4][1]
    path = os.path.join(OUT, "fake_lossless" + cr.video_extension(quality))
    w = cr.VideoWriter(path, 256, 64, 10.0, False, quality, bit16=True)
    import numpy as np
    for _ in range(10):
        w.write((np.random.rand(64, 256) * 4095).astype(np.uint16) << 4)
    w.close()
    win.review.load(path)
    print(f"FFV1 file -> source={win.review.source}, "
          f"message={win.review.view.text()[:60]!r}")
    if win.review.source is None and not win.review.view.text():
        fail.append("no explanation shown for an undecodable file")


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
    QtCore.QTimer.singleShot(3500, run_next)


QtCore.QTimer.singleShot(3000, run_next)
QtCore.QTimer.singleShot(180000, lambda: (fail.append("timed out"), os._exit(1)))
app.exec_()
