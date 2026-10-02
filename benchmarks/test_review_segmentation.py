"""Review tab vessel overlay with swappable segmenters, end to end, without a camera.

Copies a burst into a temporary recordings folder, stabilizes it from the
Review tab, then: runs the quick built-in segmenter and checks the overlay
on the Stabilized mean, Stabilized and Raw views; adds a segmenter of its own
through LIMBUS_SEGMENTERS, runs it and switches between the two (each keeps
its result); cancels a run and fails one on purpose, and checks the previous
overlay survives both; edits the segmenter and checks the overlay is flagged
as made by an earlier version; and checks the V key and the Masks toggle.
Screenshots of each overlay are kept for a visual check.

usage: python benchmarks/test_review_segmentation.py <shots dir> <burst folder>
"""
import os
import shutil
import sys
import tempfile
import time

from bench_common import REPO

TMP = tempfile.mkdtemp(prefix="review_segmentation_")
PLUGINS = os.path.join(TMP, "my_segmenters")
os.makedirs(PLUGINS)
os.environ["LIMBUS_SEGMENTERS"] = PLUGINS      # before the app lists the segmenters

sys.path.insert(0, os.path.abspath(REPO))
import camera_recorder as cr  # noqa: E402
from PyQt5 import QtCore, QtTest, QtWidgets  # noqa: E402

SHOTS = os.path.abspath(sys.argv[1])
BURST = os.path.abspath(sys.argv[2])
TIMEOUT_S = 10 * 60
fail = []

PLUGIN = '''"""A test segmenter: three straight lines, a made-up junction kind, a mask."""
import os
import time

LABEL = "Test lines"
DESCRIPTION = "Three straight vessels, a made-up junction kind and a mask."
VERSION = "{version}"


def segment(inputs):
    if os.environ.get("TEST_SEGMENTER_FAIL"):
        raise RuntimeError("asked to fail")
    time.sleep(float(os.environ.get("TEST_SEGMENTER_SLEEP", "0")))
    import numpy as np
    h, w = inputs.shape
    ov = inputs.overlay()
    for k in range(3):
        y = h * (k + 1) / 4
        ov.add_vessel([[10, y], [w - 10, y]], radius=[2 + k, 2 + k], group=k)
    ov.add_junction(w / 2, h / 2, "test kind", [1, 2])
    mask = np.zeros((h, w), bool)
    mask[: h // 2, : w // 2] = True
    ov.add_mask("top-left", mask)
    return ov
'''


class StubMain:
    def __init__(self, output_dir):
        self.output_dir = output_dir


def check(ok, what):
    print(("  ok    " if ok else "  FAIL  ") + what, flush=True)
    if not ok:
        fail.append(what)


def pump(seconds):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.02)


def wait_job():
    started = time.time()
    while tab.proc is not None and time.time() - started < TIMEOUT_S:
        pump(0.2)
    return tab.job_log.toPlainText().splitlines()


def frame_vessels():
    for r in range(tab.frame_form.rowCount()):
        label = tab.frame_form.itemAt(r, QtWidgets.QFormLayout.LabelRole)
        field = tab.frame_form.itemAt(r, QtWidgets.QFormLayout.FieldRole)
        if label and field and label.widget().text() == "Vessels":
            return field.widget().text()
    return None


def legend_texts():
    out = []
    for i in range(tab.legend.count()):
        w = tab.legend.itemAt(i).widget()
        if isinstance(w, QtWidgets.QLabel) and w.text():
            out.append(w.text())
    return out


def choose(sid):
    tab.segmenter_combo.setCurrentIndex(tab.segmenter_combo.findData(sid))
    pump(0.2)


def find_vessels(sid):
    choose(sid)
    tab.vessel_btn.click()
    pump(0.3)
    running = tab.proc is not None and tab.vessel_btn.text() == "Cancel"
    log = wait_job()
    return running, log


def views(label):
    """The overlay on each view; returns the frame panel's Vessels line per view."""
    got = {}
    for view in ("mean", "stabilized", "raw"):
        tab.view_combo.setCurrentIndex(tab.view_combo.findData(view))
        tab.slider.setValue(len(tab.source) // 2)
        pump(0.2)
        got[view] = frame_vessels()
        tab.grab().save(os.path.join(SHOTS, f"{label}_{view}.png"))
    return got


app = QtWidgets.QApplication(sys.argv)
os.makedirs(SHOTS, exist_ok=True)
try:
    with open(os.path.join(PLUGINS, "test_lines.py"), "w", encoding="utf-8") as f:
        f.write(PLUGIN.format(version="1"))
    recordings = os.path.join(TMP, "recordings")
    name = os.path.basename(BURST)
    shutil.copytree(BURST, os.path.join(recordings, name))
    tab = cr.ReviewTab(StubMain(recordings))
    tab.resize(1500, 950)
    tab.show()
    tab.refresh()
    pump(0.3)
    listed = [tab.segmenter_combo.itemData(i) for i in range(tab.segmenter_combo.count())]
    print(f"segmenters listed: {listed}")
    check({"stabilization_mask", "vesselmap", "test_lines"} <= set(listed),
          "the built-in segmenters and the one in LIMBUS_SEGMENTERS are listed")

    print("stabilize first:")
    tab.method_combo.setCurrentIndex(tab.method_combo.findData("nonrigid"))
    choose("stabilization_mask")
    check(not tab.vessel_btn.isEnabled() and "Stabilize with this method first" in tab.vessel_status.text(),
          "Find vessels waits for a stabilization result")
    tab.stab_btn.click()
    pump(0.5)
    check(not tab.vessel_btn.isEnabled() and tab.vessel_status.text().startswith("Busy"),
          "one analysis job at a time: Find vessels waits while stabilizing")
    log = wait_job()
    check(tab.result is not None and tab.result.ok, f"stabilized: {log[-2] if len(log) > 1 else log}")

    print("the quick built-in segmenter:")
    check(tab.vessels is None and tab.vessel_btn.text() == "Find vessels"
          and not tab.vessel_files_btn.isEnabled(), "nothing found yet, no folder to open")
    running, log = find_vessels("stabilization_mask")
    check(running and log[-1] == "finished" and tab.vessels is not None,
          f"found: {tab.vessel_status.text().splitlines()[0] if tab.vessels else log[-3:]}")
    check(tab.vessel_files_btn.isEnabled() and tab.vessel_btn.text() == "Re-run",
          "its result folder can be opened, and the button reads Re-run")
    tab.vessel_parts["masks"].setChecked(True)
    got = views("quick")
    check(all(v and v.endswith("drawn") or (v and "drawn" in v) for v in got.values()),
          f"drawn on every view: {got}")
    check("stabilized views only" in (got["raw"] or ""), "masks are left off raw frames, and it says so")
    check(any("mask" in t for t in legend_texts()), f"legend: {legend_texts()}")

    print("a segmenter of our own, and switching:")
    choose("test_lines")
    check(tab.vessels is None and "Not found with Test lines" in tab.vessel_status.text(),
          "switching to a segmenter that hasn't run shows no overlay")
    running, log = find_vessels("test_lines")
    check(tab.vessels is not None and len(tab.vessels.vessels) == 3, "its overlay is drawn")
    check("test kind" in legend_texts(), f"its own junction kind is in the legend: {legend_texts()}")
    views("test_lines")
    quick_summary = None
    choose("stabilization_mask")
    quick_summary = tab.vessels.summary if tab.vessels else None
    check(quick_summary is not None and "vessels" in quick_summary,
          "switching back shows the quick overlay again, without re-running")
    choose("test_lines")

    print("V toggles the overlay:")
    tab.view_combo.setCurrentIndex(tab.view_combo.findData("stabilized"))
    pump(0.2)
    shown = frame_vessels()
    QtTest.QTest.keyClick(tab, QtCore.Qt.Key_V)
    pump(0.2)
    hidden = frame_vessels()
    QtTest.QTest.keyClick(tab, QtCore.Qt.Key_V)
    pump(0.2)
    check(shown is not None and hidden is None and frame_vessels() is not None,
          f"V hides and shows it ({shown!r} -> {hidden!r})")

    print("cancel and failure keep the previous overlay:")
    made = tab.vessels.made
    os.environ["TEST_SEGMENTER_SLEEP"] = "60"
    tab.vessel_btn.click()
    pump(3.0)
    check(tab.proc is not None and not tab.segmenter_combo.isEnabled(),
          "while it runs the segmenter can't be switched")
    tab.vessel_btn.click()                       # Cancel
    pump(0.5)
    os.environ.pop("TEST_SEGMENTER_SLEEP")
    check(tab.proc is None and tab.vessels is not None and tab.vessels.made == made,
          "cancelled: the previous overlay is still there")
    os.environ["TEST_SEGMENTER_FAIL"] = "1"
    running, log = find_vessels("test_lines")
    os.environ.pop("TEST_SEGMENTER_FAIL")
    check(log[-1] == "exited with code 1" and any("asked to fail" in line for line in log)
          and tab.vessels is not None and tab.vessels.made == made,
          "failed: says why, and the previous overlay is still there")

    print("an edited segmenter:")
    with open(os.path.join(PLUGINS, "test_lines.py"), "w", encoding="utf-8") as f:
        f.write(PLUGIN.format(version="2"))
    tab.refresh()
    pump(0.3)
    check("earlier version of this segmenter" in tab.vessel_status.text(),
          f"its old overlay is flagged: {tab.vessel_status.text()!r}")
    tab.grab().save(os.path.join(SHOTS, "stale_status.png"))
finally:
    if "tab" in globals():
        tab.shutdown()
    shutil.rmtree(TMP, ignore_errors=True)

print(f"\nscreenshots in {SHOTS}")
print("RESULT:", "FAIL " + "; ".join(fail) if fail else "PASS")
sys.exit(1 if fail else 0)
