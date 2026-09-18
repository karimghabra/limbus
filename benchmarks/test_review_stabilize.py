"""Review tab stabilization, end to end, without a camera.

Copies the given bursts into a temporary recordings folder, opens the Review
tab on its own, and for each burst and method: presses Stabilize, waits for
the analysis process, then plays the Raw / Stabilized / Stabilized mean
views. Also cancels a run part-way and checks the summary covers every
burst. Screenshots of each view are kept for a visual check.

usage: python benchmarks/test_review_stabilize.py <shots dir> <burst folder> [<burst folder> ...]
"""
import glob
import os
import shutil
import sys
import tempfile
import time

from bench_common import REPO

sys.path.insert(0, os.path.abspath(REPO))
import camera_recorder as cr  # noqa: E402
from PyQt5 import QtWidgets  # noqa: E402

SHOTS = os.path.abspath(sys.argv[1])
BURSTS = [os.path.abspath(p) for p in sys.argv[2:]]
TIMEOUT_S = 20 * 60
fail = []


class StubMain:
    """All the Review tab needs from the main window."""

    def __init__(self, output_dir):
        self.output_dir = output_dir


def form_rows(form):
    rows = {}
    for r in range(form.rowCount()):
        label = form.itemAt(r, QtWidgets.QFormLayout.LabelRole)
        field = form.itemAt(r, QtWidgets.QFormLayout.FieldRole)
        if label and field:
            rows[label.widget().text()] = field.widget().text()
    return rows


def pump(seconds):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.02)


def select(name, method, view="raw"):
    for row in range(tab.listing.count()):
        if os.path.basename(tab.listing.item(row).data(0x0100)) == name:
            tab.listing.setCurrentRow(row)
            break
    else:
        fail.append(f"{name} not in the listing")
    tab.method_combo.setCurrentIndex(tab.method_combo.findData(method))
    tab.view_combo.setCurrentIndex(tab.view_combo.findData(view))
    pump(0.2)


def shot(label):
    path = os.path.join(SHOTS, f"{label}.png")
    tab.grab().save(path)
    return path


def stabilize(name, method):
    select(name, method)
    before = tab.stab_status.text()
    tab.stab_btn.click()
    started = time.time()
    pump(0.5)
    if tab.proc is None:
        fail.append(f"{name}/{method}: no analysis process started")
        return
    if tab.stab_btn.text() != "Cancel":
        fail.append(f"{name}/{method}: button reads {tab.stab_btn.text()!r} while running")
    while tab.proc is not None and time.time() - started < TIMEOUT_S:
        pump(0.2)
    log = tab.stab_log.toPlainText().splitlines()
    print(f"\n{name} / {method}: {time.time() - started:.0f}s, status before: {before!r}")
    print("  log tail: " + " | ".join(log[-4:]))
    print(f"  status: {tab.stab_status.text()!r}")
    if tab.proc is not None:
        fail.append(f"{name}/{method}: still running after {TIMEOUT_S}s")
        tab.shutdown()
        return
    if "finished" not in log[-1]:
        fail.append(f"{name}/{method}: process ended with {log[-1]!r}")
    if not (tab.result and tab.result.ok):
        fail.append(f"{name}/{method}: no usable result loaded")
        return
    if "Stability index" not in tab.stab_status.text():
        fail.append(f"{name}/{method}: status shows no metrics")
    if "older code" in tab.stab_status.text():
        fail.append(f"{name}/{method}: a fresh result reported as out of date")


def views(name, method):
    select(name, method, "stabilized")
    res = tab.result
    n = len(tab.source)
    used = [i for i in range(n) if res.used(i)]
    unused = [i for i in range(n) if not res.used(i)]
    for i in (used[:1] + used[len(used) // 2:len(used) // 2 + 1] + unused[:1]):
        tab.slider.setValue(i)
        pump(0.1)
        rows = form_rows(tab.frame_form)
        if tab.view.pixmap() is None or tab.view.pixmap().isNull():
            fail.append(f"{name}/{method}: frame {i} rendered nothing")
        state = rows.get("Stabilized")
        expect_used = i in used
        if (state == "used") != expect_used:
            fail.append(f"{name}/{method}: frame {i} labelled {state!r}, used={expect_used}")
        print(f"  frame {i}: {state}; pixel range {rows.get('Pixel range')}")
    shot(f"{name}_{method}_stabilized")
    # playback must advance through stabilized frames
    tab.slider.setValue(0)
    tab._toggle_play()
    pump(1.0)
    tab._stop()
    if tab.index == 0:
        fail.append(f"{name}/{method}: stabilized playback did not advance")

    tab.view_combo.setCurrentIndex(tab.view_combo.findData("mean"))
    pump(0.2)
    rows = form_rows(tab.frame_form)
    print(f"  mean view: {rows}")
    rgb, lo, hi, missing = res.mean_image()
    magenta = int((rgb == cr.NO_DATA_RGB).all(axis=2).sum())
    if tab.view.pixmap() is None or tab.view.pixmap().isNull():
        fail.append(f"{name}/{method}: mean rendered nothing")
    if missing > 0 and magenta == 0:
        fail.append(f"{name}/{method}: no-data regions not drawn magenta")
    tab._toggle_play()
    if tab.timer.isActive():
        fail.append(f"{name}/{method}: playback started on the still mean view")
        tab._stop()
    shot(f"{name}_{method}_mean")
    tab.view_combo.setCurrentIndex(tab.view_combo.findData("raw"))
    pump(0.1)
    shot(f"{name}_{method}_raw")


def cancel(name):
    select(name, "nonrigid")
    tab.stab_btn.click()
    pump(4.0)
    if tab.proc is None:
        fail.append("cancel: the run finished before it could be cancelled")
        return
    tab.stab_btn.click()
    pump(0.5)
    base = cr.stabilization_base(tab.source.path)
    leftovers = glob.glob(os.path.join(base, "*", name, "_work"))
    print(f"\ncancel: proc={tab.proc}, leftover work dirs {leftovers}, "
          f"status {tab.stab_status.text()!r}")
    if tab.proc is not None:
        fail.append("cancel: process still attached")
    if leftovers:
        fail.append(f"cancel: working maps left behind {leftovers}")
    if tab.stab_btn.text() == "Cancel":
        fail.append("cancel: button still reads Cancel")


app = QtWidgets.QApplication(sys.argv)
os.makedirs(SHOTS, exist_ok=True)
tmp = tempfile.mkdtemp(prefix="review_stabilize_")
try:
    recordings = os.path.join(tmp, "recordings")
    for b in BURSTS:
        shutil.copytree(b, os.path.join(recordings, os.path.basename(b)))
    tab = cr.ReviewTab(StubMain(recordings))
    tab.resize(1500, 900)
    tab.show()
    tab.refresh()
    pump(0.3)
    if not tab.analysis:
        fail.append("analysis package not found by the Review tab")
    names = [os.path.basename(b) for b in BURSTS]

    select(names[0], "translation")
    if "Not stabilized" not in tab.stab_status.text():
        fail.append(f"fresh burst status reads {tab.stab_status.text()!r}")
    select(names[0], "translation", "stabilized")
    if "No stabilization result" not in tab.view.text():
        fail.append("stabilized view without a result doesn't say so")

    cancel(names[0])
    for name in names:
        for method in ("translation", "nonrigid"):
            stabilize(name, method)
            views(name, method)

    summary = os.path.join(cr.stabilization_base(os.path.join(recordings, names[0])),
                           "translation", "summary.csv")
    with open(summary, encoding="utf-8") as f:
        rows = f.read().splitlines()[1:]
    print(f"\ntranslation summary rows: {len(rows)} (bursts {len(names)})")
    if len(rows) != len(names):
        fail.append(f"summary has {len(rows)} rows for {len(names)} bursts")
    tab.shutdown()
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print(f"\nscreenshots in {SHOTS}")
print("RESULT:", "FAIL " + "; ".join(fail) if fail else "PASS")
sys.exit(1 if fail else 0)
