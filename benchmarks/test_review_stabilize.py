"""Review tab stabilization, end to end, without a camera.

Copies the given bursts into a temporary recordings folder, opens the Review
tab on its own, and for each burst and method: presses Stabilize, waits for
the analysis process, then plays the Raw / Stabilized / Stabilized mean
views, checking the Stabilized view shows only the frames the result used
(stepping, playback and the slider skip the rest) while Raw shows them all.
Also cancels a run part-way and checks the summary covers every burst. Screenshots of each view are kept for a visual check.

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
        return False
    if tab.stab_btn.text() != "Cancel":
        fail.append(f"{name}/{method}: button reads {tab.stab_btn.text()!r} while running")
    while tab.proc is not None and time.time() - started < TIMEOUT_S:
        pump(0.2)
    log = tab.job_log.toPlainText().splitlines()
    print(f"\n{name} / {method}: {time.time() - started:.0f}s, status before: {before!r}")
    print("  log tail: " + " | ".join(log[-4:]))
    print(f"  status: {tab.stab_status.text()!r}")
    if tab.proc is not None:
        fail.append(f"{name}/{method}: still running after {TIMEOUT_S}s")
        tab.shutdown()
        return False
    if "finished" not in log[-1]:
        fail.append(f"{name}/{method}: process ended with {log[-1]!r}")
    if tab.result is not None and tab.result.metrics.get("status") == "skipped":
        # the analysis may decline a burst (too few frames agree): that is a
        # result too, as long as the status line says why
        if "Skipped" not in tab.stab_status.text():
            fail.append(f"{name}/{method}: skipped without saying why: {tab.stab_status.text()!r}")
        return False
    if not (tab.result and tab.result.ok):
        fail.append(f"{name}/{method}: no usable result loaded")
        return False
    if "Stability index" not in tab.stab_status.text():
        fail.append(f"{name}/{method}: status shows no metrics")
    if "older code" in tab.stab_status.text():
        fail.append(f"{name}/{method}: a fresh result reported as out of date")
    return True


def views(name, method):
    select(name, method, "stabilized")
    res = tab.result
    n = len(tab.source)
    used = [i for i in range(n) if res.used(i)]
    unused = [i for i in range(n) if not res.used(i)]
    print(f"  used {len(used)} of {n} frames; left out {unused}")
    # the Stabilized view shows only the frames the stabilization used: the
    # slider put on one it left out moves on to the next one it used, the
    # way it moved
    for i in (used[:1] + used[len(used) // 2:len(used) // 2 + 1] + unused[:1]):
        was = tab.index
        tab.slider.setValue(i)
        pump(0.1)
        rows = form_rows(tab.frame_form)
        if tab.view.pixmap() is None or tab.view.pixmap().isNull():
            fail.append(f"{name}/{method}: frame {i} rendered nothing")
        state = rows.get("Stabilized") or ""
        if not state.startswith("used") or not res.used(tab.index):
            fail.append(f"{name}/{method}: slider on {i} shows frame {tab.index}, labelled {state!r}")
        if i in used and tab.index != i:
            fail.append(f"{name}/{method}: slider on used frame {i} moved to {tab.index}")
        if i not in used:
            ahead = [u for u in used if (u > i if i > was else u < i)]
            expect = (min(ahead) if i > was else max(ahead)) if ahead else (used[-1] if i > was else used[0])
            if tab.index != expect:
                fail.append(f"{name}/{method}: slider {was} -> left-out {i} went to {tab.index}, not {expect}")
        print(f"  slider {i} -> frame {tab.index}: {state}; pixel range {rows.get('Pixel range')}")
    shot(f"{name}_{method}_stabilized")
    # stepping visits every used frame and only those, both ways
    tab.slider.setValue(used[0])
    pump(0.05)
    seen = [tab.index]
    for _ in range(n + 2):
        tab._step(+1)
        seen.append(tab.index)
    back = [tab.index]
    for _ in range(n + 2):
        tab._step(-1)
        back.append(tab.index)
    fwd = sorted(set(seen))
    if fwd != used or sorted(set(back)) != used or seen[-1] != used[-1] or back[-1] != used[0]:
        fail.append(f"{name}/{method}: stepping visited {fwd} / {sorted(set(back))}, used {used}")
    # the slider moved one frame at a time (arrow keys) never stalls on a
    # frame left out, and never shows one
    tab.slider.setValue(used[0])
    keys = [tab.index]
    for _ in range(n + 2):
        tab.slider.setValue(min(n - 1, tab.slider.value() + 1))
        keys.append(tab.index)
    if sorted(set(keys)) != used:
        fail.append(f"{name}/{method}: arrow keys visited {sorted(set(keys))}, used {used}")
    # playback advances through the used frames only, and loops
    tab.slider.setValue(used[0])
    played = []
    for _ in range(2 * len(used)):
        tab._advance()
        played.append(tab.index)
    if any(not res.used(i) for i in played) or sorted(set(played)) != used:
        fail.append(f"{name}/{method}: playback showed {sorted(set(played))}, used {used}")
    tab.slider.setValue(used[0])
    tab._toggle_play()
    pump(1.0)
    tab._stop()
    if tab.index == used[0]:
        fail.append(f"{name}/{method}: stabilized playback did not advance")
    # the Raw view still shows every frame
    tab.view_combo.setCurrentIndex(tab.view_combo.findData("raw"))
    tab.slider.setValue(0)
    pump(0.05)
    raw_seen = [tab.index]
    for _ in range(n + 2):
        tab._step(+1)
        raw_seen.append(tab.index)
    if sorted(set(raw_seen)) != list(range(n)):
        fail.append(f"{name}/{method}: Raw stepping visited {sorted(set(raw_seen))} of {n}")
    if unused:
        tab.slider.setValue(unused[0])
        pump(0.05)
        if tab.index != unused[0]:
            fail.append(f"{name}/{method}: Raw view moved off left-out frame {unused[0]}")
    tab.view_combo.setCurrentIndex(tab.view_combo.findData("stabilized"))
    pump(0.1)
    if unused and not res.used(tab.index):
        fail.append(f"{name}/{method}: switching to Stabilized kept left-out frame {tab.index}")

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
            if stabilize(name, method):
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
