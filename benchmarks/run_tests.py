"""Run the whole verification suite against the connected camera.

usage: python run_tests.py [work_dir]

Each test exits non-zero on failure; the summary at the end is what matters.
"""
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "test_run")
PY = sys.executable

TESTS = (
    ("burst pixels bit-exact (Mono8/12/12p)", ["test_burst_writer.py"]),
    ("frame-rate restamp (mp4 + mkv)", ["test_restamp.py"]),
    ("camera settings, ROI, video, Mono8", ["test_regression.py",
                                            os.path.join(WORK, "regress")]),
    ("codec presets: bit depth and drops", ["test_codecs_camera.py",
                                            os.path.join(WORK, "codecs"), "8"]),
    ("Mono12 burst at 40 fps + metadata", ["test_burst_gui.py",
                                           os.path.join(WORK, "burst"),
                                           "200", "40", "958"]),
    ("400 fps strip burst", ["test_burst_gui.py",
                             os.path.join(WORK, "strip"),
                             "2000", "400", "60", "2000"]),
    ("ROI placement + Review tab", ["test_review_tab.py",
                                    os.path.join(WORK, "review")]),
    ("Record button bursts (hold/tap/close)", ["test_record_burst.py",
                                               os.path.join(WORK, "record")]),
)

if __name__ == "__main__":
    os.makedirs(WORK, exist_ok=True)
    results = []
    for name, args in TESTS:
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}", flush=True)
        t0 = time.time()
        rc = subprocess.run([PY, *args], cwd=HERE).returncode
        results.append((name, rc, time.time() - t0))
    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    for name, rc, secs in results:
        print(f"  {'PASS' if rc == 0 else 'FAIL'}  {name}  ({secs:.0f}s)")
    shutil.rmtree(WORK, ignore_errors=True)
    sys.exit(0 if all(rc == 0 for _, rc, _ in results) else 1)
