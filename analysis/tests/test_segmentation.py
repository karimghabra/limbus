"""The swappable segmentation behind the Review tab's vessel overlay.

Checks the contract a segmenter plugs into (segmentation/README.md) without
the app: that segmenters are found without being imported, that one added in
a LIMBUS_SEGMENTERS folder can stand in for a built-in one, that the runner
writes a valid overlay and replaces the previous one only on success, that
overlays made on an earlier stabilization or by an earlier version of the
segmenter are flagged, and that the quick built-in segmenter finds the
vessels of a synthetic burst.

usage: python analysis/tests/test_segmentation.py      (from the repository root)
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
ROOT = os.path.dirname(ANALYSIS)
sys.path.insert(0, ANALYSIS)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
sys.path.insert(0, HERE)
import segmentation  # noqa: E402
from segmentation import overlay  # noqa: E402
from stabilize.methods import run as stabilize  # noqa: E402
from test_nonrigid_smoke import render, scene, write_burst  # noqa: E402

fail = []


def check(ok, what):
    print(("  ok    " if ok else "  FAIL  ") + what, flush=True)
    if not ok:
        fail.append(what)


PLUGIN = '''"""A test segmenter: straight lines where the burst's vessels are not."""
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
        ov.add_vessel([[10, y], [w - 10, y]], radius=[2 + k, 2 + k], group=1, note="line")
    ov.add_junction(w / 2, h / 2, "test kind", [1, 2])
    mask = np.zeros((h, w), bool)
    mask[: h // 2, : w // 2] = True
    ov.add_mask("top-left", mask)
    with open(os.path.join(inputs.out_dir, "extra.txt"), "w") as f:
        f.write("a segmenter's own file")
    return ov
'''


def run_segmenter(sid, result, env=None):
    e = dict(os.environ, PYTHONIOENCODING="utf-8", **(env or {}))
    e["PYTHONPATH"] = ANALYSIS
    p = subprocess.run([sys.executable, "-m", "segmentation", "run", sid, result],
                       cwd=ANALYSIS, env=e, capture_output=True, text=True, encoding="utf-8")
    return p.returncode, p.stdout + p.stderr


def main():
    tmp = tempfile.mkdtemp(prefix="segmentation_test_")
    try:
        plugins = os.path.join(tmp, "my_segmenters")
        os.makedirs(plugins)
        with open(os.path.join(plugins, "test_lines.py"), "w", encoding="utf-8") as f:
            f.write(PLUGIN.format(version="1"))
        # a file that would break the listing if it were imported
        with open(os.path.join(plugins, "import_bomb.py"), "w", encoding="utf-8") as f:
            f.write('raise SystemExit("imported while listing")\nLABEL = "Bomb"\n'
                    'def segment(inputs):\n    pass\n')
        with open(os.path.join(plugins, "_helper.py"), "w", encoding="utf-8") as f:
            f.write("def segment(inputs):\n    pass\n")
        os.environ[segmentation.ENV_VAR] = plugins

        print("discovery:")
        found = {s.id: s for s in segmentation.list_segmenters()}
        check({"stabilization_mask", "vesselmap", "test_lines", "import_bomb"} <= set(found)
              and "_helper" not in found,
              f"found {sorted(found)} without importing any of them")
        check(found["test_lines"].label == "Test lines" and found["test_lines"].version == "1",
              "LABEL and VERSION read from the file's source")
        # a folder in LIMBUS_SEGMENTERS stands in for a built-in of the same name
        shutil.copy(os.path.join(plugins, "test_lines.py"),
                    os.path.join(plugins, "stabilization_mask.py"))
        check(segmentation.find("stabilization_mask").path.startswith(plugins),
              "a segmenter in LIMBUS_SEGMENTERS replaces the built-in one of its name")
        os.remove(os.path.join(plugins, "stabilization_mask.py"))

        print("a stabilized synthetic burst:")
        rng = np.random.default_rng(3)
        folder = os.path.join(tmp, "recordings", "burst_seg")
        write_burst(folder, render(rng, scene(np.random.default_rng(1000), 1000, 720), 1000, 720,
                                   [(rng.normal(0, 1.5), rng.normal(0, 1.5), 0) for _ in range(30)]))
        out = os.path.join(tmp, "stabilization")
        rec = stabilize("translation", folder, out, log=lambda s: None)
        result = os.path.join(out, "translation", "burst_seg")
        check(rec["status"] == "ok", f"stabilized ({rec['processing']['backend']})")

        print("the quick built-in segmenter:")
        t0 = time.time()
        code, log = run_segmenter("stabilization_mask", result)
        doc = overlay.read(segmentation.result_dir(result, "stabilization_mask")) if code == 0 else None
        check(code == 0 and doc is not None and len(doc["vessels"]) >= 5 and doc["masks"],
              f"ran in {time.time() - t0:.1f}s: {doc['summary'] if doc else log[-300:]}")
        if doc:
            # its centrelines lie on the synthetic vessels (dark), not on
            # tissue: compared with as many points placed at random
            import tifffile
            mean = tifffile.imread(os.path.join(result, "mean_stabilized.tif"))
            h, w = mean.shape
            pts = np.vstack([v["points"] for v in doc["vessels"]])
            rnd = np.random.default_rng(0).uniform((0, 0), (w - 1, h - 1), pts.shape)

            def median_at(p):
                return float(np.nanmedian(mean[np.rint(p[:, 1]).astype(int),
                                               np.rint(p[:, 0]).astype(int)]))
            on, anywhere = median_at(np.clip(pts, 0, (w - 1, h - 1))), median_at(rnd)
            check(on < 0.93 * anywhere, f"its centrelines lie on vessels: median intensity "
                                        f"{on:.0f} DN on them, {anywhere:.0f} at random points")

        print("a segmenter from LIMBUS_SEGMENTERS, run by the runner:")
        code, log = run_segmenter("test_lines", result)
        folder_tl = segmentation.result_dir(result, "test_lines")
        doc = overlay.read(folder_tl) if code == 0 else None
        check(code == 0 and doc and len(doc["vessels"]) == 3 and doc["junctions"][0]["kind"] == "test kind"
              and os.path.exists(os.path.join(folder_tl, "extra.txt"))
              and os.path.exists(doc["masks"][0][1]),
              "overlay.json, its mask and the segmenter's own file are written")
        check(doc and doc["segmenter"]["sha256"] == found["test_lines"].source_hash()
              and doc["source"]["method"] == "translation" and doc["shape"] == (720, 1000),
              "the overlay records its segmenter, source and frame size")
        check(doc and overlay.staleness(doc, result, segmentation.find("test_lines")) == [],
              "a fresh overlay is current")

        print("replacing only on success:")
        code, log = run_segmenter("test_lines", result, {"TEST_SEGMENTER_FAIL": "1"})
        check(code == 1 and "asked to fail" in log and os.path.exists(os.path.join(folder_tl, "overlay.json")),
              "a failed run exits 1, says why, and keeps the previous overlay")
        before = overlay.read(folder_tl)["made_utc"]
        p = subprocess.Popen([sys.executable, "-m", "segmentation", "run", "test_lines", result],
                             cwd=ANALYSIS, env=dict(os.environ, PYTHONPATH=ANALYSIS,
                                                    TEST_SEGMENTER_SLEEP="30"),
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)
        p.kill()
        p.wait()
        check(overlay.read(folder_tl)["made_utc"] == before, "a killed run keeps the previous overlay")

        print("stale overlays:")
        with open(found["test_lines"].path, "w", encoding="utf-8") as f:
            f.write(PLUGIN.format(version="2"))
        doc = overlay.read(folder_tl)
        check(overlay.staleness(doc, result, segmentation.find("test_lines"))
              == ["made by an earlier version of this segmenter"],
              "an edited segmenter flags its old overlays")
        mean_path = os.path.join(result, "mean_stabilized.tif")
        os.utime(mean_path, ns=(time.time_ns(), time.time_ns() + 10**9))
        check("found on an earlier stabilization" in overlay.staleness(doc, result),
              "re-stabilizing flags the overlays made on the old mean")

        print("the overlay file:")
        ov = overlay.Overlay((50, 60))
        for bad, what in ((lambda: ov.add_vessel([[1, 2]]), "one point"),
                          (lambda: ov.add_vessel([[1, 2], [3, np.nan]]), "NaN"),
                          (lambda: ov.add_vessel([[1, 2], [3, 4]], radius=[1]), "radius length"),
                          (lambda: ov.add_mask("m", np.zeros((5, 5))), "mask size")):
            try:
                bad()
                check(False, f"rejects a vessel/mask with {what}")
            except ValueError:
                check(True, f"rejects a vessel/mask with {what}")
        vid = ov.add_vessel(np.array([[1.234567, 2], [30, 40]]), radius=np.array([1.5, 2.5]), depth=3)
        ov.add_junction(5, 6, "crossing")
        ov.write(os.path.join(tmp, "ov"), {"segmenter": {}, "source": {}})
        back = overlay.read(os.path.join(tmp, "ov"))
        v = back["vessels"][0]
        check(v["id"] == vid and np.allclose(v["points"], [[1.23, 2], [30, 40]])
              and v["props"] == {"depth": 3} and back["summary"] == "1 vessels · crossing 1",
              "round trip: points to 0.01 px, radius, props, summary")
        with open(os.path.join(tmp, "ov", "overlay.json"), encoding="utf-8") as f:
            d = json.load(f)
        d["version"] = 99
        with open(os.path.join(tmp, "ov", "overlay.json"), "w", encoding="utf-8") as f:
            json.dump(d, f)
        try:
            overlay.read(os.path.join(tmp, "ov"))
            check(False, "refuses an overlay version it doesn't know")
        except ValueError as exc:
            check("version 99" in str(exc), "refuses an overlay version it doesn't know")
    finally:
        os.environ.pop(segmentation.ENV_VAR, None)
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nRESULT:", "FAIL " + "; ".join(fail) if fail else "PASS")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
