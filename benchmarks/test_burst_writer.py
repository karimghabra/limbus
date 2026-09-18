"""TiffBurstWriter must save exactly the sensor values for every pixel format.

Needs no camera. Feeds frames in the form the pylon converter delivers them
(Mono8 as plain uint8; Mono12/Mono12p as uint16 with the 12 bits MSB-aligned)
and checks the saved TIFFs hold the true sensor DN, bit for bit, and that the
manifest describes them correctly.

This exists because Mono8 bursts were once written entirely black: the writer
shifted every frame right by 16 - bit_depth, which zeroes 8-bit data.

usage: python test_burst_writer.py
"""
import glob
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import tifffile

from bench_common import REPO

sys.path.insert(0, os.path.abspath(REPO))
import camera_recorder as cr  # noqa: E402

rng = np.random.default_rng(7)
fail = []
tmp = tempfile.mkdtemp(prefix="burstwriter_")

CASES = (
    # label, bit_depth, dtype as delivered, true DN range, MSB-aligned?
    ("Mono8", 8, np.uint8, 255, False),
    ("Mono12", 12, np.uint16, 4095, True),
    ("Mono12p", 12, np.uint16, 4095, True),
)

for label, bits, dtype, top, msb in CASES:
    # true sensor values, including both extremes so clipping or loss shows
    truth = rng.integers(0, top + 1, size=(6, 40, 64)).astype(dtype)
    truth[:, 0, 0], truth[:, 0, 1] = 0, top
    delivered = (truth << (16 - bits)) if msb else truth

    folder = os.path.join(tmp, label)
    writer = cr.TiffBurstWriter(folder, {"frame_bytes": delivered[0].nbytes},
                                bit_depth=bits)
    for frame in delivered:
        writer.write(np.ascontiguousarray(frame), {"exposure_us": 1000.0})
    written, dropped = writer.close()

    saved = np.stack([tifffile.imread(p) for p in
                      sorted(glob.glob(os.path.join(folder, "*.tif")))])
    with open(os.path.join(folder, "manifest.json"), encoding="utf-8") as f:
        pv = json.load(f)["pixel_values"]

    exact = saved.dtype == truth.dtype and np.array_equal(saved, truth)
    print(f"{label:8s} delivered {delivered.dtype} max={int(delivered.max())} -> "
          f"saved {saved.dtype} range {int(saved.min())}-{int(saved.max())} "
          f"exact={exact} | manifest dtype={pv['dtype']} "
          f"max_value={pv['max_value']} | written={written} dropped={dropped}")
    if not exact:
        fail.append(f"{label}: saved pixels differ from the sensor values")
    if int(saved.max()) == 0:
        fail.append(f"{label}: saved frames are entirely zero")
    if pv["dtype"] != np.dtype(dtype).name:
        fail.append(f"{label}: manifest says {pv['dtype']}, "
                    f"files are {np.dtype(dtype).name}")
    if pv["max_value"] != top:
        fail.append(f"{label}: manifest max_value {pv['max_value']}, expected {top}")

shutil.rmtree(tmp, ignore_errors=True)
print("\nRESULT:", "FAIL " + "; ".join(fail) if fail else "PASS")
sys.exit(1 if fail else 0)
