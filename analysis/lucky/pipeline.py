"""Per-burst lucky stage (METHODS.md §L1–§L4):

    saved transforms -> aligned frames -> plain mean (pass 1)
                     -> lucky fusion, full burst and split halves (pass 2)
                     -> annotation of the mean and of each fused image
                     -> comparison, with the stabilization's own consensus-
                        mask skeleton as the reference it replaces
"""
import csv
import json
import os
import platform
import time
from datetime import datetime, timezone

import cv2
import numpy as np
import tifffile
from skimage.morphology import skeletonize

from . import __version__
from .align import AlignedBurst
from .annotate import annotate, calibrate, contrast_image, graph_stats, segment_table
from .config import params_hash
from .evaluate import background_noise, centreline_agreement, edge_sharpness
from .fuse import LuckyAccumulator


def fuse_burst(burst_path, stab_root, lp, method="auto", halves=True, log=None):
    """Fuse one burst at every power in lp.powers.

    With halves=True, alternate used frames are also accumulated separately
    (and summed for the full result) for split-half reproducibility (§L4).
    Returns {'images': {label: image}, 'halves': [{label: image}] * 2, ...}.
    """
    ab = AlignedBurst(burst_path, stab_root, method)
    t0 = time.time()
    # common brightness: illumination and auto-exposure change the level
    # frame to frame
    gains = {}
    ref_level = None

    def gain(i, img, seen):
        nonlocal ref_level
        if i not in gains:
            level = float(np.median(img[seen]))
            ref_level = ref_level or level
            gains[i] = ref_level / max(level, 1e-6)
        return gains[i]

    template = None
    if any(p > 0 for p in lp.powers):
        # pass 1: the plain mean, which pass 2 measures each frame against
        acc = np.zeros(ab.shape, np.float64)
        cnt = np.zeros(ab.shape, np.float64)
        for i, img, seen in ab.frames():
            if seen.any():
                acc += np.where(seen, img * gain(i, img, seen), 0)
                cnt += seen
        ok = cnt >= max(1.0, lp.min_coverage * len(gains))
        template = np.where(ok, acc / np.maximum(cnt, 1), np.nan).astype(np.float32)
    parts = [LuckyAccumulator(ab.shape, lp, template) for _ in range(2 if halves else 1)]
    k = 0
    for i, img, seen in ab.frames():
        if seen.any():
            parts[k % len(parts)].add(img, seen, gain(i, img, seen))
            k += 1
    full = parts[0] if len(parts) == 1 else parts[0].merge(parts[1])
    images, n_eff, valid = full.result(lp.min_coverage)
    out = {"aligned": ab, "images": images, "n_eff": n_eff, "valid": valid,
           "frames": full.frames}
    if halves:
        res = [p.result(lp.min_coverage) for p in parts]
        out["halves"] = [r[0] for r in res]
        out["half_valid"] = [r[2] for r in res]
    if log:
        log(f"  fused {full.frames} frames ({ab.method} alignment) at "
            f"p = {', '.join(f'{p:g}' for p in lp.powers)} ({time.time() - t0:.0f}s)")
    return out


def process(burst_path, stab_root, out_root, lp, ap, method="auto", log=print):
    """Fuse, annotate and compare one burst into <out_root>/<burst>/."""
    started = time.time()
    res = fuse_burst(burst_path, stab_root, lp, method, halves=True, log=log)
    ab = res["aligned"]
    out_dir = os.path.join(out_root, ab.burst.name)
    os.makedirs(out_dir, exist_ok=True)
    valid = res["valid"]
    # score away from the field border, where the largest filter runs off it
    border = int(2 * max(ap.sigmas_px))
    region = cv2.erode(valid.astype(np.uint8), np.ones((2 * border + 1,) * 2, np.uint8)) > 0

    # ONE Frangi constant for the burst, from the plain mean, used for every
    # image: differences between annotations then come from the images
    c = calibrate(contrast_image(res["images"]["p0"], valid, ap), valid & region, ap)
    both = region & res["half_valid"][0] & res["half_valid"][1]
    rows, annotations = {}, {}
    for label, img in res["images"].items():
        a = annotate(img, ap, c, valid)
        halves = [annotate(h[label], ap, c, hv)
                  for h, hv in zip(res["halves"], res["half_valid"])]
        split = centreline_agreement(halves[0]["skeleton"], halves[1]["skeleton"], 2.0, both)
        rows[label] = {
            **a["stats"],
            "split_half_f1": split["f1"],
            "edge_sharpness": edge_sharpness(a["contrast"], a["mask"] & region,
                                             a["skeleton"], region),
            "background_noise": background_noise(a["contrast"], a["mask"], region),
            "n_eff_per_band": res["n_eff"][label],
        }
        annotations[label] = a
        log(f"  {label:<3s} split-half F1 {split['f1']:.3f}, centreline "
            f"{a['stats']['centreline_px_per_mpx']:.0f} px/Mpx in "
            f"{a['stats']['components_per_mpx']:.0f} pieces/Mpx, edge sharpness "
            f"{rows[label]['edge_sharpness']:.3f}, noise {rows[label]['background_noise']:.4f}")

    label = f"p{lp.output:g}"
    lucky, a = res["images"][label], annotations[label]

    # the reference being replaced: the stabilization's consensus mask,
    # skeletonized (stabilize §6, §9)
    baseline, cons_skel = None, None
    cons_path = os.path.join(ab.result_dir, "consensus_mask.tif")
    if os.path.exists(cons_path):
        cons = (tifffile.imread(cons_path) > 0) & valid
        cons_skel = skeletonize(cons)
        baseline = {**graph_stats(cons_skel, cons, valid),
                    # how much of the old skeleton the new annotation also has
                    "found_by_lucky_annotation": centreline_agreement(
                        cons_skel, a["skeleton"], 3.0, region)["precision"]}

    tifffile.imwrite(os.path.join(out_dir, "lucky.tif"), lucky)
    tifffile.imwrite(os.path.join(out_dir, "mean.tif"), res["images"]["p0"])
    tifffile.imwrite(os.path.join(out_dir, "centreline.tif"), a["skeleton"].astype(np.uint8) * 255)
    tifffile.imwrite(os.path.join(out_dir, "vessel_mask.tif"), a["mask"].astype(np.uint8) * 255)
    tifffile.imwrite(os.path.join(out_dir, "vessel_width.tif"), a["width"])
    tifffile.imwrite(os.path.join(out_dir, "vesselness.tif"), a["vesselness"])
    segments = segment_table(a["skeleton"], a["width"])
    with open(os.path.join(out_dir, "segments.csv"), "w", newline="", encoding="utf-8") as f:
        if segments:
            wr = csv.DictWriter(f, fieldnames=list(segments[0].keys()))
            wr.writeheader()
            wr.writerows(segments)
    write_compare_png(os.path.join(out_dir, "compare.png"), res["images"]["p0"], lucky,
                      cons_skel, a["skeleton"], valid)

    record = {
        "status": "ok",
        "burst": ab.burst.name,
        "alignment": ab.method,
        "alignment_result": ab.result_dir,
        "frames_fused": res["frames"],
        "output_image": label,
        "segments": len(segments),
        "by_image": rows,
        "consensus_skeleton": baseline,
        "annotation_contrast_constant": c,
        "processing": {
            "lucky_version": __version__,
            "params_hash": params_hash(lp, ap),
            "lucky_params": lp.to_dict(),
            "annotate_params": ap.to_dict(),
            "seconds": round(time.time() - started, 1),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "tifffile": tifffile.__version__,
        },
    }
    with open(os.path.join(out_dir, "lucky.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    log(f"  done: {len(segments)} segments ({time.time() - started:.0f}s)")
    return record


def _u8(img, lo, hi):
    return np.clip((np.nan_to_num(img, nan=lo) - lo) / max(hi - lo, 1e-9) * 255,
                   0, 255).astype(np.uint8)


def write_compare_png(path, mean, lucky, skel_consensus, skel_lucky, valid):
    """Top: plain mean | lucky image, on one contrast scale. Bottom: the
    stabilization's consensus-mask skeleton (red) on the mean | the lucky
    annotation (green) on the lucky image. Magenta = no data (stabilize §11)."""
    lo, hi = np.nanpercentile(mean[valid], [0.5, 99.5])
    panels = []
    for img, skel, colour in ((mean, skel_consensus, (0, 0, 255)),
                              (lucky, skel_lucky, (0, 255, 0))):
        rgb = cv2.cvtColor(_u8(img, lo, hi), cv2.COLOR_GRAY2BGR)
        rgb[~valid] = (255, 0, 255)
        ov = rgb.copy()
        if skel is not None:
            ov[cv2.dilate(skel.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0] = colour
        panels.append((rgb, ov))
    sep = np.full((mean.shape[0], 8, 3), 255, np.uint8)
    top = np.hstack([panels[0][0], sep, panels[1][0]])
    bottom = np.hstack([panels[0][1], sep, panels[1][1]])
    cv2.imwrite(path, np.vstack([top, np.full((8, top.shape[1], 3), 255, np.uint8), bottom]))
