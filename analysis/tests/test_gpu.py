"""The GPU backend against the CPU path it stands in for (METHODS.md §14).

Three levels, on a synthetic burst with rotation, jitter and glare:

  primitives   each GPU primitive against the OpenCV / NumPy function it
               replaces, on the same input
  stages       registration, metrics and the non-rigid refinement given the
               SAME vessel maps: registration must reproduce the CPU
               trajectory exactly, the metrics to rounding
  end to end   the whole pipeline on each backend: GPU vesselness differs from
               OpenCV's by float32 rounding, so a frame whose sub-pixel peak
               is a near-tie may land one refinement step away (0.05
               working-scale px); everything else must agree

Skips (exit code 0) when there is no CUDA GPU.

usage: python analysis/tests/test_gpu.py      (from the repository root)
"""
import csv
import json
import os
import shutil
import sys
import tempfile
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
sys.path.insert(0, HERE)
from stabilize import gpu_stages as S  # noqa: E402
from stabilize.bursts import load_burst, read_frame  # noqa: E402
from stabilize.config import NonrigidParams, Params  # noqa: E402
from stabilize.fields import FieldEvaluator, Grid  # noqa: E402
from stabilize.gpu import Maps, select  # noqa: E402
from stabilize.metrics import (closure_error, mask_consensus, residual_motion,  # noqa: E402
                               rotation_diagnostic)
from stabilize.methods import run  # noqa: E402
from stabilize.nonrigid import _refine  # noqa: E402
from stabilize.pipeline import process_burst  # noqa: E402
from stabilize.quality import frame_stats  # noqa: E402
from stabilize.register import (SubpixelRefiner, build_templates, phase_correlate,  # noqa: E402
                                register_groupwise, shift)
from stabilize.vessels import (contrast_constant, envelope, glare_mask, to_working,  # noqa: E402
                               vesselness, working_sigmas)
from test_nonrigid_smoke import render, scene, write_burst  # noqa: E402

fail = []


def check(ok, what):
    print(("  ok    " if ok else "  FAIL  ") + what, flush=True)
    if not ok:
        fail.append(what)


def make_burst(tmp):
    """40 frames, 1000x720: jitter, a 0.5 deg change of pose, and a clipped
    glare spot fixed to the camera."""
    rng = np.random.default_rng(11)
    poses = [(rng.normal(0, 2.0), rng.normal(0, 2.0), 0.0 if t < 25 else 0.5) for t in range(40)]
    frames = render(np.random.default_rng(5), scene(np.random.default_rng(1000), 1000, 720),
                    1000, 720, poses)
    for f in frames:
        cv2.circle(f, (700, 200), 20, 4095, -1)
    folder = os.path.join(tmp, "recordings", "burst_gpu_check")
    write_burst(folder, frames)
    return folder


def primitives(gpu, folder):
    print("primitives (GPU vs the CPU function each replaces):")
    p = Params()
    files = load_burst(folder).files
    raw = np.stack([read_frame(files[i]) for i in (0, 1, 20, 30)])
    fs = 4095
    x = gpu.tensor(raw).float()
    w_cpu = np.stack([to_working(im, 0.5) for im in raw])
    w_gpu = gpu.to_working(x, 0.5)
    check(np.array_equal(w_gpu.cpu().numpy(), w_cpu), "to_working: identical")
    st_c = np.array([frame_stats(im, fs, p) for im in w_cpu]).T
    st_g = np.array(gpu.frame_stats(w_gpu, fs, p))
    check(np.abs(st_c - st_g).max() < 1e-3, f"frame_stats: max |diff| {np.abs(st_c - st_g).max():.1e}")
    sig = working_sigmas(p, 0.5, w_cpu.shape[1])
    c = contrast_constant(w_cpu[0], sig, p)
    v_cpu = np.stack([vesselness(im, sig, c, p) for im in w_cpu])
    v_gpu = gpu.vesselness(w_gpu, sig, c, p)
    dv = np.abs(v_gpu.cpu().numpy() - v_cpu)
    # Second derivatives of heavily blurred images lose most of their float32
    # digits to cancellation, differently in each filter implementation; at
    # saddle points (|l1| ~ |l2|) that can even swap the eigenvalue order. So
    # vesselness agrees to rounding almost everywhere, with a few isolated
    # pixels further off; what matters is that the vessel masks built from
    # it agree (checked with the envelope below).
    check(np.percentile(dv, 99.9) < 1e-3 and (dv > 1e-3).mean() < 1e-3,
          f"vesselness (values 0-1): 99.9th pct |diff| {np.percentile(dv, 99.9):.1e}, "
          f"{100 * (dv > 1e-3).mean():.3f}% of pixels > 1e-3")
    g_cpu = np.stack([glare_mask(im, fs, p, 0.5) for im in w_cpu])
    check(g_cpu.any() and np.array_equal(gpu.glare(w_gpu, fs, p, 0.5).cpu().numpy(), g_cpu),
          f"glare_mask: identical ({int(g_cpu.sum())} glare px)")
    thr = float(np.percentile(v_cpu, 90))
    m_cpu = np.stack([envelope(v, thr) for v in v_cpu])
    check(np.array_equal(gpu.envelope(gpu.tensor(v_cpu), thr).cpu().numpy(), m_cpu),
          "envelope: identical from the same vesselness")
    m_gpu = gpu.envelope(v_gpu, thr).cpu().numpy()
    check((m_gpu != m_cpu).mean() < 1e-4,
          f"envelope from each backend's own vesselness: {100 * (m_gpu != m_cpu).mean():.4f}% "
          "of pixels differ")
    worst = 0.0
    for sub in ((slice(None), slice(None)), (slice(0, 83), slice(0, 240)),
                (slice(0, 125), slice(0, 250)), (slice(40, 200), slice(100, 260))):
        A = np.ascontiguousarray(v_cpu[[0, 0, 1]][(slice(None),) + sub])
        B = np.ascontiguousarray(v_cpu[[1, 2, 3]][(slice(None),) + sub])
        win = cv2.createHanningWindow((A.shape[2], A.shape[1]), cv2.CV_32F)
        ref = []
        for a, b in zip(A, B):
            (dx, dy), r = phase_correlate(a, b, win)
            ref.append((dx, dy, r))
        got = np.stack(gpu.phase_correlator(A.shape[1:])(gpu.tensor(A), gpu.tensor(B)), 1)
        worst = max(worst, float(np.abs(got - np.array(ref)).max()))
    check(worst < 1e-3, f"phaseCorrelate (incl. padded and odd sizes): max |diff| {worst:.1e}")
    rc = SubpixelRefiner(v_cpu.shape[1:])
    rc.set_template(v_cpu[0])
    rg = gpu.refiner(v_cpu.shape[1:])
    rg.set_template(gpu.tensor(v_cpu[0]))
    coarse = np.array([[0.4, 0.3], [2.0, -1.0], [-3.0, 2.5], [0.0, 0.0]])
    dg, cg = rg(rg.spectra(gpu.tensor(v_cpu)), np.arange(4), coarse, 3.0)
    res = [rc(v_cpu[k], coarse[k], 3.0) for k in range(4)]
    dc = np.array([r[0] for r in res])
    cc = np.array([r[1] for r in res])
    check(np.abs(dg - dc).max() < 1e-9 and np.abs(cg - cc).max() < 1e-9,
          f"SubpixelRefiner: max |diff| {np.abs(dg - dc).max():.1e} px, confidence "
          f"{np.abs(cg - cc).max():.1e}")
    d = np.array([[0.3137, -0.71], [4.74, -18.28], [-21.5, 16.0], [0.0, 0.0]])
    sc = np.stack([shift(v_cpu[k], d[k]) for k in range(4)])
    check(np.abs(sc - gpu.shift(gpu.tensor(v_cpu), d).cpu().numpy()).max() < 1e-4,
          f"shift: max |diff| {np.abs(sc - gpu.shift(gpu.tensor(v_cpu), d).cpu().numpy()).max():.1e}")
    h, w = v_cpu.shape[1:]
    grid = Grid(79.5, 79.5, 80, 3, 5)
    ev, evg = FieldEvaluator((h, w), grid), gpu.fields((h, w), grid)
    rng = np.random.default_rng(0)
    A = np.zeros((4, 2, 3))
    A[:, :, :2] = rng.normal(0, 3e-3, (4, 2, 2))
    A[:, :, 2] = rng.normal(0, 5, (4, 2))
    L = rng.normal(0, 2, (4, 3, 5, 2)).astype(np.float32)
    fx, fy = evg.dense(A, L)
    dfield = max(float(np.abs(np.stack(ev.dense(A[k], L[k])) -
                              np.stack([fx[k].cpu().numpy(), fy[k].cpu().numpy()])).max()) for k in range(4))
    wc = np.stack([ev.warp(v_cpu[k], *ev.dense(A[k], L[k])) for k in range(4)])
    dwarp = float(np.abs(wc - evg.warp(gpu.tensor(v_cpu), fx, fy).cpu().numpy()).max())
    check(dfield < 1e-4 and dwarp < 1e-3, f"fields: dense max |diff| {dfield:.1e} px, remap {dwarp:.1e}")


def stages(gpu, folder, tmp):
    print("stages, given the same vessel maps:")
    p, npar = Params(), NonrigidParams()
    quiet = lambda s: None  # noqa: E731
    out = os.path.join(tmp, "same_maps")
    rec = process_burst(folder, os.path.join(out, "translation"), p, log=quiet, keep_work=True)
    wd = rec.pop("_work_dir")
    try:
        V, M, O = (np.load(os.path.join(wd, f"{k}.npy"), mmap_mode="r") for k in "VMO")
        name = os.path.basename(folder)
        with open(os.path.join(out, "translation", name, "transforms.csv"), encoding="utf-8") as f:
            good = np.array([r["passed_gate"] == "1" for r in csv.DictReader(f)])
        scale = rec["processing"]["working_scale"]
        maps = Maps.load(gpu, V, M, O)
        rc = register_groupwise(V, M, good, p, scale, O=O)
        rg = S.register_groupwise(gpu, maps, good, p, scale)
        dt = np.abs(rc["traj"] - rg["traj"]).max()
        check(dt < 1e-6 and np.array_equal(rc["registered"], rg["registered"])
              and rc["iterations"] == rg["iterations"],
              f"register_groupwise: trajectory max |diff| {dt:.1e} px, same flags and iterations")
        traj = rc["traj"]
        used = np.flatnonzero(rc["registered"] & good)
        align = S.ShiftWarp(gpu, traj)
        for aligned in (False, True):
            a, ca = mask_consensus(M, traj, used, p, aligned=aligned, O=O)
            b, cb = S.mask_consensus(gpu, maps, used, p, aligned=aligned, warp=align)
            check(abs(a["overlap"] - b["overlap"]) < 1e-9 and abs(a["dice"] - b["dice"]) < 1e-6
                  and np.array_equal(ca, cb),
                  f"mask_consensus (aligned={aligned}): overlap {a['overlap']:.6f} / {b['overlap']:.6f}")
        _, tv_c, _ = build_templates(V, M, traj, used, O)
        _, tv_g, _ = S.build_templates(gpu, maps, align, used)
        pairs = [(residual_motion(V, traj, used, scale, p),
                  S.residual_motion(gpu, maps, used, scale, p, align)),
                 (closure_error(V, used, scale, p), S.closure_error(gpu, maps, used, scale, p)),
                 (rotation_diagnostic(V, traj, tv_c, used, scale, p),
                  S.rotation_diagnostic(gpu, maps, tv_g, used, scale, p, align))]
        dm = max(abs(a[k] - b[k]) for a, b in pairs for k in a)
        check(dm < 1e-3, f"residual, closure and rotation diagnostics: max |diff| {dm:.1e} px")
        burst = load_burst(folder)
        tr_dir = os.path.join(out, "translation", name)
        fields = {}
        for label, g in (("cpu", None), ("gpu", gpu)):
            r = _refine(burst, rec, tr_dir, wd, os.path.join(out, "nonrigid_" + label, name), p, npar,
                        quiet, time.time(), g)
            fields[label] = (r, np.load(os.path.join(out, "nonrigid_" + label, name, "fields.npz")))
        (r1, z1), (r2, z2) = fields["cpu"], fields["gpu"]
        dl = np.abs(z1["local"] - z2["local"])
        check(np.array_equal(z1["used"], z2["used"]) and np.median(dl) < 1e-3
              and abs(r1["stability_index"] - r2["stability_index"]) < 1e-3,
              f"non-rigid: same frames used, local field median |diff| {np.median(dl):.1e} px "
              f"(max {dl.max():.1e}), affine max |diff| {np.abs(z1['affine'] - z2['affine']).max():.1e}, "
              f"index {r1['stability_index']:.4f} / {r2['stability_index']:.4f}")
    finally:
        shutil.rmtree(wd, ignore_errors=True)


def end_to_end(folder, tmp):
    print("end to end, each backend from the raw frames:")
    name = os.path.basename(folder)
    trajs = {}
    for device in ("cpu", "cuda"):
        out = os.path.join(tmp, "e2e_" + device)
        run("nonrigid", folder, out, log=lambda s: None, device=device)
        with open(os.path.join(out, "translation", name, "transforms.csv"), encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        trajs[device] = (np.array([[float(r["dx_px"]), float(r["dy_px"])] for r in rows]),
                         np.array([r["registered"] == "1" for r in rows]))
    (tc, rc), (tg, rg) = trajs["cpu"], trajs["cuda"]
    both = rc & rg
    d = tc[both] - tg[both]
    rel = np.hypot(*(d - np.median(d, axis=0)).T)
    step = 0.05 / Params().working_scale          # one refinement step, full-res px
    check((rc == rg).mean() >= 0.95 and np.mean(rel < 1e-3) >= 0.9 and rel.max() <= 2 * step + 1e-3,
          f"translation: registered flags agree on {100 * (rc == rg).mean():.0f}% of frames; "
          f"{100 * np.mean(rel < 1e-3):.0f}% of trajectories identical, max |diff| {rel.max():.3f} px "
          f"(refinement step {step:.2f} px)")
    for key in ("translation", "nonrigid"):
        a, b = (json.load(open(os.path.join(tmp, "e2e_" + device, key, name, "metrics.json"),
                               encoding="utf-8")) for device in ("cpu", "cuda"))
        check(a["status"] == b["status"] == "ok"
              and abs(a["stability_index"] - b["stability_index"]) < 0.005,
              f"{key}: stability index {a['stability_index']:.4f} (cpu) / {b['stability_index']:.4f} "
              f"(gpu), {a['processing']['seconds']:.1f} s / {b['processing']['seconds']:.1f} s")


def main():
    gpu = select("auto")
    if gpu is None:
        print("SKIPPED: no CUDA GPU (PyTorch with CUDA finds none)")
        sys.exit(0)
    print(gpu.describe())
    tmp = tempfile.mkdtemp(prefix="stabilize_gpu_")
    try:
        folder = make_burst(tmp)
        primitives(gpu, folder)
        stages(gpu, folder, tmp)
        end_to_end(folder, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nRESULT:", "FAIL " + "; ".join(fail) if fail else "PASS")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
