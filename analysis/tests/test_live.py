"""Live stabilization against known motion (stabilize/live.py, METHODS.md §15).

A synthetic stream with known eye motion goes through the app's side of
live stabilization (LiveClient), which runs the worker process on the GPU,
at a camera's pace (60 fps), frame by frame:

  area A    jitter: locks on, and the offsets follow the true motion
  blink     bright, featureless frames: not registered, and not a loss
  area A    carries on in the same reference frame
  area B    another area: the eye is lost, and it locks on B afresh
  area A    back again: it rejoins A's reference frame, so the offsets
            match the true motion with the same constant as at the start

Skips (exit code 0) without a CUDA GPU, after checking the worker says so.

usage: python analysis/tests/test_live.py      (from the repository root)
"""
import os
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
sys.path.insert(0, HERE)
from stabilize.gpu import select  # noqa: E402
from stabilize.live import LiveClient  # noqa: E402
from test_nonrigid_smoke import render, scene  # noqa: E402

W, H, FPS = 1000, 480, 60.0
fail = []


def check(ok, what):
    print(("  ok    " if ok else "  FAIL  ") + what, flush=True)
    if not ok:
        fail.append(what)


def stream():
    """Frames (MSB-aligned 16-bit, as the camera sends them), each frame's
    true content motion (NaN where there is none to find), and which part
    of the story each frame belongs to."""
    rng = np.random.default_rng(4)
    a = scene(np.random.default_rng(1000), W, H)
    b = scene(np.random.default_rng(2000), W, H)
    walk = np.cumsum(rng.normal(0, 0.8, (600, 2)), 0)
    parts = [("A", 180), ("blink", 20), ("A2", 120), ("B", 150), ("A3", 150)]
    frames, truth, label = [], [], []
    t = 0
    for name, n in parts:
        if name == "blink":
            gy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
            for _ in range(n):
                f = 3000 + 200 * gy + rng.normal(0, 30, (H, W)).astype(np.float32)
                frames.append(np.clip(np.rint(f), 0, 4095).astype(np.uint16))
                truth.append((np.nan, np.nan))
                label.append(name)
            continue
        sc = b if name == "B" else a
        poses = [(walk[t + k, 0], walk[t + k, 1], 0.0) for k in range(n)]
        if name == "A3":            # back to A, somewhere else in its field
            poses = [(x + 12.0, y - 8.0, r) for x, y, r in poses]
        frames += render(rng, sc, W, H, poses)
        truth += [(x, y) if name != "B" else (np.nan, np.nan) for x, y, _ in poses]
        label += [name] * n
        t += n
    return (np.stack(frames).astype(np.uint16) << 4), np.array(truth), np.array(label)


def main():
    if select() is None:
        states = []
        c = LiveClient((H, W), np.uint16, on_state=lambda k, t: states.append((k, t)),
                       device="cpu").start()
        end = time.time() + 30
        while time.time() < end and not any(k == "error" for k, _ in states):
            time.sleep(0.1)
        c.stop()
        check(any(k == "error" and "GPU" in t for k, t in states), "without a GPU, the worker says so")
        print("\nSKIPPED: no CUDA GPU" if not fail else "\nRESULT: FAIL")
        sys.exit(1 if fail else 0)

    print("rendering the stream ...", flush=True)
    frames, truth, label = stream()
    results, states = {}, []
    t_start = [None]

    def on_results(rows, info):
        for i, dx, dy, ok, conf in rows:
            results[i] = (dx, dy, ok)

    def on_state(kind, text):
        states.append((kind, text, max(results) if results else -1))

    client = LiveClient((H, W), np.uint16, on_results, on_state).start()
    t0 = time.time()
    while not client.ready and time.time() - t0 < 60:
        time.sleep(0.01)
    check(client.ready, f"the worker starts ({time.time() - t0:.1f} s)")
    t_start[0] = time.perf_counter()
    for i, f in enumerate(frames):
        target = t_start[0] + i / FPS
        while time.perf_counter() < target:
            time.sleep(0.0005)
        client.offer(f, i)
    end = time.time() + 5
    while time.time() < end and len(results) < len(frames):
        time.sleep(0.05)
    client.stop()
    print("states:", [(k, t, i) for k, t, i in states])

    check(client.skipped == 0 and len(results) == len(frames),
          f"every frame stabilized at {FPS:.0f} fps: {len(results)}/{len(frames)}, "
          f"{client.skipped} skipped")
    idx = np.arange(len(frames))
    got = np.array([results.get(i, (np.nan, np.nan, False)) for i in idx], object)
    d = np.array([[g[0], g[1]] for g in got], float)
    ok = np.array([bool(g[2]) for g in got])

    def err(part, const):
        sel = (label == part) & ok
        e = d[sel] - truth[sel] - const
        return np.sqrt(np.mean(np.sum(e ** 2, 1))), int(sel.sum()), int((label == part).sum())

    first = (label == "A") & ok
    const = np.median(d[first] - truth[first], 0)
    rms, n_ok, n = err("A", const)
    check(n_ok >= 0.8 * n and rms < 0.2, f"area A: {n_ok}/{n} registered, RMS error {rms:.3f} px")
    blink = label == "blink"
    in_blink = [k for k, t, i in states if k == "lost" and blink[max(i, 0)]]
    check(not ok[blink].any() and not in_blink,
          f"blink: {int(ok[blink].sum())} of {int(blink.sum())} blink frames registered, "
          f"{len(in_blink)} losses of lock")
    rms, n_ok, n = err("A2", const)
    check(n_ok >= 0.8 * n and rms < 0.2, f"area A after the blink: {n_ok}/{n} registered, "
                                         f"RMS error {rms:.3f} px (same reference)")
    kinds = [k for k, _, _ in states]
    lost_b = [i for k, _, i in states if k == "lost" and label[max(i, 0)] == "B"]
    locked_b = [t for k, t, i in states if k == "locked" and label[max(i, 0)] == "B"]
    check(bool(lost_b) and bool(locked_b) and "new reference" in locked_b[0],
          f"area B: the eye is lost, and it locks on B afresh ({locked_b[:1]})")
    locked_a3 = [(t, i) for k, t, i in states if k == "locked" and label[max(i, 0)] == "A3"]
    rms, n_ok, n = err("A3", const)
    # it takes RELOCK_SECS to call the eye lost and a calibration's worth of
    # frames to lock on again; after that, A3 should register as A did
    after = (label == "A3") & (idx > locked_a3[0][1]) if locked_a3 else np.zeros(len(idx), bool)
    check(bool(locked_a3) and "same reference" in locked_a3[0][0]
          and ok[after].mean() >= 0.8 and rms < 0.5,
          f"back in area A: rejoins its reference ({locked_a3[0][0] if locked_a3 else None!r}); "
          f"{int(ok[after].sum())}/{int(after.sum())} registered after the re-lock, "
          f"RMS error {rms:.3f} px against the original constant")
    check(kinds.count("error") == 0, "no errors")
    print("\nRESULT:", "FAIL " + "; ".join(fail) if fail else "PASS")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
