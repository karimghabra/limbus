"""Live stabilization: camera frames stabilized as they arrive (METHODS.md §15).

The offline methods register a whole burst at once, each frame against a
template made from all of them. Live, only the frames seen so far exist. So
each frame is registered as it arrives against a template made from the first
frames (calibration) and kept current, with the translation method's own
steps and parameters (§5-§8): vesselness and envelope masks, coarse
correlation of the masks with the template, then the half-whitened sub-pixel
refinement started from the last position; the quality gate compares each
frame with the calibration frames.

It runs on the GPU (§14), in a process of its own, so the camera app never
loads PyTorch and a slow or failed stabilizer can't stall frame grabbing:

    LiveClient   in the app: starts the worker, passes it frames through
                 shared memory without ever waiting, and hands back each
                 frame's offset.
    serve()      the worker: python -m stabilize.live ...

Frames the worker receives together are registered together, so when the
camera outruns it the batches grow and it keeps up at a little more latency.
"""
import argparse
import collections
import os
import secrets
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from multiprocessing.connection import Client, Listener
from multiprocessing.shared_memory import SharedMemory

import numpy as np

CALIBRATION_FRAMES = 24      # frames that set the constants and the first template
TEMPLATE_MEMORY = 64         # recent frames in the template fade at this rate (frames)
RELOCK_SECS = 1.0            # clear frames not registering for this long: start again
REFERENCES_KEPT = 4          # earlier locks a re-lock may rejoin
MAX_BATCH = 32
SLOTS = 32                   # frames in flight between the app and the worker
LOG = os.path.join(tempfile.gettempdir(), "limbus_live_stabilization.log")


class RealtimeStabilizer:
    """Causal translation stabilization of a stream of frames on the GPU."""

    def __init__(self, gpu, shape, full_scale, params=None):
        from .config import Params
        from .vessels import working_sigmas
        self.gpu, self.torch = gpu, gpu.torch
        self.p = p = params or Params()
        H, W = shape
        self.shape, self.fs = tuple(shape), full_scale
        self.scale = p.working_scale if H >= p.min_height_to_downscale else 1.0
        self.hw = (int(np.rint(H * self.scale)), int(np.rint(W * self.scale)))
        self.sigmas = working_sigmas(p, self.scale, self.hw[0])
        self.pc = gpu.phase_correlator(self.hw)
        self.refiner = gpu.refiner(self.hw)
        self.max_step = p.fine_max_step_px * self.scale
        self.decay = 1.0 - 1.0 / TEMPLATE_MEMORY
        # this template's place in the stream's reference frame (working px):
        # 0 for the first lock; after an automatic re-lock, where the new
        # template sits in the old one, so the view doesn't jump
        self.origin = np.zeros(2)

    def calibrate(self, frames):
        """Constants (vesselness contrast, envelope threshold, the gate's
        reference statistics) and the first template, from the first frames:
        chained, then registered once against the template the chain makes."""
        from .vessels import contrast_constant
        g, p = self.gpu, self.p
        w = g.to_working(g.tensor(np.stack(frames)).float(), self.scale)
        self.gate_ref = g.frame_stats(w, self.fs, p)[:2]
        self.c = contrast_constant(w[len(frames) // 2].cpu().numpy(), self.sigmas, p)
        v, glare = self._vessels(w)
        self.threshold = float(np.percentile(v.cpu().numpy(), p.envelope_percentile))
        m = g.envelope(v, self.threshold).float()
        o = (~glare).float()
        dx, dy, _ = self.pc(v[:-1], v[1:])
        d = np.vstack([[0.0, 0.0], np.cumsum(np.stack([dx, dy], 1), 0)])
        d -= np.median(d, 0)
        self._set_template(v, m, o, d)
        d, conf = self.refiner(self.refiner.spectra(v), np.arange(len(frames)), d, self.max_step)
        keep = conf >= p.fine_min_response
        if keep.sum() < max(3, len(frames) // 4):
            raise RuntimeError("the first frames don't show enough vessel structure to lock on")
        d -= np.median(d[keep], 0)
        k = self.gpu.tensor(np.flatnonzero(keep))
        self._set_template(v.index_select(0, k), m.index_select(0, k), o.index_select(0, k), d[keep])
        self.prev = d[keep][-1]
        # The template is these calibration frames, kept for good, plus the
        # recently registered frames, which fade. Recent frames let it follow
        # the eye into tissue the calibration never saw; the permanent part
        # anchors it: a template made only of recent frames re-learns each
        # frame's small registration error and drifts (0.22 -> 0.41 px from
        # offline over 800 frames of 15-22-26; anchored, 0.22 throughout).
        self.base = (self.vsum, self.msum, self.cover)
        self.vsum, self.msum, self.cover = (t * 0 for t in self.base)
        self._refresh()

    def _vessels(self, w):
        v = self.gpu.vesselness(w, self.sigmas, self.c, self.p)
        glare = self.gpu.glare(w, self.fs, self.p, self.scale)
        return self.torch.where(glare, 0.0, v), glare

    def _set_template(self, v, m, o, d):
        g = self.gpu
        self.vsum = g.shift(v, d).sum(0)
        self.msum = g.shift(m, d).sum(0)
        self.cover = g.shift(o, d).sum(0)
        self._refresh()

    def _refresh(self):
        torch = self.torch
        vsum, msum, cover = self.vsum, self.msum, self.cover
        if hasattr(self, "base"):
            vsum, msum, cover = vsum + self.base[0], msum + self.base[1], cover + self.base[2]
        ok = cover > 0
        denom = torch.where(ok, cover, 1.0)
        self.p_template = torch.where(ok, msum / denom, 0.0)
        self.v_template = torch.where(ok, vsum / denom, 0.0)
        self.p_spec = self.pc.spectrum(self.p_template[None])
        self.refiner.set_template(self.v_template)

    def _register(self, v, m, prior):
        """Offsets and confidences of maps v, m (B, h, w) against the template:
        refined from `prior`, and, as offline, also from the coarse estimate
        where that is confident and lands elsewhere (after a saccade); the
        fine stage's confidence decides."""
        p = self.p
        dx, dy, rc = self.pc(self.p_spec, m)
        d_coarse = np.stack([dx, dy], 1)
        spectra = self.refiner.spectra(v)
        rows = np.arange(len(v))
        best, conf = self.refiner(spectra, rows, prior, self.max_step)
        alt = (rc >= p.coarse_min_response) & (np.hypot(*(d_coarse - prior).T) > self.max_step)
        if alt.any():
            d2, c2 = self.refiner(spectra, rows[alt], d_coarse[alt], self.max_step)
            better = c2 > conf[alt]
            sel = rows[alt][better]
            best[sel], conf[sel] = d2[better], c2[better]
        return best, conf

    def locate(self, other):
        """Where this stabilizer's template sits in `other`'s (working px), and
        the refinement's confidence: to carry the reference frame across an
        automatic re-lock."""
        d, conf = other._register(self.v_template[None], self.p_template[None], np.zeros((1, 2)))
        return d[0], float(conf[0])

    def clear(self, frames):
        """The quality gate against the calibration frames: which of these
        frames are free of blinks and lighting jumps."""
        g = self.gpu
        w = g.to_working(g.tensor(np.asarray(frames)).float(), self.scale)
        return self._clear(*g.frame_stats(w, self.fs, self.p)[:2])

    def _clear(self, ls, mn):
        from .quality import robust_z
        p = self.p
        ref_ls, ref_mn = self.gate_ref
        z_s = np.array([robust_z(np.r_[ref_ls, x])[-1] for x in ls])
        z_m = np.array([robust_z(np.r_[ref_mn, x])[-1] for x in mn])
        blurred = (z_s < p.sharpness_min_z) & (np.exp(ls - np.median(ref_ls)) < p.sharpness_min_ratio)
        lighting = ((np.abs(z_m) > p.brightness_max_abs_z)
                    & (np.abs(mn / np.median(ref_mn) - 1) > p.brightness_max_rel_change))
        return ~blurred & ~lighting

    def process(self, frames):
        """Register frames (B, H, W) that arrived since the last call. Returns
        offsets d (B, 2) in full-resolution px (the frame's content has moved
        by d: show it shifted by -d), registered (B,), confidence (B,), and
        clear (B,): passed the quality gate (no blink, no lighting jump)."""
        g, torch, p = self.gpu, self.torch, self.p
        B = len(frames)
        w = g.to_working(g.tensor(np.asarray(frames)).float(), self.scale)
        ls, mn, _ = g.frame_stats(w, self.fs, p)
        clear = self._clear(ls, mn)
        v, glare = self._vessels(w)
        m = g.envelope(v, self.threshold).float()
        best, conf = self._register(v, m, np.repeat(self.prev[None], B, 0))
        ok = (conf >= p.fine_min_response) & clear
        if ok.any():
            k = np.flatnonzero(ok)
            self.prev = best[k[-1]]
            # the template's recent part: this batch's registered frames in,
            # older ones fading (the calibration frames stay)
            kt = g.tensor(k)
            n = len(k)
            both = torch.cat([v.index_select(0, kt), m.index_select(0, kt),
                              (~glare).float().index_select(0, kt)])
            moved = g.shift(both, np.tile(best[k], (3, 1)))
            fade = self.decay ** n
            self.vsum = self.vsum * fade + moved[:n].sum(0)
            self.msum = self.msum * fade + moved[n:2 * n].sum(0)
            self.cover = self.cover * fade + moved[2 * n:].sum(0)
            self._refresh()
        return (best + self.origin) / self.scale, ok, conf, clear


# ---- the worker ---------------------------------------------------------------------

def warm_up(gpu, shape, dtype, full_scale):
    """Run every GPU step once on made-up frames of the stream's size, so the
    first real frames don't wait while CUDA compiles kernels and plans FFTs
    (over a second, the first time)."""
    rng = np.random.default_rng(0)
    H, W = shape
    top = 255 if dtype == np.uint8 else 4095 << 4
    base = rng.random((H + 16, W + 16))
    base = np.cumsum(np.cumsum(base - 0.5, 0), 1)           # smooth, structured
    base = (base - base.min()) / (np.ptp(base) + 1e-9) * 0.8 * top
    frames = [base[k:k + H, k:k + W].astype(dtype) for k in range(8)]
    stab = RealtimeStabilizer(gpu, shape, full_scale)
    try:
        stab.calibrate(frames)
        stab.process(np.stack(frames[:2]))
    except RuntimeError:
        pass
    gpu.torch.cuda.synchronize()


def serve(args):
    """Receive frames, stabilize them, send back offsets, until told to stop
    or the app goes away. Messages in: ("frame", index, slot), ("relock",),
    ("stop",). Out: ("state", kind, text) with kind locking / locked / lost /
    error, ("free", slots) as soon as frames are copied out of shared memory,
    and ("results", [(index, dx, dy, registered, confidence), ...], info)."""
    host, port = args.address.rsplit(":", 1)
    conn = Client((host, int(port)), authkey=bytes.fromhex(args.authkey))
    shm = None
    try:
        from .gpu import select
        gpu = select(args.device)
        if gpu is None:
            conn.send(("state", "error", "Live stabilization needs an NVIDIA GPU and PyTorch "
                       "with CUDA (see requirements-analysis.txt); none was found."))
            return
        H, W = args.shape
        dtype = np.dtype(args.dtype)
        shm = SharedMemory(name=args.shm)
        if os.name != "nt":
            # the app created the memory and removes it; don't let this
            # process's resource tracker remove it too
            from multiprocessing import resource_tracker
            resource_tracker.unregister(shm._name, "shared_memory")
        ring = np.ndarray((args.slots, H, W), dtype, buffer=shm.buf)
        full_scale = 255 if dtype == np.uint8 else 65535     # 16-bit frames are MSB-aligned
        warm_up(gpu, (H, W), dtype, full_scale)
        # the templates locked on before, newest last: after losing the eye,
        # the new lock is placed in the reference frame of whichever of them
        # it matches, so coming back to an area doesn't make the view jump
        stab, calibration = None, []
        earlier = collections.deque(maxlen=REFERENCES_KEPT)
        conn.send(("state", "locking", gpu.describe()))
        failing_since = None
        while True:
            msgs = [conn.recv()]
            while len(msgs) < MAX_BATCH and conn.poll():
                msgs.append(conn.recv())
            if any(m[0] == "stop" for m in msgs):
                break
            if any(m[0] == "relock" for m in msgs):
                # asked for: a fresh reference frame, centred on the view now
                stab, calibration = None, []
                earlier.clear()
                conn.send(("state", "locking", "locking on again"))
            frames = [m for m in msgs if m[0] == "frame"]
            if not frames:
                continue
            ids = [m[1] for m in frames]
            slots = [m[2] for m in frames]
            batch = ring[slots]                              # a copy
            conn.send(("free", slots))
            t0 = time.perf_counter()
            rows = [(i, 0.0, 0.0, False, 0.0) for i in ids]
            if stab is None:
                # after losing the eye, calibrate only on frames the last
                # reference calls clear: not on the blink that lost it
                keep = earlier[-1].clear(batch) if earlier else np.ones(len(batch), bool)
                calibration.extend(batch[keep])
                if len(calibration) >= CALIBRATION_FRAMES:
                    stab = RealtimeStabilizer(gpu, (H, W), full_scale)
                    try:
                        stab.calibrate(calibration[-CALIBRATION_FRAMES:])
                        note = ""
                        if earlier:
                            note = "the view has moved: a new reference"
                            for ref in reversed(earlier):
                                d, conf = stab.locate(ref)
                                if conf >= stab.p.fine_min_response:
                                    stab.origin = ref.origin + d
                                    note = "back on the same reference"
                                    break
                        conn.send(("state", "locked", note))
                        failing_since = None
                    except RuntimeError as exc:
                        stab = None
                        conn.send(("state", "locking", str(exc)))
                    calibration = []
            else:
                d, ok, conf, clear = stab.process(batch)
                rows = [(i, float(a), float(b), bool(c), float(e))
                        for i, (a, b), c, e in zip(ids, d, ok, conf)]
                # lost: clear frames that stop registering, for a while.
                # Blinks and lighting jumps don't count: the eye comes back
                now = time.monotonic()
                if ok.any():
                    failing_since = None
                elif clear.any():
                    failing_since = failing_since or now
                    if now - failing_since > RELOCK_SECS:
                        earlier.append(stab)
                        stab, calibration = None, []
                        conn.send(("state", "lost", f"clear frames stopped registering for "
                                                     f"{RELOCK_SECS:.0f} s: locking on again"))
            conn.send(("results", rows, {"batch": len(ids),
                                         "ms": 1000 * (time.perf_counter() - t0)}))
    except (EOFError, ConnectionError, BrokenPipeError):
        pass                                                 # the app went away
    except Exception:
        try:
            conn.send(("state", "error", traceback.format_exc()))
        except OSError:
            pass
        raise
    finally:
        if shm is not None:
            shm.close()
        conn.close()


# ---- the app's side ---------------------------------------------------------------------

class LiveClient:
    """Runs the worker and feeds it. offer() is called from the camera's grab
    thread and never waits: when every slot is in flight the frame is only
    counted as skipped. Callbacks come from a thread of this client's own:
    on_results(rows, info) and on_state(kind, text), with kind locking,
    locked, lost, error, stopped, or reshape (frames no longer match the
    size or type this client was made for: make a new one)."""

    def __init__(self, shape, dtype, on_results=None, on_state=None, slots=SLOTS,
                 device=None, analysis_dir=None, python=None):
        self.shape = tuple(int(v) for v in shape[:2])
        self.dtype = np.dtype(dtype)
        self.on_results = on_results or (lambda rows, info: None)
        self.on_state = on_state or (lambda kind, text: None)
        self.slots = slots
        self.device = device
        self.analysis_dir = analysis_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.python = python or sys.executable
        H, W = self.shape
        self.shm = SharedMemory(create=True, size=slots * H * W * self.dtype.itemsize)
        self.ring = np.ndarray((slots, H, W), self.dtype, buffer=self.shm.buf)
        self.free = collections.deque(range(slots))
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()
        self.key = secrets.token_bytes(16)
        self.listener = Listener(("127.0.0.1", 0), authkey=self.key)
        self.conn = None
        self.proc = None
        self.connected = False
        self.accepted = False
        self.ready = False           # the worker has warmed up and wants frames
        self.closed = False
        self.offered = self.skipped = 0
        self._reshaped = False

    def start(self):
        host, port = self.listener.address
        cmd = [self.python, "-u", "-m", "stabilize.live", "--address", f"{host}:{port}",
               "--authkey", self.key.hex(), "--shm", self.shm.name, "--slots", str(self.slots),
               "--shape", str(self.shape[0]), str(self.shape[1]), "--dtype", self.dtype.name]
        if self.device:
            cmd += ["--device", self.device]
        env = dict(os.environ, PYTHONPATH=self.analysis_dir, PYTHONIOENCODING="utf-8")
        self._log = open(LOG, "w", encoding="utf-8")
        self.proc = subprocess.Popen(cmd, cwd=self.analysis_dir, env=env, stdout=self._log,
                                     stderr=subprocess.STDOUT,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        threading.Thread(target=self._run, daemon=True).start()
        threading.Thread(target=self._watch, daemon=True).start()
        return self

    def _watch(self):
        """If the worker dies before it connects, stop waiting for it."""
        self.proc.wait()
        if not self.accepted and not self.closed:
            self.on_state("error", f"the stabilizer stopped before it started: see {LOG}")
            self.listener.close()

    def _run(self):
        try:
            self.conn = self.listener.accept()
            self.accepted = self.connected = True
            while True:
                msg = self.conn.recv()
                if msg[0] == "free":
                    with self._lock:
                        self.free.extend(msg[1])
                elif msg[0] == "results":
                    self.on_results(msg[1], msg[2])
                elif msg[0] == "state":
                    if msg[1] == "locking":
                        self.ready = True
                    self.on_state(msg[1], msg[2])
        except (EOFError, OSError):
            pass
        finally:
            self.connected = False
            if not self.closed:
                self.on_state("stopped", f"the stabilizer stopped: see {LOG}")

    def offer(self, frame, index):
        """Hand a frame to the worker if a slot is free. Colour frames are
        registered in grey. Returns whether it was sent. Never raises: it
        runs in the camera's grab thread, where an exception stops the camera."""
        try:
            return self._offer(frame, index)
        except Exception:
            return False

    def _offer(self, frame, index):
        if not self.ready or not self.connected or self.closed:
            return False
        if frame.ndim == 3:
            import cv2
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frame.shape != self.shape or frame.dtype != self.dtype:
            if not self._reshaped:
                self._reshaped = True
                self.on_state("reshape", f"frames are now {frame.shape[1]}x{frame.shape[0]} "
                                         f"{frame.dtype}")
            return False
        with self._lock:
            if not self.free:
                self.skipped += 1
                return False
            slot = self.free.popleft()
        self.ring[slot] = frame
        try:
            with self._send_lock:
                self.conn.send(("frame", int(index), slot))
        except OSError:
            return False
        self.offered += 1
        return True

    def relock(self):
        self._send(("relock",))

    def _send(self, msg):
        if self.connected:
            try:
                with self._send_lock:
                    self.conn.send(msg)
            except OSError:
                pass

    def stop(self):
        """Stop the worker and release the shared memory."""
        if self.closed:
            return
        self.closed = True
        self._send(("stop",))
        if self.proc is not None:
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(5)
        for close in (getattr(self.conn, "close", None), self.listener.close):
            try:
                if close:
                    close()
            except OSError:
                pass
        if hasattr(self, "_log"):
            self._log.close()
        del self.ring
        self.shm.close()
        try:
            self.shm.unlink()
        except FileNotFoundError:
            pass


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m stabilize.live", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--address", required=True)
    ap.add_argument("--authkey", required=True)
    ap.add_argument("--shm", required=True)
    ap.add_argument("--slots", type=int, required=True)
    ap.add_argument("--shape", type=int, nargs=2, required=True)
    ap.add_argument("--dtype", required=True)
    ap.add_argument("--device", default=None)
    serve(ap.parse_args(argv))


if __name__ == "__main__":
    main()
