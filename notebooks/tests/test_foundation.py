"""Behavioural tests of the foundation notebook (notebooks/foundation.py), run end to end on synthetic bursts.

Each case writes a small burst the way the recorder does (one 12-bit TIFF per frame and a manifest.json), runs the
whole notebook on it as a script and checks what it prints and what it saves to preprocessed/<burst>.npz:

- glint: a clipped glint is in view for part of the burst. The frame gate must keep those frames, and still drop
  the blurred ones. The saved file must hold the kept frames, the fixations and each frame's shift from its
  reference frame, and those shifts must match the motion the burst was made with.
- short fixations: every fixation is shorter than the notebook needs to judge a pixel's fluctuation over time. The
  notebook must say so and finish, rather than fail after stabilizing.

The notebook is exported with `marimo export script` and run with its animated previews switched off.

usage: python notebooks/tests/test_foundation.py   (from the repository root, in the notebooks' environment:
notebooks/requirements.txt). It also runs under pytest.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
NOTEBOOK = ROOT / "notebooks" / "foundation.py"
sys.path.insert(0, str(ROOT / "benchmarks"))
from bench_common import _vessels  # noqa: E402  the vessel-network generator of the recording benchmarks

SIZE = 320          # pixels on a side of a frame
FPS = 30.0
SATURATION = 4095   # the camera's largest 12-bit value


def make_scene(rng, size):
    """A tissue picture in DN: uneven light, fine texture and a vessel network, larger than a frame."""
    illumination = cv2.resize(cv2.GaussianBlur(rng.random((size // 16, size // 16)).astype(np.float32), (0, 0), 2),
                              (size, size), interpolation=cv2.INTER_CUBIC)
    illumination = 0.8 + 0.4 * (illumination - illumination.min()) / (np.ptp(illumination) + 1e-6)
    texture = cv2.GaussianBlur(rng.standard_normal((size, size)).astype(np.float32), (0, 0), 1.5)
    texture /= texture.std()
    # the generator draws a fixed number of vessels for a whole camera frame: cut the scene from a field of that
    # size, so the vessels are as far apart as in a recording and open tissue lies between them
    field = 1200
    vessels = _vessels(rng, field, field)[(field - size) // 2:(field + size) // 2, (field - size) // 2:(field + size) // 2]
    return 1800.0 * illumination * (1 + 0.02 * texture) * (1 - vessels)


def render(scene, positions, rng, glint_frames=(), blurred_frames=(), glint_at=(160.0, 160.0)):
    """Render frames of the scene seen with the eye at each (dy, dx) position, with shot noise.

    A glint on the tissue (it moves with it) is in view in glint_frames; blurred_frames are smeared."""
    margin = (scene.shape[0] - SIZE) // 2
    frames = []
    for number, (dy, dx) in enumerate(positions):
        # the tissue moves by (dy, dx) in the frame: frame pixel p shows scene pixel p - (dy, dx) + margin
        view = np.float32([[1, 0, margin - dx], [0, 1, margin - dy]])
        frame = cv2.warpAffine(scene, view, (SIZE, SIZE), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        if number in glint_frames:
            # a clipped spot of about 40 pixels, the size of the glints in the recordings
            yy, xx = np.mgrid[:SIZE, :SIZE]
            frame = np.where(np.hypot(yy - (glint_at[0] + dy), xx - (glint_at[1] + dx)) < 3.5, SATURATION, frame)
        if number in blurred_frames:
            frame = cv2.GaussianBlur(frame, (0, 0), 5)
        frame = frame + rng.standard_normal(frame.shape).astype(np.float32) * np.sqrt(np.maximum(frame, 0) / 8)
        frames.append(np.clip(np.rint(frame), 0, SATURATION).astype(np.uint16))
    return frames


def write_burst(folder, frames):
    """Write frames as the recorder does: one TIFF per frame, and the frame rate in manifest.json."""
    folder.mkdir(parents=True)
    for number, frame in enumerate(frames):
        Image.fromarray(frame).save(folder / f"frame_{number:06d}.tif")
    (folder / "manifest.json").write_text(json.dumps({"capture": {"effective_fps": FPS}}), encoding="utf-8")


def fixational_path(rng, frames, start=(0.0, 0.0), step=1.0):
    """A random walk of small steps, as the eye jitters within a fixation."""
    return np.asarray(start) + np.cumsum(rng.normal(0, step, (frames, 2)), axis=0)


def notebook_script(work):
    """Export the notebook to a plain script in the folder work, with its animated previews off."""
    path = Path(work) / "foundation_script.py"
    subprocess.run([sys.executable, "-m", "marimo", "export", "script", str(NOTEBOOK), "-o", str(path)],
                   check=True, capture_output=True)
    text = path.read_text(encoding="utf-8")
    assert text.count("SHOW_PREVIEWS = True") == 1, "the notebook's SHOW_PREVIEWS setting has moved"
    path.write_text(text.replace("SHOW_PREVIEWS = True", "SHOW_PREVIEWS = False"), encoding="utf-8")
    return path


def run_notebook(burst, work):
    """Run the whole notebook on a burst in the folder work; return its printed log and the saved result."""
    environment = dict(os.environ, LIMBUS_BURST=str(burst), MPLBACKEND="Agg")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment.setdefault(name, "1")
    done = subprocess.run([sys.executable, str(notebook_script(work))], cwd=work, env=environment,
                          capture_output=True, text=True)
    log = done.stdout + done.stderr
    assert done.returncode == 0, f"the notebook failed on {burst.name}:\n{log[-4000:]}"
    with np.load(Path(work) / "preprocessed" / f"{burst.name}.npz") as saved:
        return log, {key: saved[key] for key in saved.files}


def check_alignment_keys(result, frames):
    """The saved kept frames, fixations, references and shifts must describe every kept frame consistently."""
    kept = result["gated_indices"]
    starts, lengths = result["fixation_starts"], result["fixation_lengths"]
    assert len(starts) == len(lengths) == len(result["references"]) == len(result["to_anchor"])
    assert starts[0] == 0 and np.all(starts[1:] == starts[:-1] + lengths[:-1]) and lengths.sum() == len(kept)
    assert np.all((result["references"] >= starts) & (result["references"] < starts + lengths))
    assert result["shifts"].shape == (len(kept), 2)
    assert np.allclose(result["frame_times"], kept / FPS)
    assert np.all(np.diff(kept) > 0) and kept[0] >= 0 and kept[-1] < frames


def test_glint_frames_are_kept():
    rng = np.random.default_rng(7)
    frames = 80
    glint_frames = range(20, 50)   # in view for 30 of 80 frames: the gate's median is that of frames without it
    blurred_frames = (5, 60, 70)
    positions = fixational_path(rng, frames)
    scene = make_scene(rng, SIZE + 96)
    with tempfile.TemporaryDirectory() as work:
        burst = Path(work) / "burst_glint"
        write_burst(burst, render(scene, positions, rng, glint_frames, blurred_frames))
        log, result = run_notebook(burst, work)

    kept = result["gated_indices"]
    dropped = sorted(set(range(frames)) - set(kept.tolist()))
    assert set(glint_frames) <= set(kept.tolist()), f"frames with the glint in view were dropped: {dropped}"
    assert set(blurred_frames) <= set(dropped), f"blurred frames were kept; dropped: {dropped}"
    assert len(dropped) <= len(blurred_frames) + 2, f"too many frames dropped: {dropped}"
    assert "Glare left out of the scores" in log

    check_alignment_keys(result, frames)
    # each saved shift is the kept frame's motion from its fixation's reference frame
    for start, length, reference in zip(result["fixation_starts"], result["fixation_lengths"], result["references"]):
        members = np.arange(start, start + length)
        true_shifts = positions[kept[members]] - positions[kept[reference]]
        error = np.linalg.norm(result["shifts"][members] - true_shifts, axis=1)
        assert np.median(error) < 0.3, f"saved shifts are off the true motion by {np.median(error):.2f} px (median)"


def test_short_fixations_finish():
    rng = np.random.default_rng(11)
    lengths = (12, 14, 16)  # with the slow movement taken out, at most 18 frames per pixel: fewer than MIN_FRAMES
    starts = ((0.0, 0.0), (45.0, -40.0), (-30.0, 35.0))  # a saccade of 50 to 90 px between fixations
    positions = np.concatenate([fixational_path(rng, n, start, 0.7) for n, start in zip(lengths, starts)])
    scene = make_scene(rng, SIZE + 256)
    with tempfile.TemporaryDirectory() as work:
        burst = Path(work) / "burst_short"
        write_burst(burst, render(scene, positions, rng))
        log, result = run_notebook(burst, work)

    assert "No pixel is judged" in log, "the notebook did not report that no pixel's fluctuation could be judged"
    check_alignment_keys(result, len(positions))
    assert len(result["fixation_starts"]) == len(lengths), f"fixations found: {result['fixation_lengths'].tolist()}"
    assert not result["quiet_pixels"].any(), "pixels were called quiet without being judged"
    assert result["background_pixels"].any(axis=(1, 2)).all(), "a fixation was left without a background"


if __name__ == "__main__":
    failed = []
    for test in (test_glint_frames_are_kept, test_short_fixations_finish):
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception as error:  # a failed check, or a key missing from the saved result
            failed.append(test.__name__)
            print(f"FAIL {test.__name__}: {error}")
    sys.exit(1 if failed else 0)
