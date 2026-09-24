# Camera Video Recorder

A simple app for recording video from lab cameras. Supports:

- **Basler** cameras (via the pylon driver) — e.g. ace acA1920-40um,
  acA1300-200um
- **The Imaging Source polarization cameras** (via Aravis/USB3 Vision) —
  e.g. DZK 33UX250
- **The Imaging Source** standard mono/color cameras and other UVC
  cameras (via the standard Linux driver) — e.g. DMK 33UX250

## How to start it

Double-click **Camera Recorder** on the Desktop, or run:

```bash
./run.sh
```

## How to use it

1. The live camera image appears automatically when the app opens. If more
   than one camera is plugged in, pick one from the dropdown at the top
   right. Press **⟳** to re-scan after plugging in a camera.
2. The big red **● Record** button works two ways:
   - **Hold it down** to record a short clip — recording runs for exactly
     as long as you hold, and stops the moment you let go.
   - **Tap it once** to start a longer recording that keeps running
     hands-free, then tap again (**■ Stop**) to finish.

   Anything held for less than half a second counts as a tap, so a clip
   is always at least that long. Videos are saved as `.mp4` files in the
   `recordings` folder, named by date and time.
3. **📷 Snapshot** saves a single still image (`.png`).
4. **⦿ Capture burst** saves a fixed number of frames as individual
   lossless TIFFs — see *TIFF bursts* below. This is the mode to use for
   measurement: unlike video, it keeps all 12 bits.
5. Camera settings (right-hand panel):
   - **Frame rate** — frames per second of preview and recording.
   - **Auto brightness** — let the camera pick exposure and gain
     automatically (on by default). Turn it off for manual control. Either
     way, the *Live values* panel shows what the camera is currently
     using, and bursts record it frame by frame.
   - **Pixel format** — `Mono8` (256 grey levels, fastest), `Mono12`
     (4096 levels, the full measurement range) or `Mono12p` (the same 12
     bits packed smaller on the bus; on the acA1920-40um it reads out no
     faster than `Mono12`). Changing this reconnects the camera, which
     takes a few seconds; the camera reloads its factory defaults, but the
     app re-applies your exposure, gain, frame rate and advanced settings
     afterwards.
   - **Resolution** — the region of interest. **Set** applies the typed
     size, **Full** returns to the sensor's specified imaging area. The
     hint underneath shows the highest frame rate possible at that ROI,
     measured from the camera rather than assumed.
   - **Top of ROI** — which sensor row the region starts at, so a short,
     fast band can be placed over the vessel you care about instead of
     sitting in the middle of the sensor. **Centre** re-centres it.
   - **Speed preset** — picks an ROI height for a target frame rate and
     asks the camera for that rate, which also caps how long
     auto-exposure may expose. Frame rate costs vertical field of view:
     see *Measured performance* below.
   - **Exposure** — how long each frame gathers light, in microseconds.
     Longer means brighter, but caps the frame rate: a warning appears
     when the exposure is what's limiting the rate.
   - **Gain** — electronic brightness boost (higher = brighter but noisier).
   - **Advanced** — black level, gamma, and the limits auto-brightness
     works within (target brightness, longest exposure and highest gain it
     may choose). Controls the camera doesn't offer are hidden.
   - **Live values** — exposure, gain, black level, measured and maximum
     frame rate, readout time, sensor temperature and the count of
     incomplete frames, re-read from the camera twice a second. Values
     marked *(auto)* are being chosen by the camera, and you can watch
     them adapt.
6. **Save to** — change or open the folder where recordings are stored.

## TIFF bursts

Set the **Length** — either a number of frames or a number of seconds
(the frame count is then worked out from the rate the camera is actually
achieving) — press **⦿ Capture burst**, and every frame is saved as a
16-bit TIFF in a new `burst_<date>_<time>` folder. Pixel values are true
sensor DN (0–4095 for 12-bit), *not* rescaled to fill the 16-bit range, so
measurements match what the sensor reported.

Each burst folder holds:

- `frame_000000.tif`, … — one lossless image per frame. Each file also
  carries its own metadata in the TIFF description tag.
- `manifest.json` — camera model, serial, firmware and link speed; the
  requested pixel format, ROI and frame rate; the camera's own reported
  state read at the *start* and at the *end* of the burst; how many frames
  were written and dropped and the effective frame rate; and the software
  versions used.
- `frames.csv` — one row per frame: index, host timestamp, the camera's
  own timestamp, and the **exposure and gain that frame was actually taken
  with**. These come from the camera's chunk data, attached to each image
  by the camera itself, so auto-exposure adaptation can be reconstructed
  frame by frame afterwards rather than inferred.

If the disk can't keep up, frames are dropped rather than silently
delaying the burst, and the count is reported both on screen and in
`manifest.json`.

## Review tab

Every capture can be played back without leaving the app. The **Review**
tab lists what is in the output folder, newest first, and selects each new
burst or recording as soon as it finishes.

- **Scrub and step** through the frames, or play them at any rate;
  **Real time** plays at the rate they were actually captured.
- **This frame** shows the exposure, gain and camera timestamp recorded
  for the frame on screen — the values that frame was taken with, so you
  can see auto-exposure adapting as it plays.
- **Capture** summarises the burst: frames, effective rate, frames
  dropped, pixel format, ROI and where the band sat on the sensor.
- **Stretch contrast** scales each frame between its own darkest and
  brightest pixel, which makes faint detail visible. It changes frame to
  frame, so don't judge brightness by it.

TIFF bursts and H.264 `.mp4` files play here. Lossless FFV1 usually plays
too; 12-bit HEVC may not, since it depends on the codecs OpenCV was built
with — those files open in VLC or ImageJ, and the tab says so rather than
failing silently.

### Stabilizing a burst

The **Stabilization** panel runs the offline stabilization in
`analysis/stabilize` on the selected TIFF burst:

- **Method** — *Translation (validated)* shifts each frame, and has been
  checked against synthetic bursts with known motion (0.06–0.12 px RMS
  error). *Non-rigid (experimental)* also corrects rotation and
  magnification between fixations, which translation leaves as doubled
  vessels toward the corners of wide frames; it is not yet validated the
  same way.
- **Stabilize** runs it in a separate process, so the window stays
  responsive; progress appears underneath, and **Cancel** stops it. A
  burst takes from under a minute (short strips) to a few minutes (full
  frames).
- **View** — *Raw*, *Stabilized* (every frame warped by its correction,
  with frames the result didn't use labelled) or *Stabilized mean* (the
  average of the frames used; **magenta** marks regions too few frames
  saw, never black, which would look like a vessel).
- The status line gives the **stability index** (how well vessel masks
  agree across frames after stabilization), the usable fraction and
  rejected frames, and warns when a result was made by older code or
  settings.

Results go to a `stabilization/<method>/<burst>/` folder **beside** the
recordings folder — raw bursts are never written to. See
[`analysis/stabilize/METHODS.md`](analysis/stabilize/METHODS.md) for how
every step works and how it was validated.

## Analysis on another computer

Stabilization and vessel annotation need no camera, driver or GUI — only
Python 3.10+ and a few packages:

```bash
git clone https://github.com/karimghabra/limbus.git
cd limbus
python -m venv venv
venv/Scripts/pip install -r requirements-analysis.txt      # Windows
# ./venv/bin/pip install -r requirements-analysis.txt     # macOS / Linux
```

**Reference bursts.** Five raw bursts (1.2 GB download, 1.7 GB extracted)
are published as assets of the GitHub release `reference-data-v1`, so
algorithms can be compared on identical input. Fetch and verify them
(SHA-256 of every file) with:

```bash
python tools/fetch_reference_data.py
```

They land in `reference_data/`; [`reference_data/README.md`](reference_data/README.md)
describes what each burst exercises.

**Command line.** From the `analysis` folder:

```bash
cd analysis
python -m stabilize ../reference_data                                   # every burst, translation
python -m stabilize ../reference_data/burst_2026-09-16_15-50-52 --method nonrigid
```

Results go to `stabilization/<method>/<burst>/` beside the input folder
(or `--out DIR`), with `summary.html` ranking every burst. Re-running skips
results that are already current — same code version and parameters.

**Lucky image and vessel annotation.** Once a burst is stabilized:

```bash
python -m lucky ../reference_data                  # after python -m stabilize ... --method nonrigid
```

This re-warps each raw frame with the saved transforms and fuses them band
by band. At each place, fine detail is weighted towards the frames that
were sharp there, which gives `lucky.tif`. It then traces the vessels:
centrelines, a half-maximum vessel mask, widths, and a per-segment table.
Results go to `stabilization/lucky/<burst>/`, with `compare.png` setting the
new annotation beside the skeleton of the stabilization's consensus mask,
and `summary.csv` comparing every burst. See
[`analysis/lucky/METHODS.md`](analysis/lucky/METHODS.md) for how it works
and how it was validated.

**Tests.** From the repository root:

```bash
python analysis/tests/test_synthetic.py       # translation vs known motion
python analysis/tests/test_nonrigid_smoke.py  # non-rigid sanity checks
python analysis/tests/test_lucky_synthetic.py # annotation vs known vessels
```

## Repository layout

| Path | Contents |
|---|---|
| `camera_recorder.py`, `run.bat`, `run.sh` | the recorder app and its launchers |
| `analysis/stabilize/` | offline stabilization package, with `METHODS.md` |
| `analysis/lucky/` | lucky fusion and vessel annotation of stabilized bursts, with `METHODS.md` |
| `analysis/tests/` | synthetic ground-truth and smoke tests |
| `benchmarks/` | camera, encoder and disk benchmarks, and GUI tests of the app |
| `tools/` | building and fetching the reference-data release |
| `reference_data/` | manifest and description of the reference bursts (data fetched, not committed) |

## Troubleshooting

- **"No camera found"** — check the USB cable and close any other camera
  program (Pylon Viewer, tcam-capture — only one program can use a camera
  at a time), then press ⟳ or Retry.
- Frame rate lower than requested — the exposure time is too long for that
  frame rate. Shorten the exposure or lower the frame rate.
- If the encoder can't keep up, the app reports dropped frames; either way
  the saved video is re-stamped so it always plays at true speed.
- **Quality / codec.** The first three presets are H.264 `.mp4`, playable
  anywhere, but **H.264 here carries only 10 of the camera's 12 bits** —
  libx264 in this build has no 12-bit support and silently downgrades a
  12-bit request. *High quality* preserves full detail but noisy scenes
  make large files (up to ~1 GB/min); *Balanced* and *Compact* cap the
  bitrate for predictable sizes at some cost in fine detail. Two presets
  keep all 12 bits, both `.mkv`:
  - **12-bit HEVC** — visually lossy but true 12-bit, and fast enough to
    keep up with this camera at any rate it can produce.
  - **12-bit lossless (FFV1)** — mathematically identical to the sensor
    data, but the slowest and by far the largest (~2.5 MB per frame).
- Less image noise = smaller files and cleaner video: more light on the
  subject lets auto-brightness use less gain.
- **For measurement, prefer TIFF bursts** over any video format: they are
  lossless, keep all 12 bits, and carry per-frame metadata.
- The DZK 33UX250 is a *polarization* camera: every 2×2 pixel block holds
  four polarizer angles (0°/45°/90°/135°). Recordings store this raw
  mosaic, so polarization analysis can be done later on the saved files.

## Technical notes

### Measured performance (i7-10810U, 6 cores / 12 threads, NVMe SSD)

With the Basler acA1920-40um. Re-measure with the scripts in
`benchmarks/` after any hardware change — `ENCODER_THROUGHPUT_MBS` and the
per-codec `throughput_mbs` values in `camera_recorder.py` come from them,
and drive the default frame rates and "can't keep up" warnings.

| Stage | Sustained rate (1928×1208, 12-bit) |
|---|---|
| Camera, Mono12 (full frame) | **32.0 fps** — the hard limit |
| Camera, Mono12 at 1928×958 ROI | **40.0 fps** |
| Camera, Mono8 (full frame) | 41.3 fps |
| H.264 10-bit encode | 86–113 fps (400–530 MB/s) |
| HEVC 12-bit encode | 45 fps (210 MB/s) |
| FFV1 lossless encode | 22 fps (103 MB/s) |
| TIFF to disk | 407 fps (1895 MB/s) |

The 12-bit frame rate is limited by **sensor readout**, roughly 25 µs per
row, not by the bus, the encoder or the disk: `Mono12p` packing and
raising `DeviceLinkThroughputLimit` both change nothing. Shorter ROIs are
therefore the only way to go faster — the app shows the achievable rate
for the current ROI, measured from the camera, under the Resolution row.

Encoder figures come from realistic noisy vessel imagery; a flat or
blank test pattern encodes far faster (149 fps versus 86) and would
overstate every one of them.

- Python + PyQt5 GUI; frames are piped to `ffmpeg` for H.264 encoding.
  Basler cameras are driven with
  [pypylon](https://github.com/basler/pypylon); The Imaging Source 33U
  cameras are UVC-compliant, so they use the standard Linux V4L2 driver
  (OpenCV capture + `v4l2-ctl` for exposure/gain control) — no extra SDK
  needed.
- The virtual environment lives in `venv/`. To recreate it:

```bash
python3 -m venv --system-site-packages venv && ./venv/bin/pip install -r requirements.txt
```

  On Windows: `python -m venv venv` then
  `venv\Scripts\pip install -r requirements.txt`, and install the Basler
  pylon runtime for the camera's USB driver.

- On startup the app resets the camera to factory defaults, then applies
  full resolution, 8-bit pixel format, 10 ms exposure, 30 fps.
