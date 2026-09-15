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
4. Camera settings (right-hand panel):
   - **Frame rate** — frames per second of preview and recording.
   - **Auto brightness** — let the camera pick exposure and gain
     automatically (on by default). Turn it off for manual control.
   - **Bit depth** — 8-bit (256 grey levels) or 12-bit (4096 levels) on
     cameras that support it. 12-bit gives far more dynamic range for
     measurement. Snapshots keep all 12 bits as 16-bit PNG files; video
     keeps 10 of them, and records at a lower frame rate (see below).
     Changing this reconnects the camera, which takes a few seconds.
   - **Exposure** — how long each frame gathers light (longer = brighter,
     but limits the maximum frame rate).
   - **Gain** — electronic brightness boost (higher = brighter but noisier).
5. **Save to** — change or open the folder where videos are stored.

## Troubleshooting

- **"No camera found"** — check the USB cable and close any other camera
  program (Pylon Viewer, tcam-capture — only one program can use a camera
  at a time), then press ⟳ or Retry.
- Frame rate lower than requested — the exposure time is too long for that
  frame rate. Shorten the exposure or lower the frame rate.
- Videos are standard H.264 `.mp4` files and play in VLC, browsers, etc.
- High-resolution cameras (like the 5-megapixel DZK 33UX250) default to
  15 fps because that is what the Raspberry Pi can encode at full
  resolution. You can raise the frame rate, but if the encoder can't keep
  up the app will report dropped frames; either way the saved video is
  re-stamped so it always plays at true speed.
- **Quality** setting: *High quality* (default) preserves full image
  detail but noisy scenes can produce very large files (up to ~1 GB per
  minute) — the status bar shows file size and free disk space while
  recording, and recording stops automatically if the disk is nearly
  full. *Balanced* and *Compact files* limit the bitrate for smaller,
  predictable file sizes at some cost in fine detail.
- Less image noise = smaller files and cleaner video: more light on the
  subject lets auto-brightness use less gain.
- **12-bit frame rates.** The Pi encodes 12-bit video at roughly 10 fps
  at 1928×1208, so that is the default in 12-bit mode. The camera itself
  still grabs much faster, so the live preview and snapshots are
  unaffected. You can raise the frame rate, but the app will start
  reporting dropped frames. For full-speed recording use 8-bit; for
  maximum measurement accuracy use 12-bit snapshots.
- The DZK 33UX250 is a *polarization* camera: every 2×2 pixel block holds
  four polarizer angles (0°/45°/90°/135°). Recordings store this raw
  mosaic, so polarization analysis can be done later on the saved files.

## Technical notes

- Python + PyQt5 GUI; frames are piped to `ffmpeg` for H.264 encoding.
  Basler cameras are driven with
  [pypylon](https://github.com/basler/pypylon); The Imaging Source 33U
  cameras are UVC-compliant, so they use the standard Linux V4L2 driver
  (OpenCV capture + `v4l2-ctl` for exposure/gain control) — no extra SDK
  needed.
- The virtual environment lives in `venv/`. To recreate it:

```bash
python3 -m venv --system-site-packages venv && ./venv/bin/pip install pypylon
```

- On startup the app resets the camera to factory defaults, then applies
  full resolution, 8-bit pixel format, 10 ms exposure, 30 fps.
