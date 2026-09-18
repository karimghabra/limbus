# LIMBUS reference data

Five raw TIFF bursts from the LIMBUS camera (Basler acA1920-40um). They are a fixed, shared input set, so stabilization and analysis algorithms can be test-run on other computers and the results compared against the same bytes. The images are too large for the git repository. They are published as zip assets on the GitHub release `reference-data-v1`. `manifest.json` in this folder lists every zip and every file inside it, with sizes and SHA-256 hashes.

## The bursts

| # | Burst | Description | Frames | Geometry | Download / extracted |
|---|-------|-------------|-------:|----------|---------------------:|
| 1 | `burst_2026-09-16_15-50-52` | Clean fixation with one saccade at ~3.9 s; two distinct eye poses (reference case for single-mode / non-rigid stabilization). | 140 | 1920x1200 Mono12 @ 32.2 fps | 476.4 / 645.2 MB |
| 2 | `burst_2026-09-16_15-45-12` | Pronounced peripheral misalignment (rotation/magnification jitter); the 'doubling' test case. | 84 | 1920x1200 Mono12 @ 32.2 fps | 264.6 / 387.1 MB |
| 3 | `burst_2026-09-16_15-40-55` | Hard case: low usable fraction (~37%) and non-converging translation registration. | 63 | 1920x1200 Mono12 @ 32.3 fps | 187.3 / 290.3 MB |
| 4 | `burst_2026-09-16_15-31-37` | Short-ROI geometry, clean. | 160 | 1920x500 Mono12 @ 73.7 fps | 208.5 / 307.3 MB |
| 5 | `burst_2026-09-16_12-57-20` | Thin strip geometry (little vertical capture range). | 296 | 1920x100 Mono12p @ 148.7 fps | 78.8 / 113.8 MB |

Total: 1215.6 MB to download and 1743.8 MB on disk once extracted (MB = 10^6 bytes). The frame rate is the rate the camera actually delivered. Full-frame bursts 2 and 3 requested 74 fps, and burst 1 requested 30 fps. At 1920x1200 the sensor readout limits all three to about 32 fps.

## Fetching

Run this from the repository root with any Python 3.8+. It uses only the standard library.

```
python tools/fetch_reference_data.py
```

For each burst the script:

1. Downloads `<burst>.zip` from the release and shows progress in MB.
2. Checks the zip's size and SHA-256 against `manifest.json`. If they do not match, that burst is not extracted and an error names the zip.
3. Extracts the zip into a temporary folder.
4. Checks every extracted file's size and SHA-256.
5. Moves the files into place only if all of them pass.

If a burst is already present and every file passes the check, it is skipped. If any file is missing or altered, the burst is fetched again. The exit code is 0 when every requested burst is present and verified, 1 if any burst failed, and 2 for bad arguments or an unreadable manifest.

Options:

- `--source DIR` uses zips already in a local folder (for example, copied from a USB drive) instead of downloading them. They are verified the same way.
- `--only NAME` fetches just one burst. You can repeat it, for example `--only burst_2026-09-16_12-57-20`.
- `--dest DIR` sets where the burst folders go. The default is this folder, `reference_data/`.
- `--keep-zips` keeps downloaded zips in `--dest`. The next run reuses them after checking them again.
- `--manifest FILE` uses a different manifest. The default is `reference_data/manifest.json`.

## Where files land

With the default `--dest`, each burst is extracted to `reference_data/<burst>/`, for example `reference_data/burst_2026-09-16_12-57-20/frame_000000.tif`. Every file is byte-identical to the original recording. In `manifest.json`, file paths are given relative to `--dest`, in the form `<burst>/<file>`.

## Layout of a burst folder

- `frame_NNNNNN.tif`: one uncompressed 16-bit greyscale TIFF per frame, numbered from `000000`. The pixel values are right-aligned 12-bit sensor DN in the range 0-4095. They are not rescaled to the 16-bit range, so divide by 4095, not 65535, to normalise. Burst 5 was transferred from the camera as packed Mono12p but is stored in the same unpacked 16-bit form as the others.
- `manifest.json`: the recording's own metadata. It includes the camera model and serial number, requested acquisition settings (pixel format, ROI size and offsets, requested fps), camera state at the start and end (exposure, gain, gamma, auto-exposure limits, temperature), capture statistics (frames written and dropped, time span, effective fps), and the software versions used.
- `frames.csv`: one row per frame. Columns: `camera_timestamp_ns`, `exposure_us`, `gain`, `camera_counter`, `index`, `host_time_utc`, `host_monotonic_s`, `filename`. `camera_timestamp_ns` is the camera's hardware timestamp. Exposure and gain are the per-frame values reported by the camera, which can vary within a burst because auto-exposure was on.

## Rebuilding the release (maintainers)

`python tools/make_reference_release.py` zips the five bursts from `recordings/` into `../release_assets/`. It checks every zipped file against its source and rewrites this folder's `manifest.json`. It also writes `RELEASE_NOTES.md` next to the zips. Upload the five zips to the `reference-data-v1` release under exactly the names in the manifest.

## Consent, licence and caveat

Published with the informed consent of the imaged participant.

Licence: TO BE DECIDED by the repository owner before public reuse.

Caveat: Parallel "doubled" vessel lines are present in individual raw frames and are not a stabilization artefact; whether they are anatomy or an optical ghost is unresolved.
