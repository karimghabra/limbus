# The Windows installer

`LIMBUS-Setup-<version>.exe` installs the app on a Windows 10/11 (x64)
laptop that has nothing installed: no Python, no ffmpeg, no GPU. Everything
the app runs comes with it. The only other software a Basler camera needs is
Basler's **pylon** (for its USB driver).

GitHub builds it (`.github/workflows/windows-installer.yml`) on every push to
`main` and every pull request, tests it end to end on a clean runner, and
keeps it as a build artifact. Pushing a tag `vX.Y.Z` that matches `VERSION`
also publishes it as a GitHub release:

```bash
git tag v0.1.0 && git push origin v0.1.0
```

## What gets installed

The setup needs no administrator rights. By default it installs for the
current user, in `%LOCALAPPDATA%\Programs\LIMBUS`; it can also install for
all users. It adds:

| | |
|---|---|
| `runtime\` | the app's own Python 3.12 (python.org's relocatable build) with every package pinned in `installer/requirements.txt`: CPU-only PyTorch, OpenCV, PyQt5, pypylon, SciPy, scikit-image, and imageio-ffmpeg, whose static ffmpeg records H.264, 12-bit HEVC and FFV1. The Visual C++ runtime DLLs sit beside `python.exe`. |
| `app\` | the repository's `camera_recorder.py`, `analysis\`, `vesselmap\`, `assets\` and `e2e\`, laid out as in git, plus `build_info.json` (version, commit, packages) |
| Start menu | **LIMBUS**, which runs `runtime\pythonw.exe -s app\camera_recorder.py`, and **LIMBUS Self-Test** (see below); a desktop shortcut is optional |

The install folder's path can be at most about 110 characters: the deepest
file inside it (in PyTorch) is about 140, and Windows limits a file path to
260. The setup refuses a deeper folder before copying anything; the default
is far shorter.

Installed, the app keeps recordings in `Documents\LIMBUS\recordings` (and
stabilization results beside them), never in its program folder, and
writes errors to `%LOCALAPPDATA%\LIMBUS\limbus.log`. An uninstall removes
the program folder and shortcuts but leaves the recordings. Installing a new
version over an old one replaces `runtime\` and `app\` wholesale.

Without a GPU, stabilization and vesselmap run on the CPU (slower, same
results), and the Camera tab's live stabilization says it needs a GPU.

## Self-test

**LIMBUS Self-Test** (or `runtime\python.exe -s app\e2e\app_scenarios.py`)
drives the installed app end to end, with a replay camera playing a
synthetic vessel scene that drifts by a known amount. It records with every
video preset, captures bursts and a snapshot (checked bit-exact), plays them
back, stabilizes with both methods (checked against the known drift), finds
vessels with both segmenters, and checks that the app, its analysis, and
every DLL they load come from the installation alone. With a Basler camera
plugged in, it also records a short video and burst through pylon. It takes
a few minutes, mostly vesselmap on the CPU.

## Building it locally

On 64-bit Windows with Python 3.8+, git, and
[Inno Setup 6](https://jrsoftware.org/isinfo.php):

```bash
python installer/build_bundle.py
```

```bash
iscc installer\limbus.iss
```

The first command assembles `build\bundle\` (about 1.5 GB; downloads are
cached in `build\cache` and pip's cache). The second packs it into
`dist\LIMBUS-Setup-<version>.exe`. To test that installer the way CI does:

```bash
python e2e/run_e2e.py --installer dist\LIMBUS-Setup-0.1.0.exe
```

That checks the installer refuses a folder too deep for Windows' path limit,
installs it silently for the current user where it installs by default (or
into `--dir`), installs it again over itself (an upgrade), runs the self-test
with no Python or ffmpeg on `PATH`, starts the app from its Start-menu
shortcut and closes it, then uninstalls it and checks nothing is left behind.
It refuses to run if LIMBUS is already installed for the user.

## Changing what's in it

- **A package**: change its pin in `installer/requirements.txt` (pin
  everything, dependencies included, to versions tested together).
- **App files**: `APP_PATHS` in `build_bundle.py` lists what goes in `app\`.
  Only files committed to git are packed, so local data never ends up in it.
- **The version**: edit `VERSION`.
