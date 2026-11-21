# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Pixel scale: the recordings do not store the magnification; a millimetre ruler imaged through the camera gives
  3.26 µm/px (docs/synthetic_conjunctiva_anatomy.md §1.3). Analyses default to pixels unless told the scale.
- Synthetic data with exact ground truth: `synth/` (`synth.simulate`, `synth.write_burst`, `notebooks/synthetic.py`).
  Every anatomical or physiological constant lives in `synth/anatomy.py` with its source in
  docs/synthetic_conjunctiva_anatomy.md; `synth.check` must keep passing (`python -m pytest synth/tests`).
- Notebooks are marimo files; `marimo check notebooks/<name>.py` catches names defined in two cells.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
