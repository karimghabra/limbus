"""Vessel segmentation for the Review tab's overlay, with the segmenter swappable.

A *segmenter* is one Python file in segmentation/plugins/ (or in a folder
listed in the LIMBUS_SEGMENTERS environment variable). The Review tab lists
every one it finds; choosing one and pressing Find vessels runs

    python -m segmentation run <id> <stabilization result folder> [--burst DIR]

in a separate process, which calls the file's segment() and writes what it
returns as <result folder>/segmentation/<id>/overlay.json. The app only ever
reads that file, so a segmenter can be rewritten, replaced or added without
touching the app. See README.md in this folder for how to write one.

Nothing here imports a segmenter to list it: its LABEL, DESCRIPTION and
VERSION are read from the file's source, so a segmenter's own dependencies
(PyTorch, say) are loaded only in the process that runs it.
"""
import ast
import hashlib
import importlib.util
import os
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
BUILTIN_DIR = os.path.join(HERE, "plugins")
ENV_VAR = "LIMBUS_SEGMENTERS"
RESULTS = "segmentation"          # <stabilization result>/segmentation/<id>/


@dataclass
class Segmenter:
    id: str                 # the file name without .py: also its results folder
    path: str
    label: str
    description: str
    version: str

    def source_hash(self):
        return source_hash(self.path)


def source_hash(path):
    """Fingerprint of a segmenter's file, recorded in each overlay: an overlay
    made by an earlier version of the file can then be flagged. Line endings
    are normalised, so a checkout with CRLF matches one with LF."""
    with open(path, "rb") as f:
        data = f.read().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()[:12]


def plugin_dirs():
    """Where segmenters are looked for: LIMBUS_SEGMENTERS folders first, so a
    segmenter under development there can stand in for a built-in one of the
    same name, then segmentation/plugins/."""
    extra = [d for d in os.environ.get(ENV_VAR, "").split(os.pathsep) if d.strip()]
    return [os.path.abspath(d) for d in extra] + [BUILTIN_DIR]


def _constants(path):
    """The module-level string constants and docstring of a file, without
    running it."""
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), path)
    out = {"__doc__": ast.get_docstring(tree) or ""}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
            out[node.targets[0].id] = node.value.value
    return out


def list_segmenters():
    """Every segmenter found, in a stable order (built-in and extra folders
    alike, sorted by label). Files that start with '_' are skipped, as are
    files that don't define segment()."""
    found = {}
    for folder in plugin_dirs():
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if not name.endswith(".py") or name.startswith("_"):
                continue
            sid = name[:-3]
            if sid in found:
                continue
            path = os.path.join(folder, name)
            try:
                c = _constants(path)
                with open(path, encoding="utf-8") as f:
                    if "def segment(" not in f.read():
                        continue
            except (OSError, SyntaxError, UnicodeDecodeError):
                continue
            doc = c["__doc__"].strip().splitlines()
            found[sid] = Segmenter(
                id=sid, path=path, label=c.get("LABEL") or sid,
                description=c.get("DESCRIPTION") or " ".join(doc).strip(),
                version=c.get("VERSION", ""))
    return sorted(found.values(), key=lambda s: s.label.lower())


def find(sid):
    for s in list_segmenters():
        if s.id == sid:
            return s
    raise KeyError(f"no segmenter {sid!r}; found: {', '.join(s.id for s in list_segmenters())}")


def load(segmenter):
    """Import a segmenter's file as a module of its own (so a file named like
    a package it uses, e.g. vesselmap.py, doesn't shadow that package)."""
    spec = importlib.util.spec_from_file_location(f"_segmenter_{segmenter.id}", segmenter.path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "segment", None)):
        raise TypeError(f"{segmenter.path} has no segment(inputs) function")
    return module


def result_dir(stabilization_result, sid):
    """Where segmenter `sid` writes for this stabilization result."""
    return os.path.join(stabilization_result, RESULTS, sid)
