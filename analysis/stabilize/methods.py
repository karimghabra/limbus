"""The stabilization methods, and where their results live.

Results for a burst are written to <out>/<method>/<burst>/ with the same
files for every method (METHODS.md §11), so the Review tab and the summary
read them the same way. A result is current only if it was produced by this
version of the code with these parameters — file times alone can't tell.
"""
import json
import os

from . import __version__
from .config import NonrigidParams, Params

METHODS = {
    "translation": {"label": "Translation (validated)", "experimental": False},
    "nonrigid": {"label": "Non-rigid (experimental)", "experimental": True},
}


def params_hash(method):
    if method == "translation":
        return Params().params_hash()
    if method == "nonrigid":
        return NonrigidParams().params_hash(Params())
    raise ValueError(f"unknown method {method!r}")


def result_dir(out_base, method, burst_name):
    return os.path.join(out_base, method, burst_name)


def read_result(out_base, method, burst_name):
    path = os.path.join(result_dir(out_base, method, burst_name), "metrics.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def is_current(burst_path, out_base, method):
    name = os.path.basename(os.path.normpath(burst_path))
    rec = read_result(out_base, method, name)
    if not rec or rec.get("status") not in ("ok", "skipped"):
        return False
    proc = rec.get("processing", {})
    if (proc.get("stabilize_version") != __version__
            or proc.get("params_hash") != params_hash(method)):
        return False
    path = os.path.join(result_dir(out_base, method, name), "metrics.json")
    return os.path.getmtime(path) >= os.path.getmtime(burst_path)


def run(method, burst_path, out_base, log=print):
    """Stabilize one burst with the named method; returns its metrics record."""
    if method == "translation":
        from .pipeline import process_burst
        return process_burst(burst_path, os.path.join(out_base, method), Params(), log=log,
                             method=method)
    if method == "nonrigid":
        from .nonrigid import process_burst_nonrigid
        return process_burst_nonrigid(burst_path, out_base, Params(), NonrigidParams(), log=log)
    raise ValueError(f"unknown method {method!r}")
