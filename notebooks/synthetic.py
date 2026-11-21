import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import sys
    from pathlib import Path

    import marimo as mo

    # the notebook may be opened through a symlink: import the synth package from the checkout it really lives in
    REPOSITORY = Path(__file__).resolve().parents[1]
    for _path in (REPOSITORY, REPOSITORY / "notebooks"):
        if str(_path) not in sys.path:
            sys.path.insert(0, str(_path))
    return REPOSITORY, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # A synthetic conjunctiva

    *Under construction.* A patch of bulbar conjunctiva built from anatomy and physiology: an arteriole-to-venule
    network with anastomoses in two depth layers, blood flow that conserves mass at every junction, red cells
    carried through each vessel's circular cross-section, and the camera that records them.
    """)
    return


if __name__ == "__main__":
    app.run()
