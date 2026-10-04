# vesselnet

Learning the vessel graph of real averaged frames from synthetic vesselscene scenes. The learned part proposes
the graph, and `vesselmap`'s render-and-residual optimisation finishes it.

- [PLAN.md](PLAN.md): the plan, written for a GPU session. Start at §0.
- [RESULTS.md](RESULTS.md): the results log, one entry per iteration.
- [pilot/](pilot/): the CPU proof of concept.

This folder is mirrored in [vesselscene](https://github.com/karimghabra/vesselscene) on its branch
`claude/vesselnet-plan`. The work needs both repositories (PLAN.md §0). Choose one home for the code and results
at the start of iteration 0, and keep only a pointer in the other.
