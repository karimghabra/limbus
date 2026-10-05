# vesselnet

Learning the vessel graph of real averaged frames from synthetic vesselscene scenes. The learned part proposes
the graph, and `vesselmap`'s render-and-residual optimisation finishes it.

- [PLAN.md](PLAN.md): the plan, written for a GPU session. Start at §0.
- [RESULTS.md](RESULTS.md): the results log, one entry per iteration.
- [pilot/](pilot/): the CPU proof of concept.

This folder is the only home of vesselnet's code and results. The work also needs
[vesselscene](https://github.com/karimghabra/vesselscene) checked out beside this repository (PLAN.md §0); its
`vesselnet/` folder is only a pointer here.
