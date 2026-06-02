# DiBS: Diffusion-Informed Branch Selection

DiBS is a diffusion-informed branch-selection method for complete symbolic
solvers. It uses a learned diffusion prior to improve candidate value ordering
at high-leverage branching states while preserving the correctness and
completeness guarantees of the underlying search procedure.

## Motivation

Classical constraint satisfaction solvers are complete, but hard instances can
still trigger expensive long-tail search when an early branch is explored in
the wrong order. Learned models capture useful global structure, but using them
as end-to-end solvers may sacrifice exactness.

DiBS combines the two approaches:

- The symbolic solver keeps constraint propagation, variable selection,
  backtracking, and completeness.
- The diffusion model evaluates the current partial assignment.
- Candidate values are reordered using diffusion preferences and a lightweight
  consistency signal.
- No candidate is pruned, so the symbolic solver remains complete.

The current implementation includes the Sudoku solver and an adaptation of the
same branch-ordering principle to satisfiable 3-SAT instances.

## Repository Structure

```text
.
├── DiBS/       # DiBS solver and experiment programs
├── dataset/    # placeholder for local datasets
└── model/      # placeholder for local checkpoints
```

Datasets, pretrained checkpoints, generated outputs, and third-party model
implementations are intentionally not included in this repository.

## Code Overview

- `DiBS/solver.py`: complete Sudoku search with diffusion-informed ordering.
- `DiBS/heuristic.py`: branch-ranking logic and consistency-aware scoring.
- `DiBS/model_wrapper.py`: adapter for a local diffusion checkpoint.
- `DiBS/sat_solver_dibs.py`: DiBS adaptation for satisfiable 3-SAT.
- `DiBS/run_table*.py`: experiment entry points used in the project.

## Citation

Citation information will be added after the paper is publicly available.
