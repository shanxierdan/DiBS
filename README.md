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

## Acknowledgement

DiBS uses the discrete diffusion model from
[HKUNLP/diffusion-vs-ar](https://github.com/HKUNLP/diffusion-vs-ar) as its
learned branch-ordering prior. We thank the authors of
[*Beyond Autoregression: Discrete Diffusion for Complex Reasoning and
Planning*](https://arxiv.org/abs/2410.14157) for making their model and code
publicly available. Their work is an important foundation for this project.

If you use DiBS, cite this:
```bibtex
@misc{liu2026dibsdiffusioninformedbranchselection,
      title={DiBS: Diffusion-Informed Branch Selection}, 
      author={Bo Liu and Yuan Xie and Yuan Gao and Xiaolong Luo and Peng Ye and Tao Chen and Fujun Han},
      year={2026},
      eprint={2606.06518},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2606.06518}, 
}
```

please also cite:

```bibtex
@article{ye2024beyond,
  title={Beyond Autoregression: Discrete Diffusion for Complex Reasoning and Planning},
  author={Ye, Jiacheng and Gao, Jiahui and Gong, Shansan and Zheng, Lin and Jiang, Xin and Li, Zhenguo and Kong, Lingpeng},
  journal={arXiv preprint arXiv:2410.14157},
  year={2024}
}
```

## Code Overview

- `DiBS/solver.py`: complete Sudoku search with diffusion-informed ordering.
- `DiBS/heuristic.py`: branch-ranking logic and consistency-aware scoring.
- `DiBS/model_wrapper.py`: adapter for a local diffusion checkpoint.
- `DiBS/sat_solver_dibs.py`: DiBS adaptation for satisfiable 3-SAT.
- `DiBS/run_table*.py`: experiment entry points used in the project.

## Citation

Citation information will be added after the paper is publicly available.
