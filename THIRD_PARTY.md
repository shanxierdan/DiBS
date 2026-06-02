# Third-Party Components and Data

This repository contains original DiBS code and selected adapted components.
Large third-party datasets, pretrained checkpoints, caches, and experiment
outputs are intentionally not redistributed.

## Adapted Training Code

`model/diffusion-vs-ar/` is adapted from
[HKUNLP/diffusion-vs-ar](https://github.com/HKUNLP/diffusion-vs-ar) and keeps
its Apache License 2.0 text in `model/diffusion-vs-ar/LICENSE`.

## Optional Baselines

The following upstream repositories are not vendored in this release:

- [locuslab/SATNet](https://github.com/locuslab/SATNet)
- [Chrixtar/SRM](https://github.com/Chrixtar/SRM)
- [t-dillon/tdoku](https://github.com/t-dillon/tdoku)

Clone them separately when reproducing experiments that require those
baselines, and follow their respective licenses.

## Data

Small synthetic generalized Sudoku and N-Queens samples generated for this
project are included under `dataset/`. External datasets such as Kaggle Sudoku,
SATNet Sudoku, and Royle or tdoku collections must be obtained from their
original sources and processed locally with the scripts under `dataset/`.
