# Table 4 Extension: Unified Pipeline (Generalized Sudoku + N-Queens)

## Goal

One end-to-end pipeline:

1. external source discovery and strict admission
2. train/test dataset build
3. family model training
4. DiBS-full vs MRV+FC evaluation
5. LaTeX table export

## Tasks

- generalized Sudoku: `4x4`, `16x16`, `25x25`
- N-Queens: `8`, `9`, `10`

## Default Data Scale

- per size: `train=5000`, `test=500`
- if accepted external data is insufficient, synthetic generation fills the gap
- default is strict unique generation (`--allow-duplicates` disabled)

## Commands

### 1) Discover external sources

```bash
cd DiBS
python3 dataset/discover_external_data.py
```

Outputs:

- `dataset/accepted_sources.json`
- `dataset/rejected_sources.json`

### 2) Build unified datasets

```bash
cd DiBS
python3 dataset/build_table4_extension_data.py \
  --train-count 5000 --test-count 500 --seed 42 \
  --max-generation-attempts 1000000
```

Output root:

- `dataset/table4_extension/`

### 3) Train task-family models

```bash
cd DiBS
GPUS=0,1,2,3,4,5,6,7 EPOCHS=12 BATCH_SIZE=64 \
  bash model/diffusion-vs-ar/scripts/extension/train_gen_sudoku_mdm.sh
GPUS=0,1,2,3,4,5,6,7 EPOCHS=12 BATCH_SIZE=64 \
  bash model/diffusion-vs-ar/scripts/extension/train_nqueens_mdm.sh
```

Output:

- `model/diffusion-vs-ar/output/extension/checkpoints_registry.json`

### 4) Run unified evaluation

```bash
cd DiBS
python3 DiBS/run_table4_extension.py --tasks all --model-device cuda
```

Outputs:

- per-instance jsonl
- per-task summary json
- `*_all_summaries.json`
- `*_table4_extension.tex`
- `*_meta.json`

### 5) One-click full pipeline

```bash
cd DiBS
GEN_SUDOKU_GPUS=0,1,2,3,4,5,6,7 \
NQUEENS_GPUS=0,1,2,3,4,5,6,7 \
MODEL_DEVICE=cuda \
bash run_table4_extension_pipeline.sh
```

## Notes

- Extension models are now neural priors trained from extension datasets and loaded from `.pt` checkpoints.
- Table1/2/3 scripts and main `DiBS/solver.py` workflow remain unchanged.
