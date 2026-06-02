# Table5 SAT Plan and Usage

## Goal and Boundary
- Goal: add a standalone Table5 pipeline for `3sat5/3sat7/3sat9` with `PySAT/CDCL` and `DiBS-CP`.
- Boundary: do not change behavior of existing Table1/2/3/4 runners or Sudoku `DiBS/solver.py`.

## New Files
- `DiBS/table5_sat_data.py`: 3SAT dataset parser and adapter.
- `DiBS/sat_solver_dibs.py`: DPLL-style DiBS-CP SAT solver.
- `DiBS/table5_experiment.py`: unified Table5 evaluator + exports.
- `model/diffusion-vs-ar/scripts/3-sat/train-table5-mdm.sh`: Table5-oriented MDM training launcher.
- `table5.sh`: one-command orchestration (train/eval/resume).

## Input/Output Convention
- Input datasets (read-only):
  - `model/diffusion-vs-ar/data/3sat5_{train,test}.jsonl`
  - `model/diffusion-vs-ar/data/3sat7_{train,test}.jsonl`
  - `model/diffusion-vs-ar/data/3sat9_{train,test}.jsonl`
- Output root:
  - `DiBS/results/parallel/Table_5/<RUN_ID>/`
- Output files:
  - `per_instance/<task>/<solver>.jsonl`
  - `summaries/<task>_<solver>_summary.json`
  - `<RUN_ID>_all_summaries.json`
  - `<RUN_ID>_table5.tex`
  - `<RUN_ID>_report.md`

## Run ID and Resume
- Run id is required/strongly recommended: `YYYYMMDD-HHMMSS` or custom.
- Resume behavior:
  - `--resume` skips already-recorded `instance_id` per solver/task jsonl.
  - Safe to rerun after interruption.

## One-command Usage
```bash
bash table5.sh --run-id 20260430-table5 --gpu 1
```

Skip train, eval only:
```bash
bash table5.sh --run-id 20260430-table5 --skip-train --resume --gpu 1
```

Direct evaluator:
```bash
python3 DiBS/table5_experiment.py \
  --run-id 20260430-table5 \
  --tasks 3sat5,3sat7,3sat9 \
  --solvers pysat,dibs_cp \
  --workers 16 \
  --timeout-ms 0 \
  --max-nodes 1000000 \
  --resume
```

## Regression Protection Smoke
```bash
python3 DiBS/run_table1_complete.py --max-puzzles 5
python3 DiBS/run_table3_experiments.py --max-puzzles 20 --experiment alpha
```
