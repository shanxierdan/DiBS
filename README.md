# DiBS 实验运行指南

本文档介绍如何运行 DiBS 论文实验。
稳定主链路（Table1/2/3）保持不变；Table4 已升级为统一扩展实验（广义数独 + N-Queens）。

## 开源范围

本仓库包含 DiBS 求解器、实验脚本、训练代码、数据构建器、可直接发布的小型合成数据和论文源码。

为控制仓库体积并遵守第三方数据许可，仓库不直接分发模型 checkpoint、运行日志、缓存、完整实验输出或第三方原始数据。运行依赖模型的实验前，请将 checkpoint 放入本地目录并通过 `--model` 或 `DIBS_CHECKPOINT_PATH` 指定路径。第三方代码与数据来源见 [`THIRD_PARTY.md`](THIRD_PARTY.md)。

## 目录结构

```text
sudoku/
├── DiBS/
│   ├── run_table1_complete.py
│   ├── run_table2_experiments.py
│   ├── run_table3_experiments.py
│   ├── run_table4_extension.py
│   ├── run_explore_1.py
│   ├── run_explore_2.py
│   └── solver.py
├── dataset/
│   ├── prepared_data/
│   ├── discover_external_data.py
│   ├── build_table4_extension_data.py
│   └── table4_extension/           # 构建后生成
├── model/diffusion-vs-ar/scripts/extension/
│   ├── train_gen_sudoku_mdm.sh
│   └── train_nqueens_mdm.sh
├── Table1.md
├── Table2.md
├── Table3.md
└── Table4_extension.md
```

## Table 1: Baseline 对比实验

```bash
cd DiBS
python3 DiBS/run_table1_complete.py --workers 16
```

输出目录：`DiBS/results/parallel/Table_1/`

## Table 2: 多数据集泛化对比

```bash
cd DiBS
python3 DiBS/run_table2_experiments.py --gpus "0,1,2,3" --solvers "MRV,DiBS"
```

输出目录：`DiBS/results/parallel/Table_2/`

## Table 3: 消融实验与调参

```bash
cd DiBS
python3 DiBS/run_table3_experiments.py --gpus "0,1,2,3"
```

输出目录：`DiBS/results/parallel/Table_3/`

## Table 4: 统一扩展实验（先数据，再训练，再对比）

默认覆盖任务：

- generalized Sudoku: `4x4`, `16x16`, `25x25`
- N-Queens: `8`, `9`, `10`

### 1) 外部数据发现（严格准入）

```bash
cd DiBS
python3 dataset/discover_external_data.py
```

### 2) 构建统一 train/test 数据

```bash
cd DiBS
python3 dataset/build_table4_extension_data.py \
  --train-count 5000 --test-count 500 --seed 42 \
  --max-generation-attempts 1000000
```

### 3) 训练任务族模型

```bash
cd DiBS
GPUS=0,1,2,3,4,5,6,7 EPOCHS=12 BATCH_SIZE=64 \
  bash model/diffusion-vs-ar/scripts/extension/train_gen_sudoku_mdm.sh
GPUS=0,1,2,3,4,5,6,7 EPOCHS=12 BATCH_SIZE=64 \
  bash model/diffusion-vs-ar/scripts/extension/train_nqueens_mdm.sh
```

### 4) 统一评测（MRV+FC vs DiBS-full）

```bash
cd DiBS
python3 DiBS/run_table4_extension.py --tasks all --model-device cuda
```

### 5) 一键全链路（推荐）

```bash
cd DiBS
GEN_SUDOKU_GPUS=0,1,2,3,4,5,6,7 \
NQUEENS_GPUS=0,1,2,3,4,5,6,7 \
MODEL_DEVICE=cuda \
bash run_table4_extension_pipeline.sh
```

默认使用 run-centric 目录管理：
- `experiments/table4_runs/<RUN_ID>/data`
- `experiments/table4_runs/<RUN_ID>/models`
- `experiments/table4_runs/<RUN_ID>/eval`
- `experiments/table4_runs/latest`（软链接）

### 6) Full-scale 模式（更大规模 + 更难任务）

```bash
cd DiBS
bash run_table4_fullscale_pipeline.sh
```

默认 full-scale 任务：
- generalized Sudoku: `16x16,25x25`
- N-Queens: `12,14,16`

如需更强调高难样本训练（提升 unknown 准确率）可加：

```bash
GEN_SUDOKU_EPOCHS=60 \
GEN_SUDOKU_HARD_WEIGHT=4.0 \
GEN_SUDOKU_MEDIUM_WEIGHT=2.0 \
GEN_SUDOKU_UNKNOWN_RATIO_WEIGHT=0.8 \
NQUEENS_EPOCHS=40 \
NQUEENS_HARD_WEIGHT=4.0 \
bash run_table4_extension_pipeline.sh
```

### 7) 难度分层报告

```bash
cd DiBS
python3 DiBS/report_table4_by_difficulty.py --run-id <RUN_ID>
```

输出目录：`experiments/table4_runs/<RUN_ID>/eval/`（full-scale 对应 `experiments/table4_fullscale_runs/<RUN_ID>/eval/`）
- `<RUN_ID>_difficulty_report.{json,md}`
- `<RUN_ID>_givens_ratio_gain_report.{json,md}`（DiBS 相对 MRV+FC 的按已填格比例增益）

## 快速 smoke test

```bash
cd DiBS
python3 dataset/discover_external_data.py
python3 dataset/build_table4_extension_data.py --train-count 100 --test-count 20 --seed 42
bash model/diffusion-vs-ar/scripts/extension/train_gen_sudoku_mdm.sh
bash model/diffusion-vs-ar/scripts/extension/train_nqueens_mdm.sh
python3 DiBS/run_table4_extension.py --max-puzzles 20
```

## 说明

- Table1/2/3 与 `DiBS/solver.py` 主链路未改动。
- Table4 统一扩展实验默认中等规模数据，外部数据不足时自动生成补齐，默认严格去重。
- `dataset/build_table4_extension_data.py` 支持 `--sudoku-sizes` 和 `--nqueens-sizes`，并写入 `difficulty` 标签。

## Exploratory Experiments

探索实验输出统一写入：`experiments/explore_runs/<RUN_ID>/`

### 资源检查（CPU/GPU）

```bash
cd DiBS
nproc
uptime
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
```

### Explore 1：难度-增益相关性（自动可行难度桶）

```bash
cd DiBS
python3 DiBS/run_explore_1.py \
  --run-id 20260417-exp1 \
  --model <LOCAL_CHECKPOINT_DIR> \
  --per-bucket 500 \
  --bucket-count 10 \
  --min-givens 17 \
  --gpu 1 \
  --workers-baseline 32 \
  --workers-dibs 1
```

### Explore 2：多步去噪调用（独立于 Explore 1，DiBS-only）

```bash
cd DiBS
python3 DiBS/run_explore_2.py \
  --run-id 20260417-exp2 \
  --dataset-csv <LOCAL_DATASET_CSV> \
  --max-puzzles 5000 \
  --steps 1,2,4,8 \
  --denoise-strategy mdm_iterative \
  --mdm-decoding-strategy deterministic-cosine \
  --gpu 1 \
  --workers-dibs 1
```

### 断点续跑

```bash
python3 DiBS/run_explore_1.py --run-id 20260417-exp1 --resume
python3 DiBS/run_explore_2.py --run-id 20260417-exp2 --resume
```

### 输出结构

- `explore_1/data/prepared_merged.jsonl`
- `explore_1/data/prepared_merged_stats.json`
- `explore_1/data/sampled_puzzles.jsonl`
- `explore_1/results/per_instance/<solver>/givens_<g>.jsonl`
- `explore_1/results/explore_1_global_summary.json`
- `explore_1/reports/explore_1_report.md`
- `explore_2/results/per_instance/DiBS_step_<k>/all.jsonl`
- `explore_2/results/explore_2_global_summary.json`
- `explore_2/reports/explore_2_report.md`

### GPU 建议

- 探索实验优先使用空闲卡（你当前机器通常 `GPU1` 更空闲）。
- `workers-dibs` 建议先从 `1` 开始，避免多进程模型占用导致显存/吞吐抖动。
