#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
MDM_ROOT="$ROOT_DIR/model/diffusion-vs-ar"
export WANDB_DISABLED=true
export WANDB_MODE=disabled
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
TASKS="${TASKS:-3sat5,3sat7,3sat9}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
PROCS="${PROCS:-8}"
PORT="${PORT:-20109}"
MODEL_NAME="${MODEL_NAME:-model_config_tiny}"
EPOCHS="${EPOCHS:-600}"
BSZ="${BSZ:-128}"
LR="${LR:-1e-3}"
DIFF_STEPS="${DIFF_STEPS:-20}"
ALPHA="${ALPHA:-0.25}"
GAMMA="${GAMMA:-1.0}"
VAL_SIZE="${VAL_SIZE:-448}"
EVAL_STEPS="${EVAL_STEPS:-100}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
SAVE_STEPS="${SAVE_STEPS:-500}"

OUT_BASE="$ROOT_DIR/model/diffusion-vs-ar/output/3sat_table5/$RUN_ID"
mkdir -p "$OUT_BASE"
pushd "$MDM_ROOT" >/dev/null

IFS=',' read -r -a TASK_ARR <<< "$TASKS"
for task in "${TASK_ARR[@]}"; do
  exp="$OUT_BASE/$task"
  mkdir -p "$exp"
  echo "[train-table5-mdm] task=$task out=$exp"
  LAUNCH_ARGS=()
  if [[ "${PROCS}" -ge 2 ]]; then
    LAUNCH_ARGS+=(--multi_gpu --num_machines 1 --mixed_precision fp16 --num_processes "$PROCS" --main_process_port "$PORT")
  else
    LAUNCH_ARGS+=(--num_processes 1)
  fi

  CUDA_VISIBLE_DEVICES="$GPUS" \
  accelerate launch "${LAUNCH_ARGS[@]}" \
    "./src/train_bash.py" \
    --stage mdm --overwrite_output_dir \
    --cache_dir "./cache" \
    --dataset_dir "./data" \
    --model_name_or_path "$MODEL_NAME" \
    --do_train \
    --dataset "${task}_train" \
    --finetuning_type full \
    --cutoff_len 325 \
    --output_dir "$exp" \
    --overwrite_cache \
    --per_device_train_batch_size "$BSZ" \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps "$LOGGING_STEPS" \
    --val_size "$VAL_SIZE" \
    --per_device_eval_batch_size 32 \
    --evaluation_strategy steps \
    --eval_steps "$EVAL_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --learning_rate "$LR" \
    --num_train_epochs "$EPOCHS" \
    --plot_loss \
    --run_name "table5_${task}_${RUN_ID}" \
    --report_to none \
    --preprocessing_num_workers 8 \
    --fp16 \
    --save_total_limit 2 \
    --remove_unused_columns False \
    --diffusion_steps "$DIFF_STEPS" \
    --save_safetensors False \
    --token_reweighting True \
    --time_reweighting linear \
    --topk_decoding True \
    --alpha "$ALPHA" \
    --gamma "$GAMMA" \
    > "$exp/train.log" 2>&1
done

python3 - <<PY
import json
from pathlib import Path
run_id = "${RUN_ID}"
base = Path("${OUT_BASE}")
registry = {"run_id": run_id, "tasks": {}, "created_at": __import__("time").strftime("%Y-%m-%d %H:%M:%S")}
for d in sorted(base.iterdir()):
    if d.is_dir():
        registry["tasks"][d.name] = {"checkpoint_dir": str(d), "train_log": str(d / "train.log")}
rp = base / "checkpoints_registry.json"
rp.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
print("Registry updated:", rp)
PY
popd >/dev/null
