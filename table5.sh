#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
TASKS="${TASKS:-3sat5,3sat7,3sat9}"
SOLVERS="${SOLVERS:-pysat,cp_first_pos,cp_mrv_pos,cp_mrv_jw,dibs_mdm}"
TIMEOUT_MS="${TIMEOUT_MS:-0}"
MAX_NODES="${MAX_NODES:-1000000}"
MRV_THRESHOLD="${MRV_THRESHOLD:-2}"
SMART_CALL="${SMART_CALL:-1}"
JW_MARGIN_THRESHOLD="${JW_MARGIN_THRESHOLD:-0.25}"
MAX_INSTANCES="${MAX_INSTANCES:-0}"
GPU="${GPU:-0}"
WORKERS="${WORKERS:-1}"
SAT_MODEL_REGISTRY="${SAT_MODEL_REGISTRY:-$ROOT/model/diffusion-vs-ar/output/3sat_table5/best_models_registry.json}"
SKIP_TRAIN=0
SKIP_EVAL=0
RESUME=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --tasks) TASKS="$2"; shift 2 ;;
    --solvers) SOLVERS="$2"; shift 2 ;;
    --timeout-ms) TIMEOUT_MS="$2"; shift 2 ;;
    --max-nodes) MAX_NODES="$2"; shift 2 ;;
    --mrv-threshold) MRV_THRESHOLD="$2"; shift 2 ;;
    --smart-call) SMART_CALL="$2"; shift 2 ;;
    --jw-margin-threshold) JW_MARGIN_THRESHOLD="$2"; shift 2 ;;
    --max-instances) MAX_INSTANCES="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --sat-model-registry) SAT_MODEL_REGISTRY="$2"; shift 2 ;;
    --skip-train) SKIP_TRAIN=1; shift ;;
    --skip-eval) SKIP_EVAL=1; shift ;;
    --resume) RESUME=1; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "============================================================"
echo "Table5 SAT Unified Pipeline"
echo "============================================================"
echo "RUN_ID         : $RUN_ID"
echo "TASKS          : $TASKS"
echo "SOLVERS        : $SOLVERS"
echo "TIMEOUT_MS     : $TIMEOUT_MS"
echo "MAX_NODES      : $MAX_NODES"
echo "MRV_THRESHOLD  : $MRV_THRESHOLD"
echo "SMART_CALL     : $SMART_CALL"
echo "JW_MARGIN      : $JW_MARGIN_THRESHOLD"
echo "MAX_INSTANCES  : $MAX_INSTANCES"
echo "GPU            : $GPU"
echo "WORKERS        : $WORKERS"
echo "MODEL_REGISTRY : $SAT_MODEL_REGISTRY"
echo "============================================================"

if [[ "$SKIP_TRAIN" -eq 0 ]]; then
  echo "[1/2] Train Table5 MDM models"
  RUN_ID="$RUN_ID" TASKS="$TASKS" GPUS="$GPU" PROCS=1 \
    bash "$ROOT/model/diffusion-vs-ar/scripts/3-sat/train-table5-mdm.sh"
else
  echo "[1/2] Train skipped"
fi

if [[ "$SKIP_EVAL" -eq 0 ]]; then
  echo "[2/2] Run Table5 evaluation"
  CMD=(python3 "$ROOT/DiBS/table5_experiment.py"
    --run-id "$RUN_ID"
    --tasks "$TASKS"
    --solvers "$SOLVERS"
    --timeout-ms "$TIMEOUT_MS"
    --max-nodes "$MAX_NODES"
    --mrv-threshold "$MRV_THRESHOLD"
    --smart-call "$SMART_CALL"
    --jw-margin-threshold "$JW_MARGIN_THRESHOLD"
    --max-instances "$MAX_INSTANCES"
    --workers "$WORKERS"
    --device "cuda:$GPU"
    --sat-model-registry "$SAT_MODEL_REGISTRY")
  if [[ "$RESUME" -eq 1 ]]; then
    CMD+=(--resume)
  fi
  "${CMD[@]}"
else
  echo "[2/2] Eval skipped"
fi

echo "Pipeline done."
echo "Run dir: $ROOT/DiBS/results/parallel/Table_5/$RUN_ID"
