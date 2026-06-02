#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -----------------------------
# Config (override via env vars)
# -----------------------------
TRAIN_COUNT="${TRAIN_COUNT:-1500}"
TEST_COUNT="${TEST_COUNT:-500}"
SEED="${SEED:-42}"

TASKS="${TASKS:-all}"                  # all or "generalized_sudoku:4x4,nqueens:8"
MAX_PUZZLES="${MAX_PUZZLES:-}"         # empty => full test split
TIMEOUT_MS="${TIMEOUT_MS:-60000}"
MAX_NODES="${MAX_NODES:-1000000}"
ALPHA="${ALPHA:-0.8}"
SMART_INTERVAL="${SMART_INTERVAL:-5}"

GEN_SUDOKU_GPUS="${GEN_SUDOKU_GPUS:-auto}"
NQUEENS_GPUS="${NQUEENS_GPUS:-auto}"
SUDOKU_SIZES="${SUDOKU_SIZES:-4x4,16x16,25x25}"
NQUEENS_SIZES="${NQUEENS_SIZES:-8,9,10}"
GEN_SUDOKU_SIZES="${GEN_SUDOKU_SIZES:-$SUDOKU_SIZES}"
GEN_SUDOKU_EPOCHS="${GEN_SUDOKU_EPOCHS:-12}"
NQUEENS_EPOCHS="${NQUEENS_EPOCHS:-12}"
GEN_SUDOKU_BATCH_SIZE="${GEN_SUDOKU_BATCH_SIZE:-64}"
NQUEENS_BATCH_SIZE="${NQUEENS_BATCH_SIZE:-64}"
GEN_SUDOKU_LR="${GEN_SUDOKU_LR:-3e-4}"
NQUEENS_LR="${NQUEENS_LR:-3e-4}"
GEN_SUDOKU_WORKERS="${GEN_SUDOKU_WORKERS:-4}"
NQUEENS_WORKERS="${NQUEENS_WORKERS:-4}"
GEN_SUDOKU_EASY_WEIGHT="${GEN_SUDOKU_EASY_WEIGHT:-1.0}"
GEN_SUDOKU_MEDIUM_WEIGHT="${GEN_SUDOKU_MEDIUM_WEIGHT:-1.5}"
GEN_SUDOKU_HARD_WEIGHT="${GEN_SUDOKU_HARD_WEIGHT:-3.0}"
GEN_SUDOKU_UNKNOWN_RATIO_WEIGHT="${GEN_SUDOKU_UNKNOWN_RATIO_WEIGHT:-0.5}"
NQUEENS_EASY_WEIGHT="${NQUEENS_EASY_WEIGHT:-1.0}"
NQUEENS_MEDIUM_WEIGHT="${NQUEENS_MEDIUM_WEIGHT:-1.5}"
NQUEENS_HARD_WEIGHT="${NQUEENS_HARD_WEIGHT:-3.0}"
NQUEENS_UNKNOWN_RATIO_WEIGHT="${NQUEENS_UNKNOWN_RATIO_WEIGHT:-0.5}"
GEN_SUDOKU_DISABLE_WEIGHTED_SAMPLING="${GEN_SUDOKU_DISABLE_WEIGHTED_SAMPLING:-0}"
NQUEENS_DISABLE_WEIGHTED_SAMPLING="${NQUEENS_DISABLE_WEIGHTED_SAMPLING:-0}"
GEN_SUDOKU_EVAL_SPLIT="${GEN_SUDOKU_EVAL_SPLIT:-test}"
NQUEENS_EVAL_SPLIT="${NQUEENS_EVAL_SPLIT:-test}"
GEN_SUDOKU_EVAL_BATCH_SIZE="${GEN_SUDOKU_EVAL_BATCH_SIZE:-128}"
NQUEENS_EVAL_BATCH_SIZE="${NQUEENS_EVAL_BATCH_SIZE:-128}"
GEN_SUDOKU_EVAL_EVERY="${GEN_SUDOKU_EVAL_EVERY:-1}"
NQUEENS_EVAL_EVERY="${NQUEENS_EVAL_EVERY:-1}"
GEN_SUDOKU_HIDDEN_SIZE="${GEN_SUDOKU_HIDDEN_SIZE:-512}"
NQUEENS_HIDDEN_SIZE="${NQUEENS_HIDDEN_SIZE:-512}"
GEN_SUDOKU_NUM_LAYERS="${GEN_SUDOKU_NUM_LAYERS:-6}"
NQUEENS_NUM_LAYERS="${NQUEENS_NUM_LAYERS:-6}"
GEN_SUDOKU_NUM_HEADS="${GEN_SUDOKU_NUM_HEADS:-8}"
NQUEENS_NUM_HEADS="${NQUEENS_NUM_HEADS:-8}"
GEN_SUDOKU_DROPOUT="${GEN_SUDOKU_DROPOUT:-0.1}"
NQUEENS_DROPOUT="${NQUEENS_DROPOUT:-0.1}"
GEN_SUDOKU_DIFFUSION_STEPS="${GEN_SUDOKU_DIFFUSION_STEPS:-20}"
NQUEENS_DIFFUSION_STEPS="${NQUEENS_DIFFUSION_STEPS:-20}"
GEN_SUDOKU_TOKEN_REWEIGHTING="${GEN_SUDOKU_TOKEN_REWEIGHTING:-1}"
NQUEENS_TOKEN_REWEIGHTING="${NQUEENS_TOKEN_REWEIGHTING:-1}"
GEN_SUDOKU_LOSS_ALPHA="${GEN_SUDOKU_LOSS_ALPHA:-0.25}"
NQUEENS_LOSS_ALPHA="${NQUEENS_LOSS_ALPHA:-0.25}"
GEN_SUDOKU_LOSS_GAMMA="${GEN_SUDOKU_LOSS_GAMMA:-1.0}"
NQUEENS_LOSS_GAMMA="${NQUEENS_LOSS_GAMMA:-1.0}"
GEN_SUDOKU_TIME_REWEIGHTING="${GEN_SUDOKU_TIME_REWEIGHTING:-linear}"
NQUEENS_TIME_REWEIGHTING="${NQUEENS_TIME_REWEIGHTING:-linear}"
GEN_SUDOKU_LR_SCHEDULER="${GEN_SUDOKU_LR_SCHEDULER:-cosine}"
NQUEENS_LR_SCHEDULER="${NQUEENS_LR_SCHEDULER:-cosine}"
GEN_SUDOKU_WARMUP_RATIO="${GEN_SUDOKU_WARMUP_RATIO:-0.03}"
NQUEENS_WARMUP_RATIO="${NQUEENS_WARMUP_RATIO:-0.03}"
GEN_SUDOKU_GRAD_ACC_STEPS="${GEN_SUDOKU_GRAD_ACC_STEPS:-1}"
NQUEENS_GRAD_ACC_STEPS="${NQUEENS_GRAD_ACC_STEPS:-1}"
GEN_SUDOKU_MAX_GRAD_NORM="${GEN_SUDOKU_MAX_GRAD_NORM:-1.0}"
NQUEENS_MAX_GRAD_NORM="${NQUEENS_MAX_GRAD_NORM:-1.0}"
MAX_GEN_ATTEMPTS="${MAX_GEN_ATTEMPTS:-1000000}"
ALLOW_DUPLICATES="${ALLOW_DUPLICATES:-0}"
UNIQUE_CHECK="${UNIQUE_CHECK:-1}"
PER_RECORD_MAX_ATTEMPTS="${PER_RECORD_MAX_ATTEMPTS:-200}"
UNIQUENESS_NODES="${UNIQUENESS_NODES:-300000}"
UNIQUENESS_TIMEOUT_SEC="${UNIQUENESS_TIMEOUT_SEC:-1.0}"
DATA_PROGRESS_EVERY="${DATA_PROGRESS_EVERY:-500}"
DIFFICULTY_STRATIFICATION="${DIFFICULTY_STRATIFICATION:-0}"
GENERATION_DIFFICULTY="${GENERATION_DIFFICULTY:-easy}"
MODEL_DEVICE="${MODEL_DEVICE:-auto}"
WORKERS="${WORKERS:-1}"
GPUS="${GPUS:-}"
RESUME="${RESUME:-1}"
RUN_ID="${RUN_ID:-}"
RUNS_ROOT="${RUNS_ROOT:-$ROOT/experiments/table4_runs}"
FORCE_REBUILD_DATA="${FORCE_REBUILD_DATA:-0}"
SKIP_DATA_BUILD="${SKIP_DATA_BUILD:-0}"
SKIP_EXTERNAL_DISCOVER="${SKIP_EXTERNAL_DISCOVER:-1}"
STRICT_REUSE_CHECKS="${STRICT_REUSE_CHECKS:-1}"
REQUIRE_RUN_ID_FOR_RESUME="${REQUIRE_RUN_ID_FOR_RESUME:-1}"
GEN_SUDOKU_MIN_BEST_METRIC="${GEN_SUDOKU_MIN_BEST_METRIC:-}"
NQUEENS_MIN_BEST_METRIC="${NQUEENS_MIN_BEST_METRIC:-}"
FORCE_RETRAIN_GEN_SUDOKU="${FORCE_RETRAIN_GEN_SUDOKU:-0}"
FORCE_RETRAIN_NQUEENS="${FORCE_RETRAIN_NQUEENS:-0}"
GENERATION_WORKERS="${GENERATION_WORKERS:-1}"
TRAIN_UNIQUE_CHECK="${TRAIN_UNIQUE_CHECK:-1}"
TEST_UNIQUE_CHECK="${TEST_UNIQUE_CHECK:-1}"

mkdir -p "$RUNS_ROOT"

resolve_latest_run_id() {
  python3 - "$RUNS_ROOT" <<'PY'
from pathlib import Path
root = Path(__import__("sys").argv[1])
dirs = []
for p in root.iterdir():
    if p.name == "latest":
        continue
    try:
        if p.is_symlink():
            continue
        if not p.is_dir():
            continue
        dirs.append(p)
    except OSError:
        continue
dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
print(dirs[0].name if dirs else "")
PY
}

if [[ -z "$RUN_ID" ]]; then
  if [[ "$RESUME" == "1" ]]; then
    if [[ "$REQUIRE_RUN_ID_FOR_RESUME" == "1" ]]; then
      echo "ERROR: RESUME=1 requires explicit RUN_ID (timestamp)."
      echo "Example: RUN_ID=20260407-153000 RESUME=1 bash run_table4_extension_pipeline.sh"
      exit 1
    fi
    RUN_ID="$(resolve_latest_run_id)"
  fi
  if [[ -z "$RUN_ID" ]]; then
    RUN_ID="$(date +%Y%m%d-%H%M%S)"
  fi
fi
if [[ "$RUN_ID" == "latest" ]]; then
  RUN_ID="$(date +%Y%m%d-%H%M%S)"
fi

RUN_DIR="$RUNS_ROOT/$RUN_ID"
DATA_ROOT="$RUN_DIR/data"
MODEL_ROOT="$RUN_DIR/models"
EVAL_ROOT="$RUN_DIR/eval"
REGISTRY="$MODEL_ROOT/checkpoints_registry.json"
MANIFEST_DIR="$RUN_DIR/manifests"
STATUS_PATH="$RUN_DIR/status.json"
CONFIG_PATH="$RUN_DIR/config.json"
INDEX_PATH="$RUNS_ROOT/runs_index.jsonl"
LATEST_LINK="$RUNS_ROOT/latest"

mkdir -p "$RUN_DIR" "$DATA_ROOT" "$MODEL_ROOT" "$EVAL_ROOT" "$MANIFEST_DIR"

write_config() {
  python3 - "$CONFIG_PATH" <<'PY'
import json, os, time
path = __import__("sys").argv[1]
keys = [
  "RUN_ID","RUNS_ROOT","ROOT","TRAIN_COUNT","TEST_COUNT","SEED","TASKS","MAX_PUZZLES",
  "TIMEOUT_MS","MAX_NODES","ALPHA","SMART_INTERVAL","MODEL_DEVICE","WORKERS","GPUS",
  "SUDOKU_SIZES","NQUEENS_SIZES","GEN_SUDOKU_SIZES","GEN_SUDOKU_GPUS","NQUEENS_GPUS",
  "GEN_SUDOKU_EPOCHS","NQUEENS_EPOCHS","GEN_SUDOKU_BATCH_SIZE","NQUEENS_BATCH_SIZE",
  "GEN_SUDOKU_LR","NQUEENS_LR","GEN_SUDOKU_WORKERS","NQUEENS_WORKERS",
  "GEN_SUDOKU_EASY_WEIGHT","GEN_SUDOKU_MEDIUM_WEIGHT","GEN_SUDOKU_HARD_WEIGHT","GEN_SUDOKU_UNKNOWN_RATIO_WEIGHT",
  "NQUEENS_EASY_WEIGHT","NQUEENS_MEDIUM_WEIGHT","NQUEENS_HARD_WEIGHT","NQUEENS_UNKNOWN_RATIO_WEIGHT",
  "GEN_SUDOKU_DISABLE_WEIGHTED_SAMPLING","NQUEENS_DISABLE_WEIGHTED_SAMPLING",
  "GEN_SUDOKU_EVAL_SPLIT","NQUEENS_EVAL_SPLIT","GEN_SUDOKU_EVAL_BATCH_SIZE","NQUEENS_EVAL_BATCH_SIZE",
  "GEN_SUDOKU_EVAL_EVERY","NQUEENS_EVAL_EVERY",
  "GEN_SUDOKU_HIDDEN_SIZE","NQUEENS_HIDDEN_SIZE","GEN_SUDOKU_NUM_LAYERS","NQUEENS_NUM_LAYERS",
  "GEN_SUDOKU_NUM_HEADS","NQUEENS_NUM_HEADS","GEN_SUDOKU_DROPOUT","NQUEENS_DROPOUT",
  "GEN_SUDOKU_DIFFUSION_STEPS","NQUEENS_DIFFUSION_STEPS","GEN_SUDOKU_TOKEN_REWEIGHTING","NQUEENS_TOKEN_REWEIGHTING",
  "GEN_SUDOKU_LOSS_ALPHA","NQUEENS_LOSS_ALPHA","GEN_SUDOKU_LOSS_GAMMA","NQUEENS_LOSS_GAMMA",
  "GEN_SUDOKU_TIME_REWEIGHTING","NQUEENS_TIME_REWEIGHTING",
  "GEN_SUDOKU_LR_SCHEDULER","NQUEENS_LR_SCHEDULER","GEN_SUDOKU_WARMUP_RATIO","NQUEENS_WARMUP_RATIO",
  "GEN_SUDOKU_GRAD_ACC_STEPS","NQUEENS_GRAD_ACC_STEPS","GEN_SUDOKU_MAX_GRAD_NORM","NQUEENS_MAX_GRAD_NORM",
  "MAX_GEN_ATTEMPTS","ALLOW_DUPLICATES","RESUME","REQUIRE_RUN_ID_FOR_RESUME","FORCE_REBUILD_DATA","SKIP_DATA_BUILD","SKIP_EXTERNAL_DISCOVER","STRICT_REUSE_CHECKS",
  "GENERATION_WORKERS","TRAIN_UNIQUE_CHECK","TEST_UNIQUE_CHECK",
  "GEN_SUDOKU_MIN_BEST_METRIC","NQUEENS_MIN_BEST_METRIC",
  "UNIQUE_CHECK","PER_RECORD_MAX_ATTEMPTS","UNIQUENESS_NODES","UNIQUENESS_TIMEOUT_SEC",
  "DATA_PROGRESS_EVERY","DIFFICULTY_STRATIFICATION","GENERATION_DIFFICULTY",
  "FORCE_RETRAIN_GEN_SUDOKU","FORCE_RETRAIN_NQUEENS",
  "DATA_ROOT","MODEL_ROOT","EVAL_ROOT","REGISTRY"
]
d = {k: os.environ.get(k) for k in keys}
d["written_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
with open(path, "w", encoding="utf-8") as f:
  json.dump(d, f, indent=2, ensure_ascii=False)
PY
}

write_status() {
  local phase="$1"
  local state="$2"
  python3 - "$STATUS_PATH" "$phase" "$state" <<'PY'
import json, time, sys
from pathlib import Path
path = Path(sys.argv[1]); phase=sys.argv[2]; state=sys.argv[3]
if path.exists():
  d = json.loads(path.read_text(encoding="utf-8"))
else:
  d = {"run_id": path.parent.name, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "phases": {}}
d["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
d["phases"][phase] = state
path.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
PY
}

append_index() {
  python3 - "$INDEX_PATH" "$RUN_ID" "$RUN_DIR" "$CONFIG_PATH" <<'PY'
import json, time, sys
idx, run_id, run_dir, cfg = sys.argv[1:]
row = {
  "run_id": run_id,
  "run_dir": run_dir,
  "config": cfg,
  "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
}
with open(idx, "a", encoding="utf-8") as f:
  f.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
}

update_latest_link() {
  rm -f "$LATEST_LINK"
  ln -s "$(realpath "$RUN_DIR")" "$LATEST_LINK"
}

write_config
if [[ ! -f "$RUN_DIR/.indexed" ]]; then
  append_index
  touch "$RUN_DIR/.indexed"
fi
update_latest_link

echo "============================================================"
echo "Table4 Unified Run-Centric Pipeline"
echo "============================================================"
echo "RUN_ID            : $RUN_ID"
echo "RUN_DIR           : $RUN_DIR"
echo "TRAIN_COUNT       : $TRAIN_COUNT"
echo "TEST_COUNT        : $TEST_COUNT"
echo "SEED              : $SEED"
echo "TASKS             : $TASKS"
echo "MAX_PUZZLES       : ${MAX_PUZZLES:-<full>}"
echo "TIMEOUT_MS        : $TIMEOUT_MS"
echo "MAX_NODES         : $MAX_NODES"
echo "ALPHA             : $ALPHA"
echo "SMART_INTERVAL    : $SMART_INTERVAL"
echo "MODEL_DEVICE      : $MODEL_DEVICE"
echo "WORKERS           : $WORKERS"
echo "GPUS              : ${GPUS:-<none>}"
echo "RESUME            : $RESUME"
echo "MAX_GEN_ATTEMPTS  : $MAX_GEN_ATTEMPTS"
echo "ALLOW_DUPLICATES  : $ALLOW_DUPLICATES"
echo "UNIQUE_CHECK      : $UNIQUE_CHECK"
echo "TRAIN_UNIQUE      : $TRAIN_UNIQUE_CHECK"
echo "TEST_UNIQUE       : $TEST_UNIQUE_CHECK"
echo "PER_RECORD_ATTEMPTS: $PER_RECORD_MAX_ATTEMPTS"
echo "UNIQUENESS_NODES  : $UNIQUENESS_NODES"
echo "UNIQUENESS_TIMEOUT: $UNIQUENESS_TIMEOUT_SEC"
echo "GEN_WORKERS       : $GENERATION_WORKERS"
echo "DATA_PROGRESS     : every $DATA_PROGRESS_EVERY generated"
echo "DIFF_STRATIFIED   : $DIFFICULTY_STRATIFICATION"
echo "GEN_DIFFICULTY    : $GENERATION_DIFFICULTY"
echo "SKIP_DATA_BUILD   : $SKIP_DATA_BUILD"
echo "SKIP_DISCOVER     : $SKIP_EXTERNAL_DISCOVER"
echo "STRICT_REUSE      : $STRICT_REUSE_CHECKS"
echo "MIN_BEST_METRIC   : gen_sudoku=${GEN_SUDOKU_MIN_BEST_METRIC:-<none>} nqueens=${NQUEENS_MIN_BEST_METRIC:-<none>}"
echo "SUDOKU_SIZES      : $SUDOKU_SIZES"
echo "NQUEENS_SIZES     : $NQUEENS_SIZES"
echo "GEN weights       : e=$GEN_SUDOKU_EASY_WEIGHT m=$GEN_SUDOKU_MEDIUM_WEIGHT h=$GEN_SUDOKU_HARD_WEIGHT unk=$GEN_SUDOKU_UNKNOWN_RATIO_WEIGHT"
echo "NQ  weights       : e=$NQUEENS_EASY_WEIGHT m=$NQUEENS_MEDIUM_WEIGHT h=$NQUEENS_HARD_WEIGHT unk=$NQUEENS_UNKNOWN_RATIO_WEIGHT"
echo "GEN model/mdm     : h=$GEN_SUDOKU_HIDDEN_SIZE L=$GEN_SUDOKU_NUM_LAYERS H=$GEN_SUDOKU_NUM_HEADS drop=$GEN_SUDOKU_DROPOUT T=$GEN_SUDOKU_DIFFUSION_STEPS rw=$GEN_SUDOKU_TOKEN_REWEIGHTING"
echo "NQ  model/mdm     : h=$NQUEENS_HIDDEN_SIZE L=$NQUEENS_NUM_LAYERS H=$NQUEENS_NUM_HEADS drop=$NQUEENS_DROPOUT T=$NQUEENS_DIFFUSION_STEPS rw=$NQUEENS_TOKEN_REWEIGHTING"
echo "GEN optim         : scheduler=$GEN_SUDOKU_LR_SCHEDULER warmup=$GEN_SUDOKU_WARMUP_RATIO grad_acc=$GEN_SUDOKU_GRAD_ACC_STEPS max_grad_norm=$GEN_SUDOKU_MAX_GRAD_NORM"
echo "NQ  optim         : scheduler=$NQUEENS_LR_SCHEDULER warmup=$NQUEENS_WARMUP_RATIO grad_acc=$NQUEENS_GRAD_ACC_STEPS max_grad_norm=$NQUEENS_MAX_GRAD_NORM"
echo "DATA_ROOT         : $DATA_ROOT"
echo "MODEL_ROOT        : $MODEL_ROOT"
echo "EVAL_ROOT         : $EVAL_ROOT"
echo "REGISTRY          : $REGISTRY"
echo "============================================================"

cd "$ROOT"

is_dataset_ready() {
  python3 - "$DATA_ROOT" "$TRAIN_COUNT" "$TEST_COUNT" "$SUDOKU_SIZES" "$NQUEENS_SIZES" <<'PY'
import sys
from pathlib import Path
data_root=Path(sys.argv[1]); train_n=int(sys.argv[2]); test_n=int(sys.argv[3])
sudoku_sizes=[s.strip() for s in sys.argv[4].split(",") if s.strip()]
nqueens_sizes=[s.strip() for s in sys.argv[5].split(",") if s.strip()]
tasks=[("generalized_sudoku",s) for s in sudoku_sizes] + [("nqueens",s) for s in nqueens_sizes]
for fam,size in tasks:
  for split,need in (("train",train_n),("test",test_n)):
    p=data_root/fam/size/f"{split}.jsonl"
    if not p.exists():
      print(f"missing:{p}")
      sys.exit(1)
    cnt=sum(1 for _ in p.open("r",encoding="utf-8"))
    if cnt!=need:
      print(f"count_mismatch:{p}:{cnt}!={need}")
      sys.exit(1)
print("ready")
PY
}

is_dataset_compatible() {
  python3 - "$DATA_ROOT" "$TRAIN_COUNT" "$TEST_COUNT" "$SEED" "$SUDOKU_SIZES" "$NQUEENS_SIZES" "$UNIQUE_CHECK" "$TRAIN_UNIQUE_CHECK" "$TEST_UNIQUE_CHECK" "$PER_RECORD_MAX_ATTEMPTS" "$UNIQUENESS_NODES" "$UNIQUENESS_TIMEOUT_SEC" "$STRICT_REUSE_CHECKS" "$DIFFICULTY_STRATIFICATION" "$GENERATION_DIFFICULTY" <<'PY'
import json,sys
from pathlib import Path

data_root=Path(sys.argv[1])
train_n=int(sys.argv[2]); test_n=int(sys.argv[3]); seed=int(sys.argv[4])
sudoku_sizes=sorted([s.strip() for s in sys.argv[5].split(",") if s.strip()])
nqueens_sizes=sorted([s.strip() for s in sys.argv[6].split(",") if s.strip()])
unique_check_global=(sys.argv[7]!="0")
train_unique=(sys.argv[8]!="0") and unique_check_global
test_unique=(sys.argv[9]!="0") and unique_check_global
per_record_max_attempts=int(sys.argv[10])
uniqueness_nodes=int(sys.argv[11])
uniqueness_timeout=float(sys.argv[12])
strict=(sys.argv[13]!="0")
diff_stratified=(sys.argv[14]!="0")
generation_difficulty=sys.argv[15].strip().lower()
expected_difficulty_schema="givens_equal_width_v2" if diff_stratified else "single_level_v1"
expected_difficulty_levels=["very_hard","hard","medium","easy","very_easy"]

if not strict:
    print("compatible (strict off)")
    sys.exit(0)

meta_path=data_root/"meta.json"
if not meta_path.exists():
    print(f"incompatible: missing {meta_path}")
    sys.exit(1)
meta=json.loads(meta_path.read_text(encoding="utf-8"))

def mismatch(msg):
    print(f"incompatible: {msg}")
    sys.exit(1)

if int(meta.get("train_count",-1)) != train_n:
    mismatch(f"train_count {meta.get('train_count')} != {train_n}")
if int(meta.get("test_count",-1)) != test_n:
    mismatch(f"test_count {meta.get('test_count')} != {test_n}")
if int(meta.get("seed",-1)) != seed:
    mismatch(f"seed {meta.get('seed')} != {seed}")
if sorted(meta.get("sudoku_sizes",[])) != sudoku_sizes:
    mismatch(f"sudoku_sizes {meta.get('sudoku_sizes')} != {sudoku_sizes}")
if sorted(meta.get("nqueens_sizes",[])) != nqueens_sizes:
    mismatch(f"nqueens_sizes {meta.get('nqueens_sizes')} != {nqueens_sizes}")
if "ensure_unique_train" in meta or "ensure_unique_test" in meta:
    if bool(meta.get("ensure_unique_train", False)) != train_unique:
        mismatch(f"ensure_unique_train {meta.get('ensure_unique_train')} != {train_unique}")
    if bool(meta.get("ensure_unique_test", False)) != test_unique:
        mismatch(f"ensure_unique_test {meta.get('ensure_unique_test')} != {test_unique}")
else:
    # backward compatibility with older meta schema
    if bool(meta.get("ensure_unique", True)) != unique_check_global:
        mismatch(f"ensure_unique {meta.get('ensure_unique')} != {unique_check_global}")
if int(meta.get("per_record_max_attempts",-1)) != per_record_max_attempts:
    mismatch(f"per_record_max_attempts {meta.get('per_record_max_attempts')} != {per_record_max_attempts}")
if int(meta.get("uniqueness_nodes",-1)) != uniqueness_nodes:
    mismatch(f"uniqueness_nodes {meta.get('uniqueness_nodes')} != {uniqueness_nodes}")
if abs(float(meta.get("uniqueness_timeout_sec",-1.0)) - uniqueness_timeout) > 1e-9:
    mismatch(f"uniqueness_timeout_sec {meta.get('uniqueness_timeout_sec')} != {uniqueness_timeout}")
if meta.get("difficulty_schema") != expected_difficulty_schema:
    mismatch(f"difficulty_schema {meta.get('difficulty_schema')} != {expected_difficulty_schema}")
if list(meta.get("difficulty_levels", [])) != expected_difficulty_levels:
    mismatch(f"difficulty_levels {meta.get('difficulty_levels')} != {expected_difficulty_levels}")
if bool(meta.get("difficulty_stratified", True)) != diff_stratified:
    mismatch(f"difficulty_stratified {meta.get('difficulty_stratified')} != {diff_stratified}")
if not diff_stratified:
    if str(meta.get("generation_difficulty", "")).strip().lower() != generation_difficulty:
        mismatch(f"generation_difficulty {meta.get('generation_difficulty')} != {generation_difficulty}")

print("compatible")
PY
}

is_family_model_reusable() {
  local family="$1"
  local expected_sizes="$2"
  local min_best_metric="$3"
  python3 - "$REGISTRY" "$family" "$expected_sizes" "$DATA_ROOT" "$STRICT_REUSE_CHECKS" "$min_best_metric" <<'PY'
import json, os, sys
from pathlib import Path

registry=Path(sys.argv[1])
family=sys.argv[2]
expected_sizes=[s for s in sys.argv[3].split(",") if s]
data_root=Path(sys.argv[4]).resolve()
strict=(sys.argv[5]!="0")
min_best_raw=sys.argv[6].strip()

if not registry.exists():
    print("model not reusable: missing registry")
    sys.exit(1)

d=json.loads(registry.read_text(encoding="utf-8"))
item=d.get(family)
if not item:
    print(f"model not reusable: missing family entry {family}")
    sys.exit(1)

ckpt=Path(item.get("checkpoint",""))
if not ckpt.exists():
    print(f"model not reusable: checkpoint missing {ckpt}")
    sys.exit(1)

meta=item.get("meta",{})
sizes=meta.get("sizes",[])
if not all(s in sizes for s in expected_sizes):
    print(f"model not reusable: sizes mismatch, have={sizes}, expect={expected_sizes}")
    sys.exit(1)

if not strict:
    print("reusable (strict off)")
    sys.exit(0)

def to_bool01(v):
    return str(v).strip() not in ("0","false","False","")

def eq_float(a,b,eps=1e-9):
    return abs(float(a)-float(b)) <= eps

def fail(msg):
    print(f"model not reusable: {msg}")
    sys.exit(1)

meta_data_root = meta.get("data_root", "")
if meta_data_root:
    try:
        if Path(meta_data_root).resolve() != data_root:
            fail(f"data_root {meta_data_root} != {data_root}")
    except Exception:
        fail(f"invalid stored data_root {meta_data_root}")

prefix = "GEN_SUDOKU" if family == "gen_sudoku" else "NQUEENS"
model_cfg = meta.get("model_config", {})
mdm_cfg = meta.get("mdm_config", {})
weights = meta.get("weights", {})

expect_model = {
    "hidden_size": int(os.environ.get(f"{prefix}_HIDDEN_SIZE", "0")),
    "num_layers": int(os.environ.get(f"{prefix}_NUM_LAYERS", "0")),
    "num_heads": int(os.environ.get(f"{prefix}_NUM_HEADS", "0")),
    "dropout": float(os.environ.get(f"{prefix}_DROPOUT", "0")),
}
for k,v in expect_model.items():
    if k not in model_cfg:
        fail(f"missing model_config.{k}")
    if isinstance(v, float):
        if not eq_float(model_cfg[k], v):
            fail(f"model_config.{k}={model_cfg[k]} != {v}")
    else:
        if int(model_cfg[k]) != v:
            fail(f"model_config.{k}={model_cfg[k]} != {v}")

expect_mdm = {
    "diffusion_steps": int(os.environ.get(f"{prefix}_DIFFUSION_STEPS", "0")),
    "token_reweighting": to_bool01(os.environ.get(f"{prefix}_TOKEN_REWEIGHTING", "1")),
    "loss_alpha": float(os.environ.get(f"{prefix}_LOSS_ALPHA", "0")),
    "loss_gamma": float(os.environ.get(f"{prefix}_LOSS_GAMMA", "0")),
    "time_reweighting": os.environ.get(f"{prefix}_TIME_REWEIGHTING", ""),
    "lr_scheduler": os.environ.get(f"{prefix}_LR_SCHEDULER", ""),
    "warmup_ratio": float(os.environ.get(f"{prefix}_WARMUP_RATIO", "0")),
    "gradient_accumulation_steps": int(os.environ.get(f"{prefix}_GRAD_ACC_STEPS", "1")),
    "max_grad_norm": float(os.environ.get(f"{prefix}_MAX_GRAD_NORM", "0")),
}
for k,v in expect_mdm.items():
    if k not in mdm_cfg:
        fail(f"missing mdm_config.{k}")
    got = mdm_cfg[k]
    if isinstance(v, bool):
        if bool(got) != v:
            fail(f"mdm_config.{k}={got} != {v}")
    elif isinstance(v, float):
        if not eq_float(got, v):
            fail(f"mdm_config.{k}={got} != {v}")
    elif isinstance(v, int):
        if int(got) != v:
            fail(f"mdm_config.{k}={got} != {v}")
    else:
        if str(got) != str(v):
            fail(f"mdm_config.{k}={got} != {v}")

expect_weights = {
    "easy_weight": float(os.environ.get(f"{prefix}_EASY_WEIGHT", "0")),
    "medium_weight": float(os.environ.get(f"{prefix}_MEDIUM_WEIGHT", "0")),
    "hard_weight": float(os.environ.get(f"{prefix}_HARD_WEIGHT", "0")),
    "unknown_ratio_weight": float(os.environ.get(f"{prefix}_UNKNOWN_RATIO_WEIGHT", "0")),
}
for k,v in expect_weights.items():
    if k not in weights:
        fail(f"missing weights.{k}")
    if not eq_float(weights[k], v):
        fail(f"weights.{k}={weights[k]} != {v}")

if min_best_raw:
    min_best=float(min_best_raw)
    best=float(meta.get("best_metric", -1.0))
    if best < min_best:
        fail(f"best_metric {best} < min_required {min_best}")

print("reusable")
PY
}

if [[ "$SKIP_EXTERNAL_DISCOVER" == "1" ]]; then
  echo
  echo "[0/3] Skip external source discover"
  write_status discover skipped
else
  echo
  echo "[0/3] Discover external sources"
  write_status discover running
  python3 "$ROOT/dataset/discover_external_data.py"
  cp -f "$ROOT/dataset/accepted_sources.json" "$MANIFEST_DIR/accepted_sources.json" 2>/dev/null || true
  cp -f "$ROOT/dataset/rejected_sources.json" "$MANIFEST_DIR/rejected_sources.json" 2>/dev/null || true
  write_status discover done
fi

echo
echo "[1/3] Build unified datasets"
write_status data running
if [[ "$SKIP_DATA_BUILD" == "1" ]]; then
  if is_dataset_ready >/dev/null 2>&1 && is_dataset_compatible >/dev/null 2>&1; then
    echo "Skip data build enabled and dataset is ready."
  else
    echo "ERROR: SKIP_DATA_BUILD=1 but dataset is not ready for current sizes/counts."
    exit 1
  fi
elif [[ "$FORCE_REBUILD_DATA" == "1" ]]; then
  echo "Force rebuild enabled."
  BUILD_CMD=(
    python3 "$ROOT/dataset/build_table4_extension_data.py"
    --train-count "$TRAIN_COUNT"
    --test-count "$TEST_COUNT"
    --seed "$SEED"
    --max-generation-attempts "$MAX_GEN_ATTEMPTS"
    --per-record-max-attempts "$PER_RECORD_MAX_ATTEMPTS"
    --uniqueness-nodes "$UNIQUENESS_NODES"
    --uniqueness-timeout-sec "$UNIQUENESS_TIMEOUT_SEC"
    --generation-workers "$GENERATION_WORKERS"
    --progress-every "$DATA_PROGRESS_EVERY"
    --generation-difficulty "$GENERATION_DIFFICULTY"
    --sudoku-sizes "$SUDOKU_SIZES"
    --nqueens-sizes "$NQUEENS_SIZES"
    --output-root "$DATA_ROOT"
  )
  if [[ "$ALLOW_DUPLICATES" == "1" ]]; then
    BUILD_CMD+=(--allow-duplicates)
  fi
  if [[ "$UNIQUE_CHECK" == "0" ]]; then
    BUILD_CMD+=(--disable-unique-check)
  fi
  if [[ "$TRAIN_UNIQUE_CHECK" == "0" ]]; then
    BUILD_CMD+=(--disable-train-unique-check)
  fi
  if [[ "$TEST_UNIQUE_CHECK" == "0" ]]; then
    BUILD_CMD+=(--disable-test-unique-check)
  fi
  if [[ "$DIFFICULTY_STRATIFICATION" == "0" ]]; then
    BUILD_CMD+=(--disable-difficulty-stratification)
  fi
  "${BUILD_CMD[@]}"
elif [[ "$RESUME" == "1" ]] && is_dataset_ready >/dev/null 2>&1 && is_dataset_compatible >/dev/null 2>&1; then
  echo "Dataset reusable under current config, skip build."
else
  if [[ "$RESUME" == "1" ]]; then
    echo "Dataset exists but is not reusable under current config; rebuild."
    is_dataset_compatible || true
  fi
  BUILD_CMD=(
    python3 "$ROOT/dataset/build_table4_extension_data.py"
    --train-count "$TRAIN_COUNT"
    --test-count "$TEST_COUNT"
    --seed "$SEED"
    --max-generation-attempts "$MAX_GEN_ATTEMPTS"
    --per-record-max-attempts "$PER_RECORD_MAX_ATTEMPTS"
    --uniqueness-nodes "$UNIQUENESS_NODES"
    --uniqueness-timeout-sec "$UNIQUENESS_TIMEOUT_SEC"
    --generation-workers "$GENERATION_WORKERS"
    --progress-every "$DATA_PROGRESS_EVERY"
    --generation-difficulty "$GENERATION_DIFFICULTY"
    --sudoku-sizes "$SUDOKU_SIZES"
    --nqueens-sizes "$NQUEENS_SIZES"
    --output-root "$DATA_ROOT"
  )
  if [[ "$ALLOW_DUPLICATES" == "1" ]]; then
    BUILD_CMD+=(--allow-duplicates)
  fi
  if [[ "$UNIQUE_CHECK" == "0" ]]; then
    BUILD_CMD+=(--disable-unique-check)
  fi
  if [[ "$TRAIN_UNIQUE_CHECK" == "0" ]]; then
    BUILD_CMD+=(--disable-train-unique-check)
  fi
  if [[ "$TEST_UNIQUE_CHECK" == "0" ]]; then
    BUILD_CMD+=(--disable-test-unique-check)
  fi
  if [[ "$DIFFICULTY_STRATIFICATION" == "0" ]]; then
    BUILD_CMD+=(--disable-difficulty-stratification)
  fi
  "${BUILD_CMD[@]}"
fi
write_status data done

echo
echo "[2/3] Train family models"
write_status train running
run_train_gen_sudoku() {
  DATA_ROOT="$DATA_ROOT" OUTPUT_ROOT="$MODEL_ROOT" GPUS="$GEN_SUDOKU_GPUS" SIZES="$GEN_SUDOKU_SIZES" \
  EPOCHS="$GEN_SUDOKU_EPOCHS" BATCH_SIZE="$GEN_SUDOKU_BATCH_SIZE" LEARNING_RATE="$GEN_SUDOKU_LR" \
  NUM_WORKERS="$GEN_SUDOKU_WORKERS" SEED="$SEED" \
  EASY_WEIGHT="$GEN_SUDOKU_EASY_WEIGHT" MEDIUM_WEIGHT="$GEN_SUDOKU_MEDIUM_WEIGHT" HARD_WEIGHT="$GEN_SUDOKU_HARD_WEIGHT" \
  UNKNOWN_RATIO_WEIGHT="$GEN_SUDOKU_UNKNOWN_RATIO_WEIGHT" \
  HIDDEN_SIZE="$GEN_SUDOKU_HIDDEN_SIZE" NUM_LAYERS="$GEN_SUDOKU_NUM_LAYERS" NUM_HEADS="$GEN_SUDOKU_NUM_HEADS" DROPOUT="$GEN_SUDOKU_DROPOUT" \
  DIFFUSION_STEPS="$GEN_SUDOKU_DIFFUSION_STEPS" TOKEN_REWEIGHTING="$GEN_SUDOKU_TOKEN_REWEIGHTING" LOSS_ALPHA="$GEN_SUDOKU_LOSS_ALPHA" LOSS_GAMMA="$GEN_SUDOKU_LOSS_GAMMA" TIME_REWEIGHTING="$GEN_SUDOKU_TIME_REWEIGHTING" \
  LR_SCHEDULER="$GEN_SUDOKU_LR_SCHEDULER" WARMUP_RATIO="$GEN_SUDOKU_WARMUP_RATIO" GRAD_ACC_STEPS="$GEN_SUDOKU_GRAD_ACC_STEPS" MAX_GRAD_NORM="$GEN_SUDOKU_MAX_GRAD_NORM" \
  DISABLE_WEIGHTED_SAMPLING="$GEN_SUDOKU_DISABLE_WEIGHTED_SAMPLING" \
  EVAL_SPLIT="$GEN_SUDOKU_EVAL_SPLIT" EVAL_BATCH_SIZE="$GEN_SUDOKU_EVAL_BATCH_SIZE" EVAL_EVERY="$GEN_SUDOKU_EVAL_EVERY" \
    bash "$ROOT/model/diffusion-vs-ar/scripts/extension/train_gen_sudoku_mdm.sh"
}
run_train_nqueens() {
  DATA_ROOT="$DATA_ROOT" OUTPUT_ROOT="$MODEL_ROOT" GPUS="$NQUEENS_GPUS" SIZES="$NQUEENS_SIZES" \
  EPOCHS="$NQUEENS_EPOCHS" BATCH_SIZE="$NQUEENS_BATCH_SIZE" LEARNING_RATE="$NQUEENS_LR" \
  NUM_WORKERS="$NQUEENS_WORKERS" SEED="$SEED" \
  EASY_WEIGHT="$NQUEENS_EASY_WEIGHT" MEDIUM_WEIGHT="$NQUEENS_MEDIUM_WEIGHT" HARD_WEIGHT="$NQUEENS_HARD_WEIGHT" \
  UNKNOWN_RATIO_WEIGHT="$NQUEENS_UNKNOWN_RATIO_WEIGHT" \
  HIDDEN_SIZE="$NQUEENS_HIDDEN_SIZE" NUM_LAYERS="$NQUEENS_NUM_LAYERS" NUM_HEADS="$NQUEENS_NUM_HEADS" DROPOUT="$NQUEENS_DROPOUT" \
  DIFFUSION_STEPS="$NQUEENS_DIFFUSION_STEPS" TOKEN_REWEIGHTING="$NQUEENS_TOKEN_REWEIGHTING" LOSS_ALPHA="$NQUEENS_LOSS_ALPHA" LOSS_GAMMA="$NQUEENS_LOSS_GAMMA" TIME_REWEIGHTING="$NQUEENS_TIME_REWEIGHTING" \
  LR_SCHEDULER="$NQUEENS_LR_SCHEDULER" WARMUP_RATIO="$NQUEENS_WARMUP_RATIO" GRAD_ACC_STEPS="$NQUEENS_GRAD_ACC_STEPS" MAX_GRAD_NORM="$NQUEENS_MAX_GRAD_NORM" \
  DISABLE_WEIGHTED_SAMPLING="$NQUEENS_DISABLE_WEIGHTED_SAMPLING" \
  EVAL_SPLIT="$NQUEENS_EVAL_SPLIT" EVAL_BATCH_SIZE="$NQUEENS_EVAL_BATCH_SIZE" EVAL_EVERY="$NQUEENS_EVAL_EVERY" \
    bash "$ROOT/model/diffusion-vs-ar/scripts/extension/train_nqueens_mdm.sh"
}
if [[ "$FORCE_RETRAIN_GEN_SUDOKU" == "1" ]]; then
  echo "Force retrain gen_sudoku enabled."
  run_train_gen_sudoku
elif [[ "$RESUME" == "1" ]] && is_family_model_reusable "gen_sudoku" "$GEN_SUDOKU_SIZES" "$GEN_SUDOKU_MIN_BEST_METRIC" >/dev/null 2>&1; then
  echo "gen_sudoku model reusable, skip training."
else
  if [[ "$RESUME" == "1" ]]; then
    echo "gen_sudoku model exists but is not reusable; retrain."
    is_family_model_reusable "gen_sudoku" "$GEN_SUDOKU_SIZES" "$GEN_SUDOKU_MIN_BEST_METRIC" || true
  fi
  run_train_gen_sudoku
fi

if [[ "$FORCE_RETRAIN_NQUEENS" == "1" ]]; then
  echo "Force retrain nqueens enabled."
  run_train_nqueens
elif [[ "$RESUME" == "1" ]] && is_family_model_reusable "nqueens" "$NQUEENS_SIZES" "$NQUEENS_MIN_BEST_METRIC" >/dev/null 2>&1; then
  echo "nqueens model reusable, skip training."
else
  if [[ "$RESUME" == "1" ]]; then
    echo "nqueens model exists but is not reusable; retrain."
    is_family_model_reusable "nqueens" "$NQUEENS_SIZES" "$NQUEENS_MIN_BEST_METRIC" || true
  fi
  run_train_nqueens
fi
write_status train done

echo
echo "[3/3] Run unified evaluation"
write_status eval running
RUN_CMD=(
  python3 "$ROOT/DiBS/run_table4_extension.py"
  --run-id "$RUN_ID"
  --data-root "$DATA_ROOT"
  --output "$EVAL_ROOT"
  --registry "$REGISTRY"
  --tasks "$TASKS"
  --timeout-ms "$TIMEOUT_MS"
  --max-nodes "$MAX_NODES"
  --alpha "$ALPHA"
  --smart-interval "$SMART_INTERVAL"
  --seed "$SEED"
  --model-device "$MODEL_DEVICE"
  --workers "$WORKERS"
  --sudoku-sizes "$SUDOKU_SIZES"
  --nqueens-sizes "$NQUEENS_SIZES"
)
if [[ "$RESUME" == "1" ]]; then
  RUN_CMD+=(--resume)
fi
if [[ -n "${MAX_PUZZLES}" ]]; then
  RUN_CMD+=(--max-puzzles "$MAX_PUZZLES")
fi
if [[ -n "${GPUS}" ]]; then
  RUN_CMD+=(--gpus "$GPUS")
fi
"${RUN_CMD[@]}"
write_status eval done

echo
echo "Pipeline done."
echo "Run dir: $RUN_DIR"
echo "Config : $CONFIG_PATH"
echo "Status : $STATUS_PATH"
