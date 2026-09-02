#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPELINE_CONFIG="${PIPELINE_CONFIG:-$ROOT/configs/full_scale.env}"
MODE="${1:-preflight}"
[[ -f "$PIPELINE_CONFIG" ]] || {
  echo "Missing PIPELINE_CONFIG: $PIPELINE_CONFIG" >&2
  echo "Copy configs/full_scale.env.example to configs/full_scale.env and edit it." >&2
  exit 2
}
set -a
# shellcheck disable=SC1090
source "$PIPELINE_CONFIG"
set +a

for name in PYTHON_BIN MODEL_PATH EAGLE_ROOT DATA_ROOT RUN_ROOT GPU_IDS GLOBAL_BATCH; do
  [[ -n "${!name:-}" ]] || { echo "Missing $name in $PIPELINE_CONFIG" >&2; exit 2; }
done
mkdir -p "$RUN_ROOT/run_configs" "$RUN_ROOT/evaluation_manifests" "$RUN_ROOT/evaluation"

STAGE1_OUTPUT="$RUN_ROOT/stage1_coarse"
STAGE2_OUTPUT="$RUN_ROOT/stage2_dense_balanced"
STAGE1_RECIPE="$DATA_ROOT/recipes/stage1_coarse.json"
STAGE2_RECIPE="$DATA_ROOT/recipes/stage2_dense_balanced.json"
VALIDATION_RECIPE="$DATA_ROOT/recipes/validation_all_tasks.json"

write_stage_config() {
  local stage="$1" recipe="$2" output="$3" init_adapter="$4" warmup="$5" expected="$6"
  local destination="$RUN_ROOT/run_configs/$stage.env"
  {
    printf 'PYTHON_BIN=%q\n' "$PYTHON_BIN"
    printf 'MODEL_PATH=%q\n' "$MODEL_PATH"
    printf 'EAGLE_ROOT=%q\n' "$EAGLE_ROOT"
    printf 'DATA_ROOT=%q\n' "$DATA_ROOT"
    printf 'TRAIN_RECIPE=%q\n' "$recipe"
    printf 'VALIDATION_RECIPE=%q\n' "$VALIDATION_RECIPE"
    printf 'OUTPUT_DIR=%q\n' "$output"
    printf 'SMOKE_DIR=%q\n' "$RUN_ROOT/smoke/$stage"
    printf 'GPU_IDS=%q\n' "$GPU_IDS"
    printf 'GLOBAL_BATCH=%q\n' "$GLOBAL_BATCH"
    printf 'MAX_STEPS=0\nALLOWED_PADDING_RECORDS=auto\n'
    printf 'EXPECTED_TRAIN_RECORDS=%q\n' "$expected"
    printf 'EXPECTED_VALIDATION_RECORDS=95740\n'
    printf 'SAMPLE_ORDER=%q\n' "${SAMPLE_ORDER:-shuffled}"
    printf 'VALIDATION_RECORDS=%q\n' "${VALIDATION_RECORDS:-1000}"
    printf 'VALIDATION_SAMPLING=%q\n' "${VALIDATION_SAMPLING:-stratified_files}"
    printf 'LORA_RANK=%q\n' "${LORA_RANK:-16}"
    printf 'IMAGE_TOKEN_LIMIT=%q\n' "${IMAGE_TOKEN_LIMIT:-6000}"
    printf 'MAX_SEQUENCE=%q\n' "${MAX_SEQUENCE:-8192}"
    printf 'VISUAL_CONTEXT=%q\n' "${VISUAL_CONTEXT:-preprojector_magnified_roi}"
    printf 'MAGNIFIED_ROI_PIXELS=%q\n' "${MAGNIFIED_ROI_PIXELS:-380}"
    printf 'MAGNIFIED_ROI_STRIDE=%q\n' "${MAGNIFIED_ROI_STRIDE:-1}"
    printf 'LEARNING_RATE=%q\n' "${LEARNING_RATE:-1e-5}"
    printf 'WARMUP_STEPS=%q\n' "$warmup"
    printf 'CHECKPOINT_STEPS=%q\n' "${CHECKPOINT_STEPS:-500}"
    printf 'EVAL_STEPS=%q\n' "${EVAL_STEPS:-500}"
    printf 'WORKERS=%q\n' "${WORKERS:-4}"
    printf 'SEED=%q\n' "${SEED:-20260901}"
    printf 'VISION_ATTENTION=%q\n' "${VISION_ATTENTION:-auto}"
    printf 'SERIAL_MODEL_LOAD=%q\n' "${SERIAL_MODEL_LOAD:-1}"
    printf 'WANDB_PROJECT=%q\n' "${WANDB_PROJECT:-}"
    printf 'WANDB_NAME=%q\n' "${WANDB_NAME_PREFIX:-pixel-pivr-full-scale}-$stage"
    printf 'INIT_ADAPTER=%q\n' "$init_adapter"
  } > "$destination"
  echo "$destination"
}

stage1_config() {
  write_stage_config stage1 "$STAGE1_RECIPE" "$STAGE1_OUTPUT" "" "${WARMUP_STEPS_STAGE1:-1000}" 388410
}

require_stage1_complete() {
  [[ -f "$STAGE1_OUTPUT/done.json" ]] || { echo "Stage 1 is not complete: $STAGE1_OUTPUT/done.json" >&2; exit 2; }
  "$PYTHON_BIN" - "$STAGE1_OUTPUT/done.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
if not value.get("complete_one_pass"):
    raise SystemExit("Stage 1 done.json does not certify one complete pass")
PY
  [[ -e "$STAGE1_OUTPUT/best.pt" ]] || { echo "Stage 1 best.pt is missing" >&2; exit 2; }
}

stage2_config() {
  require_stage1_complete
  write_stage_config stage2 "$STAGE2_RECIPE" "$STAGE2_OUTPUT" "$STAGE1_OUTPUT/best.pt" "${WARMUP_STEPS_STAGE2:-1000}" 1862021
}

run_stage() {
  local config="$1" action="$2"
  RUN_CONFIG="$config" bash "$ROOT/scripts/train_distributed.sh" "$action"
}

case "$MODE" in
  preflight)
    "$PYTHON_BIN" "$ROOT/tools/preflight.py" --model "$MODEL_PATH" \
      --eagle-root "$EAGLE_ROOT" --data-root "$DATA_ROOT" --global-batch "$GLOBAL_BATCH"
    run_stage "$(stage1_config)" check
    ;;
  audit-stage1|smoke-stage1|train-stage1|status-stage1)
    run_stage "$(stage1_config)" "${MODE%-stage1}"
    ;;
  audit-stage2|smoke-stage2|train-stage2|status-stage2)
    run_stage "$(stage2_config)" "${MODE%-stage2}"
    ;;
  prepare-eval)
    "$PYTHON_BIN" "$ROOT/tools/prepare_evaluation.py" \
      --data-root "$DATA_ROOT" --output "$RUN_ROOT/evaluation_manifests"
    ;;
  evaluate)
    PIPELINE_CONFIG="$PIPELINE_CONFIG" bash "$ROOT/scripts/evaluate_all.sh"
    ;;
  all)
    run_stage "$(stage1_config)" check
    run_stage "$(stage1_config)" smoke
    run_stage "$(stage1_config)" train
    run_stage "$(stage2_config)" smoke
    run_stage "$(stage2_config)" train
    PIPELINE_CONFIG="$PIPELINE_CONFIG" bash "$ROOT/scripts/evaluate_all.sh"
    ;;
  status)
    run_stage "$(stage1_config)" status
    if [[ -f "$STAGE1_OUTPUT/done.json" ]]; then
      run_stage "$(stage2_config)" status
    fi
    ;;
  *)
    echo "Usage: $0 {preflight|audit-stage1|smoke-stage1|train-stage1|status-stage1|audit-stage2|smoke-stage2|train-stage2|status-stage2|prepare-eval|evaluate|all|status}" >&2
    exit 2
    ;;
esac
