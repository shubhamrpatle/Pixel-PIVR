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

for name in PYTHON_BIN MODEL_PATH EAGLE_ROOT DATA_ROOT RUN_ROOT GPU_IDS GLOBAL_BATCH \
  CODE_REVISION DATA_REVISION MODEL_REVISION EAGLE_REVISION; do
  [[ -n "${!name:-}" ]] || { echo "Missing $name in $PIPELINE_CONFIG" >&2; exit 2; }
done
for name in CODE_REVISION DATA_REVISION MODEL_REVISION EAGLE_REVISION; do
  [[ "${!name}" =~ ^[0-9a-f]{40}$ ]] || {
    echo "$name must be an immutable 40-character commit SHA, got ${!name@Q}" >&2
    exit 2
  }
done
ACTUAL_CODE_REVISION="$(git -C "$ROOT" rev-parse HEAD)"
[[ "$ACTUAL_CODE_REVISION" == "$CODE_REVISION" ]] || {
  echo "Code revision mismatch: $ACTUAL_CODE_REVISION != $CODE_REVISION" >&2
  exit 2
}
SOURCE_DIRTY="$(git -C "$ROOT" status --porcelain --untracked-files=normal)"
[[ -z "$SOURCE_DIRTY" ]] || {
  echo "Pixel-PIVR checkout has uncommitted source files:" >&2
  printf '%s\n' "$SOURCE_DIRTY" >&2
  exit 2
}
"$PYTHON_BIN" "$ROOT/tools/source_manifest.py" check
mkdir -p "$RUN_ROOT/run_configs" "$RUN_ROOT/evaluation_manifests" "$RUN_ROOT/evaluation"

[[ "${ARCHITECTURE_CONTRACT:-}" == "pixel_crop_144to384_v2" ]] || {
  echo "ARCHITECTURE_CONTRACT must be pixel_crop_144to384_v2" >&2
  exit 2
}
[[ "${VISUAL_CONTEXT:-}" == "pixel_reencoded" ]] || {
  echo "The magnified-v2 pipeline requires VISUAL_CONTEXT=pixel_reencoded" >&2
  exit 2
}
[[ "${SOURCE_CROP_SIDE:-}" == "144" && "${LOCAL_INPUT_SIDE:-}" == "384" ]] || {
  echo "The release crop contract is exactly 144 source pixels -> 384 input pixels" >&2
  exit 2
}
[[ "${MAGNIFIED_ROI_PIXELS:-}" == "384" ]] || {
  echo "MAGNIFIED_ROI_PIXELS must be 384 for the pixel-reencoded release contract" >&2
  exit 2
}
[[ "${LOSS_BALANCING:-}" == "source_query_task" ]] || {
  echo "LOSS_BALANCING must be source_query_task for the full-scale release" >&2
  exit 2
}

STAGE1_OUTPUT="$RUN_ROOT/stage1_coarse"
STAGE2_OUTPUT="$RUN_ROOT/stage2_dense_balanced"
STAGE1_RECIPE="$DATA_ROOT/recipes/stage1_coarse.json"
STAGE2_RECIPE="$DATA_ROOT/recipes/stage2_dense_balanced.json"
VALIDATION_RECIPE="$DATA_ROOT/recipes/validation_monitor_all_tasks.json"
DATA_MANIFEST="$DATA_ROOT/manifest.json"

manifest_count() {
  local key="$1"
  "$PYTHON_BIN" - "$DATA_MANIFEST" "$key" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))["training_contract"][sys.argv[2]]
if not isinstance(value, int) or value <= 0:
    raise SystemExit(f"Invalid training_contract count {sys.argv[2]}={value!r}")
print(value)
PY
}

STAGE1_RECORDS="$(manifest_count stage1_records)"
STAGE2_RECORDS="$(manifest_count stage2_records)"
VALIDATION_RECORDS_TOTAL="$(manifest_count validation_monitor_records)"

require_training_approval() {
  [[ "${FULL_SCALE_APPROVED:-NO}" == "YES" ]] || {
    echo "Training is locked. Run preflight and the stage smoke test, inspect it, then set FULL_SCALE_APPROVED=YES." >&2
    exit 2
  }
}

require_smoke_success() {
  local stage="$1" config="$2"
  local receipt="$RUN_ROOT/smoke/$stage/latest_success.json"
  [[ -f "$receipt" ]] || {
    echo "No successful $stage smoke receipt was found at $receipt" >&2
    exit 2
  }
  "$PYTHON_BIN" - "$receipt" "$config" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

receipt, config = map(Path, sys.argv[1:])
value = json.loads(receipt.read_text(encoding="utf-8"))
actual = hashlib.sha256(config.read_bytes()).hexdigest()
if value.get("run_config_sha256") != actual:
    raise SystemExit(
        "Smoke receipt does not match the current stage config; rerun the smoke test"
    )
done = Path(str(value.get("done_json") or ""))
if not done.is_file():
    raise SystemExit(f"Smoke completion file is missing: {done}")
PY
}

write_stage_config() {
  local stage="$1" recipe="$2" output="$3" init_adapter="$4" warmup="$5" expected="$6" learning_rate="$7"
  local destination="$RUN_ROOT/run_configs/$stage.env"
  {
    printf 'PYTHON_BIN=%q\n' "$PYTHON_BIN"
    printf 'MODEL_PATH=%q\n' "$MODEL_PATH"
    printf 'EAGLE_ROOT=%q\n' "$EAGLE_ROOT"
    printf 'DATA_ROOT=%q\n' "$DATA_ROOT"
    printf 'ARCHITECTURE_CONTRACT=%q\n' "$ARCHITECTURE_CONTRACT"
    printf 'SOURCE_CROP_SIDE=%q\n' "$SOURCE_CROP_SIDE"
    printf 'LOCAL_INPUT_SIDE=%q\n' "$LOCAL_INPUT_SIDE"
    printf 'TRAIN_RECIPE=%q\n' "$recipe"
    printf 'VALIDATION_RECIPE=%q\n' "$VALIDATION_RECIPE"
    printf 'OUTPUT_DIR=%q\n' "$output"
    printf 'SMOKE_DIR=%q\n' "$RUN_ROOT/smoke/$stage"
    printf 'GPU_IDS=%q\n' "$GPU_IDS"
    printf 'GLOBAL_BATCH=%q\n' "$GLOBAL_BATCH"
    printf 'MAX_STEPS=0\nALLOWED_PADDING_RECORDS=auto\n'
    printf 'EXPECTED_TRAIN_RECORDS=%q\n' "$expected"
    printf 'EXPECTED_VALIDATION_RECORDS=%q\n' "$VALIDATION_RECORDS_TOTAL"
    printf 'SAMPLE_ORDER=%q\n' "${SAMPLE_ORDER:-shuffled}"
    printf 'VALIDATION_RECORDS=%q\n' "${VALIDATION_RECORDS:-0}"
    printf 'VALIDATION_SAMPLING=%q\n' "${VALIDATION_SAMPLING:-first}"
    printf 'LORA_RANK=%q\n' "${LORA_RANK:-16}"
    printf 'LOSS_BALANCING=%q\n' "${LOSS_BALANCING:-source_query_task}"
    printf 'IMAGE_TOKEN_LIMIT=%q\n' "${IMAGE_TOKEN_LIMIT:-6000}"
    printf 'MAX_SEQUENCE=%q\n' "${MAX_SEQUENCE:-32768}"
    printf 'VISUAL_CONTEXT=%q\n' "${VISUAL_CONTEXT:-pixel_reencoded}"
    printf 'MAGNIFIED_ROI_PIXELS=%q\n' "${MAGNIFIED_ROI_PIXELS:-384}"
    printf 'MAGNIFIED_ROI_STRIDE=%q\n' "${MAGNIFIED_ROI_STRIDE:-1}"
    printf 'LEARNING_RATE=%q\n' "$learning_rate"
    printf 'WEIGHT_DECAY=%q\n' "${WEIGHT_DECAY:-0.01}"
    printf 'MAX_GRAD_NORM=%q\n' "${MAX_GRAD_NORM:-1.0}"
    printf 'WARMUP_STEPS=%q\n' "$warmup"
    printf 'CHECKPOINT_STEPS=%q\n' "${CHECKPOINT_STEPS:-500}"
    printf 'KEEP_RECENT_CHECKPOINTS=%q\n' "${KEEP_RECENT_CHECKPOINTS:-1}"
    printf 'EVAL_STEPS=%q\n' "${EVAL_STEPS:-500}"
    printf 'LOG_STEPS=%q\n' "${LOG_STEPS:-50}"
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
  write_stage_config stage1 "$STAGE1_RECIPE" "$STAGE1_OUTPUT" "" \
    "${WARMUP_STEPS_STAGE1:-600}" "$STAGE1_RECORDS" "${LEARNING_RATE_STAGE1:-1e-5}"
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
  write_stage_config stage2 "$STAGE2_RECIPE" "$STAGE2_OUTPUT" "$STAGE1_OUTPUT/best.pt" \
    "${WARMUP_STEPS_STAGE2:-1500}" "$STAGE2_RECORDS" "${LEARNING_RATE_STAGE2:-5e-6}"
}

run_stage() {
  local config="$1" action="$2"
  RUN_CONFIG="$config" bash "$ROOT/scripts/train_distributed.sh" "$action"
}

case "$MODE" in
  preflight)
    PREFLIGHT_ARGS=(
      --model "$MODEL_PATH" --eagle-root "$EAGLE_ROOT" --data-root "$DATA_ROOT"
      --run-root "$RUN_ROOT" --global-batch "$GLOBAL_BATCH" --gpu-ids "$GPU_IDS"
      --required-gpus "${REQUIRED_GPUS:-8}"
      --required-gpu-name "${REQUIRED_GPU_NAME:-A100}"
      --minimum-gpu-memory-gib "${MINIMUM_GPU_MEMORY_GIB:-75}"
      --minimum-run-free-gib "${MINIMUM_RUN_FREE_GIB:-100}"
      --max-sequence "${MAX_SEQUENCE:-32768}"
      --image-token-limit "${IMAGE_TOKEN_LIMIT:-6000}"
      --loss-balancing "$LOSS_BALANCING"
      --code-root "$ROOT"
      --expected-code-revision "$CODE_REVISION"
      --download-receipt "$(dirname "$DATA_ROOT")/download_receipt.json"
      --expected-data-revision "$DATA_REVISION"
      --expected-model-revision "$MODEL_REVISION"
      --expected-eagle-revision "$EAGLE_REVISION"
    )
    [[ "${VERIFY_IMAGE_HASHES:-0}" == "1" ]] && PREFLIGHT_ARGS+=(--verify-image-hashes)
    [[ "${REQUIRE_FLASH_ATTN:-0}" == "1" ]] && PREFLIGHT_ARGS+=(--require-flash-attn)
    "$PYTHON_BIN" "$ROOT/tools/preflight.py" "${PREFLIGHT_ARGS[@]}"
    run_stage "$(stage1_config)" check
    ;;
  audit-stage1|smoke-stage1|status-stage1)
    run_stage "$(stage1_config)" "${MODE%-stage1}"
    ;;
  train-stage1)
    require_training_approval
    config="$(stage1_config)"
    require_smoke_success stage1 "$config"
    run_stage "$config" train
    ;;
  audit-stage2|smoke-stage2|status-stage2)
    run_stage "$(stage2_config)" "${MODE%-stage2}"
    ;;
  train-stage2)
    require_training_approval
    config="$(stage2_config)"
    require_smoke_success stage2 "$config"
    run_stage "$config" train
    ;;
  prepare-eval)
    "$PYTHON_BIN" "$ROOT/tools/prepare_evaluation.py" \
      --data-root "$DATA_ROOT" --output "$RUN_ROOT/evaluation_manifests"
    ;;
  evaluate)
    PIPELINE_CONFIG="$PIPELINE_CONFIG" bash "$ROOT/scripts/evaluate_all.sh"
    ;;
  all)
    require_training_approval
    PIPELINE_CONFIG="$PIPELINE_CONFIG" "$0" preflight
    stage1="$(stage1_config)"
    run_stage "$stage1" smoke
    require_smoke_success stage1 "$stage1"
    run_stage "$stage1" train
    stage2="$(stage2_config)"
    run_stage "$stage2" smoke
    require_smoke_success stage2 "$stage2"
    run_stage "$stage2" train
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
