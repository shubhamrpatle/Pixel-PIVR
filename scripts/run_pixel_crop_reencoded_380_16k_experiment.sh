#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RUN_CONFIG="${RUN_CONFIG:-$ROOT/configs/pixel_crop_reencoded_380_16k.env}"
MODE="${1:-check}"

[[ -f "$RUN_CONFIG" ]] || {
  echo "Missing RUN_CONFIG: $RUN_CONFIG" >&2
  echo "Copy configs/pixel_crop_reencoded_380_16k.env.example and set absolute paths." >&2
  exit 2
}
set -a
# shellcheck disable=SC1090
source "$RUN_CONFIG"
set +a

required=(
  PYTHON_BIN DATA_PREP_SCRIPT COMPACT_PREP_SCRIPT SOURCE_DATASET ORIGINAL_DATASET
  CROP_DATASET CROP_MEDIA_ROOT PREPARED_DATASET HOLDOUT_HASHES
  DATASET_AUDIT_REPORT CROP_SIDE
)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { echo "RUN_CONFIG is missing $name" >&2; exit 2; }
done
for path in "$PYTHON_BIN" "$DATA_PREP_SCRIPT" "$COMPACT_PREP_SCRIPT" \
  "$SOURCE_DATASET" "$ORIGINAL_DATASET" "$HOLDOUT_HASHES"; do
  [[ -e "$path" ]] || { echo "Missing preparation input: $path" >&2; exit 2; }
done
[[ "$CROP_SIDE" == "380" ]] || {
  echo "This controlled experiment requires CROP_SIDE=380, found $CROP_SIDE" >&2
  exit 2
}

LOG_ROOT="$ROOT/runs/logs"
mkdir -p "$LOG_ROOT" "$(dirname "$DATASET_AUDIT_REPORT")"

prepare_data() {
  echo "Preparing exact 380x380 lossless local crops."
  "$PYTHON_BIN" "$DATA_PREP_SCRIPT" \
    --source "$SOURCE_DATASET" \
    --original "$ORIGINAL_DATASET" \
    --output "$CROP_DATASET" \
    --media-root "$CROP_MEDIA_ROOT" \
    --crop-side "$CROP_SIDE" \
    --require-exact-crop-size \
    >"$LOG_ROOT/pixel_crop_reencoded_380_prepare_crops.log" 2>&1

  echo "Converting Round 2 to the compact point-containment prompt."
  "$PYTHON_BIN" "$COMPACT_PREP_SCRIPT" \
    --source "$CROP_DATASET" \
    --output "$PREPARED_DATASET" \
    --seed "${SEED:-20260902}" \
    >"$LOG_ROOT/pixel_crop_reencoded_380_prepare_compact.log" 2>&1
}

verify_data() {
  "$PYTHON_BIN" "$ROOT/tools/verify_pixel_crop_reencoded_dataset.py" \
    --crop-dataset "$CROP_DATASET" \
    --compact-dataset "$PREPARED_DATASET" \
    --holdout-hashes "$HOLDOUT_HASHES" \
    --report "$DATASET_AUDIT_REPORT" \
    --crop-side "$CROP_SIDE" \
    --expected-train "${EXPECTED_TRAIN_RECORDS:-16000}" \
    --expected-validation "${EXPECTED_VALIDATION_RECORDS:-1000}"
}

case "$MODE" in
  prepare)
    prepare_data
    verify_data
    ;;
  verify-data)
    verify_data
    ;;
  audit)
    verify_data
    bash "$ROOT/scripts/train_distributed.sh" audit
    ;;
  check)
    verify_data
    bash "$ROOT/scripts/train_distributed.sh" check
    ;;
  smoke)
    verify_data
    bash "$ROOT/scripts/train_distributed.sh" smoke
    ;;
  train)
    verify_data
    bash "$ROOT/scripts/train_distributed.sh" train
    ;;
  evaluate)
    verify_data
    bash "$ROOT/scripts/evaluate_pixel_crop_reencoded_380_balanced100.sh" run
    ;;
  preflight)
    prepare_data
    verify_data
    bash "$ROOT/scripts/train_distributed.sh" audit
    bash "$ROOT/scripts/train_distributed.sh" check
    bash "$ROOT/scripts/train_distributed.sh" smoke
    ;;
  all)
    prepare_data
    verify_data
    bash "$ROOT/scripts/train_distributed.sh" audit
    bash "$ROOT/scripts/train_distributed.sh" check
    bash "$ROOT/scripts/train_distributed.sh" smoke
    bash "$ROOT/scripts/train_distributed.sh" train
    bash "$ROOT/scripts/evaluate_pixel_crop_reencoded_380_balanced100.sh" run
    ;;
  status)
    bash "$ROOT/scripts/train_distributed.sh" status
    bash "$ROOT/scripts/evaluate_pixel_crop_reencoded_380_balanced100.sh" status
    ;;
  *)
    echo "Usage: $0 {prepare|verify-data|audit|check|smoke|train|evaluate|preflight|all|status}" >&2
    exit 2
    ;;
esac
