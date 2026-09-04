#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_CONFIG="${RUN_CONFIG:-$ROOT/configs/large_scale.env}"
MODE="${1:-check}"

if [[ ! -f "$RUN_CONFIG" ]]; then
  echo "Missing RUN_CONFIG: $RUN_CONFIG" >&2
  echo "Copy configs/large_scale.env.example and edit its absolute paths." >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "$RUN_CONFIG"
set +a

required=(
  PYTHON_BIN MODEL_PATH EAGLE_ROOT DATA_ROOT
  OUTPUT_DIR SMOKE_DIR GPU_IDS GLOBAL_BATCH
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "RUN_CONFIG is missing $name" >&2
    exit 2
  fi
done
for path in "$PYTHON_BIN" "$MODEL_PATH" "$EAGLE_ROOT" "$DATA_ROOT"; do
  [[ -e "$path" ]] || { echo "Missing configured path: $path" >&2; exit 2; }
done

if [[ -n "${ARCHITECTURE_CONTRACT:-}" ]]; then
  "$PYTHON_BIN" - "$DATA_ROOT/manifest.json" \
    "$ARCHITECTURE_CONTRACT" "${VISUAL_CONTEXT:-}" \
    "${SOURCE_CROP_SIDE:-}" "${LOCAL_INPUT_SIDE:-}" \
    "${MAGNIFIED_ROI_PIXELS:-}" "${IMAGE_TOKEN_LIMIT:-}" \
    "${MAX_SEQUENCE:-}" "${LOSS_BALANCING:-}" <<'PY'
import json, sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
if not manifest_path.is_file():
    raise SystemExit(f"Missing dataset manifest: {manifest_path}")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
expected = {
    "architecture": "pixel_crop_144to384_v2",
    "visual_context": "pixel_reencoded",
    "source_crop": "144",
    "local_input": "384",
    "magnified_roi": "384",
    "image_tokens": "6000",
    "sequence": "32768",
    "loss_balancing": "source_query_task",
}
actual = dict(zip(expected, sys.argv[2:]))
if actual != expected:
    raise SystemExit(f"Full-scale architecture contract mismatch: {actual} != {expected}")
if manifest.get("schema_version") != "pixel-pivr-hf-hbb-magnified-v2":
    raise SystemExit(f"Wrong dataset schema: {manifest.get('schema_version')!r}")
crop = manifest.get("pixel_reentry") or {}
if crop.get("source_crop_side_pixels") != 144 or crop.get("local_input_side_pixels") != 384:
    raise SystemExit(f"Wrong dataset crop contract: {crop}")
if crop.get("fallback_edge_margin_pixels") != 2.0:
    raise SystemExit(f"Wrong dataset fallback edge margin: {crop}")
PY
fi

load_recipe_annotations() {
  local recipe="$1"
  "$PYTHON_BIN" - "$recipe" "$DATA_ROOT" <<'PY'
import json
import sys
from pathlib import Path

recipe = Path(sys.argv[1])
root = Path(sys.argv[2])
payload = json.loads(recipe.read_text(encoding="utf-8"))
paths = payload.get("annotation")
if not isinstance(paths, list) or not paths:
    raise SystemExit(f"Recipe must contain a non-empty annotation list: {recipe}")
for value in paths:
    path = Path(value)
    print(path if path.is_absolute() else root / path)
PY
}

if [[ -n "${TRAIN_RECIPE:-}" ]]; then
  [[ -f "$TRAIN_RECIPE" ]] || { echo "Missing TRAIN_RECIPE: $TRAIN_RECIPE" >&2; exit 2; }
  mapfile -t TRAIN_DATA_ARGS < <(load_recipe_annotations "$TRAIN_RECIPE")
elif [[ -n "${TRAIN_DATA:-}" ]]; then
  IFS=':' read -r -a TRAIN_DATA_ARGS <<< "$TRAIN_DATA"
else
  echo "Set TRAIN_RECIPE or TRAIN_DATA" >&2
  exit 2
fi

if [[ -n "${VALIDATION_RECIPE:-}" ]]; then
  [[ -f "$VALIDATION_RECIPE" ]] || { echo "Missing VALIDATION_RECIPE: $VALIDATION_RECIPE" >&2; exit 2; }
  mapfile -t VALIDATION_DATA_ARGS < <(load_recipe_annotations "$VALIDATION_RECIPE")
elif [[ -n "${VALIDATION_DATA:-}" ]]; then
  IFS=':' read -r -a VALIDATION_DATA_ARGS <<< "$VALIDATION_DATA"
else
  echo "Set VALIDATION_RECIPE or VALIDATION_DATA" >&2
  exit 2
fi

for path in "${TRAIN_DATA_ARGS[@]}" "${VALIDATION_DATA_ARGS[@]}"; do
  [[ -f "$path" ]] || { echo "Missing annotation shard: $path" >&2; exit 2; }
done

count_records() {
  "$PYTHON_BIN" - "$@" <<'PY'
import sys
from pathlib import Path

total = 0
for value in sys.argv[1:]:
    with Path(value).open("rb") as handle:
        total += sum(1 for line in handle if line.strip())
print(total)
PY
}

TRAIN_RECORDS="$(count_records "${TRAIN_DATA_ARGS[@]}")"
VALIDATION_RECORD_COUNT="$(count_records "${VALIDATION_DATA_ARGS[@]}")"
if [[ "${EXPECTED_TRAIN_RECORDS:-0}" != "0" && \
      "$TRAIN_RECORDS" != "$EXPECTED_TRAIN_RECORDS" ]]; then
  echo "Train count changed: $TRAIN_RECORDS != $EXPECTED_TRAIN_RECORDS" >&2
  exit 2
fi
if [[ "${EXPECTED_VALIDATION_RECORDS:-0}" != "0" && \
      "$VALIDATION_RECORD_COUNT" != "$EXPECTED_VALIDATION_RECORDS" ]]; then
  echo "Validation count changed: $VALIDATION_RECORD_COUNT != $EXPECTED_VALIDATION_RECORDS" >&2
  exit 2
fi

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
WORLD_SIZE="${#GPUS[@]}"
if (( WORLD_SIZE < 1 || GLOBAL_BATCH % WORLD_SIZE != 0 )); then
  echo "GLOBAL_BATCH=$GLOBAL_BATCH must be divisible by GPU count $WORLD_SIZE" >&2
  exit 2
fi
GRADIENT_ACCUMULATION=$((GLOBAL_BATCH / WORLD_SIZE))
PADDING_RECORDS="${ALLOWED_PADDING_RECORDS:-auto}"
if [[ "$PADDING_RECORDS" == "auto" ]]; then
  PADDING_RECORDS=$(( (GLOBAL_BATCH - TRAIN_RECORDS % GLOBAL_BATCH) % GLOBAL_BATCH ))
elif [[ ! "$PADDING_RECORDS" =~ ^[0-9]+$ ]]; then
  echo "ALLOWED_PADDING_RECORDS must be auto or a non-negative integer" >&2
  exit 2
fi
ONE_PASS_STEPS=$(( (TRAIN_RECORDS + PADDING_RECORDS) / GLOBAL_BATCH ))

export PYTHONPATH="$ROOT/src:$EAGLE_ROOT:${PYTHONPATH:-}"
export LOCANY_STRICT_COVERAGE=1
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$ROOT/.cache/torch_extensions}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$ROOT/.cache/triton}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
mkdir -p "$HF_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$ROOT/runs/logs"

gpu_uuid() {
  local gpu="$1"
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits |
    awk -F', ' -v wanted="$gpu" '$1 == wanted {print $2}'
}

gpu_has_compute() {
  local uuid
  uuid="$(gpu_uuid "$1")"
  [[ -n "$uuid" ]] || return 0
  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits 2>/dev/null |
    awk -F', ' -v wanted="$uuid" '$1 == wanted {found=1} END {exit !found}'
}

require_gpus_free() {
  local gpu busy=0
  for gpu in "${GPUS[@]}"; do
    if gpu_has_compute "$gpu"; then
      echo "GPU $gpu already has a compute process" >&2
      busy=1
    fi
  done
  (( busy == 0 ))
}

common_args() {
  TRAIN_ARGS=(
    --model "$MODEL_PATH"
    --eagle-root "$EAGLE_ROOT"
    --data-root "$DATA_ROOT"
    --train-data "${TRAIN_DATA_ARGS[@]}"
    --validation-data "${VALIDATION_DATA_ARGS[@]}"
    --expected-train-records "${EXPECTED_TRAIN_RECORDS:-0}"
    --expected-validation-records "${EXPECTED_VALIDATION_RECORDS:-0}"
    --lora-rank "${LORA_RANK:-16}"
    --loss-balancing "${LOSS_BALANCING:-none}"
    --image-token-limit "${IMAGE_TOKEN_LIMIT:-1024}"
    --max-sequence "${MAX_SEQUENCE:-32768}"
    --visual-context "${VISUAL_CONTEXT:-pixel_reencoded}"
    --magnified-roi-pixels "${MAGNIFIED_ROI_PIXELS:-384}"
    --magnified-roi-stride "${MAGNIFIED_ROI_STRIDE:-1}"
    --multiscale-roi-pixels "${MULTISCALE_ROI_PIXELS:-196,378,756}"
    --multiscale-target-patches "${MULTISCALE_TARGET_PATCHES:-27}"
    --multiscale-fusion-hidden "${MULTISCALE_FUSION_HIDDEN:-128}"
    --multiscale-preferred-scale "${MULTISCALE_PREFERRED_SCALE:-1}"
    --gradient-accumulation "$GRADIENT_ACCUMULATION"
    --allowed-padding-records "$PADDING_RECORDS"
    --sample-order "${SAMPLE_ORDER:-shuffled}"
    --learning-rate "${LEARNING_RATE:-1e-5}"
    --weight-decay "${WEIGHT_DECAY:-0.01}"
    --max-grad-norm "${MAX_GRAD_NORM:-1.0}"
    --checkpoint-steps "${CHECKPOINT_STEPS:-500}"
    --keep-recent-checkpoints "${KEEP_RECENT_CHECKPOINTS:-1}"
    --eval-steps "${EVAL_STEPS:-500}"
    --log-steps "${LOG_STEPS:-50}"
    --validation-records "${VALIDATION_RECORDS:-0}"
    --validation-sampling "${VALIDATION_SAMPLING:-stratified_files}"
    --workers "${WORKERS:-2}"
    --seed "${SEED:-20260901}"
    --vision-attention "${VISION_ATTENTION:-auto}"
  )
  if [[ "${SERIAL_MODEL_LOAD:-1}" == "1" ]]; then
    TRAIN_ARGS+=(--serial-model-load)
  else
    TRAIN_ARGS+=(--no-serial-model-load)
  fi
  if [[ -n "${INIT_ADAPTER:-}" ]]; then
    TRAIN_ARGS+=(--init-adapter "$INIT_ADAPTER")
  fi
  if [[ -n "${WANDB_PROJECT:-}" ]]; then
    TRAIN_ARGS+=(--wandb-project "$WANDB_PROJECT")
    [[ -n "${WANDB_NAME:-}" ]] && TRAIN_ARGS+=(--wandb-name "$WANDB_NAME")
    [[ -n "${WANDB_RUN_ID:-}" ]] && TRAIN_ARGS+=(--wandb-run-id "$WANDB_RUN_ID")
  fi
}

run_torch() {
  local destination="$1" steps="$2" smoke_flag="$3"
  local warmup_steps="${WARMUP_STEPS:-100}"
  local saved_wandb_project="${WANDB_PROJECT:-}"
  local saved_validation_records="${VALIDATION_RECORDS:-0}"
  local saved_expected_train_records="${EXPECTED_TRAIN_RECORDS:-0}"
  local saved_padding_records="$PADDING_RECORDS"
  local saved_train_records="$TRAIN_RECORDS"
  local -a saved_train_data_args=("${TRAIN_DATA_ARGS[@]}")
  [[ -n "$smoke_flag" ]] && warmup_steps=1
  # Smoke runs use timestamped output directories and must not append their
  # diagnostic steps to the stable W&B run reserved for the complete experiment.
  [[ -n "$smoke_flag" ]] && WANDB_PROJECT=""
  [[ -n "$smoke_flag" ]] && VALIDATION_RECORDS="${SMOKE_VALIDATION_RECORDS:-8}"
  if [[ -n "$smoke_flag" ]]; then
    local smoke_records=$((steps * GLOBAL_BATCH))
    local smoke_recipe_root="$destination/hard_smoke_data"
    mkdir -p "$smoke_recipe_root"
    "$PYTHON_BIN" "$ROOT/tools/build_hard_smoke_recipe.py" \
      --jsonl "${TRAIN_DATA_ARGS[@]}" \
      --output "$smoke_recipe_root" \
      --records "$smoke_records"
    mapfile -t TRAIN_DATA_ARGS < <(
      load_recipe_annotations "$smoke_recipe_root/hard_smoke_recipe.json"
    )
    TRAIN_RECORDS="$smoke_records"
    EXPECTED_TRAIN_RECORDS="$smoke_records"
    PADDING_RECORDS=0
  fi
  common_args
  WANDB_PROJECT="$saved_wandb_project"
  VALIDATION_RECORDS="$saved_validation_records"
  EXPECTED_TRAIN_RECORDS="$saved_expected_train_records"
  PADDING_RECORDS="$saved_padding_records"
  TRAIN_RECORDS="$saved_train_records"
  TRAIN_DATA_ARGS=("${saved_train_data_args[@]}")
  TRAIN_ARGS+=(
    --output "$destination"
    --max-steps "$steps"
    --warmup-steps "$warmup_steps"
  )
  [[ -n "$smoke_flag" ]] && TRAIN_ARGS+=(--smoke)
  CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON_BIN" -m torch.distributed.run \
    --standalone --nproc_per_node "$WORLD_SIZE" \
    -m pixel_pivr.train "${TRAIN_ARGS[@]}"
}

case "$MODE" in
  audit)
    AUDIT_ARGS=(
      --jsonl "${TRAIN_DATA_ARGS[@]}" "${VALIDATION_DATA_ARGS[@]}" \
      --data-root "$DATA_ROOT" \
      --report "${AUDIT_REPORT:-$ROOT/runs/magnified_preprojector_audit.json}" \
      --exact-loader \
      --model "$MODEL_PATH" \
      --eagle-root "$EAGLE_ROOT" \
      --image-token-limit "${IMAGE_TOKEN_LIMIT:-1024}" \
      --max-sequence "${MAX_SEQUENCE:-32768}" \
      --visual-context "${VISUAL_CONTEXT:-pixel_reencoded}" \
      --magnified-roi-pixels "${MAGNIFIED_ROI_PIXELS:-384}" \
      --magnified-roi-stride "${MAGNIFIED_ROI_STRIDE:-1}"
      --multiscale-target-patches "${MULTISCALE_TARGET_PATCHES:-27}"
    )
    if [[ -n "${HOLDOUT_HASHES:-}" ]]; then
      [[ -f "$HOLDOUT_HASHES" ]] || {
        echo "Missing HOLDOUT_HASHES: $HOLDOUT_HASHES" >&2
        exit 2
      }
      AUDIT_ARGS+=(--holdout-hashes "$HOLDOUT_HASHES")
    fi
    "$PYTHON_BIN" -m pixel_pivr.audit "${AUDIT_ARGS[@]}"
    ;;
  check)
    "$PYTHON_BIN" "$ROOT/tools/verify_release.py"
    require_gpus_free
    echo "Pixel-PIVR inputs are present; $WORLD_SIZE GPUs are free."
    echo "Global records/update: $GLOBAL_BATCH; accumulation/rank: $GRADIENT_ACCUMULATION"
    echo "Train records: $TRAIN_RECORDS; explicit padding: $PADDING_RECORDS; one-pass steps: $ONE_PASS_STEPS"
    echo "Loss balancing: ${LOSS_BALANCING:-none}"
    echo "Validation records: $VALIDATION_RECORD_COUNT; monitor: ${VALIDATION_RECORDS:-0} (${VALIDATION_SAMPLING:-stratified_files})"
    ;;
  smoke)
    require_gpus_free
    destination="$SMOKE_DIR/$(date +%Y%m%d_%H%M%S)"
    run_torch "$destination" "${SMOKE_STEPS:-2}" smoke
    [[ -f "$destination/done.json" ]] || {
      echo "Smoke failed to produce done.json: $destination" >&2
      exit 1
    }
    "$PYTHON_BIN" - "$RUN_CONFIG" "$destination/done.json" \
      "$SMOKE_DIR/latest_success.json" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

config, done, output = map(Path, sys.argv[1:])
payload = {
    "schema_version": "pixel-pivr-smoke-receipt-v1",
    "run_config": str(config.resolve()),
    "run_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
    "done_json": str(done.resolve()),
}
output.parent.mkdir(parents=True, exist_ok=True)
temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
temporary.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
os.replace(temporary, output)
PY
    echo "Smoke passed: $destination"
    ;;
  train)
    require_gpus_free
    log="$ROOT/runs/logs/$(basename "$OUTPUT_DIR").log"
    echo "Training log: $log"
    run_torch "$OUTPUT_DIR" "${MAX_STEPS:-0}" "" 2>&1 | tee -a "$log"
    ;;
  status)
    for path in "$OUTPUT_DIR/run_contract.json" "$OUTPUT_DIR/status.json" \
      "$OUTPUT_DIR/interrupted.json" "$OUTPUT_DIR/done.json"; do
      if [[ -f "$path" ]]; then
        echo "== $path"
        sed -n '1,220p' "$path"
      fi
    done
    ;;
  *)
    echo "Usage: RUN_CONFIG=/path/config.env $0 {audit|check|smoke|train|status}" >&2
    exit 2
    ;;
esac
