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
  PYTHON_BIN MODEL_PATH EAGLE_ROOT DATA_ROOT TRAIN_DATA VALIDATION_DATA
  OUTPUT_DIR SMOKE_DIR GPU_IDS GLOBAL_BATCH
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "RUN_CONFIG is missing $name" >&2
    exit 2
  fi
done
for path in "$PYTHON_BIN" "$MODEL_PATH" "$EAGLE_ROOT" "$DATA_ROOT" \
  "$TRAIN_DATA" "$VALIDATION_DATA"; do
  [[ -e "$path" ]] || { echo "Missing configured path: $path" >&2; exit 2; }
done

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
WORLD_SIZE="${#GPUS[@]}"
if (( WORLD_SIZE < 1 || GLOBAL_BATCH % WORLD_SIZE != 0 )); then
  echo "GLOBAL_BATCH=$GLOBAL_BATCH must be divisible by GPU count $WORLD_SIZE" >&2
  exit 2
fi
GRADIENT_ACCUMULATION=$((GLOBAL_BATCH / WORLD_SIZE))

export PYTHONPATH="$ROOT/src:$EAGLE_ROOT:${PYTHONPATH:-}"
export LOCANY_STRICT_COVERAGE=1
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$ROOT/.cache/torch_extensions}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$ROOT/.cache/triton}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
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
    --train-data "$TRAIN_DATA"
    --validation-data "$VALIDATION_DATA"
    --expected-train-records "${EXPECTED_TRAIN_RECORDS:-0}"
    --expected-validation-records "${EXPECTED_VALIDATION_RECORDS:-0}"
    --lora-rank "${LORA_RANK:-16}"
    --image-token-limit "${IMAGE_TOKEN_LIMIT:-1024}"
    --max-sequence "${MAX_SEQUENCE:-8192}"
    --visual-context "${VISUAL_CONTEXT:-pixel_reencoded}"
    --magnified-roi-pixels "${MAGNIFIED_ROI_PIXELS:-380}"
    --magnified-roi-stride "${MAGNIFIED_ROI_STRIDE:-1}"
    --gradient-accumulation "$GRADIENT_ACCUMULATION"
    --allowed-padding-records "${ALLOWED_PADDING_RECORDS:-0}"
    --learning-rate "${LEARNING_RATE:-1e-5}"
    --checkpoint-steps "${CHECKPOINT_STEPS:-500}"
    --eval-steps "${EVAL_STEPS:-500}"
    --validation-records "${VALIDATION_RECORDS:-0}"
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
  [[ -n "$smoke_flag" ]] && warmup_steps=1
  common_args
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
      --jsonl "$TRAIN_DATA" "$VALIDATION_DATA" \
      --data-root "$DATA_ROOT" \
      --report "${AUDIT_REPORT:-$ROOT/runs/magnified_preprojector_audit.json}" \
      --exact-loader \
      --model "$MODEL_PATH" \
      --eagle-root "$EAGLE_ROOT" \
      --image-token-limit "${IMAGE_TOKEN_LIMIT:-1024}" \
      --max-sequence "${MAX_SEQUENCE:-8192}" \
      --visual-context "${VISUAL_CONTEXT:-pixel_reencoded}" \
      --magnified-roi-pixels "${MAGNIFIED_ROI_PIXELS:-380}" \
      --magnified-roi-stride "${MAGNIFIED_ROI_STRIDE:-1}"
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
    ;;
  smoke)
    require_gpus_free
    destination="$SMOKE_DIR/$(date +%Y%m%d_%H%M%S)"
    run_torch "$destination" "${SMOKE_STEPS:-2}" smoke
    [[ -f "$destination/done.json" ]] || {
      echo "Smoke failed to produce done.json: $destination" >&2
      exit 1
    }
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
