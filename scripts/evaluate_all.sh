#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPELINE_CONFIG="${PIPELINE_CONFIG:-$ROOT/configs/full_scale.env}"
[[ -f "$PIPELINE_CONFIG" ]] || { echo "Missing PIPELINE_CONFIG: $PIPELINE_CONFIG" >&2; exit 2; }
set -a
# shellcheck disable=SC1090
source "$PIPELINE_CONFIG"
set +a

for name in PYTHON_BIN MODEL_PATH DATA_ROOT RUN_ROOT; do
  [[ -n "${!name:-}" ]] || { echo "Missing $name in $PIPELINE_CONFIG" >&2; exit 2; }
done
if [[ -z "${EVAL_ADAPTER:-}" ]]; then
  [[ -f "$RUN_ROOT/stage2_dense_balanced/done.json" ]] || {
    echo "Default Stage 2 evaluation requires a completed run. Set EVAL_ADAPTER explicitly for a diagnostic checkpoint." >&2
    exit 2
  }
  ADAPTER="$RUN_ROOT/stage2_dense_balanced/best.pt"
else
  ADAPTER="$EVAL_ADAPTER"
fi
[[ -e "$ADAPTER" ]] || { echo "Missing evaluation adapter: $ADAPTER" >&2; exit 2; }
MANIFEST_ROOT="$RUN_ROOT/evaluation_manifests"
ADAPTER_TAG="$(basename "$(readlink -f "$ADAPTER")" .pt)"
[[ "${ARCHITECTURE_CONTRACT:-}" == "pixel_crop_144to384_v2" ]] || {
  echo "Evaluation requires ARCHITECTURE_CONTRACT=pixel_crop_144to384_v2" >&2
  exit 2
}
[[ "${EVAL_CROP_SIDE:-}" == "144" && "${EVAL_LOCAL_RESIZE_SIDE:-}" == "384" ]] || {
  echo "Evaluation must use EVAL_CROP_SIDE=144 and EVAL_LOCAL_RESIZE_SIDE=384" >&2
  exit 2
}
[[ "${EVAL_GEOMETRY_PREFIX_MODE:-}" == "box_only" && "${EVAL_GLOBAL_FALLBACK:-0}" == "1" ]] || {
  echo "Magnified-v2 requires box-only PBD6 and observable global fallback" >&2
  exit 2
}
EVAL_TAG="${ADAPTER_TAG}_it${IMAGE_TOKEN_LIMIT:-6000}_${VISUAL_CONTEXT:-pixel_reencoded}_${EVAL_PREFIX_CACHE_MODE:-shared}_wave${EVAL_WAVE_SIZE:-200}_crop144to384_boxonly_fallback"
OUTPUT_ROOT="$RUN_ROOT/evaluation/$EVAL_TAG"
mkdir -p "$MANIFEST_ROOT" "$OUTPUT_ROOT/logs"

"$PYTHON_BIN" "$ROOT/tools/prepare_evaluation.py" \
  --data-root "$DATA_ROOT" --output "$MANIFEST_ROOT"

IFS=',' read -r -a GPUS <<< "${EVAL_GPU_IDS:-${GPU_IDS:-0}}"
(( ${#GPUS[@]} > 0 )) || { echo "EVAL_GPU_IDS is empty" >&2; exit 2; }

gpu_uuid() {
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits |
    awk -F', ' -v wanted="$1" '$1 == wanted {print $2}'
}
for gpu in "${GPUS[@]}"; do
  uuid="$(gpu_uuid "$gpu")"
  [[ -n "$uuid" ]] || { echo "Unknown GPU index: $gpu" >&2; exit 2; }
  if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits 2>/dev/null |
      grep -Fxq "$uuid"; then
    echo "GPU $gpu already has a compute process" >&2
    exit 2
  fi
done

SPECS=(
  "detection_DIOR"
  "detection_DOTAv2"
  "grounding_DIOR-RSVG"
  "grounding_VRSBench-VG"
  "pointing_DOTAv2-Balanced100"
)

run_shard() {
  local spec="$1" gpu="$2" manifest="$3" shard_index="$4"
  local output="$OUTPUT_ROOT/$spec/shards/shard-$shard_index"
  local log="$OUTPUT_ROOT/logs/${spec}_shard-${shard_index}.log"
  if [[ -f "$output/summary.json" ]]; then
    echo "SKIP complete: $spec shard $shard_index"
    return 0
  fi
  mkdir -p "$output"
  local none_arg=()
  [[ "${EVAL_ALLOW_NONE:-1}" == "1" ]] && none_arg+=(--allow-none)
  local fallback_arg=()
  [[ "${EVAL_GLOBAL_FALLBACK:-0}" == "1" ]] && fallback_arg+=(--global-fallback)
  echo "START $spec shard $shard_index on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m pixel_pivr.infer \
    --model "$MODEL_PATH" \
    --adapter "$ADAPTER" \
    --manifest "$manifest" \
    --data-root "$DATA_ROOT" \
    --output "$output" \
    --device cuda:0 \
    --dtype bfloat16 \
    --wave-size "${EVAL_WAVE_SIZE:-200}" \
    --prefix-cache-mode "${EVAL_PREFIX_CACHE_MODE:-shared}" \
    --geometry-prefix-mode "${EVAL_GEOMETRY_PREFIX_MODE:-box_only}" \
    --crop-side "${EVAL_CROP_SIDE:-144}" \
    --local-resize-side "${EVAL_LOCAL_RESIZE_SIDE:-384}" \
    --fallback-edge-margin "${EVAL_FALLBACK_EDGE_MARGIN:-2}" \
    --image-token-limit "${IMAGE_TOKEN_LIMIT:-6000}" \
    --visual-context "${VISUAL_CONTEXT:-preprojector_magnified_roi}" \
    --point-address-prompt-schema "${POINT_ADDRESS_PROMPT_SCHEMA:-compact}" \
    --magnified-roi-pixels "${MAGNIFIED_ROI_PIXELS:-384}" \
    --magnified-roi-stride "${MAGNIFIED_ROI_STRIDE:-1}" \
    --point-max-new-tokens "${EVAL_POINT_MAX_NEW_TOKENS:-4096}" \
    --seed "${SEED:-20260901}" \
    --resume \
    "${fallback_arg[@]}" \
    "${none_arg[@]}" 2>&1 | tee -a "$log"
  echo "COMPLETE $spec shard $shard_index"
}

for spec in "${SPECS[@]}"; do
  manifest="$MANIFEST_ROOT/$spec.jsonl"
  output="$OUTPUT_ROOT/$spec"
  if [[ -f "$output/summary.json" ]]; then
    echo "SKIP complete merged benchmark: $spec"
    continue
  fi
  shard_root="$MANIFEST_ROOT/shards/$spec"
  "$PYTHON_BIN" "$ROOT/tools/shard_evaluation_manifest.py" \
    --input "$manifest" --output-dir "$shard_root" --shards "${#GPUS[@]}"
  mapfile -t shard_manifests < <(find "$shard_root" -maxdepth 1 -type f -name 'part-*.jsonl' | sort)
  (( ${#shard_manifests[@]} > 0 )) || { echo "No shards generated for $spec" >&2; exit 2; }
  pids=()
  labels=()
  for index in "${!shard_manifests[@]}"; do
    shard_index="$(printf '%05d' "$index")"
    run_shard "$spec" "${GPUS[index]}" "${shard_manifests[index]}" "$shard_index" &
    pids+=("$!")
    labels+=("$spec/$shard_index")
  done
  failed=0
  for index in "${!pids[@]}"; do
    if ! wait "${pids[index]}"; then
      echo "FAILED ${labels[index]}" >&2
      failed=1
    fi
  done
  (( failed == 0 )) || exit 1
  expected_records="$(awk 'NF {count++} END {print count+0}' "$manifest")"
  "$PYTHON_BIN" "$ROOT/tools/merge_inference_shards.py" \
    --shard-root "$output/shards" --output "$output" \
    --expected-records "$expected_records"
done

"$PYTHON_BIN" - "$OUTPUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summaries = {}
for path in sorted(root.glob("*/summary.json")):
    summaries[path.parent.name] = json.loads(path.read_text(encoding="utf-8"))
destination = root / "all_metrics.json"
destination.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"Wrote {destination}")
PY
