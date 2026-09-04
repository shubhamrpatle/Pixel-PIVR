#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_CONFIG="${RUN_CONFIG:-$ROOT/configs/pixel_crop_reencoded_380_16k.env}"
MODE="${1:-check}"

[[ -f "$RUN_CONFIG" ]] || { echo "Missing RUN_CONFIG: $RUN_CONFIG" >&2; exit 2; }
set -a
# shellcheck disable=SC1090
source "$RUN_CONFIG"
set +a

ADAPTER="${EVAL_ADAPTER:-$OUTPUT_DIR/best.pt}"
required=(
  PYTHON_BIN MODEL_PATH EAGLE_ROOT DATA_ROOT OUTPUT_DIR CROP_DATASET
  PREPARED_DATASET CROP_SIDE EVAL_MANIFEST EVAL_OUTPUT EVAL_GPU_IDS
)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { echo "RUN_CONFIG is missing $name" >&2; exit 2; }
done
for path in "$PYTHON_BIN" "$MODEL_PATH" "$EAGLE_ROOT" "$DATA_ROOT" \
  "$ADAPTER" "$EVAL_MANIFEST" "$CROP_DATASET/manifest.json" \
  "$PREPARED_DATASET/manifest.json"; do
  [[ -e "$path" ]] || { echo "Missing evaluation input: $path" >&2; exit 2; }
done

IFS=',' read -r -a GPUS <<< "$EVAL_GPU_IDS"
(( ${#GPUS[@]} >= 2 )) || { echo "EVAL_GPU_IDS requires two GPUs" >&2; exit 2; }
export PYTHONPATH="$ROOT/src:$EAGLE_ROOT:${PYTHONPATH:-}"
mkdir -p "$EVAL_OUTPUT/logs"

gpu_uuid() {
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits |
    awk -F', ' -v wanted="$1" '$1 == wanted {print $2}'
}

require_gpu_free() {
  local uuid
  uuid="$(gpu_uuid "$1")"
  [[ -n "$uuid" ]] || { echo "Unknown GPU $1" >&2; exit 2; }
  if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits 2>/dev/null |
      grep -Fxq "$uuid"; then
    echo "GPU $1 already has a compute process" >&2
    exit 2
  fi
}

verify() {
  [[ -f "$OUTPUT_DIR/done.json" ]] || {
    echo "Training is not complete: $OUTPUT_DIR/done.json" >&2
    exit 2
  }
  "$PYTHON_BIN" - "$OUTPUT_DIR/done.json" "$ADAPTER" "$EVAL_MANIFEST" \
    "$CROP_DATASET/manifest.json" "$PREPARED_DATASET/manifest.json" "$CROP_SIDE" <<'PY'
import json
import sys
from pathlib import Path

done_path, adapter, eval_manifest, crop_manifest, compact_manifest = map(
    Path, sys.argv[1:6]
)
crop_side = int(sys.argv[6])
done = json.loads(done_path.read_text(encoding="utf-8"))
assert done["complete_one_pass"] is True
assert done["steps"] == 4000
assert done["record_exposures"] == 16000
assert done["padding_record_exposures"] == 0
assert adapter.resolve() == Path(done["best"]).resolve()

crop = json.loads(crop_manifest.read_text(encoding="utf-8"))
compact = json.loads(compact_manifest.read_text(encoding="utf-8"))
assert crop["crop_side"] == crop_side == 380
assert crop["require_exact_crop_size"] is True
assert compact["arms"]["global_local"]["train"]["records"] == 16000

rows = [json.loads(line) for line in eval_manifest.read_text(encoding="utf-8").splitlines() if line]
assert len(rows) == 100
assert len({row["sample_key"] for row in rows}) == 100
assert sum(len(row.get("gt") or []) for row in rows) == 9045

checkpoint = __import__("torch").load(adapter, map_location="cpu", weights_only=False)
config = checkpoint.get("config") or {}
assert config.get("visual_context") == "pixel_reencoded"
assert int(config.get("image_token_limit")) == 6000
assert int(config.get("max_sequence")) == 8192
print(
    f"verified adapter={adapter.resolve()} crop={crop_side} train=16000 "
    "evaluation_images=100 gt=9045 prediction_cap=None"
)
PY
}

run_arm() {
  local name="$1" gpu="$2" wave="$3" destination="$EVAL_OUTPUT/$1"
  mkdir -p "$destination"
  if [[ -f "$destination/summary.json" ]]; then
    echo "SKIP completed $name"
    return
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m pixel_pivr.infer \
    --model "$MODEL_PATH" \
    --adapter "$ADAPTER" \
    --manifest "$EVAL_MANIFEST" \
    --data-root / \
    --output "$destination" \
    --device cuda:0 \
    --dtype bfloat16 \
    --wave-size "$wave" \
    --prefix-cache-mode shared \
    --image-token-limit "${IMAGE_TOKEN_LIMIT:-6000}" \
    --visual-context pixel_reencoded \
    --point-address-prompt-schema compact \
    --crop-side "$CROP_SIDE" \
    --point-max-new-tokens 4096 \
    --nms-iou 0.5 \
    --allow-none \
    --seed 0 \
    --resume
}

summarize() {
  "$PYTHON_BIN" - "$EVAL_OUTPUT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
payload = {
    "experiment": "pixel_crop_reencoded_380_native6000",
    "crop_side": 380,
    "image_token_limit": 6000,
    "prediction_cap": None,
    "arms": {},
}
for name in ("sequential", "wave200"):
    path = root / name / "summary.json"
    if not path.is_file():
        raise SystemExit(f"missing summary: {path}")
    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["images"] == 100 and row["prediction_cap"] is None
    payload["arms"][name] = row
(root / "comparison.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
}

case "$MODE" in
  check)
    verify
    require_gpu_free "${GPUS[0]}"
    require_gpu_free "${GPUS[1]}"
    echo "Pixel-crop re-encoded Balanced-100 evaluation is ready"
    ;;
  run)
    verify
    require_gpu_free "${GPUS[0]}"
    require_gpu_free "${GPUS[1]}"
    run_arm sequential "${GPUS[0]}" 1 >"$EVAL_OUTPUT/logs/sequential.log" 2>&1 & p1=$!
    run_arm wave200 "${GPUS[1]}" "${EVAL_WAVE_SIZE:-200}" >"$EVAL_OUTPUT/logs/wave200.log" 2>&1 & p2=$!
    failed=0
    wait "$p1" || failed=1
    wait "$p2" || failed=1
    (( failed == 0 )) || {
      echo "Evaluation failed; inspect $EVAL_OUTPUT/logs" >&2
      exit 1
    }
    summarize
    ;;
  status)
    for name in sequential wave200; do
      echo "== $name =="
      [[ -f "$EVAL_OUTPUT/$name/summary.json" ]] && cat "$EVAL_OUTPUT/$name/summary.json"
      [[ -f "$EVAL_OUTPUT/$name/predictions.jsonl" ]] && wc -l "$EVAL_OUTPUT/$name/predictions.jsonl"
      tail -n 5 "$EVAL_OUTPUT/logs/$name.log" 2>/dev/null || true
    done
    ;;
  *)
    echo "Usage: $0 {check|run|status}" >&2
    exit 2
    ;;
esac
