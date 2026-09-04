#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_CONFIG="${RUN_CONFIG:-$ROOT/configs/adaptive_multiscale_16k.env}"
MODE="${1:-check}"

[[ -f "$RUN_CONFIG" ]] || { echo "Missing RUN_CONFIG: $RUN_CONFIG" >&2; exit 2; }
set -a
# shellcheck disable=SC1090
source "$RUN_CONFIG"
set +a

ADAPTER="${EVAL_ADAPTER:-$OUTPUT_DIR/best.pt}"
for name in PYTHON_BIN MODEL_PATH DATA_ROOT OUTPUT_DIR EVAL_MANIFEST EVAL_OUTPUT EVAL_GPU_IDS; do
  [[ -n "${!name:-}" ]] || { echo "RUN_CONFIG is missing $name" >&2; exit 2; }
done
for path in "$PYTHON_BIN" "$MODEL_PATH" "$DATA_ROOT" "$ADAPTER" "$EVAL_MANIFEST"; do
  [[ -e "$path" ]] || { echo "Missing evaluation input: $path" >&2; exit 2; }
done

IFS=',' read -r -a GPUS <<< "$EVAL_GPU_IDS"
(( ${#GPUS[@]} >= 2 )) || { echo "EVAL_GPU_IDS needs two GPUs" >&2; exit 2; }
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
  "$PYTHON_BIN" - "$OUTPUT_DIR/done.json" "$ADAPTER" "$EVAL_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

done_path, adapter, manifest = map(Path, sys.argv[1:])
done = json.loads(done_path.read_text(encoding="utf-8"))
assert done["complete_one_pass"] is True
assert done["steps"] == 4000
assert done["record_exposures"] == 16000
assert adapter.resolve() == Path(done["best"]).resolve()
rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
assert len(rows) == 100
assert sum(len(row.get("gt") or []) for row in rows) == 9045
assert len({row["sample_key"] for row in rows}) == 100
labels = {}
for row in rows:
    for target in row.get("gt") or []:
        label = target["label"]
        labels[label] = labels.get(label, 0) + 1
for label, expected in {"small vehicle": 5650, "large vehicle": 108, "plane": 73}.items():
    assert labels.get(label) == expected, (
        f"evaluation manifest has an invalid {label!r} count: "
        f"{labels.get(label, 0)} != {expected}"
    )
assert not labels.get("vehicle") and not labels.get("airplane"), (
    "evaluation manifest contains collapsed DOTAv2 labels"
)
checkpoint = __import__("torch").load(adapter, map_location="cpu", weights_only=False)
config = checkpoint.get("config") or {}
assert config.get("visual_context") == "adaptive_multiscale_preprojector_roi"
controller = (checkpoint.get("trainable_state") or {}).get("controller")
assert isinstance(controller, dict) and controller
print(f"verified adapter={adapter.resolve()} manifest_images=100 gt=9045")
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
    --visual-context adaptive_multiscale_preprojector_roi \
    --point-address-prompt-schema compact \
    --magnified-roi-stride "${MAGNIFIED_ROI_STRIDE:-1}" \
    --multiscale-roi-pixels "${MULTISCALE_ROI_PIXELS:-196,378,756}" \
    --multiscale-target-patches "${MULTISCALE_TARGET_PATCHES:-27}" \
    --multiscale-fusion-hidden "${MULTISCALE_FUSION_HIDDEN:-128}" \
    --multiscale-preferred-scale "${MULTISCALE_PREFERRED_SCALE:-1}" \
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
payload = {}
for name in ("sequential", "wave200"):
    path = root / name / "summary.json"
    if not path.is_file():
        raise SystemExit(f"missing summary: {path}")
    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["images"] == 100 and row["prediction_cap"] is None
    payload[name] = row
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
    echo "Adaptive multi-scale Balanced-100 evaluation is ready"
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
    (( failed == 0 )) || { echo "Evaluation failed; inspect $EVAL_OUTPUT/logs" >&2; exit 1; }
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
  *) echo "Usage: $0 {check|run|status}" >&2; exit 2 ;;
esac
