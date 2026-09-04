#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-install}"
CODE_REVISION="${CODE_REVISION:-$(git -C "$ROOT" rev-parse HEAD)}"
WORK_ROOT="${WORK_ROOT:-$(dirname "$ROOT")/pixel-pivr-assets}"
VENV="${VENV:-$ROOT/.venv}"
PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3.10}"
EAGLE_DIR="${EAGLE_DIR:-$WORK_ROOT/Eagle}"
EAGLE_REVISION="${EAGLE_REVISION:-8442db3b79f7fd2357e468e6eecdd9b6a82049ff}"
MODEL_REPO="${MODEL_REPO:-nvidia/LocateAnything-3B}"
MODEL_REVISION="${MODEL_REVISION:-c32291ca5e996f5a7a485845b4f57a233936bba0}"
MODEL_DIR="${MODEL_DIR:-$WORK_ROOT/LocateAnything-3B}"
DATA_REPO="${DATA_REPO:-shubhampatle/Pixel-PIVR-Magnified-v2}"
DATA_REVISION="${DATA_REVISION:-}"
DATA_DIR="${DATA_DIR:-$WORK_ROOT/Pixel-PIVR-Magnified-v2}"
PATCH="$ROOT/patches/eagle_virtual_crop_v1.patch"
EAGLE_TOOLS_SHA256="0acc434441dd79bbe1890d721df48ba71c7abcb7f7ff7c6cbae93bf34c4a34cc"
EAGLE_TRAINER_SHA256="1108e6f5074e39b971f3da0ee056b0c92ec0b3ab56a80c98ceab2b470119f2e0"

require_data_revision() {
  [[ "$DATA_REVISION" =~ ^[0-9a-f]{40}$ ]] || {
    echo "Set DATA_REVISION to the verified 40-character Hugging Face commit SHA." >&2
    exit 2
  }
}

install_environment() {
  command -v "$PYTHON_BOOTSTRAP" >/dev/null || {
    echo "Missing $PYTHON_BOOTSTRAP; install Python 3.10 first." >&2
    exit 2
  }
  "$PYTHON_BOOTSTRAP" - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"Python 3.10 is required, got {sys.version}")
PY
  "$PYTHON_BOOTSTRAP" -m venv "$VENV"
  "$VENV/bin/python" -m pip install --upgrade pip==25.2 wheel==0.45.1 setuptools==80.9.0
  "$VENV/bin/python" -m pip install \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1 torchvision==0.20.1
  "$VENV/bin/python" -m pip install -r "$ROOT/requirements.txt"
  "$VENV/bin/python" -m pip install -e "${ROOT}[test]"
  if [[ "${INSTALL_FLASH_ATTN:-1}" == "1" ]]; then
    command -v nvcc >/dev/null || {
      echo "FlashAttention requires the CUDA 12.1 toolkit (nvcc), not only an NVIDIA driver." >&2
      exit 2
    }
    local torch_cuda nvcc_cuda
    torch_cuda="$("$VENV/bin/python" -c 'import torch; print(torch.version.cuda or "")')"
    nvcc_cuda="$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | tail -n 1)"
    if [[ "$torch_cuda" != "12.1" || "$nvcc_cuda" != "$torch_cuda" ]]; then
      echo "CUDA toolkit mismatch: PyTorch uses ${torch_cuda:-unknown}, nvcc uses ${nvcc_cuda:-unknown}; load CUDA 12.1." >&2
      exit 2
    fi
    MAX_JOBS="${FLASH_ATTN_MAX_JOBS:-8}" \
      "$VENV/bin/python" -m pip install --no-build-isolation flash-attn==2.7.4.post1
  fi
  "$VENV/bin/python" -m pip check
  "$VENV/bin/python" "$ROOT/tools/verify_release.py"
  echo "Environment installed and source verified: $VENV"
}

prepare_eagle() {
  command -v git >/dev/null || { echo "git is required" >&2; exit 2; }
  mkdir -p "$WORK_ROOT"
  if [[ ! -d "$EAGLE_DIR/.git" ]]; then
    git clone https://github.com/NVlabs/Eagle.git "$EAGLE_DIR"
  else
    local status head dirty_paths expected_dirty_paths
    status="$(git -C "$EAGLE_DIR" status --porcelain)"
    head="$(git -C "$EAGLE_DIR" rev-parse HEAD)"
    if [[ -n "$status" ]]; then
      dirty_paths="$(git -C "$EAGLE_DIR" status --porcelain | cut -c4- | sort)"
      expected_dirty_paths=$'Embodied/eaglevl/train/locany_finetune_magi_stream.py\nEmbodied/eaglevl/train/tools.py'
      if [[ "$head" == "$EAGLE_REVISION" && \
            "$dirty_paths" == "$expected_dirty_paths" ]] && \
         grep -q 'PIXEL_PIVR_VIRTUAL_CROP_V1' \
           "$EAGLE_DIR/Embodied/eaglevl/train/tools.py" && \
         grep -q 'strict coverage refuses sample replacement' \
           "$EAGLE_DIR/Embodied/eaglevl/train/locany_finetune_magi_stream.py" && \
         git -C "$EAGLE_DIR" apply --reverse --check "$PATCH"; then
        if [[ "$(sha256sum "$EAGLE_DIR/Embodied/eaglevl/train/tools.py" | awk '{print $1}')" == "$EAGLE_TOOLS_SHA256" && \
              "$(sha256sum "$EAGLE_DIR/Embodied/eaglevl/train/locany_finetune_magi_stream.py" | awk '{print $1}')" == "$EAGLE_TRAINER_SHA256" ]]; then
          echo "Eagle already has the exact tested Pixel-PIVR patch."
          return
        fi
      fi
      echo "Refusing to modify an unexpectedly dirty Eagle checkout: $EAGLE_DIR" >&2
      git -C "$EAGLE_DIR" status --short >&2
      exit 2
    fi
  fi
  git -C "$EAGLE_DIR" fetch --quiet origin "$EAGLE_REVISION"
  git -C "$EAGLE_DIR" checkout --detach "$EAGLE_REVISION"
  git -C "$EAGLE_DIR" apply --check "$PATCH"
  git -C "$EAGLE_DIR" apply "$PATCH"
  [[ "$(git -C "$EAGLE_DIR" rev-parse HEAD)" == "$EAGLE_REVISION" ]] || {
    echo "Eagle revision verification failed" >&2
    exit 2
  }
  grep -q 'PIXEL_PIVR_VIRTUAL_CROP_V1' "$EAGLE_DIR/Embodied/eaglevl/train/tools.py"
  grep -q 'strict coverage refuses sample replacement' \
    "$EAGLE_DIR/Embodied/eaglevl/train/locany_finetune_magi_stream.py"
  git -C "$EAGLE_DIR" apply --reverse --check "$PATCH"
  [[ "$(sha256sum "$EAGLE_DIR/Embodied/eaglevl/train/tools.py" | awk '{print $1}')" == "$EAGLE_TOOLS_SHA256" ]] || {
    echo "Patched Eagle tools.py checksum verification failed" >&2
    exit 2
  }
  [[ "$(sha256sum "$EAGLE_DIR/Embodied/eaglevl/train/locany_finetune_magi_stream.py" | awk '{print $1}')" == "$EAGLE_TRAINER_SHA256" ]] || {
    echo "Patched Eagle trainer checksum verification failed" >&2
    exit 2
  }
}

download_assets() {
  [[ -x "$VENV/bin/hf" ]] || {
    echo "Missing $VENV/bin/hf; run '$0 install' first." >&2
    exit 2
  }
  require_data_revision
  prepare_eagle
  export HF_XET_HIGH_PERFORMANCE=1
  "$VENV/bin/hf" download "$MODEL_REPO" \
    --revision "$MODEL_REVISION" --local-dir "$MODEL_DIR" --max-workers 16
  "$VENV/bin/hf" download "$DATA_REPO" --repo-type dataset \
    --revision "$DATA_REVISION" --local-dir "$DATA_DIR" --max-workers 32
  "$VENV/bin/python" "$ROOT/tools/build_hf_upload_bundle.py" verify \
    --output "$DATA_DIR" --verify-member-hashes --allow-materialized-images
  "$VENV/bin/python" "$ROOT/tools/materialize_hf_dataset.py" \
    --data-root "$DATA_DIR"
  "$VENV/bin/python" - "$WORK_ROOT/download_receipt.json" <<PY
import json, pathlib
payload = {
    "dataset": {"repo": "$DATA_REPO", "revision": "$DATA_REVISION", "path": str(pathlib.Path("$DATA_DIR").resolve())},
    "model": {"repo": "$MODEL_REPO", "revision": "$MODEL_REVISION", "path": str(pathlib.Path("$MODEL_DIR").resolve())},
    "eagle": {"revision": "$EAGLE_REVISION", "path": str(pathlib.Path("$EAGLE_DIR").resolve()), "patch": "eagle_virtual_crop_v1.patch"},
}
path = pathlib.Path("$WORK_ROOT/download_receipt.json")
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(path)
PY
  echo "Eagle: $EAGLE_DIR"
  echo "Model: $MODEL_DIR"
  echo "Dataset: $DATA_DIR"
}

case "$MODE" in
  install)
    install_environment
    ;;
  download)
    download_assets
    ;;
  verify-data)
    "$VENV/bin/python" "$ROOT/tools/package_hf_magnified_v2.py" verify \
      --output "$DATA_DIR" --verify-image-hashes
    ;;
  verify-bundle)
    "$VENV/bin/python" "$ROOT/tools/build_hf_upload_bundle.py" verify \
      --output "$DATA_DIR" --verify-member-hashes --allow-materialized-images
    ;;
  check)
    require_data_revision
    PREFLIGHT_ARGS=(
      --model "$MODEL_DIR" --eagle-root "$EAGLE_DIR/Embodied"
      --data-root "$DATA_DIR" --run-root "${RUN_ROOT:?Set RUN_ROOT}"
      --global-batch "${GLOBAL_BATCH:-8}"
      --gpu-ids "${GPU_IDS:-0,1,2,3,4,5,6,7}"
      --required-gpus "${REQUIRED_GPUS:-8}"
      --required-gpu-name "${REQUIRED_GPU_NAME:-A100}"
      --minimum-gpu-memory-gib "${MINIMUM_GPU_MEMORY_GIB:-75}"
      --minimum-run-free-gib "${MINIMUM_RUN_FREE_GIB:-100}"
      --max-sequence "${MAX_SEQUENCE:-32768}"
      --image-token-limit "${IMAGE_TOKEN_LIMIT:-6000}"
      --loss-balancing "${LOSS_BALANCING:-source_query_task}"
      --code-root "$ROOT" --expected-code-revision "$CODE_REVISION"
      --download-receipt "$WORK_ROOT/download_receipt.json"
      --expected-data-revision "$DATA_REVISION"
      --expected-model-revision "$MODEL_REVISION"
      --expected-eagle-revision "$EAGLE_REVISION"
      --require-flash-attn
    )
    [[ "${VERIFY_IMAGE_HASHES:-1}" == "1" ]] && PREFLIGHT_ARGS+=(--verify-image-hashes)
    "$VENV/bin/python" "$ROOT/tools/preflight.py" "${PREFLIGHT_ARGS[@]}"
    ;;
  *)
    echo "Usage: $0 {install|download|verify-bundle|verify-data|check}" >&2
    exit 2
    ;;
esac
