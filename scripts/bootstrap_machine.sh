#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-install}"
WORK_ROOT="${WORK_ROOT:-$(dirname "$ROOT")/pixel-pivr-assets}"
VENV="${VENV:-$ROOT/.venv}"
EAGLE_DIR="${EAGLE_DIR:-$WORK_ROOT/Eagle}"
EAGLE_REVISION="${EAGLE_REVISION:-8442db3b79f7fd2357e468e6eecdd9b6a82049ff}"
MODEL_DIR="${MODEL_DIR:-$WORK_ROOT/LocateAnything-3B}"
DATA_DIR="${DATA_DIR:-$WORK_ROOT/Pixel-PIVR-data}"

case "$MODE" in
  install)
    python3 -m venv "$VENV"
    "$VENV/bin/python" -m pip install --upgrade pip wheel
    "$VENV/bin/python" -m pip install -r "$ROOT/requirements.txt"
    "$VENV/bin/python" -m pip install -e "$ROOT"
    echo "Environment installed: $VENV"
    ;;
  download)
    command -v git >/dev/null || { echo "git is required" >&2; exit 2; }
    command -v hf >/dev/null || {
      echo "Install/login to the Hugging Face CLI first: curl -LsSf https://hf.co/cli/install.sh | bash" >&2
      exit 2
    }
    mkdir -p "$WORK_ROOT"
    if [[ ! -d "$EAGLE_DIR/.git" ]]; then
      git clone https://github.com/NVlabs/Eagle.git "$EAGLE_DIR"
      git -C "$EAGLE_DIR" checkout --detach "$EAGLE_REVISION"
    elif [[ "$(git -C "$EAGLE_DIR" rev-parse HEAD)" != "$EAGLE_REVISION" ]]; then
      echo "Existing Eagle checkout is not the tested revision $EAGLE_REVISION: $EAGLE_DIR" >&2
      echo "Use a fresh EAGLE_DIR or explicitly set EAGLE_REVISION to the intended commit." >&2
      exit 2
    fi
    hf download nvidia/LocateAnything-3B --local-dir "$MODEL_DIR"
    hf download shubhampatle/Pixel-PIVR --repo-type dataset --local-dir "$DATA_DIR"
    echo "Eagle: $EAGLE_DIR"
    echo "Model: $MODEL_DIR"
    echo "Dataset: $DATA_DIR"
    ;;
  check)
    "$VENV/bin/python" "$ROOT/tools/preflight.py" \
      --model "$MODEL_DIR" \
      --eagle-root "$EAGLE_DIR/Embodied" \
      --data-root "$DATA_DIR" \
      --global-batch "${GLOBAL_BATCH:-4}"
    ;;
  *)
    echo "Usage: $0 {install|download|check}" >&2
    exit 2
    ;;
esac
