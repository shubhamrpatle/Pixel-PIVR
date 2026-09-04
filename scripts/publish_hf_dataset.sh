#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-check}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
HF_BIN="${HF_BIN:-$ROOT/.venv/bin/hf}"
BUNDLE_ROOT="${BUNDLE_ROOT:?Set BUNDLE_ROOT to the verified upload-bundle directory}"
HF_REPO="${HF_REPO:-shubhampatle/Pixel-PIVR-Magnified-v2}"
HF_WORKERS="${HF_WORKERS:-16}"

[[ -x "$PYTHON_BIN" ]] || { echo "Missing Python: $PYTHON_BIN" >&2; exit 2; }
[[ -x "$HF_BIN" ]] || { echo "Missing HF CLI: $HF_BIN" >&2; exit 2; }
[[ -f "$BUNDLE_ROOT/BUNDLE_MANIFEST.json" ]] || {
  echo "Not a Pixel-PIVR upload bundle: $BUNDLE_ROOT" >&2
  exit 2
}
[[ "$HF_REPO" != "shubhampatle/Pixel-PIVR" ]] || {
  echo "Refusing to overwrite the legacy Pixel-PIVR dataset repository." >&2
  exit 2
}

verify_local() {
  "$PYTHON_BIN" "$ROOT/tools/build_hf_upload_bundle.py" verify \
    --output "$BUNDLE_ROOT" --verify-member-hashes
}

resolved_revision() {
  "$PYTHON_BIN" - "$HF_REPO" <<'PY'
import sys
from huggingface_hub import HfApi

print(HfApi().dataset_info(sys.argv[1]).sha)
PY
}

require_private_repo() {
  "$PYTHON_BIN" - "$HF_REPO" <<'PY'
import sys
from huggingface_hub import HfApi

info = HfApi().dataset_info(sys.argv[1])
if not info.private:
    raise SystemExit(
        "Refusing upload: the dataset repository is public. Review all source "
        "redistribution terms before changing its visibility."
    )
print(f"verified private dataset repository: {sys.argv[1]}")
PY
}

case "$MODE" in
  check)
    verify_local
    ;;
  upload)
    verify_local
    "$HF_BIN" auth whoami >/dev/null
    "$HF_BIN" repo create "$HF_REPO" --repo-type dataset --private --exist-ok
    require_private_repo
    export HF_XET_HIGH_PERFORMANCE=1
    "$HF_BIN" upload-large-folder "$HF_REPO" "$BUNDLE_ROOT" \
      --repo-type dataset --num-workers "$HF_WORKERS" \
      --exclude bundle_verification.json remote_verification.json '.cache/**'
    revision="$(resolved_revision)"
    "$PYTHON_BIN" "$ROOT/tools/verify_hf_snapshot.py" \
      --repo-id "$HF_REPO" --local-dir "$BUNDLE_ROOT" --revision "$revision"
    printf 'VERIFIED_DATA_REVISION=%s\n' "$revision"
    ;;
  verify-remote)
    verify_local
    revision="${DATA_REVISION:?Set DATA_REVISION to the remote 40-character commit SHA}"
    [[ "$revision" =~ ^[0-9a-f]{40}$ ]] || {
      echo "DATA_REVISION must be a 40-character lowercase commit SHA." >&2
      exit 2
    }
    "$PYTHON_BIN" "$ROOT/tools/verify_hf_snapshot.py" \
      --repo-id "$HF_REPO" --local-dir "$BUNDLE_ROOT" --revision "$revision"
    ;;
  *)
    echo "Usage: BUNDLE_ROOT=/path/to/bundle $0 {check|upload|verify-remote}" >&2
    exit 2
    ;;
esac
