#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_ROOT="${WORK_ROOT:?Set WORK_ROOT to the absolute asset directory}"
RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT to the absolute experiment directory}"
DATA_REVISION="${DATA_REVISION:?Set DATA_REVISION to the verified 40-character Hugging Face commit SHA}"
OUTPUT="${PIPELINE_CONFIG:-$ROOT/configs/full_scale.env}"

[[ "$WORK_ROOT" == /* && "$RUN_ROOT" == /* && "$OUTPUT" == /* ]] || {
  echo "WORK_ROOT, RUN_ROOT, and PIPELINE_CONFIG must be absolute paths." >&2
  exit 2
}
[[ "$DATA_REVISION" =~ ^[0-9a-f]{40}$ ]] || {
  echo "DATA_REVISION must be a 40-character lowercase commit SHA." >&2
  exit 2
}
CODE_REVISION="$(git -C "$ROOT" rev-parse HEAD)"
[[ "$CODE_REVISION" =~ ^[0-9a-f]{40}$ ]] || {
  echo "Could not resolve the Pixel-PIVR Git commit." >&2
  exit 2
}
if [[ -e "$OUTPUT" && "${FORCE:-0}" != "1" ]]; then
  echo "Refusing to overwrite $OUTPUT; set FORCE=1 only for a new run." >&2
  exit 2
fi

"${PYTHON_BOOTSTRAP:-python3.10}" - \
  "$ROOT/configs/full_scale.env.example" "$OUTPUT" "$ROOT" \
  "$WORK_ROOT" "$RUN_ROOT" "$CODE_REVISION" "$DATA_REVISION" \
  "${WANDB_PROJECT-pixel-pivr}" <<'PY'
import shlex
import sys
from pathlib import Path

template, output, code_root, work_root, run_root = map(Path, sys.argv[1:6])
code_revision, data_revision, wandb_project = sys.argv[6:9]
replacements = {
    "PYTHON_BIN": code_root / ".venv/bin/python",
    "MODEL_PATH": work_root / "LocateAnything-3B",
    "EAGLE_ROOT": work_root / "Eagle/Embodied",
    "DATA_ROOT": work_root / "Pixel-PIVR-Magnified-v2",
    "RUN_ROOT": run_root,
    "CODE_REVISION": code_revision,
    "DATA_REVISION": data_revision,
    "WANDB_PROJECT": wandb_project,
}
lines = []
seen = set()
for line in template.read_text(encoding="utf-8").splitlines():
    key = line.split("=", 1)[0] if "=" in line and not line.startswith("#") else None
    if key in replacements:
        raw = replacements[key]
        value = str(raw.resolve()) if isinstance(raw, Path) else str(raw)
        line = f"{key}={shlex.quote(value)}"
        seen.add(key)
    lines.append(line)
missing = sorted(set(replacements) - seen)
if missing:
    raise SystemExit(f"Template is missing required keys: {missing}")
output.parent.mkdir(parents=True, exist_ok=True)
temporary = output.with_suffix(output.suffix + ".tmp")
temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
temporary.replace(output)
print(output)
PY

echo "Configuration written with FULL_SCALE_APPROVED=NO."
echo "Run preflight and smoke tests before changing approval to YES."
