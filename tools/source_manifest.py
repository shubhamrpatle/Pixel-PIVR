#!/usr/bin/env python3
"""Update or verify the portable source SHA-256 manifest."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.sha256"
EXCLUDED_DIRECTORIES = {
    ".git",
    ".cache",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "checkpoints",
    "data",
    "outputs",
    "runs",
    "wandb",
}


def files() -> list[Path]:
    output = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if not path.is_file() or path == MANIFEST:
            continue
        if any(part in EXCLUDED_DIRECTORIES for part in relative.parts):
            continue
        if path.suffix == ".pyc" or (relative.parts[0] == "configs" and path.suffix == ".env"):
            continue
        output.append(path)
    return sorted(output, key=lambda value: value.relative_to(ROOT).as_posix())


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def content() -> str:
    return "".join(
        f"{digest(path)}  ./{path.relative_to(ROOT).as_posix()}\n" for path in files()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("update", "check"))
    args = parser.parse_args()
    expected = content()
    if args.mode == "update":
        temporary = MANIFEST.with_suffix(".sha256.tmp")
        temporary.write_text(expected, encoding="utf-8")
        temporary.replace(MANIFEST)
        print(f"Updated {MANIFEST} with {len(files())} files")
        return
    actual = MANIFEST.read_text(encoding="utf-8") if MANIFEST.is_file() else ""
    if actual != expected:
        raise SystemExit("MANIFEST.sha256 is stale; run tools/source_manifest.py update")
    print(f"Verified {len(files())} source files")


if __name__ == "__main__":
    main()
