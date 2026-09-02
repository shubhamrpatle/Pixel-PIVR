#!/usr/bin/env python3
"""Build a minimal, portable file list for a private Pixel-PIVR run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path, block_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def within_root(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Asset is outside data root {root}: {resolved}") from exc


def regular_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    for current, directories, files in os.walk(root):
        directories[:] = sorted(
            name
            for name in directories
            if name
            not in {
                ".git",
                "__pycache__",
                ".cache",
                "wandb",
                "runs",
                "outputs",
            }
        )
        for name in sorted(files):
            path = Path(current) / name
            if not path.is_symlink() and path.is_file() and path.suffix != ".pyc":
                yield path


def image_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("path"), str):
        return value["path"]
    raise TypeError(f"Unsupported image item: {value!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--eagle-root", type=Path, required=True)
    parser.add_argument("--holdout-hashes", type=Path, required=True)
    parser.add_argument("--output-list", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    required = (
        data_root,
        args.dataset_dir,
        args.model,
        args.eagle_root,
        args.holdout_hashes,
    )
    for path in required:
        if not path.exists():
            parser.error(f"Missing input: {path}")

    assets: set[Path] = set()
    split_stats: dict[str, dict[str, Any]] = {}
    image_paths: set[Path] = set()
    for split in ("train", "validation", "smoke"):
        jsonl = args.dataset_dir / f"{split}.jsonl"
        if not jsonl.is_file():
            parser.error(f"Missing split: {jsonl}")
        records = 0
        split_images: set[Path] = set()
        with jsonl.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise ValueError(f"Blank line in {jsonl}:{line_number}")
                row = json.loads(line)
                values = row.get("image")
                values = values if isinstance(values, list) else [values]
                for value in values:
                    raw = Path(image_value(value))
                    resolved = raw.resolve() if raw.is_absolute() else (data_root / raw).resolve()
                    if not resolved.is_file():
                        raise FileNotFoundError(resolved)
                    within_root(resolved, data_root)
                    split_images.add(resolved)
                    image_paths.add(resolved)
                records += 1
        split_stats[split] = {
            "records": records,
            "unique_images": len(split_images),
            "jsonl_sha256": sha256_file(jsonl),
        }

    for root in (args.dataset_dir, args.model, args.eagle_root):
        assets.update(regular_files(root.resolve()))
    assets.add(args.holdout_hashes.resolve())
    assets.update(image_paths)

    relative = sorted(within_root(path, data_root) for path in assets)
    for path in relative:
        if "\n" in str(path):
            raise ValueError(f"Newline in asset path is unsupported: {path!r}")
    missing = [str(path) for path in relative if not (data_root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"Manifest contains missing files: {missing[:10]}")

    args.output_list.parent.mkdir(parents=True, exist_ok=True)
    args.output_list.write_text(
        "".join(f"{path.as_posix()}\n" for path in relative), encoding="utf-8"
    )
    total_bytes = sum((data_root / path).stat().st_size for path in relative)
    report = {
        "schema_version": "pixel-pivr-private-assets-v1",
        "data_root": str(data_root),
        "files": len(relative),
        "bytes": total_bytes,
        "gib": total_bytes / 1024**3,
        "unique_training_images": len(image_paths),
        "splits": split_stats,
        "holdout_hashes_sha256": sha256_file(args.holdout_hashes),
        "rsync_file_list": str(args.output_list.resolve()),
        "rsync_file_list_sha256": sha256_file(args.output_list),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
