#!/usr/bin/env python3
"""Safely materialize Pixel-PIVR image archives after a Hub download."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path
from typing import Any

from build_hf_upload_bundle import BUNDLE_SCHEMA, parse_checksums
from package_hf_dataset import read_jsonl, sha256_file, write_json
from package_hf_magnified_v2 import verify_package


def inventory(root: Path) -> dict[str, dict[str, Any]]:
    values = {}
    for _number, row in read_jsonl(root / "IMAGE_INVENTORY.jsonl"):
        relative = str(row["path"])
        if relative in values:
            raise ValueError(f"Duplicate image inventory path: {relative}")
        values[relative] = row
    return values


def existing_image_ok(path: Path, expected: dict[str, Any]) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == int(expected["bytes"])
        and sha256_file(path) == str(expected["sha256"])
    )


def materialize(root: Path) -> dict[str, Any]:
    root = root.resolve()
    archives = json.loads((root / "ARCHIVE_INVENTORY.json").read_text(encoding="utf-8"))
    if archives.get("schema_version") != BUNDLE_SCHEMA:
        raise ValueError("Dataset is not the tested archived magnified-v2 bundle")
    expected = inventory(root)
    checksum = parse_checksums(root / "SHA256SUMS")
    for record in archives["archives"]:
        relative = str(record["path"])
        path = root / relative
        if not path.is_file() or path.stat().st_size != int(record["archive_bytes"]):
            raise FileNotFoundError(f"Missing/partial image archive: {path}")
        if sha256_file(path) != str(record["sha256"]) or checksum.get(relative) != str(record["sha256"]):
            raise ValueError(f"Image archive checksum mismatch: {path}")

    missing_bytes = sum(
        int(row["bytes"])
        for relative, row in expected.items()
        if not existing_image_ok(root / relative, row)
    )
    free = shutil.disk_usage(root).free
    if free < missing_bytes + 5 * 1024**3:
        raise RuntimeError(
            f"Need at least {(missing_bytes + 5 * 1024**3) / 1024**3:.1f} GiB free "
            f"to materialize images, found {free / 1024**3:.1f} GiB"
        )

    written = skipped = 0
    seen: set[str] = set()
    for record in archives["archives"]:
        path = root / str(record["path"])
        with tarfile.open(path, "r") as archive:
            for member in archive:
                relative = member.name
                if not member.isfile() or relative not in expected or relative in seen:
                    raise ValueError(f"Unexpected/duplicate archive member: {relative}")
                if Path(relative).is_absolute() or ".." in Path(relative).parts:
                    raise ValueError(f"Unsafe archive member: {relative}")
                row = expected[relative]
                if member.size != int(row["bytes"]):
                    raise ValueError(f"Archive member size mismatch: {relative}")
                destination = root / relative
                if existing_image_ok(destination, row):
                    skipped += 1
                    seen.add(relative)
                    continue
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"Cannot read archive member: {relative}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(destination.suffix + f".tmp.{os.getpid()}")
                digest = hashlib.sha256()
                size = 0
                with contextlib.closing(source), temporary.open("wb") as handle:
                    for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        handle.write(block)
                        digest.update(block)
                        size += len(block)
                if size != int(row["bytes"]) or digest.hexdigest() != str(row["sha256"]):
                    temporary.unlink(missing_ok=True)
                    raise ValueError(f"Extracted image checksum mismatch: {relative}")
                os.replace(temporary, destination)
                written += 1
                seen.add(relative)
    if seen != set(expected):
        raise RuntimeError(f"Image materialization coverage mismatch: {len(seen)} != {len(expected)}")

    package = verify_package(root, verify_image_hashes=True)
    report = {
        "status": "passed",
        "schema_version": BUNDLE_SCHEMA,
        "images": len(seen),
        "newly_materialized": written,
        "already_verified": skipped,
        "full_package_verification": package,
    }
    write_json(root / "materialization.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.data_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
