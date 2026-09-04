#!/usr/bin/env python3
"""Build or verify a low-file-count Hugging Face distribution bundle."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from package_hf_dataset import read_jsonl, sha256_file, write_json


GIB = 1024**3
BUNDLE_SCHEMA = "pixel-pivr-hf-archive-bundle-v1"
GENERATED_REPORTS = {"bundle_verification.json", "remote_verification.json"}


def package_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or ".cache" in relative.parts:
            continue
        if relative.parts[0] == "images" or path.name in {
            "SHA256SUMS",
            "verification.json",
            "IMAGE_INVENTORY.jsonl",
            *GENERATED_REPORTS,
        }:
            continue
        yield path


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def image_inventory(root: Path) -> list[dict[str, Any]]:
    rows = [row for _number, row in read_jsonl(root / "IMAGE_INVENTORY.jsonl")]
    paths = [str(row["path"]) for row in rows]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("IMAGE_INVENTORY.jsonl must be sorted and unique")
    return rows


def tar_info(tar: tarfile.TarFile, source: Path, relative: str) -> tarfile.TarInfo:
    info = tar.gettarinfo(str(source), arcname=relative)
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    info.mode = 0o644
    return info


def write_archive(
    destination: Path, root: Path, rows: list[dict[str, Any]], split: str, index: int
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    image_bytes = 0
    with tarfile.open(temporary, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for row in rows:
            relative = str(row["path"])
            source = root / relative
            if not source.is_file() or source.stat().st_size != int(row["bytes"]):
                raise ValueError(f"Missing or size-mismatched materialized image: {source}")
            with source.open("rb") as handle:
                archive.addfile(tar_info(archive, source, relative), handle)
            image_bytes += int(row["bytes"])
    os.replace(temporary, destination)
    return {
        "path": destination.relative_to(destination.parents[1]).as_posix(),
        "split": split,
        "index": index,
        "members": len(rows),
        "member_bytes": image_bytes,
        "archive_bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "first_member": str(rows[0]["path"]),
        "last_member": str(rows[-1]["path"]),
    }


def archive_images(
    source: Path, output: Path, rows: list[dict[str, Any]], target_bytes: int
) -> list[dict[str, Any]]:
    by_split: dict[str, list[dict[str, Any]]] = {name: [] for name in ("train", "validation", "test")}
    for row in rows:
        parts = Path(str(row["path"])).parts
        if len(parts) < 3 or parts[0] != "images" or parts[1] not in by_split:
            raise ValueError(f"Invalid image inventory path: {row['path']!r}")
        by_split[parts[1]].append(row)
    records = []
    for split, values in by_split.items():
        chunk: list[dict[str, Any]] = []
        chunk_bytes = 0
        index = 0
        for row in values:
            size = int(row["bytes"])
            if chunk and chunk_bytes + size > target_bytes:
                name = f"images-{split}-part-{index:05d}.tar"
                records.append(write_archive(output / "archives" / name, source, chunk, split, index))
                index += 1
                chunk, chunk_bytes = [], 0
            chunk.append(row)
            chunk_bytes += size
        if chunk:
            name = f"images-{split}-part-{index:05d}.tar"
            records.append(write_archive(output / "archives" / name, source, chunk, split, index))
    return records


def checksum_entries(output: Path, rows: list[dict[str, Any]]) -> dict[str, str]:
    entries = {
        path.relative_to(output).as_posix(): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
        and ".cache" not in path.relative_to(output).parts
        and path.name not in {"SHA256SUMS", *GENERATED_REPORTS}
    }
    for row in rows:
        relative = str(row["path"])
        digest = str(row["sha256"])
        if relative in entries:
            raise ValueError(f"An archived image also exists as a bundle file: {relative}")
        entries[relative] = digest
    return dict(sorted(entries.items()))


def write_checksums(output: Path, entries: dict[str, str]) -> None:
    with (output / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for relative, digest in entries.items():
            handle.write(f"{digest}  {relative}\n")


def build(source: Path, output: Path, target_bytes: int) -> dict[str, Any]:
    source, output = source.resolve(), output.resolve()
    if target_bytes <= 0:
        raise ValueError("Archive target size must be positive")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be absent or empty: {output}")
    for required in ("manifest.json", "IMAGE_INVENTORY.jsonl", "annotations", "recipes", "images"):
        if not (source / required).exists():
            raise FileNotFoundError(source / required)
    output.mkdir(parents=True, exist_ok=True)
    for path in package_files(source):
        relative = path.relative_to(source)
        if relative.as_posix() in {"README.md", "manifest.json"}:
            continue
        link_or_copy(path, output / relative)

    rows = image_inventory(source)
    link_or_copy(source / "IMAGE_INVENTORY.jsonl", output / "IMAGE_INVENTORY.jsonl")
    archives = archive_images(source, output, rows, target_bytes)
    archive_manifest = {
        "schema_version": BUNDLE_SCHEMA,
        "archive_format": "uncompressed POSIX tar with deterministic metadata",
        "archives": archives,
        "archive_count": len(archives),
        "image_members": sum(int(row["members"]) for row in archives),
        "image_bytes": sum(int(row["member_bytes"]) for row in archives),
        "archive_bytes": sum(int(row["archive_bytes"]) for row in archives),
    }
    write_json(output / "ARCHIVE_INVENTORY.json", archive_manifest)

    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    manifest["distribution"] = {
        "schema_version": BUNDLE_SCHEMA,
        "image_transport": "deterministic_tar_shards",
        "archive_inventory": "ARCHIVE_INVENTORY.json",
        "materialized_image_inventory": "IMAGE_INVENTORY.jsonl",
        "training_requires_materialization": True,
        "reason": "Keep the Hub repository well below 100,000 files.",
    }
    write_json(output / "manifest.json", manifest)

    readme = (source / "README.md").read_text(encoding="utf-8")
    readme += """

## Archive transport

Images are transported in deterministic uncompressed tar shards so the Hub
repository remains well below its recommended 100,000-file ceiling. The JSONL
paths still point to `images/...`. Run the code repository bootstrap command;
it validates every archive, extracts images atomically, and verifies every image
SHA-256 before training is unlocked. Do not train directly from the tar files.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")

    bundle = {
        "schema_version": BUNDLE_SCHEMA,
        "source_schema_version": manifest["schema_version"],
        "source_materialized_package": source.name,
        "target_archive_bytes": int(target_bytes),
        "archives": archive_manifest,
        "remote_physical_files_excluding_reports": sum(
            1
            for path in output.rglob("*")
            if path.is_file() and path.name not in GENERATED_REPORTS
        )
        + 2,
        "logical_image_files_after_materialization": len(rows),
    }
    write_json(output / "BUNDLE_MANIFEST.json", bundle)
    entries = checksum_entries(output, rows)
    write_checksums(output, entries)
    return verify(output, verify_member_hashes=False)


def parse_checksums(path: Path) -> dict[str, str]:
    entries = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if "  " not in line:
            raise ValueError(f"Malformed checksum line {number}")
        digest, relative = line.split("  ", 1)
        if len(digest) != 64 or relative in entries:
            raise ValueError(f"Invalid checksum line {number}")
        entries[relative] = digest
    return entries


def verify(
    root: Path,
    *,
    verify_member_hashes: bool,
    allow_materialized_images: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    bundle = json.loads((root / "BUNDLE_MANIFEST.json").read_text(encoding="utf-8"))
    archive_manifest = json.loads((root / "ARCHIVE_INVENTORY.json").read_text(encoding="utf-8"))
    if bundle.get("schema_version") != BUNDLE_SCHEMA or archive_manifest.get("schema_version") != BUNDLE_SCHEMA:
        raise ValueError("Wrong archive bundle schema")
    if (root / "images").exists() and not allow_materialized_images:
        raise ValueError("Upload bundle must not contain a materialized images directory")
    rows = image_inventory(root)
    expected_images = {str(row["path"]): row for row in rows}
    checksums = parse_checksums(root / "SHA256SUMS")
    if set(expected_images) - set(checksums):
        raise ValueError("SHA256SUMS does not cover every eventual image")
    for relative, row in expected_images.items():
        if checksums.get(relative) != str(row["sha256"]):
            raise ValueError(f"Logical image checksum mismatch: {relative}")

    physical = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and ".cache" not in path.relative_to(root).parts
        and path.relative_to(root).parts[0] != "images"
        and path.name not in {"SHA256SUMS", *GENERATED_REPORTS}
    }
    expected_physical = {path for path in checksums if not path.startswith("images/")}
    if physical != expected_physical:
        raise ValueError(
            f"Bundle physical checksum coverage differs: missing={sorted(physical-expected_physical)[:10]}, "
            f"extra={sorted(expected_physical-physical)[:10]}"
        )
    for relative in physical:
        if sha256_file(root / relative) != checksums[relative]:
            raise ValueError(f"Physical file checksum mismatch: {relative}")

    seen: set[str] = set()
    split_counts: Counter[str] = Counter()
    payload_bytes = 0
    for record in archive_manifest["archives"]:
        archive_path = root / str(record["path"])
        if sha256_file(archive_path) != record["sha256"]:
            raise ValueError(f"Archive checksum mismatch: {archive_path}")
        members = 0
        member_bytes = 0
        with tarfile.open(archive_path, "r") as archive:
            for member in archive:
                if not member.isfile() or member.name not in expected_images or member.name in seen:
                    raise ValueError(f"Unexpected/duplicate archive member: {member.name}")
                expected = expected_images[member.name]
                if member.size != int(expected["bytes"]):
                    raise ValueError(f"Archive member size mismatch: {member.name}")
                if verify_member_hashes:
                    handle = archive.extractfile(member)
                    if handle is None:
                        raise ValueError(f"Cannot read archive member: {member.name}")
                    import hashlib

                    digest = hashlib.sha256()
                    with contextlib.closing(handle):
                        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                            digest.update(block)
                    if digest.hexdigest() != expected["sha256"]:
                        raise ValueError(f"Archive member hash mismatch: {member.name}")
                seen.add(member.name)
                split_counts[Path(member.name).parts[1]] += 1
                members += 1
                member_bytes += member.size
                payload_bytes += member.size
        if members != int(record["members"]) or member_bytes != int(record["member_bytes"]):
            raise ValueError(f"Archive accounting mismatch: {archive_path}")
    if seen != set(expected_images):
        raise ValueError(f"Archived image coverage mismatch: {len(seen)} != {len(expected_images)}")
    report = {
        "status": "passed",
        "schema_version": BUNDLE_SCHEMA,
        "physical_upload_files": len(physical) + 1,
        "archives": len(archive_manifest["archives"]),
        "archived_images": len(seen),
        "archived_image_bytes": payload_bytes,
        "images_by_split": dict(sorted(split_counts.items())),
        "archive_payload_sha256_verified": bool(verify_member_hashes),
    }
    declared_physical = int(bundle.get("remote_physical_files_excluding_reports", -1))
    if declared_physical != report["physical_upload_files"]:
        raise ValueError(
            "Physical upload file count differs from BUNDLE_MANIFEST: "
            f"{report['physical_upload_files']} != {declared_physical}"
        )
    write_json(root / "bundle_verification.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "verify"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive-gib", type=float, default=4.0)
    parser.add_argument("--verify-member-hashes", action="store_true")
    parser.add_argument(
        "--allow-materialized-images",
        action="store_true",
        help="Ignore an already verified images/ tree when rechecking downloaded archives.",
    )
    args = parser.parse_args()
    if args.mode == "build":
        if args.source is None:
            parser.error("build requires --source")
        result = build(args.source, args.output, int(args.archive_gib * GIB))
    else:
        result = verify(
            args.output,
            verify_member_hashes=args.verify_member_hashes,
            allow_materialized_images=args.allow_materialized_images,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
