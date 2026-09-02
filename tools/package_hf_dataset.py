#!/usr/bin/env python3
"""Package the verified full-scale Pixel-PIVR corpus for Hugging Face Hub.

The source curriculum references images in the research workspace. This tool
materializes one content-addressed image per split, rewrites every annotation to
portable paths, shards JSONL files, and refuses train/validation/test leakage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


MIB = 1024 * 1024
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def sha256_file(path: Path, block_bytes: int = 8 * MIB) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL row: {path}:{line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"Expected object: {path}:{line_number}")
            yield line_number, row


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def normalize_suffix(path: Path) -> str:
    suffix = path.suffix.lower()
    return suffix if suffix in IMAGE_SUFFIXES else ".img"


def image_items(row: Mapping[str, Any]) -> list[Any]:
    value = row.get("image")
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def image_path_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("path"), str):
        return value["path"]
    raise TypeError(f"Unsupported image value: {value!r}")


class ShardedWriter:
    def __init__(self, directory: Path, max_bytes: int) -> None:
        self.directory = directory
        self.max_bytes = max_bytes
        self.index = -1
        self.handle = None
        self.current_bytes = 0
        self.paths: list[Path] = []
        self.records = 0

    def _open(self) -> None:
        if self.handle is not None:
            self.handle.close()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index += 1
        path = self.directory / f"part-{self.index:05d}.jsonl"
        self.paths.append(path)
        self.handle = path.open("w", encoding="utf-8")
        self.current_bytes = 0

    def write(self, row: Mapping[str, Any]) -> None:
        encoded = (json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n").encode("utf-8")
        if self.handle is None or (self.current_bytes and self.current_bytes + len(encoded) > self.max_bytes):
            self._open()
        assert self.handle is not None
        self.handle.write(encoded.decode("utf-8"))
        self.current_bytes += len(encoded)
        self.records += 1

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None


class Packager:
    def __init__(
        self,
        source_root: Path,
        project_root: Path,
        curriculum: Path,
        output: Path,
        *,
        link_mode: str,
        shard_mib: int,
        verify_content: bool,
    ) -> None:
        self.source_root = source_root.resolve()
        self.project_root = project_root.resolve()
        self.curriculum = curriculum.resolve()
        self.output = output.resolve()
        self.link_mode = link_mode
        self.shard_bytes = shard_mib * MIB
        self.verify_content = verify_content
        self.path_hash_cache: dict[Path, str] = {}
        self.hash_paths: dict[str, dict[str, str]] = defaultdict(dict)
        self.split_hashes: dict[str, set[str]] = defaultdict(set)
        self.image_bytes: Counter[str] = Counter()
        self.image_counts: Counter[str] = Counter()
        self.record_stats: Counter[str] = Counter()
        self.dataset_stats: Counter[str] = Counter()
        self.output_files: list[Path] = []

    def resolve_image(self, raw: str, *, fallback_root: Path | None = None) -> Path:
        path = Path(raw)
        candidates = [path] if path.is_absolute() else []
        if fallback_root is not None and not path.is_absolute():
            candidates.append(fallback_root / path)
            candidates.append(fallback_root / path.name)
        if not path.is_absolute():
            candidates.extend((self.source_root / path, self.project_root / path))
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"Cannot resolve image {raw!r}; tried {candidates}")

    def content_hash(self, path: Path, expected: str | None = None) -> str:
        path = path.resolve()
        actual = self.path_hash_cache.get(path)
        if actual is None:
            if expected and not self.verify_content:
                actual = expected
            else:
                actual = sha256_file(path)
            self.path_hash_cache[path] = actual
        if expected and actual != expected:
            raise ValueError(f"Image hash mismatch for {path}: metadata={expected}, actual={actual}")
        return actual

    def materialize_image(self, path: Path, split: str, expected_hash: str | None = None) -> tuple[str, str]:
        digest = self.content_hash(path, expected_hash)
        suffix = normalize_suffix(path)
        relative = Path("images") / split / digest[:2] / f"{digest}{suffix}"
        prior = self.hash_paths[split].get(digest)
        if prior is not None:
            return prior, digest
        destination = self.output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if sha256_file(destination) != digest:
                raise ValueError(f"Existing destination has wrong content: {destination}")
        elif self.link_mode == "hardlink":
            try:
                os.link(path, destination)
            except OSError:
                shutil.copy2(path, destination)
        else:
            shutil.copy2(path, destination)
        portable = relative.as_posix()
        self.hash_paths[split][digest] = portable
        self.split_hashes[split].add(digest)
        self.image_counts[split] += 1
        self.image_bytes[split] += path.stat().st_size
        return portable, digest

    def rewrite_training_row(self, row: dict[str, Any], split: str) -> dict[str, Any]:
        values = image_items(row)
        if len(values) != 1:
            raise ValueError(f"Full-scale cache PIVR expects one source image, got {len(values)}")
        raw = image_path_value(values[0])
        expected = str(row.get("meta", {}).get("image_content_sha256") or "") or None
        source = self.resolve_image(raw)
        portable, digest = self.materialize_image(source, split, expected)
        row["image"] = portable
        row.setdefault("meta", {})["image_content_sha256"] = digest
        return row

    def package_training(self) -> dict[str, Any]:
        definitions = {
            "stage1_coarse": {
                "round1_global_point": sorted((self.curriculum / "round1_global_point/stage1_coarse").glob("*.jsonl")),
                "round2_point_box": sorted((self.curriculum / "round2_point_box/stage1_coarse").glob("*.jsonl")),
            },
            "stage2_dense_balanced": {
                "round1_global_point": sorted((self.curriculum / "round1_global_point/stage2_dense_balanced").glob("*.jsonl")),
                "round2_point_box": sorted((self.curriculum / "round2_point_box/stage2_dense_balanced").glob("*.jsonl")),
            },
        }
        recipes: dict[str, Any] = {}
        for stage, routes in definitions.items():
            writers: dict[tuple[str, str], ShardedWriter] = {}
            for route, sources in routes.items():
                if not sources:
                    raise FileNotFoundError(f"No source shards for {stage}/{route}")
                for source in sources:
                    for _, row in read_jsonl(source):
                        task = str(row.get("meta", {}).get("task") or "")
                        if task not in {"detection", "grounding", "pointing"}:
                            raise ValueError(f"Unsupported training task {task!r} in {source}")
                        if route == "round2_point_box" and task == "pointing":
                            raise ValueError("Pointing must not have a point-to-box refinement record")
                        row = self.rewrite_training_row(row, "train")
                        key = (task, route)
                        writer = writers.setdefault(
                            key,
                            ShardedWriter(self.output / "annotations/train" / stage / task / route, self.shard_bytes),
                        )
                        writer.write(row)
                        meta = row["meta"]
                        self.record_stats[f"train.{stage}.{task}.{route}.records"] += 1
                        self.record_stats[f"train.{stage}.{task}.{route}.targets"] += int(meta.get("target_count", 0))
                        self.dataset_stats[f"train.{stage}.{task}.{meta.get('dataset', 'unknown')}.records"] += 1
            for writer in writers.values():
                writer.close()
                self.output_files.extend(writer.paths)
            recipes[stage] = {
                task: {
                    route: [str(path.relative_to(self.output).as_posix()) for path in writers[(task, route)].paths]
                    for route in routes
                    if (task, route) in writers
                }
                for task in ("detection", "grounding", "pointing")
            }
        return recipes

    def package_validation(self) -> dict[str, Any]:
        routes = {
            "round1_global_point": sorted((self.curriculum / "round1_global_point/validation").glob("*.jsonl")),
            "round2_point_box": sorted((self.curriculum / "round2_point_box/validation").glob("*.jsonl")),
        }
        writers: dict[tuple[str, str], ShardedWriter] = {}
        for route, sources in routes.items():
            if not sources:
                raise FileNotFoundError(f"No validation shards for {route}")
            for source in sources:
                for _, row in read_jsonl(source):
                    task = str(row.get("meta", {}).get("task") or "")
                    if task not in {"detection", "grounding", "pointing"}:
                        raise ValueError(f"Unsupported validation task {task!r}")
                    row = self.rewrite_training_row(row, "validation")
                    key = (task, route)
                    writer = writers.setdefault(
                        key,
                        ShardedWriter(self.output / "annotations/validation" / task / route, self.shard_bytes),
                    )
                    writer.write(row)
                    self.record_stats[f"validation.{task}.{route}.records"] += 1
                    self.record_stats[f"validation.{task}.{route}.targets"] += int(row["meta"].get("target_count", 0))
        for writer in writers.values():
            writer.close()
            self.output_files.extend(writer.paths)

        monitor_source = self.curriculum / "validation/round2_refinement_monitor1000.jsonl"
        monitor_writer = ShardedWriter(self.output / "annotations/validation_monitor/round2_point_box", self.shard_bytes)
        for _, row in read_jsonl(monitor_source):
            monitor_writer.write(self.rewrite_training_row(row, "validation"))
        monitor_writer.close()
        self.output_files.extend(monitor_writer.paths)
        return {
            task: {
                route: [str(path.relative_to(self.output).as_posix()) for path in writers[(task, route)].paths]
                for route in routes
                if (task, route) in writers
            }
            for task in ("detection", "grounding", "pointing")
        } | {
            "monitor": [str(path.relative_to(self.output).as_posix()) for path in monitor_writer.paths]
        }

    def rewrite_test_image(
        self,
        row: dict[str, Any],
        *,
        fallback_root: Path,
        field: str = "image_path",
    ) -> tuple[dict[str, Any], str]:
        raw = str(row.get(field) or row.get("file_name") or "")
        source = self.resolve_image(raw, fallback_root=fallback_root)
        portable, digest = self.materialize_image(source, "test")
        row[field] = portable
        row["image_content_sha256"] = digest
        return row, digest

    def package_public_tests(self) -> dict[str, Any]:
        benchmark_root = self.project_root / "dataset/benchmark/HBB_type3_public_v1"
        specs = [
            ("detection", "DIOR", benchmark_root / "DIOR_detection.jsonl", self.project_root / "dataset/benchmark/DIOR/images"),
            ("detection", "DOTAv2", benchmark_root / "DOTAv2_detection.jsonl", self.project_root / "dataset/benchmark/DOTAv2/images"),
            ("grounding", "DIOR-RSVG", benchmark_root / "DIOR-RSVG_grounding.jsonl", self.project_root / "dataset/grounding/DIOR-RSVG/JPEGImages"),
            ("grounding", "VRSBench-VG", benchmark_root / "VRSBench-VG_grounding.jsonl", self.project_root / "dataset/grounding/VRSBench/Images_val"),
        ]
        outputs: dict[str, Any] = defaultdict(dict)
        for task, dataset, source, image_root in specs:
            writer = ShardedWriter(self.output / "annotations/test" / task / dataset, self.shard_bytes)
            for _, row in read_jsonl(source):
                row, _ = self.rewrite_test_image(row, fallback_root=image_root)
                writer.write(row)
                self.record_stats[f"test.{task}.{dataset}.records"] += 1
            writer.close()
            self.output_files.extend(writer.paths)
            outputs[task][dataset] = [str(path.relative_to(self.output).as_posix()) for path in writer.paths]
        outputs["pointing"] = self.package_pointing_test()
        return dict(outputs)

    def package_pointing_test(self) -> dict[str, Any]:
        source = self.project_root / "dataset/eval_balanced100/DOTAv2/annotations/DOTAv2_balanced100.jsonl"
        image_root = self.project_root / "dataset/eval_balanced100/DOTAv2/images"
        writer = ShardedWriter(self.output / "annotations/test/pointing/DOTAv2-Balanced100", self.shard_bytes)
        for _, source_row in read_jsonl(source):
            row, digest = self.rewrite_test_image(dict(source_row), fallback_root=image_root)
            width, height = int(row["width"]), int(row["height"])
            by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for obj in row.get("objects", []):
                x1, y1, x2, y2 = map(float, obj["hbox"])
                point = [round(((x1 + x2) / 2) / width * 1000), round(((y1 + y2) / 2) / height * 1000)]
                by_class[str(obj["class_name"])].append(
                    {"hbox": [x1, y1, x2, y2], "point_normalized": point}
                )
            for class_name in sorted(by_class):
                answer = f"<ref>{class_name}</ref>" + "".join(
                    f"<box><{item['point_normalized'][0]}><{item['point_normalized'][1]}></box>"
                    for item in by_class[class_name]
                )
                record_id = hashlib.sha256(f"{digest}\0{class_name}".encode()).hexdigest()[:24]
                portable_row = {
                    "conversations": [
                        {"from": "human", "value": f"<image>\nPoint to: {class_name}."},
                        {"from": "gpt", "value": answer},
                    ],
                    "image": row["image_path"],
                    "image_id": str(row.get("image_id") or row.get("file_name") or digest),
                    "width": width,
                    "height": height,
                    "target_class": class_name,
                    "gt_hboxes": [item["hbox"] for item in by_class[class_name]],
                    "gt_points_normalized": [
                        item["point_normalized"] for item in by_class[class_name]
                    ],
                    "meta": {
                        "benchmark": "DOTAv2-Balanced100",
                        "coordinate_space": "normalized_0_1000",
                        "geometry_mode": "point",
                        "image_content_sha256": digest,
                        "record_id": f"pixel_pivr_point_test_{record_id}",
                        "target_class": class_name,
                        "target_count": len(by_class[class_name]),
                        "task": "pointing",
                        "test_protocol": "class-wise points scored by containment in unmatched same-class HBB",
                    },
                }
                writer.write(portable_row)
                self.record_stats["test.pointing.DOTAv2-Balanced100.records"] += 1
                self.record_stats["test.pointing.DOTAv2-Balanced100.targets"] += len(by_class[class_name])
        writer.close()
        self.output_files.extend(writer.paths)
        return {"DOTAv2-Balanced100": [str(path.relative_to(self.output).as_posix()) for path in writer.paths]}

    def assert_split_separation(self) -> dict[str, int]:
        overlaps = {
            "train_validation": len(self.split_hashes["train"] & self.split_hashes["validation"]),
            "train_test": len(self.split_hashes["train"] & self.split_hashes["test"]),
            "validation_test": len(self.split_hashes["validation"] & self.split_hashes["test"]),
        }
        if any(overlaps.values()):
            raise ValueError(f"Image leakage detected: {overlaps}")
        return overlaps

    def build(self) -> dict[str, Any]:
        if self.output.exists() and any(self.output.iterdir()):
            raise FileExistsError(f"Output must be absent or empty: {self.output}")
        self.output.mkdir(parents=True, exist_ok=True)
        recipes = {
            "train": self.package_training(),
            "validation": self.package_validation(),
            "test": self.package_public_tests(),
        }
        overlaps = self.assert_split_separation()
        manifest = {
            "schema_version": "pixel-pivr-hf-hbb-v1",
            "source_curriculum": self.curriculum.name,
            "coordinate_space": "normalized_0_1000",
            "geometry": "horizontal bounding boxes and points",
            "records": dict(sorted(self.record_stats.items())),
            "records_by_source_dataset": dict(sorted(self.dataset_stats.items())),
            "images": {
                split: {
                    "unique": self.image_counts[split],
                    "bytes": self.image_bytes[split],
                    "gib": self.image_bytes[split] / 1024**3,
                }
                for split in ("train", "validation", "test")
            },
            "image_hash_overlap": overlaps,
            "recipes": recipes,
            "notes": {
                "stage_overlap": "Stage 1 and Stage 2 are curricula within train and may reuse images.",
                "validation": "Independent checkpoint-selection split; excluded from train and public benchmarks.",
                "pointing_test": "DOTAv2 Balanced-100 diagnostic, not a standardized public pointing benchmark.",
            },
        }
        manifest_path = self.output / "manifest.json"
        write_json(manifest_path, manifest)
        self.output_files.append(manifest_path)
        self.write_recipes(recipes)
        self.write_readme(manifest)
        self.write_checksums()
        return manifest

    def write_recipes(self, recipes: Mapping[str, Any]) -> None:
        for stage in ("stage1_coarse", "stage2_dense_balanced"):
            files = []
            for task in ("detection", "grounding", "pointing"):
                for route in ("round1_global_point", "round2_point_box"):
                    files.extend(recipes["train"][stage].get(task, {}).get(route, []))
            payload = {"data_root": ".", "annotation": files, "repeat_time": 1.0, "data_augment": False}
            path = self.output / "recipes" / f"{stage}.json"
            write_json(path, payload)
            self.output_files.append(path)
        validation_files = []
        for task in ("detection", "grounding", "pointing"):
            for route in ("round1_global_point", "round2_point_box"):
                validation_files.extend(recipes["validation"].get(task, {}).get(route, []))
        path = self.output / "recipes/validation_all_tasks.json"
        write_json(path, {"data_root": ".", "annotation": validation_files, "repeat_time": 1.0, "data_augment": False})
        self.output_files.append(path)

    def write_readme(self, manifest: Mapping[str, Any]) -> None:
        images = manifest["images"]
        records = manifest["records"]
        readme = f"""---
pretty_name: Pixel-PIVR Remote-Sensing HBB
license: other
task_categories:
- object-detection
- visual-question-answering
language:
- en
tags:
- remote-sensing
- visual-grounding
- pointing
- locate-anything
- pixel-pivr
---

# Pixel-PIVR Remote-Sensing HBB Dataset

This is the portable, full-scale HBB training/evaluation package for Pixel-PIVR.
It contains content-addressed images, LocateAnything-compatible JSONL annotations,
and separate task files for detection, phrase grounding, and pointing.

## Splits

| Split | Purpose | Unique images | Size |
|---|---|---:|---:|
| `train/stage1_coarse` | Coarse all-task adaptation | shared train pool | - |
| `train/stage2_dense_balanced` | Dense specialization and replay | shared train pool | - |
| `validation` | Checkpoint selection only | {images['validation']['unique']:,} | {images['validation']['gib']:.2f} GiB |
| `test` | Frozen public benchmarks and pointing diagnostic | {images['test']['unique']:,} | {images['test']['gib']:.2f} GiB |

The combined training pool contains **{images['train']['unique']:,} unique images**
({images['train']['gib']:.2f} GiB). Stage 1 and Stage 2 are curricula, so an image
may intentionally occur in both stages. Image SHA-256 overlap is exactly zero
between train, validation, and test.

## Training Records

| Stage | Round 1 global records | Round 2 refinement records |
|---|---:|---:|
| Stage 1 coarse | {sum(v for k,v in records.items() if k.startswith('train.stage1_coarse.') and k.endswith('.round1_global_point.records')):,} | {sum(v for k,v in records.items() if k.startswith('train.stage1_coarse.') and k.endswith('.round2_point_box.records')):,} |
| Stage 2 dense balanced | {sum(v for k,v in records.items() if k.startswith('train.stage2_dense_balanced.') and k.endswith('.round1_global_point.records')):,} | {sum(v for k,v in records.items() if k.startswith('train.stage2_dense_balanced.') and k.endswith('.round2_point_box.records')):,} |
| Validation | {sum(v for k,v in records.items() if k.startswith('validation.') and k.endswith('.round1_global_point.records')):,} | {sum(v for k,v in records.items() if k.startswith('validation.') and k.endswith('.round2_point_box.records')):,} |

Round 1 teaches target-complete global point discovery. Round 2 maps one unique
point address to exactly one HBB or `<box>None</box>`. Coordinates are integer
values normalized to `[0, 1000]`.

## Directory Layout

```text
annotations/
  train/<stage>/<task>/<round>/part-*.jsonl
  validation/<task>/<round>/part-*.jsonl
  validation_monitor/round2_point_box/part-*.jsonl
  test/<task>/<benchmark>/part-*.jsonl
images/<train|validation|test>/<sha-prefix>/<sha256>.<ext>
recipes/{{stage1_coarse,stage2_dense_balanced,validation_all_tasks}}.json
manifest.json
SHA256SUMS
```

## Download

```bash
hf download shubhampatle/Pixel-PIVR --repo-type dataset --local-dir Pixel-PIVR-data
```

Downloads resume automatically when the same command and local directory are used.

## Train and Evaluate Directly

Clone the code repository and install it, then point `DATA_ROOT` at this dataset.
Each recipe contains portable paths relative to the downloaded dataset root.

```bash
git clone https://github.com/shubhamrpatle/Pixel-PIVR.git
cd Pixel-PIVR
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

cp configs/full_scale.env.example configs/full_scale.env
# Set DATA_ROOT=/absolute/path/Pixel-PIVR-data and edit the other four paths.
PIPELINE_CONFIG=$PWD/configs/full_scale.env bash scripts/run_full_pipeline.sh preflight
PIPELINE_CONFIG=$PWD/configs/full_scale.env bash scripts/run_full_pipeline.sh smoke-stage1
PIPELINE_CONFIG=$PWD/configs/full_scale.env bash scripts/run_full_pipeline.sh train-stage1
PIPELINE_CONFIG=$PWD/configs/full_scale.env bash scripts/run_full_pipeline.sh smoke-stage2
PIPELINE_CONFIG=$PWD/configs/full_scale.env bash scripts/run_full_pipeline.sh train-stage2
PIPELINE_CONFIG=$PWD/configs/full_scale.env bash scripts/run_full_pipeline.sh evaluate
```

The pipeline reads the three portable recipes, derives exact one-pass steps,
selects a task-balanced held-out validation monitor, and gates Stage 2 on a
completed Stage 1. Test files are never combined with training or validation.

## Test Protocols

- HBB detection: DIOR test and DOTAv2 validation.
- Phrase grounding: DIOR-RSVG test and VRSBench-VG validation.
- Pointing: DOTAv2 Balanced-100 class-wise diagnostic. It is not claimed as a
  standardized public pointing benchmark.

## Source and License Notice

This package is a derived research compilation. Source datasets retain their own
licenses and terms, which may differ. The corpus includes records derived from
LAE-1M/LAE-FOD, DIOR/DIOR-RSVG, DOTAv2, DOTA-v1.5, FAIR1M, xView, HRSC2016,
RSOD, NWPU-VHR-10, Power-Plant, KFGOD, SODA-A, and VRSBench. KFGOD is marked
CC BY-SA 4.0; VRSBench's local card contains inconsistent CC BY 4.0 front matter
and CC BY-NC 4.0 prose, so users should follow the more restrictive source terms
unless the authors clarify them. Verify every original dataset's current terms
before redistribution or commercial use.

No model weights are included. See `manifest.json` for exact counts and leakage
audits, and use `sha256sum -c SHA256SUMS` after download.
"""
        path = self.output / "README.md"
        path.write_text(readme, encoding="utf-8")
        self.output_files.append(path)

    def write_checksums(self) -> None:
        checksum_path = self.output / "SHA256SUMS"
        files = sorted(path for path in self.output_files if path.is_file() and path != checksum_path)
        with checksum_path.open("w", encoding="utf-8") as handle:
            for path in files:
                handle.write(f"{sha256_file(path)}  {path.relative_to(self.output).as_posix()}\n")


def verify_package(root: Path, *, verify_image_hashes: bool) -> dict[str, Any]:
    root = root.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    jsonl_files = sorted((root / "annotations").rglob("*.jsonl"))
    image_references: dict[str, set[str]] = defaultdict(set)
    records = 0
    missing = []
    bad_hashes = []
    for path in jsonl_files:
        split = path.relative_to(root / "annotations").parts[0]
        split = "train" if split == "train" else "validation" if split.startswith("validation") else "test"
        for _, row in read_jsonl(path):
            values = image_items(row)
            if not values and row.get("image_path"):
                values = [row["image_path"]]
            for value in values:
                relative = Path(image_path_value(value))
                image = root / relative
                if not image.is_file():
                    missing.append(relative.as_posix())
                    continue
                digest = image.stem
                image_references[split].add(digest)
            records += 1
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} referenced images; first={missing[:5]}")
    overlaps = {
        "train_validation": len(image_references["train"] & image_references["validation"]),
        "train_test": len(image_references["train"] & image_references["test"]),
        "validation_test": len(image_references["validation"] & image_references["test"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"Split leakage: {overlaps}")
    if verify_image_hashes:
        for image in sorted((root / "images").rglob("*")):
            if image.is_file() and sha256_file(image) != image.stem:
                bad_hashes.append(str(image.relative_to(root)))
        if bad_hashes:
            raise ValueError(f"Content-address mismatches: {bad_hashes[:5]}")
    result = {
        "status": "passed",
        "jsonl_files": len(jsonl_files),
        "records": records,
        "referenced_unique_images": {key: len(value) for key, value in sorted(image_references.items())},
        "image_hash_overlap": overlaps,
        "manifest_schema": manifest.get("schema_version"),
        "full_image_hash_verification": verify_image_hashes,
    }
    write_json(root / "verification.json", result)
    return result


def finalize_existing(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing built manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    packager = Packager(
        root,
        root,
        root,
        root,
        link_mode="hardlink",
        shard_mib=180,
        verify_content=False,
    )
    packager.write_readme(manifest)
    packager.output_files = sorted((root / "annotations").rglob("*.jsonl"))
    packager.output_files.extend(sorted((root / "recipes").glob("*.json")))
    packager.output_files.extend((manifest_path, root / "README.md"))
    if (root / "verification.json").is_file():
        packager.output_files.append(root / "verification.json")
    packager.write_checksums()
    return {
        "status": "finalized",
        "root": str(root),
        "checksummed_metadata_files": len(packager.output_files),
    }


def plan(curriculum: Path) -> dict[str, Any]:
    paths = sorted(curriculum.rglob("*.jsonl"))
    records = 0
    bytes_total = 0
    for path in paths:
        bytes_total += path.stat().st_size
        with path.open("rb") as handle:
            records += sum(1 for line in handle if line.strip())
    return {
        "curriculum": str(curriculum.resolve()),
        "jsonl_files": len(paths),
        "jsonl_bytes": bytes_total,
        "jsonl_gib": bytes_total / 1024**3,
        "jsonl_records_including_indexes_and_quarantine": records,
    }


def main() -> None:
    release_root = Path(__file__).resolve().parents[1]
    workspace_root = release_root.parent
    project_root = workspace_root / "Zero_shot_ anlysis_LA"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "build", "finalize", "verify"))
    parser.add_argument("--source-root", type=Path, default=workspace_root)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=project_root,
    )
    parser.add_argument(
        "--curriculum",
        type=Path,
        default=project_root / "dataset/sft/pixel_pivr_type2_hbb_full_v1",
    )
    parser.add_argument("--output", type=Path, default=workspace_root / "Pixel-PIVR-HF")
    parser.add_argument("--link-mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--shard-mib", type=int, default=180)
    parser.add_argument("--trust-record-hashes", action="store_true")
    parser.add_argument("--verify-image-hashes", action="store_true")
    args = parser.parse_args()

    for path in (args.source_root, args.project_root, args.curriculum):
        if not path.exists():
            parser.error(f"Missing input: {path}")
    if args.shard_mib < 16:
        parser.error("--shard-mib must be at least 16")

    if args.mode == "plan":
        result = plan(args.curriculum)
    elif args.mode == "build":
        packager = Packager(
            args.source_root,
            args.project_root,
            args.curriculum,
            args.output,
            link_mode=args.link_mode,
            shard_mib=args.shard_mib,
            verify_content=not args.trust_record_hashes,
        )
        result = packager.build()
    elif args.mode == "finalize":
        result = finalize_existing(args.output)
    else:
        result = verify_package(args.output, verify_image_hashes=args.verify_image_hashes)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
