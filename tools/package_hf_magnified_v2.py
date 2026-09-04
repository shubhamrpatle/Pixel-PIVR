#!/usr/bin/env python3
"""Build and verify the portable 144-to-384 Pixel-PIVR full-scale corpus."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from package_hf_dataset import (  # noqa: E402
    Packager,
    ShardedWriter,
    image_items,
    image_path_value,
    normalize_suffix,
    read_jsonl,
    sha256_file,
    write_json,
)
from pixel_pivr.virtual_crop import (  # noqa: E402
    ANSWER_SCHEMA,
    CONTAINMENT_TOLERANCE_PIXELS,
    FALLBACK_EDGE_MARGIN_PIXELS,
    LOCAL_INPUT_SIDE,
    PROMPT_SCHEMA,
    SOURCE_CROP_SIDE,
    VIRTUAL_CROP_SCHEMA,
    box_token,
    compact_prompt,
    geometry_in_crop,
    local_box_touches_fallback_margin,
    point_inside,
    point_centered_crop,
    target_fits_crop,
    transform_round2_records,
)


SCHEMA = "pixel-pivr-hf-hbb-magnified-v2"
PROCESSOR_ALIGNED_SIDE = 392


def conversation_text(row: Mapping[str, Any], role: str) -> str:
    accepted = {"human", "user"} if role == "human" else {"gpt", "assistant"}
    values = [
        str(turn.get("value") or "")
        for turn in row.get("conversations") or []
        if str(turn.get("from") or "").lower() in accepted
    ]
    if len(values) != 1:
        raise ValueError(f"Expected exactly one {role} turn, found {len(values)}")
    return values[0]


def image_digest_from_path(path: Path) -> str:
    stem = path.stem
    if len(stem) != 64 or any(char not in "0123456789abcdef" for char in stem):
        raise ValueError(f"Image is not content addressed: {path}")
    return stem


class MagnifiedPackager(Packager):
    def __init__(
        self,
        *args: Any,
        source_crop_side: int,
        local_input_side: int,
        existing_image_root: Path | None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.source_crop_side = int(source_crop_side)
        self.local_input_side = int(local_input_side)
        self.existing_image_root = (
            None if existing_image_root is None else existing_image_root.resolve()
        )
        self.dimension_cache: dict[Path, tuple[int, int]] = {}
        self.transform_stats: Counter[str] = Counter()
        self.monitor_stats: Counter[str] = Counter()
        self.validation_monitor_records = 0
        self.image_output_paths: set[Path] = set()

    def dimensions(self, path: Path) -> tuple[int, int]:
        path = path.resolve()
        value = self.dimension_cache.get(path)
        if value is None:
            with Image.open(path) as image:
                value = tuple(map(int, image.size))
            self.dimension_cache[path] = value
        return value

    def materialize_image(
        self, path: Path, split: str, expected_hash: str | None = None
    ) -> tuple[str, str]:
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
        else:
            reusable = None
            if self.existing_image_root is not None:
                candidates = list(
                    (self.existing_image_root / "images" / split / digest[:2]).glob(
                        f"{digest}.*"
                    )
                )
                if len(candidates) > 1:
                    raise ValueError(f"Ambiguous reusable image for {digest}: {candidates}")
                reusable = candidates[0] if candidates else None
            source = reusable or path
            if reusable is not None and self.verify_content and sha256_file(reusable) != digest:
                raise ValueError(f"Reusable image hash mismatch: {reusable}")
            if self.link_mode == "hardlink":
                try:
                    os.link(source, destination)
                except OSError:
                    import shutil

                    shutil.copy2(source, destination)
            else:
                import shutil

                shutil.copy2(source, destination)
        portable = relative.as_posix()
        self.hash_paths[split][digest] = portable
        self.split_hashes[split].add(digest)
        self.image_counts[split] += 1
        self.image_bytes[split] += destination.stat().st_size
        self.image_output_paths.add(destination)
        return portable, digest

    def rewrite_training_rows(
        self,
        row: dict[str, Any],
        split: str,
        *,
        stats_scope: str | None = None,
    ) -> list[dict[str, Any]]:
        stats_scope = stats_scope or split
        values = image_items(row)
        if len(values) != 1:
            raise ValueError(f"Source curriculum must contain one image, got {len(values)}")
        raw = image_path_value(values[0])
        expected = str(row.get("meta", {}).get("image_content_sha256") or "") or None
        source = self.resolve_image(raw)
        portable, digest = self.materialize_image(source, split, expected)
        route = str(row.get("meta", {}).get("pivr_route") or "")
        if route == "global_point_discovery":
            output = copy.deepcopy(row)
            output["image"] = portable
            output.setdefault("meta", {}).update(
                {
                    "image_content_sha256": digest,
                    "pivr_round": 1,
                    "pivr_visual_context": "global",
                    "pivr_visual_inputs": 1,
                }
            )
            self.transform_stats[f"{stats_scope}.round1_global"] += 1
            return [output]
        if route != "point_indexed_visual_reentry":
            raise ValueError(f"Unknown PIVR route {route!r}")
        width, height = self.dimensions(source)
        outputs = transform_round2_records(
            row,
            portable_image=portable,
            width=width,
            height=height,
            source_crop_side=self.source_crop_side,
            local_input_side=self.local_input_side,
        )
        for output in outputs:
            output["meta"]["image_content_sha256"] = digest
            context = str(output["meta"]["pivr_visual_context"])
            polarity = str(output["meta"].get("pivr_reentry_polarity") or "unknown")
            task = str(output["meta"].get("task") or "unknown")
            self.transform_stats[f"{stats_scope}.round2.{context}"] += 1
            self.transform_stats[
                f"{stats_scope}.round2.{task}.{polarity}.{context}"
            ] += 1
            reason = str(output["meta"].get("pivr_fallback_reason") or "none")
            self.transform_stats[f"{stats_scope}.round2.fallback_reason.{reason}"] += 1
        self.transform_stats[f"{stats_scope}.round2.source_records"] += 1
        self.transform_stats[f"{stats_scope}.round2.output_records"] += len(outputs)
        return outputs

    def rewrite_training_row(self, row: dict[str, Any], split: str) -> dict[str, Any]:
        """Reject accidental use of the legacy one-input/one-output package path."""
        outputs = self.rewrite_training_rows(row, split)
        if len(outputs) != 1:
            raise RuntimeError(
                "Magnified-v2 expanded a Round-2 row; use rewrite_training_rows"
            )
        return outputs[0]

    def _write_training_outputs(
        self,
        *,
        source_row: dict[str, Any],
        split: str,
        task: str,
        stage: str | None,
        route: str,
        writers: dict[tuple[str, str], ShardedWriter],
    ) -> None:
        for row in self.rewrite_training_rows(source_row, split):
            key = (task, route)
            prefix = (
                self.output / "annotations" / split / str(stage) / task / route
                if stage is not None
                else self.output / "annotations" / split / task / route
            )
            writer = writers.setdefault(key, ShardedWriter(prefix, self.shard_bytes))
            writer.write(row)
            meta = row["meta"]
            stat_prefix = (
                f"train.{stage}.{task}.{route}"
                if stage is not None
                else f"validation.{task}.{route}"
            )
            self.record_stats[f"{stat_prefix}.records"] += 1
            self.record_stats[f"{stat_prefix}.targets"] += int(
                meta.get("target_count", 0)
            )
            if stage is not None:
                self.dataset_stats[
                    f"train.{stage}.{task}.{meta.get('dataset', 'unknown')}.records"
                ] += 1
                if stage == "stage2_dense_balanced":
                    partition = (
                        "replay" if meta.get("stage2_replay") is True else "dense"
                    )
                    self.record_stats[
                        f"train.{stage}.partition.{partition}.records"
                    ] += 1
                    self.record_stats[
                        f"train.{stage}.partition.{partition}.{task}.{route}.records"
                    ] += 1

    def package_training(self) -> dict[str, Any]:
        definitions = {
            "stage1_coarse": {
                "round1_global_point": sorted(
                    (self.curriculum / "round1_global_point/stage1_coarse").glob("*.jsonl")
                ),
                "round2_point_box": sorted(
                    (self.curriculum / "round2_point_box/stage1_coarse").glob("*.jsonl")
                ),
            },
            "stage2_dense_balanced": {
                "round1_global_point": sorted(
                    (self.curriculum / "round1_global_point/stage2_dense_balanced").glob("*.jsonl")
                ),
                "round2_point_box": sorted(
                    (self.curriculum / "round2_point_box/stage2_dense_balanced").glob("*.jsonl")
                ),
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
                            raise ValueError("Pointing must not have a point-to-box record")
                        self._write_training_outputs(
                            source_row=row,
                            split="train",
                            task=task,
                            stage=stage,
                            route=route,
                            writers=writers,
                        )
            for writer in writers.values():
                writer.close()
                self.output_files.extend(writer.paths)
            recipes[stage] = {
                task: {
                    route: [
                        str(path.relative_to(self.output).as_posix())
                        for path in writers[(task, route)].paths
                    ]
                    for route in routes
                    if (task, route) in writers
                }
                for task in ("detection", "grounding", "pointing")
            }
        return recipes

    def package_validation(self) -> dict[str, Any]:
        routes = {
            "round1_global_point": sorted(
                (self.curriculum / "round1_global_point/validation").glob("*.jsonl")
            ),
            "round2_point_box": sorted(
                (self.curriculum / "round2_point_box/validation").glob("*.jsonl")
            ),
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
                    self._write_training_outputs(
                        source_row=row,
                        split="validation",
                        task=task,
                        stage=None,
                        route=route,
                        writers=writers,
                    )
        for writer in writers.values():
            writer.close()
            self.output_files.extend(writer.paths)

        monitor_source = self.curriculum / "validation/round2_refinement_monitor1000.jsonl"
        monitor_writer = ShardedWriter(
            self.output / "annotations/validation_monitor/round2_point_box",
            self.shard_bytes,
        )
        for _, row in read_jsonl(monitor_source):
            for output in self.rewrite_training_rows(
                row, "validation", stats_scope="validation_monitor"
            ):
                monitor_writer.write(output)
                meta = output.get("meta") or {}
                self.monitor_stats[
                    ".".join(
                        (
                            str(meta.get("task") or "unknown"),
                            str(meta.get("pivr_reentry_polarity") or "unknown"),
                            str(meta.get("pivr_round2_route") or "unknown"),
                        )
                    )
                ] += 1
        monitor_writer.close()
        self.output_files.extend(monitor_writer.paths)
        round1_monitor = []
        for task in ("detection", "grounding", "pointing"):
            writer = writers.get((task, "round1_global_point"))
            if writer is not None:
                round1_monitor.extend(
                    path.relative_to(self.output).as_posix() for path in writer.paths
                )
                self.monitor_stats[f"{task}.all.round1_global"] += writer.records
        self.validation_monitor_records = sum(
            writer.records
            for (_task, route), writer in writers.items()
            if route == "round1_global_point"
        ) + monitor_writer.records
        return {
            task: {
                route: [
                    str(path.relative_to(self.output).as_posix())
                    for path in writers[(task, route)].paths
                ]
                for route in routes
                if (task, route) in writers
            }
            for task in ("detection", "grounding", "pointing")
        } | {
            "monitor": round1_monitor + [
                str(path.relative_to(self.output).as_posix())
                for path in monitor_writer.paths
            ]
        }

    def write_recipes(self, recipes: Mapping[str, Any]) -> None:
        for stage in ("stage1_coarse", "stage2_dense_balanced"):
            files = []
            for task in ("detection", "grounding", "pointing"):
                for route in ("round1_global_point", "round2_point_box"):
                    files.extend(recipes["train"][stage].get(task, {}).get(route, []))
            write_json(
                self.output / "recipes" / f"{stage}.json",
                {
                    "schema_version": "pixel-pivr-recipe-v2",
                    "data_root": ".",
                    "annotation": files,
                    "repeat_time": 1.0,
                    "data_augment": False,
                    "coverage": "every listed record exactly once before explicit batch padding",
                },
            )
            self.output_files.append(self.output / "recipes" / f"{stage}.json")

        validation_files = []
        for task in ("detection", "grounding", "pointing"):
            for route in ("round1_global_point", "round2_point_box"):
                validation_files.extend(recipes["validation"].get(task, {}).get(route, []))
        for name, files, purpose in (
            (
                "validation_all_tasks.json",
                validation_files,
                "full independent validation pool; never used for gradient updates",
            ),
            (
                "validation_monitor_all_tasks.json",
                recipes["validation"]["monitor"],
                "fixed checkpoint-selection monitor spanning Round 1 and Round 2",
            ),
        ):
            path = self.output / "recipes" / name
            write_json(
                path,
                {
                    "schema_version": "pixel-pivr-recipe-v2",
                    "data_root": ".",
                    "annotation": files,
                    "repeat_time": 1.0,
                    "data_augment": False,
                    "purpose": purpose,
                },
            )
            self.output_files.append(path)

    def build(self) -> dict[str, Any]:
        if self.source_crop_side != SOURCE_CROP_SIDE or self.local_input_side != LOCAL_INPUT_SIDE:
            raise ValueError(
                "The release contract is frozen at source_crop_side=144 and local_input_side=384"
            )
        if self.output.exists() and any(self.output.iterdir()):
            raise FileExistsError(f"Output must be absent or empty: {self.output}")
        self.output.mkdir(parents=True, exist_ok=True)
        recipes = {
            "train": self.package_training(),
            "validation": self.package_validation(),
            "test": self.package_public_tests(),
        }
        overlaps = self.assert_split_separation()
        stage2_replay_records = int(
            self.record_stats[
                "train.stage2_dense_balanced.partition.replay.records"
            ]
        )
        stage2_dense_records = int(
            self.record_stats[
                "train.stage2_dense_balanced.partition.dense.records"
            ]
        )
        stage2_total_records = stage2_dense_records + stage2_replay_records
        stage2_replay_query_records = sum(
            int(
                self.record_stats[
                    "train.stage2_dense_balanced.partition.replay."
                    f"{task}.round1_global_point.records"
                ]
            )
            for task in ("detection", "grounding", "pointing")
        )
        stage2_dense_query_records = sum(
            int(
                self.record_stats[
                    "train.stage2_dense_balanced.partition.dense."
                    f"{task}.round1_global_point.records"
                ]
            )
            for task in ("detection", "grounding", "pointing")
        )
        stage2_total_query_records = (
            stage2_dense_query_records + stage2_replay_query_records
        )
        def stage_records(stage: str) -> int:
            return sum(
                int(self.record_stats[
                    f"train.{stage}.{task}.{route}.records"
                ])
                for task in ("detection", "grounding", "pointing")
                for route in ("round1_global_point", "round2_point_box")
            )

        validation_records = sum(
            int(self.record_stats[f"validation.{task}.{route}.records"])
            for task in ("detection", "grounding", "pointing")
            for route in ("round1_global_point", "round2_point_box")
        )
        manifest = {
            "schema_version": SCHEMA,
            "source_curriculum": self.curriculum.name,
            "coordinate_space": "normalized_0_1000",
            "geometry": "horizontal bounding boxes and points",
            "pixel_reentry": {
                "source_crop_side_pixels": self.source_crop_side,
                "containment_tolerance_pixels": CONTAINMENT_TOLERANCE_PIXELS,
                "fallback_edge_margin_pixels": FALLBACK_EDGE_MARGIN_PIXELS,
                "local_input_side_pixels": self.local_input_side,
                "la_processor_aligned_side_pixels": PROCESSOR_ALIGNED_SIDE,
                "upscale_factor": self.local_input_side / self.source_crop_side,
                "resample": "PIL.Image.Resampling.LANCZOS",
                "la_alignment_resample": "PIL.Image.Resampling.BICUBIC",
                "virtual_crop_schema": VIRTUAL_CROP_SCHEMA,
                "prompt_schema": PROMPT_SCHEMA,
                "answer_schema": ANSWER_SCHEMA,
                "positive_overflow_policy": (
                    "local None gate plus paired global point-to-box fallback; "
                    "complete local edge boxes retain their target and receive a "
                    "paired global fallback; no oracle routing, clipping, or target loss"
                ),
                "negative_fallback_policy": (
                    "local None plus paired global None, matching inference retries"
                ),
                "inference_routing": (
                    "accept a valid interior local box; retry globally after local "
                    "None, invalid geometry, or a crop-edge box"
                ),
                "moonvit_encodes_for_local_route": 2,
            },
            "records": dict(sorted(self.record_stats.items())),
            "records_by_source_dataset": dict(sorted(self.dataset_stats.items())),
            "transformation_records": dict(sorted(self.transform_stats.items())),
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
            "training_contract": {
                "curriculum": "Stage 1 coarse, then Stage 2 dense plus source-defined replay",
                "stage1_records": stage_records("stage1_coarse"),
                "stage2_records": stage_records("stage2_dense_balanced"),
                "stage2_dense_records": stage2_dense_records,
                "stage2_replay_records": stage2_replay_records,
                "stage2_replay_percent": (
                    100.0 * stage2_replay_records / stage2_total_records
                ),
                "stage2_source_query_dense_records": stage2_dense_query_records,
                "stage2_source_query_replay_records": stage2_replay_query_records,
                "stage2_source_query_replay_percent": (
                    100.0
                    * stage2_replay_query_records
                    / stage2_total_query_records
                ),
                "validation_records": validation_records,
                "validation_monitor_records": self.validation_monitor_records,
                "validation_monitor_composition": dict(sorted(self.monitor_stats.items())),
                "recommended_image_token_limit": 6000,
                "required_max_sequence": 32768,
                "one_exact_record_pass_per_stage": True,
            },
            "notes": {
                "stage_overlap": "Stage 1 and Stage 2 are curricula within train and may reuse images.",
                "validation": "Independent checkpoint-selection split; excluded from train and public benchmarks.",
                "pointing_test": "DOTAv2 Balanced-100 diagnostic, not a standardized public pointing benchmark.",
                "storage": "Virtual crops reference source images; local crop pixels are produced in memory.",
            },
        }
        manifest_path = self.output / "manifest.json"
        write_json(manifest_path, manifest)
        self.output_files.append(manifest_path)
        self.write_recipes(recipes)
        self.write_readme(manifest)
        self.write_image_inventory()
        self.write_checksums()
        return manifest

    def write_image_inventory(self) -> None:
        path = self.output / "IMAGE_INVENTORY.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for image in sorted(self.image_output_paths):
                relative = image.relative_to(self.output).as_posix()
                handle.write(
                    json.dumps(
                        {
                            "bytes": image.stat().st_size,
                            "path": relative,
                            "sha256": image_digest_from_path(image),
                        },
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                )
        self.output_files.append(path)

    def write_readme(self, manifest: Mapping[str, Any]) -> None:
        records = manifest["training_contract"]
        images = manifest["images"]
        scopes = ("train", "validation", "validation_monitor")
        local_rows = sum(
            int(manifest["transformation_records"].get(
                f"{scope}.round2.global_plus_pixel_crop_144to384_local_gate", 0
            ))
            for scope in scopes
        )
        fallback_rows = sum(
            int(manifest["transformation_records"].get(
                f"{scope}.round2.global_only_point_box_fallback", 0
            ))
            for scope in scopes
        )
        text = f"""---
pretty_name: Pixel-PIVR 144-to-384 Remote-Sensing HBB
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

# Pixel-PIVR 144-to-384 Full-Scale Dataset

This is the versioned `{SCHEMA}` package. It is the **real pixel re-encoding**
variant, not the older cached pre-projector ROI variant.

Round 1 encodes the global image and predicts target-complete point addresses.
Round 2 first encodes the same global image plus a deterministic 144 x 144
source-pixel crop resized to 384 x 384 with Lanczos, then predicts exactly one box
or `<box>None</box>`. There are {local_rows:,} local-gate rows. A local `None`,
invalid box, or crop-edge box triggers one of {fallback_rows:,} paired global
fallback rows. These include both large positive targets and negative addresses,
so inference never selects a route using ground truth. No target or box is clipped.

LocateAnything's unchanged image processor subsequently aligns 384 x 384 to
392 x 392 (28 x 28 MoonViT patches, or 14 x 14 = 196 projected visual tokens).

| Split | Records | Unique images | Image bytes |
|---|---:|---:|---:|
| Stage 1 coarse | {records['stage1_records']:,} | shared train pool | - |
| Stage 2 dense + replay | {records['stage2_records']:,} | shared train pool | - |
| Validation | {records['validation_records']:,} | {images['validation']['unique']:,} | {images['validation']['gib']:.2f} GiB |
| Checkpoint-selection monitor | {records['validation_monitor_records']:,} | same validation pool | - |
| Test | benchmark protocols | {images['test']['unique']:,} | {images['test']['gib']:.2f} GiB |

The train, validation, and test image SHA-256 sets have zero overlap. Stage 1 and
Stage 2 may intentionally reuse training images. Use the recipes under `recipes/`;
do not construct splits from directory names.

Stage 2 contains {records['stage2_dense_records']:,} dense optimizer records and
{records['stage2_replay_records']:,} replay optimizer records
({records['stage2_replay_percent']:.2f}% replay after point-to-box expansion).
Before expansion, its Round-1 source-query schedule contains
{records['stage2_source_query_dense_records']:,} dense and
{records['stage2_source_query_replay_records']:,} replay records
({records['stage2_source_query_replay_percent']:.2f}% replay). These percentages
describe different units and must not be interchanged.

## Integrity

`SHA256SUMS` covers every annotation, recipe, manifest, documentation file, and
every image. `IMAGE_INVENTORY.jsonl` provides a machine-readable image inventory.
The training repository preflight additionally verifies this schema, crop contract,
record counts, GPU topology, sequence budget, and the pinned Eagle compatibility
patch before training can start.

## Download

```bash
hf download shubhampatle/Pixel-PIVR-Magnified-v2 \
  --repo-type dataset --local-dir Pixel-PIVR-Magnified-v2
```

The full operator guide is in the Pixel-PIVR code repository under
`docs/A100_8GPU_FULL_SCALE_RUNBOOK.md`.

## License note

This is a derived research compilation. Every source dataset retains its own terms.
Review the source licenses before redistribution or commercial use. No model weights
are included.
"""
        path = self.output / "README.md"
        path.write_text(text, encoding="utf-8")
        self.output_files.append(path)

    def write_checksums(self) -> None:
        checksum_path = self.output / "SHA256SUMS"
        metadata = sorted(
            path for path in set(self.output_files) if path.is_file() and path != checksum_path
        )
        images = sorted(self.image_output_paths)
        with checksum_path.open("w", encoding="utf-8") as handle:
            for path in metadata:
                handle.write(
                    f"{sha256_file(path)}  {path.relative_to(self.output).as_posix()}\n"
                )
            for path in images:
                handle.write(
                    f"{image_digest_from_path(path)}  {path.relative_to(self.output).as_posix()}\n"
                )


def validate_virtual_crop(
    value: Mapping[str, Any], *, source: str, width: int, height: int
) -> None:
    if value.get("virtual_crop") is not True or value.get("schema") != VIRTUAL_CROP_SCHEMA:
        raise ValueError("Round-2 local input has the wrong virtual-crop schema")
    if value.get("path") != source:
        raise ValueError("Virtual crop does not reference its global source image")
    crop = value.get("crop_xyxy")
    if not isinstance(crop, list) or len(crop) != 4 or any(
        isinstance(item, bool) or not isinstance(item, int) for item in crop
    ):
        raise ValueError(f"Invalid crop_xyxy: {crop!r}")
    left, top, right, bottom = crop
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError(f"Crop {crop} is outside image {(width, height)}")
    if right - left != SOURCE_CROP_SIDE or bottom - top != SOURCE_CROP_SIDE:
        raise ValueError(f"Virtual crop is not {SOURCE_CROP_SIDE} x {SOURCE_CROP_SIDE}")
    if value.get("resize_hw") != [LOCAL_INPUT_SIDE, LOCAL_INPUT_SIDE]:
        raise ValueError("Virtual crop is not resized to the frozen 384 x 384 input")
    if str(value.get("resample") or "").lower() != "lanczos":
        raise ValueError("Virtual crop does not use Lanczos resampling")


def verify_package(root: Path, *, verify_image_hashes: bool) -> dict[str, Any]:
    root = root.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA:
        raise ValueError(f"Wrong schema: {manifest.get('schema_version')!r}")
    pixel = manifest.get("pixel_reentry") or {}
    expected_pixel = {
        "source_crop_side_pixels": SOURCE_CROP_SIDE,
        "containment_tolerance_pixels": CONTAINMENT_TOLERANCE_PIXELS,
        "fallback_edge_margin_pixels": FALLBACK_EDGE_MARGIN_PIXELS,
        "local_input_side_pixels": LOCAL_INPUT_SIDE,
        "la_processor_aligned_side_pixels": PROCESSOR_ALIGNED_SIDE,
        "virtual_crop_schema": VIRTUAL_CROP_SCHEMA,
        "prompt_schema": PROMPT_SCHEMA,
        "answer_schema": ANSWER_SCHEMA,
    }
    for key, value in expected_pixel.items():
        if pixel.get(key) != value:
            raise ValueError(f"Manifest crop contract mismatch for {key}: {pixel.get(key)!r}")

    image_dimensions: dict[str, tuple[int, int]] = {}
    referenced: dict[str, set[str]] = defaultdict(set)
    counts: Counter[str] = Counter()
    local_fallback_expected: set[str] = set()
    global_fallback_seen: set[str] = set()
    record_ids: dict[str, set[str]] = defaultdict(set)
    observed_stage2_partitions: Counter[str] = Counter()
    errors: list[str] = []
    for path in sorted((root / "annotations").rglob("*.jsonl")):
        relative = path.relative_to(root / "annotations")
        primary = relative.parts[0]
        split = "train" if primary == "train" else "validation" if primary.startswith("validation") else "test"
        is_training_contract = primary in {"train", "validation", "validation_monitor"}
        if primary == "train":
            record_scope = "/".join(relative.parts[:2])
        elif primary == "validation":
            record_scope = "validation"
        elif primary == "validation_monitor":
            record_scope = "validation_monitor"
        else:
            record_scope = "/".join(relative.parts[:3])
        for line_number, row in read_jsonl(path):
            try:
                values = image_items(row)
                if not values and row.get("image_path"):
                    values = [row["image_path"]]
                if not values:
                    raise ValueError("Record has no image")
                global_path = image_path_value(values[0])
                global_file = root / global_path
                if not global_file.is_file():
                    raise FileNotFoundError(global_path)
                digest = image_digest_from_path(global_file)
                referenced[split].add(digest)
                if is_training_contract:
                    prompt = conversation_text(row, "human")
                    answer = conversation_text(row, "gpt")
                    meta = row.get("meta") or {}
                    record_id = str(meta.get("record_id") or "")
                    if not record_id:
                        raise ValueError("Record has no record_id")
                    if record_id in record_ids[record_scope]:
                        raise ValueError(
                            f"Duplicate record_id {record_id!r} in {record_scope}"
                        )
                    record_ids[record_scope].add(record_id)
                    if (
                        primary == "train"
                        and len(relative.parts) > 1
                        and relative.parts[1] == "stage2_dense_balanced"
                    ):
                        partition = (
                            "replay" if meta.get("stage2_replay") is True else "dense"
                        )
                        observed_stage2_partitions[partition] += 1
                        observed_stage2_partitions[
                            f"{partition}.{meta.get('task', 'unknown')}."
                            f"{relative.parts[-2]}"
                        ] += 1
                    route = str(meta.get("pivr_route") or "")
                    if route == "global_point_discovery":
                        if len(values) != 1 or prompt.count("<image>") != 1:
                            raise ValueError("Round 1 must have one global visual input")
                        counts[f"{split}.round1"] += 1
                    elif route == "point_indexed_visual_reentry":
                        if meta.get("answer_format") != ANSWER_SCHEMA:
                            raise ValueError("Round 2 answer schema mismatch")
                        obsolete = {
                            "pivr_local_context_source",
                            "pivr_local_roi_may_not_cover_full_target",
                        } & set(meta)
                        if obsolete:
                            raise ValueError(
                                "Round 2 retains superseded cached-ROI metadata: "
                                f"{sorted(obsolete)}"
                            )
                        if "<ref>" in answer or not answer.startswith("<box>"):
                            raise ValueError("Round 2 must use box-only output")
                        if answer == "<box>None</box>":
                            if int(meta.get("target_count", -1)) != 0:
                                raise ValueError("None output has a nonzero target count")
                        elif int(meta.get("target_count", -1)) != 1:
                            raise ValueError("Positive box output does not have target_count=1")
                        context = str(meta.get("pivr_visual_context") or "")
                        source_record_id = str(meta.get("pivr_source_record_id") or "")
                        if not source_record_id:
                            raise ValueError("Round 2 has no pivr_source_record_id")
                        fallback_pair_key = (
                            f"{relative.parent.parent.as_posix()}::{source_record_id}"
                        )
                        if context == "global_plus_pixel_crop_144to384_local_gate":
                            if len(values) != 2 or not isinstance(values[1], Mapping):
                                raise ValueError("Local Round 2 must have global plus virtual crop")
                            size = image_dimensions.get(global_path)
                            if size is None:
                                with Image.open(global_file) as image:
                                    size = tuple(map(int, image.size))
                                image_dimensions[global_path] = size
                            validate_virtual_crop(
                                values[1], source=global_path, width=size[0], height=size[1]
                            )
                            if prompt.count("<image>") != 2:
                                raise ValueError("Local Round-2 prompt must contain two image placeholders")
                            target = meta.get("pivr_local_target_box")
                            point = meta.get("pivr_local_point")
                            global_point = meta.get("pivr_global_point")
                            global_target = meta.get("pivr_global_target_box")
                            reference = str(meta.get("pivr_reference") or "").strip()
                            if not reference:
                                raise ValueError("Round 2 has no pivr_reference")
                            expected_crop = list(
                                point_centered_crop(
                                    global_point,
                                    size[0],
                                    size[1],
                                    SOURCE_CROP_SIDE,
                                )
                            )
                            if values[1]["crop_xyxy"] != expected_crop:
                                raise ValueError(
                                    "Virtual crop differs from the deterministic point-centred crop"
                                )
                            if meta.get("pivr_containment_tolerance_pixels") != 0.0:
                                raise ValueError("Local containment tolerance must be exactly zero")
                            if float(meta.get("pivr_fallback_edge_margin_pixels", -1)) != float(
                                FALLBACK_EDGE_MARGIN_PIXELS
                            ):
                                raise ValueError("Local fallback edge margin is not frozen")
                            if target is not None and not point_inside(point, target):
                                raise ValueError("Local point is outside the local GT box")
                            crop = values[1]["crop_xyxy"]
                            exact_fit = bool(
                                global_target is None
                                or target_fits_crop(
                                    global_target, crop, size[0], size[1]
                                )
                            )
                            if bool(meta.get("pivr_target_fully_contained")) != exact_fit:
                                raise ValueError("Local target containment flag is not exact")
                            if target is not None and not exact_fit:
                                raise ValueError("A clipped target was emitted in local coordinates")
                            expected_point, expected_target = geometry_in_crop(
                                global_point,
                                global_target if exact_fit else None,
                                crop,
                                size[0],
                                size[1],
                            )
                            if point != expected_point or target != expected_target:
                                raise ValueError(
                                    "Stored local point/box differs from exact global-to-crop geometry"
                                )
                            if prompt != compact_prompt(reference, expected_point, "local"):
                                raise ValueError("Local Round-2 prompt differs from the frozen template")
                            if answer != box_token(expected_target):
                                raise ValueError("Local Round-2 answer differs from its exact target")
                            touches_edge = bool(
                                expected_target is not None
                                and local_box_touches_fallback_margin(
                                    expected_target,
                                    LOCAL_INPUT_SIDE,
                                    FALLBACK_EDGE_MARGIN_PIXELS,
                                )
                            )
                            if bool(
                                meta.get("pivr_target_touches_local_fallback_margin")
                            ) != touches_edge:
                                raise ValueError("Local crop-edge fallback flag is inconsistent")
                            fallback_required = bool(meta.get("pivr_fallback_required"))
                            expected_fallback = bool(
                                global_target is None or not exact_fit or touches_edge
                            )
                            if fallback_required != expected_fallback:
                                raise ValueError("Local fallback flag differs from observable policy")
                            expected_reason = (
                                "negative_address"
                                if global_target is None
                                else (
                                    "target_not_fully_contained_in_fixed_144px_crop"
                                    if not exact_fit
                                    else "complete_target_touches_local_fallback_margin"
                                    if touches_edge
                                    else None
                                )
                            )
                            if meta.get("pivr_fallback_reason") != expected_reason:
                                raise ValueError("Local fallback reason is inconsistent")
                            if fallback_required:
                                if (global_target is None or not exact_fit) and answer != "<box>None</box>":
                                    raise ValueError("Negative/overflow local fallback must emit None")
                                if touches_edge and answer == "<box>None</box>":
                                    raise ValueError("Complete edge target must retain its local box")
                                if fallback_pair_key in local_fallback_expected:
                                    raise ValueError("Duplicate fallback-required local source record")
                                local_fallback_expected.add(fallback_pair_key)
                            elif answer == "<box>None</box>" or target is None:
                                raise ValueError("A local None row must request a global fallback")
                            counts[f"{split}.round2.local"] += 1
                        elif context == "global_only_point_box_fallback":
                            if len(values) != 1 or prompt.count("<image>") != 1:
                                raise ValueError("Global fallback must have one visual input")
                            if meta.get("pivr_fallback_reason") not in {
                                "negative_address",
                                "target_not_fully_contained_in_fixed_144px_crop",
                                "complete_target_touches_local_fallback_margin",
                            }:
                                raise ValueError("Global fallback has no valid reason")
                            size = image_dimensions.get(global_path)
                            if size is None:
                                with Image.open(global_file) as image:
                                    size = tuple(map(int, image.size))
                                image_dimensions[global_path] = size
                            reference = str(meta.get("pivr_reference") or "").strip()
                            global_point = meta.get("pivr_global_point")
                            global_target = meta.get("pivr_global_target_box")
                            if prompt != compact_prompt(
                                reference, global_point, "global_fallback"
                            ):
                                raise ValueError(
                                    "Global fallback prompt differs from the frozen template"
                                )
                            if answer != box_token(global_target):
                                raise ValueError(
                                    "Global fallback answer differs from its exact target"
                                )
                            expected_crop = list(
                                point_centered_crop(
                                    global_point,
                                    size[0],
                                    size[1],
                                    SOURCE_CROP_SIDE,
                                )
                            )
                            if meta.get("pivr_crop_xyxy_pixels") != expected_crop:
                                raise ValueError(
                                    "Global fallback crop metadata is not deterministic"
                                )
                            exact_fit = bool(
                                global_target is None
                                or target_fits_crop(
                                    global_target,
                                    expected_crop,
                                    size[0],
                                    size[1],
                                )
                            )
                            expected_point, expected_local_target = geometry_in_crop(
                                global_point,
                                global_target if exact_fit else None,
                                expected_crop,
                                size[0],
                                size[1],
                            )
                            touches_edge = bool(
                                expected_local_target is not None
                                and local_box_touches_fallback_margin(
                                    expected_local_target,
                                    LOCAL_INPUT_SIDE,
                                    FALLBACK_EDGE_MARGIN_PIXELS,
                                )
                            )
                            expected_reason = (
                                "negative_address"
                                if global_target is None
                                else (
                                    "target_not_fully_contained_in_fixed_144px_crop"
                                    if not exact_fit
                                    else "complete_target_touches_local_fallback_margin"
                                    if touches_edge
                                    else None
                                )
                            )
                            if meta.get("pivr_fallback_reason") != expected_reason:
                                raise ValueError("Global fallback reason is inconsistent")
                            if expected_reason is None:
                                raise ValueError(
                                    "A local-interior positive must not have a global fallback row"
                                )
                            if float(meta.get("pivr_fallback_edge_margin_pixels", -1)) != float(
                                FALLBACK_EDGE_MARGIN_PIXELS
                            ):
                                raise ValueError("Global fallback edge margin is not frozen")
                            if bool(meta.get("pivr_target_fully_contained")) != exact_fit:
                                raise ValueError("Global fallback containment flag is inconsistent")
                            if bool(
                                meta.get("pivr_target_touches_local_fallback_margin")
                            ) != touches_edge:
                                raise ValueError("Global fallback crop-edge flag is inconsistent")
                            if meta.get("pivr_local_point") != expected_point:
                                raise ValueError("Global fallback local point metadata is inconsistent")
                            if meta.get("pivr_local_target_box") != expected_local_target:
                                raise ValueError("Global fallback local target metadata is inconsistent")
                            if fallback_pair_key in global_fallback_seen:
                                raise ValueError("Duplicate global fallback source record")
                            global_fallback_seen.add(fallback_pair_key)
                            counts[f"{split}.round2.global_fallback"] += 1
                        else:
                            raise ValueError(f"Wrong Round-2 visual context: {context!r}")
                    else:
                        raise ValueError(f"Unknown PIVR route: {route!r}")
                counts[f"{split}.records_including_monitor"] += 1
            except Exception as exc:
                errors.append(f"{path}:{line_number}: {exc}")
                if len(errors) >= 20:
                    raise ValueError("Package validation failed:\n" + "\n".join(errors))
    if errors:
        raise ValueError("Package validation failed:\n" + "\n".join(errors))
    if local_fallback_expected != global_fallback_seen:
        missing = sorted(local_fallback_expected - global_fallback_seen)[:20]
        extra = sorted(global_fallback_seen - local_fallback_expected)[:20]
        raise ValueError(
            f"Local/global fallback pairing mismatch: missing={missing}, extra={extra}"
        )

    overlaps = {
        "train_validation": len(referenced["train"] & referenced["validation"]),
        "train_test": len(referenced["train"] & referenced["test"]),
        "validation_test": len(referenced["validation"] & referenced["test"]),
    }
    if any(overlaps.values()) or overlaps != manifest.get("image_hash_overlap"):
        raise ValueError(f"Image split leakage or manifest mismatch: {overlaps}")

    inventory_path = root / "IMAGE_INVENTORY.jsonl"
    inventory = []
    for _, row in read_jsonl(inventory_path):
        image = root / str(row["path"])
        if not image.is_file() or image.stat().st_size != int(row["bytes"]):
            raise ValueError(f"Image inventory mismatch: {image}")
        if image_digest_from_path(image) != row["sha256"]:
            raise ValueError(f"Image inventory digest/path mismatch: {image}")
        inventory.append(row)
    expected_images = sum(int(manifest["images"][split]["unique"]) for split in ("train", "validation", "test"))
    if len(inventory) != expected_images:
        raise ValueError(f"Image inventory count {len(inventory)} != {expected_images}")

    inventory_paths = {str(row["path"]) for row in inventory}
    actual_image_paths = {
        path.relative_to(root).as_posix()
        for path in (root / "images").rglob("*")
        if path.is_file()
    }
    if inventory_paths != actual_image_paths:
        missing = sorted(inventory_paths - actual_image_paths)[:20]
        extra = sorted(actual_image_paths - inventory_paths)[:20]
        raise ValueError(
            f"Image inventory/file mismatch: missing={missing}, extra={extra}"
        )

    inventory_by_split = Counter(Path(value).parts[1] for value in inventory_paths)
    bytes_by_split = Counter()
    for row in inventory:
        bytes_by_split[Path(str(row["path"])).parts[1]] += int(row["bytes"])
    for split in ("train", "validation", "test"):
        expected = manifest["images"][split]
        if inventory_by_split[split] != int(expected["unique"]):
            raise ValueError(
                f"{split} image count {inventory_by_split[split]} != {expected['unique']}"
            )
        if bytes_by_split[split] != int(expected["bytes"]):
            raise ValueError(
                f"{split} image bytes {bytes_by_split[split]} != {expected['bytes']}"
            )
        if len(referenced[split]) != int(expected["unique"]):
            raise ValueError(
                f"{split} referenced hashes {len(referenced[split])} != {expected['unique']}"
            )

    def recipe_count(name: str) -> int:
        recipe = json.loads((root / "recipes" / name).read_text(encoding="utf-8"))
        files = recipe.get("annotation")
        if not isinstance(files, list) or not files:
            raise ValueError(f"Recipe {name} has no annotation files")
        total = 0
        for value in files:
            path = root / str(value)
            if not path.is_file():
                raise FileNotFoundError(f"Recipe {name} references missing {value}")
            total += sum(1 for _ in read_jsonl(path))
        return total

    contract = manifest["training_contract"]
    recipe_counts = {
        "stage1_records": recipe_count("stage1_coarse.json"),
        "stage2_records": recipe_count("stage2_dense_balanced.json"),
        "validation_records": recipe_count("validation_all_tasks.json"),
        "validation_monitor_records": recipe_count(
            "validation_monitor_all_tasks.json"
        ),
    }
    for key, value in recipe_counts.items():
        if value != int(contract[key]):
            raise ValueError(f"Recipe count {key}={value} != manifest {contract[key]}")
    if int(contract["stage2_dense_records"]) + int(
        contract["stage2_replay_records"]
    ) != int(contract["stage2_records"]):
        raise ValueError("Stage-2 dense/replay accounting does not sum to the recipe")
    observed_partition_records = {
        key: int(observed_stage2_partitions[key]) for key in ("dense", "replay")
    }
    declared_partition_records = {
        "dense": int(contract["stage2_dense_records"]),
        "replay": int(contract["stage2_replay_records"]),
    }
    if observed_partition_records != declared_partition_records:
        raise ValueError(
            "Observed Stage-2 partitions differ from the manifest: "
            f"{observed_partition_records} != {declared_partition_records}"
        )
    observed_replay_percent = (
        100.0
        * int(contract["stage2_replay_records"])
        / int(contract["stage2_records"])
    )
    if abs(observed_replay_percent - float(contract["stage2_replay_percent"])) > 1e-12:
        raise ValueError("Stage-2 replay percentage is inconsistent")

    checksum_path = root / "SHA256SUMS"
    checksum_entries: dict[str, str] = {}
    for line_number, line in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or "  " not in line:
            raise ValueError(f"Malformed SHA256SUMS line {line_number}")
        digest, relative = line.split("  ", 1)
        if len(digest) != 64 or relative in checksum_entries:
            raise ValueError(f"Invalid/duplicate SHA256SUMS line {line_number}")
        checksum_entries[relative] = digest
    checksum_expected = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and ".cache" not in path.relative_to(root).parts
        and path.name not in {
            "SHA256SUMS",
            "verification.json",
            "materialization.json",
            "bundle_verification.json",
            "remote_verification.json",
        }
    }
    if set(checksum_entries) != checksum_expected:
        missing = sorted(checksum_expected - set(checksum_entries))[:20]
        extra = sorted(set(checksum_entries) - checksum_expected)[:20]
        raise ValueError(f"SHA256SUMS coverage mismatch: missing={missing}, extra={extra}")
    for relative, digest in checksum_entries.items():
        path = root / relative
        if relative.startswith("images/"):
            if digest != image_digest_from_path(path):
                raise ValueError(f"Image checksum/path mismatch in SHA256SUMS: {relative}")
        elif sha256_file(path) != digest:
            raise ValueError(f"Metadata checksum mismatch: {relative}")

    bad_hashes = []
    if verify_image_hashes:
        for index, row in enumerate(inventory, 1):
            image = root / str(row["path"])
            if sha256_file(image) != row["sha256"]:
                bad_hashes.append(str(row["path"]))
                if len(bad_hashes) >= 20:
                    break
            if index % 10000 == 0:
                print(f"verified-image-hashes {index}/{len(inventory)}", flush=True)
        if bad_hashes:
            raise ValueError(f"Image SHA-256 mismatches: {bad_hashes}")

    result = {
        "status": "passed",
        "schema_version": SCHEMA,
        "annotation_counts": dict(sorted(counts.items())),
        "image_inventory_records": len(inventory),
        "referenced_unique_images": {key: len(value) for key, value in sorted(referenced.items())},
        "image_hash_overlap": overlaps,
        "full_image_hash_verification": bool(verify_image_hashes),
        "crop_contract": expected_pixel,
        "fallback_pairs": len(global_fallback_seen),
        "record_id_scopes": {
            key: len(value) for key, value in sorted(record_ids.items())
        },
        "stage2_partitions": dict(sorted(observed_stage2_partitions.items())),
        "recipe_counts": recipe_counts,
        "sha256sum_entries": len(checksum_entries),
    }
    write_json(root / "verification.json", result)
    return result


def main() -> None:
    workspace = ROOT.parent
    project = workspace / "Zero_shot_ anlysis_LA"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "build", "verify"))
    parser.add_argument("--source-root", type=Path, default=workspace)
    parser.add_argument("--project-root", type=Path, default=project)
    parser.add_argument(
        "--curriculum",
        type=Path,
        default=project / "dataset/sft/pixel_pivr_type2_hbb_full_v1",
    )
    parser.add_argument("--output", type=Path, default=workspace / "Pixel-PIVR-HF-v2")
    parser.add_argument("--existing-image-root", type=Path, default=workspace / "Pixel-PIVR-HF")
    parser.add_argument("--link-mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--shard-mib", type=int, default=180)
    parser.add_argument("--source-crop-side", type=int, default=SOURCE_CROP_SIDE)
    parser.add_argument("--local-input-side", type=int, default=LOCAL_INPUT_SIDE)
    parser.add_argument("--trust-record-hashes", action="store_true")
    parser.add_argument("--verify-image-hashes", action="store_true")
    args = parser.parse_args()
    if args.mode == "plan":
        files = sorted(args.curriculum.rglob("*.jsonl"))
        result = {
            "schema": SCHEMA,
            "source_crop_side": args.source_crop_side,
            "local_input_side": args.local_input_side,
            "source_jsonl_files": len(files),
            "source_jsonl_bytes": sum(path.stat().st_size for path in files),
        }
    elif args.mode == "build":
        packager = MagnifiedPackager(
            args.source_root,
            args.project_root,
            args.curriculum,
            args.output,
            link_mode=args.link_mode,
            shard_mib=args.shard_mib,
            verify_content=not args.trust_record_hashes,
            source_crop_side=args.source_crop_side,
            local_input_side=args.local_input_side,
            existing_image_root=args.existing_image_root,
        )
        result = packager.build()
    else:
        result = verify_package(
            args.output, verify_image_hashes=args.verify_image_hashes
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
