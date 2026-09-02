"""Audit Pixel-PIVR JSONL before expensive distributed training."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from PIL import Image
import torch

from .data import make_dataset
from .io import atomic_json, sha256_file


REF_RE = re.compile(r"<ref>\s*(.*?)\s*</ref>", re.IGNORECASE | re.DOTALL)
POINT_RE = re.compile(r"<box>\s*<(\d+)>\s*<(\d+)>\s*</box>", re.IGNORECASE)
HBB_RE = re.compile(
    r"<box>\s*<(\d+)>\s*<(\d+)>\s*<(\d+)>\s*<(\d+)>\s*</box>",
    re.IGNORECASE,
)
NONE_RE = re.compile(r"<box>\s*None\s*</box>", re.IGNORECASE)


def assistant_text(row: Mapping[str, Any]) -> str:
    return "".join(
        str(turn.get("value") or "")
        for turn in row.get("conversations") or []
        if str(turn.get("from") or "").lower() in {"gpt", "assistant"}
    )


def resolve_path(raw: str, data_root: Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else data_root / path


def validate_image_item(item: Any, data_root: Path) -> None:
    if isinstance(item, str):
        path = resolve_path(item, data_root)
        if not path.is_file():
            raise FileNotFoundError(path)
        return
    if not isinstance(item, Mapping) or not item.get("virtual_crop"):
        raise TypeError(f"Unsupported image item: {item!r}")
    path = resolve_path(str(item.get("path") or ""), data_root)
    crop = item.get("crop_xyxy")
    if not path.is_file():
        raise FileNotFoundError(path)
    if not isinstance(crop, list) or len(crop) != 4 or not all(
        isinstance(value, int) for value in crop
    ):
        raise ValueError(f"Invalid virtual crop: {crop!r}")
    with Image.open(path) as image:
        width, height = image.size
    left, top, right, bottom = crop
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError(
            f"Virtual crop {crop} is outside {path} with size {(width, height)}"
        )


def validate_row(
    row: Mapping[str, Any],
    data_root: Path,
    *,
    visual_context: str = "pixel_reencoded",
) -> dict[str, Any]:
    meta = row.get("meta") or {}
    route = str(meta.get("pivr_route") or "")
    if route not in {"global_point_discovery", "point_indexed_visual_reentry"}:
        raise ValueError(f"Unknown or missing meta.pivr_route: {route!r}")
    geometry_modes = {
        str(meta.get("geometry_mode") or "").lower(),
        str(meta.get("pivr_source_geometry_mode") or "").lower(),
    }
    if "obb" in geometry_modes:
        raise ValueError("This tested release is HBB-only; OBB records are not accepted")
    answer = assistant_text(row)
    if "<quad>" in answer.lower():
        raise ValueError("This tested release does not accept <quad> supervision")
    refs = REF_RE.findall(answer)
    points = POINT_RE.findall(answer)
    boxes = HBB_RE.findall(answer)
    none_count = len(NONE_RE.findall(answer))
    images = row.get("image")

    if route == "global_point_discovery":
        if isinstance(images, list):
            raise ValueError("Global point discovery must contain exactly one image")
        validate_image_item(images, data_root)
        if not refs:
            raise ValueError("Global discovery answer has no <ref> group")
        if len(points) + none_count < len(refs):
            raise ValueError("A global <ref> group has neither points nor explicit None")
        for point in points:
            if any(not 0 <= int(value) <= 1000 for value in point):
                raise ValueError(f"Point coordinate outside [0,1000]: {point}")
    else:
        magnified = visual_context == "preprojector_magnified_roi"
        if magnified:
            if isinstance(images, list) or not isinstance(images, str):
                raise ValueError("Magnified re-entry requires one global image")
            validate_image_item(images, data_root)
        else:
            if not isinstance(images, list) or len(images) != 2:
                raise ValueError("Re-entry must contain [global image, local image]")
            for item in images:
                validate_image_item(item, data_root)
        if len(refs) != 1:
            raise ValueError(f"Re-entry requires one <ref>, found {len(refs)}")
        if len(boxes) + none_count != 1:
            raise ValueError("Re-entry requires exactly one HBB or explicit None")
        if boxes:
            values = tuple(int(value) for value in boxes[0])
            if any(not 0 <= value <= 1000 for value in values):
                raise ValueError(f"HBB coordinate outside [0,1000]: {values}")
            x1, y1, x2, y2 = values
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"Degenerate HBB: {values}")
            address_point = (
                meta.get("pivr_global_point")
                if magnified
                else meta.get("pivr_local_point")
            )
            if address_point is None and magnified:
                address_point = meta.get("pivr_branch_point")
            if address_point is not None:
                px, py = map(float, address_point)
                if not (x1 <= px <= x2 and y1 <= py <= y2):
                    raise ValueError(
                        f"Address {address_point} is outside target {values}"
                    )
        if meta.get("pivr_target_fully_contained") is not True:
            raise ValueError("Re-entry target containment is not explicitly true")

    expected = int(meta.get("target_count") or 0)
    observed = len(points) if route == "global_point_discovery" else len(boxes)
    if expected != observed:
        raise ValueError(f"meta.target_count mismatch: {expected} != {observed}")
    return {
        "route": route,
        "targets": observed,
        "none_groups": none_count,
        "image_hash": str(meta.get("image_content_sha256") or ""),
    }


def basic_audit(
    paths: list[Path],
    data_root: Path,
    holdout_hashes: set[str],
    *,
    visual_context: str = "pixel_reencoded",
) -> dict[str, Any]:
    counters = Counter()
    hashes: set[str] = set()
    errors = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    if not line.strip():
                        raise ValueError("blank JSONL line")
                    row = json.loads(line)
                    result = validate_row(
                        row, data_root, visual_context=visual_context
                    )
                    counters["records"] += 1
                    counters[f"route::{result['route']}"] += 1
                    counters["targets"] += result["targets"]
                    counters["none_groups"] += result["none_groups"]
                    if result["image_hash"]:
                        hashes.add(result["image_hash"])
                except Exception as exc:  # audit should report several failures at once
                    errors.append(
                        {"file": str(path), "line": line_number, "error": repr(exc)}
                    )
                    if len(errors) >= 100:
                        break
        if len(errors) >= 100:
            break
    overlap = sorted(hashes & holdout_hashes)
    if overlap:
        errors.append(
            {
                "error": "benchmark_hash_overlap",
                "count": len(overlap),
                "examples": overlap[:10],
            }
        )
    return {
        "status": "passed" if not errors else "failed",
        "files": [
            {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in paths
        ],
        "counts": dict(sorted(counters.items())),
        "unique_image_hashes": len(hashes),
        "holdout_hashes": len(holdout_hashes),
        "holdout_overlap": len(overlap),
        "errors": errors,
    }


def exact_loader_audit(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoProcessor

    os.environ["LOCANY_STRICT_COVERAGE"] = "1"
    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=not args.allow_download,
        use_fast=False,
    )
    processor.image_processor.in_token_limit = int(args.image_token_limit)
    processor.tokenizer.model_max_length = int(args.max_sequence)
    dataset = make_dataset(
        args.jsonl,
        processor,
        data_root=args.data_root,
        eagle_root=args.eagle_root,
        block_size=6,
    )
    if args.visual_context == "preprojector_magnified_roi":
        from .magnified_modeling import MagnifiedROIMetadataDataset

        dataset = MagnifiedROIMetadataDataset(dataset)
    maximum = 0
    supervised = 0
    limit = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    for index in range(limit):
        sample = dataset[index]
        length = int(sample["input_ids"].numel())
        local_tokens = 0
        if args.visual_context == "preprojector_magnified_roi" and bool(
            torch.as_tensor(sample["magnified_roi_enabled"]).item()
        ):
            grid_h, grid_w = (
                int(value) for value in torch.as_tensor(sample["image_grid_hws"])[0]
            )
            patch_size = int(processor.image_processor.patch_size)
            side = max(2, int(round(args.magnified_roi_pixels / patch_size)))
            window_h, window_w = min(side, grid_h), min(side, grid_w)
            local_h = (window_h - 2) // args.magnified_roi_stride + 1
            local_w = (window_w - 2) // args.magnified_roi_stride + 1
            local_tokens = local_h * local_w
            length += local_tokens + 2
        maximum = max(maximum, length)
        supervised += int(sample["labels"].ne(-100).sum().item())
        if length > args.max_sequence:
            raise RuntimeError(
                f"Exact loader sequence {index} exceeds budget: {length} > {args.max_sequence}"
            )
        if index and index % 1000 == 0:
            print(f"exact-loader {index}/{limit}", flush=True)
    return {
        "records_checked": limit,
        "dataset_records": len(dataset),
        "max_post_mtp_tokens": maximum,
        "supervised_tokens": supervised,
        "max_sequence": args.max_sequence,
        "visual_context": args.visual_context,
        "magnified_roi_pixels": args.magnified_roi_pixels,
        "magnified_roi_stride": args.magnified_roi_stride,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--holdout-hashes", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--exact-loader", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--eagle-root", type=Path)
    parser.add_argument("--image-token-limit", type=int, default=1024)
    parser.add_argument("--max-sequence", type=int, default=8192)
    parser.add_argument(
        "--visual-context",
        choices=("pixel_reencoded", "preprojector_magnified_roi"),
        default="pixel_reencoded",
    )
    parser.add_argument("--magnified-roi-pixels", type=int, default=380)
    parser.add_argument(
        "--magnified-roi-stride", type=int, choices=(1, 2), default=1
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    for path in [args.data_root, *args.jsonl]:
        if not path.exists():
            parser.error(f"Missing input: {path}")
    if args.exact_loader and (args.model is None or args.eagle_root is None):
        parser.error("--exact-loader requires --model and --eagle-root")
    return args


def main() -> None:
    args = parse_args()
    holdout = set()
    if args.holdout_hashes is not None:
        holdout = {
            line.strip()
            for line in args.holdout_hashes.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    report = basic_audit(
        args.jsonl,
        args.data_root,
        holdout,
        visual_context=args.visual_context,
    )
    if args.exact_loader and report["status"] == "passed":
        report["exact_loader"] = exact_loader_audit(args)
    atomic_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
