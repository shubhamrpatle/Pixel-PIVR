#!/usr/bin/env python3
"""Verify a fixed-source-crop Pixel-PIVR training corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image


REF_BOX = re.compile(
    r"^<ref>(?P<label>.*?)</ref><box>(?P<body>None|(?:<\d+>){4})</box>$",
    re.DOTALL,
)
POINT_PROMPT = re.compile(
    r"^<image><image>\nLocate the single (?P<label>.+?) containing point "
    r"<box><(?P<x>\d+)><(?P<y>\d+)></box> in horizontal box format\. "
    r"Return None if absent\.$"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                yield line_number, json.loads(line)


def assistant_text(record: Mapping[str, Any]) -> str:
    conversations = record.get("conversations") or []
    if len(conversations) != 2:
        raise AssertionError("record must contain one human and one GPT turn")
    if conversations[0].get("from") != "human" or conversations[1].get("from") != "gpt":
        raise AssertionError("conversation roles are not human/GPT")
    return str(conversations[1].get("value") or "")


def human_text(record: Mapping[str, Any]) -> str:
    return str((record.get("conversations") or [{}])[0].get("value") or "")


def point_inside(point: Sequence[int], box: Sequence[int]) -> bool:
    return (
        int(box[0]) <= int(point[0]) <= int(box[2])
        and int(box[1]) <= int(point[1]) <= int(box[3])
    )


def load_holdouts(path: Path) -> set[str]:
    return {
        line.strip().split()[0]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def verify_split(
    *,
    crop_path: Path,
    compact_path: Path,
    expected_records: int,
    crop_side: int,
    output_side: int,
    holdouts: set[str],
) -> tuple[dict[str, Any], set[str]]:
    crop_rows = list(read_jsonl(crop_path))
    compact_rows = list(read_jsonl(compact_path))
    if len(crop_rows) != expected_records or len(compact_rows) != expected_records:
        raise AssertionError(
            f"record count mismatch: crop={len(crop_rows)}, compact={len(compact_rows)}, "
            f"expected={expected_records}"
        )

    routes: Counter[str] = Counter()
    polarities: Counter[str] = Counter()
    datasets: Counter[str] = Counter()
    fallbacks: Counter[str] = Counter()
    hashes: set[str] = set()
    local_paths: set[Path] = set()
    local_positive = 0
    local_negative = 0

    for (crop_line, crop), (compact_line, record) in zip(crop_rows, compact_rows):
        if crop_line != compact_line:
            raise AssertionError("line alignment changed")
        crop_meta = crop.get("meta") or {}
        meta = record.get("meta") or {}
        for key in ("image_content_sha256", "pivr_route", "target_count"):
            if crop_meta.get(key) != meta.get(key):
                raise AssertionError(f"line {crop_line}: compact conversion changed {key}")
        if crop.get("image") != record.get("image"):
            raise AssertionError(f"line {crop_line}: compact conversion changed images")

        image_hash = str(meta.get("image_content_sha256") or "")
        if len(image_hash) != 64:
            raise AssertionError(f"line {crop_line}: invalid source image SHA256")
        if image_hash in holdouts:
            raise AssertionError(f"line {crop_line}: held-out image leaked into training data")
        hashes.add(image_hash)
        route = str(meta.get("pivr_route"))
        routes[route] += 1
        polarities[str(meta.get("polarity") or "mixed")] += 1
        datasets[str(meta.get("dataset") or "unknown")] += 1
        if meta.get("pivr_fallback_reason"):
            fallbacks[str(meta["pivr_fallback_reason"])] += 1

        if route == "global_point_discovery":
            if isinstance(record.get("image"), list):
                raise AssertionError(f"line {crop_line}: Round 1 must use one global image")
            continue
        if route != "point_indexed_visual_reentry":
            raise AssertionError(f"line {crop_line}: unknown route {route!r}")

        images = record.get("image")
        if not isinstance(images, list) or len(images) != 2:
            raise AssertionError(f"line {crop_line}: Round 2 must use global and local images")
        if int(meta.get("pivr_crop_side_requested", -1)) != crop_side:
            raise AssertionError(f"line {crop_line}: requested crop side is not {crop_side}")
        if list(meta.get("pivr_crop_size") or []) != [output_side, output_side]:
            raise AssertionError(
                f"line {crop_line}: encoded local view is not {output_side} square"
            )
        source_size = list(meta.get("pivr_source_crop_size") or [crop_side, crop_side])
        if source_size != [crop_side, crop_side]:
            raise AssertionError(
                f"line {crop_line}: source crop is not {crop_side} square"
            )
        crop_xyxy = [int(value) for value in meta.get("pivr_crop_xyxy_pixels") or []]
        if len(crop_xyxy) != 4 or crop_xyxy[2] - crop_xyxy[0] != crop_side or crop_xyxy[3] - crop_xyxy[1] != crop_side:
            raise AssertionError(f"line {crop_line}: crop bounds disagree with crop size")

        local_path = Path(str(images[1]))
        if not local_path.is_file():
            raise FileNotFoundError(local_path)
        with Image.open(local_path) as image:
            if image.size != (output_side, output_side):
                raise AssertionError(f"line {crop_line}: PNG dimensions are {image.size}")
            if image.format != "PNG":
                raise AssertionError(f"line {crop_line}: local view is not lossless PNG")
        if local_path in local_paths:
            raise AssertionError(f"line {crop_line}: local crop is reused by another record")
        local_paths.add(local_path)

        prompt_match = POINT_PROMPT.fullmatch(human_text(record))
        answer_match = REF_BOX.fullmatch(assistant_text(record))
        if prompt_match is None or answer_match is None:
            raise AssertionError(f"line {crop_line}: compact prompt/answer grammar mismatch")
        if prompt_match.group("label") != answer_match.group("label"):
            raise AssertionError(f"line {crop_line}: prompt and answer labels differ")
        local_point = [int(prompt_match.group("x")), int(prompt_match.group("y"))]
        if local_point != [int(value) for value in meta.get("pivr_local_point") or []]:
            raise AssertionError(f"line {crop_line}: prompt point and local point differ")
        local_target = meta.get("pivr_local_target_box")
        if local_target is None:
            if answer_match.group("body") != "None" or int(meta.get("target_count", -1)) != 0:
                raise AssertionError(f"line {crop_line}: negative target grammar mismatch")
            local_negative += 1
        else:
            local_target = [int(value) for value in local_target]
            expected_body = "".join(f"<{value}>" for value in local_target)
            if answer_match.group("body") != expected_body or int(meta.get("target_count", -1)) != 1:
                raise AssertionError(f"line {crop_line}: positive target grammar mismatch")
            if not point_inside(local_point, local_target):
                raise AssertionError(f"line {crop_line}: point is outside local target")
            local_positive += 1
        if meta.get("pivr_target_fully_contained") is not True:
            raise AssertionError(f"line {crop_line}: target is not certified fully contained")

    if len(hashes) != expected_records:
        raise AssertionError(
            f"expected one unique source image per record, found {len(hashes)}/{expected_records}"
        )
    return (
        {
            "records": expected_records,
            "unique_source_images": len(hashes),
            "routes": dict(sorted(routes.items())),
            "polarities": dict(sorted(polarities.items())),
            "datasets": dict(sorted(datasets.items())),
            "fallbacks": dict(sorted(fallbacks.items())),
            "exact_local_crops": len(local_paths),
            "local_positive": local_positive,
            "local_negative": local_negative,
        },
        hashes,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-dataset", type=Path, required=True)
    parser.add_argument("--compact-dataset", type=Path, required=True)
    parser.add_argument("--holdout-hashes", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--crop-side", type=int, default=380)
    parser.add_argument("--output-side", type=int, default=None)
    parser.add_argument("--expected-train", type=int, default=16000)
    parser.add_argument("--expected-validation", type=int, default=1000)
    args = parser.parse_args()
    output_side = int(args.output_side or args.crop_side)

    crop_manifest_path = args.crop_dataset / "manifest.json"
    compact_manifest_path = args.compact_dataset / "manifest.json"
    crop_manifest = json.loads(crop_manifest_path.read_text(encoding="utf-8"))
    compact_manifest = json.loads(compact_manifest_path.read_text(encoding="utf-8"))
    if int(crop_manifest.get("crop_side", -1)) != args.crop_side:
        raise AssertionError("crop manifest has the wrong crop side")
    if int(crop_manifest.get("output_side", args.crop_side)) != output_side:
        raise AssertionError("crop manifest has the wrong output side")
    if crop_manifest.get("require_exact_crop_size") is not True:
        raise AssertionError("crop manifest does not require exact crop dimensions")
    if crop_manifest.get("source_benchmark_hash_overlap") != 0:
        raise AssertionError("source manifest reports benchmark leakage")
    if crop_manifest.get("source_train_validation_hash_overlap") != 0:
        raise AssertionError("source manifest reports train/validation overlap")
    if Path(str(compact_manifest.get("source"))).resolve() != args.crop_dataset.resolve():
        raise AssertionError("compact corpus points to a different crop corpus")
    if compact_manifest.get("source_manifest_sha256") != sha256_file(crop_manifest_path):
        raise AssertionError("compact corpus source-manifest digest changed")
    if compact_manifest.get("source_benchmark_hash_overlap") != 0:
        raise AssertionError("compact manifest reports benchmark leakage")

    holdouts = load_holdouts(args.holdout_hashes)
    report: dict[str, Any] = {
        "schema_version": "pixel-crop-reencoded-fixed-source-audit-v2",
        "crop_side": args.crop_side,
        "output_side": output_side,
        "upscale_factor": output_side / float(args.crop_side),
        "visual_path": (
            "independent_pixel_crop_to_Lanczos_resize_to_MoonViT_to_2x2_merge_to_shared_projector"
            if output_side != args.crop_side
            else "independent_lossless_crop_to_MoonViT_to_2x2_merge_to_shared_projector"
        ),
        "per_image_premerge_patch_token_limit_for_training": 6000,
        "splits": {},
    }
    split_hashes: dict[str, set[str]] = {}
    for split, count in (
        ("train", args.expected_train),
        ("validation", args.expected_validation),
        ("smoke", 8),
    ):
        crop_path = args.crop_dataset / f"{split}.jsonl"
        compact_path = args.compact_dataset / "global_local" / f"{split}.jsonl"
        stats, hashes = verify_split(
            crop_path=crop_path,
            compact_path=compact_path,
            expected_records=count,
            crop_side=args.crop_side,
            output_side=output_side,
            holdouts=holdouts,
        )
        expected_digest = compact_manifest["arms"]["global_local"][split]["sha256"]
        if sha256_file(compact_path) != expected_digest:
            raise AssertionError(f"{split} JSONL digest differs from manifest")
        stats["jsonl"] = str(compact_path.resolve())
        stats["sha256"] = expected_digest
        report["splits"][split] = stats
        split_hashes[split] = hashes

    overlap = split_hashes["train"] & split_hashes["validation"]
    if overlap:
        raise AssertionError(f"train/validation image overlap: {len(overlap)}")
    report["train_validation_image_overlap"] = 0
    report["heldout_image_overlap"] = 0
    report["target_truncation"] = 0
    report["crop_manifest_sha256"] = sha256_file(crop_manifest_path)
    report["compact_manifest_sha256"] = sha256_file(compact_manifest_path)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
