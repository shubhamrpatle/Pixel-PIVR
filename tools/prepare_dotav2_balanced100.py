#!/usr/bin/env python3
"""Build the frozen class-resolved DOTAv2 Balanced-100 manifest."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.prepare_evaluation import DETECTION_ONTOLOGIES, rows  # noqa: E402


EXPECTED_CLASS_COUNTS = {
    "airport": 18,
    "baseball diamond": 27,
    "basketball court": 47,
    "bridge": 18,
    "container crane": 11,
    "ground track field": 29,
    "harbor": 111,
    "helicopter": 112,
    "helipad": 5,
    "large vehicle": 108,
    "plane": 73,
    "roundabout": 30,
    "ship": 2388,
    "small vehicle": 5650,
    "soccer ball field": 30,
    "storage tank": 175,
    "swimming pool": 105,
    "tennis court": 108,
}
EXPECTED_IMAGES = 100
EXPECTED_TARGETS = 9045
FORBIDDEN_COLLAPSED_LABELS = {"airplane", "vehicle"}

if sum(EXPECTED_CLASS_COUNTS.values()) != EXPECTED_TARGETS:
    raise RuntimeError("Internal Balanced-100 target-count contract is inconsistent")


def _validated_hbox(value: Any, *, image_id: str, object_index: int) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(
            f"Image {image_id!r} object {object_index} has an invalid HBB: {value!r}"
        )
    box = [float(item) for item in value]
    if not all(math.isfinite(item) for item in box):
        raise ValueError(
            f"Image {image_id!r} object {object_index} has a non-finite HBB"
        )
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(
            f"Image {image_id!r} object {object_index} has a degenerate HBB: {box}"
        )
    return box


def balanced100_rows(
    annotation_paths: Iterable[Path], image_root: Path
) -> list[dict[str, Any]]:
    """Convert source rows without allowing convenience labels to replace raw GT."""
    ontology = DETECTION_ONTOLOGIES["DOTAv2"]
    classes = tuple(ontology["classes"])
    class_set = set(classes)
    image_root = image_root.resolve()
    output: list[dict[str, Any]] = []
    seen_image_ids: set[str] = set()

    for index, source in enumerate(rows(annotation_paths)):
        image_id = str(source.get("image_id", index))
        if image_id in seen_image_ids:
            raise ValueError(f"Duplicate DOTAv2 Balanced-100 image_id: {image_id!r}")
        seen_image_ids.add(image_id)

        raw_image = Path(str(source["image_path"]))
        image = raw_image if raw_image.is_absolute() else image_root / raw_image
        image = image.resolve()
        if not image.is_file():
            raise FileNotFoundError(f"Missing Balanced-100 image: {image}")

        gt = []
        for object_index, obj in enumerate(source.get("objects", [])):
            # raw_class_name is mandatory here. Falling back to class_name caused
            # plane -> airplane and both vehicle subclasses -> vehicle, corrupting
            # class-aware TP/FP/FN accounting in the historical pilot evaluation.
            label = str(obj.get("raw_class_name") or "").strip()
            if not label:
                raise ValueError(
                    f"Image {image_id!r} object {object_index} is missing raw_class_name"
                )
            if label not in class_set:
                raise ValueError(
                    f"Image {image_id!r} object {object_index} has raw label "
                    f"{label!r} outside {ontology['name']}"
                )
            gt.append(
                {
                    "label": label,
                    "hbox": _validated_hbox(
                        obj.get("hbox"),
                        image_id=image_id,
                        object_index=object_index,
                    ),
                }
            )

        output.append(
            {
                "sample_key": f"DOTAv2:balanced100:{image_id}",
                "image_id": image_id,
                "image": str(image),
                "task": "detection",
                "benchmark": "DOTAv2-Balanced100",
                "classes": list(classes),
                "class_ontology": ontology["name"],
                "gt_label_field": "raw_class_name",
                "class_prompts": dict(ontology["class_prompts"]),
                "label_aliases": dict(ontology["label_aliases"]),
                "gt": gt,
                "gt_coordinate_space": "pixel",
            }
        )
    return output


def validate_balanced100_rows(
    values: Iterable[Mapping[str, Any]],
    *,
    expected_images: int = EXPECTED_IMAGES,
    expected_class_counts: Mapping[str, int] = EXPECTED_CLASS_COUNTS,
) -> Counter[str]:
    values = list(values)
    sample_keys = [str(row.get("sample_key")) for row in values]
    if len(values) != expected_images:
        raise ValueError(
            f"Frozen Balanced-100 image count changed: {len(values)} != {expected_images}"
        )
    if len(set(sample_keys)) != len(sample_keys):
        raise ValueError("Frozen Balanced-100 manifest contains duplicate sample_key values")

    counts: Counter[str] = Counter(
        str(target["label"])
        for row in values
        for target in (row.get("gt") or [])
    )
    expected = Counter(expected_class_counts)
    if expected_images == EXPECTED_IMAGES and expected == Counter(EXPECTED_CLASS_COUNTS):
        if sum(counts.values()) != EXPECTED_TARGETS:
            raise ValueError(
                f"Frozen Balanced-100 target count changed: "
                f"{sum(counts.values())} != {EXPECTED_TARGETS}"
            )
    if counts != expected:
        missing = expected - counts
        extra = counts - expected
        raise ValueError(
            "Frozen Balanced-100 class distribution changed: "
            f"missing={dict(missing)} extra={dict(extra)} actual={dict(counts)}"
        )
    collapsed = sorted(FORBIDDEN_COLLAPSED_LABELS.intersection(counts))
    if collapsed:
        raise ValueError(f"Collapsed DOTAv2 labels are forbidden: {collapsed}")
    return counts


def write_manifest(destination: Path, values: Iterable[Mapping[str, Any]]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, sort_keys=True) + "\n")
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, action="append", required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    values = balanced100_rows(args.annotation, args.image_root)
    counts = validate_balanced100_rows(values)
    write_manifest(args.output, values)
    print(
        json.dumps(
            {
                "class_counts": dict(sorted(counts.items())),
                "class_ontology": "dotav2_raw_18_class",
                "images": len(values),
                "output": str(args.output.resolve()),
                "targets": sum(counts.values()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
