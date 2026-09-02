#!/usr/bin/env python3
"""Convert the portable HF test annotations into Pixel-PIVR manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


DIOR_CLASSES = (
    "airplane", "airport", "groundtrackfield", "harbor", "baseballfield",
    "overpass", "basketballcourt", "ship", "bridge", "stadium",
    "storagetank", "tenniscourt", "expressway service area", "trainstation",
    "expressway toll station", "vehicle", "golffield", "windmill", "chimney",
    "dam",
)
DOTA_CLASSES = (
    "plane", "ship", "storage tank", "baseball diamond", "tennis court",
    "basketball court", "ground track field", "harbor", "bridge",
    "large vehicle", "small vehicle", "helicopter", "roundabout",
    "soccer ball field", "swimming pool", "container crane", "airport",
    "helipad",
)


def rows(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise ValueError(f"Blank line: {path}:{line_number}")
                yield json.loads(line)


def write_rows(path: Path, values: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, sort_keys=True) + "\n")
            count += 1
    temporary.replace(path)
    return count


def detection_rows(paths: list[Path], benchmark: str) -> Iterable[dict[str, Any]]:
    classes = DIOR_CLASSES if benchmark == "DIOR" else DOTA_CLASSES
    for index, row in enumerate(rows(paths)):
        yield {
            "sample_key": f"{benchmark}:detection:{row.get('image_id', index)}",
            "image_id": str(row.get("image_id", index)),
            "image": row["image_path"],
            "task": "detection",
            "benchmark": benchmark,
            "classes": list(classes),
            "gt": [
                {"label": obj["class_name"], "hbox": obj["hbox"]}
                for obj in row.get("objects", [])
            ],
            "gt_coordinate_space": "pixel",
        }


def grounding_rows(paths: list[Path], benchmark: str) -> Iterable[dict[str, Any]]:
    for index, row in enumerate(rows(paths)):
        query = str(row["query"]).strip().rstrip(".")
        sample_id = str(row.get("sample_id", index))
        yield {
            "sample_key": f"{benchmark}:grounding:{sample_id}",
            "image_id": str(row.get("image_id", sample_id)),
            "image": row["image_path"],
            "task": "grounding",
            "benchmark": benchmark,
            "classes": [query],
            "point_prompt": (
                "Point to a single instance that matches the following description: "
                f"{query}."
            ),
            "gt": [{"label": query, "hbox": row["gt_hbox"]}],
            "gt_coordinate_space": "pixel",
        }


def pointing_rows(paths: list[Path], benchmark: str) -> Iterable[dict[str, Any]]:
    for index, row in enumerate(rows(paths)):
        boxes = row.get("gt_hboxes")
        if not boxes:
            raise ValueError(
                "Pointing annotation has no gt_hboxes. Rebuild or refresh the HF "
                "package with the current package_hf_dataset.py."
            )
        label = str(row.get("target_class") or row["meta"]["target_class"])
        record_id = str(row.get("meta", {}).get("record_id") or index)
        prompt = str(row["conversations"][0]["value"]).replace("<image>\n", "", 1)
        yield {
            "sample_key": f"{benchmark}:pointing:{record_id}",
            "image_id": str(row.get("image_id") or record_id),
            "image": row["image"],
            "task": "pointing",
            "benchmark": benchmark,
            "classes": [label],
            "point_prompt": prompt,
            "gt": [{"label": label, "hbox": box} for box in boxes],
            "gt_coordinate_space": "pixel",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.data_root.resolve()
    specs = (
        ("detection", "DIOR", detection_rows),
        ("detection", "DOTAv2", detection_rows),
        ("grounding", "DIOR-RSVG", grounding_rows),
        ("grounding", "VRSBench-VG", grounding_rows),
        ("pointing", "DOTAv2-Balanced100", pointing_rows),
    )
    summary = {}
    for task, benchmark, converter in specs:
        source = sorted((root / "annotations/test" / task / benchmark).glob("part-*.jsonl"))
        if not source:
            raise FileNotFoundError(f"No test annotations for {task}/{benchmark}")
        destination = args.output / f"{task}_{benchmark}.jsonl"
        summary[f"{task}/{benchmark}"] = {
            "records": write_rows(destination, converter(source, benchmark)),
            "manifest": str(destination.resolve()),
        }
    summary_path = args.output / "manifest_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
