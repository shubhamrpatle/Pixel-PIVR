#!/usr/bin/env python3
"""Merge resumed Pixel-PIVR inference shards and recompute all metrics."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from pixel_pivr.geometry import precision_recall_f1
from pixel_pivr.infer import update_metrics


def rows(path: Path) -> list[dict[str, Any]]:
    output = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"Blank line at {path}:{number}")
            output.append(json.loads(line))
    return output


def atomic_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        temporary = Path(handle.name)
        for value in values:
            handle.write(json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-records", type=int, required=True)
    args = parser.parse_args()
    shard_dirs = sorted(path for path in args.shard_root.glob("shard-*") if path.is_dir())
    if not shard_dirs:
        raise FileNotFoundError(f"No shard directories under {args.shard_root}")
    predictions: dict[str, dict[str, Any]] = {}
    executions: dict[str, dict[str, Any]] = {}
    summaries = []
    for shard in shard_dirs:
        required = [shard / name for name in ("predictions.jsonl", "execution.jsonl", "summary.json")]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Incomplete inference shard {shard}: {missing}")
        summary = json.loads((shard / "summary.json").read_text(encoding="utf-8"))
        summaries.append(summary)
        for row in rows(shard / "predictions.jsonl"):
            key = str(row.get("sample_key") or "")
            if not key or key in predictions:
                raise ValueError(f"Missing/duplicate prediction sample_key: {key!r}")
            predictions[key] = row
        for row in rows(shard / "execution.jsonl"):
            key = str(row.get("sample_key") or "")
            if not key or key in executions:
                raise ValueError(f"Missing/duplicate execution sample_key: {key!r}")
            executions[key] = row
    if len(predictions) != args.expected_records or set(predictions) != set(executions):
        raise RuntimeError(
            f"Merged coverage mismatch: predictions={len(predictions)}, "
            f"executions={len(executions)}, expected={args.expected_records}"
        )
    contract_fields = (
        "model", "adapter", "image_token_limit", "visual_context", "crop_side",
        "local_resize_side", "geometry_prefix_mode", "global_fallback",
        "fallback_edge_margin", "prefix_cache_mode", "wave_size", "seed",
    )
    for field in contract_fields:
        values = {json.dumps(summary.get(field), sort_keys=True) for summary in summaries}
        if len(values) != 1:
            raise RuntimeError(f"Shard summaries disagree on {field}: {values}")

    detection: Counter = Counter()
    pointing: Counter = Counter()
    grounding: Counter = Counter()
    tasks: Counter = Counter()
    ordered_predictions = [predictions[key] for key in sorted(predictions)]
    ordered_executions = [executions[key] for key in sorted(executions)]
    for row in ordered_predictions:
        tasks[str(row.get("task") or "detection")] += 1
        update_metrics(row, detection, pointing, grounding)
    first = summaries[0]
    summary: dict[str, Any] = {
        key: first.get(key) for key in contract_fields
    }
    summary.update(
        {
            "schema_version": "pixel-pivr-sharded-inference-v1",
            "images": len(ordered_predictions),
            "prediction_cap": None,
            "nms_iou": first.get("nms_iou"),
            "tasks": dict(sorted(tasks.items())),
            "point_stage_seconds": sum(
                float(row.get("point_stage_seconds", 0.0)) for row in ordered_executions
            ),
            "refinement_seconds": sum(
                float(row.get("end_to_end_seconds", 0.0)) for row in ordered_executions
            ),
            "sharded_evaluation": {
                "shards": len(shard_dirs),
                "source_directories": [str(path.resolve()) for path in shard_dirs],
                "metrics_recomputed_from_raw_predictions": True,
            },
        }
    )
    if detection:
        summary["detection_iou_0_5"] = {
            **dict(detection),
            **precision_recall_f1(detection["tp"], detection["fp"], detection["fn"]),
        }
    if grounding:
        queries = int(grounding["queries"])
        summary["grounding"] = {
            "queries": queries,
            "acc_0_5": grounding["acc_0_5"] / queries,
            "acc_0_7": grounding["acc_0_7"] / queries,
            "mean_iou": grounding["iou_sum"] / queries,
        }
    if pointing:
        summary["pointing_containment"] = {
            **dict(pointing),
            **precision_recall_f1(pointing["tp"], pointing["fp"], pointing["fn"]),
        }
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(args.output / "predictions.jsonl", ordered_predictions)
    atomic_jsonl(args.output / "execution.jsonl", ordered_executions)
    temporary = args.output / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output / "summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
