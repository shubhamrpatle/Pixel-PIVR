#!/usr/bin/env python3
"""Build a tiny, deterministic hard-record recipe for GPU smoke testing."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from pixel_pivr.audit import assistant_text, loader_stress_rows
from pixel_pivr.data import ROUTES, TASKS, annotation_scope
from pixel_pivr.io import atomic_json, count_jsonl, sha256_file


def candidate_score(row: dict[str, Any]) -> dict[str, int]:
    meta = row.get("meta") or {}
    source_size = meta.get("pivr_source_image_size") or [0, 0]
    if not isinstance(source_size, (list, tuple)) or len(source_size) != 2:
        source_size = [0, 0]
    images = row.get("image")
    return {
        "conversation_chars": sum(
            len(str(turn.get("value") or ""))
            for turn in row.get("conversations") or []
        ),
        "target_count": int(meta.get("target_count") or 0),
        "source_image_area": int(source_size[0]) * int(source_size[1]),
        "visual_inputs": len(images) if isinstance(images, list) else 1,
        "explicit_none": int("<box>none</box>" in assistant_text(row).lower()),
    }


def interleave(rankings: Iterable[list[dict[str, Any]]]) -> Iterable[dict[str, Any]]:
    values = [ranking for ranking in rankings if ranking]
    depth = 0
    while values:
        next_values = []
        for ranking in values:
            if depth < len(ranking):
                yield ranking[depth]
                next_values.append(ranking)
        values = next_values
        depth += 1


def select(paths: list[Path], count: int) -> list[dict[str, Any]]:
    rows, evidence = loader_stress_rows(paths)
    if len(rows) != len(evidence):
        raise RuntimeError("Hard-record selector returned inconsistent evidence")

    offsets: dict[Path, int] = {}
    cursor = 0
    for path in paths:
        resolved = path.resolve()
        offsets[resolved] = cursor
        cursor += count_jsonl(resolved)

    candidates = []
    for row, proof in zip(rows, evidence):
        source = Path(str(proof["source"])).resolve()
        task, route = annotation_scope(source)
        line = int(proof["line"])
        candidates.append(
            {
                "row": row,
                "source": str(source),
                "line": line,
                "global_index": offsets[source] + line - 1,
                "task": task,
                "route": route,
                "record_id": str(proof["record_id"]),
                "reasons": list(proof["reasons"]),
                **candidate_score(row),
            }
        )
    # A small recipe can have fewer distinct stress extrema than smoke slots
    # because one row may be longest, largest, and highest-target at once. Keep
    # every hard candidate, then fill deterministically from real recipe rows.
    if len(candidates) < count:
        known_locations = {
            (str(candidate["source"]), int(candidate["line"]))
            for candidate in candidates
        }
        for path in paths:
            source = path.resolve()
            with source.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    location = (str(source), line_number)
                    if location in known_locations:
                        continue
                    row = json.loads(line)
                    task, route = annotation_scope(source)
                    record_id = str((row.get("meta") or {}).get("record_id") or "")
                    if not record_id:
                        raise ValueError(
                            f"Smoke-fill candidate has no record_id in {source}:{line_number}"
                        )
                    candidates.append(
                        {
                            "row": row,
                            "source": str(source),
                            "line": line_number,
                            "global_index": offsets[source] + line_number - 1,
                            "task": task,
                            "route": route,
                            "record_id": record_id,
                            "reasons": ["deterministic_fill"],
                            **candidate_score(row),
                        }
                    )
                    known_locations.add(location)
                    if len(candidates) == count:
                        break
            if len(candidates) == count:
                break
    if len(candidates) < count:
        raise RuntimeError(
            f"Recipe contains only {len(candidates)} unique records for {count} slots"
        )

    def ranked(key: str, *, require: Any = None) -> list[dict[str, Any]]:
        values = candidates
        if require is not None:
            values = [row for row in values if require(row)]
        return sorted(
            values,
            key=lambda row: (
                -int(row[key]),
                -int(row["conversation_chars"]),
                -int(row["target_count"]),
                str(row["source"]),
                int(row["line"]),
            ),
        )

    rankings = []
    for task in TASKS:
        for route in ROUTES:
            scoped = [
                row
                for row in candidates
                if row["task"] == task and row["route"] == route
            ]
            if scoped:
                rankings.append(
                    sorted(
                        scoped,
                        key=lambda row: (
                            -int(row["conversation_chars"]),
                            -int(row["target_count"]),
                            str(row["source"]),
                            int(row["line"]),
                        ),
                    )
                )
    rankings.extend(
        [
            ranked("conversation_chars"),
            ranked("target_count"),
            ranked("source_image_area"),
            ranked("visual_inputs"),
            ranked("conversation_chars", require=lambda row: row["visual_inputs"] > 1),
            ranked("conversation_chars", require=lambda row: row["explicit_none"] > 0),
        ]
    )

    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for candidate in interleave(rankings):
        index = int(candidate["global_index"])
        if index in seen:
            continue
        selected.append(candidate)
        seen.add(index)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"Selected {len(selected)} unique records, expected {count}")
    return selected


def write_recipe(paths: list[Path], output: Path, count: int) -> dict[str, Any]:
    selected = select(paths, count)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in selected:
        grouped[(candidate["task"], candidate["route"])].append(candidate)

    annotation = []
    for task in TASKS:
        for route in ROUTES:
            candidates = grouped.get((task, route)) or []
            if not candidates:
                continue
            destination = output / "annotations" / task / route / "part-00000.jsonl"
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + f".tmp.{os.getpid()}")
            with temporary.open("w", encoding="utf-8") as handle:
                for candidate in candidates:
                    handle.write(
                        json.dumps(candidate["row"], ensure_ascii=True, separators=(",", ":"))
                        + "\n"
                    )
            os.replace(temporary, destination)
            annotation.append(str(destination.resolve()))

    recipe = output / "hard_smoke_recipe.json"
    atomic_json(
        recipe,
        {
            "schema_version": "pixel-pivr-hard-smoke-recipe-v1",
            "annotation": annotation,
            "records": len(selected),
        },
    )
    report = {
        "schema_version": "pixel-pivr-hard-smoke-selection-v1",
        "records": len(selected),
        "recipe": str(recipe.resolve()),
        "recipe_sha256": sha256_file(recipe),
        "route_counts": {
            f"{task}.{route}": len(values)
            for (task, route), values in sorted(grouped.items())
        },
        "selection": [
            {key: value for key, value in candidate.items() if key != "row"}
            for candidate in selected
        ],
    }
    atomic_json(output / "hard_smoke_selection.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records", type=int, required=True)
    args = parser.parse_args()
    if args.records < 1:
        parser.error("--records must be positive")
    missing = [str(path) for path in args.jsonl if not path.is_file()]
    if missing:
        parser.error(f"Missing JSONL inputs: {missing}")
    print(
        json.dumps(
            write_recipe(args.jsonl, args.output.resolve(), args.records),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
