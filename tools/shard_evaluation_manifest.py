#!/usr/bin/env python3
"""Split one evaluation JSONL into deterministic, balanced GPU shards."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


def atomic_lines(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(row + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, required=True)
    args = parser.parse_args()
    if args.shards < 1:
        parser.error("--shards must be positive")
    lines = [line for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Empty evaluation manifest: {args.input}")
    keys: set[str] = set()
    for number, line in enumerate(lines, 1):
        row = json.loads(line)
        key = str(row.get("sample_key") or "")
        if not key or key in keys:
            raise ValueError(f"Missing/duplicate sample_key at {args.input}:{number}: {key!r}")
        keys.add(key)
    shard_count = min(args.shards, len(lines))
    buckets = [[] for _ in range(shard_count)]
    for index, line in enumerate(lines):
        buckets[index % shard_count].append(line)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    expected_names = {f"part-{index:05d}.jsonl" for index in range(shard_count)}
    unexpected = [
        path for path in args.output_dir.glob("part-*.jsonl") if path.name not in expected_names
    ]
    if unexpected:
        raise RuntimeError(f"Stale shard files present: {unexpected}")
    counts = []
    for index, bucket in enumerate(buckets):
        destination = args.output_dir / f"part-{index:05d}.jsonl"
        atomic_lines(destination, bucket)
        counts.append(len(bucket))
    summary = {
        "schema_version": "pixel-pivr-evaluation-shards-v1",
        "source": str(args.input.resolve()),
        "records": len(lines),
        "shards": shard_count,
        "records_per_shard": counts,
    }
    (args.output_dir / "shards.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
