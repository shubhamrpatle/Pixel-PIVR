#!/usr/bin/env python3
"""Static destination-machine checks for full-scale Pixel-PIVR runs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from pixel_pivr.io import count_jsonl


EXPECTED_SCHEMA = "pixel-pivr-hf-hbb-v1"


def recipe_paths(root: Path, recipe: Path) -> list[Path]:
    payload = json.loads(recipe.read_text(encoding="utf-8"))
    values = payload.get("annotation")
    if not isinstance(values, list) or not values:
        raise ValueError(f"Recipe has no annotation list: {recipe}")
    paths = [Path(value) if Path(value).is_absolute() else root / value for value in values]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Recipe has missing shards: {missing[:5]}")
    return paths


def inspect_recipe(root: Path, recipe: Path, global_batch: int) -> dict[str, Any]:
    paths = recipe_paths(root, recipe)
    records = sum(count_jsonl(path) for path in paths)
    padding = (-records) % global_batch
    return {
        "recipe": str(recipe.resolve()),
        "shards": len(paths),
        "records": records,
        "padding_records": padding,
        "one_pass_optimizer_steps": (records + padding) // global_batch,
    }


def model_weights(model: Path) -> list[str]:
    index = model / "model.safetensors.index.json"
    if index.is_file():
        payload = json.loads(index.read_text(encoding="utf-8"))
        names = sorted(set((payload.get("weight_map") or {}).values()))
        missing = [name for name in names if not (model / name).is_file()]
        if not names or missing:
            raise FileNotFoundError(f"Model weight index is incomplete: {missing[:5]}")
        return names
    names = [name for name in ("model.safetensors", "pytorch_model.bin") if (model / name).is_file()]
    if not names:
        raise FileNotFoundError(f"No model weights or weight index found under {model}")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--eagle-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--global-batch", type=int, default=4)
    args = parser.parse_args()
    if sys.version_info < (3, 10):
        raise RuntimeError("Pixel-PIVR requires Python 3.10 or newer")
    if args.global_batch <= 0:
        raise ValueError("--global-batch must be positive")

    model = args.model.resolve()
    eagle = args.eagle_root.resolve()
    data = args.data_root.resolve()
    required = (
        model / "config.json",
        eagle / "eaglevl/train/locany_finetune_magi_stream.py",
        data / "manifest.json",
        data / "recipes/stage1_coarse.json",
        data / "recipes/stage2_dense_balanced.json",
        data / "recipes/validation_all_tasks.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required assets: {missing}")

    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != EXPECTED_SCHEMA:
        raise ValueError(
            f"Dataset schema {manifest.get('schema_version')!r} != {EXPECTED_SCHEMA!r}"
        )
    overlap = manifest.get("image_hash_overlap") or {}
    if any(int(value) for value in overlap.values()):
        raise ValueError(f"Dataset split leakage is nonzero: {overlap}")

    image_inventory = {}
    for split in ("train", "validation", "test"):
        expected = manifest["images"][split]
        paths = [path for path in (data / "images" / split).rglob("*") if path.is_file()]
        actual_files = len(paths)
        actual_bytes = sum(path.stat().st_size for path in paths)
        if actual_files != int(expected["unique"]) or actual_bytes != int(expected["bytes"]):
            raise ValueError(
                f"Incomplete {split} image payload: files={actual_files}, bytes={actual_bytes}; "
                f"expected files={expected['unique']}, bytes={expected['bytes']}"
            )
        image_inventory[split] = {"files": actual_files, "bytes": actual_bytes}

    packages = {}
    for name in ("torch", "transformers", "peft", "PIL"):
        spec = importlib.util.find_spec(name)
        if spec is None:
            raise ModuleNotFoundError(f"Required package is not installed: {name}")
        module = __import__(name)
        packages[name] = str(getattr(module, "__version__", "installed"))

    recipes = {
        stage: inspect_recipe(data, data / "recipes" / filename, args.global_batch)
        for stage, filename in (
            ("stage1", "stage1_coarse.json"),
            ("stage2", "stage2_dense_balanced.json"),
            ("validation", "validation_all_tasks.json"),
        )
    }
    report = {
        "status": "passed",
        "python": sys.version.split()[0],
        "model": str(model),
        "model_weight_files": model_weights(model),
        "eagle_root": str(eagle),
        "data_root": str(data),
        "dataset_schema": manifest["schema_version"],
        "image_hash_overlap": overlap,
        "image_inventory": image_inventory,
        "global_batch": args.global_batch,
        "packages": packages,
        "recipes": recipes,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
