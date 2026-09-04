#!/usr/bin/env python3
"""Fail-closed destination-machine checks for full-scale Pixel-PIVR."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from pixel_pivr.audit import exact_loader_audit, loader_stress_rows
from pixel_pivr.data import task_loss_balance_contract
from pixel_pivr.io import count_jsonl

from package_hf_magnified_v2 import (
    FALLBACK_EDGE_MARGIN_PIXELS,
    LOCAL_INPUT_SIDE,
    PROCESSOR_ALIGNED_SIDE,
    SCHEMA,
    SOURCE_CROP_SIDE,
    verify_package,
)
from prepare_evaluation import DETECTION_ONTOLOGIES, detection_rows


EAGLE_REVISION = "8442db3b79f7fd2357e468e6eecdd9b6a82049ff"
EAGLE_PATCH_MARKER = "PIXEL_PIVR_VIRTUAL_CROP_V1"
EAGLE_STRICT_MARKER = "strict coverage refuses sample replacement"
EAGLE_PATCHED_FILE_SHA256 = {
    "eaglevl/train/tools.py": "0acc434441dd79bbe1890d721df48ba71c7abcb7f7ff7c6cbae93bf34c4a34cc",
    "eaglevl/train/locany_finetune_magi_stream.py": "1108e6f5074e39b971f3da0ee056b0c92ec0b3ab56a80c98ceab2b470119f2e0",
}
EXPECTED_PACKAGES = {
    # import name: (distribution name, tested version prefix)
    "torch": ("torch", "2.5.1"),
    "torchvision": ("torchvision", "0.20.1"),
    "transformers": ("transformers", "4.57.1"),
    "tokenizers": ("tokenizers", "0.22.0"),
    "peft": ("peft", "0.12.0"),
    "accelerate": ("accelerate", "1.5.2"),
    "deepspeed": ("deepspeed", "0.15.4"),
    "liger_kernel": ("liger-kernel", "0.3.1"),
    "datasets": ("datasets", "5.0.0"),
    "huggingface_hub": ("huggingface-hub", "0.36.2"),
    "numpy": ("numpy", "1.26.4"),
    "PIL": ("Pillow", "11.1.0"),
    "cv2": ("opencv-python-headless", "4.11.0.86"),
}
EXPECTED_FLASH_ATTN = "2.7.4.post1"


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


def inspect_recipe(
    root: Path,
    recipe: Path,
    global_batch: int,
    *,
    loss_balancing: str | None = None,
) -> dict[str, Any]:
    paths = recipe_paths(root, recipe)
    signature = [
        {"path": str(path.resolve()), "records": count_jsonl(path)} for path in paths
    ]
    records = sum(int(row["records"]) for row in signature)
    padding = (-records) % global_batch
    result = {
        "recipe": str(recipe.resolve()),
        "shards": len(paths),
        "records": records,
        "padding_records": padding,
        "one_pass_optimizer_steps": (records + padding) // global_batch,
    }
    if loss_balancing is not None:
        result["source_query_task_loss_balance"] = task_loss_balance_contract(
            signature, loss_balancing
        )
    return result


def inspect_detection_ontologies(
    root: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    """Verify that portable evaluation retains each benchmark's raw labels."""
    output: dict[str, Any] = {}
    configured = manifest["recipes"]["test"]["detection"]
    for benchmark, relative_paths in configured.items():
        paths = [root / value for value in relative_paths]
        converted = list(detection_rows(paths, benchmark))
        expected_records = int(
            manifest["records"][f"test.detection.{benchmark}.records"]
        )
        if len(converted) != expected_records:
            raise RuntimeError(
                f"{benchmark} portable evaluation count changed: "
                f"{len(converted)} != {expected_records}"
            )
        expected_classes = list(DETECTION_ONTOLOGIES[benchmark]["classes"])
        if any(row["classes"] != expected_classes for row in converted):
            raise RuntimeError(f"{benchmark} prompt ontology is inconsistent")
        expected_prompts = dict(DETECTION_ONTOLOGIES[benchmark]["class_prompts"])
        if any(row["class_prompts"] != expected_prompts for row in converted):
            raise RuntimeError(f"{benchmark} model-facing class prompts are inconsistent")
        labels = Counter(
            target["label"] for row in converted for target in row.get("gt", [])
        )
        if benchmark == "DOTAv2":
            for required in ("plane", "small vehicle", "large vehicle"):
                if labels[required] <= 0:
                    raise RuntimeError(
                        f"DOTAv2 lost required raw class {required!r}"
                    )
            if labels["airplane"] or labels["vehicle"]:
                raise RuntimeError(
                    "DOTAv2 GT was collapsed to airplane/vehicle during conversion"
                )
        if benchmark == "DIOR" and labels["overpass"] <= 0:
            raise RuntimeError("DIOR lost the raw overpass class during conversion")
        output[benchmark] = {
            "ontology": DETECTION_ONTOLOGIES[benchmark]["name"],
            "records": len(converted),
            "targets": sum(labels.values()),
            "class_target_counts": dict(sorted(labels.items())),
            "class_prompts": expected_prompts,
            "label_aliases": dict(
                DETECTION_ONTOLOGIES[benchmark]["label_aliases"]
            ),
        }
    return output


def exact_loader_stress(
    *,
    recipe_groups: list[list[Path]],
    data_root: Path,
    model: Path,
    eagle_root: Path,
    image_token_limit: int,
    max_sequence: int,
) -> dict[str, Any]:
    """Run the real patched Eagle loader on hard rows from every release shard."""
    paths: list[Path] = []
    seen: set[Path] = set()
    for group in recipe_groups:
        for path in group:
            resolved = path.resolve()
            if resolved not in seen:
                paths.append(resolved)
                seen.add(resolved)
    rows, selection = loader_stress_rows(paths)
    if not rows or len(rows) != len(selection):
        raise RuntimeError("Exact-loader stress selection is empty or inconsistent")
    serialized = "".join(
        json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    with tempfile.TemporaryDirectory(prefix="pixel-pivr-loader-stress-") as directory:
        stress_path = Path(directory) / "hard_records.jsonl"
        stress_path.write_text(serialized, encoding="utf-8")
        exact = exact_loader_audit(
            argparse.Namespace(
                model=model,
                allow_download=False,
                image_token_limit=int(image_token_limit),
                max_sequence=int(max_sequence),
                jsonl=[stress_path],
                data_root=data_root,
                eagle_root=eagle_root,
                visual_context="pixel_reencoded",
                magnified_roi_pixels=384,
                magnified_roi_stride=1,
                multiscale_target_patches=27,
                limit=None,
            )
        )
    maximum_index = exact.get("max_post_mtp_index")
    maximum_source = (
        selection[int(maximum_index)] if maximum_index is not None else None
    )
    return {
        "annotation_shards": len(paths),
        "selected_records": len(rows),
        "selection_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "selection_policy": [
            "longest_conversation",
            "largest_source_image",
            "highest_target_count",
            "explicit_none_if_present",
        ],
        "maximum_source": maximum_source,
        "exact_loader": exact,
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


def verify_eagle(
    eagle_root: Path, expected_revision: str, patch: Path
) -> dict[str, Any]:
    tools = eagle_root / "eaglevl/train/tools.py"
    source = tools.read_text(encoding="utf-8")
    if EAGLE_PATCH_MARKER not in source:
        raise RuntimeError(
            f"Eagle is missing the required virtual-crop patch marker {EAGLE_PATCH_MARKER}"
        )
    trainer = eagle_root / "eaglevl/train/locany_finetune_magi_stream.py"
    if EAGLE_STRICT_MARKER not in trainer.read_text(encoding="utf-8"):
        raise RuntimeError("Eagle is missing the strict no-truncation/no-replacement guard")
    repository = eagle_root.parent
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    if expected_revision != EAGLE_REVISION:
        raise RuntimeError(
            f"Requested Eagle revision {expected_revision} != tested {EAGLE_REVISION}"
        )
    if revision != expected_revision:
        raise RuntimeError(f"Eagle revision {revision} != expected {expected_revision}")
    status = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    ).splitlines()
    changed = sorted(line[3:] for line in status if len(line) >= 4)
    expected_changed = sorted(
        (
            "Embodied/eaglevl/train/locany_finetune_magi_stream.py",
            "Embodied/eaglevl/train/tools.py",
        )
    )
    if changed != expected_changed:
        raise RuntimeError(
            f"Eagle has unexpected modified paths: {changed} != {expected_changed}"
        )
    patched_hashes = {
        relative: hashlib.sha256((eagle_root / relative).read_bytes()).hexdigest()
        for relative in EAGLE_PATCHED_FILE_SHA256
    }
    if patched_hashes != EAGLE_PATCHED_FILE_SHA256:
        raise RuntimeError(
            "Patched Eagle file hashes differ from the tested release:\n"
            + json.dumps(
                {
                    relative: {
                        "observed": patched_hashes.get(relative),
                        "expected": expected,
                    }
                    for relative, expected in EAGLE_PATCHED_FILE_SHA256.items()
                    if patched_hashes.get(relative) != expected
                },
                indent=2,
                sort_keys=True,
            )
        )
    reverse = subprocess.run(
        ["git", "-C", str(repository), "apply", "--reverse", "--check", str(patch)],
        capture_output=True,
        text=True,
    )
    if reverse.returncode:
        raise RuntimeError(
            "Eagle changes are not exactly reversible by the release patch:\n"
            + reverse.stderr
        )
    return {
        "repository_revision": revision,
        "virtual_crop_patch": EAGLE_PATCH_MARKER,
        "strict_coverage_guard": EAGLE_STRICT_MARKER,
        "modified_paths": changed,
        "patched_file_sha256": patched_hashes,
        "patch": str(patch.resolve()),
    }


def verify_release_identity(
    *,
    code_root: Path,
    code_revision: str,
    receipt_path: Path,
    data_root: Path,
    data_revision: str,
    model_path: Path,
    model_revision: str,
    eagle_root: Path,
    eagle_revision: str,
) -> dict[str, Any]:
    for name, value in (
        ("code", code_revision),
        ("dataset", data_revision),
        ("model", model_revision),
        ("Eagle", eagle_revision),
    ):
        if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
            raise RuntimeError(f"{name} revision is not an immutable 40-character SHA: {value!r}")
    actual_code = subprocess.check_output(
        ["git", "-C", str(code_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_code != code_revision:
        raise RuntimeError(f"Code revision {actual_code} != expected {code_revision}")
    dirty = subprocess.check_output(
        ["git", "-C", str(code_root), "status", "--porcelain", "--untracked-files=normal"],
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError(f"Pixel-PIVR checkout is not clean:\n{dirty}")

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected = {
        "dataset": {"revision": data_revision, "path": str(data_root)},
        "model": {"revision": model_revision, "path": str(model_path)},
        "eagle": {"revision": eagle_revision, "path": str(eagle_root.parent)},
    }
    mismatches: dict[str, Any] = {}
    for section, values in expected.items():
        observed = receipt.get(section) or {}
        for key, value in values.items():
            if observed.get(key) != value:
                mismatches[f"{section}.{key}"] = {
                    "observed": observed.get(key),
                    "expected": value,
                }
    if mismatches:
        raise RuntimeError(
            "Asset download receipt does not match this run:\n"
            + json.dumps(mismatches, indent=2, sort_keys=True)
        )
    return {
        "code_revision": actual_code,
        "data_revision": data_revision,
        "model_revision": model_revision,
        "eagle_revision": eagle_revision,
        "download_receipt": str(receipt_path),
    }


def inspect_packages(require_flash_attn: bool) -> dict[str, str]:
    values: dict[str, str] = {}
    for name, (distribution, expected) in EXPECTED_PACKAGES.items():
        importlib.import_module(name)
        actual = importlib.metadata.version(distribution)
        if not actual.startswith(expected):
            raise RuntimeError(
                f"{distribution} version {actual!r} != tested {expected!r}"
            )
        values[name] = actual
    if require_flash_attn:
        importlib.import_module("flash_attn")
        actual = importlib.metadata.version("flash-attn")
        if actual != EXPECTED_FLASH_ATTN:
            raise RuntimeError(
                f"flash_attn version {actual!r} != tested {EXPECTED_FLASH_ATTN!r}"
            )
        values["flash_attn"] = actual
    return values


def inspect_gpus(
    gpu_ids: list[int], required_name: str, minimum_memory_gib: float
) -> list[dict[str, Any]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    discovered = {}
    for line in output.splitlines():
        index, name, memory = [value.strip() for value in line.split(",", 2)]
        discovered[int(index)] = {"index": int(index), "name": name, "memory_mib": int(memory)}
    missing = sorted(set(gpu_ids) - set(discovered))
    if missing:
        raise RuntimeError(f"Configured GPUs do not exist: {missing}")
    selected = [discovered[index] for index in gpu_ids]
    for gpu in selected:
        if required_name.lower() not in gpu["name"].lower():
            raise RuntimeError(
                f"GPU {gpu['index']} is {gpu['name']!r}, expected name containing {required_name!r}"
            )
        if gpu["memory_mib"] < minimum_memory_gib * 1024:
            raise RuntimeError(
                f"GPU {gpu['index']} has {gpu['memory_mib']} MiB, below {minimum_memory_gib:.1f} GiB"
            )
    return selected


def model_context_limit(model: Path) -> int:
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    text = config.get("text_config") or {}
    value = int(text.get("max_position_embeddings") or config.get("max_position_embeddings") or 0)
    if value <= 0:
        raise ValueError("Could not determine Qwen max_position_embeddings")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--eagle-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--global-batch", type=int, default=8)
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--required-gpus", type=int, default=8)
    parser.add_argument("--required-gpu-name", default="A100")
    parser.add_argument("--minimum-gpu-memory-gib", type=float, default=75.0)
    parser.add_argument("--minimum-run-free-gib", type=float, default=100.0)
    parser.add_argument("--max-sequence", type=int, default=32768)
    parser.add_argument("--image-token-limit", type=int, default=6000)
    parser.add_argument(
        "--loss-balancing", choices=("source_query_task",), default="source_query_task"
    )
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--expected-code-revision", required=True)
    parser.add_argument("--download-receipt", type=Path, required=True)
    parser.add_argument("--expected-data-revision", required=True)
    parser.add_argument("--expected-model-revision", required=True)
    parser.add_argument("--expected-eagle-revision", required=True)
    parser.add_argument("--verify-image-hashes", action="store_true")
    parser.add_argument("--require-flash-attn", action="store_true")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError(f"Tested environment requires Python 3.10, got {sys.version.split()[0]}")
    if args.global_batch <= 0 or args.max_sequence <= 0 or args.image_token_limit <= 0:
        raise ValueError("Batch and token limits must be positive")

    gpu_ids = [int(value.strip()) for value in args.gpu_ids.split(",") if value.strip()]
    if len(gpu_ids) != args.required_gpus or len(set(gpu_ids)) != len(gpu_ids):
        raise RuntimeError(f"Expected {args.required_gpus} unique GPU IDs, received {gpu_ids}")
    if args.global_batch % len(gpu_ids):
        raise RuntimeError(
            f"Global batch {args.global_batch} is not divisible by {len(gpu_ids)} GPUs"
        )

    model = args.model.resolve()
    eagle = args.eagle_root.resolve()
    data = args.data_root.resolve()
    code_root = args.code_root.resolve()
    receipt = args.download_receipt.resolve()
    required = (
        model / "config.json",
        eagle / "eaglevl/train/locany_finetune_magi_stream.py",
        eagle / "eaglevl/train/tools.py",
        data / "manifest.json",
        data / "SHA256SUMS",
        data / "IMAGE_INVENTORY.jsonl",
        data / "recipes/stage1_coarse.json",
        data / "recipes/stage2_dense_balanced.json",
        data / "recipes/validation_all_tasks.json",
        data / "recipes/validation_monitor_all_tasks.json",
        receipt,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required assets: {missing}")

    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA:
        raise ValueError(f"Dataset schema {manifest.get('schema_version')!r} != {SCHEMA!r}")
    crop = manifest.get("pixel_reentry") or {}
    required_crop = {
        "source_crop_side_pixels": SOURCE_CROP_SIDE,
        "local_input_side_pixels": LOCAL_INPUT_SIDE,
        "la_processor_aligned_side_pixels": PROCESSOR_ALIGNED_SIDE,
        "fallback_edge_margin_pixels": FALLBACK_EDGE_MARGIN_PIXELS,
    }
    for key, expected in required_crop.items():
        if crop.get(key) != expected:
            raise ValueError(f"Dataset {key}={crop.get(key)!r}, expected {expected!r}")
    if args.max_sequence != int(manifest["training_contract"]["required_max_sequence"]):
        raise RuntimeError("MAX_SEQUENCE does not match the dataset training contract")
    if args.image_token_limit != int(manifest["training_contract"]["recommended_image_token_limit"]):
        raise RuntimeError("IMAGE_TOKEN_LIMIT does not match the dataset training contract")
    context_limit = model_context_limit(model)
    if context_limit < args.max_sequence:
        raise RuntimeError(f"Requested sequence {args.max_sequence} exceeds model context {context_limit}")

    package_audit = verify_package(data, verify_image_hashes=args.verify_image_hashes)
    detection_ontologies = inspect_detection_ontologies(data, manifest)
    recipes = {
        stage: inspect_recipe(
            data,
            data / "recipes" / filename,
            args.global_batch,
            loss_balancing=(
                args.loss_balancing if stage in {"stage1", "stage2"} else None
            ),
        )
        for stage, filename in (
            ("stage1", "stage1_coarse.json"),
            ("stage2", "stage2_dense_balanced.json"),
            ("validation_full", "validation_all_tasks.json"),
            ("validation_monitor", "validation_monitor_all_tasks.json"),
        )
    }
    for stage, manifest_key in (("stage1", "stage1_records"), ("stage2", "stage2_records")):
        expected = int(manifest["training_contract"][manifest_key])
        if recipes[stage]["records"] != expected:
            raise RuntimeError(f"{stage} recipe count changed")
    if recipes["validation_full"]["records"] != int(
        manifest["training_contract"]["validation_records"]
    ):
        raise RuntimeError("Full validation recipe count changed")
    if recipes["validation_monitor"]["records"] != int(
        manifest["training_contract"]["validation_monitor_records"]
    ):
        raise RuntimeError("Validation monitor recipe count changed")

    storage = None
    if args.run_root is not None:
        run_root = args.run_root.resolve()
        run_root.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(run_root)
        free_gib = usage.free / 1024**3
        if free_gib < args.minimum_run_free_gib:
            raise RuntimeError(
                f"Run filesystem has {free_gib:.1f} GiB free, below {args.minimum_run_free_gib:.1f} GiB"
            )
        storage = {"path": str(run_root), "free_gib": free_gib}

    eagle_report = verify_eagle(
        eagle,
        args.expected_eagle_revision,
        code_root / "patches/eagle_virtual_crop_v1.patch",
    )
    loader_stress = exact_loader_stress(
        recipe_groups=[
            recipe_paths(data, data / "recipes" / filename)
            for filename in (
                "stage1_coarse.json",
                "stage2_dense_balanced.json",
                "validation_all_tasks.json",
                "validation_monitor_all_tasks.json",
            )
        ],
        data_root=data,
        model=model,
        eagle_root=eagle,
        image_token_limit=args.image_token_limit,
        max_sequence=args.max_sequence,
    )

    report = {
        "status": "passed",
        "python": sys.version.split()[0],
        "model": str(model),
        "model_context_limit": context_limit,
        "model_weight_files": model_weights(model),
        "eagle_root": str(eagle),
        "eagle": eagle_report,
        "release_identity": verify_release_identity(
            code_root=code_root,
            code_revision=args.expected_code_revision,
            receipt_path=receipt,
            data_root=data,
            data_revision=args.expected_data_revision,
            model_path=model,
            model_revision=args.expected_model_revision,
            eagle_root=eagle,
            eagle_revision=args.expected_eagle_revision,
        ),
        "data_root": str(data),
        "dataset_schema": manifest["schema_version"],
        "package_audit": package_audit,
        "detection_evaluation_ontologies": detection_ontologies,
        "global_batch": args.global_batch,
        "gradient_accumulation_per_rank": args.global_batch // len(gpu_ids),
        "gpus": inspect_gpus(gpu_ids, args.required_gpu_name, args.minimum_gpu_memory_gib),
        "packages": inspect_packages(args.require_flash_attn),
        "recipes": recipes,
        "exact_loader_stress": loader_stress,
        "storage": storage,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
