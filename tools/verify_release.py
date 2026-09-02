#!/usr/bin/env python3
"""Static release audit that does not load model weights or GPUs."""

from __future__ import annotations

import ast
import json
import os
import py_compile
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src/pixel_pivr"
FORBIDDEN_IMPORTS = {
    "zero_shot_analysis",
    "scripts.train_pointrelay_pbd_lora_pilot",
    "adaptive_memory_pbd",
    "dense_address_field_pbd",
    "ravel_pbd",
}
FORBIDDEN_PATHS = {
    "/home/shubham/Grounding",
    "/data1/tmp_shubham",
    "/data2/shubham",
    "/share/data/drive_1",
}


def imported_modules(tree: ast.AST) -> set[str]:
    output = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            output.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            output.add(node.module)
    return output


def main() -> None:
    required = [
        ROOT / "README.md",
        ROOT / "pyproject.toml",
        ROOT / "requirements.txt",
        ROOT / "reference/release_verification.json",
        ROOT / "reference/magnified_preprojector_verification.json",
        ROOT / "scripts/train_distributed.sh",
        SRC / "train.py",
        SRC / "decoder.py",
        SRC / "audit.py",
        SRC / "infer.py",
        SRC / "magnified_roi.py",
        SRC / "magnified_modeling.py",
        SRC / "magnified_decoder.py",
        ROOT / "configs/magnified_preprojector_16k.env.example",
        ROOT / "docs/MAGNIFIED_PREPROJECTOR.md",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing release files: {missing}")

    files = sorted(SRC.glob("*.py"))
    for path in files:
        py_compile.compile(str(path), doraise=True)
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        imports = imported_modules(tree)
        forbidden = sorted(
            value
            for value in FORBIDDEN_IMPORTS
            if any(module == value or module.startswith(value + ".") for module in imports)
        )
        if forbidden:
            raise RuntimeError(f"{path} imports unrelated project code: {forbidden}")
        leaked = sorted(value for value in FORBIDDEN_PATHS if value in source)
        if leaked:
            raise RuntimeError(f"{path} contains machine-specific paths: {leaked}")

    result_path = ROOT / "reference/matched_16k_4k_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["training"]["records"] == 16000
    assert result["training"]["optimizer_steps"] == 4000
    assert result["pixel_pivr"]["sequential"]["true_positives"] == 564
    assert result["pixel_pivr"]["wave200"]["true_positives"] == 559
    verification = json.loads(
        (ROOT / "reference/release_verification.json").read_text(encoding="utf-8")
    )
    assert verification["data_audit"]["status"] == "passed"
    assert verification["distributed_training_smoke"]["status"] == "passed"
    assert verification["inference_smoke"]["status"] == "passed"
    magnified = json.loads(
        (ROOT / "reference/magnified_preprojector_verification.json").read_text(
            encoding="utf-8"
        )
    )
    assert magnified["status"] == "passed"
    assert magnified["mechanism"]["local_visual_tokens"] == 676
    assert magnified["exact_loader_smoke"]["maximum_augmented_sequence_tokens"] <= 8192
    assert magnified["training_smoke"]["includes_point_reentry_backward"] is True
    assert magnified["data_audit"]["benchmark_holdout_overlap"] == 0
    assert magnified["native_global_path_equivalence"]["exact_equal"] is True
    assert magnified["inference_smoke"]["global_moonvit_encodes"] == 1
    assert magnified["inference_smoke"]["local_moonvit_encodes"] == 0
    assert magnified["wave_input_deduplication"][
        "token_ids_exactly_match_repeated_processor_path"
    ] is True

    shell = ROOT / "scripts/train_distributed.sh"
    if not os.access(shell, os.X_OK):
        raise RuntimeError(f"Launcher is not executable: {shell}")
    print(
        json.dumps(
            {
                "status": "passed",
                "python_files": len(files),
                "forbidden_imports": 0,
                "machine_specific_source_paths": 0,
                "reference_contract": "matched 16K/4K",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    main()
