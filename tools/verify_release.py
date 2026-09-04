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
        ROOT / "reference/pixel_crop_reencoded_380_verification.json",
        ROOT / "reference/full_scale_magnified_v2_verification.json",
        ROOT / "scripts/train_distributed.sh",
        ROOT / "scripts/bootstrap_machine.sh",
        ROOT / "scripts/configure_a100_node.sh",
        ROOT / "scripts/run_full_pipeline.sh",
        ROOT / "scripts/evaluate_all.sh",
        ROOT / "scripts/publish_hf_dataset.sh",
        ROOT / "tools/build_asset_manifest.py",
        ROOT / "tools/preflight.py",
        ROOT / "tools/prepare_evaluation.py",
        ROOT / "tools/source_manifest.py",
        ROOT / "tools/build_hf_upload_bundle.py",
        ROOT / "tools/materialize_hf_dataset.py",
        ROOT / "tools/package_hf_magnified_v2.py",
        ROOT / "tools/verify_hf_snapshot.py",
        ROOT / "tools/shard_evaluation_manifest.py",
        ROOT / "tools/merge_inference_shards.py",
        ROOT / "tools/build_hard_smoke_recipe.py",
        ROOT / "patches/eagle_virtual_crop_v1.patch",
        SRC / "train.py",
        SRC / "decoder.py",
        SRC / "audit.py",
        SRC / "infer.py",
        SRC / "magnified_roi.py",
        SRC / "magnified_modeling.py",
        SRC / "magnified_decoder.py",
        ROOT / "configs/magnified_preprojector_16k.env.example",
        ROOT / "configs/full_scale.env.example",
        ROOT / "docs/A100_8GPU_FULL_SCALE_RUNBOOK.md",
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
    pixel_crop = json.loads(
        (ROOT / "reference/pixel_crop_reencoded_380_verification.json").read_text(
            encoding="utf-8"
        )
    )
    assert pixel_crop["status"] == "passed"
    assert pixel_crop["dataset"]["train_records"] == 16000
    assert pixel_crop["dataset"]["benchmark_holdout_overlap"] == 0
    assert pixel_crop["dataset"]["target_truncation"] == 0
    assert pixel_crop["exact_loader"]["records_checked"] == 17000
    assert pixel_crop["exact_loader"]["maximum_post_mtp_sequence_tokens"] <= 8192
    assert pixel_crop["training_smoke"]["local_reentry_exposures"] > 0

    full_scale = json.loads(
        (ROOT / "reference/full_scale_magnified_v2_verification.json").read_text(
            encoding="utf-8"
        )
    )
    assert full_scale["status"] == "local_release_passed_remote_pending"
    assert full_scale["architecture_contract"] == "pixel_crop_144to384_v2"
    assert full_scale["dataset"]["materialized_images"] == 124325
    assert not any(full_scale["dataset"]["image_hash_overlap"].values())
    assert full_scale["training"]["stage1"]["one_pass_optimizer_steps"] == 57533
    assert full_scale["training"]["stage2"]["one_pass_optimizer_steps"] == 262572
    assert full_scale["training"]["loss_balancing"] == "source_query_task"
    assert full_scale["exact_loader_stress"]["records_skipped"] == 0
    assert full_scale["exact_loader_stress"]["maximum_post_mtp_sequence_tokens"] <= 32768
    evaluation = full_scale["evaluation_contract"]
    assert evaluation["detection"]["DIOR"] == {
        "records": 586,
        "targets": 3379,
        "classes": 20,
        "ontology": "dior_raw_20_class",
        "keeps_overpass_distinct": True,
    }
    assert evaluation["detection"]["DOTAv2"] == {
        "records": 874,
        "targets": 29329,
        "classes": 18,
        "ontology": "dotav2_raw_18_class",
        "keeps_plane_and_vehicle_sizes_distinct": True,
    }
    assert evaluation["grounding"]["DIOR-RSVG"] == 7500
    assert evaluation["grounding"]["VRSBench-VG_leakage_controlled"] == 16154
    assert evaluation["pointing"]["DOTAv2-Balanced100_targets"] == 9045
    assert evaluation["benchmark_images_used_for_training_or_validation"] == 0
    execution = full_scale["execution_safety"]
    assert execution["checkpoint_schema"] == "pixel-pivr-lora-checkpoint-v5"
    assert execution["completion_schema"] == "pixel-pivr-done-v3"
    assert execution["run_contract_schema"] == "pixel-pivr-run-contract-v5"
    assert execution["scheduled_annotation_checksums_verified_each_launch"] is True
    assert execution["scheduled_annotation_files"] == 47
    assert execution["scheduled_train_and_monitor_records"] == 2562980
    assert execution["required_gpu_processes"] == 8
    assert full_scale["hub_bundle"]["archive_member_sha256_verification"] is True
    assert full_scale["hub_bundle"]["remote_verification"] == "pending"

    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    for requirement in (
        "torch==2.5.1",
        "transformers==4.57.1",
        "liger-kernel==0.3.1",
        "datasets==5.0.0",
        "fsspec[http]==2026.4.0",
    ):
        assert requirement in requirements

    for shell in sorted((ROOT / "scripts").glob("*.sh")):
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
