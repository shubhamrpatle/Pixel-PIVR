from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from pixel_pivr.data import LossBalancedDataset, task_loss_balance_contract
from pixel_pivr.modeling import weighted_training_objective
from pixel_pivr.train import (
    PARAMETER_SYNC_POLICY,
    completion_payload,
    truncate_curve,
    verify_resume_contract,
)


def signature(task: str, route: str, records: int) -> dict[str, object]:
    return {
        "path": f"/data/annotations/train/stage/{task}/{route}/part.jsonl",
        "records": records,
    }


class DummyDataset:
    def __init__(self, records: int) -> None:
        self.records = records

    def __len__(self) -> int:
        return self.records

    def __getitem__(self, index: int) -> dict[str, int]:
        return {"index": index}


class LossBalancingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.signature = [
            signature("detection", "round1_global_point", 2),
            signature("detection", "round2_point_box", 8),
            signature("grounding", "round1_global_point", 3),
            signature("grounding", "round2_point_box", 1),
            signature("pointing", "round1_global_point", 5),
        ]

    def test_source_query_task_weights_preserve_source_mixture(self) -> None:
        contract = task_loss_balance_contract(
            self.signature, "source_query_task"
        )
        self.assertAlmostEqual(contract["task_weights"]["detection"], 0.38)
        self.assertAlmostEqual(contract["task_weights"]["grounding"], 1.425)
        self.assertAlmostEqual(contract["task_weights"]["pointing"], 1.9)
        self.assertAlmostEqual(contract["mean_weight"], 1.0)
        weighted = {
            task: contract["flat_records"][task]
            * contract["task_weights"][task]
            / 19
            for task in ("detection", "grounding", "pointing")
        }
        self.assertAlmostEqual(weighted["detection"], 0.2)
        self.assertAlmostEqual(weighted["grounding"], 0.3)
        self.assertAlmostEqual(weighted["pointing"], 0.5)

    def test_dataset_attaches_weight_at_file_boundaries(self) -> None:
        contract = task_loss_balance_contract(
            self.signature, "source_query_task"
        )
        dataset = LossBalancedDataset(DummyDataset(19), self.signature, contract)
        self.assertAlmostEqual(dataset[0]["pixel_pivr_loss_weight"], 0.38)
        self.assertAlmostEqual(dataset[9]["pixel_pivr_loss_weight"], 0.38)
        self.assertAlmostEqual(dataset[10]["pixel_pivr_loss_weight"], 1.425)
        self.assertAlmostEqual(dataset[13]["pixel_pivr_loss_weight"], 1.425)
        self.assertAlmostEqual(dataset[14]["pixel_pivr_loss_weight"], 1.9)

    def test_weighted_objective_keeps_native_metric(self) -> None:
        native = torch.tensor(2.0, requires_grad=True)
        objective, metrics = weighted_training_objective(
            native, {"pixel_pivr_loss_weight": 1.5}
        )
        objective.backward()
        self.assertEqual(float(objective), 3.0)
        self.assertEqual(float(native.grad), 1.5)
        self.assertEqual(metrics["native_loss"], 2.0)
        self.assertEqual(metrics["loss"], 3.0)

    def test_resume_curve_discards_only_uncheckpointed_steps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training_curve.jsonl"
            path.write_text(
                "".join(
                    json.dumps({"step": step, "loss": step / 10}) + "\n"
                    for step in range(1, 6)
                ),
                encoding="utf-8",
            )
            self.assertEqual(truncate_curve(path, "step", 3), 2)
            retained = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([row["step"] for row in retained], [1, 2, 3])

    def test_completion_requires_exact_exposure_and_both_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "best.pt").write_bytes(b"best")
            (output / "last.pt").write_bytes(b"last")
            payload = completion_payload(
                output=output,
                step=3,
                exposure=24,
                required_exposures=24,
                train_records=23,
                world_size=8,
                gradient_accumulation=1,
                global_records_per_step=8,
                parameter_sync={"policy": PARAMETER_SYNC_POLICY},
                loss_balancing={"policy": "source_query_task"},
            )
            self.assertTrue(payload["complete_one_pass"])
            self.assertEqual(payload["padding_record_exposures"], 1)
            with self.assertRaisesRegex(RuntimeError, "Cannot certify completion"):
                completion_payload(
                    output=output,
                    step=3,
                    exposure=23,
                    required_exposures=24,
                    train_records=23,
                    world_size=8,
                    gradient_accumulation=1,
                    global_records_per_step=8,
                    parameter_sync={"policy": PARAMETER_SYNC_POLICY},
                    loss_balancing={"policy": "source_query_task"},
                )

    def test_resume_contract_rejects_learning_rate_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, data = root / "model", root / "data"
            model.mkdir()
            data.mkdir()
            args = SimpleNamespace(
                model=model,
                data_root=data,
                lora_rank=16,
                image_token_limit=6000,
                max_sequence=32768,
                visual_context="pixel_reencoded",
                magnified_roi_pixels=384,
                magnified_roi_stride=1,
                gradient_accumulation=1,
                allowed_padding_records=1,
                keep_recent_checkpoints=1,
                checkpoint_steps=1000,
                eval_steps=5000,
                log_steps=50,
                seed=20260904,
                learning_rate=1e-5,
                weight_decay=0.01,
                max_grad_norm=1.0,
                warmup_steps=600,
                workers=4,
                validation_sampling="first",
                validation_records=0,
                sample_order="shuffled",
            )
            train_signature = [{"path": "/train.jsonl", "records": 23}]
            validation_signature = [{"path": "/validation.jsonl", "records": 2}]
            loss_balancing = {"policy": "source_query_task"}
            config = {
                "base_model": str(model.resolve()),
                "data_root": str(data.resolve()),
                "train_data": train_signature,
                "validation_data": validation_signature,
                "lora_rank": 16,
                "image_token_limit": 6000,
                "max_sequence": 32768,
                "visual_context": "pixel_reencoded",
                "magnified_roi_pixels": 384,
                "magnified_roi_stride": 1,
                "gradient_accumulation": 1,
                "allowed_padding_records": 1,
                "keep_recent_checkpoints": 1,
                "checkpoint_steps": 1000,
                "eval_steps": 5000,
                "log_steps": 50,
                "max_steps": 3,
                "world_size": 8,
                "seed": 20260904,
                "learning_rate": 1e-5,
                "weight_decay": 0.01,
                "max_grad_norm": 1.0,
                "warmup_steps": 600,
                "workers": 4,
                "vision_attention": "flash_attention_2",
                "validation_sampling": "first",
                "validation_records": 0,
                "init_adapter": None,
                "sample_order": "shuffled",
                "synchronized_trainable_initialization": True,
                "distributed_parameter_sync": PARAMETER_SYNC_POLICY,
                "loss_balancing": loss_balancing,
            }
            kwargs = {
                "args": args,
                "world_size": 8,
                "max_steps": 3,
                "train_signature": train_signature,
                "validation_signature": validation_signature,
                "init_adapter_signature": None,
                "loss_balancing": loss_balancing,
                "vision_attention": "flash_attention_2",
            }
            verify_resume_contract({"config": config}, **kwargs)
            args.learning_rate = 2e-5
            with self.assertRaisesRegex(RuntimeError, "learning_rate"):
                verify_resume_contract({"config": config}, **kwargs)


if __name__ == "__main__":
    unittest.main()
