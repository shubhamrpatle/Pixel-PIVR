from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pixel_pivr.infer import row_class_prompts, row_label
from tools.prepare_dotav2_balanced100 import (
    balanced100_rows,
    validate_balanced100_rows,
)
from tools.prepare_evaluation import DIOR_CLASSES, DOTA_CLASSES, detection_rows


class PrepareEvaluationTests(unittest.TestCase):
    def test_balanced100_builder_requires_and_preserves_raw_labels(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image_root = root / "images"
            image_root.mkdir()
            (image_root / "example.png").write_bytes(b"test image placeholder")
            source = root / "balanced100.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "image_id": "example",
                        "image_path": "example.png",
                        "objects": [
                            {
                                "raw_class_name": "plane",
                                "class_name": "airplane",
                                "hbox": [1, 2, 3, 4],
                            },
                            {
                                "raw_class_name": "small vehicle",
                                "class_name": "vehicle",
                                "hbox": [5, 6, 7, 8],
                            },
                            {
                                "raw_class_name": "large vehicle",
                                "class_name": "vehicle",
                                "hbox": [9, 10, 11, 12],
                            },
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            values = balanced100_rows([source], image_root)

        counts = validate_balanced100_rows(
            values,
            expected_images=1,
            expected_class_counts={
                "plane": 1,
                "small vehicle": 1,
                "large vehicle": 1,
            },
        )
        self.assertEqual(
            [target["label"] for target in values[0]["gt"]],
            ["plane", "small vehicle", "large vehicle"],
        )
        self.assertEqual(values[0]["class_ontology"], "dotav2_raw_18_class")
        self.assertEqual(values[0]["gt_label_field"], "raw_class_name")
        self.assertNotIn("vehicle", counts)
        self.assertNotIn("airplane", counts)

    def test_balanced100_builder_fails_without_raw_label(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image_root = root / "images"
            image_root.mkdir()
            (image_root / "example.png").write_bytes(b"test image placeholder")
            source = root / "balanced100.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "image_id": "example",
                        "image_path": "example.png",
                        "objects": [
                            {"class_name": "vehicle", "hbox": [1, 2, 3, 4]}
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "missing raw_class_name"):
                balanced100_rows([source], image_root)

    def test_balanced100_validation_rejects_collapsed_labels(self) -> None:
        values = [
            {
                "sample_key": "DOTAv2:balanced100:bad",
                "gt": [{"label": "vehicle", "hbox": [1, 2, 3, 4]}],
            }
        ]
        with self.assertRaisesRegex(ValueError, "Collapsed DOTAv2 labels"):
            validate_balanced100_rows(
                values,
                expected_images=1,
                expected_class_counts={"vehicle": 1},
            )

    def test_dotav2_preserves_raw_18_class_ontology(self) -> None:
        with TemporaryDirectory() as directory:
            source = Path(directory) / "part-00000.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "image_id": "example",
                        "image_path": "images/test/example.png",
                        "objects": [
                            {
                                "raw_class_name": "plane",
                                "class_name": "airplane",
                                "hbox": [1, 2, 3, 4],
                            },
                            {
                                "raw_class_name": "small vehicle",
                                "class_name": "vehicle",
                                "hbox": [5, 6, 7, 8],
                            },
                            {
                                "raw_class_name": "large vehicle",
                                "class_name": "vehicle",
                                "hbox": [9, 10, 11, 12],
                            },
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            row = list(detection_rows([source], "DOTAv2"))[0]

        self.assertEqual(tuple(row["classes"]), DOTA_CLASSES)
        self.assertEqual(row["class_ontology"], "dotav2_raw_18_class")
        self.assertEqual(row["gt_label_field"], "raw_class_name")
        self.assertEqual(
            [target["label"] for target in row["gt"]],
            ["plane", "small vehicle", "large vehicle"],
        )
        self.assertEqual(row_label("airplane", row), "plane")
        self.assertEqual(row_label("vehicle", row), "vehicle")
        self.assertEqual(row_class_prompts(list(DOTA_CLASSES), row), list(DOTA_CLASSES))

    def test_dior_uses_readable_prompts_but_scores_raw_labels(self) -> None:
        with TemporaryDirectory() as directory:
            source = Path(directory) / "part-00000.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "image_id": "example",
                        "image_path": "images/test/example.png",
                        "objects": [
                            {
                                "raw_class_name": "groundtrackfield",
                                "class_name": "groundtrackfield",
                                "hbox": [1, 2, 3, 4],
                            },
                            {
                                "raw_class_name": "overpass",
                                "class_name": "bridge",
                                "hbox": [5, 6, 7, 8],
                            },
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            row = list(detection_rows([source], "DIOR"))[0]

        prompts = row_class_prompts(list(DIOR_CLASSES), row)
        self.assertEqual(
            prompts[list(DIOR_CLASSES).index("groundtrackfield")],
            "ground track field",
        )
        self.assertEqual(row_label("ground track field", row), "groundtrackfield")
        self.assertEqual(row_label("overpass", row), "overpass")
        self.assertEqual(row_label("bridge", row), "bridge")
        self.assertEqual(
            [target["label"] for target in row["gt"]],
            ["groundtrackfield", "overpass"],
        )

    def test_unknown_detection_label_fails_closed(self) -> None:
        with TemporaryDirectory() as directory:
            source = Path(directory) / "part-00000.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "image_id": "bad",
                        "image_path": "images/test/bad.png",
                        "objects": [
                            {
                                "raw_class_name": "unknown aircraft",
                                "class_name": "airplane",
                                "hbox": [1, 2, 3, 4],
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "outside ontology"):
                list(detection_rows([source], "DOTAv2"))


if __name__ == "__main__":
    unittest.main()
