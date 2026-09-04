from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from pixel_pivr.audit import loader_stress_rows, validate_row
from pixel_pivr.data import verify_packaged_data_signatures


def make_image(path: Path) -> None:
    Image.new("RGB", (64, 64), "white").save(path)


class DataContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_packaged_signatures_must_match_checksum_ledger(self) -> None:
        shard = self.root / "annotations/train/detection/round1_global_point/part.jsonl"
        shard.parent.mkdir(parents=True)
        shard.write_text('{"sample": 1}\n', encoding="utf-8")
        digest = hashlib.sha256(shard.read_bytes()).hexdigest()
        relative = shard.relative_to(self.root).as_posix()
        (self.root / "SHA256SUMS").write_text(
            f"{digest}  {relative}\n", encoding="utf-8"
        )
        signature = [{"path": str(shard), "sha256": digest, "records": 1}]
        result = verify_packaged_data_signatures(self.root, signature)
        self.assertEqual(result["scheduled_annotation_files_verified"], 1)

        signature[0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            verify_packaged_data_signatures(self.root, signature)

    def test_global_point_contract(self) -> None:
        make_image(self.root / "global.png")
        row = {
        "conversations": [
            {"from": "human", "value": "<image>\nPoint to: ship."},
            {
                "from": "gpt",
                "value": "<ref>ship</ref><box><500><500></box>",
            },
        ],
        "image": "global.png",
        "meta": {
            "pivr_route": "global_point_discovery",
            "geometry_mode": "point",
            "target_count": 1,
        },
        }
        result = validate_row(row, self.root)
        self.assertEqual(result["targets"], 1)


    def test_reentry_contract(self) -> None:
        make_image(self.root / "global.png")
        row = {
        "conversations": [
            {"from": "human", "value": "<image><image>\nAddress A1."},
            {
                "from": "gpt",
                "value": "<box><400><400><600><600></box>",
            },
        ],
        "image": [
            "global.png",
            {
                "virtual_crop": True,
                "schema": "pixel-pivr-virtual-crop-v1",
                "path": "global.png",
                "crop_xyxy": [0, 0, 64, 64],
                "resize_hw": [384, 384],
                "resample": "lanczos",
            },
        ],
        "meta": {
            "pivr_route": "point_indexed_visual_reentry",
            "pivr_round2_route": "local_first",
            "geometry_mode": "hbb",
            "pivr_local_point": [500, 500],
            "pivr_target_fully_contained": True,
            "target_count": 1,
        },
        }
        result = validate_row(row, self.root)
        self.assertEqual(result["targets"], 1)


    def test_reentry_rejects_point_outside_target(self) -> None:
        make_image(self.root / "global.png")
        row = {
        "conversations": [
            {"from": "human", "value": "<image><image>\nAddress A1."},
            {
                "from": "gpt",
                "value": "<box><400><400><600><600></box>",
            },
        ],
        "image": [
            "global.png",
            {
                "virtual_crop": True,
                "schema": "pixel-pivr-virtual-crop-v1",
                "path": "global.png",
                "crop_xyxy": [0, 0, 64, 64],
                "resize_hw": [384, 384],
                "resample": "lanczos",
            },
        ],
        "meta": {
            "pivr_route": "point_indexed_visual_reentry",
            "pivr_round2_route": "local_first",
            "geometry_mode": "hbb",
            "pivr_local_point": [800, 800],
            "pivr_target_fully_contained": True,
            "target_count": 1,
        },
        }
        with self.assertRaisesRegex(ValueError, "outside target"):
            validate_row(row, self.root)

    def test_pixel_reentry_global_fallback_uses_box_only_global_coordinates(self) -> None:
        make_image(self.root / "global.png")
        row = {
            "conversations": [
                {"from": "human", "value": "<image>\nAddress A1."},
                {"from": "gpt", "value": "<box><400><400><600><600></box>"},
            ],
            "image": "global.png",
            "meta": {
                "record_id": "fallback",
                "pivr_route": "point_indexed_visual_reentry",
                "pivr_round2_route": "global_fallback",
                "geometry_mode": "hbb",
                "pivr_global_point": [500, 500],
                "pivr_target_fully_contained": False,
                "target_count": 1,
            },
        }
        result = validate_row(row, self.root)
        self.assertEqual(result["targets"], 1)

    def test_loader_stress_selection_covers_hard_and_none_rows(self) -> None:
        path = self.root / "records.jsonl"
        rows = [
            {
                "conversations": [{"from": "gpt", "value": "short"}],
                "meta": {
                    "record_id": "none",
                    "target_count": 0,
                    "pivr_source_image_size": [64, 64],
                },
            },
            {
                "conversations": [{"from": "gpt", "value": "x" * 100}],
                "meta": {
                    "record_id": "long",
                    "target_count": 2,
                    "pivr_source_image_size": [32, 32],
                },
            },
            {
                "conversations": [{"from": "gpt", "value": "medium"}],
                "meta": {
                    "record_id": "large",
                    "target_count": 1,
                    "pivr_source_image_size": [128, 128],
                },
            },
        ]
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        selected, evidence = loader_stress_rows([path])
        self.assertEqual(
            {row["meta"]["record_id"] for row in selected},
            {"none", "long", "large"},
        )
        self.assertEqual(len(selected), len(evidence))

    def test_magnified_reentry_uses_one_global_image_and_global_coordinates(self) -> None:
        make_image(self.root / "image.png")
        row = {
            "conversations": [
                {
                    "from": "human",
                    "value": (
                        "<image>\nLocate the single ship containing point "
                        "<box><500><500></box> in horizontal box format."
                    ),
                },
                {
                    "from": "gpt",
                    "value": "<ref>ship</ref><box><450><450><550><550></box>",
                },
            ],
            "image": "image.png",
            "meta": {
                "pivr_route": "point_indexed_visual_reentry",
                "geometry_mode": "hbb",
                "pivr_global_point": [500, 500],
                "pivr_target_fully_contained": True,
                "target_count": 1,
            },
        }
        result = validate_row(
            row, self.root, visual_context="preprojector_magnified_roi"
        )
        self.assertEqual(result["targets"], 1)
