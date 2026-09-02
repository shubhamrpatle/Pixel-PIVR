from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from pixel_pivr.audit import validate_row


def make_image(path: Path) -> None:
    Image.new("RGB", (64, 64), "white").save(path)


class DataContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

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
        make_image(self.root / "crop.png")
        row = {
        "conversations": [
            {"from": "human", "value": "<image><image>\nAddress A1."},
            {
                "from": "gpt",
                "value": "<ref>ship</ref><box><400><400><600><600></box>",
            },
        ],
        "image": ["global.png", "crop.png"],
        "meta": {
            "pivr_route": "point_indexed_visual_reentry",
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
        make_image(self.root / "crop.png")
        row = {
        "conversations": [
            {"from": "human", "value": "<image><image>\nAddress A1."},
            {
                "from": "gpt",
                "value": "<ref>ship</ref><box><400><400><600><600></box>",
            },
        ],
        "image": ["global.png", "crop.png"],
        "meta": {
            "pivr_route": "point_indexed_visual_reentry",
            "geometry_mode": "hbb",
            "pivr_local_point": [800, 800],
            "pivr_target_fully_contained": True,
            "target_count": 1,
        },
        }
        with self.assertRaisesRegex(ValueError, "outside target"):
            validate_row(row, self.root)

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
