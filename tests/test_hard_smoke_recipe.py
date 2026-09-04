from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tools.build_hard_smoke_recipe import write_recipe


class HardSmokeRecipeTests(unittest.TestCase):
    def test_selection_covers_every_available_task_route(self) -> None:
        scopes = (
            ("detection", "round1_global_point"),
            ("detection", "round2_point_box"),
            ("grounding", "round1_global_point"),
            ("grounding", "round2_point_box"),
            ("pointing", "round1_global_point"),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for task, route in scopes:
                path = root / "annotations" / task / route / "part.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                values = []
                for index in range(4):
                    target_count = index
                    answer = (
                        "<box>None</box>"
                        if index == 0
                        else "<box><1><2><3><4></box>" * index
                    )
                    values.append(
                        {
                            "conversations": [
                                {"from": "human", "value": "x" * (10 + index)},
                                {"from": "gpt", "value": answer},
                            ],
                            "image": ["global.jpg", {"virtual_crop": True}]
                            if route == "round2_point_box"
                            else "global.jpg",
                            "meta": {
                                "record_id": f"{task}-{route}-{index}",
                                "target_count": target_count,
                                "pivr_source_image_size": [100 + index, 200 + index],
                            },
                        }
                    )
                path.write_text(
                    "".join(json.dumps(value) + "\n" for value in values),
                    encoding="utf-8",
                )
                paths.append(path)

            report = write_recipe(paths, root / "smoke", 16)
            self.assertEqual(report["records"], 16)
            self.assertEqual(set(report["route_counts"]), {
                f"{task}.{route}" for task, route in scopes
            })
            recipe = json.loads(
                Path(report["recipe"]).read_text(encoding="utf-8")
            )
            self.assertEqual(recipe["records"], 16)
            observed = sum(
                sum(1 for line in Path(path).open(encoding="utf-8") if line.strip())
                for path in recipe["annotation"]
            )
            self.assertEqual(observed, 16)


if __name__ == "__main__":
    unittest.main()
