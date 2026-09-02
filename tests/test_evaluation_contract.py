from __future__ import annotations

import unittest

from pixel_pivr.infer import pointing_counts, sample_seed, update_metrics
from collections import Counter


class EvaluationContractTests(unittest.TestCase):
    def test_pointing_uses_one_to_one_containment(self) -> None:
        gt = [
            {"label": "ship", "hbox": [0, 0, 10, 10]},
            {"label": "ship", "hbox": [20, 20, 30, 30]},
        ]
        points = [
            {"label": "ship", "point": [5, 5]},
            {"label": "ship", "point": [6, 6]},
            {"label": "ship", "point": [25, 25]},
        ]
        self.assertEqual(pointing_counts(gt, points), (2, 1, 0))

    def test_resume_metric_reduction(self) -> None:
        detection, pointing, grounding = Counter(), Counter(), Counter()
        update_metrics(
            {"task": "grounding", "metric": {"iou": 0.75}},
            detection,
            pointing,
            grounding,
        )
        self.assertEqual(grounding["queries"], 1)
        self.assertEqual(grounding["acc_0_5"], 1)
        self.assertEqual(grounding["acc_0_7"], 1)

    def test_sample_seed_is_stable_and_sample_specific(self) -> None:
        self.assertEqual(sample_seed(7, "sample-a"), sample_seed(7, "sample-a"))
        self.assertNotEqual(sample_seed(7, "sample-a"), sample_seed(7, "sample-b"))


if __name__ == "__main__":
    unittest.main()
