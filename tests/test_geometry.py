from __future__ import annotations

import unittest

from pixel_pivr.geometry import (
    class_nms,
    detection_counts,
    parse_labeled_points,
    parse_single_hbb,
    point_centered_crop,
)


class GeometryTests(unittest.TestCase):
    def test_parse_points_preserves_active_ref_and_deduplicates(self) -> None:
        answer = (
            "<ref>ship</ref><box><100><200></box><box><100><200></box>"
            "<ref>vehicle</ref><box><200><200><400><600></box>"
        )
        points = parse_labeled_points(answer, 1000, 500)
        self.assertEqual(
            [(row["label"], row["point_norm"]) for row in points],
            [("ship", [100, 200]), ("vehicle", [300, 400])],
        )

    def test_parse_atomic_hbb(self) -> None:
        prediction = parse_single_hbb(
            "<ref>ship</ref><box><100><200><400><600></box>",
            "ship",
            200,
            100,
        )
        self.assertIsNotNone(prediction)
        self.assertEqual(prediction["hbox"], [20.0, 20.0, 80.0, 60.0])

    def test_crop_is_clamped_without_padding(self) -> None:
        self.assertEqual(point_centered_crop([3, 4], 100, 80, 32), (0, 0, 32, 32))
        self.assertEqual(
            point_centered_crop([98, 79], 100, 80, 32), (68, 48, 100, 80)
        )

    def test_class_nms_and_matching_are_class_aware(self) -> None:
        predictions = [
            {"label": "ship", "hbox": [0, 0, 10, 10], "score": 1.0},
            {"label": "ship", "hbox": [1, 1, 10, 10], "score": 1.0},
            {"label": "vehicle", "hbox": [0, 0, 10, 10], "score": 1.0},
        ]
        kept = class_nms(predictions, 0.5)
        self.assertEqual(len(kept), 2)
        gt = [
            {"label": "ship", "hbox": [0, 0, 10, 10]},
            {"label": "vehicle", "hbox": [0, 0, 10, 10]},
        ]
        self.assertEqual(detection_counts(gt, kept), (2, 0, 0))
