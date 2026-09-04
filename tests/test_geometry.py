from __future__ import annotations

import unittest

from pixel_pivr.geometry import (
    box_touches_frame,
    cached_feature_point_prompt,
    class_nms,
    detection_counts,
    global_fallback_point_prompt,
    parse_labeled_points,
    parse_single_hbb,
    point_centered_crop,
    local_hbox_to_global,
    resize_local_point,
)


class GeometryTests(unittest.TestCase):
    def test_compact_point_prompt_matches_training_contract(self) -> None:
        self.assertEqual(
            cached_feature_point_prompt("small vehicle", [72, 72], 144, 144),
            "Locate the single small vehicle containing point "
            "<box><500><500></box> in the local view. Return its complete "
            "horizontal box, or None if the complete boundary is unavailable.",
        )
        self.assertEqual(
            global_fallback_point_prompt("small vehicle", [512, 256], 1024, 1024),
            "Locate the single small vehicle containing point "
            "<box><500><250></box> in the global image. Return its horizontal "
            "box, or None if absent.",
        )

    def test_crop_edge_boxes_request_fallback(self) -> None:
        self.assertTrue(box_touches_frame([1, 20, 100, 120], 384, 384, 2))
        self.assertTrue(box_touches_frame([20, 20, 383, 120], 384, 384, 2))
        self.assertFalse(box_touches_frame([3, 20, 380, 120], 384, 384, 2))

    def test_144_crop_magnified_to_384_round_trips_geometry(self) -> None:
        point = resize_local_point([72, 36], [144, 144], [384, 384])
        self.assertEqual(point, [192.0, 96.0])
        global_box = local_hbox_to_global(
            [96, 48, 288, 336], [100, 200, 244, 344], [384, 384]
        )
        self.assertEqual(global_box, [136.0, 218.0, 208.0, 326.0])

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
