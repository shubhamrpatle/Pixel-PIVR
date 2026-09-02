from __future__ import annotations

import unittest

import torch
from torch import nn

from pixel_pivr.magnified_roi import (
    MoonViTFeatureCache,
    extract_magnified_preprojector_roi,
    insert_virtual_image,
    merge_2x2,
    sliding_2x2,
)


class MagnifiedROITest(unittest.TestCase):
    def test_stride_two_matches_native_merger_order(self) -> None:
        features = torch.arange(4 * 6 * 3, dtype=torch.float32).view(24, 3)
        native = merge_2x2(features, (4, 6))
        sliding, grid_hw = sliding_2x2(features.view(4, 6, 3), stride=2)
        self.assertEqual(grid_hw, (2, 3))
        torch.testing.assert_close(sliding, native, rtol=0, atol=0)

    def test_380_pixel_stride_one_roi_has_676_tokens(self) -> None:
        hidden = 3
        features = torch.arange(74 * 74 * hidden, dtype=torch.float32).view(
            74 * 74, hidden
        )
        cache = MoonViTFeatureCache(
            unmerged=features,
            projected_global=torch.empty(37 * 37, 5),
            patch_grid_hw=(74, 74),
        )
        projector = nn.Linear(4 * hidden, 5, bias=False)
        roi = extract_magnified_preprojector_roi(
            cache,
            (500, 500),
            projector,
            requested_pixels=380,
            patch_size=14,
            stride=1,
        )
        self.assertEqual(roi.window_patch_hw, (27, 27))
        self.assertEqual(roi.effective_pixels_hw, (378, 378))
        self.assertEqual(roi.projector_grid_hw, (26, 26))
        self.assertEqual(tuple(roi.features.shape), (676, 5))

    def test_border_roi_shifts_without_padding(self) -> None:
        cache = MoonViTFeatureCache(
            unmerged=torch.randn(40 * 50, 2),
            projected_global=torch.randn(20 * 25, 4),
            patch_grid_hw=(40, 50),
        )
        roi = extract_magnified_preprojector_roi(
            cache,
            (0, 1000),
            nn.Linear(8, 4),
            requested_pixels=380,
            patch_size=14,
            stride=1,
        )
        self.assertEqual(roi.window_yx, (13, 0))
        self.assertEqual(roi.center_rc, (39, 0))
        self.assertEqual(roi.window_patch_hw, (27, 27))

    def test_virtual_image_preserves_targets_and_positions(self) -> None:
        ids = torch.tensor([10, 20, 99, 99, 21, 30, 31])
        labels = torch.tensor([-100, -100, -100, -100, -100, 30, 31])
        positions = torch.arange(ids.numel())
        new_ids, new_labels, new_positions = insert_virtual_image(
            ids,
            labels,
            positions,
            image_token_id=99,
            image_start_id=20,
            image_end_id=21,
            visual_tokens=3,
        )
        self.assertEqual(
            new_ids.tolist(), [10, 20, 99, 99, 21, 20, 99, 99, 99, 21, 30, 31]
        )
        self.assertEqual(new_labels[-2:].tolist(), [30, 31])
        self.assertTrue(new_labels[5:10].eq(-100).all())
        self.assertEqual(new_positions.tolist(), list(range(12)))


if __name__ == "__main__":
    unittest.main()
