from __future__ import annotations

import unittest

import torch
from torch import nn

from pixel_pivr.adaptive_multiscale import (
    AdaptiveScaleFusion,
    extract_resampled_preprojector_roi,
)
from pixel_pivr.magnified_roi import MoonViTFeatureCache
from pixel_pivr.lora import restore_trainable_state, trainable_state


class AdaptiveMultiScaleTest(unittest.TestCase):
    def setUp(self) -> None:
        hidden = 3
        grid = torch.arange(74 * 74 * hidden, dtype=torch.float32).view(
            74 * 74, hidden
        )
        self.cache = MoonViTFeatureCache(
            unmerged=grid,
            projected_global=torch.empty(37 * 37, 5),
            patch_grid_hw=(74, 74),
        )
        self.projector = nn.Linear(4 * hidden, 5, bias=False)

    def test_three_fields_share_one_fixed_projected_grid(self) -> None:
        rois = [
            extract_resampled_preprojector_roi(
                self.cache,
                (500, 500),
                self.projector,
                requested_pixels=pixels,
                target_patches=27,
                patch_size=14,
                stride=1,
            )
            for pixels in (196, 378, 756)
        ]
        self.assertEqual([roi.source_patch_hw for roi in rois], [(14, 14), (27, 27), (54, 54)])
        self.assertEqual([tuple(roi.features.shape) for roi in rois], [(676, 5)] * 3)
        self.assertFalse(torch.equal(rois[0].features, rois[2].features))

    def test_fusion_keeps_token_count_and_prefers_medium_at_initialization(self) -> None:
        fusion = AdaptiveScaleFusion(8, 3, gate_hidden=4, preferred_scale=1)
        inputs = [torch.randn(12, 8) for _ in range(3)]
        output, weights = fusion(inputs)
        self.assertEqual(tuple(output.shape), (12, 8))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
        self.assertGreater(float(weights[1]), float(weights[0]))
        self.assertGreater(float(weights[1]), float(weights[2]))

    def test_fusion_receives_gradients(self) -> None:
        fusion = AdaptiveScaleFusion(8, 3, gate_hidden=4, preferred_scale=1)
        inputs = [torch.randn(12, 8, dtype=torch.bfloat16) for _ in range(3)]
        output, _weights = fusion(inputs)
        self.assertEqual(output.dtype, torch.bfloat16)
        output.square().mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in fusion.parameters()))

    def test_controller_round_trips_with_lora_checkpoint(self) -> None:
        class DummyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.language_model_lora_A = nn.Parameter(torch.randn(2, 2))
                self.pixel_pivr_controller = AdaptiveScaleFusion(8, 3, gate_hidden=4)

        model = DummyModel()
        saved = trainable_state(model)
        expected = {
            name: value.clone()
            for name, value in model.pixel_pivr_controller.state_dict().items()
        }
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        restore_trainable_state(model, saved, require_controller=True)
        for name, value in model.pixel_pivr_controller.state_dict().items():
            torch.testing.assert_close(value, expected[name])


if __name__ == "__main__":
    unittest.main()
