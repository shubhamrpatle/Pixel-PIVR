"""Wave decoder for adaptive multi-scale cached-feature re-entry."""

from __future__ import annotations

from typing import Any

import torch

from .adaptive_multiscale import AdaptiveScaleFusion, extract_resampled_preprojector_roi
from .decoder import AddressedCrop
from .lora import base_causal_lm, restore_trainable_state
from .magnified_decoder import MagnifiedPreProjectorWaveDecoder
from .magnified_roi import MoonViTFeatureCache


class AdaptiveMultiScaleWaveDecoder(MagnifiedPreProjectorWaveDecoder):
    """Fuse several point-centred cached feature fields before PBD6 decoding."""

    def __init__(
        self,
        worker: Any,
        *,
        image_token_limit: int = 6000,
        prefix_cache_mode: str = "shared",
        roi_pixels: tuple[int, ...] = (196, 378, 756),
        target_patches: int = 27,
        roi_stride: int = 1,
        fusion_hidden: int = 128,
        preferred_scale: int = 1,
    ) -> None:
        if len(roi_pixels) < 2 or len(set(roi_pixels)) != len(roi_pixels):
            raise ValueError("Adaptive ROI scales must contain at least two unique values")
        super().__init__(
            worker,
            image_token_limit=image_token_limit,
            prefix_cache_mode=prefix_cache_mode,
            roi_pixels=int(roi_pixels[preferred_scale]),
            roi_stride=roi_stride,
        )
        self.roi_scales = tuple(int(value) for value in roi_pixels)
        self.target_patches = int(target_patches)
        self.fusion_hidden = int(fusion_hidden)
        self.preferred_scale = int(preferred_scale)
        hidden_size = int(base_causal_lm(self.model.language_model).config.hidden_size)
        self.fusion = AdaptiveScaleFusion(
            hidden_size,
            len(self.roi_scales),
            gate_hidden=self.fusion_hidden,
            preferred_scale=self.preferred_scale,
        ).to(device=self.device)
        self.model.pixel_pivr_controller = self.fusion
        restore_trainable_state(
            self.model,
            worker.checkpoint["trainable_state"],
            require_controller=True,
        )
        self.fusion.eval()
        self._scale_weight_sum = torch.zeros(
            len(self.roi_scales), dtype=torch.float64
        )
        self._scale_weight_count = 0

    def _local_features(self, cache: MoonViTFeatureCache) -> list[torch.Tensor]:
        values = []
        for branch in self._active_branches:
            if not isinstance(branch, AddressedCrop) or branch.point_normalized_xy is None:
                raise ValueError("Every adaptive address requires a normalized point")
            rois = [
                extract_resampled_preprojector_roi(
                    cache,
                    branch.point_normalized_xy,
                    self.model.mlp1,
                    requested_pixels=scale,
                    target_patches=self.target_patches,
                    patch_size=int(self.model.vision_model.patch_size),
                    stride=self.roi_stride,
                )
                for scale in self.roi_scales
            ]
            fused, weights = self.fusion([roi.features for roi in rois])
            self._scale_weight_sum += weights.detach().double().cpu()
            self._scale_weight_count += 1
            values.append(fused)
        return values

    @torch.no_grad()
    def decode_image(self, *args: Any, **kwargs: Any):
        self._scale_weight_sum.zero_()
        self._scale_weight_count = 0
        outputs, execution = super().decode_image(*args, **kwargs)
        mean_weights = (
            (self._scale_weight_sum / self._scale_weight_count).tolist()
            if self._scale_weight_count
            else [0.0] * len(self.roi_scales)
        )
        execution.update(
            {
                "visual_context": "adaptive_multiscale_preprojector_roi",
                "adaptive_roi_scales_pixels": list(self.roi_scales),
                "adaptive_target_patches": self.target_patches,
                "adaptive_local_tokens": (self.target_patches - 1) ** 2,
                "adaptive_mean_scale_weights": mean_weights,
                "adaptive_fusion_hidden": self.fusion_hidden,
            }
        )
        return outputs, execution


__all__ = ["AdaptiveMultiScaleWaveDecoder"]
