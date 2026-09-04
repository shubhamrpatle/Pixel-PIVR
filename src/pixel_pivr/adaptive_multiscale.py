"""Single-encode adaptive multi-scale feature re-entry for Pixel-PIVR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .lora import base_causal_lm
from .magnified_modeling import MagnifiedROIMetadataDataset
from .magnified_roi import (
    MoonViTFeatureCache,
    encode_global_cache,
    insert_virtual_image,
    normalized_grid_index,
    sliding_2x2,
)
from .modeling import chunked_cross_entropy, weighted_training_objective


@dataclass(frozen=True)
class ResampledROI:
    features: torch.Tensor
    source_patch_hw: tuple[int, int]
    source_window_yx: tuple[int, int]
    target_patch_hw: tuple[int, int]
    projector_grid_hw: tuple[int, int]
    requested_pixels: int
    effective_pixels_hw: tuple[int, int]


def extract_resampled_preprojector_roi(
    cache: MoonViTFeatureCache,
    point_normalized_xy: Sequence[float] | torch.Tensor,
    projector: Any,
    *,
    requested_pixels: int,
    target_patches: int = 27,
    patch_size: int = 14,
    stride: int = 1,
) -> ResampledROI:
    """Crop one cached MoonViT field, resize it, and apply LA's projector."""
    requested = int(requested_pixels)
    target = int(target_patches)
    patch = int(patch_size)
    if requested <= 0 or target < 2 or patch <= 0:
        raise ValueError("ROI pixels and patch size must be positive; target must be >=2")
    point = torch.as_tensor(point_normalized_xy, dtype=torch.float32).flatten()
    if point.numel() != 2:
        raise ValueError(f"Expected normalized (x,y), got {point.tolist()}")
    grid_h, grid_w = cache.patch_grid_hw
    source_side = max(2, int(round(requested / patch)))
    source_h, source_w = min(source_side, grid_h), min(source_side, grid_w)
    row = normalized_grid_index(float(point[1]), grid_h)
    col = normalized_grid_index(float(point[0]), grid_w)
    start_y = min(max(row - source_h // 2, 0), grid_h - source_h)
    start_x = min(max(col - source_w // 2, 0), grid_w - source_w)
    grid = cache.unmerged.view(grid_h, grid_w, cache.unmerged.shape[-1])
    source = grid[start_y : start_y + source_h, start_x : start_x + source_w]
    # Resizing cached features changes their relative spatial scale without claiming
    # to recover pixel frequencies that MoonViT did not encode.
    resized = F.interpolate(
        source.permute(2, 0, 1).unsqueeze(0).float(),
        size=(target, target),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).permute(1, 2, 0).to(dtype=source.dtype)
    groups, projector_grid = sliding_2x2(resized, stride=int(stride))
    projected = projector(groups)
    return ResampledROI(
        features=projected,
        source_patch_hw=(source_h, source_w),
        source_window_yx=(start_y, start_x),
        target_patch_hw=(target, target),
        projector_grid_hw=projector_grid,
        requested_pixels=requested,
        effective_pixels_hw=(source_h * patch, source_w * patch),
    )


class AdaptiveScaleFusion(nn.Module):
    """Content-conditioned fusion that keeps one fixed local-token grid."""

    def __init__(
        self,
        hidden_size: int,
        scale_count: int,
        gate_hidden: int = 128,
        preferred_scale: int = 1,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or scale_count < 2 or gate_hidden <= 0:
            raise ValueError("Invalid adaptive scale-fusion dimensions")
        if not 0 <= preferred_scale < scale_count:
            raise ValueError("preferred_scale is outside the scale list")
        self.hidden_size = int(hidden_size)
        self.scale_count = int(scale_count)
        self.gate_hidden = int(gate_hidden)
        self.preferred_scale = int(preferred_scale)
        self.gate_in = nn.Linear(self.hidden_size * self.scale_count, self.gate_hidden)
        self.gate_out = nn.Linear(self.gate_hidden, self.scale_count)
        nn.init.xavier_uniform_(self.gate_in.weight)
        nn.init.zeros_(self.gate_in.bias)
        nn.init.zeros_(self.gate_out.weight)
        bias = torch.full((self.scale_count,), -1.0)
        bias[self.preferred_scale] = 1.0
        with torch.no_grad():
            self.gate_out.bias.copy_(bias)

    def forward(self, scale_features: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if len(scale_features) != self.scale_count:
            raise ValueError(f"Expected {self.scale_count} scales, got {len(scale_features)}")
        shapes = {tuple(value.shape) for value in scale_features}
        if len(shapes) != 1:
            raise ValueError(f"Scale token shapes differ: {sorted(shapes)}")
        stacked = torch.stack(tuple(scale_features), dim=0)
        if int(stacked.shape[-1]) != self.hidden_size:
            raise ValueError("Projected scale width differs from fusion width")
        descriptors = stacked.float().mean(dim=1).reshape(1, -1)
        descriptors = F.layer_norm(descriptors, descriptors.shape[-1:])
        logits = self.gate_out(F.silu(self.gate_in(descriptors)))
        weights = torch.softmax(logits, dim=-1).to(dtype=stacked.dtype).squeeze(0)
        fused = (stacked * weights[:, None, None]).sum(dim=0).to(dtype=stacked.dtype)
        return fused, weights


class AdaptiveMultiScaleTrainingModel(nn.Module):
    """Frozen MoonViT/projector with Qwen LoRA and a learned scale gate."""

    def __init__(
        self,
        model: nn.Module,
        *,
        roi_pixels: Sequence[int],
        target_patches: int,
        roi_stride: int,
        fusion_hidden: int,
        preferred_scale: int,
        image_start_id: int,
        image_end_id: int,
        max_sequence: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.roi_pixels = tuple(int(value) for value in roi_pixels)
        self.target_patches = int(target_patches)
        self.roi_stride = int(roi_stride)
        self.image_start_id = int(image_start_id)
        self.image_end_id = int(image_end_id)
        self.max_sequence = int(max_sequence)
        hidden_size = int(base_causal_lm(model.language_model).config.hidden_size)
        self.fusion = AdaptiveScaleFusion(
            hidden_size,
            len(self.roi_pixels),
            gate_hidden=int(fusion_hidden),
            preferred_scale=int(preferred_scale),
        ).to(device=next(model.parameters()).device)
        model.pixel_pivr_controller = self.fusion

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.vision_model.eval()
        self.model.mlp1.eval()
        self.fusion.train(mode)
        return self

    def local_features(
        self,
        cache: MoonViTFeatureCache,
        point: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[ResampledROI]]:
        with torch.no_grad():
            rois = [
                extract_resampled_preprojector_roi(
                    cache,
                    point,
                    self.model.mlp1,
                    requested_pixels=value,
                    target_patches=self.target_patches,
                    patch_size=int(self.model.vision_model.patch_size),
                    stride=self.roi_stride,
                )
                for value in self.roi_pixels
            ]
        fused, weights = self.fusion([roi.features.detach() for roi in rois])
        return fused, weights, rois

    def forward(self, sample: Mapping[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        device = next(self.model.parameters()).device
        input_ids = sample["input_ids"].to(device)
        labels = sample["labels"].to(device)
        position_ids = sample["position_ids"].to(device)
        pixel_values = sample["pixel_values"].to(device=device, dtype=torch.bfloat16)
        image_grid_hws = torch.as_tensor(
            sample["image_grid_hws"], device=device, dtype=torch.int32
        )
        with torch.no_grad():
            cache = encode_global_cache(self.model, pixel_values, image_grid_hws)

        projected = cache.projected_global
        roi_tokens = 0
        scale_weights = torch.zeros(len(self.roi_pixels), device=device)
        enabled = bool(torch.as_tensor(sample["magnified_roi_enabled"]).item())
        if enabled:
            local, scale_weights, _rois = self.local_features(
                cache, sample["magnified_roi_point"]
            )
            roi_tokens = int(local.shape[0])
            input_ids, labels, position_ids = insert_virtual_image(
                input_ids,
                labels,
                position_ids,
                image_token_id=int(self.model.image_token_index),
                image_start_id=self.image_start_id,
                image_end_id=self.image_end_id,
                visual_tokens=roi_tokens,
            )
            projected = torch.cat((cache.projected_global, local), dim=0)
        if int(input_ids.numel()) > self.max_sequence:
            raise RuntimeError(
                f"Adaptive multi-scale sequence exceeds budget: {input_ids.numel()} > {self.max_sequence}"
            )
        batched_ids = input_ids.unsqueeze(0)
        selected = batched_ids.eq(int(self.model.image_token_index))
        if int(selected.sum()) != int(projected.shape[0]):
            raise RuntimeError("Adaptive multi-scale visual placeholder mismatch")
        embeddings = self.model.language_model.get_input_embeddings()(batched_ids)
        flat = embeddings.reshape(-1, embeddings.shape[-1]).clone()
        flat[selected.reshape(-1)] = projected
        embeddings = flat.view_as(embeddings)
        causal = base_causal_lm(self.model.language_model)
        outputs = causal.model(
            input_ids=None,
            inputs_embeds=embeddings,
            attention_mask=None,
            position_ids=position_ids.unsqueeze(0),
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        native = chunked_cross_entropy(
            outputs.last_hidden_state,
            labels.unsqueeze(0),
            causal.lm_head.weight,
        )
        objective, metrics = weighted_training_objective(native, sample)
        metrics.update({
            "global_visual_tokens": float(cache.projected_global.shape[0]),
            "local_visual_tokens": float(roi_tokens),
            "scale_active": float(enabled),
        })
        for index, value in enumerate(scale_weights.detach().float().tolist()):
            metrics[f"scale_weight_{index}"] = float(value)
        return objective, metrics


__all__ = [
    "AdaptiveMultiScaleTrainingModel",
    "AdaptiveScaleFusion",
    "MagnifiedROIMetadataDataset",
    "ResampledROI",
    "extract_resampled_preprojector_roi",
]
