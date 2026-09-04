"""Training adapter for single-encode magnified pre-projector PIVR."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn
from torch.utils.data import Dataset

from .lora import base_causal_lm
from .magnified_roi import (
    encode_global_cache,
    extract_magnified_preprojector_roi,
    insert_virtual_image,
)
from .modeling import chunked_cross_entropy, weighted_training_objective


REENTRY_ROUTES = {
    "point_indexed_visual_reentry",
    "cache_aware_point_reentry",
}


def _raw_row(base: Any, index: int) -> Mapping[str, Any]:
    if not hasattr(base, "lazy_loader") or not hasattr(base, "active_indices"):
        raise TypeError("Expected a LazySupervisedDatasetMTP-compatible dataset")
    return base.lazy_loader[base.active_indices[index]]


class MagnifiedROIMetadataDataset(Dataset):
    """Attach a global point to Round-2 rows without changing their JSONL."""

    def __init__(self, base: Dataset) -> None:
        self.base = base
        if not hasattr(base, "lazy_loader") or not hasattr(base, "active_indices"):
            raise TypeError("Expected a LazySupervisedDatasetMTP-compatible dataset")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.base[index])
        row = _raw_row(self.base, index)
        meta = dict(row.get("meta") or {})
        route = str(meta.get("pivr_route") or "")
        phase = str(meta.get("pivr_final_phase") or "")
        enabled = route in REENTRY_ROUTES or phase == "branch"
        point = meta.get("pivr_global_point")
        if point is None:
            point = meta.get("pivr_branch_point")
        if enabled and (
            not isinstance(point, (list, tuple))
            or len(point) != 2
            or any(not 0 <= float(value) <= 1000 for value in point)
        ):
            raise RuntimeError(
                f"Magnified Round-2 row {index} has no valid normalized global point: {point}"
            )
        images = row.get("image")
        if isinstance(images, list) or not isinstance(images, str):
            raise RuntimeError(
                "Magnified pre-projector PIVR requires exactly one real global image "
                f"per record; index={index}"
            )
        sample["magnified_roi_enabled"] = torch.tensor(enabled, dtype=torch.bool)
        sample["magnified_roi_point"] = torch.tensor(
            point if enabled else (0, 0), dtype=torch.float32
        )
        return sample


class MagnifiedPreProjectorTrainingModel(nn.Module):
    """Frozen one-pass MoonViT/projector with trainable Qwen LoRA."""

    def __init__(
        self,
        model: nn.Module,
        *,
        roi_pixels: int,
        roi_stride: int,
        image_start_id: int,
        image_end_id: int,
        max_sequence: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.roi_pixels = int(roi_pixels)
        self.roi_stride = int(roi_stride)
        self.image_start_id = int(image_start_id)
        self.image_end_id = int(image_end_id)
        self.max_sequence = int(max_sequence)

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.vision_model.eval()
        self.model.mlp1.eval()
        return self

    def forward(
        self, sample: Mapping[str, Any]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        device = next(self.model.parameters()).device
        input_ids = sample["input_ids"].to(device)
        labels = sample["labels"].to(device)
        position_ids = sample["position_ids"].to(device)
        pixel_values = sample["pixel_values"].to(
            device=device, dtype=torch.bfloat16
        )
        image_grid_hws = torch.as_tensor(
            sample["image_grid_hws"], device=device, dtype=torch.int32
        )
        with torch.no_grad():
            cache = encode_global_cache(self.model, pixel_values, image_grid_hws)

        enabled = bool(torch.as_tensor(sample["magnified_roi_enabled"]).item())
        projected = cache.projected_global
        roi_tokens = 0
        effective_h = 0
        effective_w = 0
        if enabled:
            with torch.no_grad():
                roi = extract_magnified_preprojector_roi(
                    cache,
                    sample["magnified_roi_point"],
                    self.model.mlp1,
                    requested_pixels=self.roi_pixels,
                    patch_size=int(self.model.vision_model.patch_size),
                    stride=self.roi_stride,
                )
            roi_tokens = int(roi.features.shape[0])
            effective_h, effective_w = roi.effective_pixels_hw
            input_ids, labels, position_ids = insert_virtual_image(
                input_ids,
                labels,
                position_ids,
                image_token_id=int(self.model.image_token_index),
                image_start_id=self.image_start_id,
                image_end_id=self.image_end_id,
                visual_tokens=roi_tokens,
            )
            projected = torch.cat((cache.projected_global, roi.features), dim=0)
        if int(input_ids.numel()) > self.max_sequence:
            raise RuntimeError(
                "Magnified PIVR sequence exceeds budget: "
                f"{input_ids.numel()} > {self.max_sequence}"
            )

        batched_ids = input_ids.unsqueeze(0)
        selected = batched_ids.eq(int(self.model.image_token_index))
        if int(selected.sum()) != int(projected.shape[0]):
            raise RuntimeError(
                "Magnified visual placeholder mismatch: "
                f"ids={int(selected.sum())}, features={int(projected.shape[0])}"
            )
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
            "local_effective_pixels_h": float(effective_h),
            "local_effective_pixels_w": float(effective_w),
            "moonvit_encodes": 1.0,
        })
        return objective, metrics
