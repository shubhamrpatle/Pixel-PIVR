"""Sequential/wave decoding with one cached pre-merge MoonViT feature grid."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from .decoder import AddressedCrop, PixelPIVRWaveDecoder, image_patch_lengths
from .magnified_roi import (
    MoonViTFeatureCache,
    encode_global_cache,
    extract_magnified_preprojector_roi,
    insert_virtual_image,
)


class MagnifiedPreProjectorWaveDecoder(PixelPIVRWaveDecoder):
    """Use dense local projector windows without re-encoding local pixels."""

    def __init__(
        self,
        worker: Any,
        *,
        image_token_limit: int = 6000,
        prefix_cache_mode: str = "shared",
        roi_pixels: int = 380,
        roi_stride: int = 1,
    ) -> None:
        super().__init__(
            worker,
            image_token_limit=image_token_limit,
            prefix_cache_mode=prefix_cache_mode,
        )
        self.roi_pixels = int(roi_pixels)
        self.roi_stride = int(roi_stride)
        if self.roi_pixels <= 0 or self.roi_stride not in (1, 2):
            raise ValueError("roi_pixels must be positive and roi_stride must be 1 or 2")
        self.image_start_id = int(
            self.tokenizer.convert_tokens_to_ids(self.processor.image_start_token)
        )
        self._feature_cache: MoonViTFeatureCache | None = None
        self._active_branches: Sequence[AddressedCrop] = ()

    def _messages(
        self, global_image: Any, branch: AddressedCrop
    ) -> list[dict[str, Any]]:
        if branch.point_normalized_xy is None:
            raise ValueError(f"Address {branch.address_id} has no normalized point")
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": global_image},
                    {"type": "text", "text": branch.question},
                ],
            }
        ]

    def _prepare_batch(
        self, global_image: Any, branches: Sequence[AddressedCrop]
    ) -> tuple[dict[str, Any], list[int]]:
        if not branches:
            raise ValueError("Cannot prepare an empty address wave")
        self._active_branches = branches
        conversations = [self._messages(global_image, branch) for branch in branches]
        texts = [
            self.processor.py_apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            for messages in conversations
        ]
        # Every branch shares one global image. Expand its placeholders ourselves so
        # the processor does not resize and patchify the same pixels N times.
        images, videos = self.processor.process_vision_info(conversations[0])
        if images is None or len(images) != 1 or videos is not None:
            raise RuntimeError("Magnified decoding expects exactly one global image")
        image_inputs = self.processor.image_processor(
            images=images, return_tensors="pt"
        )
        lengths = image_patch_lengths(image_inputs["image_grid_hws"])
        if len(lengths) != 1:
            raise RuntimeError(
                f"Expected one global image grid, found {len(lengths)}"
            )
        merge_kernel = self.processor.image_processor.merge_kernel_size
        merged_tokens = lengths[0] // int(merge_kernel[0] * merge_kernel[1])
        placeholder = (
            f"<image 1>{self.processor.image_start_token}"
            f"{self.processor.image_token * merged_tokens}"
            f"{self.processor.image_end_token}"
        )
        expanded_texts = []
        for text in texts:
            if text.count("<image-1>") != 1:
                raise RuntimeError("Each address prompt must contain one global image")
            expanded_texts.append(text.replace("<image-1>", placeholder, 1))
        text_inputs = self.tokenizer(
            expanded_texts,
            return_tensors="pt",
            padding=True,
        )
        inputs = {
            **text_inputs,
            "pixel_values": image_inputs["pixel_values"],
            "image_grid_hws": image_inputs["image_grid_hws"],
        }
        return inputs, lengths

    def _ensure_feature_cache(
        self,
        inputs: Mapping[str, Any],
        patch_lengths: Sequence[int],
    ) -> MoonViTFeatureCache:
        grids = torch.as_tensor(inputs["image_grid_hws"], dtype=torch.int32)
        if int(grids.shape[0]) != 1 or len(patch_lengths) != 1:
            raise RuntimeError("Expected one preprocessed global image per wave")
        first_grid = grids[0]
        if self._feature_cache is None:
            self._feature_cache = encode_global_cache(
                self.model,
                inputs["pixel_values"].to(device=self.device, dtype=self.dtype),
                first_grid.unsqueeze(0).to(device=self.device),
            )
        elif self._feature_cache.patch_grid_hw != tuple(
            int(value) for value in first_grid.tolist()
        ):
            raise RuntimeError("Global MoonViT grid changed between decoding waves")
        return self._feature_cache

    def _local_features(
        self, cache: MoonViTFeatureCache
    ) -> list[torch.Tensor]:
        values = []
        for branch in self._active_branches:
            if branch.point_normalized_xy is None:
                raise ValueError(f"Address {branch.address_id} has no normalized point")
            roi = extract_magnified_preprojector_roi(
                cache,
                branch.point_normalized_xy,
                self.model.mlp1,
                requested_pixels=self.roi_pixels,
                patch_size=int(self.model.vision_model.patch_size),
                stride=self.roi_stride,
            )
            values.append(roi.features)
        return values

    def _insert_local_placeholders(
        self,
        inputs: Mapping[str, Any],
        local_features: Sequence[torch.Tensor],
    ) -> None:
        ids = inputs["input_ids"]
        attention = inputs["attention_mask"]
        rows = []
        for row_index, local in enumerate(local_features):
            row = ids[row_index][attention[row_index].to(dtype=torch.bool)]
            labels = row.new_full(row.shape, -100)
            positions = torch.arange(row.numel(), dtype=row.dtype, device=row.device)
            inserted, _labels, _positions = insert_virtual_image(
                row,
                labels,
                positions,
                image_token_id=int(self.model.image_token_index),
                image_start_id=self.image_start_id,
                image_end_id=self.image_end_id,
                visual_tokens=int(local.shape[0]),
            )
            rows.append(inserted)
        padded_ids, padded_attention = self._left_pad_rows(rows)
        inputs["input_ids"] = padded_ids
        inputs["attention_mask"] = padded_attention

    def _project_shared_prefix_and_locals(
        self,
        inputs: Mapping[str, Any],
        patch_lengths: Sequence[int],
        cached_global: torch.Tensor | None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        cache = self._ensure_feature_cache(inputs, patch_lengths)
        locals_ = self._local_features(cache)
        self._insert_local_placeholders(inputs, locals_)
        return cache.projected_global, locals_

    def _project_cached_global_and_locals(
        self,
        inputs: Mapping[str, Any],
        patch_lengths: Sequence[int],
        cached_global: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache = self._ensure_feature_cache(inputs, patch_lengths)
        locals_ = self._local_features(cache)
        self._insert_local_placeholders(inputs, locals_)
        rows = [
            value
            for local in locals_
            for value in (cache.projected_global, local)
        ]
        return cache.projected_global, torch.cat(rows, dim=0)

    @torch.no_grad()
    def decode_image(
        self,
        global_image: Any,
        branches: Sequence[AddressedCrop],
        *,
        requested_wave_size: int,
        allow_none: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        self._feature_cache = None
        self._active_branches = ()
        try:
            outputs, execution = super().decode_image(
                global_image,
                branches,
                requested_wave_size=requested_wave_size,
                allow_none=allow_none,
            )
            execution.update(
                {
                    "visual_context": "preprojector_magnified_roi",
                    "local_moonvit_encodes": 0,
                    "moonvit_unmerged_cache_count": 1 if branches else 0,
                    "magnified_roi_requested_pixels": self.roi_pixels,
                    "magnified_roi_effective_pixels": int(
                        round(self.roi_pixels / int(self.model.vision_model.patch_size))
                    )
                    * int(self.model.vision_model.patch_size),
                    "magnified_roi_stride": self.roi_stride,
                }
            )
            return outputs, execution
        finally:
            self._feature_cache = None
            self._active_branches = ()
