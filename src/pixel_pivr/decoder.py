"""Shared-prefix sequential/wave PBD6 decoding for pixel-reencoded PIVR."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from .lora import base_causal_lm
from .shared_prefix import SharedPrefixCache, legacy_key_values


@dataclass(frozen=True)
class AddressedCrop:
    """One predicted instance address and optional point-centered pixel crop."""

    address_id: str
    label: str
    crop: Any
    question: str
    point_normalized_xy: tuple[int, int] | None = None


def image_patch_lengths(image_grid_hws: Any) -> list[int]:
    grid = torch.as_tensor(image_grid_hws, dtype=torch.long)
    if grid.ndim != 2 or grid.shape[1] not in (2, 3):
        raise ValueError(f"Unexpected image_grid_hws shape: {tuple(grid.shape)}")
    return [int(row.prod().item()) for row in grid]


class PixelPIVRWaveDecoder:
    """Decode exactly one native PBD6 HBB for every known point address."""

    def __init__(
        self,
        worker: Any,
        *,
        image_token_limit: int = 1024,
        prefix_cache_mode: str = "shared",
    ) -> None:
        self.worker = worker
        self.model = worker.model
        self.processor = worker.processor
        self.tokenizer = worker.tokenizer
        self.device = worker.device
        self.dtype = worker.dtype
        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is None or not hasattr(image_processor, "in_token_limit"):
            raise RuntimeError("LocateAnything processor has no in_token_limit")
        image_processor.in_token_limit = int(image_token_limit)
        self.tokenizer.padding_side = "left"
        self.mask_id = int(self.model.token_ids["default_mask_token_id"])
        self.n_future_tokens = 6
        if prefix_cache_mode not in {"recompute", "shared"}:
            raise ValueError("prefix_cache_mode must be recompute or shared")
        self.prefix_cache_mode = prefix_cache_mode
        self.image_end_id = int(
            self.tokenizer.convert_tokens_to_ids(self.processor.image_end_token)
        )

    def _messages(
        self, global_image: Any, branch: AddressedCrop
    ) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": global_image},
                    {"type": "image", "image": branch.crop},
                    {"type": "text", "text": branch.question},
                ],
            }
        ]

    def _prepare_batch(
        self, global_image: Any, branches: Sequence[AddressedCrop]
    ) -> tuple[dict[str, Any], list[int]]:
        conversations = [self._messages(global_image, branch) for branch in branches]
        texts = [
            self.processor.py_apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            for messages in conversations
        ]
        images, videos = self.processor.process_vision_info(conversations)
        inputs = self.processor(
            text=texts,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
        )
        lengths = image_patch_lengths(inputs["image_grid_hws"])
        expected_images = 2 * len(branches)
        if len(lengths) != expected_images:
            raise RuntimeError(
                f"Expected {expected_images} global/local images, found {len(lengths)}"
            )
        return inputs, lengths

    def _project_cached_global_and_locals(
        self,
        inputs: Mapping[str, Any],
        patch_lengths: Sequence[int],
        cached_global: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pixels = inputs["pixel_values"]
        chunks = list(torch.split(pixels, list(patch_lengths), dim=0))
        grids = torch.as_tensor(inputs["image_grid_hws"], dtype=torch.int32)
        if len(chunks) % 2:
            raise RuntimeError("Global/local image list is not paired")

        local_chunks = chunks[1::2]
        local_grids = grids[1::2]
        if cached_global is None:
            encoded_pixels = torch.cat([chunks[0], *local_chunks], dim=0).to(
                device=self.device, dtype=self.dtype
            )
            encoded_grids = torch.cat([grids[0:1], local_grids], dim=0).to(
                device=self.device
            )
            features = self.model.extract_feature(encoded_pixels, encoded_grids)
            projected_global = self.model.mlp1(features[0])
            projected_locals = [self.model.mlp1(value) for value in features[1:]]
        else:
            encoded_pixels = torch.cat(local_chunks, dim=0).to(
                device=self.device, dtype=self.dtype
            )
            encoded_grids = local_grids.to(device=self.device)
            features = self.model.extract_feature(encoded_pixels, encoded_grids)
            projected_global = cached_global
            projected_locals = [self.model.mlp1(value) for value in features]

        visual_rows = []
        for local in projected_locals:
            visual_rows.extend((projected_global, local))
        return projected_global, torch.cat(visual_rows, dim=0)

    def _project_shared_prefix_and_locals(
        self,
        inputs: Mapping[str, Any],
        patch_lengths: Sequence[int],
        cached_global: torch.Tensor | None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        pixels = inputs["pixel_values"]
        chunks = list(torch.split(pixels, list(patch_lengths), dim=0))
        grids = torch.as_tensor(inputs["image_grid_hws"], dtype=torch.int32)
        if len(chunks) % 2:
            raise RuntimeError("Global/local image list is not paired")
        global_grids = grids[0::2]
        first_grid = global_grids[0]
        if any(not torch.equal(grid, first_grid) for grid in global_grids[1:]):
            raise RuntimeError("Repeated global images produced different grids")
        local_chunks = chunks[1::2]
        local_grids = grids[1::2]
        if cached_global is None:
            encoded_pixels = torch.cat([chunks[0], *local_chunks], dim=0).to(
                device=self.device, dtype=self.dtype
            )
            encoded_grids = torch.cat([grids[0:1], local_grids], dim=0).to(
                device=self.device
            )
            features = self.model.extract_feature(encoded_pixels, encoded_grids)
            projected_global = self.model.mlp1(features[0])
            projected_locals = [self.model.mlp1(value) for value in features[1:]]
        else:
            features = self.model.extract_feature(
                torch.cat(local_chunks, dim=0).to(
                    device=self.device, dtype=self.dtype
                ),
                local_grids.to(device=self.device),
            )
            projected_global = cached_global
            projected_locals = [self.model.mlp1(value) for value in features]
        return projected_global, projected_locals

    def _split_shared_prefix_rows(
        self,
        inputs: Mapping[str, Any],
        projected_global: torch.Tensor,
        projected_locals: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        if int(input_ids.shape[0]) != len(projected_locals):
            raise RuntimeError("Shared-prefix text and local-visual counts differ")
        image_index = int(self.model.image_token_index)
        global_tokens = int(projected_global.shape[0])
        common_prefix: torch.Tensor | None = None
        suffix_rows: list[torch.Tensor] = []
        for row_index, projected_local in enumerate(projected_locals):
            ids = input_ids[row_index][attention_mask[row_index].to(dtype=torch.bool)]
            selected = torch.where(ids.eq(image_index))[0]
            if int(selected.numel()) < global_tokens:
                raise RuntimeError("Global visual placeholder count is too small")
            global_positions = selected[:global_tokens]
            expected = torch.arange(
                int(global_positions[0]),
                int(global_positions[0]) + global_tokens,
                dtype=global_positions.dtype,
                device=global_positions.device,
            )
            if not torch.equal(global_positions, expected):
                raise RuntimeError("Global visual placeholders are not contiguous")
            last_global = int(global_positions[-1].item())
            if int(ids[last_global + 1]) != self.image_end_id:
                raise RuntimeError("Global image is not followed by </img>")
            split_at = last_global + 2
            candidate = ids[:split_at]
            if common_prefix is None:
                common_prefix = candidate
            elif not torch.equal(common_prefix, candidate):
                raise RuntimeError("Address rows do not share one global prefix")
            suffix = ids[split_at:]
            if int(suffix.eq(image_index).sum().item()) != int(
                projected_local.shape[0]
            ):
                raise RuntimeError("Local visual placeholder count differs")
            suffix_rows.append(suffix)
        if common_prefix is None:
            raise RuntimeError("Cannot split an empty address batch")
        return common_prefix, suffix_rows

    def _left_pad_rows(
        self, rows: Sequence[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not rows or any(int(row.numel()) == 0 for row in rows):
            raise RuntimeError("Every address must have a nonempty suffix")
        maximum = max(int(row.numel()) for row in rows)
        pad_id = int(
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )
        input_ids = rows[0].new_full((len(rows), maximum), pad_id)
        attention = rows[0].new_zeros((len(rows), maximum))
        for row_index, row in enumerate(rows):
            input_ids[row_index, -int(row.numel()) :] = row
            attention[row_index, -int(row.numel()) :] = 1
        return input_ids, attention

    def _prepare_geometry_inputs(
        self,
        branches: Sequence[AddressedCrop],
        *,
        input_dtype: torch.dtype,
        attention_dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reference_rows = [
            self.tokenizer.encode(f"<ref>{branch.label}</ref>", add_special_tokens=False)
            for branch in branches
        ]
        if any(not row for row in reference_rows):
            raise RuntimeError("A point-address reference tokenized to an empty sequence")
        maximum_reference = max(len(row) for row in reference_rows)
        pad_id = int(
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )
        reference_ids = torch.full(
            (len(branches), maximum_reference),
            pad_id,
            dtype=input_dtype,
            device=device,
        )
        reference_attention = torch.zeros(
            (len(branches), maximum_reference),
            dtype=attention_dtype,
            device=device,
        )
        for index, row in enumerate(reference_rows):
            reference_ids[index, -len(row) :] = torch.as_tensor(
                row, dtype=input_dtype, device=device
            )
            reference_attention[index, -len(row) :] = 1
        geometry_ids = torch.cat(
            (
                reference_ids,
                reference_ids[:, -1:].clone(),
                torch.full(
                    (len(branches), self.n_future_tokens - 1),
                    self.mask_id,
                    dtype=input_dtype,
                    device=device,
                ),
            ),
            dim=1,
        )
        geometry_attention = torch.cat(
            (
                reference_attention,
                torch.ones(
                    (len(branches), self.n_future_tokens),
                    dtype=attention_dtype,
                    device=device,
                ),
            ),
            dim=1,
        )
        return geometry_ids, geometry_attention

    def _results_from_logits(
        self,
        logits: torch.Tensor,
        branches: Sequence[AddressedCrop],
        *,
        allow_none: bool,
    ) -> list[dict[str, Any]]:
        box_start = int(self.model.token_ids["box_start_token_id"])
        box_end = int(self.model.token_ids["box_end_token_id"])
        none_id = int(self.model.token_ids["none_token_id"])
        coord_start = int(self.model.token_ids["coord_start_token_id"])
        coord_end = int(self.model.token_ids["coord_end_token_id"])
        results: list[dict[str, Any]] = []
        for index, branch in enumerate(branches):
            row = logits[index].float()
            if allow_none and row[1, none_id] >= row[
                1, coord_start : coord_end + 1
            ].max():
                tokens = torch.as_tensor(
                    [box_start, none_id, box_end],
                    dtype=torch.long,
                    device=logits.device,
                )
                block_type = "empty_box"
            else:
                coordinates = []
                for lane in range(1, 5):
                    lane_logits = row[lane, coord_start : coord_end + 1]
                    values, indices = torch.topk(lane_logits, k=5)
                    coordinate = int(
                        torch.round(
                            (indices.float() * values.softmax(dim=0)).sum()
                        ).item()
                    )
                    coordinates.append(max(0, min(1000, coordinate)))
                x1, x2 = sorted((coordinates[0], coordinates[2]))
                y1, y2 = sorted((coordinates[1], coordinates[3]))
                tokens = torch.as_tensor(
                    [
                        box_start,
                        coord_start + x1,
                        coord_start + y1,
                        coord_start + x2,
                        coord_start + y2,
                        box_end,
                    ],
                    dtype=torch.long,
                    device=logits.device,
                )
                block_type = "coord_box"
            block = self.tokenizer.decode(tokens, skip_special_tokens=False)
            results.append(
                {
                    "address_id": branch.address_id,
                    "label": branch.label,
                    "block_type": block_type,
                    "block_tokens": [int(value) for value in tokens.tolist()],
                    "answer": f"<ref>{branch.label}</ref>{block}",
                }
            )
        return results

    def _encode_shared_prefix(
        self, prefix_ids: torch.Tensor, projected_global: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        prefix_ids = prefix_ids.unsqueeze(0).to(self.device)
        attention = torch.ones_like(prefix_ids)
        if int(prefix_ids.eq(int(self.model.image_token_index)).sum()) != int(
            projected_global.shape[0]
        ):
            raise RuntimeError("Shared-prefix visual placeholder count differs")
        causal = base_causal_lm(self.model.language_model)
        outputs = causal.model(
            input_ids=prefix_ids,
            visual_features=projected_global,
            image_token_index=int(self.model.image_token_index),
            attention_mask=attention,
            position_ids=torch.arange(
                int(prefix_ids.shape[1]), device=self.device
            ).unsqueeze(0),
            past_key_values=None,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        return legacy_key_values(outputs.past_key_values)

    def _decode_shared_atomic_blocks(
        self,
        prefix_ids: torch.Tensor,
        suffix_rows: Sequence[torch.Tensor],
        projected_locals: Sequence[torch.Tensor],
        prefix_key_values: Sequence[tuple[torch.Tensor, torch.Tensor]],
        branches: Sequence[AddressedCrop],
        *,
        allow_none: bool,
    ) -> list[dict[str, Any]]:
        suffix_ids, suffix_attention = self._left_pad_rows(suffix_rows)
        suffix_ids = suffix_ids.to(self.device)
        suffix_attention = suffix_attention.to(self.device)
        batch_size = len(branches)
        prefix_attention = torch.ones(
            (batch_size, int(prefix_ids.numel())),
            dtype=suffix_attention.dtype,
            device=self.device,
        )
        context_attention = torch.cat((prefix_attention, suffix_attention), dim=1)
        context_positions = context_attention.long().cumsum(-1) - 1
        context_positions.masked_fill_(context_attention == 0, 1)
        projected = torch.cat(list(projected_locals), dim=0)
        if int(suffix_ids.eq(int(self.model.image_token_index)).sum()) != int(
            projected.shape[0]
        ):
            raise RuntimeError("Shared-prefix local placeholder count differs")
        causal = base_causal_lm(self.model.language_model)
        suffix_outputs = causal.model(
            input_ids=suffix_ids,
            visual_features=projected,
            image_token_index=int(self.model.image_token_index),
            attention_mask=context_attention,
            position_ids=context_positions[:, -int(suffix_ids.shape[1]) :],
            past_key_values=SharedPrefixCache(prefix_key_values),
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        geometry_ids, geometry_attention = self._prepare_geometry_inputs(
            branches,
            input_dtype=suffix_ids.dtype,
            attention_dtype=suffix_attention.dtype,
            device=suffix_ids.device,
        )
        full_attention = torch.cat((context_attention, geometry_attention), dim=1)
        positions = full_attention.long().cumsum(-1) - 1
        positions.masked_fill_(full_attention == 0, 1)
        positions[:, -self.n_future_tokens :] -= 1
        outputs = causal.model(
            input_ids=geometry_ids,
            visual_features=None,
            image_token_index=None,
            attention_mask=full_attention,
            position_ids=positions[:, -int(geometry_ids.shape[1]) :],
            past_key_values=suffix_outputs.past_key_values,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state[:, -self.n_future_tokens :, :]
        logits = torch.nn.functional.linear(hidden, causal.lm_head.weight)
        return self._results_from_logits(logits, branches, allow_none=allow_none)

    def _decode_atomic_blocks(
        self,
        inputs: Mapping[str, Any],
        projected: torch.Tensor,
        branches: Sequence[AddressedCrop],
        *,
        allow_none: bool,
    ) -> list[dict[str, Any]]:
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        batch_size = int(input_ids.shape[0])
        if batch_size != len(branches):
            raise RuntimeError("Text batch and branch batch differ")

        causal = base_causal_lm(self.model.language_model)
        prefix_position_ids = attention_mask.long().cumsum(-1) - 1
        prefix_position_ids.masked_fill_(attention_mask == 0, 1)
        prefix_outputs = causal.model(
            input_ids=input_ids,
            visual_features=projected,
            image_token_index=int(self.model.image_token_index),
            attention_mask=attention_mask,
            position_ids=prefix_position_ids,
            past_key_values=None,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        reference_rows = [
            self.tokenizer.encode(
                f"<ref>{branch.label}</ref>", add_special_tokens=False
            )
            for branch in branches
        ]
        if any(not row for row in reference_rows):
            raise RuntimeError("A point-address reference tokenized to an empty sequence")
        maximum_reference = max(len(row) for row in reference_rows)
        pad_id = int(
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )
        reference_ids = torch.full(
            (batch_size, maximum_reference),
            pad_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        reference_attention = torch.zeros(
            (batch_size, maximum_reference),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        for index, row in enumerate(reference_rows):
            reference_ids[index, -len(row) :] = torch.as_tensor(
                row, dtype=input_ids.dtype, device=input_ids.device
            )
            reference_attention[index, -len(row) :] = 1

        repeated_last = reference_ids[:, -1:].clone()
        masks = torch.full(
            (batch_size, self.n_future_tokens - 1),
            self.mask_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        geometry_ids = torch.cat((reference_ids, repeated_last, masks), dim=1)
        geometry_attention = torch.cat(
            (
                reference_attention,
                torch.ones(
                    (batch_size, self.n_future_tokens),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
            ),
            dim=1,
        )
        full_attention = torch.cat((attention_mask, geometry_attention), dim=1)
        full_position_ids = full_attention.long().cumsum(-1) - 1
        full_position_ids.masked_fill_(full_attention == 0, 1)
        full_position_ids[:, -self.n_future_tokens :] -= 1
        geometry_position_ids = full_position_ids[:, -geometry_ids.shape[1] :]

        outputs = causal.model(
            input_ids=geometry_ids,
            visual_features=None,
            image_token_index=None,
            attention_mask=full_attention,
            position_ids=geometry_position_ids,
            past_key_values=prefix_outputs.past_key_values,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state[:, -self.n_future_tokens :, :]
        logits = torch.nn.functional.linear(hidden, causal.lm_head.weight)

        box_start = int(self.model.token_ids["box_start_token_id"])
        box_end = int(self.model.token_ids["box_end_token_id"])
        none_id = int(self.model.token_ids["none_token_id"])
        coord_start = int(self.model.token_ids["coord_start_token_id"])
        coord_end = int(self.model.token_ids["coord_end_token_id"])
        results: list[dict[str, Any]] = []
        for index, branch in enumerate(branches):
            row = logits[index].float()
            coordinate_slice = row[1, coord_start : coord_end + 1]
            if allow_none and row[1, none_id] >= coordinate_slice.max():
                tokens = torch.as_tensor(
                    [box_start, none_id, box_end],
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
                block_type = "empty_box"
            else:
                coordinates = []
                for lane in range(1, 5):
                    lane_logits = row[lane, coord_start : coord_end + 1]
                    values, indices = torch.topk(lane_logits, k=5)
                    coordinate = int(
                        torch.round((indices.float() * values.softmax(dim=0)).sum())
                        .item()
                    )
                    coordinates.append(max(0, min(1000, coordinate)))
                x1, x2 = sorted((coordinates[0], coordinates[2]))
                y1, y2 = sorted((coordinates[1], coordinates[3]))
                tokens = torch.as_tensor(
                    [
                        box_start,
                        coord_start + x1,
                        coord_start + y1,
                        coord_start + x2,
                        coord_start + y2,
                        box_end,
                    ],
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
                block_type = "coord_box"
            block = self.tokenizer.decode(tokens, skip_special_tokens=False)
            results.append(
                {
                    "address_id": branch.address_id,
                    "label": branch.label,
                    "block_type": block_type,
                    "block_tokens": [int(value) for value in tokens.tolist()],
                    "answer": f"<ref>{branch.label}</ref>{block}",
                }
            )
        return results

    @torch.no_grad()
    def decode_image(
        self,
        global_image: Any,
        branches: Sequence[AddressedCrop],
        *,
        requested_wave_size: int,
        allow_none: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if requested_wave_size < 1 or requested_wave_size > 200:
            raise ValueError("requested_wave_size must be in [1, 200]")
        if not branches:
            return [], {
                "requested_wave_size": int(requested_wave_size),
                "effective_wave_sizes": [],
                "waves": 0,
                "addresses": 0,
                "model_seconds": 0.0,
                "end_to_end_seconds": 0.0,
                "peak_gpu_mb": 0.0,
                "prefix_cache_mode": self.prefix_cache_mode,
                "qwen_global_prefix_forwards": 0,
                "qwen_branch_suffix_forwards": 0,
                "qwen_geometry_forwards": 0,
            }

        torch.cuda.reset_peak_memory_stats()
        started_e2e = time.perf_counter()
        model_seconds = 0.0
        cursor = 0
        active_width = min(int(requested_wave_size), len(branches))
        cached_global: torch.Tensor | None = None
        shared_prefix_ids: torch.Tensor | None = None
        shared_prefix_key_values: tuple[
            tuple[torch.Tensor, torch.Tensor], ...
        ] | None = None
        outputs: list[dict[str, Any]] = []
        effective_sizes: list[int] = []

        while cursor < len(branches):
            width = min(active_width, len(branches) - cursor)
            chunk = branches[cursor : cursor + width]
            try:
                inputs, lengths = self._prepare_batch(global_image, chunk)
                torch.cuda.synchronize()
                model_started = time.perf_counter()
                if self.prefix_cache_mode == "shared":
                    cached_global, projected_locals = (
                        self._project_shared_prefix_and_locals(
                            inputs, lengths, cached_global
                        )
                    )
                    candidate_prefix, suffix_rows = self._split_shared_prefix_rows(
                        inputs, cached_global, projected_locals
                    )
                    if shared_prefix_ids is None:
                        shared_prefix_ids = candidate_prefix.clone()
                        shared_prefix_key_values = self._encode_shared_prefix(
                            shared_prefix_ids, cached_global
                        )
                    elif not torch.equal(shared_prefix_ids, candidate_prefix):
                        raise RuntimeError("Global Qwen prefix changed between waves")
                    if shared_prefix_key_values is None:
                        raise RuntimeError("Shared Qwen prefix was not initialized")
                    decoded = self._decode_shared_atomic_blocks(
                        shared_prefix_ids,
                        suffix_rows,
                        projected_locals,
                        shared_prefix_key_values,
                        chunk,
                        allow_none=allow_none,
                    )
                else:
                    cached_global, projected = self._project_cached_global_and_locals(
                        inputs, lengths, cached_global
                    )
                    decoded = self._decode_atomic_blocks(
                        inputs, projected, chunk, allow_none=allow_none
                    )
                torch.cuda.synchronize()
                model_seconds += time.perf_counter() - model_started
            except torch.cuda.OutOfMemoryError:
                if "inputs" in locals():
                    del inputs
                if "projected_locals" in locals():
                    del projected_locals
                if "projected" in locals():
                    del projected
                torch.cuda.empty_cache()
                if width == 1:
                    raise
                active_width = max(1, width // 2)
                continue
            outputs.extend(decoded)
            effective_sizes.append(width)
            cursor += width

        prefix_cache_bytes = 0
        if shared_prefix_key_values is not None:
            prefix_cache_bytes = sum(
                int(keys.numel() * keys.element_size())
                + int(values.numel() * values.element_size())
                for keys, values in shared_prefix_key_values
            )
        shared_mode = self.prefix_cache_mode == "shared"

        return outputs, {
            "requested_wave_size": int(requested_wave_size),
            "effective_wave_sizes": effective_sizes,
            "maximum_effective_wave_size": max(effective_sizes),
            "waves": len(effective_sizes),
            "addresses": len(branches),
            "model_seconds": float(model_seconds),
            "end_to_end_seconds": float(time.perf_counter() - started_e2e),
            "peak_gpu_mb": float(torch.cuda.max_memory_allocated() / 1024**2),
            "prefix_cache_mode": self.prefix_cache_mode,
            "fully_shared_qwen_prefix": shared_mode,
            "shared_prefix_sequence_tokens": (
                int(shared_prefix_ids.numel())
                if shared_prefix_ids is not None
                else 0
            ),
            "persistent_prefix_kv_copies": 1 if shared_mode else 0,
            "persistent_prefix_kv_mb": float(prefix_cache_bytes / 1024**2),
            "qwen_global_prefix_forwards": (
                1 if shared_mode else len(effective_sizes)
            ),
            "qwen_branch_suffix_forwards": (
                len(effective_sizes) if shared_mode else 0
            ),
            "qwen_geometry_forwards": len(effective_sizes),
            "qwen_total_forwards": (
                1 + 2 * len(effective_sizes)
                if shared_mode
                else 2 * len(effective_sizes)
            ),
            "global_moonvit_encodes": 1,
            "local_moonvit_encodes": len(effective_sizes),
            "native_geometry_block_tokens": 6,
            "geometry_decoding": "point_addressed_constrained_pbd6",
            "allow_none_for_predicted_points": bool(allow_none),
        }
