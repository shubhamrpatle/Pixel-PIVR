"""Minimal training wrapper for the tested Pixel-PIVR Qwen-LoRA objective."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from .lora import base_causal_lm


IGNORE_INDEX = -100


def chunked_cross_entropy(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    chunk: int = 128,
) -> torch.Tensor:
    shifted_hidden = hidden_states[:, :-1].reshape(-1, hidden_states.shape[-1])
    shifted_labels = labels[:, 1:].reshape(-1)
    valid = shifted_labels.ne(IGNORE_INDEX)
    shifted_hidden = shifted_hidden[valid]
    shifted_labels = shifted_labels[valid]
    if shifted_labels.numel() == 0:
        raise RuntimeError("Pixel-PIVR sample has no supervised tokens")
    total = hidden_states.new_zeros((), dtype=torch.float32)
    for start in range(0, int(shifted_labels.numel()), int(chunk)):
        stop = min(int(shifted_labels.numel()), start + int(chunk))
        logits = F.linear(shifted_hidden[start:stop], weight).float()
        total = total + F.cross_entropy(
            logits, shifted_labels[start:stop], reduction="sum"
        )
    return total / shifted_labels.numel()


class PixelPIVRTrainingModel(nn.Module):
    """Frozen vision/projector plus trainable Qwen LoRA and native MTP loss."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.vision_model.eval()
        self.model.mlp1.eval()
        return self

    def forward(
        self, sample: Mapping[str, Any]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        device = next(self.model.parameters()).device
        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        labels = sample["labels"].unsqueeze(0).to(device)
        position_ids = sample["position_ids"].unsqueeze(0).to(device)
        pixel_values = sample["pixel_values"].to(
            device=device, dtype=torch.bfloat16
        )
        image_grid_hws = torch.as_tensor(
            sample["image_grid_hws"], device=device, dtype=torch.int32
        )

        with torch.no_grad():
            visual = self.model.extract_feature(pixel_values, image_grid_hws)
            projected = self.model.mlp1(torch.cat(visual, dim=0))

        selected = input_ids.eq(int(self.model.image_token_index))
        if int(selected.sum().item()) != int(projected.shape[0]):
            raise RuntimeError(
                "Visual placeholder mismatch: "
                f"ids={int(selected.sum())}, projected={int(projected.shape[0])}"
            )

        embeddings = self.model.language_model.get_input_embeddings()(input_ids)
        flat = embeddings.reshape(-1, embeddings.shape[-1]).clone()
        flat[selected.reshape(-1)] = projected
        embeddings = flat.view_as(embeddings)
        causal = base_causal_lm(self.model.language_model)
        outputs = causal.model(
            input_ids=None,
            inputs_embeds=embeddings,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        native = chunked_cross_entropy(
            outputs.last_hidden_state, labels, causal.lm_head.weight
        )
        return native, {
            "loss": float(native.detach().float()),
            "native_loss": float(native.detach().float()),
        }

