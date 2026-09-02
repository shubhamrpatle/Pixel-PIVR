"""LocateAnything Qwen-LoRA configuration and checkpoint state helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn


def base_causal_lm(language_model: nn.Module) -> nn.Module:
    base = (
        language_model.get_base_model()
        if hasattr(language_model, "get_base_model")
        else language_model
    )
    if not hasattr(base, "model") or not hasattr(base, "lm_head"):
        raise RuntimeError("Cannot resolve Qwen decoder and LM head")
    return base


def qwen_lora_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    values = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if "lora_" in name and "language_model" in name
    ]
    if not values:
        raise RuntimeError("No Qwen LoRA parameters were found")
    return values


def configure_lora_only(model: nn.Module, rank: int) -> dict[str, int]:
    if rank <= 0:
        raise ValueError("LoRA rank must be positive")
    model.requires_grad_(False)
    model.wrap_llm_lora(
        r=int(rank),
        lora_alpha=2 * int(rank),
        lora_dropout=0.05,
    )
    model.requires_grad_(False)
    values = qwen_lora_parameters(model)
    for _name, parameter in values:
        parameter.requires_grad_(True)
    return {
        "rank": int(rank),
        "tensors": len(values),
        "parameters": sum(parameter.numel() for _name, parameter in values),
    }


def trainable_state(model: nn.Module) -> dict[str, Any]:
    lora = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" in name
    }
    if not lora:
        raise RuntimeError("Refusing to save an empty LoRA state")
    return {"lora": lora, "controller": None}


def restore_trainable_state(model: nn.Module, payload: Mapping[str, Any]) -> None:
    live = dict(model.named_parameters())
    saved = payload.get("lora")
    if not isinstance(saved, Mapping) or not saved:
        raise RuntimeError("Checkpoint has no LoRA state")
    missing = sorted(set(saved) - set(live))
    if missing:
        raise RuntimeError(f"Adapter keys are absent from live model: {missing[:3]}")
    unexpected_live = sorted(
        name
        for name, parameter in live.items()
        if parameter.requires_grad and "lora_" in name and name not in saved
    )
    if unexpected_live:
        raise RuntimeError(
            f"Live model has LoRA keys absent from checkpoint: {unexpected_live[:3]}"
        )
    with torch.no_grad():
        for name, tensor in saved.items():
            parameter = live[name]
            if tuple(parameter.shape) != tuple(tensor.shape):
                raise RuntimeError(
                    f"Adapter shape mismatch for {name}: "
                    f"{tuple(parameter.shape)} != {tuple(tensor.shape)}"
                )
            parameter.copy_(tensor.to(parameter.device, parameter.dtype))


def load_adapter_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    configure: bool = True,
) -> dict[str, Any]:
    checkpoint = torch.load(
        Path(checkpoint_path), map_location="cpu", weights_only=False
    )
    config = checkpoint.get("config") or {}
    if configure:
        configure_lora_only(model, int(config["lora_rank"]))
    restore_trainable_state(model, checkpoint["trainable_state"])
    return checkpoint

