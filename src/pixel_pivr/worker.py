"""Minimal LocateAnything loader for Pixel-PIVR point and box inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .attention import (
    locateanything_attention_dispatch,
    verify_locateanything_attention,
)
from .lora import load_adapter_checkpoint


def choose_vision_attention() -> str:
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return "sdpa"
    return "flash_attention_2"


class LocateAnythingPixelPIVRWorker:
    def __init__(
        self,
        model_path: str | Path,
        adapter_path: str | Path,
        *,
        device: str = "cuda",
        dtype: str = "bfloat16",
        local_files_only: bool = True,
        allow_unsynchronized_adapter: bool = False,
    ) -> None:
        from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        self.device = device
        self.dtype = getattr(torch, dtype)
        common = {
            "trust_remote_code": True,
            "local_files_only": bool(local_files_only),
        }
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, **common)
        self.processor = AutoProcessor.from_pretrained(
            model_path, use_fast=False, **common
        )
        config = AutoConfig.from_pretrained(model_path, **common)
        config._attn_implementation = "sdpa"
        config._attn_implementation_autoset = False
        config.text_config._attn_implementation = "sdpa"
        config.text_config._attn_implementation_autoset = False
        vision_attention = choose_vision_attention()
        config.vision_config._attn_implementation = vision_attention
        config.vision_config._attn_implementation_autoset = False
        attention_dispatch = locateanything_attention_dispatch(vision_attention)
        self.model = AutoModel.from_pretrained(
            model_path,
            config=config,
            dtype=self.dtype,
            trust_remote_code=True,
            local_files_only=bool(local_files_only),
            low_cpu_mem_usage=True,
            attn_implementation=attention_dispatch,
        ).to(device)
        verify_locateanything_attention(self.model, vision_attention)
        self.checkpoint = load_adapter_checkpoint(
            self.model,
            adapter_path,
            allow_unsynchronized_distributed=allow_unsynchronized_adapter,
        )
        self.model.eval()

    @torch.no_grad()
    def predict_points(
        self,
        image: Any,
        question: str,
        *,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = self.processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=images, videos=videos, return_tensors="pt"
        ).to(self.device)
        response = self.model.generate(
            pixel_values=inputs["pixel_values"].to(self.dtype),
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_grid_hws=inputs.get("image_grid_hws"),
            tokenizer=self.tokenizer,
            max_new_tokens=int(max_new_tokens),
            use_cache=True,
            generation_mode="hybrid",
            decode_mode="baseline",
            temperature=float(temperature),
            do_sample=True,
            top_p=float(top_p),
            repetition_penalty=1.1,
            verbose=False,
            return_debug_info=False,
        )
        return str(response[0] if isinstance(response, tuple) else response)
