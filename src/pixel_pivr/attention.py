"""Attention dispatch shared by Pixel-PIVR training and inference."""

from __future__ import annotations


def locateanything_attention_dispatch(vision_attention: str) -> dict[str, str]:
    """Keep Qwen on SDPA while selecting MoonViT's tested implementation.

    Transformers recursively applies a scalar ``attn_implementation`` to every
    sub-config. LocateAnything needs a dictionary here; otherwise a final
    ``"sdpa"`` argument silently overrides the earlier MoonViT setting.
    """
    if vision_attention not in {"sdpa", "flash_attention_2"}:
        raise ValueError(f"Unsupported MoonViT attention: {vision_attention!r}")
    return {
        "": "sdpa",
        "text_config": "sdpa",
        "vision_config": vision_attention,
    }


def verify_locateanything_attention(model: object, vision_attention: str) -> None:
    """Fail if Transformers did not apply the requested nested dispatch."""
    text_config = getattr(getattr(model, "language_model", None), "config", None)
    vision_config = getattr(getattr(model, "vision_model", None), "config", None)
    actual_text = getattr(text_config, "_attn_implementation", None)
    actual_vision = getattr(vision_config, "_attn_implementation", None)
    if actual_text != "sdpa" or actual_vision != vision_attention:
        raise RuntimeError(
            "LocateAnything attention dispatch mismatch: "
            f"text={actual_text!r}, vision={actual_vision!r}, "
            f"expected text='sdpa', vision={vision_attention!r}"
        )
