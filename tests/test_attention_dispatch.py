from types import SimpleNamespace

import pytest

from pixel_pivr.attention import (
    locateanything_attention_dispatch,
    verify_locateanything_attention,
)


def fake_model(text: str = "sdpa", vision: str = "flash_attention_2"):
    return SimpleNamespace(
        language_model=SimpleNamespace(
            config=SimpleNamespace(_attn_implementation=text)
        ),
        vision_model=SimpleNamespace(
            config=SimpleNamespace(_attn_implementation=vision)
        ),
    )


def test_attention_dispatch_is_submodel_specific():
    assert locateanything_attention_dispatch("flash_attention_2") == {
        "": "sdpa",
        "text_config": "sdpa",
        "vision_config": "flash_attention_2",
    }


def test_attention_runtime_guard_accepts_expected_values():
    verify_locateanything_attention(fake_model(), "flash_attention_2")


def test_attention_runtime_guard_rejects_recursive_sdpa_override():
    with pytest.raises(RuntimeError, match="dispatch mismatch"):
        verify_locateanything_attention(fake_model(vision="sdpa"), "flash_attention_2")
