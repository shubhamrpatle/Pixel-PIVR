"""Single-copy decoder-prefix cache used by Pixel-PIVR wave inference."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from transformers.cache_utils import Cache, CacheLayerMixin


class SharedPrefixCacheLayer(CacheLayerMixin):
    """Store one immutable prefix and a mutable branch suffix for one layer."""

    is_sliding = False

    def __init__(self, prefix_keys: torch.Tensor, prefix_values: torch.Tensor) -> None:
        super().__init__()
        if prefix_keys.shape != prefix_values.shape or prefix_keys.ndim != 4:
            raise ValueError("Shared-prefix K/V tensors must be matching rank four")
        if int(prefix_keys.shape[0]) != 1:
            raise ValueError("Persistent shared-prefix K/V must have batch size one")
        self.prefix_keys = prefix_keys
        self.prefix_values = prefix_values
        self.suffix_keys: torch.Tensor | None = None
        self.suffix_values: torch.Tensor | None = None
        self.keys = prefix_keys
        self.values = prefix_values
        self.dtype = prefix_keys.dtype
        self.device = prefix_keys.device
        self.is_initialized = True

    def lazy_initialization(self, key_states: torch.Tensor) -> None:
        if key_states.ndim != 4:
            raise ValueError("Shared-prefix suffix K/V tensors must be rank four")

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Mapping[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del cache_kwargs
        if key_states.shape != value_states.shape or key_states.ndim != 4:
            raise ValueError("Shared-prefix suffix K/V tensors must match")
        if self.suffix_keys is None:
            self.suffix_keys = key_states
            self.suffix_values = value_states
        else:
            if int(key_states.shape[0]) != int(self.suffix_keys.shape[0]):
                raise ValueError("Shared-prefix branch batch changed within one wave")
            self.suffix_keys = torch.cat((self.suffix_keys, key_states), dim=-2)
            self.suffix_values = torch.cat((self.suffix_values, value_states), dim=-2)
        batch_size = int(key_states.shape[0])
        return (
            torch.cat(
                (self.prefix_keys.expand(batch_size, -1, -1, -1), self.suffix_keys),
                dim=-2,
            ),
            torch.cat(
                (
                    self.prefix_values.expand(batch_size, -1, -1, -1),
                    self.suffix_values,
                ),
                dim=-2,
            ),
        )

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        return self.get_seq_length() + int(cache_position.shape[0]), 0

    def get_seq_length(self) -> int:
        suffix = 0 if self.suffix_keys is None else int(self.suffix_keys.shape[-2])
        return int(self.prefix_keys.shape[-2]) + suffix

    def get_max_cache_shape(self) -> int:
        return -1

    @property
    def max_batch_size(self) -> int:
        return 1 if self.suffix_keys is None else int(self.suffix_keys.shape[0])

    @property
    def max_cache_len(self) -> int:
        return self.get_seq_length()


class SharedPrefixCache(Cache):
    """Transformers cache with one persistent global prefix."""

    def __init__(
        self, prefix_key_values: Sequence[tuple[torch.Tensor, torch.Tensor]]
    ) -> None:
        if not prefix_key_values:
            raise ValueError("Cannot create an empty shared-prefix cache")
        super().__init__(
            layers=[
                SharedPrefixCacheLayer(keys, values)
                for keys, values in prefix_key_values
            ]
        )


def legacy_key_values(
    cache: Any,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Normalize a batch-one Qwen cache to its layer-wise legacy tensors."""
    if cache is None:
        raise RuntimeError("Qwen did not return a prefix KV cache")
    if hasattr(cache, "to_legacy_cache"):
        cache = cache.to_legacy_cache()
    values = tuple((keys, items) for keys, items in cache)
    if not values:
        raise RuntimeError("Qwen returned an empty prefix KV cache")
    if any(int(keys.shape[0]) != 1 for keys, _items in values):
        raise RuntimeError("Shared Qwen prefix cache must have batch size one")
    return values
