from __future__ import annotations

import torch
import unittest

from pixel_pivr.decoder import image_patch_lengths
from pixel_pivr.shared_prefix import SharedPrefixCache


class DecoderContractTests(unittest.TestCase):
    def test_patch_lengths_support_hw_and_thw(self) -> None:
        self.assertEqual(image_patch_lengths(torch.tensor([[2, 3], [4, 5]])), [6, 20])
        self.assertEqual(
            image_patch_lengths(torch.tensor([[1, 2, 3], [2, 3, 4]])), [6, 24]
        )

    def test_patch_lengths_rejects_unknown_shape(self) -> None:
        with self.assertRaises(ValueError):
            image_patch_lengths(torch.ones(2, 4, dtype=torch.long))

    def test_shared_prefix_cache_keeps_one_persistent_prefix(self) -> None:
        prefix_keys = torch.randn(1, 2, 3, 4)
        prefix_values = torch.randn(1, 2, 3, 4)
        cache = SharedPrefixCache(((prefix_keys, prefix_values),))
        pointer = cache.layers[0].prefix_keys.data_ptr()
        suffix_keys = torch.randn(5, 2, 2, 4)
        suffix_values = torch.randn(5, 2, 2, 4)
        keys, values = cache.update(suffix_keys, suffix_values, 0)
        self.assertEqual(tuple(cache.layers[0].prefix_keys.shape), (1, 2, 3, 4))
        self.assertEqual(cache.layers[0].prefix_keys.data_ptr(), pointer)
        self.assertEqual(tuple(keys.shape), (5, 2, 5, 4))
        self.assertEqual(tuple(values.shape), (5, 2, 5, 4))
