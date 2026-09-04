from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from pixel_pivr.lora import load_adapter_checkpoint


class AdapterSafetyTests(unittest.TestCase):
    def test_unsynchronized_distributed_adapter_is_rejected_before_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "adapter.pt"
            torch.save(
                {
                    "config": {"world_size": 2, "lora_rank": 16},
                    "trainable_state": {},
                },
                checkpoint,
            )
            with self.assertRaisesRegex(RuntimeError, "not synchronized"):
                load_adapter_checkpoint(nn.Linear(1, 1), checkpoint)


if __name__ == "__main__":
    unittest.main()
