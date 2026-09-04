from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from pixel_pivr.train import (
    PARAMETER_SYNC_POLICY,
    synchronize_trainable_parameters,
)


def _sync_worker(rank: int, world_size: int, rendezvous: str, output: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
    )
    try:
        layer = nn.Linear(4, 3, bias=True)
        with torch.no_grad():
            layer.weight.fill_(float(rank + 1))
            layer.bias.fill_(float(10 + rank))
        layer.bias.requires_grad_(False)

        audit = synchronize_trainable_parameters(layer, world_size)
        Path(f"{output}.{rank}.json").write_text(
            json.dumps(
                {
                    "weight": layer.weight.detach().tolist(),
                    "bias": layer.bias.detach().tolist(),
                    "audit": audit,
                }
            ),
            encoding="utf-8",
        )
    finally:
        dist.destroy_process_group()


class DistributedParameterSyncTests(unittest.TestCase):
    def test_trainable_parameters_are_broadcast_from_rank_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            rendezvous = tmp_path / "distributed-init"
            output = tmp_path / "replica"
            mp.spawn(
                _sync_worker,
                args=(2, str(rendezvous), str(output)),
                nprocs=2,
                join=True,
            )

            rank0 = json.loads(
                (tmp_path / "replica.0.json").read_text(encoding="utf-8")
            )
            rank1 = json.loads(
                (tmp_path / "replica.1.json").read_text(encoding="utf-8")
            )
            self.assertEqual(rank0["weight"], rank1["weight"])
            self.assertEqual(rank0["weight"], [[1.0] * 4] * 3)
            self.assertEqual(rank0["bias"], [10.0] * 3)
            self.assertEqual(rank1["bias"], [11.0] * 3)
            self.assertEqual(
                rank0["audit"],
                rank1["audit"],
            )
            self.assertEqual(
                rank0["audit"],
                {
                    "policy": PARAMETER_SYNC_POLICY,
                    "tensors": 1,
                    "parameters": 12,
                    "source_rank": 0,
                    "world_size": 2,
                },
            )


if __name__ == "__main__":
    unittest.main()
