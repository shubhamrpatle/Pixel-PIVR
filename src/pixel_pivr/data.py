"""LocateAnything/Eagle dataset integration with explicit path boundaries."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

from torch.utils.data import DataLoader, Subset


def activate_eagle(eagle_root: Path) -> None:
    eagle_root = eagle_root.resolve()
    expected = eagle_root / "eaglevl/train/locany_finetune_magi_stream.py"
    if not expected.is_file():
        raise FileNotFoundError(
            f"--eagle-root must point to Eagle/Embodied; missing {expected}"
        )
    if str(eagle_root) not in sys.path:
        sys.path.insert(0, str(eagle_root))


def make_dataset(
    paths: Sequence[Path],
    processor: Any,
    *,
    data_root: Path,
    eagle_root: Path,
    block_size: int = 6,
) -> Any:
    activate_eagle(eagle_root)
    from eaglevl.train.locany_finetune_magi_stream import (  # type: ignore
        LazySupervisedDatasetMTP,
    )

    annotations = [str(path.resolve()) for path in paths]
    return LazySupervisedDatasetMTP(
        "pixel_pivr",
        {
            "annotation": annotations,
            "root": str(data_root.resolve()),
            "repeat_time": 1.0,
            "data_augment": False,
        },
        processor,
        block_size=int(block_size),
    )


def make_loader(dataset: Any, indices: list[int], workers: int) -> DataLoader:
    kwargs: dict[str, Any] = {
        "dataset": Subset(dataset, indices),
        "batch_size": None,
        "shuffle": False,
        "num_workers": int(workers),
        "pin_memory": True,
    }
    if workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**kwargs)

