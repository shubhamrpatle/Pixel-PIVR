"""LocateAnything/Eagle dataset integration with explicit path boundaries."""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter
import hashlib
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from torch.utils.data import DataLoader, Subset


TASKS = ("detection", "grounding", "pointing")
ROUTES = ("round1_global_point", "round2_point_box")


def verify_packaged_data_signatures(
    data_root: Path,
    *signature_groups: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Match every scheduled annotation shard to the package checksum ledger."""
    root = data_root.resolve()
    checksum_path = root / "SHA256SUMS"
    if not checksum_path.is_file():
        raise FileNotFoundError(f"Missing package checksum ledger: {checksum_path}")
    expected: dict[str, str] = {}
    for number, line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if "  " not in line:
            raise ValueError(f"Malformed SHA256SUMS line {number}")
        digest, relative = line.split("  ", 1)
        if len(digest) != 64 or relative in expected:
            raise ValueError(f"Invalid SHA256SUMS line {number}")
        expected[relative] = digest

    checked = 0
    for signature in signature_groups:
        for row in signature:
            path = Path(str(row["path"])).resolve()
            try:
                relative = path.relative_to(root).as_posix()
            except ValueError as exc:
                raise ValueError(
                    f"Scheduled annotation is outside DATA_ROOT: {path}"
                ) from exc
            declared = expected.get(relative)
            if declared is None:
                raise ValueError(
                    f"Scheduled annotation is absent from SHA256SUMS: {relative}"
                )
            if str(row["sha256"]) != declared:
                raise ValueError(f"Scheduled annotation checksum mismatch: {relative}")
            checked += 1
    return {
        "checksum_ledger": str(checksum_path),
        "checksum_ledger_sha256": hashlib.sha256(
            checksum_path.read_bytes()
        ).hexdigest(),
        "scheduled_annotation_files_verified": checked,
    }


def annotation_scope(path: str | Path) -> tuple[str, str]:
    """Resolve task and Pixel-PIVR round from a packaged annotation path."""
    parts = set(Path(path).parts)
    tasks = [value for value in TASKS if value in parts]
    routes = [value for value in ROUTES if value in parts]
    if len(tasks) != 1 or len(routes) != 1:
        raise ValueError(f"Cannot resolve task/route from annotation path: {path}")
    return tasks[0], routes[0]


def task_loss_balance_contract(
    signature: Sequence[Mapping[str, Any]], policy: str
) -> dict[str, Any]:
    """Build deterministic per-task weights without changing record exposure.

    Point-to-box expansion creates one Round-2 row per address. Without weighting,
    dense detection addresses can overwhelm the source-query task mixture. The
    source_query_task policy preserves the Round-1 source-query task proportions
    while still exposing every flattened row exactly once.
    """
    if policy not in {"none", "source_query_task"}:
        raise ValueError(f"Unknown loss-balancing policy: {policy}")
    flat: Counter[str] = Counter()
    source: Counter[str] = Counter()
    files: list[dict[str, Any]] = []
    for row in signature:
        records = int(row["records"])
        task, route = annotation_scope(str(row["path"]))
        flat[task] += records
        if route == "round1_global_point":
            source[task] += records
        files.append(
            {
                "path": str(row["path"]),
                "records": records,
                "task": task,
                "route": route,
            }
        )
    total_flat = sum(flat.values())
    total_source = sum(source.values())
    if total_flat <= 0:
        raise ValueError("Cannot balance an empty training signature")
    if policy == "source_query_task" and (
        total_source <= 0
        or any(flat[task] <= 0 or source[task] <= 0 for task in TASKS)
    ):
        raise ValueError(
            "source_query_task requires nonempty Round-1 and flattened records "
            "for detection, grounding, and pointing"
        )

    weights: dict[str, float] = {}
    for task in TASKS:
        if flat[task] <= 0:
            continue
        if policy == "none":
            weights[task] = 1.0
        else:
            desired = source[task] / total_source
            observed = flat[task] / total_flat
            weights[task] = desired / observed
    weighted_total = sum(flat[task] * weights[task] for task in weights)
    if abs(weighted_total - total_flat) > max(1e-6, total_flat * 1e-12):
        raise RuntimeError(
            f"Task weights are not mean-one: {weighted_total} != {total_flat}"
        )
    return {
        "policy": policy,
        "flat_records": {task: int(flat[task]) for task in TASKS},
        "source_query_records": {task: int(source[task]) for task in TASKS},
        "flat_record_percent": {
            task: 100.0 * flat[task] / total_flat for task in TASKS
        },
        "source_query_percent": {
            task: (100.0 * source[task] / total_source if total_source else 0.0)
            for task in TASKS
        },
        "task_weights": weights,
        "mean_weight": weighted_total / total_flat,
        "files": files,
    }


class LossBalancedDataset:
    """Attach a deterministic scalar objective weight to each Eagle sample."""

    def __init__(
        self,
        dataset: Any,
        signature: Sequence[Mapping[str, Any]],
        contract: Mapping[str, Any],
    ) -> None:
        self.dataset = dataset
        self.ends: list[int] = []
        self.weights: list[float] = []
        total = 0
        task_weights = contract["task_weights"]
        for row in signature:
            task, _route = annotation_scope(str(row["path"]))
            total += int(row["records"])
            self.ends.append(total)
            self.weights.append(float(task_weights[task]))
        if total != len(dataset):
            raise ValueError(
                f"Loss-weight ranges cover {total} records, dataset has {len(dataset)}"
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        file_index = bisect_right(self.ends, int(index))
        if file_index >= len(self.ends):
            raise IndexError(index)
        sample = dict(self.dataset[index])
        sample["pixel_pivr_loss_weight"] = self.weights[file_index]
        return sample


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
