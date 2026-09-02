"""Distributed exact-coverage Qwen-LoRA training for Pixel-PIVR."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from .data import make_dataset, make_loader
from .io import (
    append_jsonl,
    atomic_json,
    atomic_torch_save,
    data_signature,
    update_symlink,
)
from .lora import (
    base_causal_lm,
    configure_lora_only,
    restore_trainable_state,
    trainable_state,
)
from .modeling import PixelPIVRTrainingModel
from .magnified_modeling import (
    MagnifiedPreProjectorTrainingModel,
    MagnifiedROIMetadataDataset,
)


METRIC_KEYS = ("loss", "native_loss")


def stratified_file_indices(
    signature: list[Mapping[str, Any]], limit: int
) -> list[int]:
    """Select a deterministic, approximately equal monitor from every JSONL file."""
    counts = [int(row["records"]) for row in signature]
    total = sum(counts)
    if limit <= 0 or limit >= total:
        return list(range(total))
    nonempty = [index for index, count in enumerate(counts) if count > 0]
    if not nonempty:
        return []

    allocation = [0] * len(counts)
    remaining = min(limit, total)
    while remaining:
        progressed = False
        for index in nonempty:
            if allocation[index] < counts[index] and remaining:
                allocation[index] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break

    offsets = []
    cursor = 0
    for count in counts:
        offsets.append(cursor)
        cursor += count

    selected = []
    for offset, count, take in zip(offsets, counts, allocation):
        if take == 0:
            continue
        if take == count:
            local = list(range(count))
        else:
            # Midpoints of equal-width bins cover the complete file without RNG.
            local = [
                min(count - 1, ((2 * slot + 1) * count) // (2 * take))
                for slot in range(take)
            ]
        selected.extend(offset + value for value in local)
    if len(selected) != min(limit, total) or len(set(selected)) != len(selected):
        raise RuntimeError("Stratified validation selection is not unique and complete")
    return selected


def validation_indices(
    signature: list[Mapping[str, Any]], limit: int, policy: str
) -> list[int]:
    total = sum(int(row["records"]) for row in signature)
    effective = total if limit <= 0 else min(limit, total)
    if policy == "first":
        return list(range(effective))
    if policy == "stratified_files":
        return stratified_file_indices(signature, effective)
    raise ValueError(f"Unknown validation sampling policy: {policy}")


def rank_training_indices(
    records: int,
    padding: int,
    seed: int,
    order: str,
    rank: int,
    world_size: int,
    local_start: int,
    local_total: int,
) -> list[int]:
    if records <= 0 or padding < 0 or padding >= max(1, records):
        raise ValueError("Invalid training record or padding count")
    if order == "shuffled":
        schedule = np.random.default_rng(seed).permutation(records)
    elif order == "sequential":
        schedule = np.arange(records, dtype=np.int64)
    else:
        raise ValueError(f"Unknown sample order: {order}")
    if padding:
        schedule = np.concatenate((schedule, schedule[:padding]))
    start = rank + local_start * world_size
    stop = rank + local_total * world_size
    selected = schedule[start:stop:world_size]
    if int(selected.size) != local_total - local_start:
        raise RuntimeError("Per-rank training schedule length is inconsistent")
    return [int(value) for value in selected]


def rank_log(rank: int, message: str) -> None:
    print(f"[rank {rank}] {message}", flush=True)


def scheduler_factor(step: int, total_steps: int, warmup: int) -> float:
    if step < warmup:
        return float(step + 1) / max(1, warmup)
    progress = min(1.0, float(step - warmup) / max(1, total_steps - warmup))
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def distributed_context() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size


def synchronize_gradients(
    parameters: list[nn.Parameter], world_size: int
) -> None:
    if world_size == 1:
        return
    grouped: dict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = {}
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        key = (parameter.grad.device, parameter.grad.dtype)
        grouped.setdefault(key, []).append(parameter.grad)
    for gradients in grouped.values():
        flat = _flatten_dense_tensors(gradients)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(world_size)
        synchronized = _unflatten_dense_tensors(flat, gradients)
        for gradient, value in zip(gradients, synchronized):
            gradient.copy_(value)


def gather_rng_states(
    rank: int, world_size: int
) -> list[dict[str, torch.Tensor]] | None:
    state = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}
    if world_size == 1:
        return [state]
    gathered: list[dict[str, torch.Tensor] | None] | None = (
        [None] * world_size if rank == 0 else None
    )
    dist.gather_object(state, gathered, dst=0)
    if rank:
        return None
    return [value for value in gathered or [] if value is not None]


def aggregate_rows(
    rows: list[Mapping[str, float]], device: torch.device, world_size: int
) -> dict[str, float]:
    values = torch.tensor(
        [sum(float(row.get(key, 0.0)) for row in rows) for key in METRIC_KEYS]
        + [float(len(rows))],
        dtype=torch.float64,
        device=device,
    )
    if world_size > 1:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    count = float(values[-1].item())
    if count <= 0:
        raise RuntimeError("No metric rows were available for reduction")
    return {
        key: float(values[index].item() / count)
        for index, key in enumerate(METRIC_KEYS)
    }


@torch.no_grad()
def validate(
    wrapper: PixelPIVRTrainingModel,
    dataset: Any,
    indices: list[int],
    workers: int,
    *,
    rank: int,
    world_size: int,
) -> dict[str, float]:
    wrapper.eval()
    causal_model = base_causal_lm(wrapper.model.language_model)
    # LocateAnything selects its deterministic MTP mask through this flag.
    # Child modules remain in eval mode, so LoRA dropout stays disabled.
    causal_model.model.training = True
    rows: list[dict[str, float]] = []
    for sample in make_loader(dataset, indices[rank::world_size], workers):
        _loss, metrics = wrapper(sample)
        rows.append(metrics)
    causal_model.model.training = False
    wrapper.train()
    return aggregate_rows(rows, next(wrapper.parameters()).device, world_size)


def choose_vision_attention(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return "sdpa"
    return "flash_attention_2"


def shared_data_signatures(
    train_paths: list[Path],
    validation_paths: list[Path],
    *,
    rank: int,
    world_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload: list[Any] = [None]
    if rank == 0:
        payload[0] = {
            "train": data_signature(train_paths),
            "validation": data_signature(validation_paths),
        }
    if world_size > 1:
        dist.broadcast_object_list(payload, src=0)
    value = payload[0]
    if not isinstance(value, Mapping):
        raise RuntimeError("Failed to broadcast data signatures")
    return list(value["train"]), list(value["validation"])


def checkpoint_payload(
    *,
    args: argparse.Namespace,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    exposure: int,
    validation: Mapping[str, float] | None,
    lora_audit: Mapping[str, int],
    world_size: int,
    max_steps: int,
    train_signature: list[dict[str, Any]],
    validation_signature: list[dict[str, Any]],
    init_adapter_signature: dict[str, Any] | None,
    rng_states: list[dict[str, torch.Tensor]] | None,
) -> dict[str, Any]:
    return {
        "schema_version": "pixel-pivr-lora-checkpoint-v3",
        "config": {
            "experiment": str(args.visual_context),
            "base_model": str(args.model.resolve()),
            "data_root": str(args.data_root.resolve()),
            "train_data": train_signature,
            "validation_data": validation_signature,
            "lora_rank": int(args.lora_rank),
            "image_token_limit": int(args.image_token_limit),
            "max_sequence": int(args.max_sequence),
            "visual_context": str(args.visual_context),
            "magnified_roi_pixels": int(args.magnified_roi_pixels),
            "magnified_roi_stride": int(args.magnified_roi_stride),
            "gradient_accumulation": int(args.gradient_accumulation),
            "max_steps": int(max_steps),
            "world_size": int(world_size),
            "seed": int(args.seed),
            "validation_sampling": str(args.validation_sampling),
            "validation_records": int(args.validation_records),
            "init_adapter": init_adapter_signature,
            "sample_order": str(args.sample_order),
            "lora_audit": dict(lora_audit),
        },
        "step": int(step),
        "exposure": int(exposure),
        "validation": None if validation is None else dict(validation),
        "trainable_state": trainable_state(model),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state(),
        "distributed_rng_states": rng_states,
    }


def verify_resume_contract(
    payload: Mapping[str, Any],
    *,
    args: argparse.Namespace,
    world_size: int,
    max_steps: int,
    train_signature: list[dict[str, Any]],
    validation_signature: list[dict[str, Any]],
    init_adapter_signature: dict[str, Any] | None,
) -> None:
    config = payload.get("config") or {}
    expected = {
        "base_model": str(args.model.resolve()),
        "data_root": str(args.data_root.resolve()),
        "train_data": train_signature,
        "validation_data": validation_signature,
        "lora_rank": int(args.lora_rank),
        "image_token_limit": int(args.image_token_limit),
        "max_sequence": int(args.max_sequence),
        "visual_context": str(args.visual_context),
        "magnified_roi_pixels": int(args.magnified_roi_pixels),
        "magnified_roi_stride": int(args.magnified_roi_stride),
        "gradient_accumulation": int(args.gradient_accumulation),
        "max_steps": int(max_steps),
        "world_size": int(world_size),
        "seed": int(args.seed),
        "validation_sampling": str(args.validation_sampling),
        "validation_records": int(args.validation_records),
        "init_adapter": init_adapter_signature,
        "sample_order": str(args.sample_order),
    }
    mismatches = {
        key: {"saved": config.get(key), "requested": value}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "Resume contract changed:\n" + json.dumps(mismatches, indent=2)
        )


def train(args: argparse.Namespace) -> None:
    from transformers import AutoConfig, AutoModel, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    rank, local_rank, world_size = distributed_context()
    device = torch.device(f"cuda:{local_rank}")
    process_seed = int(args.seed) + rank
    torch.manual_seed(process_seed)
    torch.cuda.manual_seed(process_seed)
    np.random.seed(process_seed)
    random.seed(process_seed)
    torch.set_float32_matmul_precision("high")

    train_paths = [path.resolve() for path in args.train_data]
    validation_paths = [path.resolve() for path in args.validation_data]
    rank_log(rank, f"signing data files on cuda:{local_rank}")
    train_signature, validation_signature = shared_data_signatures(
        train_paths,
        validation_paths,
        rank=rank,
        world_size=world_size,
    )
    adapter_payload: list[Any] = [None]
    if rank == 0 and args.init_adapter is not None:
        adapter_payload[0] = data_signature([args.init_adapter.resolve()])[0]
    if world_size > 1:
        dist.broadcast_object_list(adapter_payload, src=0)
    init_adapter_signature = adapter_payload[0]
    train_records = sum(int(row["records"]) for row in train_signature)
    validation_records = sum(int(row["records"]) for row in validation_signature)
    if args.expected_train_records and train_records != args.expected_train_records:
        raise RuntimeError(
            f"Train count changed: {train_records} != {args.expected_train_records}"
        )
    if (
        args.expected_validation_records
        and validation_records != args.expected_validation_records
    ):
        raise RuntimeError(
            "Validation count changed: "
            f"{validation_records} != {args.expected_validation_records}"
        )

    global_records_per_step = world_size * int(args.gradient_accumulation)
    total_scheduled = train_records + int(args.allowed_padding_records)
    if args.max_steps:
        max_steps = int(args.max_steps)
        required_exposures = max_steps * global_records_per_step
        if not args.smoke and required_exposures != total_scheduled:
            raise RuntimeError(
                "Exact coverage requires steps*world_size*accumulation == "
                "records+explicit_padding; "
                f"got {required_exposures} != {total_scheduled}"
            )
    else:
        if total_scheduled % global_records_per_step:
            raise RuntimeError(
                f"{total_scheduled} scheduled records are not divisible by global "
                f"batch {global_records_per_step}; set --allowed-padding-records explicitly"
            )
        max_steps = total_scheduled // global_records_per_step
        required_exposures = total_scheduled
    if max_steps <= 0:
        raise RuntimeError("Derived max_steps is not positive")
    if args.warmup_steps >= max_steps:
        raise RuntimeError("Warm-up must be shorter than the complete run")

    rank_log(rank, "loading processor and configuration")
    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=not args.allow_download,
        use_fast=False,
    )
    processor.image_processor.in_token_limit = int(args.image_token_limit)
    processor.tokenizer.model_max_length = int(args.max_sequence)
    config = AutoConfig.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=not args.allow_download,
    )
    vision_attention = choose_vision_attention(args.vision_attention)
    config._attn_implementation = "sdpa"
    config._attn_implementation_autoset = False
    config.text_config._attn_implementation = "sdpa"
    config.text_config._attn_implementation_autoset = False
    config.vision_config._attn_implementation = vision_attention
    config.vision_config._attn_implementation_autoset = False
    def load_model() -> nn.Module:
        return AutoModel.from_pretrained(
            args.model,
            config=config,
            trust_remote_code=True,
            local_files_only=not args.allow_download,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).to(device)

    if args.serial_model_load and world_size > 1:
        model = None
        for loading_rank in range(world_size):
            if rank == loading_rank:
                rank_log(rank, "loading model (serialized across ranks)")
                model = load_model()
                rank_log(rank, "model load complete")
            dist.barrier(device_ids=[local_rank])
        if model is None:
            raise RuntimeError("Serialized model load did not initialize this rank")
    else:
        rank_log(rank, "loading model")
        model = load_model()
        rank_log(rank, "model load complete")

    rank_log(rank, "installing Qwen LoRA")
    lora_audit = configure_lora_only(model, args.lora_rank)
    if args.init_adapter is not None:
        initial = torch.load(
            args.init_adapter, map_location="cpu", weights_only=False
        )
        initial_rank = int((initial.get("config") or {}).get("lora_rank", -1))
        if initial_rank != args.lora_rank:
            raise RuntimeError(
                f"Initial adapter rank {initial_rank} != requested {args.lora_rank}"
            )
        restore_trainable_state(model, initial["trainable_state"])
    model.vision_model.eval()
    model.mlp1.eval()

    rank_log(rank, "constructing exact Eagle datasets")
    train_dataset = make_dataset(
        train_paths,
        processor,
        data_root=args.data_root,
        eagle_root=args.eagle_root,
        block_size=6,
    )
    validation_dataset = make_dataset(
        validation_paths,
        processor,
        data_root=args.data_root,
        eagle_root=args.eagle_root,
        block_size=6,
    )
    if args.visual_context == "preprojector_magnified_roi":
        train_dataset = MagnifiedROIMetadataDataset(train_dataset)
        validation_dataset = MagnifiedROIMetadataDataset(validation_dataset)
        image_start_id = processor.tokenizer.convert_tokens_to_ids(
            processor.image_start_token
        )
        image_end_id = processor.tokenizer.convert_tokens_to_ids(
            processor.image_end_token
        )
        wrapper = MagnifiedPreProjectorTrainingModel(
            model,
            roi_pixels=args.magnified_roi_pixels,
            roi_stride=args.magnified_roi_stride,
            image_start_id=image_start_id,
            image_end_id=image_end_id,
            max_sequence=args.max_sequence,
        ).to(device)
    else:
        wrapper = PixelPIVRTrainingModel(model).to(device)
    if len(train_dataset) != train_records or len(validation_dataset) != validation_records:
        raise RuntimeError("Eagle loader counts disagree with signed JSONL counts")
    rank_log(rank, "dataset construction complete")

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda current: scheduler_factor(current, max_steps, args.warmup_steps),
    )
    output = args.output.resolve()
    output_had_contents = output.exists() and any(output.iterdir())
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        signature_path = output / "run_contract.json"
        contract = {
            "schema_version": "pixel-pivr-run-contract-v2",
            "model": str(args.model.resolve()),
            "data_root": str(args.data_root.resolve()),
            "train_data": train_signature,
            "validation_data": validation_signature,
            "world_size": world_size,
            "gradient_accumulation": args.gradient_accumulation,
            "global_records_per_step": global_records_per_step,
            "max_steps": max_steps,
            "required_exposures": required_exposures,
            "allowed_padding_records": args.allowed_padding_records,
            "lora": lora_audit,
            "vision_attention": vision_attention,
            "visual_context": str(args.visual_context),
            "magnified_roi_pixels": int(args.magnified_roi_pixels),
            "magnified_roi_stride": int(args.magnified_roi_stride),
            "init_adapter": init_adapter_signature,
            "validation_sampling": str(args.validation_sampling),
            "validation_records": int(args.validation_records),
            "sample_order": str(args.sample_order),
        }
        if signature_path.exists():
            saved = json.loads(signature_path.read_text(encoding="utf-8"))
            if saved != contract:
                raise RuntimeError("Existing output has a different run contract")
        else:
            atomic_json(signature_path, contract)
    if world_size > 1:
        dist.barrier(device_ids=[local_rank])
    rank_log(rank, "startup synchronization complete")

    start_step = 0
    start_exposure = 0
    best_validation = float("inf")
    resume = output / "last.pt"
    if resume.is_file() or resume.is_symlink():
        if args.no_resume:
            raise RuntimeError(f"Output has a checkpoint but --no-resume was set: {resume}")
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        verify_resume_contract(
            payload,
            args=args,
            world_size=world_size,
            max_steps=max_steps,
            train_signature=train_signature,
            validation_signature=validation_signature,
            init_adapter_signature=init_adapter_signature,
        )
        restore_trainable_state(model, payload["trainable_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        scheduler.load_state_dict(payload["scheduler_state"])
        start_step = int(payload["step"])
        start_exposure = int(payload["exposure"])
        states = payload.get("distributed_rng_states") or []
        if rank < len(states):
            torch.set_rng_state(states[rank]["cpu"])
            torch.cuda.set_rng_state(states[rank]["cuda"])
        best_path = output / "best.pt"
        if best_path.exists() or best_path.is_symlink():
            best = torch.load(best_path, map_location="cpu", weights_only=False)
            best_validation = float(best["validation"]["native_loss"])
    elif args.no_resume and output_had_contents:
        raise RuntimeError(f"--no-resume requires an empty output directory: {output}")

    if start_step >= max_steps:
        if rank == 0:
            print(f"Pixel-PIVR training already complete at step {start_step}")
        if world_size > 1:
            dist.destroy_process_group()
        return
    if start_exposure % world_size:
        raise RuntimeError("Saved global exposure is not divisible by world size")

    local_total = max_steps * int(args.gradient_accumulation)
    local_start = start_exposure // world_size
    local_indices = rank_training_indices(
        train_records,
        int(args.allowed_padding_records),
        int(args.seed),
        str(args.sample_order),
        rank,
        world_size,
        local_start,
        local_total,
    )
    monitor_indices = validation_indices(
        validation_signature,
        int(args.validation_records),
        str(args.validation_sampling),
    )
    if rank == 0:
        atomic_json(
            output / "validation_monitor.json",
            {
                "policy": str(args.validation_sampling),
                "requested_records": int(args.validation_records),
                "selected_records": len(monitor_indices),
                "files": validation_signature,
                "indices_sha256": hashlib.sha256(
                    json.dumps(monitor_indices, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
            },
        )

    interrupted = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    wandb_run = None
    if rank == 0 and args.wandb_project:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("Install pixel-pivr[tracking] to enable W&B") from exc
        contract_bytes = (output / "run_contract.json").read_bytes()
        wandb_run_id = args.wandb_run_id or hashlib.sha256(
            str(output).encode("utf-8") + b"\0" + contract_bytes
        ).hexdigest()[:24]
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or output.name,
            config=json.loads((output / "run_contract.json").read_text(encoding="utf-8")),
            resume="allow",
            id=wandb_run_id,
        )
        atomic_json(
            output / "wandb.json",
            {
                "project": args.wandb_project,
                "name": args.wandb_name or output.name,
                "id": wandb_run_id,
            },
        )

    loader = make_loader(train_dataset, local_indices, args.workers)
    wrapper.train()
    optimizer.zero_grad(set_to_none=True)
    running: list[dict[str, float]] = []
    step = start_step
    local_exposure = local_start
    exposure = start_exposure
    started = time.time()

    def save(
        validation: Mapping[str, float] | None,
        *,
        reason: str,
    ) -> None:
        nonlocal best_validation
        rng_states = gather_rng_states(rank, world_size)
        if rank == 0:
            checkpoint = output / f"checkpoint-step{step:07d}.pt"
            payload = checkpoint_payload(
                args=args,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                exposure=exposure,
                validation=validation,
                lora_audit=lora_audit,
                world_size=world_size,
                max_steps=max_steps,
                train_signature=train_signature,
                validation_signature=validation_signature,
                init_adapter_signature=init_adapter_signature,
                rng_states=rng_states,
            )
            atomic_torch_save(checkpoint, payload)
            update_symlink(output / "last.pt", checkpoint)
            if validation is not None and validation["native_loss"] < best_validation:
                best_validation = validation["native_loss"]
                update_symlink(output / "best.pt", checkpoint)
            atomic_json(
                output / "status.json",
                {
                    "schema_version": "pixel-pivr-status-v2",
                    "step": step,
                    "max_steps": max_steps,
                    "exposure": exposure,
                    "required_exposures": required_exposures,
                    "validation": validation,
                    "best_validation_native_loss": best_validation,
                    "reason": reason,
                    "complete_one_pass": exposure >= train_records,
                },
            )
        if world_size > 1:
            dist.barrier(device_ids=[local_rank])

    for sample in loader:
        local_exposure += 1
        exposure = local_exposure * world_size
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, metrics = wrapper(sample)
            scaled = loss / int(args.gradient_accumulation)
        scaled.backward()
        running.append(metrics)
        if local_exposure % int(args.gradient_accumulation):
            continue

        synchronize_gradients(trainable, world_size)
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        averaged = aggregate_rows(running, device, world_size)
        running.clear()
        if rank == 0:
            record = {
                "step": step,
                "exposure": exposure,
                "world_size": world_size,
                "records_per_optimizer_step": global_records_per_step,
                "seed": int(args.seed),
                "learning_rate": scheduler.get_last_lr()[0],
                "gradient_norm": float(gradient_norm),
                "elapsed_seconds": time.time() - started,
                "peak_gpu_mb": torch.cuda.max_memory_allocated() / 1024**2,
                **averaged,
            }
            append_jsonl(output / "training_curve.jsonl", record)
            if wandb_run is not None:
                wandb_run.log({f"train/{key}": value for key, value in record.items()}, step=step)
            if step == 1 or step % args.log_steps == 0:
                print(json.dumps(record, sort_keys=True), flush=True)

        stop_flag = torch.tensor(
            int(interrupted), dtype=torch.int32, device=device
        )
        if world_size > 1:
            dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
        should_stop = bool(stop_flag.item())
        should_validate = step % args.eval_steps == 0 or step == max_steps
        should_checkpoint = (
            should_stop
            or step % args.checkpoint_steps == 0
            or should_validate
        )
        validation = None
        if should_validate:
            validation = validate(
                wrapper,
                validation_dataset,
                monitor_indices,
                args.workers,
                rank=rank,
                world_size=world_size,
            )
            if rank == 0:
                validation_row = {"validation_step": step, **validation}
                append_jsonl(output / "validation_curve.jsonl", validation_row)
                print(json.dumps(validation_row, sort_keys=True), flush=True)
                if wandb_run is not None:
                    wandb_run.log(
                        {f"validation/{key}": value for key, value in validation.items()},
                        step=step,
                    )
        if should_checkpoint:
            save(validation, reason="signal" if should_stop else "scheduled")
        if should_stop:
            if rank == 0:
                atomic_json(
                    output / "interrupted.json",
                    {"step": step, "exposure": exposure, "resume": str(output / "last.pt")},
                )
                print(f"Saved aligned interruption checkpoint at step {step}", flush=True)
            if wandb_run is not None:
                wandb_run.finish(exit_code=0)
            if world_size > 1:
                dist.destroy_process_group()
            return
        if step >= max_steps:
            break

    if step != max_steps or exposure != required_exposures:
        raise RuntimeError(
            f"Incomplete run: step={step}/{max_steps}, "
            f"exposure={exposure}/{required_exposures}"
        )
    if rank == 0:
        atomic_json(
            output / "done.json",
            {
                "schema_version": "pixel-pivr-done-v2",
                "steps": step,
                "record_exposures": exposure,
                "unique_training_records": train_records,
                "padding_record_exposures": max(0, exposure - train_records),
                "world_size": world_size,
                "gradient_accumulation": int(args.gradient_accumulation),
                "records_per_optimizer_step": global_records_per_step,
                "complete_one_pass": exposure >= train_records,
                "best": str((output / "best.pt").resolve()),
                "last": str((output / "last.pt").resolve()),
            },
        )
        (output / "interrupted.json").unlink(missing_ok=True)
        if wandb_run is not None:
            wandb_run.finish()
    if world_size > 1:
        dist.barrier(device_ids=[local_rank])
        dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--eagle-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, nargs="+", required=True)
    parser.add_argument("--validation-data", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--init-adapter", type=Path)
    parser.add_argument("--expected-train-records", type=int, default=0)
    parser.add_argument("--expected-validation-records", type=int, default=0)
    parser.add_argument("--validation-records", type=int, default=0)
    parser.add_argument(
        "--validation-sampling",
        choices=("stratified_files", "first"),
        default="stratified_files",
        help="Select validation rows evenly across files or from the concatenated prefix",
    )
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--image-token-limit", type=int, default=1024)
    parser.add_argument("--max-sequence", type=int, default=8192)
    parser.add_argument(
        "--visual-context",
        choices=("pixel_reencoded", "preprojector_magnified_roi"),
        default="pixel_reencoded",
    )
    parser.add_argument(
        "--magnified-roi-pixels",
        type=int,
        default=380,
        help="Nominal point-centred field; 380 maps to 27x27 MoonViT patches (378 px)",
    )
    parser.add_argument(
        "--magnified-roi-stride",
        type=int,
        choices=(1, 2),
        default=1,
        help="Stride 1 is the magnified path; stride 2 is its native-density control",
    )
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--allowed-padding-records", type=int, default=0)
    parser.add_argument(
        "--sample-order",
        choices=("shuffled", "sequential"),
        default="shuffled",
        help="Deterministic global shuffle is the recommended exact-coverage order",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--checkpoint-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument(
        "--vision-attention",
        choices=("auto", "flash_attention_2", "sdpa"),
        default="auto",
    )
    parser.add_argument(
        "--serial-model-load",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load the checkpoint one rank at a time to avoid shared-I/O stalls",
    )
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-run-id")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.init_adapter is not None and args.no_resume is False:
        # A new output can use init-adapter. The runtime rejects it only if a
        # same-run last.pt also exists.
        pass
    for path in [
        args.model,
        args.eagle_root,
        args.data_root,
        *args.train_data,
        *args.validation_data,
    ]:
        if not path.exists() and not path.is_symlink():
            parser.error(f"Missing input: {path}")
    if args.init_adapter is not None and not args.init_adapter.exists():
        parser.error(f"Missing initial adapter: {args.init_adapter}")
    for name in (
        "lora_rank",
        "image_token_limit",
        "max_sequence",
        "magnified_roi_pixels",
        "magnified_roi_stride",
        "gradient_accumulation",
        "checkpoint_steps",
        "eval_steps",
        "log_steps",
    ):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_steps < 0 or args.allowed_padding_records < 0:
        parser.error("Step and padding counts cannot be negative")
    return args


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
