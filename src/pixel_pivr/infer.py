"""Run standalone Pixel-PIVR HBB inference from an explicit JSONL manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import torch
from PIL import Image

from .decoder import AddressedCrop, PixelPIVRWaveDecoder
from .magnified_decoder import MagnifiedPreProjectorWaveDecoder
from .adaptive_multiscale_decoder import AdaptiveMultiScaleWaveDecoder
from .train import ADAPTIVE_CONTEXT, parse_int_csv
from .geometry import (
    box_touches_frame,
    canonical_label,
    cached_feature_point_prompt,
    class_nms,
    detection_counts,
    discovery_prompt,
    global_fallback_point_prompt,
    hbox_iou,
    local_hbox_to_global,
    parse_labeled_points,
    parse_single_hbb,
    point_address_prompt,
    point_centered_crop,
    point_inside_hbox,
    precision_recall_f1,
    resize_local_point,
)
from .io import append_jsonl, atomic_json, count_jsonl
from .worker import LocateAnythingPixelPIVRWorker


def row_label(value: Any, row: Mapping[str, Any]) -> str:
    """Canonicalize a label using aliases explicitly signed by the manifest."""
    label = canonical_label(value)
    aliases = {
        canonical_label(source): canonical_label(target)
        for source, target in (row.get("label_aliases") or {}).items()
    }
    return aliases.get(label, label)


def row_class_prompts(classes: list[str], row: Mapping[str, Any]) -> list[str]:
    """Return model-facing names while retaining canonical scoring classes."""
    prompts = {
        row_label(source, row): str(target).strip()
        for source, target in (row.get("class_prompts") or {}).items()
    }
    if any(not value for value in prompts.values()):
        raise ValueError("class_prompts contains an empty model-facing label")
    return [prompts.get(label, label) for label in classes]


def resolve_image(raw: str, data_root: Path) -> Path:
    path = Path(raw)
    path = path if path.is_absolute() else data_root / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def supplied_points(
    row: Mapping[str, Any], width: int, height: int
) -> list[dict[str, Any]] | None:
    if "points" not in row:
        return None
    coordinate_space = str(row.get("point_coordinate_space") or "pixel")
    output = []
    seen: set[tuple[str, int, int]] = set()
    for index, value in enumerate(row.get("points") or []):
        label = row_label(value.get("label"), row)
        point = [float(item) for item in value["point"]]
        if coordinate_space == "normalized_0_1000":
            point = [point[0] / 1000.0 * width, point[1] / 1000.0 * height]
        elif coordinate_space != "pixel":
            raise ValueError(f"Unsupported point coordinate space: {coordinate_space}")
        nx = max(0, min(1000, int(round(point[0] / max(1, width) * 1000))))
        ny = max(0, min(1000, int(round(point[1] / max(1, height) * 1000))))
        key = (label, nx, ny)
        if key in seen:
            continue
        seen.add(key)
        output.append(
            {
                "label": label,
                "point": point,
                "point_norm": [nx, ny],
                "source": "manifest",
                "point_id": str(value.get("point_id") or f"{label}:{nx}:{ny}:{index}"),
            }
        )
    return output


def normalize_ground_truth(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    coordinate_space = str(row.get("gt_coordinate_space") or "pixel")
    image_size = row.get("image_size")
    output = []
    for target in row.get("gt") or []:
        box = [float(value) for value in target["hbox"]]
        if coordinate_space == "normalized_0_1000":
            if not image_size:
                raise ValueError("Normalized GT requires image_size=[width,height]")
            width, height = map(float, image_size)
            box = [
                box[0] / 1000.0 * width,
                box[1] / 1000.0 * height,
                box[2] / 1000.0 * width,
                box[3] / 1000.0 * height,
            ]
        elif coordinate_space != "pixel":
            raise ValueError(f"Unsupported GT coordinate space: {coordinate_space}")
        output.append({"label": row_label(target.get("label"), row), "hbox": box})
    return output


def pointing_counts(
    ground_truth: list[dict[str, Any]], points: list[dict[str, Any]]
) -> tuple[int, int, int]:
    matched: set[int] = set()
    true_positive = false_positive = 0
    for point in points:
        candidates = [
            index
            for index, target in enumerate(ground_truth)
            if index not in matched
            and canonical_label(target.get("label")) == canonical_label(point.get("label"))
            and point_inside_hbox(point["point"], target["hbox"])
        ]
        if candidates:
            matched.add(candidates[0])
            true_positive += 1
        else:
            false_positive += 1
    return true_positive, false_positive, len(ground_truth) - len(matched)


def update_metrics(
    row: Mapping[str, Any],
    detection: Counter,
    pointing: Counter,
    grounding: Counter,
) -> None:
    metric = row.get("metric") or {}
    task = str(row.get("task") or "detection")
    if task == "detection" and metric:
        detection.update(
            tp=int(metric.get("tp", 0)),
            fp=int(metric.get("fp", 0)),
            fn=int(metric.get("fn", 0)),
        )
    elif task == "pointing" and metric:
        pointing.update(
            tp=int(metric.get("tp", 0)),
            fp=int(metric.get("fp", 0)),
            fn=int(metric.get("fn", 0)),
        )
    elif task == "grounding" and metric:
        overlap = float(metric.get("iou", 0.0))
        grounding.update(
            queries=1,
            iou_sum=overlap,
            acc_0_5=int(overlap >= 0.5),
            acc_0_7=int(overlap >= 0.7),
        )


def sample_seed(base_seed: int, sample_key: str) -> int:
    digest = hashlib.sha256(sample_key.encode("utf-8")).digest()
    return (int(base_seed) + int.from_bytes(digest[:4], "big")) % (2**31)


def run(args: argparse.Namespace) -> None:
    if not 1 <= args.wave_size <= 200:
        raise ValueError("--wave-size must be in [1, 200]")
    args.output.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output / "predictions.jsonl"
    execution_path = args.output / "execution.jsonl"
    if (predictions_path.exists() or execution_path.exists()) and not args.resume:
        raise FileExistsError(
            f"Output already contains predictions; choose a new directory or pass --resume: {args.output}"
        )

    processed: set[str] = set()
    detection_totals: Counter = Counter()
    pointing_totals: Counter = Counter()
    grounding_totals: Counter = Counter()
    tasks_seen: Counter = Counter()
    if args.resume and predictions_path.is_file():
        with predictions_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                prior = json.loads(line)
                sample_key = str(prior["sample_key"])
                if sample_key in processed:
                    raise RuntimeError(f"Duplicate prior sample_key: {sample_key}")
                processed.add(sample_key)
                tasks_seen[str(prior.get("task") or "detection")] += 1
                update_metrics(
                    prior, detection_totals, pointing_totals, grounding_totals
                )

    worker = LocateAnythingPixelPIVRWorker(
        args.model,
        args.adapter,
        device=args.device,
        dtype=args.dtype,
        local_files_only=not args.allow_download,
        allow_unsynchronized_adapter=args.allow_unsynchronized_adapter,
    )
    worker.processor.image_processor.in_token_limit = int(args.image_token_limit)
    if args.visual_context == ADAPTIVE_CONTEXT:
        decoder = AdaptiveMultiScaleWaveDecoder(
            worker,
            image_token_limit=args.image_token_limit,
            prefix_cache_mode=args.prefix_cache_mode,
            roi_pixels=args.multiscale_roi_pixels,
            target_patches=args.multiscale_target_patches,
            roi_stride=args.magnified_roi_stride,
            fusion_hidden=args.multiscale_fusion_hidden,
            preferred_scale=args.multiscale_preferred_scale,
        )
    elif args.visual_context == "preprojector_magnified_roi":
        decoder = MagnifiedPreProjectorWaveDecoder(
            worker,
            image_token_limit=args.image_token_limit,
            prefix_cache_mode=args.prefix_cache_mode,
            roi_pixels=args.magnified_roi_pixels,
            roi_stride=args.magnified_roi_stride,
        )
    else:
        decoder = PixelPIVRWaveDecoder(
            worker,
            image_token_limit=args.image_token_limit,
            prefix_cache_mode=args.prefix_cache_mode,
            geometry_prefix_mode=args.geometry_prefix_mode,
        )
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    total_point_seconds = 0.0
    total_refinement_seconds = 0.0
    if args.resume and execution_path.is_file():
        with execution_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                prior = json.loads(line)
                total_point_seconds += float(prior.get("point_stage_seconds", 0.0))
                total_refinement_seconds += float(prior.get("end_to_end_seconds", 0.0))
    with args.manifest.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"Blank manifest line at {index}")
            row = json.loads(line)
            image_id = str(row.get("image_id") or f"row-{index:08d}")
            sample_key = str(row.get("sample_key") or image_id)
            if sample_key in processed:
                continue
            seed = sample_seed(args.seed, sample_key)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            task = str(row.get("task") or "detection")
            if task not in {"detection", "grounding", "pointing"}:
                raise ValueError(f"Unsupported task {task!r} for {sample_key}")
            image_path = resolve_image(str(row["image"]), args.data_root)
            image = Image.open(image_path).convert("RGB")
            width, height = image.size
            classes = [row_label(value, row) for value in row.get("classes") or []]
            if not classes:
                raise ValueError(f"{image_id} has no explicit classes")
            prompt_classes = row_class_prompts(classes, row)

            points = supplied_points(row, width, height)
            point_answer = None
            point_seconds = 0.0
            if points is None:
                started = time.perf_counter()
                point_answer = worker.predict_points(
                    image,
                    str(row.get("point_prompt") or discovery_prompt(prompt_classes)),
                    max_new_tokens=args.point_max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                point_seconds = time.perf_counter() - started
                points = parse_labeled_points(point_answer, width, height)
                for point in points:
                    point["label"] = row_label(point.get("label"), row)
            allowed = set(classes)
            points = [point for point in points if point["label"] in allowed]
            ground_truth = normalize_ground_truth(
                {**row, "image_size": [width, height]}
            )

            if task == "pointing":
                tp, fp, fn = pointing_counts(ground_truth, points)
                metric = {"tp": tp, "fp": fp, "fn": fn}
                output_row = {
                    "sample_key": sample_key,
                    "image_id": image_id,
                    "task": task,
                    "image": str(image_path),
                    "classes": classes,
                    "point_query_output": point_answer,
                    "points": points,
                    "predictions": [],
                    "metric": metric,
                }
                append_jsonl(predictions_path, output_row)
                append_jsonl(
                    execution_path,
                    {
                        "sample_key": sample_key,
                        "image_id": image_id,
                        "point_stage_seconds": point_seconds,
                        "end_to_end_seconds": 0.0,
                    },
                )
                tasks_seen[task] += 1
                update_metrics(
                    output_row,
                    detection_totals,
                    pointing_totals,
                    grounding_totals,
                )
                total_point_seconds += point_seconds
                continue

            branches: list[AddressedCrop] = []
            tasks: list[dict[str, Any]] = []
            for point in points:
                if args.point_address_prompt_schema == "compact":
                    point_prompt = cached_feature_point_prompt
                else:
                    point_prompt = point_address_prompt
                if args.visual_context in {
                    "preprojector_magnified_roi",
                    ADAPTIVE_CONTEXT,
                }:
                    question = point_prompt(
                        str(point["label"]), point["point"], width, height
                    )
                    branches.append(
                        AddressedCrop(
                            str(point["point_id"]),
                            str(point["label"]),
                            None,
                            question,
                            tuple(int(value) for value in point["point_norm"]),
                        )
                    )
                    tasks.append(
                        {
                            "point_id": point["point_id"],
                            "label": point["label"],
                            "global_point": point["point"],
                            "local_point": point["point"],
                            "crop_box": None,
                            "crop_size": [width, height],
                        }
                    )
                else:
                    crop_box = point_centered_crop(
                        point["point"], width, height, args.crop_side
                    )
                    left, top, _right, _bottom = crop_box
                    crop = image.crop(crop_box)
                    source_crop_size = [crop.width, crop.height]
                    local_point = [
                        float(point["point"][0]) - left,
                        float(point["point"][1]) - top,
                    ]
                    if args.local_resize_side:
                        output_side = int(args.local_resize_side)
                        local_point = resize_local_point(
                            local_point, source_crop_size, [output_side, output_side]
                        )
                        crop = crop.resize(
                            (output_side, output_side),
                            resample=Image.Resampling.LANCZOS,
                        )
                    question = point_prompt(
                        str(point["label"]), local_point, crop.width, crop.height
                    )
                    branches.append(
                        AddressedCrop(
                            str(point["point_id"]), str(point["label"]), crop, question
                        )
                    )
                    tasks.append(
                        {
                            "point_id": point["point_id"],
                            "label": point["label"],
                            "global_point": point["point"],
                            "local_point": local_point,
                            "crop_box": list(crop_box),
                            "crop_size": [crop.width, crop.height],
                            "source_crop_size": source_crop_size,
                        }
                    )

            decoded, execution = decoder.decode_image(
                image,
                branches,
                requested_wave_size=args.wave_size,
                allow_none=args.allow_none,
            )
            if len(decoded) != len(tasks):
                raise RuntimeError(
                    f"Address/output mismatch for {image_id}: {len(decoded)} != {len(tasks)}"
                )
            raw_predictions = []
            refinements = []
            fallback_branches: list[AddressedCrop] = []
            fallback_tasks: list[dict[str, Any]] = []
            for address_task, result in zip(tasks, decoded):
                crop_width, crop_height = address_task["crop_size"]
                local_prediction = parse_single_hbb(
                    result["answer"], address_task["label"], crop_width, crop_height
                )
                accepted = False
                global_prediction = None
                contains_address = bool(
                    local_prediction is not None
                    and point_inside_hbox(
                        address_task["local_point"], local_prediction["hbox"]
                    )
                )
                touches_crop_edge = bool(
                    args.global_fallback
                    and address_task["crop_box"] is not None
                    and local_prediction is not None
                    and box_touches_frame(
                        local_prediction["hbox"],
                        crop_width,
                        crop_height,
                        args.fallback_edge_margin,
                    )
                )
                needs_fallback = bool(
                    args.global_fallback
                    and args.visual_context == "pixel_reencoded"
                    and address_task["crop_box"] is not None
                    and (not contains_address or touches_crop_edge)
                )
                if contains_address and not needs_fallback:
                    global_prediction = dict(local_prediction)
                    if address_task["crop_box"] is not None:
                        global_prediction["hbox"] = local_hbox_to_global(
                            local_prediction["hbox"],
                            address_task["crop_box"],
                            address_task["crop_size"],
                        )
                    global_prediction["point_id"] = address_task["point_id"]
                    global_prediction["witness_point"] = address_task["global_point"]
                    global_prediction["source"] = "pixel_pivr_constrained_pbd6"
                    raw_predictions.append(global_prediction)
                    accepted = True
                refinement = {
                    **address_task,
                    **result,
                    "accepted": accepted,
                    "local_prediction": local_prediction,
                    "local_prediction_coordinate_space": "local_view_pixels",
                    "local_contains_address": contains_address,
                    "local_touches_crop_edge": touches_crop_edge,
                    "global_fallback_requested": needs_fallback,
                    "prediction": global_prediction,
                    "prediction_coordinate_space": "global_image_pixels",
                }
                refinements.append(refinement)
                if needs_fallback:
                    fallback_branches.append(
                        AddressedCrop(
                            str(address_task["point_id"]),
                            str(address_task["label"]),
                            None,
                            global_fallback_point_prompt(
                                str(address_task["label"]),
                                address_task["global_point"],
                                width,
                                height,
                            ),
                        )
                    )
                    fallback_tasks.append(
                        {
                            "address_task": address_task,
                            "refinement_index": len(refinements) - 1,
                        }
                    )

            fallback_execution = None
            if fallback_branches:
                fallback_decoded, fallback_execution = decoder.decode_image(
                    image,
                    fallback_branches,
                    requested_wave_size=args.wave_size,
                    allow_none=True,
                )
                if len(fallback_decoded) != len(fallback_tasks):
                    raise RuntimeError(
                        "Global fallback address/output count differs: "
                        f"{len(fallback_decoded)} != {len(fallback_tasks)}"
                    )
                for fallback_task, result in zip(fallback_tasks, fallback_decoded):
                    address_task = fallback_task["address_task"]
                    prediction = parse_single_hbb(
                        result["answer"], address_task["label"], width, height
                    )
                    accepted = bool(
                        prediction is not None
                        and point_inside_hbox(
                            address_task["global_point"], prediction["hbox"]
                        )
                    )
                    global_prediction = None
                    if accepted:
                        global_prediction = dict(prediction)
                        global_prediction["point_id"] = address_task["point_id"]
                        global_prediction["witness_point"] = address_task["global_point"]
                        global_prediction["source"] = "pixel_pivr_global_fallback_pbd6"
                        raw_predictions.append(global_prediction)
                    refinements[fallback_task["refinement_index"]].update(
                        {
                            "global_fallback_result": result,
                            "global_fallback_prediction": prediction,
                            "global_fallback_accepted": accepted,
                            "accepted": accepted,
                            "prediction": global_prediction,
                        }
                    )
                execution = {
                    **execution,
                    "local_pass": execution,
                    "global_fallback_pass": fallback_execution,
                    "global_fallback_addresses": len(fallback_branches),
                    "model_seconds": float(execution["model_seconds"])
                    + float(fallback_execution["model_seconds"]),
                    "end_to_end_seconds": float(execution["end_to_end_seconds"])
                    + float(fallback_execution["end_to_end_seconds"]),
                    "peak_gpu_mb": max(
                        float(execution["peak_gpu_mb"]),
                        float(fallback_execution["peak_gpu_mb"]),
                    ),
                }
            else:
                execution["global_fallback_addresses"] = 0
            predictions = class_nms(raw_predictions, args.nms_iou)
            metric = None
            if task == "detection" and "gt" in row:
                tp, fp, fn = detection_counts(ground_truth, predictions)
                metric = {"tp": tp, "fp": fp, "fn": fn}
            elif task == "grounding":
                # The official single-object protocol scores the first valid box.
                prediction = predictions[0] if predictions else None
                overlap = (
                    hbox_iou(prediction["hbox"], ground_truth[0]["hbox"])
                    if prediction is not None and ground_truth
                    else 0.0
                )
                predictions = predictions[:1]
                metric = {"iou": overlap}
            output_row = {
                "sample_key": sample_key,
                "image_id": image_id,
                "task": task,
                "image": str(image_path),
                "classes": classes,
                "point_query_output": point_answer,
                "points": points,
                "refinements": refinements,
                "predictions": predictions,
                "metric": metric,
            }
            append_jsonl(
                predictions_path,
                output_row,
            )
            append_jsonl(
                execution_path,
                {
                    "sample_key": sample_key,
                    "image_id": image_id,
                    "point_stage_seconds": point_seconds,
                    **execution,
                },
            )
            total_point_seconds += point_seconds
            total_refinement_seconds += float(execution["end_to_end_seconds"])
            tasks_seen[task] += 1
            update_metrics(
                output_row,
                detection_totals,
                pointing_totals,
                grounding_totals,
            )
            if index % args.log_every == 0:
                print(
                    f"images={index} addresses={len(points)} predictions={len(predictions)}",
                    flush=True,
                )

    summary: dict[str, Any] = {
        "schema_version": "pixel-pivr-standalone-inference-v2",
        "images": count_jsonl(predictions_path),
        "model": str(args.model.resolve()),
        "adapter": str(args.adapter.resolve()),
        "allow_unsynchronized_adapter": bool(args.allow_unsynchronized_adapter),
        "wave_size": args.wave_size,
        "prefix_cache_mode": args.prefix_cache_mode,
        "geometry_prefix_mode": args.geometry_prefix_mode,
        "crop_side": args.crop_side,
        "local_resize_side": args.local_resize_side,
        "image_token_limit": args.image_token_limit,
        "visual_context": args.visual_context,
        "point_address_prompt_schema": args.point_address_prompt_schema,
        "global_fallback": bool(args.global_fallback),
        "fallback_edge_margin": float(args.fallback_edge_margin),
        "magnified_roi_pixels": args.magnified_roi_pixels,
        "magnified_roi_stride": args.magnified_roi_stride,
        "multiscale_roi_pixels": list(args.multiscale_roi_pixels),
        "multiscale_target_patches": args.multiscale_target_patches,
        "prediction_cap": None,
        "nms_iou": args.nms_iou,
        "seed": int(args.seed),
        "tasks": dict(sorted(tasks_seen.items())),
        "point_stage_seconds": total_point_seconds,
        "refinement_seconds": total_refinement_seconds,
    }
    if detection_totals:
        summary["detection_iou_0_5"] = {
            **dict(detection_totals),
            **precision_recall_f1(
                detection_totals["tp"], detection_totals["fp"], detection_totals["fn"]
            ),
        }
    if grounding_totals:
        queries = int(grounding_totals["queries"])
        summary["grounding"] = {
            "queries": queries,
            "acc_0_5": grounding_totals["acc_0_5"] / queries,
            "acc_0_7": grounding_totals["acc_0_7"] / queries,
            "mean_iou": grounding_totals["iou_sum"] / queries,
        }
    if pointing_totals:
        summary["pointing_containment"] = {
            **dict(pointing_totals),
            **precision_recall_f1(
                pointing_totals["tp"], pointing_totals["fp"], pointing_totals["fn"]
            ),
        }
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--wave-size", type=int, default=1)
    parser.add_argument(
        "--prefix-cache-mode",
        choices=("shared", "recompute"),
        default="shared",
        help="Share one global Qwen KV prefix per image or recompute it per wave",
    )
    parser.add_argument(
        "--geometry-prefix-mode",
        choices=("ref", "box_only"),
        default="ref",
        help=(
            "Use legacy forced <ref> context or predict the PBD6 block directly "
            "after the prompt. Magnified-v2 requires box_only."
        ),
    )
    parser.add_argument("--crop-side", type=int, default=384)
    parser.add_argument(
        "--local-resize-side",
        type=int,
        default=None,
        help=(
            "Resize each point-centred pixel crop to this square side before "
            "MoonViT; predictions are mapped back through the inverse scale."
        ),
    )
    parser.add_argument("--image-token-limit", type=int, default=1024)
    parser.add_argument(
        "--visual-context",
        choices=(
            "pixel_reencoded",
            "preprojector_magnified_roi",
            ADAPTIVE_CONTEXT,
        ),
        default="pixel_reencoded",
    )
    parser.add_argument("--magnified-roi-pixels", type=int, default=380)
    parser.add_argument(
        "--magnified-roi-stride", type=int, choices=(1, 2), default=1
    )
    parser.add_argument(
        "--multiscale-roi-pixels", type=parse_int_csv, default=(196, 378, 756)
    )
    parser.add_argument("--multiscale-target-patches", type=int, default=27)
    parser.add_argument("--multiscale-fusion-hidden", type=int, default=128)
    parser.add_argument("--multiscale-preferred-scale", type=int, default=1)
    parser.add_argument("--point-max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--point-address-prompt-schema",
        choices=("compact", "legacy_two_image"),
        default="compact",
        help=(
            "Round-2 language template. 'compact' matches current Pixel-PIVR "
            "training; legacy_two_image is retained only for older adapters."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--allow-none", action="store_true")
    parser.add_argument(
        "--global-fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Retry local None/invalid/crop-edge outputs against the full image. "
            "This is required by the magnified-v2 training contract."
        ),
    )
    parser.add_argument(
        "--fallback-edge-margin",
        type=float,
        default=2.0,
        help="Local-view pixel margin treated as an incomplete crop boundary.",
    )
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--allow-unsynchronized-adapter",
        action="store_true",
        help="Permit a known-invalid legacy distributed adapter for diagnostics only.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()
    if args.global_fallback and not args.allow_none:
        parser.error("--global-fallback requires --allow-none")
    if args.global_fallback and args.visual_context != "pixel_reencoded":
        parser.error("--global-fallback currently requires --visual-context pixel_reencoded")
    if args.global_fallback and args.geometry_prefix_mode != "box_only":
        parser.error("--global-fallback requires --geometry-prefix-mode box_only")
    if args.global_fallback and not args.local_resize_side:
        parser.error("--global-fallback requires an explicit --local-resize-side")
    if not 0 <= args.multiscale_preferred_scale < len(args.multiscale_roi_pixels):
        parser.error("--multiscale-preferred-scale is outside --multiscale-roi-pixels")
    for path in (args.model, args.adapter, args.manifest):
        if not path.exists() and not path.is_symlink():
            parser.error(f"Missing input: {path}")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
