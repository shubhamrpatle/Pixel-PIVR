"""Deterministic pixel-crop records for full-scale Pixel-PIVR training."""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence


SOURCE_CROP_SIDE = 144
LOCAL_INPUT_SIDE = 384
CONTAINMENT_TOLERANCE_PIXELS = 0.0
FALLBACK_EDGE_MARGIN_PIXELS = 2.0
VIRTUAL_CROP_SCHEMA = "pixel-pivr-virtual-crop-v1"
PROMPT_SCHEMA = "local-completeness-gated-point-box-v3"
ANSWER_SCHEMA = "pixel-pivr-box-only-hbb-or-explicit-None-v2"


def norm_to_pixel(value: int, extent: int) -> float:
    return float(value) / 1000.0 * float(extent)


def pixel_to_norm(value: float, extent: int) -> int:
    normalized = int(round(float(value) / max(1.0, float(extent)) * 1000.0))
    return max(0, min(1000, normalized))


def point_centered_crop(
    point_norm: Sequence[int], width: int, height: int, side: int = SOURCE_CROP_SIDE
) -> tuple[int, int, int, int]:
    if width < side or height < side:
        raise ValueError(
            f"Image {(width, height)} is smaller than the required {side}x{side} crop"
        )
    center_x = norm_to_pixel(int(point_norm[0]), width)
    center_y = norm_to_pixel(int(point_norm[1]), height)
    left = int(round(center_x - side / 2.0))
    top = int(round(center_y - side / 2.0))
    left = max(0, min(width - side, left))
    top = max(0, min(height - side, top))
    return left, top, left + side, top + side


def target_fits_crop(
    target_norm: Sequence[int], crop: Sequence[int], width: int, height: int
) -> bool:
    left, top, right, bottom = map(int, crop)
    x1, y1, x2, y2 = (
        norm_to_pixel(int(target_norm[0]), width),
        norm_to_pixel(int(target_norm[1]), height),
        norm_to_pixel(int(target_norm[2]), width),
        norm_to_pixel(int(target_norm[3]), height),
    )
    return (
        min(x1, x2) >= left
        and max(x1, x2) <= right
        and min(y1, y2) >= top
        and max(y1, y2) <= bottom
    )


def geometry_in_crop(
    point_norm: Sequence[int],
    target_norm: Sequence[int] | None,
    crop: Sequence[int],
    width: int,
    height: int,
) -> tuple[list[int], list[int] | None]:
    left, top, right, bottom = map(int, crop)
    crop_width = right - left
    crop_height = bottom - top
    point = [
        pixel_to_norm(norm_to_pixel(int(point_norm[0]), width) - left, crop_width),
        pixel_to_norm(norm_to_pixel(int(point_norm[1]), height) - top, crop_height),
    ]
    if target_norm is None:
        return point, None
    target_pixels = [
        norm_to_pixel(int(target_norm[0]), width) - left,
        norm_to_pixel(int(target_norm[1]), height) - top,
        norm_to_pixel(int(target_norm[2]), width) - left,
        norm_to_pixel(int(target_norm[3]), height) - top,
    ]
    target = [
        pixel_to_norm(target_pixels[0], crop_width),
        pixel_to_norm(target_pixels[1], crop_height),
        pixel_to_norm(target_pixels[2], crop_width),
        pixel_to_norm(target_pixels[3], crop_height),
    ]
    return point, target


def point_inside(point: Sequence[int], box: Sequence[int]) -> bool:
    return (
        int(box[0]) <= int(point[0]) <= int(box[2])
        and int(box[1]) <= int(point[1]) <= int(box[3])
    )


def local_box_touches_fallback_margin(
    box: Sequence[int],
    side: int = LOCAL_INPUT_SIDE,
    margin: float = FALLBACK_EDGE_MARGIN_PIXELS,
) -> bool:
    if len(box) != 4:
        raise ValueError(f"Expected HBB [x1, y1, x2, y2], got {box!r}")
    x1, y1, x2, y2 = (float(value) / 1000.0 * float(side) for value in box)
    return (
        x1 <= margin
        or y1 <= margin
        or x2 >= float(side) - margin
        or y2 >= float(side) - margin
    )


def point_token(point: Sequence[int]) -> str:
    if len(point) != 2:
        raise ValueError(f"Expected point [x, y], got {point!r}")
    return f"<box><{int(point[0])}><{int(point[1])}></box>"


def box_token(box: Sequence[int] | None) -> str:
    if box is None:
        return "<box>None</box>"
    if len(box) != 4:
        raise ValueError(f"Expected HBB [x1, y1, x2, y2], got {box!r}")
    return "<box>" + "".join(f"<{int(value)}>" for value in box) + "</box>"


def compact_prompt(reference: str, point: Sequence[int], route: str) -> str:
    if route == "local":
        return (
            "<image><image>\n"
            f"Locate the single {reference} containing point {point_token(point)} "
            "in the local view. Return its complete horizontal box, or None if "
            "the complete boundary is unavailable."
        )
    if route == "global_fallback":
        return (
            "<image>\n"
            f"Locate the single {reference} containing point {point_token(point)} "
            "in the global image. Return its horizontal box, or None if absent."
        )
    raise ValueError(f"Unknown Round-2 route: {route!r}")


def _record_id(meta: Mapping[str, Any], suffix: str) -> str:
    source = str(meta.get("record_id") or meta.get("pivr_source_record_id") or "")
    if not source:
        raise ValueError("Round-2 record has no record_id")
    return f"{source}__{suffix}"


def transform_round2_records(
    source_row: Mapping[str, Any],
    *,
    portable_image: str,
    width: int,
    height: int,
    source_crop_side: int = SOURCE_CROP_SIDE,
    local_input_side: int = LOCAL_INPUT_SIDE,
) -> list[dict[str, Any]]:
    """Create an observable local-first route and any required global fallback.

    The fixed local crop cannot contain every large target.  Selecting a global
    route directly from GT would be an oracle at inference.  Instead, every source
    row first supervises the local branch. A clipped positive is labelled None,
    meaning that a complete boundary is unavailable, and is paired with a global
    positive. A complete local target touching the inference safety margin keeps
    its exact local target and is also paired with a global positive. Negative
    local rows are paired with global negatives. These pairs calibrate every
    observable retry condition used at inference without hidden-GT routing.
    """
    row = copy.deepcopy(dict(source_row))
    meta = copy.deepcopy(dict(row.get("meta") or {}))
    # Remove fields that describe the superseded cached pre-projector ROI
    # experiment; they are false for real pixel-crop re-encoding.
    for obsolete in (
        "pivr_local_context_source",
        "pivr_local_roi_may_not_cover_full_target",
    ):
        meta.pop(obsolete, None)
    reference = str(meta.get("pivr_reference") or "").strip()
    point_global = meta.get("pivr_global_point")
    target_global = meta.get("pivr_global_target_box")
    if not reference:
        raise ValueError("Round-2 record has no pivr_reference")
    if (
        not isinstance(point_global, (list, tuple))
        or len(point_global) != 2
        or any(not 0 <= int(value) <= 1000 for value in point_global)
    ):
        raise ValueError(f"Invalid global point: {point_global!r}")
    point_global = [int(value) for value in point_global]
    if target_global is not None:
        if (
            not isinstance(target_global, (list, tuple))
            or len(target_global) != 4
            or any(not 0 <= int(value) <= 1000 for value in target_global)
        ):
            raise ValueError(f"Invalid global HBB: {target_global!r}")
        target_global = [int(value) for value in target_global]
        if not point_inside(point_global, target_global):
            raise ValueError(
                f"Address point {point_global} is outside target {target_global}"
            )

    crop = point_centered_crop(point_global, width, height, source_crop_side)
    target_is_local = target_global is None or target_fits_crop(
        target_global, crop, width, height
    )
    local_point, local_target = geometry_in_crop(
        point_global,
        target_global if target_is_local else None,
        crop,
        width,
        height,
    )
    if local_target is not None and not point_inside(local_point, local_target):
        raise ValueError(f"Local point {local_point} is outside local target {local_target}")
    target_touches_edge = bool(
        local_target is not None
        and local_box_touches_fallback_margin(
            local_target, int(local_input_side), FALLBACK_EDGE_MARGIN_PIXELS
        )
    )
    fallback_required = bool(
        target_global is None or not target_is_local or target_touches_edge
    )
    fallback_reason = (
        "negative_address"
        if target_global is None
        else (
            "target_not_fully_contained_in_fixed_144px_crop"
            if not target_is_local
            else "complete_target_touches_local_fallback_margin"
            if target_touches_edge
            else None
        )
    )

    source_record_id = str(meta.get("record_id") or "")
    local = copy.deepcopy(row)
    local_meta = copy.deepcopy(meta)
    local["conversations"] = [
        {"from": "human", "value": compact_prompt(reference, local_point, "local")},
        {"from": "gpt", "value": box_token(local_target)},
    ]
    local["image"] = [
        portable_image,
        {
            "virtual_crop": True,
            "schema": VIRTUAL_CROP_SCHEMA,
            "path": portable_image,
            "crop_xyxy": list(crop),
            "resize_hw": [int(local_input_side), int(local_input_side)],
            "resample": "lanczos",
        },
    ]
    local_meta.update(
        {
            "answer_format": ANSWER_SCHEMA,
            "record_id": _record_id(meta, "local"),
            "pivr_source_record_id": source_record_id,
            "pivr_coordinate_frame": "crop_local_normalized_0_1000",
            "pivr_crop_xyxy_pixels": list(crop),
            "pivr_containment_tolerance_pixels": CONTAINMENT_TOLERANCE_PIXELS,
            "pivr_fallback_edge_margin_pixels": FALLBACK_EDGE_MARGIN_PIXELS,
            "pivr_fallback_required": fallback_required,
            "pivr_fallback_reason": fallback_reason,
            "pivr_global_point": point_global,
            "pivr_global_target_box": target_global,
            "pivr_local_input_size": [int(local_input_side), int(local_input_side)],
            "pivr_local_point": local_point,
            "pivr_local_resample": "PIL.Image.Resampling.LANCZOS",
            "pivr_local_source_size": [int(source_crop_side), int(source_crop_side)],
            "pivr_local_target_box": local_target,
            "pivr_output_geometry_mode": "hbb",
            "pivr_prompt_schema": PROMPT_SCHEMA,
            "pivr_round": 2,
            "pivr_round2_route": "local_first",
            "pivr_source_image_size": [int(width), int(height)],
            "pivr_target_fully_contained": bool(target_is_local),
            "pivr_target_touches_local_fallback_margin": target_touches_edge,
            "pivr_upscale_factor": float(local_input_side) / float(source_crop_side),
            "pivr_view_order": ["global_context", "point_indexed_local"],
            "pivr_visual_context": "global_plus_pixel_crop_144to384_local_gate",
            "pivr_visual_inputs": 2,
            "target_count": int(local_target is not None),
        }
    )
    local["meta"] = local_meta
    outputs = [local]

    if fallback_required:
        fallback = copy.deepcopy(row)
        fallback_meta = copy.deepcopy(meta)
        fallback["conversations"] = [
            {
                "from": "human",
                "value": compact_prompt(reference, point_global, "global_fallback"),
            },
            {"from": "gpt", "value": box_token(target_global)},
        ]
        fallback["image"] = portable_image
        fallback_meta.update(
            {
                "answer_format": ANSWER_SCHEMA,
                "record_id": _record_id(meta, "global_fallback"),
                "pivr_source_record_id": source_record_id,
                "pivr_coordinate_frame": "full_image_normalized_0_1000",
                "pivr_crop_xyxy_pixels": list(crop),
                "pivr_containment_tolerance_pixels": CONTAINMENT_TOLERANCE_PIXELS,
                "pivr_fallback_edge_margin_pixels": FALLBACK_EDGE_MARGIN_PIXELS,
                "pivr_fallback_reason": fallback_reason,
                "pivr_fallback_required": True,
                "pivr_global_point": point_global,
                "pivr_global_target_box": target_global,
                "pivr_local_input_size": [int(local_input_side), int(local_input_side)],
                "pivr_local_point": local_point,
                "pivr_local_resample": "PIL.Image.Resampling.LANCZOS",
                "pivr_local_source_size": [int(source_crop_side), int(source_crop_side)],
                "pivr_local_target_box": local_target,
                "pivr_output_geometry_mode": "hbb",
                "pivr_prompt_schema": PROMPT_SCHEMA,
                "pivr_round": 2,
                "pivr_round2_route": "global_fallback",
                "pivr_source_image_size": [int(width), int(height)],
                "pivr_target_fully_contained": bool(target_is_local),
                "pivr_target_touches_local_fallback_margin": target_touches_edge,
                "pivr_upscale_factor": float(local_input_side) / float(source_crop_side),
                "pivr_view_order": ["global_context"],
                "pivr_visual_context": "global_only_point_box_fallback",
                "pivr_visual_inputs": 1,
                "target_count": int(target_global is not None),
            }
        )
        fallback["meta"] = fallback_meta
        outputs.append(fallback)

    return outputs


def transform_round2(
    source_row: Mapping[str, Any],
    *,
    portable_image: str,
    width: int,
    height: int,
    source_crop_side: int = SOURCE_CROP_SIDE,
    local_input_side: int = LOCAL_INPUT_SIDE,
) -> dict[str, Any]:
    """Compatibility helper returning the mandatory local-first record."""
    return transform_round2_records(
        source_row,
        portable_image=portable_image,
        width=width,
        height=height,
        source_crop_side=source_crop_side,
        local_input_side=local_input_side,
    )[0]
