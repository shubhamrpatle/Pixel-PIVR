"""HBB/point grammar, coordinate conversion, NMS, and diagnostic metrics."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Mapping, Sequence


REF_POINT_OR_BOX = re.compile(
    r"<ref>\s*(.*?)\s*</ref>|"
    r"<box>\s*<(\d+)>\s*<(\d+)>"
    r"(?:\s*<(\d+)>\s*<(\d+)>)?\s*</box>",
    re.IGNORECASE | re.DOTALL,
)


def canonical_label(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").split())


def point_to_norm(point: Sequence[float], width: int, height: int) -> list[int]:
    return [
        max(0, min(1000, int(round(float(point[0]) / max(1, width) * 1000)))),
        max(0, min(1000, int(round(float(point[1]) / max(1, height) * 1000)))),
    ]


def point_centered_crop(
    point: Sequence[float], width: int, height: int, side: int
) -> tuple[int, int, int, int]:
    side = max(1, min(int(side), width, height))
    x1 = int(round(float(point[0]) - side / 2.0))
    y1 = int(round(float(point[1]) - side / 2.0))
    x1 = max(0, min(width - side, x1))
    y1 = max(0, min(height - side, y1))
    return x1, y1, x1 + side, y1 + side


def resize_local_point(
    point: Sequence[float],
    source_size: Sequence[int],
    output_size: Sequence[int],
) -> list[float]:
    """Map a point from a source crop into its resized local-view frame."""
    source_width, source_height = (max(1, int(value)) for value in source_size)
    output_width, output_height = (max(1, int(value)) for value in output_size)
    return [
        float(point[0]) * output_width / source_width,
        float(point[1]) * output_height / source_height,
    ]


def local_hbox_to_global(
    hbox: Sequence[float],
    crop_xyxy: Sequence[int],
    local_size: Sequence[int],
) -> list[float]:
    """Undo local-view resizing and translate an HBB into image coordinates."""
    left, top, right, bottom = (int(value) for value in crop_xyxy)
    local_width, local_height = (max(1, int(value)) for value in local_size)
    scale_x = (right - left) / float(local_width)
    scale_y = (bottom - top) / float(local_height)
    return [
        float(hbox[0]) * scale_x + left,
        float(hbox[1]) * scale_y + top,
        float(hbox[2]) * scale_x + left,
        float(hbox[3]) * scale_y + top,
    ]


def point_address_prompt(
    label: str, point: Sequence[float], width: int, height: int
) -> str:
    x, y = point_to_norm(point, width, height)
    return (
        "Image 1 is the global scene. Image 2 is the point-indexed local view "
        "for address A1. "
        f"In Image 2, point <box><{x}><{y}></box> addresses one {label}. "
        f"Return exactly one box for that {label} in Image 2 coordinates. "
        f"If no {label} contains the address point, return None."
    )


def cached_feature_point_prompt(
    label: str, point: Sequence[float], width: int, height: int
) -> str:
    """Magnified-v2 local-completeness prompt for pixel re-encoding."""
    x, y = point_to_norm(point, width, height)
    return (
        f"Locate the single {label} containing point <box><{x}><{y}></box> "
        "in the local view. Return its complete horizontal box, or None if "
        "the complete boundary is unavailable."
    )


def pilot_compact_point_prompt(
    label: str, point: Sequence[float], width: int, height: int
) -> str:
    """Exact Round-2 prompt used by the synchronized 16K/4K pilot."""
    x, y = point_to_norm(point, width, height)
    return (
        f"Locate the single {label} containing point <box><{x}><{y}></box> "
        "in horizontal box format. Return None if absent."
    )


def global_fallback_point_prompt(
    label: str, point: Sequence[float], width: int, height: int
) -> str:
    """Training-matched retry prompt in full-image coordinates."""
    x, y = point_to_norm(point, width, height)
    return (
        f"Locate the single {label} containing point <box><{x}><{y}></box> "
        "in the global image. Return its horizontal box, or None if absent."
    )


def box_touches_frame(
    hbox: Sequence[float], width: int, height: int, margin: float
) -> bool:
    """Return whether a predicted boundary is too close to a crop boundary."""
    if len(hbox) != 4:
        raise ValueError(f"Expected HBB [x1, y1, x2, y2], got {hbox!r}")
    margin = max(0.0, float(margin))
    x1, y1, x2, y2 = map(float, hbox)
    return (
        x1 <= margin
        or y1 <= margin
        or x2 >= float(width) - margin
        or y2 >= float(height) - margin
    )


def discovery_prompt(classes: Sequence[str]) -> str:
    labels = [canonical_label(value) for value in classes]
    if not labels or any(not value for value in labels):
        raise ValueError("Point discovery requires non-empty class names")
    return (
        "Point to all instances that match the following categories: "
        + "</c>".join(labels)
        + "."
    )


def parse_labeled_points(
    answer: str, width: int, height: int
) -> list[dict[str, Any]]:
    current_label = ""
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for match in REF_POINT_OR_BOX.finditer(answer):
        if match.group(1) is not None:
            current_label = canonical_label(match.group(1))
            continue
        if not current_label:
            continue
        values = [int(value) for value in match.groups()[1:] if value is not None]
        if len(values) == 2:
            nx, ny = values
            source = "point"
        elif len(values) == 4:
            nx = int(round((values[0] + values[2]) / 2))
            ny = int(round((values[1] + values[3]) / 2))
            source = "box_center_fallback"
        else:
            continue
        nx, ny = max(0, min(1000, nx)), max(0, min(1000, ny))
        key = (current_label, nx, ny)
        if key in seen:
            continue
        seen.add(key)
        output.append(
            {
                "label": current_label,
                "point": [nx / 1000.0 * width, ny / 1000.0 * height],
                "point_norm": [nx, ny],
                "source": source,
                "point_id": f"{current_label}:{nx}:{ny}",
            }
        )
    return output


def parse_single_hbb(
    answer: str,
    expected_label: str,
    width: int,
    height: int,
) -> dict[str, Any] | None:
    current_label = ""
    boxes: list[dict[str, Any]] = []
    for match in REF_POINT_OR_BOX.finditer(answer):
        if match.group(1) is not None:
            current_label = canonical_label(match.group(1))
            continue
        values = [int(value) for value in match.groups()[1:] if value is not None]
        if len(values) != 4 or current_label != canonical_label(expected_label):
            continue
        x1, x2 = sorted((max(0, min(1000, values[0])), max(0, min(1000, values[2]))))
        y1, y2 = sorted((max(0, min(1000, values[1])), max(0, min(1000, values[3]))))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append(
            {
                "label": current_label,
                "hbox": [
                    x1 / 1000.0 * width,
                    y1 / 1000.0 * height,
                    x2 / 1000.0 * width,
                    y2 / 1000.0 * height,
                ],
                "score": 1.0,
            }
        )
    if len(boxes) > 1:
        raise RuntimeError(
            f"Atomic Pixel-PIVR branch emitted {len(boxes)} valid HBBs"
        )
    return boxes[0] if boxes else None


def point_inside_hbox(point: Sequence[float], box: Sequence[float]) -> bool:
    return (
        min(box[0], box[2]) <= point[0] <= max(box[0], box[2])
        and min(box[1], box[3]) <= point[1] <= max(box[1], box[3])
    )


def hbox_iou(left: Sequence[float], right: Sequence[float]) -> float:
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def class_nms(
    predictions: Sequence[Mapping[str, Any]], threshold: float
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        grouped[canonical_label(prediction.get("label"))].append(dict(prediction))
    kept: list[dict[str, Any]] = []
    for rows in grouped.values():
        rows.sort(key=lambda row: float(row.get("score", 1.0)), reverse=True)
        while rows:
            chosen = rows.pop(0)
            kept.append(chosen)
            rows = [
                row
                for row in rows
                if hbox_iou(chosen["hbox"], row["hbox"]) < threshold
            ]
    return kept


def detection_counts(
    ground_truth: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    iou_threshold: float = 0.5,
) -> tuple[int, int, int]:
    matched: set[int] = set()
    tp = fp = 0
    for prediction in predictions:
        candidates = []
        label = canonical_label(prediction.get("label"))
        for index, target in enumerate(ground_truth):
            if index in matched or canonical_label(target.get("label")) != label:
                continue
            overlap = hbox_iou(prediction["hbox"], target["hbox"])
            if overlap >= iou_threshold:
                candidates.append((overlap, index))
        if candidates:
            _, index = max(candidates)
            matched.add(index)
            tp += 1
        else:
            fp += 1
    return tp, fp, len(ground_truth) - len(matched)


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}
