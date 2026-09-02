"""Point-centred local context from MoonViT features before 2x2 merging.

LocateAnything normally groups non-overlapping 2x2 MoonViT cells before its
multimodal projector.  Magnified PIVR keeps that path for the global image but
also gathers a point-centred window from the unmerged MoonViT grid.  Overlapping
2x2 windows (stride 1) are sent through the *same* released projector, providing
denser local evidence without another pixel or vision-encoder pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch


@dataclass(frozen=True)
class MagnifiedROI:
    """A projected local ROI and the spatial policy used to construct it."""

    features: torch.Tensor
    patch_grid_hw: tuple[int, int]
    window_patch_hw: tuple[int, int]
    window_yx: tuple[int, int]
    center_rc: tuple[int, int]
    projector_grid_hw: tuple[int, int]
    requested_pixels: int
    effective_pixels_hw: tuple[int, int]
    stride: int


@dataclass(frozen=True)
class MoonViTFeatureCache:
    """One-image MoonViT cache before and after LA's native projector."""

    unmerged: torch.Tensor
    projected_global: torch.Tensor
    patch_grid_hw: tuple[int, int]


def normalized_grid_index(value: float, size: int) -> int:
    if size <= 0:
        raise ValueError(f"Invalid grid size: {size}")
    clipped = max(0.0, min(1000.0, float(value)))
    return min(size - 1, int(clipped * size / 1000.0))


def merge_2x2(features: torch.Tensor, grid_hw: Sequence[int]) -> torch.Tensor:
    """Exactly reproduce MoonViT's row-major non-overlapping 2x2 merger."""
    if features.ndim != 2:
        raise ValueError(f"Expected [tokens, hidden], got {tuple(features.shape)}")
    height, width = (int(value) for value in grid_hw)
    if height <= 0 or width <= 0 or height % 2 or width % 2:
        raise ValueError(f"A positive even patch grid is required, got {(height, width)}")
    if int(features.shape[0]) != height * width:
        raise ValueError(
            f"Feature/grid mismatch: {features.shape[0]} != {height}*{width}"
        )
    hidden = int(features.shape[-1])
    grid = features.view(height // 2, 2, width // 2, 2, hidden)
    return (
        grid.permute(0, 2, 1, 3, 4)
        .contiguous()
        .view((height // 2) * (width // 2), 4 * hidden)
    )


def sliding_2x2(
    grid: torch.Tensor,
    *,
    stride: int,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Flatten spatially ordered 2x2 windows for LA's unchanged projector."""
    if grid.ndim != 3:
        raise ValueError(f"Expected [height,width,hidden], got {tuple(grid.shape)}")
    height, width, hidden = (int(value) for value in grid.shape)
    step = int(stride)
    if step <= 0:
        raise ValueError(f"Stride must be positive, got {step}")
    if height < 2 or width < 2:
        raise ValueError(f"ROI must contain at least 2x2 patches, got {(height, width)}")
    # Tensor.unfold places the hidden dimension before the two kernel axes.
    windows = grid.unfold(0, 2, step).unfold(1, 2, step)
    out_h, out_w = int(windows.shape[0]), int(windows.shape[1])
    flattened = (
        windows.permute(0, 1, 3, 4, 2)
        .contiguous()
        .view(out_h * out_w, 4 * hidden)
    )
    return flattened, (out_h, out_w)


def encode_moonvit_unmerged(
    vision_model: Any,
    pixel_values: torch.Tensor,
    image_grid_hws: torch.Tensor,
) -> torch.Tensor:
    """Run MoonViT once and stop immediately before its fixed patch merger."""
    required = ("patch_embed", "encoder", "merge_kernel_size", "patch_size")
    missing = [name for name in required if not hasattr(vision_model, name)]
    if missing:
        raise TypeError(
            "The loaded vision model does not expose the LocateAnything MoonViT "
            f"interface; missing {missing}"
        )
    hidden = vision_model.patch_embed(pixel_values, image_grid_hws)
    return vision_model.encoder(hidden, image_grid_hws)


def encode_global_cache(
    model: Any,
    pixel_values: torch.Tensor,
    image_grid_hws: torch.Tensor,
) -> MoonViTFeatureCache:
    """Encode one global image and cache both unmerged and standard global tokens."""
    grid = torch.as_tensor(image_grid_hws, dtype=torch.long, device=pixel_values.device)
    if grid.ndim != 2 or tuple(grid.shape) != (1, 2):
        raise ValueError(f"Expected exactly one image grid [1,2], got {tuple(grid.shape)}")
    height, width = (int(value) for value in grid[0].tolist())
    unmerged = encode_moonvit_unmerged(model.vision_model, pixel_values, grid)
    if tuple(unmerged.shape[:1]) != (height * width,):
        raise RuntimeError(
            f"MoonViT cache/grid mismatch: {unmerged.shape[0]} != {height}*{width}"
        )
    global_groups = merge_2x2(unmerged, (height, width))
    projected_global = model.mlp1(global_groups)
    return MoonViTFeatureCache(
        unmerged=unmerged,
        projected_global=projected_global,
        patch_grid_hw=(height, width),
    )


def extract_magnified_preprojector_roi(
    cache: MoonViTFeatureCache,
    point_normalized_xy: Sequence[float] | torch.Tensor,
    projector: Any,
    *,
    requested_pixels: int = 380,
    patch_size: int = 14,
    stride: int = 1,
) -> MagnifiedROI:
    """Project an unpadded point-centred pre-merge ROI with overlapping windows."""
    requested = int(requested_pixels)
    patch = int(patch_size)
    if requested <= 0 or patch <= 0:
        raise ValueError("requested_pixels and patch_size must be positive")
    point = torch.as_tensor(point_normalized_xy, dtype=torch.float32).flatten()
    if point.numel() != 2:
        raise ValueError(f"Expected normalized (x,y), got {point.tolist()}")
    grid_h, grid_w = cache.patch_grid_hw
    if int(cache.unmerged.shape[0]) != grid_h * grid_w:
        raise ValueError("Cached unmerged features do not match their patch grid")

    # Nearest patch count makes 380 px become 27 MoonViT cells = 378 px.
    desired_side = max(2, int(round(requested / patch)))
    window_h = min(desired_side, grid_h)
    window_w = min(desired_side, grid_w)
    row = normalized_grid_index(float(point[1]), grid_h)
    col = normalized_grid_index(float(point[0]), grid_w)
    start_y = min(max(row - window_h // 2, 0), grid_h - window_h)
    start_x = min(max(col - window_w // 2, 0), grid_w - window_w)
    grid = cache.unmerged.view(grid_h, grid_w, cache.unmerged.shape[-1])
    roi_grid = grid[start_y : start_y + window_h, start_x : start_x + window_w]
    groups, projector_grid = sliding_2x2(roi_grid, stride=int(stride))
    expected_input = None
    for module in projector.modules() if hasattr(projector, "modules") else ():
        if isinstance(module, torch.nn.Linear):
            expected_input = int(module.in_features)
            break
    if expected_input is not None and int(groups.shape[-1]) != expected_input:
        raise RuntimeError(
            "Pre-projector ROI width differs from LA projector input: "
            f"{groups.shape[-1]} != {expected_input}"
        )
    projected = projector(groups)
    return MagnifiedROI(
        features=projected,
        patch_grid_hw=(grid_h, grid_w),
        window_patch_hw=(window_h, window_w),
        window_yx=(start_y, start_x),
        center_rc=(row, col),
        projector_grid_hw=projector_grid,
        requested_pixels=requested,
        effective_pixels_hw=(window_h * patch, window_w * patch),
        stride=int(stride),
    )


def insert_virtual_image(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    image_token_id: int,
    image_start_id: int,
    image_end_id: int,
    visual_tokens: int,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Insert a local virtual-image span after the real global image."""
    if any(value.ndim != 1 for value in (input_ids, labels, position_ids)):
        raise ValueError("Virtual-image insertion expects unbatched 1D tensors")
    positions = torch.where(input_ids.eq(int(image_token_id)))[0]
    if not positions.numel():
        raise RuntimeError("No global image placeholders were found")
    last_global = int(positions[-1])
    if last_global + 1 >= input_ids.numel() or int(input_ids[last_global + 1]) != int(
        image_end_id
    ):
        raise RuntimeError("Global image placeholders are not followed by </img>")
    insert_at = last_global + 2
    count = int(visual_tokens)
    if count <= 0:
        raise ValueError("A virtual local image must contain at least one token")
    virtual_ids = torch.cat(
        (
            input_ids.new_tensor([int(image_start_id)]),
            input_ids.new_full((count,), int(image_token_id)),
            input_ids.new_tensor([int(image_end_id)]),
        )
    )
    ignored = labels.new_full((count + 2,), int(ignore_index))
    first_position = int(position_ids[insert_at - 1]) + 1
    virtual_positions = torch.arange(
        first_position,
        first_position + count + 2,
        device=position_ids.device,
        dtype=position_ids.dtype,
    )
    tail_positions = position_ids[insert_at:].clone()
    tail_positions[tail_positions >= first_position] += count + 2
    return (
        torch.cat((input_ids[:insert_at], virtual_ids, input_ids[insert_at:])),
        torch.cat((labels[:insert_at], ignored, labels[insert_at:])),
        torch.cat((position_ids[:insert_at], virtual_positions, tail_positions)),
    )
