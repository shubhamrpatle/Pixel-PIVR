# Pixel-PIVR Magnified-v2 Data Contract

Every annotation is one JSON object per line with `conversations`, `image`, and
`meta`. Coordinates are integer-normalized to `[0, 1000]`. The package contains
HBB detection, phrase grounding, and pointing; OBB records are rejected.

## Round 1: global point discovery

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nPoint to all instances that match the following categories: ship</c>vehicle."
    },
    {
      "from": "gpt",
      "value": "<ref>ship</ref><box><321><418></box><ref>vehicle</ref><box>None</box>"
    }
  ],
  "image": "images/train/ab/abcdef...png",
  "meta": {
    "pivr_route": "global_point_discovery",
    "geometry_mode": "point",
    "coordinate_space": "normalized_0_1000",
    "target_count": 1,
    "image_content_sha256": "abcdef..."
  }
}
```

Every queried category has one `<ref>` group. Present objects are represented by
box-center points. Verified absent categories use exactly `<box>None</box>`.
`target_count` counts points, not None groups. Positive all-instance queries must
contain every annotated instance of each requested class.

## Round 2: local-first pixel re-entry

The second image is a virtual image descriptor, not a stored crop. The patched
Eagle loader reads the content-addressed global source image, extracts exactly
144 x 144 source pixels, resizes them to 384 x 384 with Lanczos, and supplies
both images to LocateAnything. The unchanged LA processor subsequently aligns
384 to 392 pixels, resulting in 28 x 28 MoonViT patches and 196 projected local
tokens after its native 2 x 2 merge.

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "<image><image>\nLocate the single ship containing point <box><500><500></box> in the local view. Return its complete horizontal box, or None if the complete boundary is unavailable."
    },
    {
      "from": "gpt",
      "value": "<box><420><455><610><580></box>"
    }
  ],
  "image": [
    "images/train/ab/abcdef...png",
    {
      "virtual_crop": true,
      "schema": "pixel-pivr-virtual-crop-v1",
      "path": "images/train/ab/abcdef...png",
      "crop_xyxy": [210, 348, 354, 492],
      "resize_hw": [384, 384],
      "resample": "lanczos"
    }
  ],
  "meta": {
    "pivr_route": "point_indexed_visual_reentry",
    "pivr_round2_route": "local_first",
    "pivr_visual_context": "global_plus_pixel_crop_144to384_local_gate",
    "pivr_coordinate_frame": "crop_local_normalized_0_1000",
    "pivr_local_point": [500, 500],
    "pivr_local_target_box": [420, 455, 610, 580],
    "pivr_target_fully_contained": true,
    "pivr_containment_tolerance_pixels": 0.0,
    "pivr_fallback_edge_margin_pixels": 2.0,
    "pivr_target_touches_local_fallback_margin": false,
    "target_count": 1
  }
}
```

Round 2 predicts only the native six-token box block. It does not regenerate the
known class label:

```text
positive: <box><x1><y1><x2><y2></box>
negative: <box>None</box>
```

For a positive local row, the complete GT boundary must lie inside the 144-pixel
crop with zero tolerance, and the local point must lie inside the transformed
box. Coordinates are computed analytically in the local frame; no coordinate is
clipped to make an overflowing target appear valid.

## Observable fallback supervision

A fixed 144-pixel view cannot contain every large object. Routing directly from
GT object size would leak privileged information at inference. Magnified-v2
therefore always teaches the local decision first:

- Fully contained interior positive: local input returns one local HBB and no
  fallback row.
- Fully contained edge positive: local input retains the exact local HBB and is
  paired with a global positive. This teaches the same two-pixel safety-margin
  retry used at inference without discarding valid local supervision.
- Overflowing positive: local input returns `<box>None</box>`, paired with a
  global point-to-box row returning the complete global HBB.
- Negative address: local input returns `<box>None</box>`, paired with a global
  row also returning `<box>None</box>`.

The paired global row has one image and a prompt explicitly scoped to the global
frame. At inference, only observable outcomes trigger retry: local None, invalid
geometry, a box not containing the address, or a box touching the crop edge.

## Recipes and split safety

```text
recipes/stage1_coarse.json
recipes/stage2_dense_balanced.json
recipes/validation_all_tasks.json
recipes/validation_monitor_all_tasks.json
```

The fixed monitor combines all 1,000 independent Round-1 validation records with
a task/polarity-controlled Round-2 monitor derived only from the same validation
image pool. It is used for loss and checkpoint selection, never gradients.

Split assignment is by source-image SHA-256. The package verifier requires zero
train/validation/test image-hash overlap, exact local/global fallback pairing,
all image paths present, exact recipe counts, and complete checksum coverage.
Stage 1 and Stage 2 may intentionally reuse train images and records as replay;
that is curriculum overlap, not benchmark leakage.
