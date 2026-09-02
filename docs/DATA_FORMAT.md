# Pixel-PIVR HBB Data Contract

Pixel-PIVR uses LocateAnything conversation JSONL. Every row contains
`conversations`, `image`, and `meta`.

## Global point discovery

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
  "image": "images/example.png",
  "meta": {
    "pivr_route": "global_point_discovery",
    "geometry_mode": "point",
    "coordinate_space": "normalized_0_1000",
    "target_count": 1,
    "image_content_sha256": "..."
  }
}
```

Every queried class has one `<ref>` group. Present instances are represented by
box-center points. Verified absent classes use the released
`<box>None</box>` convention. `target_count` counts points, not None groups.

## Pixel re-entry

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "<image><image>\nImage 1 is the global scene. Image 2 is the point-indexed local view for address A1. In Image 2, point <box><500><500></box> addresses one ship. Return exactly one box for that ship in Image 2 coordinates. If no ship contains the address point, return None."
    },
    {
      "from": "gpt",
      "value": "<ref>ship</ref><box><420><455><610><580></box>"
    }
  ],
  "image": ["images/example.png", "crops/example_A1.png"],
  "meta": {
    "pivr_route": "point_indexed_visual_reentry",
    "geometry_mode": "hbb",
    "coordinate_space": "normalized_0_1000",
    "pivr_local_point": [500, 500],
    "pivr_local_target_box": [420, 455, 610, 580],
    "pivr_target_fully_contained": true,
    "target_count": 1,
    "image_content_sha256": "..."
  }
}
```

The local crop is lossless. Its point and HBB are expressed in the local crop's
normalized coordinate system. Positive re-entry has exactly one HBB containing the
point. Negative re-entry has exactly one `<box>None</box>` and `target_count: 0`.

## Magnified pre-projector re-entry

The experimental `preprojector_magnified_roi` mode uses one global image rather
than a two-image list. Its prompt and answer use full-image normalized coordinates,
and metadata supplies `pivr_global_point`. No crop file is created or loaded:

```json
{
  "conversations": [
    {"from": "human", "value": "<image>\nLocate the single ship containing point <box><500><500></box> in horizontal box format. Return None if absent."},
    {"from": "gpt", "value": "<ref>ship</ref><box><470><480><540><530></box>"}
  ],
  "image": "images/example.png",
  "meta": {
    "pivr_route": "point_indexed_visual_reentry",
    "pivr_global_point": [500, 500],
    "coordinate_space": "normalized_0_1000",
    "geometry_mode": "hbb",
    "pivr_target_fully_contained": true,
    "target_count": 1
  }
}
```

## Split safety

- Split by source-image content hash, not record ID or crop filename.
- No training image hash may appear in validation or public benchmark evaluation.
- Do not generate one global point record with omitted same-class instances.
- Do not silently truncate dense point or box targets.
- Run both structural and exact-loader audits before training.
