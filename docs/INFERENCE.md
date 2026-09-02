# Standalone Inference Manifest

Each line identifies one image and the explicitly queried classes:

```json
{"image_id":"scene-1","image":"images/scene-1.png","classes":["ship","vehicle"]}
```

Pixel-PIVR first generates typed points, then refines only those points. To reuse a
frozen point scaffold, provide pixel-coordinate points:

```json
{"image_id":"scene-1","image":"images/scene-1.png","classes":["ship"],"points":[{"point_id":"p1","label":"ship","point":[314.0,221.0]}]}
```

Set `point_coordinate_space` to `normalized_0_1000` when points are normalized.

Optional pixel-coordinate HBB GT enables class-aware one-to-one IoU-0.5 diagnostic
metrics:

```json
{"image_id":"scene-1","image":"images/scene-1.png","classes":["ship"],"gt":[{"label":"ship","hbox":[280,190,350,250]}]}
```

The evaluator applies no prediction cap. It retains only HBBs containing their
address point and then applies class-wise NMS.

For single-encode magnified local features, use
`--visual-context preprojector_magnified_roi`. In this mode the point and decoded
HBB remain in full-image coordinates; `--crop-side` is ignored. The 380-pixel
request maps to a 378-pixel pre-merge MoonViT field and 676 stride-1 local tokens.

## Shared-prefix waves

`--prefix-cache-mode shared` is the default. The global image passes through
MoonViT/projector and the Qwen prefix once per image. Each point contributes a
local visual/text suffix, and each branch emits one PBD6 geometry block. A wave
size is an execution bound, not an object-count cap: 1,000 addresses with
`--wave-size 200` execute as five waves.

```bash
pixel-pivr-infer \
  --model /path/to/LocateAnything-3B \
  --adapter /path/to/best.pt \
  --manifest /path/eval.jsonl \
  --output /path/shared-wave-output \
  --prefix-cache-mode shared \
  --wave-size 200
```

For a controlled equivalence/latency comparison, rerun the same manifest with
`--prefix-cache-mode recompute`. This historical path caches projected MoonViT
features but repeats the full Qwen prefix for every active branch.
