# Standalone Inference Manifest

Each line identifies one sample, task, image, and explicitly queried classes:

```json
{"sample_key":"dota:1","task":"detection","image_id":"scene-1","image":"images/scene-1.png","classes":["ship","vehicle"]}
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
address point and then applies class-wise NMS. In the release configuration, a
local None, invalid/non-containing box, or box within two pixels of the crop edge
triggers one global point-to-box retry. The training corpus contains a paired
global row for every GT condition that invokes this policy, including complete
local boxes inside that safety margin.

Grounding rows set `task` to `grounding`, use the phrase itself as the single
reference in `classes`, and provide the exact Round-1 point prompt through
`point_prompt`. The official single-object summary reports first-box Acc@0.5,
Acc@0.7, and mIoU. Pointing rows set `task` to `pointing`; refinement is skipped
and points are matched one-to-one by containment in same-class GT HBBs.

`tools/prepare_evaluation.py` builds all five manifests from the portable HF
package. `scripts/evaluate_all.sh` schedules them over the configured GPUs and
uses `--resume`, so already written sample keys are not recomputed.

Detection uses each benchmark's original label ontology. In particular, DOTAv2
retains `plane`, `small vehicle`, and `large vehicle` as three distinct labels,
and DIOR retains `overpass`; the package's cross-dataset convenience
`class_name` field is not used for scoring. Model-facing names are signed
separately, so DIOR asks for `ground track field` while scoring the official
`groundtrackfield` label. The unambiguous DOTAv2 `airplane -> plane` alias is
accepted, but generic `vehicle` is not assigned to either vehicle subclass.
Preflight reconstructs both detection manifests and rejects missing, collapsed,
unknown, or inconsistently prompted labels.

The release mode is `--visual-context pixel_reencoded --crop-side 144
--local-resize-side 384`. Each point receives a real 144 x 144 source-pixel crop
resized to 384 x 384. The local PBD6 box is decoded in the local frame and mapped
back to the global image. `preprojector_magnified_roi` remains an ablation and is
not interchangeable with the release adapter.

## Shared-prefix waves

`--prefix-cache-mode shared` is the default. The global image passes through
MoonViT/projector and the Qwen prefix once per image. Local crops are encoded in
bounded batches, each point contributes a local visual/text suffix, and each
branch emits one PBD6 geometry block. A wave size is an execution bound, not an
object-count cap: 1,000 addresses with `--wave-size 200` execute as five waves.

```bash
pixel-pivr-infer \
  --model /path/to/LocateAnything-3B \
  --adapter /path/to/best.pt \
  --manifest /path/eval.jsonl \
  --output /path/shared-wave-output \
  --visual-context pixel_reencoded \
  --crop-side 144 \
  --local-resize-side 384 \
  --geometry-prefix-mode box_only \
  --global-fallback \
  --allow-none \
  --prefix-cache-mode shared \
  --wave-size 200
```

For a controlled equivalence/latency comparison, rerun the same manifest with
`--prefix-cache-mode recompute`. This historical path caches projected MoonViT
features but repeats the full Qwen prefix for every active branch.
