# Magnified Pre-Projector PIVR

## Purpose

This mode tests whether point-conditioned local evidence is more useful when it
is gathered before LocateAnything's fixed 2x2 MoonViT merge. It is a controlled
replacement for the post-projector cached-ROI arm: the model, 6K global image
budget, prompts, targets, Qwen LoRA scope, and training schedule remain fixed.

```text
one global image
      |
      v
MoonViT patch embedding + encoder (one pass)
      |
      +---------------- native global path ------------------+
      |                                                       |
      |  non-overlapping 2x2 merge -> released projector     |
      |                         -> global tokens [Ng, 2048]   |
      |                                                       |
      +---- point p -> 27x27 pre-merge cells ----------------+
                       -> overlapping 2x2, stride 1
                       -> released projector
                       -> local tokens [676, 2048]
                                      |
                                      v
                      [global tokens ; local tokens ; prompt]
                                      |
                                      v
                              Qwen LoRA + PBD6
```

There is no local pixel crop, no interpolation, no synthetic feature padding,
and no second MoonViT call. The global image is also resized and patchified only
once per address wave; branches share that tensor as well as the MoonViT cache.
Near an image boundary the 27x27 window shifts inward while retaining its full
field whenever the source grid is large enough.

## Exact dimensions

LocateAnything uses 14-pixel MoonViT patches, a 2x2 merge, a 1152-dimensional
vision state, and a projector whose input width is `4 * 1152 = 4608` and output
width is 2048.

For `MAGNIFIED_ROI_PIXELS=380`:

- nearest patch field: `round(380 / 14) = 27` cells;
- effective processed-image field: `27 * 14 = 378` pixels per axis;
- stride-1 projector grid: `(27 - 2) + 1 = 26` per axis;
- appended local context: `26 * 26 = 676` decoder-width tokens.

The global `IMAGE_TOKEN_LIMIT=6000` is a **pre-merge image-processor limit**. It
does not mean that Qwen receives 6,000 global visual tokens. A 1024x1024 image is
rounded by the processor to a model-compatible patch grid and normally contributes
roughly one quarter as many projected global tokens after the native 2x2 merge.
The 676 local tokens are appended only for a Round-2 point-refinement record.

## Data

Round 1 is unchanged:

```text
Human: <image> Point to all instances ...
GPT:   <ref>ship</ref><box><cx><cy></box>...
```

Round 2 contains one real global image and global normalized coordinates:

```text
Human: <image>
       Locate the single ship containing point <box><cx><cy></box>
       in horizontal box format. Return None if absent.
GPT:   <ref>ship</ref><box><x1><y1><x2><y2></box>
```

Required metadata is `pivr_route: point_indexed_visual_reentry` plus
`pivr_global_point: [cx, cy]`. The HBB and point must share full-image normalized
`[0,1000]` coordinates. The existing `pivr_cached_projected_roi_4k_v1` files meet
this contract and permit a paired post-projector versus pre-projector ablation.
The supplied configuration also signs the expected 16,000/1,000 record counts and
can reject any image hash found in a configured benchmark holdout list.

## Run

```bash
cp configs/magnified_preprojector_16k.env.example \
  configs/magnified_preprojector_16k.env
# Edit absolute paths only.

RUN_CONFIG="$PWD/configs/magnified_preprojector_16k.env" \
  bash scripts/train_distributed.sh audit
RUN_CONFIG="$PWD/configs/magnified_preprojector_16k.env" \
  bash scripts/train_distributed.sh check
RUN_CONFIG="$PWD/configs/magnified_preprojector_16k.env" \
  bash scripts/train_distributed.sh smoke
RUN_CONFIG="$PWD/configs/magnified_preprojector_16k.env" \
  bash scripts/train_distributed.sh train
```

Inference with supplied or predicted points:

```bash
CUDA_VISIBLE_DEVICES=0 pixel-pivr-infer \
  --model /path/to/LocateAnything-3B \
  --adapter /path/to/output/best.pt \
  --manifest /path/to/eval.jsonl \
  --data-root /path/to/workspace/root \
  --output /path/to/evaluation \
  --visual-context preprojector_magnified_roi \
  --image-token-limit 6000 \
  --magnified-roi-pixels 380 \
  --magnified-roi-stride 1 \
  --prefix-cache-mode shared \
  --wave-size 200
```

## Scientific boundary

This implementation makes the mechanism executable and auditable; it does not
establish that the mechanism improves accuracy. Compare it against the matched
post-projector ROI and standard LoRA controls on identical records, points,
prompts, model initialization, image-token limit, decoding, and metrics. Report
the stride-2 variant as an additional density control when feasible.
