# Pixel-Crop Re-Encoded 380 Pilot

## Hypothesis

This controlled 16K/4K experiment tests whether genuinely new local pixels improve
point-conditioned HBB refinement more than local features sampled from a single
global MoonViT pass.

```text
global image ----------------> MoonViT -> 2x2 merge -> shared projector -> global tokens

point address -> exact 380x380 lossless pixel crop
                                      |
                                      v
                         MoonViT -> 2x2 merge -> shared projector -> local tokens

global tokens + local tokens + compact point prompt -> Qwen LoRA -> one PBD6 HBB
```

MoonViT and the released projector are executed once more for every local crop, but
their parameters remain frozen. Only rank-16 adapters inside Qwen are optimized.
The local pass has independent pixels, so it can preserve detail that was removed
while resizing the full scene. This benefit is a hypothesis until the matched
evaluation completes.

In the verified loader, an exact 380x380 crop becomes a 28x28 MoonViT grid and
then 196 decoder-width local tokens after native 2x2 merging. For a 1024x1024
source, the global view becomes a 74x74 grid and 1,369 projected tokens, giving
1,565 projected visual tokens after adding the local view. The 6,000 setting is
a per-input-image pre-merge patch-token ceiling, not a combined-sample budget or
forced padding.

## Controlled Training Contract

| Item | Value |
|---|---:|
| Train records | 16,000 unique source images |
| Validation records | 1,000 held-out source images |
| Optimizer steps | 4,000 |
| Global records/update | 4 |
| Data passes | exactly 1 |
| LoRA | Qwen only, rank 16 |
| Local crop | exact 380x380, lossless PNG |
| Visual-token limit | 6,000 |
| Sequence limit | 8,192 |
| Learning rate | 1e-5 |
| Warm-up | 100 steps |
| Checkpoint/validation cadence | 500 steps |
| Seed | 20260902 |

The source split, order, prompt schema, targets, seed, effective batch, optimizer,
and benchmark are held fixed against the matched pilots. A source image smaller than
380 pixels or a target not fully contained by the crop is never padded, distorted,
or clipped; its row is routed to target-complete global point supervision.

Round-2 records use two independent image inputs:

```text
Human:
<image><image>
Locate the single {class} containing point <box><cx><cy></box> in horizontal box
format. Return None if absent.

GPT:
<ref>{class}</ref><box><x1><y1><x2><y2></box>
```

The point and target coordinates are normalized in the local crop frame. Round 1
retains the unchanged global point-discovery record.

## Prepare And Verify

Create a machine-local config from the example, then run:

```bash
RUN_CONFIG="$PWD/configs/pixel_crop_reencoded_380_16k.env" \
  bash scripts/run_pixel_crop_reencoded_380_16k_experiment.sh prepare

RUN_CONFIG="$PWD/configs/pixel_crop_reencoded_380_16k.env" \
  bash scripts/run_pixel_crop_reencoded_380_16k_experiment.sh audit

RUN_CONFIG="$PWD/configs/pixel_crop_reencoded_380_16k.env" \
  bash scripts/run_pixel_crop_reencoded_380_16k_experiment.sh smoke
```

The dataset verifier requires zero benchmark-image overlap, zero train/validation
overlap, exact 380x380 local PNGs, valid point containment, complete targets, and
the compact prompt/answer grammar. The exact LA loader audit additionally rejects
any record skipped or truncated at the 6,000/8,192 token settings.

## Train And Evaluate

```bash
RUN_CONFIG="$PWD/configs/pixel_crop_reencoded_380_16k.env" \
  bash scripts/run_pixel_crop_reencoded_380_16k_experiment.sh train

RUN_CONFIG="$PWD/configs/pixel_crop_reencoded_380_16k.env" \
  bash scripts/run_pixel_crop_reencoded_380_16k_experiment.sh evaluate
```

Training resumes only under an identical signed run contract. `best.pt` is selected
by held-out validation loss and `last.pt` stores the final aligned step. Evaluation
uses the frozen DOTAv2 Balanced-100 set (100 images, 9,045 HBB GT), no prediction
cap, class-aware one-to-one IoU@0.5 matching, and both sequential and shared-prefix
wave-200 decoding.

This is a mechanism pilot, not a benchmark or SOTA claim. Pixel re-encoding adds
MoonViT/projector compute per address; the sequential/wave comparison measures how
much shared-prefix decoding recovers at inference.
