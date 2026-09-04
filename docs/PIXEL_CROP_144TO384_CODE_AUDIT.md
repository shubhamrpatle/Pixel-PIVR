# Pixel-Crop 144-to-384 Training and Evaluation Audit

## Scope

This audit covers the historical two-GPU 16K/4K pilot and its frozen 100-image
DOTAv2 diagnostic set containing 9,045 HBB targets. The original machine-local
run directory is deliberately excluded from this portable release.

## Confirmed Defects

### 1. Round-2 prompt mismatch

Every point-indexed training row uses:

```text
Locate the single {class} containing point <box><cx><cy></box> in horizontal box format. Return None if absent.
```

The original evaluator instead used a longer two-image/address-A1 prompt that the
adapter never saw during training. A controlled evaluation retained the same
checkpoint, images, point-generation outputs, points, crops, parser, and metrics,
and changed only the Round-2 prompt:

| Round-2 prompt | TP | FP | FN | Precision | Recall | F1@0.5 |
|---|---:|---:|---:|---:|---:|---:|
| Legacy evaluator prompt | 152 | 552 | 8,893 | 21.59% | 1.68% | 3.12% |
| Training-matched compact prompt | 864 | 774 | 8,181 | 52.75% | 9.55% | 16.18% |

All 100 point lists and raw point-generation strings were exactly identical
between the two arms. The original 3.12% result is therefore not a valid measure
of the trained method under its intended prompt contract.

### 2. Unsynchronized distributed LoRA initialization

The completed run used two custom data-parallel replicas. The old trainer seeded
each rank with `seed + rank` before constructing LoRA, then manually averaged
gradients without first synchronizing the trainable parameters. PEFT initializes
LoRA A randomly, so the replicas did not represent the same model. Averaging their
gradients and saving rank 0 is not valid synchronous data-parallel optimization.

The affected checkpoint identifies `world_size=2` but contains neither
`synchronized_trainable_initialization=true` nor a synchronization policy. Its
numbers may be used only to diagnose the prompt issue; they cannot establish an
architectural gain or loss.

## Checks That Passed

- All 16,000 training and 1,000 validation records are unique and benchmark-hash
  overlap is zero.
- All 5,790 Round-2 records satisfy the compact prompt/answer grammar; all 5,118
  positive points lie inside their local target.
- All 100 sampled saved crops reproduced pixel-for-pixel from their source image.
- All 773 accepted boxes in the original evaluation were correctly inverse-scaled
  from the 384-pixel local frame into the 144-pixel source crop and translated to
  global coordinates; maximum reconstruction error was zero.
- On 50 held-out positive Round-2 records with GT points, the old adapter reached
  66% IoU@0.5 and 80% point containment. Shared-prefix and full-prefix recompute
  both scored 33/50, ruling out shared KV reuse as the failure source.
- Of 944 valid predicted point addresses, a 144-pixel crop made IoU@0.5
  geometrically impossible for 10 targets, all in the large-object stratum. This
  crop ceiling is real but too small to explain the original collapse.

## Corrections

- `pixel_pivr.infer` defaults to the compact training prompt and records its schema.
- The 144-to-384 launcher passes the compact schema explicitly.
- Local predictions are inverse-scaled and stored separately from global
  predictions to make coordinate audits unambiguous.
- Rank 0 now broadcasts every trainable tensor after construction and before
  rank-specific RNG streams and optimizer creation.
- Checkpoints record the synchronization policy. Distributed legacy checkpoints
  without this proof are rejected by default at inference and resume time.
- Regression tests cover prompt identity, 144-to-384 coordinate round trips,
  distributed parameter synchronization, and unsafe-adapter rejection.

## Required Fair Rerun

The fixed checkpoint must be trained from the released LocateAnything base using
the unchanged frozen data and hyperparameters. Only that synchronized rerun can
answer whether 144-to-384 pixel magnification improves over the matched global
control. Do not report the legacy 3.12% result as the architecture's performance.
