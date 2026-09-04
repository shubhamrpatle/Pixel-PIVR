# Pixel-Crop Magnification: 144 to 384

## Hypothesis

A native-6000 global view already represents medium and large targets reasonably
well. The controlled question here is whether genuinely magnifying a small pixel
field around each predicted point improves small-object HBB refinement.

## Visual Path

```text
global image -> MoonViT + shared projector -> global visual tokens

predicted point -> exact 144 x 144 source-pixel crop
                -> Lanczos resize to 384 x 384
                -> MoonViT + the same projector
                -> local visual tokens

[global tokens ; local tokens ; class/point prompt] -> Qwen LoRA -> one PBD6 HBB
```

This is a second MoonViT pass for each local view. It is not interpolation of a
cached feature map. The 2.667x pixel magnification is therefore real, although it
cannot create information that was absent in the original 144-pixel field.

## Geometry Contract

- Source crop: exactly `144 x 144`, shifted at image boundaries without padding.
- MoonViT input: exactly `384 x 384`, produced with PIL Lanczos interpolation.
- Point and HBB: normalized to `[0, 1000]` in the local frame.
- Evaluation: decoded local HBBs are inverse-scaled by `144 / 384` and translated
  back into full-image coordinates before IoU matching.
- Positive targets must be fully contained by the source crop. Unsafe targets are
  routed to target-complete global point supervision; they are never clipped.

## Matched Pilot

- Base model: released LocateAnything-3B.
- Training: Qwen LoRA rank 16, 4,000 optimizer steps, one exposure of 16,000 records.
- Global image token limit: 6,000.
- Maximum sequence length: 8,192.
- Seed: 20260902.
- Evaluation: the frozen DOTAv2 Balanced-100 manifest, 9,045 GT boxes, no prediction
  cap, class-aware one-to-one matching at IoU 0.5.
- Decoding: sequential and shared-prefix wave-200.
- Round-2 prompt: the compact training template, exactly:
  `Locate the single {class} containing point <box><cx><cy></box> in horizontal box format. Return None if absent.`
- Distributed correctness: rank 0 broadcasts every trainable LoRA tensor after
  construction and before rank-specific RNG streams or optimizer creation.

Run the complete experiment with:

```bash
RUN_CONFIG="$PWD/configs/pixel_crop_magnified_144to384_16k_syncfix.env" \
  bash scripts/run_pixel_crop_magnified_144to384_16k_experiment.sh all
```

The earlier two-GPU output without
`synchronized_trainable_initialization=true` is retained only as a diagnostic
artifact. The evaluator rejects such adapters by default.
