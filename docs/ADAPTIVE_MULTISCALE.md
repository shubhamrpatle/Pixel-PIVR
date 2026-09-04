# Adaptive Multi-Scale Cached-Feature PIVR

This treatment isolates one question: can point-conditioned scale selection improve
PIVR without a second MoonViT execution? It uses the same base model, 16K records,
validation set, prompts, targets, LoRA rank, seed, image budget, sequence budget,
global batch, optimizer schedule, and 4K steps as the single-scale control.

```text
global image -> MoonViT once -> unmerged feature grid [H,W,1152]
                                  |
point ----------------------------+
        | small 196 px field -> resize to 27x27 -> same projector |
        | medium 378 px field -> resize to 27x27 -> same projector| -> learned gate
        | large 756 px field -> resize to 27x27 -> same projector |    -> 676x2048

[global projected tokens ; fused local tokens ; class/point prompt] -> Qwen LoRA -> PBD6
```

The scale gate is the only new trainable component outside Qwen LoRA. It pools each
projected scale, predicts three normalized weights, and fuses corresponding tokens.
Its initial bias favors the medium field, preserving a stable approximation to the
single-scale treatment at initialization. MoonViT and LA's projector remain frozen.

The three source fields contain 14x14, 27x27, and 54x54 MoonViT cells for a 14-pixel
patch size. Bilinear feature resampling maps each field to 27x27 cells. Overlapping
2x2 stride-1 projection produces 26x26 = 676 local tokens regardless of scale, so
the Qwen sequence length does not triple.

This is feature-space scale normalization, not pixel-level super-resolution. It
cannot recreate visual frequencies absent from the cached MoonViT representation.

Run:

```bash
RUN_CONFIG="$PWD/configs/adaptive_multiscale_16k.env" \
  bash scripts/train_distributed.sh audit
RUN_CONFIG="$PWD/configs/adaptive_multiscale_16k.env" \
  bash scripts/train_distributed.sh check
RUN_CONFIG="$PWD/configs/adaptive_multiscale_16k.env" \
  bash scripts/train_distributed.sh smoke
RUN_CONFIG="$PWD/configs/adaptive_multiscale_16k.env" \
  bash scripts/train_distributed.sh train
```

Accuracy claims require the matched DOTAv2 Balanced-100 evaluation. Report both
sequential (`wave_size=1`) and shared-prefix wave (`wave_size=200`) decoding, the
learned mean scale weights, and latency split into point discovery and refinement.
