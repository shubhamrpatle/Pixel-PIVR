# Large-Scale Training

## What scales unchanged

The trainer preserves the tested mechanism:

- one record per rank at a time;
- Qwen rank-16 LoRA only;
- frozen MoonViT and projector;
- native LocateAnything MTP/PBD6 supervision;
- fixed effective global record batch;
- full held-out validation loss at a configured interval.

Scaling the number of records does not require increasing LoRA rank or changing the
output grammar.

Model loading is serialized across ranks by default (`SERIAL_MODEL_LOAD=1`). This
adds a small one-time startup cost but avoids concurrent checkpoint-I/O stalls on
shared storage. Set it to `0` only after a successful destination-machine smoke
test; it does not change training samples, gradients, or optimizer steps.

The config is sourced as a shell file. Wrap any value containing spaces in double
quotes, for example `MODEL_PATH="/data/My Models/LocateAnything-3B"`.

## Exact coverage

Let `N` be training records, `W` world size, and `A` gradient accumulation:

```text
global records/update = W * A
one-pass optimizer steps = N / (W * A)
```

With `MAX_STEPS=0`, the launcher derives this value and refuses to start unless it
is integral. Padding is never implicit. Set `ALLOWED_PADDING_RECORDS` only when a
small, explicitly recorded repeat is scientifically acceptable.

## Resume

Every checkpoint stores:

- LoRA state;
- optimizer and scheduler;
- optimizer step and exact record exposure;
- per-rank CPU/CUDA RNG state;
- world size and accumulation;
- SHA256, byte size, and line count for every train/validation JSONL;
- model and data-root paths.

Rerunning the identical command resumes `last.pt`. Changing world size, data,
schedule, seed, model, image-token limit, or sequence limit is rejected. SIGINT or
SIGTERM requests a checkpoint at the next aligned optimizer boundary.

## Stage 2

Stage 2 is a fresh run, not a Stage-1 trainer resume:

1. Wait for Stage 1 `done.json`.
2. Select Stage 1 by held-out validation loss or task metric.
3. Set `INIT_ADAPTER=/stage1/best.pt`.
4. Use Stage-2 train data and a new `OUTPUT_DIR`.
5. Start with a fresh optimizer and scheduler.

This preserves the learned adapter while preventing Stage-1 scheduler state from
leaking into Stage 2.

## Large-scale claim boundary

The standalone code is HBB-only. Mixing OBB `<quad>` records into the current PBD6
loader is rejected by the audit. OBB requires an independently tested PBD10 grammar,
decoder, and benchmark protocol.
