# Pixel-PIVR

Pixel-PIVR is a point-indexed, pixel-re-encoding extension of NVIDIA
LocateAnything-3B for remote-sensing localization. It keeps LocateAnything's
native point and six-token HBB grammar while separating localization into two
learned rounds:

```text
Global image + query
        |
        v
Frozen MoonViT + projector, Qwen LoRA
        |
        v
Grouped class-labelled point addresses
        |
        +--> 144 x 144 source-pixel crop around each point
        |          |
        |          +--> Lanczos resize to 384 x 384
        |          +--> frozen MoonViT + projector
        |
        v
Global visual context + local visual context + class + local point
        |
        v
Exactly one box-only PBD6 block, or <box>None</box>
        |
        +--> valid interior box: map to global coordinates
        `--> None/invalid/edge box: observable global retry
```

At inference, the global projected features and Qwen global-prefix KV cache are
computed once per image. Address branches are decoded sequentially or in bounded
shared-prefix waves. A wave size of 200 is a compute bound, not an object-count
cap; later addresses continue in additional waves.

## Release Scope

- Backbone: `nvidia/LocateAnything-3B`.
- Tasks: HBB detection, phrase grounding, and pointing.
- Coordinates: normalized integers in `[0, 1000]`.
- Adaptation: rank-16 Qwen-only LoRA; MoonViT and projector remain frozen.
- Stage 1: one exact pass over coarse all-task records.
- Stage 2: one exact pass over dense records plus source-defined replay.
- Objective balance: every flattened address row is visited once, with mean-one
  task weights preserving the source-query detection/grounding/pointing mixture.
- Local view: real 144-pixel crop resized to 384, not a cached-feature ROI.
- Output: native point grammar in Round 1 and box-only PBD6 in Round 2.
- Hardware release target: one node with 8 x A100 80 GB.

OBB/PBD10 and a second localization backbone are outside this release. The
matched pilot is evidence for the mechanism, not a full-scale or SOTA claim.
Full-scale execution remains deliberately locked until the final pilot result is
reviewed and `FULL_SCALE_APPROVED=YES` is set by the operator.

## Repository Layout

```text
configs/full_scale.env.example       immutable 8-GPU experiment contract
docs/A100_8GPU_FULL_SCALE_RUNBOOK.md exact destination-machine procedure
patches/eagle_virtual_crop_v1.patch  in-memory virtual-crop loader support
scripts/bootstrap_machine.sh         pinned environment and asset setup
scripts/configure_a100_node.sh        generate an absolute-path run config
scripts/run_full_pipeline.sh          preflight, smoke, Stage 1, Stage 2, eval
scripts/train_distributed.sh          exact-coverage resumable trainer
scripts/evaluate_all.sh               resumable all-GPU benchmark scheduler
src/pixel_pivr/                       model, training, and inference code
tools/package_hf_magnified_v2.py      build/verify materialized corpus
tools/build_hf_upload_bundle.py       deterministic low-file-count Hub bundle
tools/materialize_hf_dataset.py       safe archive extraction and verification
tools/preflight.py                    fail-closed remote-node audit
```

## Quick Start

Use immutable 40-character revisions supplied with the release:

```bash
export CODE_ROOT=/path/to/Pixel-PIVR
export WORK_ROOT=/path/to/pixel-pivr-assets
export RUN_ROOT=/path/to/pixel-pivr-runs/magnified-v2
export DATA_REVISION=<VERIFIED_HF_COMMIT_SHA>

git clone https://github.com/shubhamrpatle/Pixel-PIVR.git "$CODE_ROOT"
git -C "$CODE_ROOT" checkout --detach <VERIFIED_GITHUB_COMMIT_SHA>
cd "$CODE_ROOT"

bash scripts/bootstrap_machine.sh install
source .venv/bin/activate
hf auth login
bash scripts/bootstrap_machine.sh download

export PIPELINE_CONFIG="$CODE_ROOT/configs/full_scale.env"
DATA_REVISION="$DATA_REVISION" bash scripts/configure_a100_node.sh
bash scripts/run_full_pipeline.sh preflight
bash scripts/run_full_pipeline.sh smoke-stage1
```

Inspect the preflight and smoke outputs, then set only
`FULL_SCALE_APPROVED=YES` in `configs/full_scale.env` and run:

```bash
bash scripts/run_full_pipeline.sh train-stage1
bash scripts/run_full_pipeline.sh smoke-stage2
bash scripts/run_full_pipeline.sh train-stage2
bash scripts/run_full_pipeline.sh evaluate
```

See [the complete A100 runbook](docs/A100_8GPU_FULL_SCALE_RUNBOOK.md) before
starting. It covers prerequisites, revision verification, exact one-pass step
calculation, checkpoint selection, graceful interruption/resume, monitoring,
and all-task evaluation.

## Dataset Contract

Use `shubhampatle/Pixel-PIVR-Magnified-v2` only after its exact revision passes
`tools/verify_hf_snapshot.py`. The older `shubhampatle/Pixel-PIVR` snapshot is a
different schema and must not be used for this experiment.

The Hub release transports images in deterministic tar shards. Bootstrap first
verifies those archives, safely materializes `images/...`, and then verifies the
complete annotations, image inventory, crop geometry, fallback pairing, recipe
counts, and train/validation/test image-hash separation. Exact counts and the
actual record-level Stage 2 replay ratio live in the downloaded `manifest.json`.

See [the data contract](docs/DATA_FORMAT.md) and
[the Hub release procedure](docs/HUGGINGFACE_DATASET.md).

## Reproducibility and Resume

The trainer records file SHA-256 values, record counts, model/data paths, world
size, accumulation, seed, sample order, optimizer schedule, visual contract,
validation monitor, attention backend, clipping and checkpoint cadence, and
initial adapter. It stores LoRA weights, optimizer,
scheduler, per-rank RNG state, and exact record exposure. One `Ctrl+C` requests
an aligned checkpoint; rerunning the identical command resumes without changing
the one-pass permutation.

Every launch also checks each scheduled train and validation shard against the
downloaded package's signed `SHA256SUMS`; a partial or edited annotation cannot
silently become a new training run.

`best.pt` is selected by held-out monitor loss. `last.pt` is the newest aligned
checkpoint. Stage 2 initializes from Stage 1 `best.pt` but starts a fresh
optimizer and scheduler.

## Local Verification

```bash
python tools/verify_release.py
pytest -q
bash -n scripts/*.sh
python tools/source_manifest.py check
git diff --check
```

Source datasets retain their own licenses. Review every source license before
making a derived image bundle public or using it commercially.
