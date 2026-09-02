# Pixel-PIVR

Pixel-PIVR extends NVIDIA LocateAnything-3B with point-indexed visual re-entry.
The model first predicts typed point addresses from one global image, then uses
the cached global MoonViT features to extract a point-centred local feature field.
Each address is decoded into exactly one native six-token horizontal box (PBD6).

```text
global image + query
        |
        v
MoonViT + released projector (once)
        |
        +--> global projected tokens --> Qwen global-prefix KV cache (once)
        |
        +--> point-centred pre-projector ROI --> same projector
                                                |
typed point addresses --------------------------+
        |
        v
global prefix + local ROI + compact address prompt
        |
        v
one constrained PBD6 HBB per address
        |
        +--> sequential decoding (wave size 1)
        `--> shared-prefix wave decoding (wave size 2-200)
```

The full-scale recipe is a two-stage, rank-16 Qwen LoRA curriculum. MoonViT and
the released multimodal projector remain frozen. This repository does **not**
claim that the released full-scale schedule is full-parameter fine-tuning.

## Verified Scope

- Backbone: `nvidia/LocateAnything-3B`.
- Geometry: horizontal boxes and points in normalized `[0, 1000]` coordinates.
- Tasks in the full corpus: detection, phrase grounding, and pointing.
- Stage 1: coarse all-task adaptation.
- Stage 2: dense specialization with replay.
- Local context: one MoonViT pass, 27x27 pre-merge cells, overlapping 2x2
  projection at stride 1, and 676 decoder-width local tokens per address.
- Decoding: sequential or bounded shared-prefix waves; wave size is not an
  object-count cap.
- Resume: adapter, optimizer, scheduler, RNG, exposure, and signed data contract.

OBB/PBD10 and second-backbone support are not implemented in this release.

## Repository Layout

```text
configs/full_scale.env.example       complete destination-machine configuration
scripts/bootstrap_machine.sh         environment and asset setup
scripts/run_full_pipeline.sh         preflight, smoke, Stage 1, Stage 2, evaluation
scripts/train_distributed.sh         exact-coverage distributed trainer
scripts/evaluate_all.sh              resumable all-task benchmark scheduler
src/pixel_pivr/                       model, training, and inference implementation
tools/preflight.py                    model/data/environment contract check
tools/prepare_evaluation.py           portable benchmark-manifest conversion
tools/verify_release.py               source release audit
```

## 1. Installation

Clone this repository, create its dedicated environment, and install the exact
tested dependencies:

```bash
git clone https://github.com/shubhamrpatle/Pixel-PIVR.git
cd Pixel-PIVR
bash scripts/bootstrap_machine.sh install
source .venv/bin/activate
```

CUDA must be available to PyTorch. FlashAttention is optional; the code falls
back to SDPA when it is unavailable.

## 2. Download Assets

Install the Hugging Face CLI and authenticate. Authentication is mandatory while
the dataset repository is private.

```bash
curl -LsSf https://hf.co/cli/install.sh | bash
hf auth login

export WORK_ROOT=/absolute/path/to/pixel-pivr-assets
bash scripts/bootstrap_machine.sh download
```

This downloads Eagle from `https://github.com/NVlabs/Eagle.git` at tested commit
`8442db3b79f7fd2357e468e6eecdd9b6a82049ff`,
`nvidia/LocateAnything-3B`, and the `shubhampatle/Pixel-PIVR` dataset.
Downloads resume when the same command and `WORK_ROOT` are reused. Source
datasets retain their own licenses; review the dataset card before redistribution.

## 3. Dataset Contract

```text
annotations/train/<stage>/<task>/<round>/part-*.jsonl
annotations/validation/<task>/<round>/part-*.jsonl
annotations/test/<task>/<benchmark>/part-*.jsonl
images/<train|validation|test>/<sha-prefix>/<sha256>.<ext>
recipes/{stage1_coarse,stage2_dense_balanced,validation_all_tasks}.json
```

| Split | Global-point records | Point-to-box records | Total |
|---|---:|---:|---:|
| Stage 1 coarse | 123,598 | 264,812 | **388,410** |
| Stage 2 dense + replay | 50,681 | 1,811,340 | **1,862,021** |
| Validation pool | 1,000 | 94,740 | **95,740** |

There are 106,613 unique training images, 1,000 validation images, and 16,712
test images. SHA-256 overlap between train, validation, and test is zero.

Round 1 target:

```text
<ref>ship</ref><box><cx><cy></box>...
```

Round 2 target:

```text
Human: Locate the single ship containing point <box><cx><cy></box> in
       horizontal box format. Return None if absent.
GPT:   <box><x1><y1><x2><y2></box>
```

See `docs/DATA_FORMAT.md` and `docs/HUGGINGFACE_DATASET.md` for the full schema.

## 4. Configure and Check

```bash
cp configs/full_scale.env.example configs/full_scale.env
```

Edit `PYTHON_BIN`, `MODEL_PATH`, `EAGLE_ROOT`, `DATA_ROOT`, and `RUN_ROOT`, then:

```bash
PIPELINE_CONFIG="$PWD/configs/full_scale.env" \
  bash scripts/run_full_pipeline.sh preflight
```

For global batch 4, preflight must report:

| Stage | Records | Explicit repeated padding | One-pass optimizer steps |
|---|---:|---:|---:|
| Stage 1 | 388,410 | 2 | **97,103** |
| Stage 2 | 1,862,021 | 3 | **465,506** |

Padding is explicit in the run contract; no row is silently dropped. A seeded
global shuffle interleaves task/round shards while visiting every source row
exactly once; its order and resume cursor are fixed by the run contract. The
validation monitor selects 1,000 deterministic rows evenly across the five
task/round files (200 per file), rather than taking a detection-heavy prefix.

## 5. Smoke and Train

```bash
# Stage 1
PIPELINE_CONFIG="$PWD/configs/full_scale.env" \
  bash scripts/run_full_pipeline.sh smoke-stage1
PIPELINE_CONFIG="$PWD/configs/full_scale.env" \
  bash scripts/run_full_pipeline.sh train-stage1

# Stage 2 is gated on Stage 1 done.json and best.pt.
PIPELINE_CONFIG="$PWD/configs/full_scale.env" \
  bash scripts/run_full_pipeline.sh smoke-stage2
PIPELINE_CONFIG="$PWD/configs/full_scale.env" \
  bash scripts/run_full_pipeline.sh train-stage2
```

`train-stage2` initializes a fresh optimizer and scheduler from Stage 1's
lowest held-out validation-loss adapter. Both stages stop after one exact data
pass. `best.pt` tracks validation `native_loss`; `last.pt` tracks the latest
aligned checkpoint.

To run the complete sequence unattended:

```bash
PIPELINE_CONFIG="$PWD/configs/full_scale.env" \
  bash scripts/run_full_pipeline.sh all
```

### Interruption and Resume

SIGINT/SIGTERM requests a checkpoint at the next optimizer boundary. Rerun the
identical command to resume. The trainer rejects changes to data hashes, model,
world size, accumulation, schedule, seed, visual context, or validation policy.
Do not change GPU count for an in-progress output directory.

```bash
PIPELINE_CONFIG="$PWD/configs/full_scale.env" \
  bash scripts/run_full_pipeline.sh status
```

Training and validation curves are written to JSONL and, when configured, W&B:

```text
<stage>/training_curve.jsonl
<stage>/validation_curve.jsonl
<stage>/validation_monitor.json
<stage>/run_contract.json
<stage>/{best,last}.pt
```

When `WANDB_RUN_ID` is not supplied, the trainer derives a stable ID from the
output path and signed run contract, so an interrupted run resumes the same W&B
dashboard instead of creating a second curve.

## 6. Evaluation

The evaluator creates portable manifests and runs available benchmarks in
parallel across `EVAL_GPU_IDS`:

```bash
PIPELINE_CONFIG="$PWD/configs/full_scale.env" \
  bash scripts/run_full_pipeline.sh prepare-eval
PIPELINE_CONFIG="$PWD/configs/full_scale.env" \
  bash scripts/run_full_pipeline.sh evaluate
```

An interrupted benchmark resumes by `sample_key`; a completed benchmark is
skipped. Metrics are merged into:

```text
<RUN_ROOT>/evaluation/<adapter-and-decoding-tag>/all_metrics.json
```

| Task | Dataset | Protocol |
|---|---|---|
| Detection | DIOR, DOTAv2 | uncapped; class-aware one-to-one IoU@0.5 |
| Grounding | DIOR-RSVG, VRSBench-VG | first valid box; Acc@0.5, Acc@0.7, mIoU |
| Pointing | DOTAv2 Balanced-100 | one-to-one same-class GT-box containment diagnostic |

DOTAv2 Balanced-100 pointing is a diagnostic, not a standardized public
pointing benchmark. Test annotations are never loaded by the trainer.

## 7. Verification

```bash
python tools/verify_release.py
PYTHONPATH=src python -m unittest discover -s tests -v
bash -n scripts/*.sh
python tools/source_manifest.py check
```

The matched 16K/4K pilot evidence under `reference/` establishes mechanism
behavior on one seed and one diagnostic subset; it is not a full-scale or SOTA
claim.

## Common Failures

- `Repository not found`: authenticate with an account that can access the
  GitHub or private HF repository.
- `Missing Eagle loader`: set `EAGLE_ROOT` to `Eagle/Embodied`, not `Eagle`.
- GPU already in use: select free IDs; the launchers intentionally refuse sharing.
- Resume contract changed: restore the original config or use a new `RUN_ROOT`.
- Pointing annotation has no `gt_hboxes`: redownload the current dataset revision.
- W&B login failure: run `wandb login`, or set `WANDB_PROJECT=` to disable it.
