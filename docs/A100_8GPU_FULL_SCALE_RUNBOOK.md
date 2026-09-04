# Pixel-PIVR Full-Scale Runbook: 8 x A100 80 GB

This runbook is the fail-closed procedure for the two-stage HBB experiment. Do
not substitute the older `shubhampatle/Pixel-PIVR` dataset: that repository uses
the pre-magnification schema and is not the 144-to-384 pixel re-encoding corpus.

## Frozen experiment

- Base model: `nvidia/LocateAnything-3B` at commit
  `c32291ca5e996f5a7a485845b4f57a233936bba0`.
- Eagle: commit `8442db3b79f7fd2357e468e6eecdd9b6a82049ff` plus
  `patches/eagle_virtual_crop_v1.patch`.
- Trainable weights: rank-16 LoRA modules inside Qwen only.
- Frozen weights: MoonViT and LocateAnything's multimodal projector.
- Global image limit: 6,000 pre-merge MoonViT patch tokens per image.
- Qwen sequence limit: 32,768 tokens.
- Local input: exactly 144 x 144 source pixels, Lanczos-resized to 384 x 384;
  the unchanged LA processor aligns it to 392 x 392, yielding 196 projected
  local tokens.
- Stage 1: every coarse record once.
- Stage 2: every dense-plus-source-defined-replay record once, initialized from
  Stage 1 `best.pt` with a new optimizer and scheduler.
- Loss policy: visit every flattened row once, then weight each record so the
  aggregate detection/grounding/pointing contribution preserves the signed
  source-query task mixture after one-address-per-row expansion.
- Hardware contract: one process and one record per GPU, eight A100 GPUs, BF16,
  FlashAttention 2 for MoonViT, SDPA for Qwen.

`manifest.json` is the authority for exact record counts, replay percentage,
padding, and optimizer-step calculations. The launcher computes, rather than
hard-codes, `ceil(records / 8)` and records the minimal repeated batch padding.

For the frozen release corpus, the expected schedule is:

| Stage | Optimizer records | Minimal padding | Steps at global batch 8 |
|---|---:|---:|---:|
| Stage 1 coarse | 460,263 | 1 | 57,533 |
| Stage 2 dense + replay | 2,100,576 | 0 | 262,572 |

Stage 2 contains 1,960,076 dense and 140,500 replay optimizer records, or 6.69%
record-level replay after one-address-per-record expansion. At the original
Round-1 query level it contains 39,477 dense and 11,204 replay records, or
22.11% replay. These are different units. Preflight must reproduce these counts
from the downloaded manifest and recipes before the operator approves training.
For the frozen corpus, preflight also reports the following expected source-query
task proportions: Stage 1 is 50.73% detection, 43.59% grounding, and 5.68%
pointing; Stage 2 is 46.65%, 32.52%, and 20.83%, respectively. The derived
mean-one task weights are signed into checkpoints and cannot change on resume.

## 1. Machine prerequisites

Use Linux, Python 3.10, Git, an NVIDIA driver capable of CUDA 12.1 PyTorch, and
the CUDA 12.1 toolkit (`nvcc`) required to build FlashAttention. The installer
fails closed if `torch.version.cuda` and the `nvcc` major/minor version differ.
Keep at least 180
GiB free for downloaded archives plus materialized images, at least 100 GiB on
the run filesystem, and additional space for model/cache files.

```bash
nvidia-smi
python3.10 --version
nvcc --version
git --version
df -h /path/for/assets /path/for/runs
```

The launch contract requires exactly eight GPUs whose names contain `A100` and
whose reported memory is at least 75 GiB. It refuses occupied GPUs.

## 2. Clone the exact code revision

Replace `<CODE_REVISION>` with the 40-character commit reported by the release
owner. Do not train from a moving branch.

```bash
export CODE_ROOT=/path/to/Pixel-PIVR
git clone https://github.com/shubhamrpatle/Pixel-PIVR.git "$CODE_ROOT"
git -C "$CODE_ROOT" checkout --detach <CODE_REVISION>
test "$(git -C "$CODE_ROOT" rev-parse HEAD)" = <CODE_REVISION>
cd "$CODE_ROOT"
```

## 3. Install the tested environment

The installer creates a repository-local virtual environment and pins PyTorch,
Transformers, PEFT, and all Eagle dependencies. It does not use an existing
system environment.

```bash
export WORK_ROOT=/path/to/pixel-pivr-assets
export VENV="$CODE_ROOT/.venv"
export FLASH_ATTN_MAX_JOBS=16
bash scripts/bootstrap_machine.sh install
source "$VENV/bin/activate"
python -m pip check
pytest -q
wandb login
wandb status
```

If FlashAttention compilation fails, fix the compiler/CUDA environment. Do not
disable it for the release run; preflight requires it. To run without W&B,
explicitly set `WANDB_PROJECT=` before generating the run config instead of
leaving an unauthenticated W&B run enabled.

## 4. Download one immutable dataset snapshot

The release owner supplies `<DATA_REVISION>` only after remote verification has
passed. The bootstrap command resumes downloads, verifies every tar SHA-256,
verifies every archived image SHA-256, safely materializes the image tree, and
runs the complete package verifier.

```bash
hf auth login
export DATA_REPO=shubhampatle/Pixel-PIVR-Magnified-v2
export DATA_REVISION=<DATA_REVISION>
export MODEL_REPO=nvidia/LocateAnything-3B
export MODEL_REVISION=c32291ca5e996f5a7a485845b4f57a233936bba0
export EAGLE_REVISION=8442db3b79f7fd2357e468e6eecdd9b6a82049ff
export HF_XET_HIGH_PERFORMANCE=1
bash scripts/bootstrap_machine.sh download
```

Rerunning the same command is supported. It verifies existing images and writes
`$WORK_ROOT/download_receipt.json`; that receipt is mandatory at preflight.

## 5. Generate, inspect, and lock the node configuration

```bash
export RUN_ROOT=/path/to/pixel-pivr-runs/magnified-v2
export PIPELINE_CONFIG="$CODE_ROOT/configs/full_scale.env"
DATA_REVISION="$DATA_REVISION" \
  bash scripts/configure_a100_node.sh
sed -n '1,220p' "$PIPELINE_CONFIG"
```

Keep `FULL_SCALE_APPROVED=NO` initially. Do not alter world size, seed, data,
token limits, accumulation, or optimizer settings after a stage has started.

## 6. Mandatory preflight and Stage 1 smoke

The preflight checks all four revisions, a clean tracked checkout, model shards,
the patched Eagle loader, package checksums, zero train/validation/test image-hash
overlap, exact recipes, task-loss weights, dependencies, GPU topology, model
context, and disk space. It also runs the real patched Eagle/LocateAnything loader
on deterministic hard records selected from every recipe shard: longest prompt,
largest source image, highest target count, and an explicit-None example where
available. The subsequent eight-GPU smoke is the required end-to-end forward,
backward, validation, synchronization, and checkpoint gate.

```bash
PIPELINE_CONFIG="$PIPELINE_CONFIG" \
  bash scripts/run_full_pipeline.sh preflight | tee "$RUN_ROOT/preflight.log"

PIPELINE_CONFIG="$PIPELINE_CONFIG" \
  bash scripts/run_full_pipeline.sh smoke-stage1
```

Inspect the smoke `done.json`, finite loss, peak memory, and all eight ranks. The
smoke writes a signed receipt; changing the stage config invalidates it.

All eight GPUs run independent model replicas concurrently during training and
all eight receive benchmark shards during evaluation. This maximizes data-parallel
throughput without padding variable-length samples together. It intentionally does
not promise 80 GB memory saturation; adding unverified multi-record packing would
change both the optimization and exact-resume contract.

## 7. Run Stage 1 for exactly one pass

After the checks pass, change only `FULL_SCALE_APPROVED=NO` to `YES` in
`configs/full_scale.env`, then launch inside tmux:

```bash
tmux new -s pixel-pivr
cd "$CODE_ROOT"
source .venv/bin/activate
export PIPELINE_CONFIG="$CODE_ROOT/configs/full_scale.env"
bash scripts/run_full_pipeline.sh train-stage1
```

Monitor from another shell:

```bash
PIPELINE_CONFIG="$PIPELINE_CONFIG" bash scripts/run_full_pipeline.sh status
tail -f "$CODE_ROOT/runs/logs/stage1_coarse.log"
nvidia-smi
```

Stage 1 is complete only when `stage1_coarse/done.json` exists and contains
`"complete_one_pass": true`. Directory existence alone is not completion.

## 8. Smoke and run Stage 2

Stage 2 cannot start before Stage 1 completes. It loads Stage 1's lowest monitor
loss adapter from `best.pt`, then deliberately creates a fresh optimizer and
scheduler.

```bash
PIPELINE_CONFIG="$PIPELINE_CONFIG" \
  bash scripts/run_full_pipeline.sh smoke-stage2

PIPELINE_CONFIG="$PIPELINE_CONFIG" \
  bash scripts/run_full_pipeline.sh train-stage2
```

Completion requires `stage2_dense_balanced/done.json` with
`"complete_one_pass": true`. The manifest and preflight report the actual replay
fraction after one-target-per-record expansion; do not relabel it as 20% unless
the reported record-level value is 20%.

For an unattended, resumable handoff after preflight, Stage-1 smoke, and explicit
approval, the same stages and final evaluation can be chained in one tmux job:

```bash
tmux new -s pixel-pivr-full
cd "$CODE_ROOT"
source .venv/bin/activate
export PIPELINE_CONFIG="$CODE_ROOT/configs/full_scale.env"
bash scripts/run_full_pipeline.sh all
```

The command does not overlap stages: Stage 2 is gated on Stage 1 `done.json` and
starts only from Stage 1 `best.pt`. Rerunning `all` after a clean interruption
resumes the applicable stage from its aligned `last.pt`.

## 9. Interruption and exact resume

Press `Ctrl+C` once. The trainer finishes the current optimizer boundary and
writes an aligned `last.pt`. Rerun the identical command to resume optimizer,
scheduler, random states, data permutation, and exact exposure cursor. A second
interrupt can force termination before that checkpoint finishes.

The trainer rejects changed data hashes, model path, world size, accumulation,
seed, sequence limits, visual mode, validation monitor, or initial adapter.
Before model loading on every launch, each scheduled annotation SHA-256 is also
matched to the downloaded package ledger.

## 10. Evaluate with all eight GPUs

Public benchmarks are never training or validation inputs. Evaluation shards
each benchmark over all eight GPUs, resumes by sample key, and merges only after
the expected sample count is present.

```bash
PIPELINE_CONFIG="$PIPELINE_CONFIG" \
  bash scripts/run_full_pipeline.sh prepare-eval

PIPELINE_CONFIG="$PIPELINE_CONFIG" \
  bash scripts/run_full_pipeline.sh evaluate
```

The combined result is written under:

```text
$RUN_ROOT/evaluation/<adapter-and-decoding-contract>/all_metrics.json
```

DIOR and DOTAv2 detection are uncapped class-aware one-to-one IoU@0.5 metrics.
They use the original benchmark labels: DOTAv2 keeps its 18-class ontology
(`plane`, `small vehicle`, and `large vehicle` remain distinct), while DIOR
retains `overpass`. Model-facing DIOR spellings such as `ground track field`
are explicitly mapped back to the official `groundtrackfield` scoring label.
The unambiguous DOTAv2 `airplane -> plane` output alias is allowed; `vehicle`
is never mapped to a size subclass. Preflight verifies this conversion before
training starts.
DIOR-RSVG and VRSBench-VG grounding report first-box Acc@0.5, Acc@0.7, and mIoU.
DOTAv2 Balanced-100 pointing remains a diagnostic, not a public universal
pointing benchmark.

The packaged VRSBench-VG test is the leakage-controlled 16,154-query subset of
the 16,159-query official split. Five queries whose image hashes occur in the
independent checkpoint-selection pool are intentionally excluded. Report this
fact and do not present its count as the untouched official split.

## 11. Final evidence to retain

Retain both stage directories, `download_receipt.json`, the generated config,
preflight log, run contracts, training/validation curves, W&B run IDs,
`best.pt`, `last.pt`, both `done.json` files, evaluation manifests, raw
predictions, and `all_metrics.json`. These files establish exactly which code,
data, optimizer schedule, and evaluation protocol produced the result.
