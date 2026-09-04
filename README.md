# Pixel-PIVR

Pixel-PIVR extends NVIDIA LocateAnything-3B with point-indexed pixel visual
re-entry for remote-sensing localization. It retains LocateAnything's native
point and six-token horizontal-box grammar, while splitting localization into
two learned rounds:

```text
Round 1: global discovery

global image + task query
        |
        v
MoonViT + projector + Qwen LoRA
        |
        v
class-labelled point addresses

Round 2: point-indexed refinement

global image + one 144 x 144 crop around an address
        |                  |
        |                  +--> resize to 384 x 384 with Lanczos
        |                              |
        +------------------------------+
                       |
                       v
       frozen MoonViT + frozen projector
                       |
                       v
       global tokens + local tokens + class/point prompt
                       |
                       v
          Qwen LoRA predicts one PBD6 HBB or None
                       |
             +---------+----------+
             |                    |
      valid local HBB       observable retry
             |             (None/invalid/edge)
             v                    |
 map to global frame              v
                         global point-to-box pass
```

The global projected features and Qwen prefix cache are computed once per
image. Point branches can be decoded sequentially or in bounded shared-prefix
waves. `wave_size=200` is a compute bound, not an object cap: 1,000 addresses
are processed in five waves.

## What This Release Contains

This repository is the reproducible HBB Magnified-v2 release.

| Item | Contract |
|---|---|
| Base model | `nvidia/LocateAnything-3B` |
| Dataset | [`shubhampatle/Pixel-PIVR-Magnified-v2`](https://huggingface.co/datasets/shubhampatle/Pixel-PIVR-Magnified-v2) |
| Pinned dataset revision | `5be1a40c776b494a9ae6ec472b7c611aa9027698` |
| Tasks | HBB detection, phrase grounding, pointing |
| Coordinates | Integer-normalized `[0, 1000]` |
| Trainable parameters | Rank-16 LoRA modules inside Qwen |
| Frozen parameters | MoonViT and multimodal projector |
| Global image limit | 6,000 pre-merge MoonViT patch tokens/image |
| Qwen sequence limit | 32,768 tokens |
| Local visual input | 144 source pixels resized to 384 pixels |
| Round-2 output | One box-only PBD6 block or `<box>None</box>` |
| Stage 1 | One exact pass over coarse all-task records |
| Stage 2 | One exact pass over dense records plus fixed replay |
| Released hardware contract | One node with 8 x A100 80 GB |

This is **LoRA adaptation**, not full-parameter fine-tuning. OBB/PBD10 and a
second localization backbone are not included in this release. The older
`shubhampatle/Pixel-PIVR` dataset uses a different pre-magnification schema and
must not be substituted for Magnified-v2.

## Repository Layout

```text
configs/full_scale.env.example       immutable full-scale configuration
docs/A100_8GPU_FULL_SCALE_RUNBOOK.md detailed operator runbook
docs/DATA_FORMAT.md                  Round-1/Round-2 data schemas
docs/INFERENCE.md                    decoding and metric contracts
patches/eagle_virtual_crop_v1.patch  verified Eagle virtual-crop support
scripts/bootstrap_machine.sh         install, download, and materialize assets
scripts/configure_a100_node.sh        create an absolute-path run config
scripts/run_full_pipeline.sh          preflight, smoke, train, resume, evaluate
scripts/train_distributed.sh          exact-coverage distributed trainer
scripts/evaluate_all.sh               resumable multi-GPU evaluation
src/pixel_pivr/                       training and inference implementation
tools/prepare_evaluation.py           build benchmark manifests from raw labels
tools/preflight.py                    fail-closed machine/data/code audit
```

## Full-Scale Data Summary

The signed Magnified-v2 package contains 124,325 materialized images and uses
source-image SHA-256 for split assignment.

| Split | Unique images |
|---|---:|
| Train | 106,613 |
| Validation | 1,000 |
| Test | 16,712 |
| Total | 124,325 |

The frozen training schedule is:

| Stage | Optimizer records | Minimal padding | Global batch | Steps |
|---|---:|---:|---:|---:|
| Stage 1 coarse | 460,263 | 1 | 8 | 57,533 |
| Stage 2 dense + replay | 2,100,576 | 0 | 8 | 262,572 |

Stage 2 contains 1,960,076 dense and 140,500 replay optimizer records. This is
6.69% replay after one-address-per-row expansion. At the original source-query
level, replay is 22.11%; these percentages use different units and must not be
interchanged. The checkpoint-selection monitor contains 2,141 held-out records.

The downloaded `manifest.json` is authoritative. Every launcher independently
recomputes these counts and refuses a mismatch.

## Data and Supervision Format

Every sample is one JSON object per line with `conversations`, `image`, and
signed `meta`. Geometry coordinates are normalized integers in `[0, 1000]`.

Round 1 performs global point discovery. A detection example is:

```text
Human:
<image>
Point to all instances that match the following categories: ship</c>vehicle.

GPT:
<ref>ship</ref><box><321><418></box><ref>vehicle</ref><box>None</box>
```

Each present object is represented by its box-center point. Every requested
category has one `<ref>` group, and a verified absent category uses exactly
`<box>None</box>`. Detection retains every instance of every requested class.
Grounding uses the referring phrase as the reference, while pointing stops after
Round 1 and is scored by class-aware point containment.

Round 2 refines one known class/point address:

```text
Human:
<image><image>
Locate the single ship containing point <box><500><500></box> in the local view. Return its complete horizontal box, or None if the complete boundary is unavailable.

GPT positive:
<box><420><455><610><580></box>

GPT negative/unsafe local view:
<box>None</box>
```

The second image is a virtual descriptor. The patched loader extracts a real
144 x 144 source-pixel crop, resizes it to 384 x 384, and lets the normal LA
processor align it to 392 x 392. This produces 784 MoonViT patch tokens and 196
projected local tokens after LA's native 2 x 2 merge. Positive local boxes must
be completely contained; boxes are never clipped to create artificial targets.
An observable local failure or edge-touching result triggers the paired global
fallback route.

Training, validation, and test assignment is based on source-image SHA-256. The
package verifier requires zero hash overlap between those splits. Stage 2 replay
may intentionally reuse training records from Stage 1; it never imports a
validation or benchmark image.

## 1. Machine Requirements

The released full-scale configuration requires:

- Linux and Git.
- Python 3.10 exactly.
- Eight NVIDIA A100 GPUs reporting at least 75 GiB each.
- An NVIDIA driver compatible with CUDA 12.1 PyTorch.
- CUDA 12.1 toolkit and `nvcc` for FlashAttention 2.
- At least 180 GiB free for archives and materialized data.
- At least 100 GiB free on the run/checkpoint filesystem.
- Access to the model and dataset repositories on Hugging Face.

Check the node before downloading anything:

```bash
nvidia-smi
python3.10 --version
nvcc --version
git --version
df -h /path/for/assets /path/for/runs
```

The guarded launcher refuses a non-A100 node, GPUs below the memory threshold,
occupied GPUs, a dirty source checkout, incompatible CUDA/FlashAttention, or
insufficient run storage. Other hardware requires a separately validated
configuration and is not the released experiment contract.

## 2. Clone and Pin the Code

Use one parent directory on a high-capacity filesystem. The following layout
keeps source code, downloaded assets, and generated checkpoints separate:

```text
$PIXEL_PIVR_HOME/
|-- Pixel-PIVR/                         Git checkout and local .venv
|-- assets/
|   |-- Eagle/                          pinned and patched Eagle checkout
|   |-- LocateAnything-3B/              pinned base-model snapshot
|   |-- Pixel-PIVR-Magnified-v2/        Hub files plus materialized images
|   `-- download_receipt.json           immutable download identities
`-- runs/
    `-- magnified-v2/                   Stage 1, Stage 2, and evaluation outputs
```

Set `PIXEL_PIVR_HOME` to a real absolute path with at least 180 GiB free for
assets. Keep `RUN_ROOT` on a filesystem with at least 100 GiB free. They may be
different filesystems when needed.

```bash
export PIXEL_PIVR_HOME=/absolute/high-capacity/path/pixel-pivr
export CODE_ROOT="$PIXEL_PIVR_HOME/Pixel-PIVR"
export WORK_ROOT="$PIXEL_PIVR_HOME/assets"
export RUN_ROOT="$PIXEL_PIVR_HOME/runs/magnified-v2"
export VENV="$CODE_ROOT/.venv"

mkdir -p "$PIXEL_PIVR_HOME" "$WORK_ROOT" "$RUN_ROOT"
git clone https://github.com/shubhamrpatle/Pixel-PIVR.git "$CODE_ROOT"
cd "$CODE_ROOT"

# Resolve the cloned release to an immutable identity and detach from main.
export CODE_REVISION="$(git rev-parse HEAD)"
git checkout --detach "$CODE_REVISION"
test "$(git rev-parse HEAD)" = "$CODE_REVISION"
printf 'CODE_REVISION=%s\n' "$CODE_REVISION"
```

Record the printed 40-character `CODE_REVISION` with the experiment. For a
published result, all machines must check out the same recorded commit instead
of resolving `main` independently.

The exported shell variables are not persistent across new SSH logins. Put the
following block at the beginning of every later shell, changing only the first
line to the path selected above:

```bash
export PIXEL_PIVR_HOME=/absolute/high-capacity/path/pixel-pivr
export CODE_ROOT="$PIXEL_PIVR_HOME/Pixel-PIVR"
export WORK_ROOT="$PIXEL_PIVR_HOME/assets"
export RUN_ROOT="$PIXEL_PIVR_HOME/runs/magnified-v2"
export VENV="$CODE_ROOT/.venv"
export PIPELINE_CONFIG="$CODE_ROOT/configs/full_scale.env"
export DATA_REPO=shubhampatle/Pixel-PIVR-Magnified-v2
export DATA_REVISION=5be1a40c776b494a9ae6ec472b7c611aa9027698
export MODEL_REPO=nvidia/LocateAnything-3B
export MODEL_REVISION=c32291ca5e996f5a7a485845b4f57a233936bba0
export EAGLE_REVISION=8442db3b79f7fd2357e468e6eecdd9b6a82049ff
cd "$CODE_ROOT"
```

## 3. Install the Tested Environment

The bootstrap script creates a repository-local `.venv` and installs pinned
CUDA 12.1 PyTorch, Transformers, PEFT, training dependencies, tests, and
FlashAttention 2.

```bash
: "${CODE_ROOT:?Run the path-export block from Section 2 first}"
: "${WORK_ROOT:?Run the path-export block from Section 2 first}"
: "${VENV:?Run the path-export block from Section 2 first}"
export FLASH_ATTN_MAX_JOBS=16

bash scripts/bootstrap_machine.sh install
source "$VENV/bin/activate"

python -m pip check
python tools/verify_release.py
pytest -q
python tools/source_manifest.py check
```

The install intentionally fails when `torch.version.cuda` and `nvcc` are not
both CUDA 12.1. Do not disable FlashAttention for the released A100 run because
preflight requires it.

## 4. Authenticate Optional Services

Log in to Hugging Face when the model or dataset is gated/private:

```bash
hf auth login
```

W&B is optional. To enable training and validation curves:

```bash
wandb login
wandb status
export WANDB_PROJECT=pixel-pivr
```

To disable W&B, set an explicitly empty value **before** generating the config:

```bash
export WANDB_PROJECT=
```

The trainer always writes local JSONL curves even when W&B is disabled.

## 5. Download and Verify Model, Code Dependency, and Data

The public dataset is hosted at
[`shubhampatle/Pixel-PIVR-Magnified-v2`](https://huggingface.co/datasets/shubhampatle/Pixel-PIVR-Magnified-v2).
The commands below pin the verified snapshot used by this guide. Do not replace
it with `main`, because a moving revision makes resume and result reproduction
ambiguous.

```bash
export DATA_REPO=shubhampatle/Pixel-PIVR-Magnified-v2
export DATA_REVISION=5be1a40c776b494a9ae6ec472b7c611aa9027698
export MODEL_REPO=nvidia/LocateAnything-3B
export MODEL_REVISION=c32291ca5e996f5a7a485845b4f57a233936bba0
export EAGLE_REVISION=8442db3b79f7fd2357e468e6eecdd9b6a82049ff
export DATA_DIR="$WORK_ROOT/Pixel-PIVR-Magnified-v2"
export MODEL_DIR="$WORK_ROOT/LocateAnything-3B"
export EAGLE_DIR="$WORK_ROOT/Eagle"
export HF_XET_HIGH_PERFORMANCE=1

bash scripts/bootstrap_machine.sh download
```

The bootstrap invokes the equivalent dataset transfer with an absolute local
directory:

```bash
"$VENV/bin/hf" download "$DATA_REPO" \
  --repo-type dataset \
  --revision "$DATA_REVISION" \
  --local-dir "$DATA_DIR" \
  --max-workers 32
```

Do **not** run only this lower-level command for a training machine. The Hub
stores images in deterministic tar shards under `archives/`; JSONL records point
to the logical `images/...` paths. The guarded bootstrap continues by validating
all archive members, extracting the images atomically into `images/`, checking
every materialized image hash, downloading the model, preparing Eagle, and
writing the receipt required by preflight.

This single command:

1. Clones Eagle at the pinned revision.
2. Applies and verifies `patches/eagle_virtual_crop_v1.patch`.
3. Downloads LocateAnything at the pinned model revision.
4. Downloads the exact dataset revision with automatic resume support.
5. Verifies every archive size and SHA-256.
6. Safely materializes all image archives without path traversal.
7. Verifies image hashes, annotations, recipes, and split separation.
8. Writes `$WORK_ROOT/download_receipt.json` for preflight.

The current `hf download` resumes automatically when the same command is
rerun; it has no `--resume-download` option. Do not delete the downloaded
archives after materialization because later verification uses them.

Run explicit verification again at any time:

```bash
bash scripts/bootstrap_machine.sh verify-bundle
bash scripts/bootstrap_machine.sh verify-data
python -m json.tool "$WORK_ROOT/download_receipt.json"
```

After successful materialization, the on-disk layout is:

```text
$WORK_ROOT/
|-- LocateAnything-3B/                  model config, tokenizer, and weights
|-- Eagle/
|   `-- Embodied/                       patched LocateAnything training code
|-- Pixel-PIVR-Magnified-v2/
|   |-- manifest.json                   authoritative counts and contracts
|   |-- SHA256SUMS                      package and logical-image checksums
|   |-- recipes/                        frozen Stage 1/Stage 2/validation lists
|   |-- annotations/                    train, validation, and test JSONL shards
|   |-- archives/                       20 downloaded image tar shards
|   `-- images/                         124,325 materialized train/val/test images
`-- download_receipt.json               exact repo, revision, and local paths
```

Confirm that the expected roots exist before generating a training config:

```bash
test -f "$DATA_DIR/manifest.json"
test -f "$DATA_DIR/recipes/stage1_coarse.json"
test -f "$DATA_DIR/recipes/stage2_dense_balanced.json"
test -d "$DATA_DIR/images"
test -f "$MODEL_DIR/config.json"
test -d "$EAGLE_DIR/Embodied"
du -sh "$DATA_DIR" "$MODEL_DIR" "$EAGLE_DIR"
```

## 6. Generate the Node-Specific Configuration

Use the `WORK_ROOT` and `RUN_ROOT` selected in Section 2. The helper copies the
frozen template and writes the exact absolute model, Eagle, dataset, run, code,
and revision values. It refuses to overwrite an existing config unless a new
run is explicitly requested with `FORCE=1`.

```bash
export PIPELINE_CONFIG="$CODE_ROOT/configs/full_scale.env"
mkdir -p "$RUN_ROOT"

DATA_REVISION="$DATA_REVISION" \
  bash scripts/configure_a100_node.sh

sed -n '1,220p' "$PIPELINE_CONFIG"
grep -E '^(MODEL_PATH|EAGLE_ROOT|DATA_ROOT|RUN_ROOT|DATA_REVISION)=' \
  "$PIPELINE_CONFIG"
```

The generated path mapping must be:

```text
MODEL_PATH=$WORK_ROOT/LocateAnything-3B
EAGLE_ROOT=$WORK_ROOT/Eagle/Embodied
DATA_ROOT=$WORK_ROOT/Pixel-PIVR-Magnified-v2
RUN_ROOT=<the absolute run path exported in Section 2>
DATA_REVISION=5be1a40c776b494a9ae6ec472b7c611aa9027698
```

The actual config contains expanded absolute paths rather than the shell
variables shown above. If any path differs from the directory tree in Section
5, stop and correct the exported roots before preflight.

The generated config starts with `FULL_SCALE_APPROVED=NO`. Confirm at least:

```text
ARCHITECTURE_CONTRACT=pixel_crop_144to384_v2
GPU_IDS=0,1,2,3,4,5,6,7
GLOBAL_BATCH=8
VISUAL_CONTEXT=pixel_reencoded
SOURCE_CROP_SIDE=144
LOCAL_INPUT_SIDE=384
IMAGE_TOKEN_LIMIT=6000
MAX_SEQUENCE=32768
LOSS_BALANCING=source_query_task
VALIDATION_RECORDS=0
FULL_SCALE_APPROVED=NO
```

`VALIDATION_RECORDS=0` means use all 2,141 signed monitor records, not zero
validation. Once training starts, do not change world size, batch size, seed,
data revision, sequence limits, task weighting, optimizer settings, or paths.

## 7. Run Mandatory Preflight and Stage-1 Smoke

All configured GPUs must be free.

```bash
cd "$CODE_ROOT"
source "$VENV/bin/activate"
export PIPELINE_CONFIG="$CODE_ROOT/configs/full_scale.env"

mkdir -p "$RUN_ROOT"
set -o pipefail
bash scripts/run_full_pipeline.sh preflight 2>&1 | tee "$RUN_ROOT/preflight.log"
bash scripts/run_full_pipeline.sh smoke-stage1
```

Preflight checks:

- exact Git, Hugging Face, LocateAnything, and Eagle revisions;
- the clean signed source tree and patched Eagle checksums;
- all model shards, package files, recipes, and image hashes;
- zero train/validation/test image-hash overlap;
- exact record counts, task weights, and one-pass steps;
- the real LocateAnything loader on deterministic hard records;
- all eight GPUs, FlashAttention, dependency versions, and disk space.

The smoke test runs real forward, backward, validation, synchronization, and
checkpoint writing on all eight GPUs. Inspect its `done.json`, finite loss, and
peak memory before approving the long run.

Enable training by changing only the approval line:

```bash
sed -i 's/^FULL_SCALE_APPROVED=NO$/FULL_SCALE_APPROVED=YES/' "$PIPELINE_CONFIG"
grep '^FULL_SCALE_APPROVED=' "$PIPELINE_CONFIG"
```

## 8. Train Stage 1: Coarse All-Task Adaptation

Use tmux so a disconnected SSH terminal does not terminate training:

```bash
tmux new -s pixel-pivr
```

Inside the new tmux session, run:

```bash
cd "$CODE_ROOT"
source "$VENV/bin/activate"
export PIPELINE_CONFIG="$CODE_ROOT/configs/full_scale.env"
bash scripts/run_full_pipeline.sh train-stage1
```

Detach with `Ctrl+B`, then `D`. Reattach with:

```bash
tmux attach -t pixel-pivr
```

Training uses ordinary eight-way data parallelism: each GPU holds one complete
model replica and processes one record, then trainable LoRA gradients are
all-reduced and averaged. With eight GPUs, global batch 8, and accumulation 1,
each optimizer update represents eight records. The trainer uses one seeded
global permutation, shards it round-robin across ranks, and does not pack
multiple variable-length records into one sequence.

Stage 1 uses:

| Setting | Value |
|---|---:|
| Records | 460,263 + 1 deterministic padding exposure |
| Optimizer steps | 57,533 |
| Learning rate | 1e-5 |
| Warm-up | 600 steps |
| Checkpoint interval | 1,000 steps |
| Validation interval | 5,000 steps and final step |
| Coverage | Exactly one scheduled pass |

Monitor from another shell:

```bash
cd "$CODE_ROOT"
source "$VENV/bin/activate"
export PIPELINE_CONFIG="$CODE_ROOT/configs/full_scale.env"

bash scripts/run_full_pipeline.sh status-stage1
tail -f "$CODE_ROOT/runs/logs/stage1_coarse.log"
nvidia-smi
```

Stage 1 is complete only when this command prints `True`:

```bash
python - "$RUN_ROOT/stage1_coarse/done.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["complete_one_pass"])
PY
```

Directory or checkpoint existence alone does not prove completion.

## 9. Train Stage 2: Dense Specialization with Replay

Stage 2 is blocked until Stage 1 has a valid `done.json` and `best.pt`. It starts
from the Stage-1 adapter with the lowest held-out monitor loss, but deliberately
creates a fresh AdamW optimizer and cosine scheduler.

```bash
cd "$CODE_ROOT"
source "$VENV/bin/activate"
export PIPELINE_CONFIG="$CODE_ROOT/configs/full_scale.env"

bash scripts/run_full_pipeline.sh smoke-stage2
bash scripts/run_full_pipeline.sh train-stage2
```

Stage 2 uses:

| Setting | Value |
|---|---:|
| Records | 2,100,576 |
| Optimizer steps | 262,572 |
| Learning rate | 5e-6 |
| Warm-up | 1,500 steps |
| Checkpoint interval | 1,000 steps |
| Validation interval | 5,000 steps and final step |
| Coverage | Exactly one scheduled dense+replay pass |

Monitor with:

```bash
bash scripts/run_full_pipeline.sh status-stage2
tail -f "$CODE_ROOT/runs/logs/stage2_dense_balanced.log"
```

Completion requires:

```text
$RUN_ROOT/stage2_dense_balanced/done.json
```

with `"complete_one_pass": true`.

## 10. Interruption and Exact Resume

Press `Ctrl+C` once to request a graceful aligned checkpoint. The trainer
finishes the current optimizer boundary and updates `last.pt`. A second
interrupt can force termination before checkpoint writing finishes.

Resume by rerunning the **same stage command**:

```bash
bash scripts/run_full_pipeline.sh train-stage1
# or
bash scripts/run_full_pipeline.sh train-stage2
```

No explicit `--resume` flag is needed. The trainer restores LoRA weights,
optimizer, scheduler, step, exposure cursor, sample permutation, and per-rank
CPU/CUDA RNG state from `last.pt`. It rejects a changed data hash, world size,
batch/accumulation, seed, sequence budget, visual mode, validation contract, or
initial adapter.

For an already preflighted and explicitly approved unattended run, both stages
and evaluation can be chained without overlap:

```bash
bash scripts/run_full_pipeline.sh all
```

This reruns the smoke gates, starts Stage 2 only after Stage 1 completion, and
evaluates only after Stage 2 completion.

## 11. Training Loss, Validation, and Checkpoints

Each record's cross-entropy is normalized by its supervised-token count, so a
long dense answer does not gain weight merely by containing more tokens.
`source_query_task` then applies deterministic mean-one task weights to preserve
the source-query detection/grounding/pointing mixture after Round-2 expansion.

The trainer records:

```text
$RUN_ROOT/stage1_coarse/training_curve.jsonl
$RUN_ROOT/stage1_coarse/validation_curve.jsonl
$RUN_ROOT/stage1_coarse/best.pt
$RUN_ROOT/stage1_coarse/last.pt
$RUN_ROOT/stage1_coarse/done.json

$RUN_ROOT/stage2_dense_balanced/training_curve.jsonl
$RUN_ROOT/stage2_dense_balanced/validation_curve.jsonl
$RUN_ROOT/stage2_dense_balanced/best.pt
$RUN_ROOT/stage2_dense_balanced/last.pt
$RUN_ROOT/stage2_dense_balanced/done.json
```

W&B logs `train/loss`, `train/native_loss`, `train/loss_weight`, learning rate,
gradient norm, exposure, and peak memory. Validation appears under
`validation/*`. `best.pt` is selected by unweighted held-out
`validation/native_loss`; it is not selected by training loss. `last.pt` is the
newest aligned checkpoint and can differ from `best.pt`.

## 12. Evaluate the Final Model

The default evaluation adapter is Stage 2 `best.pt`. Evaluation data is never
used for gradients or checkpoint selection. Free all configured evaluation GPUs
before launching.

```bash
cd "$CODE_ROOT"
source "$VENV/bin/activate"
export PIPELINE_CONFIG="$CODE_ROOT/configs/full_scale.env"

bash scripts/run_full_pipeline.sh prepare-eval
bash scripts/run_full_pipeline.sh evaluate
```

`evaluate` also rebuilds the manifests, so `prepare-eval` is an optional visible
precheck. It shards each benchmark across all configured GPUs, resumes by
`sample_key`, verifies the expected row count, merges raw predictions, and then
writes:

```text
$RUN_ROOT/evaluation/<adapter-and-decoding-contract>/all_metrics.json
```

Each benchmark directory also contains:

```text
predictions.jsonl   raw model outputs and parsed predictions
execution.jsonl     timing and execution metadata
summary.json        merged metrics and counts
shards/             per-GPU resumable outputs
```

The frozen evaluation suite is:

| Task | Benchmark | Records | GT/notes |
|---|---|---:|---|
| HBB detection | DIOR | 586 | 3,379 boxes, 20 raw classes |
| HBB detection | DOTAv2 | 874 | 29,329 boxes, 18 raw classes |
| Grounding | DIOR-RSVG | 7,500 | Official single-box queries |
| Grounding | VRSBench-VG | 16,154 | Leakage-controlled subset |
| Pointing | DOTAv2 Balanced-100 | 285 | 9,045 targets on 100 images; diagnostic |

Detection reports uncapped class-aware one-to-one matching at IoU 0.5. There is
no `max_dets=100` prediction cap. DOTAv2 keeps `plane`, `small vehicle`, and
`large vehicle` distinct. DIOR keeps `overpass` distinct. The unambiguous model
output alias `airplane -> plane` is accepted, but generic `vehicle` is never
assigned to either size subclass.

Grounding reports first-box Acc@0.5, Acc@0.7, and mIoU. The VRSBench-VG count is
16,154 rather than 16,159 because five official queries share image hashes with
the independent validation pool and are excluded. DOTAv2 pointing is a project
diagnostic, not a universal public pointing benchmark.

To evaluate a diagnostic adapter instead, set `EVAL_ADAPTER` to its absolute
path inside a copy of `configs/full_scale.env`. Do not overwrite or mislabel the
canonical Stage-2 result directory.

## 13. Evaluation Correctness Guards

The evaluator always reconstructs manifests from the packaged raw annotations.
It never scores the package's convenience `class_name` when the official
`raw_class_name` is available.

The synchronized 16K/4K pilot has an additional frozen builder:

```bash
python tools/prepare_dotav2_balanced100.py \
  --annotation /path/to/DOTAv2_balanced100.jsonl \
  --image-root /path/to/DOTAv2/images \
  --output /path/to/DOTAv2_balanced100_raw_labels.jsonl
```

It fails unless the set has exactly 100 images, 9,045 boxes, and the complete
signed 18-class distribution. It specifically rejects the incorrect historical
collapses:

```text
small vehicle + large vehicle -> vehicle
plane -> airplane
```

Pilot inference must use `--point-address-prompt-schema pilot_compact_ref`,
which exactly matches that adapter's Round-2 training prompt. Full-scale
Magnified-v2 uses the separate `compact` local-completeness schema. The names are
deliberately distinct to prevent evaluation-time prompt drift.

## 14. Common Failures

**`DATA_REVISION must be a 40-character commit SHA`**

Use the owner-verified immutable Hugging Face revision, not a branch name.

**Hugging Face download was interrupted**

Rerun `bash scripts/bootstrap_machine.sh download` with the same paths and
revision. Do not add the removed `--resume-download` option.

**A checksum or archive verification fails**

Do not train. Rerun the download. If it fails again, compare the requested
revision with the verified release revision.

**`GPU N already has a compute process`**

Inspect `nvidia-smi`. Do not kill another user's process. Start only after every
GPU listed in `GPU_IDS` or `EVAL_GPU_IDS` is free.

**Preflight reports a dirty source checkout**

Do not bypass it. Commit intended source changes or use a fresh checkout of the
verified commit. Machine-local `.env`, `.venv`, `runs`, data, checkpoints, and
logs are already ignored.

**Training stopped without `done.json`**

The run is incomplete. Inspect `last.pt`, `status.json`, and the stage log, then
rerun the identical stage command to resume.

**Evaluation output already exists**

Evaluation resumes completed sample keys and skips complete merged benchmarks.
Use the same command for a real resume. Use a new output/config only when the
adapter or evaluation contract intentionally changes.

## 15. Local Release Verification

Run these checks before committing or transferring the repository:

```bash
cd "$CODE_ROOT"
source .venv/bin/activate

python tools/verify_release.py
pytest -q
for script in scripts/*.sh; do bash -n "$script" || exit 1; done
python tools/source_manifest.py check
git diff --check
```

## Reproducibility Evidence to Keep

Retain the exact Git and Hugging Face revisions, generated config, download
receipt, preflight report, smoke receipts, both stage directories, W&B IDs,
training/validation curves, `best.pt`, `last.pt`, `done.json`, evaluation
manifests, raw predictions, execution logs, and `all_metrics.json`. These files
are needed to reproduce and audit any reported result.

Source datasets retain their original licenses. Review every source license and
redistribution condition before publishing derived images or using the package
commercially. See [the full A100 runbook](docs/A100_8GPU_FULL_SCALE_RUNBOOK.md),
[data contract](docs/DATA_FORMAT.md), [inference contract](docs/INFERENCE.md),
and [dataset release procedure](docs/HUGGINGFACE_DATASET.md) for deeper detail.
