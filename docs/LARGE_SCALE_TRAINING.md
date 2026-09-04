# Full-Scale Training Contract

The release run is Qwen-only rank-16 LoRA adaptation of LocateAnything-3B. It is
not full-parameter fine-tuning: MoonViT and the released multimodal projector are
frozen and kept in evaluation mode.

## Eight-GPU data parallelism

Each of eight A100 80 GB GPUs owns one complete model replica and processes one
JSONL record per optimizer update. Trainable LoRA tensors are broadcast from rank
0 before the first step. Gradients are all-reduced and divided by world size, so
the update is the mean of eight records. BF16 and MoonViT FlashAttention 2 are
enabled; Qwen uses SDPA because that is the tested LocateAnything path.

This design uses all eight GPUs and avoids sequence padding between highly
variable dense records. It does not claim perfect memory saturation. Increasing
per-GPU batch or packing records would change the optimization and memory contract
and must be benchmarked as a separate experiment.

## Exact one-pass schedule

For `N` records and global batch `B=8`:

```text
padding = (-N) mod 8
optimizer steps = (N + padding) / 8
```

`MAX_STEPS=0` forces the launcher to derive this value from the signed recipe.
The schedule is one seeded global permutation distributed round-robin across the
eight ranks. Every source row occurs exactly once before only the minimal prefix
padding; no row is silently dropped. `done.json` records unique records, total
exposures, padding, and `complete_one_pass`.

The exact counts must be read from the downloaded dataset's `manifest.json`; the
launcher and preflight independently recalculate them. Stage 2's replay fraction
is also reported there at the optimizer-record level after Round-2 expansion.

## Loss normalization and task balance

For each record, cross-entropy is summed over non-ignored target tokens and divided
by that record's supervised-token count. Long dense outputs therefore do not receive
extra weight merely for having more target tokens.

Round 2 expands one source query into one row per address. A plain row mean would
make Stage 2 approximately 97% detection, even though its source-query mixture is
46.65% detection, 32.52% grounding, and 20.83% pointing. The frozen
`source_query_task` policy assigns a deterministic scalar weight by task so the
aggregate optimization contribution recovers that source-query mixture while still
visiting every flattened row once. The weights have mean one over each stage and
are derived from the signed recipe during preflight; they are not sampling
probabilities and do not change coverage.

`training_curve.jsonl` and W&B separately record weighted `loss`, unweighted
`native_loss`, and `loss_weight`. Validation remains unweighted and checkpoint
selection always uses validation `native_loss`.

## Validation and checkpoint selection

The full independent validation pool is never used for gradients. A fixed all-task
monitor combines all Round-1 validation records with a controlled Round-2 subset,
including detection, grounding, pointing, positives, negatives, local boxes, and
fallbacks. Its recipe and exact count are signed into every run contract.

Validation is sharded across all eight GPUs every 5,000 optimizer steps and at the
final step. `best.pt` tracks the lowest held-out monitor `native_loss`; `last.pt`
tracks the latest aligned checkpoint. The two can point to the same file.

## Two stages

Stage 1 trains once over coarse detection, grounding, and pointing records. Stage 2
starts from Stage 1 `best.pt`, uses all dense records plus the source-defined replay
records, and trains for one new exact pass. Stage 2 intentionally starts a fresh
AdamW optimizer and cosine scheduler; it does not resume Stage 1 optimizer state.

| Setting | Stage 1 | Stage 2 |
|---|---:|---:|
| Qwen LoRA rank | 16 | 16 |
| Learning rate | 1e-5 | 5e-6 |
| Warm-up | 600 steps | 1,500 steps |
| Weight decay | 0.01 | 0.01 |
| Gradient clipping | 1.0 | 1.0 |
| Checkpoint interval | 1,000 | 1,000 |
| Validation interval | 5,000 | 5,000 |
| Epoch-equivalent coverage | exactly 1 pass | exactly 1 pass |

The schedule decays to 10% of its peak learning rate rather than zero. All values
are part of the resume contract and cannot be changed inside an existing run.

## Resume guarantees

Every aligned checkpoint stores LoRA weights, optimizer, scheduler, global step,
exact exposure, per-rank CPU/CUDA RNG state, world size, accumulation, initial
adapter identity, model/data paths, and SHA-256/size/line count for every train and
validation shard. It also signs the learning rate, warm-up, weight decay,
gradient clipping, attention backend, worker count, and checkpoint/evaluation
cadence. One SIGINT or SIGTERM requests a checkpoint at the next optimizer
boundary. Rerunning the identical command reconstructs the same sample permutation
from the saved cursor. If the process exits after the final aligned checkpoint but
before writing `done.json`, restart reconstructs that marker from the verified final
state; orphaned checkpoint files without `last.pt` are rejected for manual review.

## Sequence and visual budgets

`IMAGE_TOKEN_LIMIT=6000` is the per-image pre-merge MoonViT patch-token ceiling,
not the final Qwen visual-token count. The 144-to-384 local view is aligned by LA to
392 x 392, creating 784 patch tokens and 196 projected tokens after the native 2 x 2
merge. A local Round-2 record includes global and local visual sequences; the Qwen
limit is 32,768. Preflight verifies the model context, and smoke exercises forward,
backward, validation, checkpoint write, and distributed synchronization.

See `docs/A100_8GPU_FULL_SCALE_RUNBOOK.md` for the exact operator sequence.
