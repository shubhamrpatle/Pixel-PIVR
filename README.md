# Pixel-PIVR

Pixel-PIVR is an isolated implementation of point-indexed visual re-entry for
NVIDIA LocateAnything-3B. It contains the tested **pixel-reencoded** mechanism
and an experimental **single-encode magnified pre-projector ROI** variant.

The release keeps LocateAnything's language reasoning and native six-token HBB
grammar. It adds a two-round inference path:

1. LocateAnything predicts typed point addresses from the global image.
2. Each predicted point defines a lossless local pixel crop. The shared MoonViT
   re-encodes that crop, while the global projected visual tokens are cached once.
3. Qwen computes the global visual/text prefix once and stores one layer-wise KV
   cache per image. Each address adds only its local visual/text suffix.
4. The shared Qwen decoder emits exactly one constrained PBD6 HBB per address.
   Independent branches run sequentially or in bounded waves; no model or
   persistent global cache is copied per point.

```text
global image + category query
             |
             v
      typed point addresses
             |
       +-----+-----------------------------+
       |                                   |
       v                                   v
global MoonViT/projector             local pixel crops
cached once                                |
       |                            shared MoonViT/projector
       v                                   |
Qwen global-prefix KV once                 |
       +-------------------+---------------+
                           v
                branch-specific suffixes
                           |
               shared-prefix Qwen LoRA
                           |
          sequential or bounded-wave PBD6
                           |
                exactly one HBB/address
```

## Scope

This repository contains the mechanism that was actually tested:

- HBB detection;
- LocateAnything-3B;
- Qwen rank-16 LoRA;
- frozen MoonViT and released multimodal projector;
- point discovery plus global/local pixel re-entry supervision;
- sequential and wave geometry decoding.

OBB, grounding, pointing, full-parameter training, and second-backbone support are
research extensions, not validated features of this release. Keeping that boundary
explicit prevents a pilot result from being presented as a completed all-task model.

The pre-projector variant is implemented and contract-tested but has not yet earned
an accuracy claim. It taps MoonViT before its fixed 2x2 merge, gathers a 378-pixel
effective point-centred field, applies overlapping 2x2 groups through LA's unchanged
projector, and appends 676 local decoder-width tokens without a second vision pass.
See `docs/MAGNIFIED_PREPROJECTOR.md`.

| Visual path | Local MoonViT calls | Local source | Tokens/address |
|---|---:|---|---:|
| Pixel-reencoded PIVR | 1 | separately encoded pixel crop | crop dependent |
| Cached post-projector ROI | 0 | already merged global tokens | 144 in the matched ablation |
| Magnified pre-projector ROI | **0** | 27x27 unmerged MoonViT cells | **676** |

## External Requirements

Pixel-PIVR does not redistribute NVIDIA LocateAnything weights or Eagle source.
Provide:

- a local LocateAnything checkpoint containing its trusted remote-code files;
- an Eagle checkout whose `Embodied` directory exposes
  `eaglevl.train.locany_finetune_magi_stream.LazySupervisedDatasetMTP`;
- Pixel-PIVR JSONL records and their referenced images.

The exact environment used for the matched run is listed in
`requirements.txt`. Install the package in a dedicated environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Data Contract

Training JSONL contains two routes:

```text
global_point_discovery
  image: one global image
  target: grouped <ref>class</ref><box><x><y></box> points

point_indexed_visual_reentry
  image: [global image, lossless local crop]
  target: <ref>class</ref><box><x1><y1><x2><y2></box>
```

Coordinates are integers normalized to `[0, 1000]`. Re-entry positives contain
exactly one local HBB, and the local address point lies inside it. See
`docs/DATA_FORMAT.md`.

Audit before training:

```bash
pixel-pivr-audit \
  --data-root /absolute/workspace/root \
  --jsonl /path/train.jsonl /path/validation.jsonl \
  --report /path/audit.json
```

Add `--exact-loader --model /path/to/LocateAnything-3B --eagle-root
/path/to/Eagle/Embodied` to pass every row through LocateAnything's real processor
and MTP loader.

## Distributed Training

Copy `configs/large_scale.env.example`, edit only the paths and desired schedule,
then run:

```bash
cp configs/large_scale.env.example configs/large_scale.env
RUN_CONFIG="$PWD/configs/large_scale.env" bash scripts/train_distributed.sh check
RUN_CONFIG="$PWD/configs/large_scale.env" bash scripts/train_distributed.sh smoke
RUN_CONFIG="$PWD/configs/large_scale.env" bash scripts/train_distributed.sh train
```

The launcher supports 1, 2, or 4 GPUs while preserving the configured global
record batch. With `MAX_STEPS=0`, it derives the exact number of optimizer updates
needed to visit every record once and refuses non-divisible coverage. Checkpoints
contain optimizer, scheduler, per-rank RNG, data signatures, exposure, and LoRA
state. Restarting the same command resumes from `last.pt`.

For a fresh second stage initialized from Stage 1's best adapter, set
`INIT_ADAPTER=/path/stage1/best.pt` and use a new output directory. This loads only
the learned adapter weights and starts a fresh optimizer/scheduler.

For the 6K single-encode magnified variant, start from
`configs/magnified_preprojector_16k.env.example` and run the mandatory `audit`
mode before `smoke` and `train`. Its example signs the expected 16K/1K split,
checks benchmark holdout hashes, and executes 4,000 exact-coverage optimizer steps
at global batch four.

## Inference

The standalone inference manifest format is documented in `docs/INFERENCE.md`.
Run sequential refinement with `--wave-size 1` or bounded wave refinement with a
value from 2 to 200. Fully shared-prefix caching is the default; use
`--prefix-cache-mode recompute` only as the historical reference path:

```bash
CUDA_VISIBLE_DEVICES=0 pixel-pivr-infer \
  --model /path/to/LocateAnything-3B \
  --adapter /path/to/best.pt \
  --manifest /path/eval.jsonl \
  --output /path/predictions \
  --prefix-cache-mode shared \
  --wave-size 200
```

## Matched Pilot Result

On the frozen DOTAv2 Balanced-100 diagnostic set (100 images, 9,045 HBB GT, no
prediction cap, class-aware one-to-one IoU-0.5 matching):

| Method | Precision | Recall | F1@0.5 |
|---|---:|---:|---:|
| Matched standard LoRA | 10.31 | 3.23 | 4.92 |
| Previous PIVR | 39.26 | 6.06 | 10.50 |
| Pixel-PIVR sequential | 37.50 | 6.24 | **10.69** |
| Pixel-PIVR wave | 37.22 | 6.18 | 10.60 |

Those table values predate fully shared Qwen-prefix caching: the historical wave
batched branches but recomputed the global decoder prefix. They remain the frozen
accuracy reference and are not presented as shared-prefix latency results. The
new path must first pass exact real-checkpoint equivalence and a matched latency
rerun. This is one-seed, one-subset mechanism evidence, not a publication-scale
or SOTA claim. Exact metadata is under `reference/`.

`reference/release_verification.json` records the standalone package's real-data
audit, two-GPU forward/backward smoke, completion rerun, and inference smoke.

## Reproducibility

Run the repository checks before transfer or upload:

```bash
python tools/verify_release.py
PYTHONPATH=src python -m unittest discover -s tests -v
```

See `docs/LARGE_SCALE_TRAINING.md` for exact-coverage and resume behavior, and
`docs/TRANSFER.md` for a verified source/GitHub transfer procedure.
