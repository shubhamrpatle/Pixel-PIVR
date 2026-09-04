# Code and Data Transfer

GitHub carries code only. Hugging Face carries the versioned dataset bundle.
Model weights, materialized images, checkpoints, caches, W&B files, and run
outputs are excluded by `.gitignore`.

## GitHub release

Before committing:

```bash
cd /path/to/Pixel-PIVR
source .venv/bin/activate
python tools/verify_release.py
PYTHONPATH=src pytest -q
bash -n scripts/*.sh
python tools/source_manifest.py check
git diff --check
git status --short
```

Commit and push only after every command passes:

```bash
git add .
git commit -m "Release magnified-v2 full-scale pipeline"
git push origin main
git rev-parse HEAD
git ls-remote origin refs/heads/main
```

The two revisions must match. Supply that 40-character commit to the A100
operator; the remote preflight rejects any other checkout or modified tracked
file.

## Dataset release

Follow `docs/HUGGINGFACE_DATASET.md`. The final flow is:

1. Build and fully verify a materialized magnified-v2 package.
2. Build deterministic image tar shards and verify every archive member.
3. Upload to `shubhampatle/Pixel-PIVR-Magnified-v2` with
   `hf upload-large-folder`.
4. Resolve the immutable Hub commit.
5. Run `tools/verify_hf_snapshot.py` against that exact remote commit.
6. Give the passed `DATA_REVISION` to the A100 operator.

Do not use the older `shubhampatle/Pixel-PIVR` snapshot for magnified-v2.

## Destination machine

The destination should not receive a manually assembled subset. Clone the exact
Git commit and let `scripts/bootstrap_machine.sh download` fetch the exact model
and dataset commits, patch the pinned Eagle checkout, validate archives, and
materialize the data. Then follow
`docs/A100_8GPU_FULL_SCALE_RUNBOOK.md` from preflight through evaluation.

Source datasets retain their own licenses. Keep the Hub repository private until
redistribution terms have been reviewed.
