# Hugging Face Magnified-v2 Release

The full-scale release is `pixel-pivr-hf-hbb-magnified-v2`. It is generated from
the leak-free `pixel_pivr_type2_hbb_full_v1` curriculum, but expands every Round-2
row into strict local-first 144-to-384 pixel re-entry supervision and any required
observable global fallback.

Do not upload or train from the older `shubhampatle/Pixel-PIVR` repository. It is
the earlier `pixel-pivr-hf-hbb-v1` schema and is not the final magnified variant.

## Build the materialized package

Use a new, empty destination. The tool refuses to overwrite an existing package.

```bash
cd /path/to/Pixel-PIVR
source .venv/bin/activate

export MATERIALIZED=/path/to/Pixel-PIVR-HF-v2-release
python tools/package_hf_magnified_v2.py plan
python tools/package_hf_magnified_v2.py build \
  --output "$MATERIALIZED" \
  --existing-image-root /path/to/verified/Pixel-PIVR-HF-v1 \
  --trust-record-hashes

python tools/package_hf_magnified_v2.py verify \
  --output "$MATERIALIZED" \
  --verify-image-hashes
```

The final verification reads every image byte. It also checks exact containment,
the frozen two-pixel edge-retry margin, box-only Round-2 grammar, local/global
fallback pairing, recipe counts, image inventory, all metadata checksums, and
zero train/validation/test image-hash overlap.

The test package uses 16,154 VRSBench-VG queries. This is intentional: five of
the 16,159 official queries share image hashes with the independent validation
pool and are excluded to keep checkpoint selection disjoint from reported test
results. The manifest records the exact retained count.

## Build the low-file-count upload bundle

Uploading more than 100,000 individual image paths is slow and fragile. The Hub
artifact therefore carries deterministic uncompressed tar shards while retaining
the logical `images/...` checksums and paths.

```bash
export BUNDLE=/path/to/Pixel-PIVR-Magnified-v2-upload
python tools/build_hf_upload_bundle.py build \
  --source "$MATERIALIZED" \
  --output "$BUNDLE" \
  --archive-gib 4

python tools/build_hf_upload_bundle.py verify \
  --output "$BUNDLE" \
  --verify-member-hashes
```

The bundle verifier reads every archive member, checks member sizes and SHA-256,
and proves that each logical image appears exactly once.

## Upload and verify the remote snapshot

Keep the repository private until source-dataset redistribution terms have been
reviewed.

```bash
hf auth login
export BUNDLE_ROOT="$BUNDLE"
export HF_REPO=shubhampatle/Pixel-PIVR-Magnified-v2
export PYTHON_BIN="$PWD/.venv/bin/python"
export HF_BIN="$PWD/.venv/bin/hf"
bash scripts/publish_hf_dataset.sh check
bash scripts/publish_hf_dataset.sh upload | tee hf_upload_and_verification.log
```

The guarded publisher refuses the legacy repository name, verifies every local
archive member before upload, creates the new repository as private, uses the
resumable `upload-large-folder` path, resolves the resulting immutable commit,
and compares the complete remote path/size/hash inventory. Rerun the identical
`upload` command after a network or terminal interruption. It does not need
`--resume-download`.

Resolve the immutable Hub commit and compare the complete remote path, size, and
cryptographic digest inventory with the local bundle:

```bash
DATA_REVISION="$(python - <<'PY'
from huggingface_hub import HfApi
print(HfApi().dataset_info('shubhampatle/Pixel-PIVR-Magnified-v2').sha)
PY
)"

DATA_REVISION="$DATA_REVISION" \
  bash scripts/publish_hf_dataset.sh verify-remote

printf 'DATA_REVISION=%s\n' "$DATA_REVISION"
```

Do not issue a remote-training handoff unless this command writes a passed
`remote_verification.json`.

## Download on the A100 node

Use the exact verified revision, never `main`:

```bash
export WORK_ROOT=/path/to/pixel-pivr-assets
export DATA_REPO=shubhampatle/Pixel-PIVR-Magnified-v2
export DATA_REVISION=<VERIFIED_40_CHARACTER_HF_COMMIT>
bash scripts/bootstrap_machine.sh download
```

Bootstrap validates all archives, materializes images atomically, verifies every
materialized image SHA-256, and writes `download_receipt.json`. Reusing the same
paths and command safely resumes or revalidates the transfer.
