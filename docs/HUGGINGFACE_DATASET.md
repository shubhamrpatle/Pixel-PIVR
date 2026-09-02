# Hugging Face Dataset Release

The full-scale HBB corpus is packaged from the independently verified
`pixel_pivr_type2_hbb_full_v1` curriculum. The packager rewrites machine-local
image references, content-addresses every image, and separates train,
validation, and test by image SHA-256.

## Build

```bash
cd /absolute/path/to/Pixel-PIVR
export PACKAGE_ROOT=/absolute/path/to/Pixel-PIVR-HF

python tools/package_hf_dataset.py plan
python tools/package_hf_dataset.py build \
  --output "$PACKAGE_ROOT" \
  --link-mode hardlink
python tools/package_hf_dataset.py verify \
  --output "$PACKAGE_ROOT"
```

`hardlink` stores no second copy of image bytes when source and destination are
on the same filesystem. Use `--link-mode copy` for a physically independent
release directory.

Run an expensive byte-level image audit before publication:

```bash
python tools/package_hf_dataset.py verify \
  --output "$PACKAGE_ROOT" \
  --verify-image-hashes
```

## Upload

Authenticate as the owner of the requested namespace, create the dataset repo,
and upload the generated directory:

```bash
source .venv/bin/activate
hf auth login
hf repo create shubhampatle/Pixel-PIVR --repo-type dataset --private --exist-ok
hf upload-large-folder shubhampatle/Pixel-PIVR \
  "$PACKAGE_ROOT" \
  --repo-type dataset \
  --num-workers 8
```

`upload-large-folder` records resumable task state inside the local folder. If
the network or terminal stops, rerun the same command to continue.

The source datasets retain their individual licenses. Resolve redistribution
permission for every included image source before making the repository public.
Keep the initial Hub repository private while completing that review.

## Use on a new machine

The code repository consumes the package without path rewriting:

```bash
hf download shubhampatle/Pixel-PIVR \
  --repo-type dataset \
  --local-dir /absolute/path/Pixel-PIVR-data

cd /absolute/path/Pixel-PIVR
cp configs/full_scale.env.example configs/full_scale.env
# Set DATA_ROOT to /absolute/path/Pixel-PIVR-data and edit the other four paths.
PIPELINE_CONFIG="$PWD/configs/full_scale.env" \
  bash scripts/run_full_pipeline.sh preflight
```

The Stage 1, Stage 2, validation, and test paths are read from package recipes.
No test annotation is accepted by the training orchestration.
