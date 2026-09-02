# Transfer And GitHub Release

The repository intentionally excludes model weights, images, JSONL datasets,
checkpoints, caches, and experiment outputs. Transfer those separately and keep
their paths in a machine-local `configs/large_scale.env`, which is ignored by
Git.

## Create A Verified Source Archive

From the directory containing `Pixel-PIVR`:

```bash
cd /path/to/Pixel-PIVR
python tools/verify_release.py
PYTHONPATH=src python -m unittest discover -s tests -v
cd ..
tar --exclude='Pixel-PIVR/.git' \
    --exclude='*/__pycache__' \
    --exclude='*.pyc' \
    --exclude='Pixel-PIVR/.cache' \
    --exclude='Pixel-PIVR/runs' \
    -czf Pixel-PIVR-source.tar.gz Pixel-PIVR
sha256sum Pixel-PIVR-source.tar.gz > Pixel-PIVR-source.tar.gz.sha256
```

On the destination machine:

```bash
sha256sum -c Pixel-PIVR-source.tar.gz.sha256
tar -xzf Pixel-PIVR-source.tar.gz
cd Pixel-PIVR
python tools/verify_release.py
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Upload To GitHub

Create an empty repository on GitHub, then run locally:

```bash
cd /path/to/Pixel-PIVR
git init
git add .
git commit -m "Release standalone Pixel-PIVR implementation"
git branch -M main
git remote add origin git@github.com:YOUR_ACCOUNT/Pixel-PIVR.git
git push -u origin main
```

Choose and add a license before making the repository public. This package does
not redistribute LocateAnything, Eagle, or model weights; review their licenses
and cite them separately.

## Data And Checkpoints

Use `rsync` with partial-transfer support for large private assets:

```bash
rsync -ahP --partial --append-verify /source/train.jsonl user@host:/data/pixel_pivr/
rsync -ahP --partial --append-verify /source/images/ user@host:/data/pixel_pivr/images/
rsync -ahP --partial --append-verify /source/LocateAnything-3B/ user@host:/models/LocateAnything-3B/
```

After transfer, run `pixel-pivr-audit` on the destination and compare the JSONL
SHA256 values in its report with the source report. Then populate
`configs/large_scale.env` and run `check`, `smoke`, and finally `train` in that
order.
