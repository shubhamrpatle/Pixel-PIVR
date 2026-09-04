#!/usr/bin/env python3
"""Compare a Hugging Face dataset revision with the local upload bundle."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.hf_api import RepoFile

from build_hf_upload_bundle import GENERATED_REPORTS, parse_checksums
from package_hf_dataset import sha256_file, write_json


def local_files(root: Path) -> dict[str, Path]:
    return {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file()
        and ".cache" not in path.relative_to(root).parts
        and path.name not in GENERATED_REPORTS
    }


def verify(repo_id: str, local: Path, revision: str | None) -> dict[str, Any]:
    local = local.resolve()
    api = HfApi()
    info = api.dataset_info(repo_id, revision=revision, files_metadata=True)
    resolved_revision = str(info.sha)
    remote: dict[str, RepoFile] = {}
    for value in api.list_repo_tree(
        repo_id, recursive=True, expand=True, revision=resolved_revision, repo_type="dataset"
    ):
        if isinstance(value, RepoFile):
            remote[value.path] = value
    expected = local_files(local)
    allowed_remote_extras = {".gitattributes"}
    missing = sorted(set(expected) - set(remote))
    extra = sorted(set(remote) - set(expected) - allowed_remote_extras)
    if missing or extra:
        raise RuntimeError(f"Remote path mismatch: missing={missing[:20]}, extra={extra[:20]}")

    checksums = parse_checksums(local / "SHA256SUMS")
    lfs_verified = downloaded_verified = 0
    with tempfile.TemporaryDirectory(prefix="pixel-pivr-hf-verify-") as cache:
        for relative, path in expected.items():
            value = remote[relative]
            if int(value.size) != path.stat().st_size:
                raise RuntimeError(f"Remote size mismatch: {relative}")
            expected_hash = (
                sha256_file(path)
                if relative == "SHA256SUMS"
                else checksums.get(relative)
            )
            if expected_hash is None:
                expected_hash = sha256_file(path)
            if value.lfs is not None and value.lfs.sha256:
                if str(value.lfs.sha256) != expected_hash:
                    raise RuntimeError(f"Remote LFS SHA-256 mismatch: {relative}")
                lfs_verified += 1
                continue
            if path.stat().st_size > 64 * 1024 * 1024:
                raise RuntimeError(
                    f"Large remote file has no cryptographic LFS/Xet digest: {relative}"
                )
            downloaded = Path(
                hf_hub_download(
                    repo_id,
                    filename=relative,
                    repo_type="dataset",
                    revision=resolved_revision,
                    cache_dir=cache,
                )
            )
            if sha256_file(downloaded) != expected_hash:
                raise RuntimeError(f"Downloaded remote checksum mismatch: {relative}")
            downloaded_verified += 1
    report = {
        "status": "passed",
        "repo_id": repo_id,
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "remote_files": len(remote),
        "expected_bundle_files": len(expected),
        "lfs_or_xet_sha256_verified": lfs_verified,
        "downloaded_git_files_verified": downloaded_verified,
        "allowed_remote_extras_present": sorted(set(remote) & allowed_remote_extras),
    }
    write_json(local / "remote_verification.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--revision")
    args = parser.parse_args()
    print(json.dumps(verify(args.repo_id, args.local_dir, args.revision), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
