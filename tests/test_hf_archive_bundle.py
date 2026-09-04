import hashlib
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from build_hf_upload_bundle import build, verify


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def test_archive_bundle_has_complete_verified_image_coverage():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "materialized"
        output = root / "upload"
        payloads = (b"first-image", b"second-image-is-larger")
        inventory = []
        for index, payload in enumerate(payloads):
            digest = hashlib.sha256(payload).hexdigest()
            relative = Path("images/train") / digest[:2] / f"{digest}.jpg"
            path = source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            inventory.append(
                {"bytes": len(payload), "path": relative.as_posix(), "sha256": digest}
            )
        inventory.sort(key=lambda row: row["path"])
        (source / "IMAGE_INVENTORY.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in inventory),
            encoding="utf-8",
        )
        write_json(source / "manifest.json", {"schema_version": "fixture-v1"})
        (source / "README.md").write_text("fixture\n", encoding="utf-8")
        (source / "annotations/train/sample.jsonl").parent.mkdir(
            parents=True, exist_ok=True
        )
        (source / "annotations/train/sample.jsonl").write_text("{}\n", encoding="utf-8")
        write_json(source / "recipes/stage1.json", {"annotation": []})

        result = build(source, output, target_bytes=len(payloads[0]))
        assert result["status"] == "passed"
        assert result["archived_images"] == 2
        assert not (output / "images").exists()
        assert len(list((output / "archives").glob("*.tar"))) == 2

        strong = verify(output, verify_member_hashes=True)
        assert strong["archive_payload_sha256_verified"] is True
        assert strong["archived_images"] == 2
