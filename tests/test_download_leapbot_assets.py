import hashlib
import json
from pathlib import Path

import pytest

from scripts.download_leapbot_assets import (
    CHECKPOINT_REVISION,
    DATASET_REVISION,
    _complete_asset_files_exist,
    _expected_assets,
    _write_download_manifest,
)


def _materialize_asset(root: Path, path: Path, revision: str, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    metadata = (
        path.parent
        / ".cache"
        / "huggingface"
        / "download"
        / f"{path.name}.metadata"
    )
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(f"{revision}\nremote-etag\n0\n")


def test_complete_manifest_hashes_exact_pinned_downloads(tmp_path):
    expected = _expected_assets(tmp_path)
    assert len(expected) == 6
    assert not _complete_asset_files_exist(tmp_path)
    for index, (path, _, revision) in enumerate(expected):
        _materialize_asset(tmp_path, path, revision, f"asset-{index}".encode())
    assert _complete_asset_files_exist(tmp_path)

    output = _write_download_manifest(tmp_path)
    payload = json.loads(output.read_text())
    claimed = payload.pop("manifest_sha256")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    assert claimed == hashlib.sha256(canonical).hexdigest()
    assert {item["revision"] for item in payload["files"]} == {
        DATASET_REVISION,
        CHECKPOINT_REVISION,
    }
    for item in payload["files"]:
        assert len(item["sha256"]) == 64
        assert item["bytes"] > 0


def test_manifest_rejects_wrong_huggingface_revision(tmp_path):
    for index, (path, _, revision) in enumerate(_expected_assets(tmp_path)):
        if index == 0:
            revision = "0" * 40
        _materialize_asset(tmp_path, path, revision, f"asset-{index}".encode())
    with pytest.raises(ValueError, match="download revision mismatch"):
        _write_download_manifest(tmp_path)
