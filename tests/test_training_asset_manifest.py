import hashlib
import json
from pathlib import Path

import pytest

from leapbot_va.conditioning_assets import (
    build_text_cache_provenance,
    write_text_cache_provenance,
)
from scripts.training_asset_manifest import (
    EXPECTED_DOWNLOAD_ASSETS,
    build_manifest,
    hash_tree,
)


TASK = "pick up the block"
PROMPT = (
    "A video recorded from a robot's point of view executing the following "
    f"instruction: {TASK}"
)
CACHE_FILENAME = (
    f"{hashlib.sha256(PROMPT.encode('utf-8')).hexdigest()}."
    "t5_len128.wan22ti2v5b.pt"
)


def _write_dataset(directory: Path, value: str) -> None:
    directory.mkdir()
    (directory / "sample").write_text(value)
    metadata = directory / "meta"
    metadata.mkdir()
    (metadata / "tasks.jsonl").write_text(json.dumps({"task": TASK}) + "\n")


def _conditioning_assets(
    tmp_path: Path,
    cache: Path,
    *,
    verification_method: str = "online_source_forward_cache_tensor_exact",
) -> tuple[Path, Path]:
    encoder = tmp_path / "text-encoder.safetensors"
    encoder.write_bytes(b"encoder")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir(exist_ok=True)
    (tokenizer / "tokenizer.json").write_bytes(b"tokenizer")
    provenance = build_text_cache_provenance(
        cache_dir=cache,
        cache_file_names=[CACHE_FILENAME],
        model_id="Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id="Wan-AI/Wan2.1-T2V-1.3B",
        redirect_common_files=True,
        context_len=128,
        text_encoder_path=encoder,
        tokenizer_path=tokenizer,
        verified_file_count=1,
        verification_method=verification_method,
    )
    write_text_cache_provenance(cache, provenance)
    return encoder, tokenizer


def _download_manifest(path: Path) -> None:
    files = [
        {
            "path": asset_path,
            "repo": repo,
            "revision": revision,
            "bytes": index + 1,
            "sha256": f"{index:064x}",
        }
        for index, (asset_path, (repo, revision)) in enumerate(
            EXPECTED_DOWNLOAD_ASSETS.items(), start=1
        )
    ]
    payload = {"schema_version": 1, "files": files}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    path.write_text(json.dumps(payload))


def test_tree_hash_changes_with_path_or_content(tmp_path):
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "a").write_bytes(b"one")
    first = hash_tree(tree)
    (tree / "a").write_bytes(b"two")
    assert hash_tree(tree)["sha256"] != first["sha256"]
    (tree / "a").rename(tree / "b")
    assert hash_tree(tree)["sha256"] != first["sha256"]


def test_manifest_binds_four_datasets_cache_vae_and_download_identity(tmp_path):
    datasets = []
    for index in range(4):
        directory = tmp_path / f"dataset-{index}"
        _write_dataset(directory, str(index))
        datasets.append(directory)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / CACHE_FILENAME).write_bytes(b"embedding")
    encoder, tokenizer = _conditioning_assets(tmp_path, cache)
    vae = tmp_path / "vae.pt"
    vae.write_bytes(b"vae")
    download = tmp_path / "download.json"
    _download_manifest(download)

    first = build_manifest(datasets, cache, vae, download, encoder, tokenizer)
    assert first["dataset_file_count"] == 8
    assert first["text_embedding_cache"]["file_count"] == 1
    assert first["vae_checkpoint"]["sha256"]
    (cache / CACHE_FILENAME).write_bytes(b"changed")
    _conditioning_assets(tmp_path, cache)
    second = build_manifest(datasets, cache, vae, download, encoder, tokenizer)
    assert second["manifest_sha256"] != first["manifest_sha256"]

    relocated = tmp_path / "relocated"
    relocated.mkdir()
    relocated_datasets = []
    for index, source in enumerate(datasets):
        destination = relocated / source.name
        _write_dataset(destination, str(index))
        relocated_datasets.append(destination)
    relocated_cache = relocated / "cache"
    relocated_cache.mkdir()
    (relocated_cache / CACHE_FILENAME).write_bytes(b"changed")
    relocated_encoder, relocated_tokenizer = _conditioning_assets(
        relocated, relocated_cache
    )
    relocated_vae = relocated / "vae.pt"
    relocated_vae.write_bytes(b"vae")
    assert (
        build_manifest(
            relocated_datasets,
            relocated_cache,
            relocated_vae,
            download,
            relocated_encoder,
            relocated_tokenizer,
        )["manifest_sha256"]
        == second["manifest_sha256"]
    )

    _conditioning_assets(
        tmp_path,
        cache,
        verification_method="source_forward_atomic_save_reload_tensor_exact",
    )
    with pytest.raises(ValueError, match="verification method mismatch"):
        build_manifest(datasets, cache, vae, download, encoder, tokenizer)


def test_download_manifest_digest_is_verified(tmp_path):
    datasets = []
    for index in range(4):
        directory = tmp_path / f"dataset-{index}"
        _write_dataset(directory, str(index))
        datasets.append(directory)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / CACHE_FILENAME).write_bytes(b"embedding")
    encoder, tokenizer = _conditioning_assets(tmp_path, cache)
    vae = tmp_path / "vae.pt"
    vae.write_bytes(b"vae")
    download = tmp_path / "download.json"
    download.write_text('{"schema_version": 1, "manifest_sha256": "bad"}')
    with pytest.raises(ValueError, match="digest mismatch"):
        build_manifest(datasets, cache, vae, download, encoder, tokenizer)


@pytest.mark.parametrize("mutation", ["path", "repo", "revision", "duplicate"])
def test_download_manifest_requires_the_exact_pinned_asset_set(tmp_path, mutation):
    download = tmp_path / "download.json"
    _download_manifest(download)
    payload = json.loads(download.read_text())
    payload.pop("manifest_sha256")
    if mutation == "path":
        payload["files"][0]["path"] = "data/unexpected.tar.gz"
    elif mutation == "repo":
        payload["files"][0]["repo"] = "yuanty/fastwam"
    elif mutation == "revision":
        payload["files"][0]["revision"] = "0" * 40
    else:
        payload["files"][0]["path"] = payload["files"][1]["path"]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    download.write_text(json.dumps(payload))

    datasets = []
    for index in range(4):
        directory = tmp_path / f"dataset-{index}"
        _write_dataset(directory, str(index))
        datasets.append(directory)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / CACHE_FILENAME).write_bytes(b"embedding")
    encoder, tokenizer = _conditioning_assets(tmp_path, cache)
    vae = tmp_path / "vae.pt"
    vae.write_bytes(b"vae")
    with pytest.raises(ValueError):
        build_manifest(datasets, cache, vae, download, encoder, tokenizer)
