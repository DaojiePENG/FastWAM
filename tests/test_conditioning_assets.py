from __future__ import annotations

from pathlib import Path

import pytest
import torch

from leapbot_va.conditioning_assets import (
    TEXT_CACHE_PROVENANCE_FILENAME,
    build_text_cache_provenance,
    build_wan_conditioning_identity,
    file_identity,
    load_and_validate_text_cache_provenance,
    tree_identity,
    write_text_cache_provenance,
)
from scripts.verify_text_cache_provenance import assert_cached_payload_exact


def _assets(tmp_path: Path):
    vae = tmp_path / "vae.safetensors"
    vae.write_bytes(b"vae")
    encoder = tmp_path / "encoder.safetensors"
    encoder.write_bytes(b"encoder")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_bytes(b"tokenizer")
    return vae, encoder, tokenizer


def _cache(tmp_path: Path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "a.pt").write_bytes(b"embedding-a")
    return cache


def test_wan_identity_hashes_actual_resolved_assets_and_is_relocation_stable(tmp_path):
    vae, encoder, tokenizer = _assets(tmp_path)
    first = build_wan_conditioning_identity(
        model_id="model",
        tokenizer_model_id="tokenizer-model",
        redirect_common_files=True,
        load_text_encoder=True,
        resolved_paths={
            "vae": vae,
            "text_encoder": encoder,
            "tokenizer": tokenizer,
        },
    )

    relocated = tmp_path / "relocated"
    relocated.mkdir()
    relocated_vae, relocated_encoder, relocated_tokenizer = _assets(relocated)
    second = build_wan_conditioning_identity(
        model_id="model",
        tokenizer_model_id="tokenizer-model",
        redirect_common_files=True,
        load_text_encoder=True,
        resolved_paths={
            "vae": relocated_vae,
            "text_encoder": relocated_encoder,
            "tokenizer": relocated_tokenizer,
        },
    )
    assert first == second

    relocated_encoder.write_bytes(b"different")
    changed = build_wan_conditioning_identity(
        model_id="model",
        tokenizer_model_id="tokenizer-model",
        redirect_common_files=True,
        load_text_encoder=True,
        resolved_paths={
            "vae": relocated_vae,
            "text_encoder": relocated_encoder,
            "tokenizer": relocated_tokenizer,
        },
    )
    assert changed["identity_sha256"] != first["identity_sha256"]


def test_conditioning_asset_identity_rejects_top_level_symlink(tmp_path):
    _, encoder, _ = _assets(tmp_path)
    link = tmp_path / "encoder-link"
    link.symlink_to(encoder)
    with pytest.raises(ValueError, match="must not be symlinks"):
        file_identity(link)


def test_text_cache_provenance_has_no_self_reference_and_rehashes_sources(tmp_path):
    _, encoder, tokenizer = _assets(tmp_path)
    cache = _cache(tmp_path)
    provenance = build_text_cache_provenance(
        cache_dir=cache,
        cache_file_names=["a.pt"],
        model_id="model",
        tokenizer_model_id="tokenizer-model",
        redirect_common_files=True,
        context_len=128,
        text_encoder_path=encoder,
        tokenizer_path=tokenizer,
        verified_file_count=1,
    )
    cache_hash = provenance["cache"]
    write_text_cache_provenance(cache, provenance)
    assert tree_identity(cache, relative_files=["a.pt"]) == cache_hash
    validated = load_and_validate_text_cache_provenance(
        cache,
        text_encoder_path=encoder,
        tokenizer_path=tokenizer,
        model_id="model",
        tokenizer_model_id="tokenizer-model",
        redirect_common_files=True,
    )
    assert validated == provenance
    assert (cache / TEXT_CACHE_PROVENANCE_FILENAME).is_file()
    with pytest.raises(ValueError, match="context_len mismatch"):
        load_and_validate_text_cache_provenance(
            cache,
            text_encoder_path=encoder,
            tokenizer_path=tokenizer,
            context_len=256,
        )

    (cache / "a.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="cache bytes"):
        load_and_validate_text_cache_provenance(
            cache,
            text_encoder_path=encoder,
            tokenizer_path=tokenizer,
        )


def test_text_cache_provenance_rejects_unlisted_cache_entries_and_wrong_encoder(tmp_path):
    _, encoder, tokenizer = _assets(tmp_path)
    cache = _cache(tmp_path)
    provenance = build_text_cache_provenance(
        cache_dir=cache,
        cache_file_names=["a.pt"],
        model_id="model",
        tokenizer_model_id="tokenizer-model",
        redirect_common_files=True,
        context_len=128,
        text_encoder_path=encoder,
        tokenizer_path=tokenizer,
        verified_file_count=1,
    )
    write_text_cache_provenance(cache, provenance)

    wrong_encoder = tmp_path / "wrong.safetensors"
    wrong_encoder.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="encoder/tokenizer"):
        load_and_validate_text_cache_provenance(
            cache,
            text_encoder_path=wrong_encoder,
            tokenizer_path=tokenizer,
        )

    (cache / "extra.pt").write_bytes(b"extra")
    with pytest.raises(ValueError, match="file set"):
        load_and_validate_text_cache_provenance(
            cache,
            text_encoder_path=encoder,
            tokenizer_path=tokenizer,
        )


def test_text_cache_provenance_requires_every_file_to_be_save_reload_verified(tmp_path):
    _, encoder, tokenizer = _assets(tmp_path)
    cache = _cache(tmp_path)
    with pytest.raises(ValueError, match="verification for every file"):
        build_text_cache_provenance(
            cache_dir=cache,
            cache_file_names=["a.pt"],
            model_id="model",
            tokenizer_model_id="tokenizer-model",
            redirect_common_files=True,
            context_len=128,
            text_encoder_path=encoder,
            tokenizer_path=tokenizer,
            verified_file_count=0,
        )


def test_online_audit_compares_context_and_mask_tensors_exactly(tmp_path):
    cache_path = tmp_path / "prompt.pt"
    context = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    mask = torch.tensor([True])
    assert_cached_payload_exact(
        {"context": context.clone(), "mask": mask.clone()},
        expected_context=context,
        expected_mask=mask,
        cache_path=cache_path,
    )
    with pytest.raises(ValueError, match="context mismatch"):
        assert_cached_payload_exact(
            {"context": context + 1, "mask": mask},
            expected_context=context,
            expected_mask=mask,
            cache_path=cache_path,
        )
    with pytest.raises(ValueError, match="mask mismatch"):
        assert_cached_payload_exact(
            {"context": context, "mask": ~mask},
            expected_context=context,
            expected_mask=mask,
            cache_path=cache_path,
        )
    with pytest.raises(ValueError, match="context dtype mismatch"):
        assert_cached_payload_exact(
            {"context": context.float(), "mask": mask},
            expected_context=context,
            expected_mask=mask,
            cache_path=cache_path,
        )
    with pytest.raises(ValueError, match="mask dtype mismatch"):
        assert_cached_payload_exact(
            {"context": context, "mask": mask.to(torch.uint8)},
            expected_context=context,
            expected_mask=mask,
            cache_path=cache_path,
        )
    with pytest.raises(ValueError, match="context is not a tensor"):
        assert_cached_payload_exact(
            {"context": [[1.0, 2.0]], "mask": mask},
            expected_context=context,
            expected_mask=mask,
            cache_path=cache_path,
        )
