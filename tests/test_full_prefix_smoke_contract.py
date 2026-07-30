from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

import scripts.full_prefix_smoke as smoke_module
from scripts.full_prefix_smoke import (
    FUTURE_VIDEO_VAE_CONTRACT,
    RUNTIME_OBSERVATION_VAE_CONTRACT,
    _extend_history_for_capacity_smoke,
    _validate_smoke_contract,
)


def _cfg(
    *,
    history_blocks: int = 50,
    synthetic_source_history_blocks: int | None = None,
    batch_size: int = 1,
    batch_repeats: int = 1,
    history_vae_batch_chunk_size: int = 1,
):
    smoke = {
        "history_blocks": history_blocks,
        "batch_repeats": batch_repeats,
        "device": "cuda:0",
        "validate_only": True,
    }
    if synthetic_source_history_blocks is not None:
        smoke["synthetic_source_history_blocks"] = synthetic_source_history_blocks
    return OmegaConf.create(
        {
            "batch_size": batch_size,
            "mixed_precision": "bf16",
            "model": {
                "history_training_mode": "incremental_full_bptt",
                "history_vae_batch_chunk_size": history_vae_batch_chunk_size,
                "replan_steps": 10,
                "action_horizon": 32,
            },
            "data": {
                "train": {
                    "full_episode_history": True,
                    "max_history_blocks": 70,
                    "replan_steps": 10,
                }
            },
            "smoke": smoke,
        }
    )


def test_real_h50_contract_is_batch_one_and_runtime_isomorphic():
    contract = _validate_smoke_contract(_cfg())

    assert contract["smoke_profile"] == "real_h50"
    assert contract["batch_size"] == 1
    assert contract["history_blocks"] == 50
    assert contract["selection_history_blocks"] == 50
    assert contract["history_provenance"] == "real_episode_prefix"
    assert contract["is_synthetic_capacity_smoke"] is False
    assert contract["history_vae_batch_chunk_size"] == 1
    assert contract["runtime_observation_vae_contract"] == (
        RUNTIME_OBSERVATION_VAE_CONTRACT
    )
    assert contract["future_video_vae_contract"] == FUTURE_VIDEO_VAE_CONTRACT


def test_h70_contract_is_explicitly_capacity_only_and_selects_real_h50():
    contract = _validate_smoke_contract(
        _cfg(history_blocks=70, synthetic_source_history_blocks=50)
    )

    assert contract["smoke_profile"] == "synthetic_h70_capacity"
    assert contract["history_blocks"] == 70
    assert contract["selection_history_blocks"] == 50
    assert contract["real_history_blocks"] == 50
    assert contract["history_provenance"] == (
        "synthetic_capacity_extension_from_real_h50"
    )
    assert contract["measurement_scope"] == "capacity_oom_only_not_loss_or_quality"
    assert contract["is_synthetic_capacity_smoke"] is True


def test_validate_only_never_loads_dataset_checkpoint_or_cuda(tmp_path, monkeypatch):
    cfg = _cfg()
    cfg.output_dir = str(tmp_path)

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("validate_only crossed into the execution path")

    monkeypatch.setattr(smoke_module, "instantiate", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)

    result = smoke_module.run_smoke(cfg)

    assert result["status"] == "validated_without_dataset_checkpoint_or_cuda"
    assert (tmp_path / "config.yaml").is_file()
    assert (tmp_path / "full_prefix_smoke.json").is_file()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"batch_size": 2}, "cfg.batch_size=1"),
        ({"batch_repeats": 2}, "smoke.batch_repeats=1"),
        ({"history_vae_batch_chunk_size": 2}, "batch_chunk_size=1"),
        ({"history_blocks": 51}, "at most a real H50"),
        (
            {"history_blocks": 70, "synthetic_source_history_blocks": 49},
            "only supported synthetic smoke",
        ),
    ],
)
def test_invalid_runtime_or_provenance_contract_is_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _validate_smoke_contract(_cfg(**kwargs))


def _capacity_sample() -> dict[str, torch.Tensor]:
    capacity = 70
    source = 50
    history_video = torch.arange(
        1 * 3 * capacity * 2 * 2, dtype=torch.float32
    ).reshape(1, 3, capacity, 2, 2)
    history_action = torch.arange(
        1 * capacity * 10 * 7, dtype=torch.float32
    ).reshape(1, capacity, 10, 7)
    history_proprio = torch.arange(
        1 * capacity * 8, dtype=torch.float32
    ).reshape(1, capacity, 8)
    history_valid = torch.zeros(1, capacity, dtype=torch.bool)
    history_valid[:, :source] = True
    positions = torch.full((1, capacity), -1, dtype=torch.long)
    positions[:, :source] = torch.arange(source)
    return {
        "history_video": history_video,
        "history_action": history_action,
        "history_proprio": history_proprio,
        "history_valid_blocks": history_valid,
        "history_block_positions": positions,
        "current_block_position": torch.tensor([source]),
        "episode_step": torch.tensor([source * 10]),
    }


def test_capacity_extension_repeats_only_last_real_block_and_advances_clocks():
    sample = _capacity_sample()
    original_video_prefix = sample["history_video"][:, :, :50].clone()
    original_action_prefix = sample["history_action"][:, :50].clone()
    original_proprio_prefix = sample["history_proprio"][:, :50].clone()

    provenance = _extend_history_for_capacity_smoke(
        sample,
        source_history_blocks=50,
        target_history_blocks=70,
        replan_steps=10,
    )

    assert provenance == {
        "source_history_blocks": 50,
        "target_history_blocks": 70,
        "repeated_source_block": 49,
    }
    torch.testing.assert_close(sample["history_video"][:, :, :50], original_video_prefix)
    torch.testing.assert_close(sample["history_action"][:, :50], original_action_prefix)
    torch.testing.assert_close(
        sample["history_proprio"][:, :50], original_proprio_prefix
    )
    torch.testing.assert_close(
        sample["history_video"][:, :, 50:70],
        sample["history_video"][:, :, 49:50].expand(-1, -1, 20, -1, -1),
    )
    torch.testing.assert_close(
        sample["history_action"][:, 50:70],
        sample["history_action"][:, 49:50].expand(-1, 20, -1, -1),
    )
    torch.testing.assert_close(
        sample["history_proprio"][:, 50:70],
        sample["history_proprio"][:, 49:50].expand(-1, 20, -1),
    )
    assert sample["history_valid_blocks"].all()
    assert sample["history_block_positions"].tolist() == [list(range(70))]
    assert sample["current_block_position"].tolist() == [70]
    assert sample["episode_step"].tolist() == [700]
