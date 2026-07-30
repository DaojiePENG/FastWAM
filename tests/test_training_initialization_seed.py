import random

import numpy as np
import torch

from fastwam.runtime import _seed_model_initialization
from leapbot_va.lora import LoRALinear


def _initialized_state(seed: int):
    _seed_model_initialization(seed)
    layer = LoRALinear(
        11,
        7,
        rank=3,
        alpha=3.0,
        device="cpu",
        dtype=torch.float32,
    )
    return {
        "python": random.random(),
        "numpy": float(np.random.random()),
        "torch": torch.rand(5),
        "lora_a": layer.lora_A.detach().clone(),
        "lora_b": layer.lora_B.detach().clone(),
    }


def test_model_initialization_seed_is_reproducible_before_trainer_creation():
    first = _initialized_state(42)
    _ = _initialized_state(314159)
    second = _initialized_state(42)

    assert first["python"] == second["python"]
    assert first["numpy"] == second["numpy"]
    torch.testing.assert_close(first["torch"], second["torch"], rtol=0, atol=0)
    torch.testing.assert_close(first["lora_a"], second["lora_a"], rtol=0, atol=0)
    torch.testing.assert_close(first["lora_b"], second["lora_b"], rtol=0, atol=0)


def test_different_model_initialization_seeds_change_lora_basis():
    first = _initialized_state(42)
    second = _initialized_state(43)
    assert not torch.equal(first["lora_a"], second["lora_a"])
