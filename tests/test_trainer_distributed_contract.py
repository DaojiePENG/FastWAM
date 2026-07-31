from __future__ import annotations

import logging

import pytest
import torch
from accelerate.utils import DistributedType
from torch import nn

from fastwam.trainer import Wan22Trainer


class _DeepSpeedEngine(nn.Module):
    def __init__(self, *, micro_batch=4, grad_accum=4, global_batch=128, zero_stage=2):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.values = {
            "train_micro_batch_size_per_gpu": micro_batch,
            "gradient_accumulation_steps": grad_accum,
            "train_batch_size": global_batch,
            "zero_optimization_stage": zero_stage,
        }

    def train_micro_batch_size_per_gpu(self):
        return self.values["train_micro_batch_size_per_gpu"]

    def gradient_accumulation_steps(self):
        return self.values["gradient_accumulation_steps"]

    def train_batch_size(self):
        return self.values["train_batch_size"]

    def zero_optimization_stage(self):
        return self.values["zero_optimization_stage"]


class _Plugin:
    def __init__(self, zero_stage=2):
        self.deepspeed_config = {"zero_optimization": {"stage": zero_stage}}


class _State:
    def __init__(self, zero_stage=2):
        self.deepspeed_plugin = _Plugin(zero_stage)


class _TopologyAccelerator:
    distributed_type = DistributedType.DEEPSPEED
    num_processes = 8

    def __init__(self, zero_stage=2):
        self.state = _State(zero_stage)


def _topology_trainer(**engine_overrides):
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.batch_size = 4
    trainer.gradient_accumulation_steps = 4
    trainer.model = _DeepSpeedEngine(**engine_overrides)
    trainer.accelerator = _TopologyAccelerator()
    return trainer


def test_deepspeed_engine_topology_matches_cfg_and_world_size(caplog):
    trainer = _topology_trainer()

    with caplog.at_level(logging.INFO):
        actual = trainer._assert_deepspeed_training_topology()

    assert actual == {
        "micro_batch_size_per_gpu": 4,
        "gradient_accumulation_steps": 4,
        "global_batch_size": 128,
        "zero_stage": 2,
    }
    assert "Verified DeepSpeed topology" in caplog.text


@pytest.mark.parametrize(
    "overrides,field",
    (
        ({"micro_batch": 2}, "micro_batch_size_per_gpu"),
        ({"grad_accum": 2}, "gradient_accumulation_steps"),
        ({"global_batch": 64}, "global_batch_size"),
        ({"zero_stage": 1}, "zero_stage"),
    ),
)
def test_deepspeed_engine_topology_mismatch_is_fatal(overrides, field):
    trainer = _topology_trainer(**overrides)

    with pytest.raises(RuntimeError, match=field):
        trainer._assert_deepspeed_training_topology()


def test_non_deepspeed_training_does_not_require_engine_accessors():
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.model = nn.Linear(2, 2)
    trainer.accelerator = type(
        "Accelerator", (), {"distributed_type": DistributedType.NO}
    )()

    assert trainer._assert_deepspeed_training_topology() is None


class _GradientAccelerator:
    device = torch.device("cpu")

    def __init__(self, *, local_norm, gathered_norms):
        self.local_norm = local_norm
        self.gathered_norms = gathered_norms
        self.clip_calls = 0

    def clip_grad_norm_(self, parameters, max_norm):
        list(parameters)
        assert max_norm == 1.0
        self.clip_calls += 1
        return torch.tensor(self.local_norm)

    def gather(self, value):
        assert value.shape == (1,)
        return torch.tensor(self.gathered_norms, dtype=value.dtype)


class _Optimizer:
    def __init__(self):
        self.step_calls = 0

    def step(self):
        self.step_calls += 1


def _gradient_trainer(*, local_norm, gathered_norms):
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.model = nn.Linear(2, 2)
    trainer.max_grad_norm = 1.0
    trainer.accelerator = _GradientAccelerator(
        local_norm=local_norm,
        gathered_norms=gathered_norms,
    )
    trainer.optimizer = _Optimizer()
    return trainer


def test_gradient_norm_is_clipped_gathered_and_averaged_before_step():
    trainer = _gradient_trainer(
        local_norm=2.0,
        gathered_norms=[2.0, 4.0, 6.0, 8.0],
    )

    result = trainer._optimizer_step_with_validated_gradients()

    assert result == 5.0
    assert trainer.accelerator.clip_calls == 1
    assert trainer.optimizer.step_calls == 1


@pytest.mark.parametrize("bad_value", (float("nan"), float("inf"), float("-inf")))
def test_nonfinite_gradient_norm_on_any_rank_fails_all_ranks_before_step(bad_value):
    # Every rank receives the same gathered vector, even when its own local norm is finite.
    rank_zero = _gradient_trainer(
        local_norm=1.0,
        gathered_norms=[1.0, bad_value, 3.0, 4.0],
    )
    rank_one = _gradient_trainer(
        local_norm=bad_value,
        gathered_norms=[1.0, bad_value, 3.0, 4.0],
    )

    for trainer in (rank_zero, rank_one):
        with pytest.raises(FloatingPointError, match="across ranks"):
            trainer._optimizer_step_with_validated_gradients()
        assert trainer.optimizer.step_calls == 0
