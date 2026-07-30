import torch
from contextlib import nullcontext
from types import SimpleNamespace

from fastwam.trainer import Wan22Trainer


def _unbatched_history_sample():
    return {
        "video": torch.randn(3, 33, 8, 8),
        "prompt": "task",
        "action": torch.randn(32, 7),
        "proprio": torch.randn(32, 8),
        "context": torch.randn(5, 6),
        "context_mask": torch.ones(5, dtype=torch.bool),
        "image_is_pad": torch.zeros(33, dtype=torch.bool),
        "action_is_pad": torch.zeros(32, dtype=torch.bool),
        "proprio_is_pad": torch.zeros(32, dtype=torch.bool),
        "history_video": torch.randn(3, 70, 8, 8),
        "history_action": torch.randn(70, 10, 7),
        "history_proprio": torch.randn(70, 8),
        "history_valid_blocks": torch.tensor([True] * 8 + [False] * 62),
        "history_block_positions": torch.tensor(list(range(8)) + [-1] * 62),
        "current_block_position": torch.tensor(8),
        "episode_step": torch.tensor(80),
        "full_episode_history": torch.tensor(True),
    }


def test_eval_batching_preserves_complete_causal_history_sample():
    source = _unbatched_history_sample()
    batched = Wan22Trainer._to_batched_eval_sample(source)

    assert batched["video"].shape == (1, 3, 33, 8, 8)
    assert batched["action"].shape == (1, 32, 7)
    assert batched["history_video"].shape == (1, 3, 70, 8, 8)
    assert batched["history_action"].shape == (1, 70, 10, 7)
    assert batched["history_proprio"].shape == (1, 70, 8)
    assert batched["history_valid_blocks"].shape == (1, 70)
    assert batched["history_block_positions"].shape == (1, 70)
    assert batched["current_block_position"].tolist() == [8]
    assert batched["episode_step"].tolist() == [80]
    assert batched["full_episode_history"].tolist() == [True]
    for key in (
        "image_is_pad",
        "action_is_pad",
        "proprio_is_pad",
        "history_video",
        "history_action",
        "history_proprio",
        "history_valid_blocks",
        "history_block_positions",
        "current_block_position",
        "episode_step",
        "full_episode_history",
    ):
        torch.testing.assert_close(batched[key][0], source[key])


def test_eval_batching_does_not_double_batch_extended_fields():
    source = _unbatched_history_sample()
    once = Wan22Trainer._to_batched_eval_sample(source)
    twice = Wan22Trainer._to_batched_eval_sample(once)

    assert twice["video"].shape == once["video"].shape
    assert twice["history_video"].shape == once["history_video"].shape
    assert twice["history_action"].shape == once["history_action"].shape
    torch.testing.assert_close(twice["history_video"], once["history_video"])
    torch.testing.assert_close(twice["history_action"], once["history_action"])


def test_history_validation_never_reports_no_memory_inference_metrics():
    class FakeModel:
        def __init__(self):
            self.dit = SimpleNamespace(training=False)

        def eval(self):
            return self

        def training_loss(self, sample):
            assert sample["history_valid_blocks"].sum().item() == 8
            return torch.tensor(1.25), {}

        def infer(self, **kwargs):
            raise AssertionError("history validation must not call FastWAM infer")

    class FakeAccelerator:
        device = torch.device("cpu")
        process_index = 0

        @staticmethod
        def unwrap_model(model):
            return model

        @staticmethod
        def autocast():
            return nullcontext()

        @staticmethod
        def gather_for_metrics(value):
            return value

    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.accelerator = FakeAccelerator()
    trainer.model = FakeModel()
    trainer.val_dataset = [_unbatched_history_sample()]
    trainer.global_step = 0

    metrics = trainer.evaluate()
    assert metrics == {
        "val_loss": 1.25,
        "history_conditioned": True,
        "rollout_metrics_skipped": True,
    }
