import json

import pytest

from fastwam.trainer import Wan22Trainer


class _Sized:
    def __init__(self, length):
        self.length = length
        self.epochs = []

    def __len__(self):
        return self.length

    def set_epoch(self, epoch):
        self.epochs.append(int(epoch))


class _Sampler:
    def __init__(self):
        self.epoch_offsets = []
        self.resume_offsets = []

    def set_epoch_offset(self, epoch):
        self.epoch_offsets.append(int(epoch))

    def set_resume_batch_offset(self, batch):
        self.resume_offsets.append(int(batch))

    def clear_resume_batch_offset(self):
        self.resume_offsets.append(0)


class _Accelerator:
    num_processes = 8

    def __init__(self):
        self.loaded = []

    def load_state(self, input_dir):
        self.loaded.append(str(input_dir))

    def wait_for_everyone(self):
        pass


def _trainer():
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.global_step = 17
    trainer.epoch = 2
    trainer.batch_in_epoch = 10
    trainer.batch_size = 2
    trainer.train_dataset = _Sized(157)
    trainer.train_loader = _Sized(10)
    trainer.train_sampler = _Sampler()
    trainer.accelerator = _Accelerator()
    return trainer


def test_saved_epoch_boundary_is_normalized_to_next_unconsumed_sample(tmp_path):
    trainer = _trainer()
    trainer._save_trainer_state(str(tmp_path))
    state = json.loads((tmp_path / "trainer_state.json").read_text())
    assert state == {
        "global_step": 17,
        "epoch": 3,
        "batch_in_epoch": 0,
        "dataset_length": 157,
        "batch_size_per_process": 2,
        "num_processes": 8,
        "micro_batches_per_epoch": 10,
    }


def test_legacy_epoch_boundary_is_normalized_when_loaded(tmp_path):
    trainer = _trainer()
    (tmp_path / "trainer_state.json").write_text(
        json.dumps({"global_step": 23, "epoch": 4, "batch_in_epoch": 10})
    )
    trainer.load_training_state(str(tmp_path))
    assert trainer.global_step == 23
    assert trainer.epoch == 5
    assert trainer.batch_in_epoch == 0
    assert trainer.train_loader.epochs == [0]
    assert trainer.train_sampler.epoch_offsets == [5]
    assert trainer.train_sampler.resume_offsets == [0]


def test_resume_rejects_changed_distributed_batch_contract(tmp_path):
    trainer = _trainer()
    (tmp_path / "trainer_state.json").write_text(
        json.dumps(
            {
                "global_step": 23,
                "epoch": 4,
                "batch_in_epoch": 3,
                "dataset_length": 157,
                "batch_size_per_process": 1,
                "num_processes": 8,
                "micro_batches_per_epoch": 10,
            }
        )
    )
    with pytest.raises(ValueError, match="batch_size_per_process"):
        trainer.load_training_state(str(tmp_path))


def test_loader_passes_gradient_accumulation_to_resumable_sampler():
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.seed = 42
    trainer.batch_size = 2
    trainer.num_workers = 0
    trainer.gradient_accumulation_steps = 3
    trainer.accelerator = _Accelerator()

    trainer._build_loader(_Sized(37))

    assert trainer.train_sampler.gradient_accumulation_steps == 3
