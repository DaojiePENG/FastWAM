from pathlib import Path
import inspect
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastwam.trainer import Wan22Trainer


class TrainerCheckpointLoadingOrderTest(unittest.TestCase):
    def test_file_resume_is_loaded_before_optimizer_and_prepare(self):
        source = inspect.getsource(Wan22Trainer.__init__)

        preload_index = source.index("self._load_resume_weights_before_optimizer()")
        optimizer_index = source.index("self.optimizer = torch.optim.AdamW(")
        prepare_index = source.index("self.accelerator.prepare(")

        self.assertLess(preload_index, optimizer_index)
        self.assertLess(preload_index, prepare_index)

    def test_full_state_resume_remains_after_prepare(self):
        source = inspect.getsource(Wan22Trainer.__init__)

        prepare_index = source.index("self.accelerator.prepare(")
        full_state_index = source.index("self._resume_full_training_state_after_prepare()")
        self.assertLess(prepare_index, full_state_index)

    def test_file_resume_loads_into_raw_model(self):
        source = inspect.getsource(Wan22Trainer._load_resume_weights_before_optimizer)
        self.assertIn("self.model.load_checkpoint", source)
        self.assertNotIn("unwrap_model", source)


if __name__ == "__main__":
    unittest.main()
