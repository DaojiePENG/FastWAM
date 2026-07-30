import numpy as np
import torch

from leapbot_va.libero import (
    canonicalize_libero_env_action,
    executed_env_actions_to_model_space,
)


class _AffineNormalizer:
    def forward(self, x):
        return x * 2.0 - 1.0


class _Processor:
    shape_meta = {"action": [{"key": "default", "shape": 3}]}
    action_state_transforms = None

    class _Container:
        normalizers = {"action": {"default": _AffineNormalizer()}}

    normalizer = _Container()


def test_executed_gripper_is_inverted_and_renormalized():
    # Environment convention is +1=close and -1=open.  Dataset convention is
    # 0=close and 1=open, which the dummy affine normalizer maps to [-1,+1].
    env = np.array([[0.2, -0.4, 1.0], [0.3, 0.5, -1.0]], dtype=np.float32)
    normalized = executed_env_actions_to_model_space(env, _Processor())
    expected = torch.tensor([[-0.6, -1.8, -1.0], [-0.4, 0.0, 1.0]])
    torch.testing.assert_close(normalized, expected)


def test_ensembled_command_not_unexecuted_prediction_is_used():
    executed = np.array([[0.0, 0.0, -1.0]], dtype=np.float32)
    normalized = executed_env_actions_to_model_space(executed, _Processor())
    assert normalized.shape == (1, 3)
    assert normalized[0, -1].item() == 1.0


def test_executed_command_is_clipped_and_gripper_is_strictly_binary():
    spec = (
        np.array([-1.0, -0.5, -1.0], dtype=np.float32),
        np.array([1.0, 0.5, 1.0], dtype=np.float32),
    )
    command = canonicalize_libero_env_action(
        np.array([2.0, -2.0, 0.0], dtype=np.float32),
        spec,
        binarize_gripper=True,
    )
    np.testing.assert_array_equal(command, np.array([1.0, -0.5, 1.0], np.float32))


def test_executed_command_rejects_nonfinite_values():
    spec = (np.full(3, -1.0, np.float32), np.full(3, 1.0, np.float32))
    with np.testing.assert_raises_regex(ValueError, "non-finite"):
        canonicalize_libero_env_action(
            np.array([0.0, np.nan, 1.0], dtype=np.float32),
            spec,
            binarize_gripper=True,
        )
