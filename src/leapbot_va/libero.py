"""LIBERO action bridge for committing commands that were actually executed."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def executed_env_actions_to_model_space(
    env_actions: np.ndarray | torch.Tensor,
    processor: Any,
) -> torch.Tensor:
    """Undo LIBERO gripper postprocessing and normalize the executed commands.

    FastWAM's LIBERO evaluator maps the dataset gripper convention ``[0, 1]``
    to the environment convention ``[+1, -1]`` and can then binarize it.  This
    function performs the exact inverse before applying the training action
    normalizer.  It intentionally consumes the commands passed to ``env.step``
    rather than the original prediction, so clipping, gripper binarization, or
    action ensembling are reflected in persistent KV memory.
    """

    actions = torch.as_tensor(env_actions, dtype=torch.float32).clone()
    if actions.ndim == 1:
        actions = actions.unsqueeze(0)
    if actions.ndim != 2:
        raise ValueError(f"env_actions must be [T,D] or [D], got {tuple(actions.shape)}")

    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError("LIBERO executed-action commit requires one merged action key")
    if processor.action_state_transforms:
        raise ValueError(
            "executed-action commit does not support coupled action/state transforms; "
            "provide a dedicated inverse transform before committing"
        )

    # eval_libero_single: raw_g -> 2*raw_g-1 -> sign flip -> optional sign().
    actions[..., -1] = (1.0 - actions[..., -1]) * 0.5
    key = action_meta[0]["key"]
    expected_dim = int(action_meta[0]["shape"])
    if actions.shape[-1] != expected_dim:
        raise ValueError(
            f"executed action dimension is {actions.shape[-1]}, expected {expected_dim}"
        )
    normalizer = processor.normalizer.normalizers["action"][key]
    return normalizer.forward(actions)
