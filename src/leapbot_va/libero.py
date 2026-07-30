"""LIBERO action bridge for committing commands that were actually executed."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def canonicalize_libero_env_action(
    action: np.ndarray | torch.Tensor,
    action_spec: tuple[np.ndarray, np.ndarray],
    *,
    binarize_gripper: bool,
) -> np.ndarray:
    """Return the exact bounded command used for both ``env.step`` and memory.

    LIBERO/robosuite controllers clip actions internally.  Performing that
    clipping explicitly makes the command persisted in Leap memory identical
    to the one applied by the controller.  Gripper thresholding happens last,
    after any action ensembling, and maps the ambiguous zero value to close.
    """

    command = np.asarray(action, dtype=np.float32).copy()
    if command.ndim != 1:
        raise ValueError(f"LIBERO env action must be [D], got {command.shape}")
    if not np.isfinite(command).all():
        raise ValueError("LIBERO env action contains non-finite values")
    if not isinstance(action_spec, (tuple, list)) or len(action_spec) != 2:
        raise ValueError("env.action_spec must be a (low, high) pair")
    low = np.asarray(action_spec[0], dtype=np.float32)
    high = np.asarray(action_spec[1], dtype=np.float32)
    if low.shape != command.shape or high.shape != command.shape:
        raise ValueError(
            "env.action_spec shape must match the action: "
            f"action={command.shape} low={low.shape} high={high.shape}"
        )
    if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(low > high):
        raise ValueError("env.action_spec contains invalid bounds")

    command = np.clip(command, low, high)
    if binarize_gripper:
        if low[-1] > -1.0 or high[-1] < 1.0:
            raise ValueError(
                "strict LIBERO gripper binarization requires action bounds containing [-1,1]"
            )
        command[-1] = 1.0 if command[-1] >= 0.0 else -1.0
    # Own the returned storage so neither the ensembler nor the environment can
    # mutate the value later recorded for the cache commit.
    return np.ascontiguousarray(command, dtype=np.float32)


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
    if not bool(torch.isfinite(actions).all().item()):
        raise ValueError("env_actions contains non-finite values")

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
    normalized = normalizer.forward(actions)
    if not bool(torch.isfinite(normalized).all().item()):
        raise ValueError("normalized executed actions contain non-finite values")
    return normalized
