#!/usr/bin/env python3
"""Validate a LeapBot evaluation checkpoint and its resumable trainer state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def _state_dict_summary(state_dict: Any, name: str) -> dict[str, int]:
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError(f"{name} must be a non-empty state dict")
    tensors = [value for value in state_dict.values() if isinstance(value, torch.Tensor)]
    if not tensors:
        raise ValueError(f"{name} contains no tensors")
    return {
        "tensor_count": len(tensors),
        "parameter_numel": int(sum(tensor.numel() for tensor in tensors)),
    }


def _directory_summary(path: Path) -> dict[str, int]:
    files = [item for item in path.rglob("*") if item.is_file()]
    return {
        "file_count": len(files),
        "bytes": int(sum(item.stat().st_size for item in files)),
    }


def validate_checkpoint(
    checkpoint: Path,
    *,
    expected_step: int,
    expected_mode: str,
    state_dir: Path | None = None,
    expected_history_training_mode: str = "packed_full_bptt",
    expected_training_strategy: str = "video_lora_action_full",
    expected_video_lora_multiplier: float = 1.0,
    expected_replan_steps: int = 10,
    expected_action_horizon: int = 32,
) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise FileNotFoundError(checkpoint)

    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint payload must be a dict, got {type(payload)!r}")

    expected_metadata = {
        "step": int(expected_step),
        "causal_mode": str(expected_mode),
        "history_training_mode": expected_history_training_mode,
        "training_strategy": expected_training_strategy,
        "training_replan_steps": int(expected_replan_steps),
        "training_action_horizon": int(expected_action_horizon),
    }
    for key, expected in expected_metadata.items():
        actual = payload.get(key)
        if actual != expected:
            raise ValueError(
                f"checkpoint metadata mismatch for {key}: expected={expected!r} "
                f"actual={actual!r}"
            )

    exit_depths = tuple(int(depth) for depth in payload.get("training_exit_depths", ()))
    if exit_depths != (30,):
        raise ValueError(f"expected training_exit_depths=(30,), got {exit_depths}")

    video_lora = payload.get("video_lora_config")
    if not isinstance(video_lora, dict):
        raise ValueError("checkpoint is missing video_lora_config")
    expected_lora = {
        "enabled": True,
        "rank": 16,
        "alpha": 16.0,
        "dropout": 0.0,
        "learning_rate_multiplier": float(expected_video_lora_multiplier),
    }
    for key, expected in expected_lora.items():
        actual = video_lora.get(key)
        if actual != expected:
            raise ValueError(
                f"video LoRA metadata mismatch for {key}: "
                f"expected={expected!r} actual={actual!r}"
            )

    result: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": int(checkpoint.stat().st_size),
        **expected_metadata,
        "training_exit_depths": list(exit_depths),
        "video_lora_config": video_lora,
        "mot": _state_dict_summary(payload.get("mot"), "mot"),
        "action_exit_heads": _state_dict_summary(
            payload.get("action_exit_heads"), "action_exit_heads"
        ),
        "video_exit_heads": _state_dict_summary(
            payload.get("video_exit_heads"), "video_exit_heads"
        ),
    }

    if state_dir is not None:
        state_dir = state_dir.expanduser().resolve()
        if not state_dir.is_dir():
            raise FileNotFoundError(state_dir)
        trainer_state_path = state_dir / "trainer_state.json"
        if not trainer_state_path.is_file():
            raise FileNotFoundError(trainer_state_path)
        trainer_state = json.loads(trainer_state_path.read_text())
        if int(trainer_state.get("global_step", -1)) != int(expected_step):
            raise ValueError(
                "trainer state/checkpoint step mismatch: "
                f"expected={expected_step} trainer_state={trainer_state}"
            )
        state_summary = _directory_summary(state_dir)
        if state_summary["file_count"] <= 1 or state_summary["bytes"] <= 0:
            raise ValueError(f"trainer state is incomplete: {state_summary}")
        result["state_dir"] = str(state_dir)
        result["trainer_state"] = trainer_state
        result["state"] = state_summary

    return result


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--expected-mode", required=True)
    parser.add_argument(
        "--expected-history-training-mode", default="packed_full_bptt"
    )
    parser.add_argument("--expected-training-strategy", default="video_lora_action_full")
    parser.add_argument("--expected-video-lora-multiplier", type=float, default=1.0)
    parser.add_argument("--expected-replan-steps", type=int, default=10)
    parser.add_argument("--expected-action-horizon", type=int, default=32)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = validate_checkpoint(
        args.checkpoint,
        expected_step=args.expected_step,
        expected_mode=args.expected_mode,
        state_dir=args.state_dir,
        expected_history_training_mode=args.expected_history_training_mode,
        expected_training_strategy=args.expected_training_strategy,
        expected_video_lora_multiplier=args.expected_video_lora_multiplier,
        expected_replan_steps=args.expected_replan_steps,
        expected_action_horizon=args.expected_action_horizon,
    )
    output = (
        args.output
        if args.output is not None
        else args.checkpoint.with_suffix(".validation.json")
    )
    _write_json(output, result)
    print(output)


if __name__ == "__main__":
    main()
