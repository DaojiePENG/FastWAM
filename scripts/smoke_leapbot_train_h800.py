#!/usr/bin/env python3
"""Run one synthetic 6B causal-history training step on an H800."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--causal-mode", default="interleaved")
    parser.add_argument("--all-exits", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("smoke_leapbot_train_h800.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("training smoke requires CUDA")

    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    exits = "[8,16,24,30]" if args.all_exits else "[30]"
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(
            config_name="train",
            overrides=[
                "task=libero_leapbot_2cam224",
                f"model.causal_mode={args.causal_mode}",
                f"model.training_exit_depths={exits}",
            ],
        )
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=args.device)
    model.load_checkpoint(str(args.checkpoint))
    model.train()

    # Two 224x224 cameras are horizontally concatenated.  The current 9 RGB
    # frames preserve FastWAM's 32-action, ratio-4 supervision layout.
    sample = {
        "video": torch.zeros(1, 3, 9, 224, 448, dtype=torch.bfloat16),
        "action": torch.zeros(1, 32, 7, dtype=torch.bfloat16),
        "proprio": torch.zeros(1, 32, 8, dtype=torch.bfloat16),
        "context": torch.zeros(1, 128, 4096, dtype=torch.bfloat16),
        "context_mask": torch.ones(1, 128, dtype=torch.bool),
        "image_is_pad": torch.zeros(1, 9, dtype=torch.bool),
        "action_is_pad": torch.zeros(1, 32, dtype=torch.bool),
        "history_video": torch.zeros(1, 3, 1, 224, 448, dtype=torch.bfloat16),
        "history_action": torch.zeros(1, 1, 10, 7, dtype=torch.bfloat16),
        "history_proprio": torch.zeros(1, 1, 8, dtype=torch.bfloat16),
        "history_valid_blocks": torch.ones(1, 1, dtype=torch.bool),
        "history_block_positions": torch.zeros(1, 1, dtype=torch.long),
        "current_block_position": torch.ones(1, dtype=torch.long),
        "episode_step": torch.full((1,), 10, dtype=torch.long),
    }
    torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    loss, metrics = model.training_loss(sample)
    loss.backward()
    result = {
        "device": torch.cuda.get_device_name(torch.device(args.device)),
        "causal_mode": args.causal_mode,
        "training_exit_depths": list(model.training_exit_depths),
        "loss": float(loss.detach()),
        "metrics": metrics,
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(torch.device(args.device))),
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
