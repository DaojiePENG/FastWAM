#!/usr/bin/env python3
"""6B H800 action-only/70-block smoke test using a release-derived checkpoint."""

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
    parser.add_argument("--dataset-stats", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--blocks", type=int, default=70)
    parser.add_argument("--exit-depth", type=int, choices=(8, 16, 24, 30), default=30)
    parser.add_argument(
        "--causal-mode",
        choices=("interleaved", "vision_causal", "action_aggregator"),
        default="interleaved",
    )
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("smoke_leapbot_h800.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("H800 smoke test requires CUDA")
    if not args.checkpoint.is_file() or not args.dataset_stats.is_file():
        raise FileNotFoundError("checkpoint and dataset stats must exist")

    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(
            config_name="sim_leapbot_libero",
            overrides=[
                f"ckpt={args.checkpoint}",
                f"EVALUATION.dataset_stats_path={args.dataset_stats}",
                f"EVALUATION.memory.exit_depth={args.exit_depth}",
                f"model.causal_mode={args.causal_mode}",
                "model.load_text_encoder=false",
            ],
        )
    model = instantiate(
        cfg.model,
        model_dtype=torch.bfloat16,
        device=args.device,
    )
    model.load_checkpoint(str(args.checkpoint))
    model.eval()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("action-only inference called a video output/decode path")

    model.video_expert.head.forward = forbidden
    for head in model.video_exit_heads.values():
        head.forward = forbidden
    model.vae.decode = forbidden

    memory = model.create_memory(
        exit_depth=args.exit_depth,
        causal_mode=args.causal_mode,
        max_history_blocks=args.blocks,
        action_horizon=32,
        replan_steps=10,
    )
    image = torch.zeros((1, 3, 224, 448), dtype=torch.bfloat16, device=args.device)
    proprio = torch.zeros((1, int(model.proprio_dim)), dtype=torch.bfloat16, device=args.device)
    context = torch.zeros((1, 128, 4096), dtype=torch.bfloat16, device=args.device)
    context_mask = torch.ones((1, 128), dtype=torch.bool, device=args.device)
    torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    replans = []
    for block in range(args.blocks):
        prediction = model.infer_action(
            prompt=None,
            input_image=image,
            action_horizon=32,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            memory=memory,
            num_inference_steps=args.inference_steps,
            seed=block,
            profile=True,
        )
        commit = model.commit_executed_actions(
            memory,
            prediction["action"][:10],
            profile=True,
        )
        replans.append({"timing": prediction["timing"], "commit": commit})

    try:
        model.infer_action(
            prompt=None,
            input_image=image,
            action_horizon=32,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            memory=memory,
            num_inference_steps=1,
        )
    except RuntimeError as exc:
        if "capacity exceeded" not in str(exc):
            raise
    else:
        raise AssertionError("history capacity guard did not reject block 71")

    result = {
        "device": torch.cuda.get_device_name(torch.device(args.device)),
        "blocks": memory.completed_blocks,
        "exit_depth": args.exit_depth,
        "causal_mode": args.causal_mode,
        "cache_bytes": memory.cache_nbytes,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(torch.device(args.device)),
        "replans": replans,
        "forbidden_video_paths_called": False,
        "capacity_guard_verified": True,
    }
    args.output.write_text(json.dumps(result, indent=2))
    model.reset_memory(memory)
    print(json.dumps({key: value for key, value in result.items() if key != "replans"}, indent=2))


if __name__ == "__main__":
    main()
