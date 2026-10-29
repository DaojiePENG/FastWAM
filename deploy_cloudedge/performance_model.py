#!/usr/bin/env python3
"""Export an untrained a6-shaped performance model, never a task policy.

The released full-MoT checkpoint is loaded by the existing factory BEFORE
rank-16 LoRA injection. New codec/projector parameters keep seeded random
initialization, and LoRA B remains zero. These artifacts support timing and
transport measurements; they contain no trained LeapBot-CE delta.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--siglip", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--out-delta", required=True)
    parser.add_argument("--out-edge-artifact", required=True)
    parser.add_argument("--task", default="libero_all_leapbotce_codec_lora")
    parser.add_argument("--device", default="cpu", help="CPU export avoids GPU allocation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--model-base-path", default=None)
    parser.add_argument("--io-overrides", default=None, help="Small trusted LIBERO mot/proprio payload")
    parser.add_argument("--out-prepared-base", default=None, help="NVMe full-MoT path after I/O replacement")
    parser.add_argument("--out-cloud-artifact", default=None)
    parser.add_argument("--initialization-note", default="Released LIBERO full MoT; new codec/projector and LoRA untrained")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.model_base_path:
        os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = args.model_base_path
    for name in ("base", "siglip", "stats"):
        if not Path(getattr(args, name)).is_file():
            raise FileNotFoundError(f"Missing {name}: {getattr(args, name)}")
    for name in ("out_delta", "out_edge_artifact"):
        output = Path(getattr(args, name))
        if output.exists() and not args.force:
            raise FileExistsError(f"Refusing to replace {output}; use --force")
        output.parent.mkdir(parents=True, exist_ok=True)

    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root), str(root / "src"), str(root / "deploy_cloudedge")]
    import numpy as np
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from server_common import compose_deploy_cfg, strip_cloud_weights
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    started = time.perf_counter()
    original_base = args.base
    if args.io_overrides:
        if not args.out_prepared_base:
            raise ValueError("--io-overrides requires --out-prepared-base")
        print("Preparing local pretrained base with matching LIBERO I/O", flush=True)
        payload = torch.load(args.base, map_location="cpu", weights_only=False, mmap=True)
        io_payload = torch.load(args.io_overrides, map_location="cpu", weights_only=False)
        payload["mot"].update(io_payload["mot"])
        payload["proprio_encoder"] = io_payload["proprio_encoder"]
        Path(args.out_prepared_base).parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, args.out_prepared_base)
        del payload, io_payload
        args.base = args.out_prepared_base
        print(f"Prepared local base {args.base}", flush=True)
    print("Building performance-only model: pretrained base, untrained new modules", flush=True)
    cfg = compose_deploy_cfg(args.task, [
        f"model.mot_pretrained_path={args.base}",
        f"model.edge_vision.checkpoint_path={args.siglip}",
        "model.skip_dit_load_from_pretrain=true",
        "model.action_dit_pretrained_path=null",
        "model.load_text_encoder=true",
        "model.redirect_common_files=true",
    ])
    # The factory loads full MoT before injecting wrappers. Do not call
    # load_checkpoint(base) again after this instantiate operation.
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=args.device)
    model.requires_grad_(False).eval()
    wrappers = model.video_lora_wrappers
    if len(wrappers) != 120 or any(torch.count_nonzero(w.lora_b).item() for w in wrappers):
        raise RuntimeError("Expected 120 zero-update rank-16 q/k/v/o LoRA wrappers")
    codec = model.cloud_kv_codec
    if codec is None or codec.num_tokens != 56 or codec.cache_dim != 3072:
        raise RuntimeError("Performance recipe does not match a6 codec dimensions")
    metadata = {
        "performance_only": True,
        "trained_leapbotce_delta": False,
        "new_modules_untrained": ["planning codec", "edge vision projector", "video LoRA (zero update)"],
        "initialization_note": args.initialization_note,
        "base_checkpoint": str(Path(args.base).resolve()),
        "original_base_checkpoint": str(Path(original_base).resolve()),
        "io_override_checkpoint": args.io_overrides,
        "siglip_checkpoint": str(Path(args.siglip).resolve()),
        "dataset_stats": str(Path(args.stats).resolve()),
        "seed": args.seed,
        "task": args.task,
        "dtype": "bfloat16",
        "num_layers": int(model.mot.num_layers),
        "num_tokens": int(codec.num_tokens),
        "cache_dim": int(codec.cache_dim),
        "payload_bytes": int(codec.num_tokens * codec.cache_dim * 2),
        "lora_wrappers": len(wrappers),
        "lora_rank": 16,
        "warning": "Timing/throughput artifact only; no task accuracy or retention claims.",
        "resolved_model_config": OmegaConf.to_container(cfg.model, resolve=True),
    }
    print(f"Model ready after {time.perf_counter() - started:.1f}s; exporting delta", flush=True)
    model.save_checkpoint(args.out_delta, step=0)
    # JSON sidecars preserve provenance without reloading the multi-GB delta.
    Path(args.out_delta + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Delta exported: {args.out_delta}", flush=True)
    scale_k = codec.scale_k.detach().cpu().clone()
    scale_v = codec.scale_v.detach().cpu().clone()
    num_layers, num_tokens = int(model.mot.num_layers), int(codec.num_tokens)
    del codec, wrappers
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(args.stats))
    # Deployment functions need only these fields, not simulator state.
    policy = SimpleNamespace(model=model, processor=processor, _prompt_cache={},
                             action_horizon=32, replan_steps=10, num_inference_steps=10,
                             sigma_shift=None, seed=None, rand_device="cpu")
    if args.out_cloud_artifact:
        # Cloud startup may load locally once, then discard the edge modules.
        Path(args.out_cloud_artifact).parent.mkdir(parents=True, exist_ok=True)
        torch.save({**metadata, "policy": policy}, args.out_cloud_artifact)
        print(f"Full CPU artifact exported: {args.out_cloud_artifact}", flush=True)
    strip_cloud_weights(model)
    model.to("cpu")
    model.device = torch.device("cpu")
    if model.video_expert is not None or model.vae is not None or model.text_encoder is not None:
        raise RuntimeError("Cloud modules remain in edge artifact")
    if any(name in model.dit for name in ("video_lora", "cloud_kv_codec")):
        raise RuntimeError("Training registry retains cloud parameters")
    artifact = {**metadata, "policy": policy, "scale_k": scale_k, "scale_v": scale_v,
                "num_layers": num_layers, "num_tokens": num_tokens}
    torch.save(artifact, args.out_edge_artifact)
    Path(args.out_edge_artifact + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"CPU edge artifact exported: {args.out_edge_artifact}", flush=True)
    print(f"Finished in {time.perf_counter() - started:.1f}s; edge parameters "
          f"{sum(p.numel() for p in model.parameters()):,}", flush=True)


if __name__ == "__main__":
    main()
