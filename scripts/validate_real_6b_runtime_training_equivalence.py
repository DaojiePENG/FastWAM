"""Validate the incremental training action path against public KV runtime.

The formal invocation loads one real LIBERO prefix and the 6B checkpoint.  It
captures the current ActionDiT call made by ``model.training_loss``, rebuilds
the same persistent real-data prefix through ``infer_action`` and
``commit_executed_actions``, and replays the captured action query against the
runtime cache.  No predicted or future-video K/V is used for the comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    ROOT_DIR / "checkpoints/fastwam_release/libero_uncond_2cam224.pt"
)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path, *, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def path_sha256(path: Path) -> str:
    """Hash one checkpoint file or a directory tree deterministically."""

    path = path.expanduser().resolve()
    if path.is_file():
        return _file_sha256(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"checkpoint directory contains no files: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_file_sha256(item)))
    return digest.hexdigest()


def compare_tensors(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> dict[str, Any]:
    if candidate.shape != reference.shape:
        raise ValueError(
            "tensor shape mismatch: "
            f"candidate={tuple(candidate.shape)} reference={tuple(reference.shape)}"
        )
    difference = candidate.float() - reference.float()
    if difference.numel():
        max_abs = float(difference.abs().max().item())
        rmse = float(difference.square().mean().sqrt().item())
    else:
        max_abs = 0.0
        rmse = 0.0
    return {
        "shape": list(candidate.shape),
        "candidate_dtype": str(candidate.dtype),
        "reference_dtype": str(reference.dtype),
        "bitwise_equal": bool(torch.equal(candidate, reference)),
        "allclose": bool(
            torch.allclose(
                candidate.float(),
                reference.float(),
                atol=float(atol),
                rtol=float(rtol),
            )
        ),
        "max_abs": max_abs,
        "rmse": rmse,
        "finite": bool(
            torch.isfinite(candidate).all().item()
            and torch.isfinite(reference).all().item()
        ),
    }


def compare_layerwise_kv(
    candidate: list[dict[str, torch.Tensor]],
    reference: list[dict[str, torch.Tensor]],
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> dict[str, Any]:
    if len(candidate) != len(reference):
        raise ValueError(
            f"KV depth mismatch: candidate={len(candidate)} reference={len(reference)}"
        )
    maximum = 0.0
    squared_error = 0.0
    element_count = 0
    bitwise_equal = True
    allclose = True
    finite = True
    token_counts: set[int] = set()
    for layer, (candidate_layer, reference_layer) in enumerate(
        zip(candidate, reference)
    ):
        for name in ("k", "v"):
            left = candidate_layer[name]
            right = reference_layer[name]
            if left.shape != right.shape:
                raise ValueError(
                    f"KV shape mismatch at layer={layer} name={name}: "
                    f"candidate={tuple(left.shape)} reference={tuple(right.shape)}"
                )
            token_counts.add(int(left.shape[1]))
            difference = left.float() - right.float()
            if difference.numel():
                maximum = max(maximum, float(difference.abs().max().item()))
                squared_error += float(difference.square().sum().item())
                element_count += int(difference.numel())
            bitwise_equal = bitwise_equal and bool(torch.equal(left, right))
            allclose = allclose and bool(
                torch.allclose(
                    left.float(), right.float(), atol=float(atol), rtol=float(rtol)
                )
            )
            finite = finite and bool(
                torch.isfinite(left).all().item()
                and torch.isfinite(right).all().item()
            )
    if len(token_counts) != 1:
        raise ValueError(f"inconsistent KV token counts across layers: {token_counts}")
    return {
        "layers": len(candidate),
        "tokens": next(iter(token_counts)),
        "bitwise_equal": bitwise_equal,
        "allclose": allclose,
        "max_abs": maximum,
        "rmse": (squared_error / max(element_count, 1)) ** 0.5,
        "finite": finite,
    }


@contextmanager
def _replace_methods(module, **replacements) -> Iterator[None]:
    originals = {name: getattr(module, name) for name in replacements}
    try:
        for name, replacement in replacements.items():
            setattr(module, name, replacement)
        yield
    finally:
        for name, original in originals.items():
            setattr(module, name, original)


def _event(
    kind: str,
    *,
    history_kv: list[dict[str, torch.Tensor]] | None,
    tokens: torch.Tensor,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "query_tokens": int(tokens.shape[1]),
        "history_tokens": 0 if history_kv is None else int(history_kv[0]["k"].shape[1]),
    }


def _set_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))


def validate_incremental_action_equivalence(
    model,
    sample: dict[str, Any],
    *,
    seed: int = 1203,
    tiled: bool = False,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> dict[str, Any]:
    """Compare a captured training action call with a public-runtime prefix."""

    if sample["video"].shape[0] != 1:
        raise ValueError("equivalence validation requires batch size 1")
    if model.history_training_mode != "incremental_full_bptt":
        raise ValueError(
            "model must use history_training_mode=incremental_full_bptt"
        )
    final_depth = int(model.mot.num_layers)
    if final_depth not in tuple(int(value) for value in model.training_exit_depths):
        raise ValueError("training_exit_depths must contain the final depth")

    history_valid = sample["history_valid_blocks"][0].to(dtype=torch.bool)
    history_count = int(history_valid.sum().item())
    if history_valid.numel() and not torch.equal(
        history_valid,
        torch.arange(history_valid.numel(), device=history_valid.device)
        < history_count,
    ):
        raise ValueError("history_valid_blocks must be a contiguous prefix")
    expected_positions = torch.arange(
        history_count,
        dtype=sample["history_block_positions"].dtype,
        device=sample["history_block_positions"].device,
    )
    if not torch.equal(
        sample["history_block_positions"][0, :history_count], expected_positions
    ):
        raise ValueError("public runtime replay requires a full episode prefix from block 0")

    device = torch.device(model.device)
    was_training = bool(model.training)
    model.eval()
    original_prefill = model.mot.prefill_expert_segment
    original_action = model.mot.forward_action_with_history
    training_events: list[dict[str, Any]] = []
    training_record: dict[str, Any] = {}

    def capture_training_prefill(*args, **kwargs):
        training_events.append(
            _event(
                f"{kwargs['expert_name']}_prefill",
                history_kv=kwargs.get("history_kv"),
                tokens=kwargs["tokens"],
            )
        )
        return original_prefill(*args, **kwargs)

    def capture_training_action(*args, **kwargs):
        if training_record:
            raise RuntimeError("expected exactly one current action call for batch size 1")
        training_events.append(
            _event(
                "action_forward",
                history_kv=kwargs.get("history_kv"),
                tokens=kwargs["action_tokens"],
            )
        )
        output = original_action(*args, **kwargs)
        if not isinstance(output, dict) or final_depth not in output:
            raise RuntimeError("training action call did not expose the final-depth hidden")
        training_record.update(
            {
                "action_tokens": kwargs["action_tokens"].detach(),
                "action_freqs": kwargs["action_freqs"].detach(),
                "action_t_mod": kwargs["action_t_mod"].detach(),
                "action_context_payload": {
                    "context": kwargs["action_context_payload"]["context"].detach(),
                    "mask": kwargs["action_context_payload"]["mask"].detach(),
                },
                "history_kv": [
                    {"k": value["k"].detach(), "v": value["v"].detach()}
                    for value in kwargs["history_kv"]
                ],
                "hidden": output[final_depth].detach(),
            }
        )
        return output

    try:
        _set_seed(seed, device)
        with _replace_methods(
            model.mot,
            prefill_expert_segment=capture_training_prefill,
            forward_action_with_history=capture_training_action,
        ), torch.no_grad():
            training_loss, training_metrics = model.training_loss(
                sample, tiled=tiled
            )
        if not training_record:
            raise RuntimeError("training loss never invoked current ActionDiT")

        runtime_events: list[dict[str, Any]] = []

        def capture_runtime_prefill(*args, **kwargs):
            runtime_events.append(
                _event(
                    f"{kwargs['expert_name']}_prefill",
                    history_kv=kwargs.get("history_kv"),
                    tokens=kwargs["tokens"],
                )
            )
            return original_prefill(*args, **kwargs)

        def capture_runtime_action(*args, **kwargs):
            runtime_events.append(
                _event(
                    "action_forward",
                    history_kv=kwargs.get("history_kv"),
                    tokens=kwargs["action_tokens"],
                )
            )
            return original_action(*args, **kwargs)

        action_horizon = int(sample["action"].shape[1])
        replan_steps = int(sample["history_action"].shape[2])
        memory = model.create_memory(
            exit_depth=final_depth,
            causal_mode=model.causal_mode,
            max_history_blocks=history_count + 1,
            action_horizon=action_horizon,
            replan_steps=replan_steps,
        )
        base_context = sample["context"][0:1]
        base_context_mask = sample["context_mask"][0:1]
        with _replace_methods(
            model.mot,
            prefill_expert_segment=capture_runtime_prefill,
            forward_action_with_history=capture_runtime_action,
        ), torch.no_grad():
            for history_index in range(history_count):
                model.infer_action(
                    prompt=None,
                    input_image=sample["history_video"][
                        0:1, :, history_index
                    ],
                    action_horizon=action_horizon,
                    proprio=sample["history_proprio"][
                        0:1, history_index
                    ],
                    context=base_context,
                    context_mask=base_context_mask,
                    num_inference_steps=1,
                    seed=seed + 10_000 + history_index,
                    rand_device="cpu",
                    tiled=tiled,
                    memory=memory,
                )
                model.commit_executed_actions(
                    memory,
                    sample["history_action"][0:1, history_index],
                )
            model.infer_action(
                prompt=None,
                input_image=sample["video"][0:1, :, 0],
                action_horizon=action_horizon,
                proprio=sample["proprio"][0:1, 0],
                context=base_context,
                context_mask=base_context_mask,
                num_inference_steps=1,
                seed=seed + 20_000,
                rand_device="cpu",
                tiled=tiled,
                memory=memory,
            )

        runtime_prefix = memory.materialize(memory.selected_segments_for_action())
        if runtime_prefix is None:
            raise RuntimeError("public runtime did not retain the current observation")
        prefix_comparison = compare_layerwise_kv(
            training_record["history_kv"],
            runtime_prefix,
            atol=atol,
            rtol=rtol,
        )
        with torch.no_grad():
            runtime_hidden = original_action(
                action_tokens=training_record["action_tokens"],
                action_freqs=training_record["action_freqs"],
                action_t_mod=training_record["action_t_mod"],
                action_context_payload=training_record[
                    "action_context_payload"
                ],
                history_kv=runtime_prefix,
                max_layers=final_depth,
            )
            if isinstance(runtime_hidden, dict):
                runtime_hidden = runtime_hidden[final_depth]
            training_hidden = training_record["hidden"]
            action_head = model._action_head_at_depth(final_depth)
            training_head = action_head(training_hidden)
            runtime_head = action_head(runtime_hidden)

        hidden_comparison = compare_tensors(
            training_hidden, runtime_hidden, atol=atol, rtol=rtol
        )
        head_comparison = compare_tensors(
            training_head, runtime_head, atol=atol, rtol=rtol
        )

        training_kinds = [event["kind"] for event in training_events]
        runtime_kinds = [event["kind"] for event in runtime_events]
        expected_training = []
        expected_runtime = []
        for _ in range(history_count):
            expected_training.extend(("video_prefill", "action_prefill"))
            expected_runtime.extend(
                ("video_prefill", "action_forward", "action_prefill")
            )
        expected_training.extend(
            ("video_prefill", "action_forward", "video_prefill")
        )
        expected_runtime.extend(("video_prefill", "action_forward"))
        training_sequence_valid = training_kinds == expected_training
        runtime_sequence_valid = runtime_kinds == expected_runtime
        training_action_path = training_kinds[:-1]
        runtime_action_path = [
            kind for kind in runtime_kinds if kind != "action_forward"
        ] + ["action_forward"]
        sequence_isomorphic = (
            training_sequence_valid
            and runtime_sequence_valid
            and training_action_path == runtime_action_path
        )
        bitwise_pass = bool(
            sequence_isomorphic
            and prefix_comparison["bitwise_equal"]
            and hidden_comparison["bitwise_equal"]
            and head_comparison["bitwise_equal"]
        )
        tolerance_pass = bool(
            sequence_isomorphic
            and prefix_comparison["allclose"]
            and hidden_comparison["allclose"]
            and head_comparison["allclose"]
        )
        return {
            "history_blocks": history_count,
            "causal_mode": str(model.causal_mode),
            "exit_depth": final_depth,
            "replan_steps": replan_steps,
            "action_horizon": action_horizon,
            "history_training_mode": str(model.history_training_mode),
            "training_loss": float(training_loss.detach().float().item()),
            "training_metrics": {
                key: float(value) for key, value in training_metrics.items()
            },
            "prefix_kv": prefix_comparison,
            "hidden": hidden_comparison,
            "head": head_comparison,
            "sequence": {
                "training": training_events,
                "runtime": runtime_events,
                "training_valid": training_sequence_valid,
                "runtime_valid": runtime_sequence_valid,
                "action_path_isomorphic": sequence_isomorphic,
            },
            "runtime_memory": {
                "completed_blocks": int(memory.completed_blocks),
                "token_counts": memory.token_counts,
                "cache_bytes": int(memory.cache_nbytes),
                "phase": memory.phase.value,
            },
            "bitwise_pass": bitwise_pass,
            "tolerance_pass": tolerance_pass,
            "atol": float(atol),
            "rtol": float(rtol),
        }
    finally:
        if was_training:
            model.train()


def _git_metadata(root: Path) -> dict[str, Any]:
    def run(*args: str) -> bytes:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
        ).stdout

    head = run("rev-parse", "HEAD").decode("ascii").strip()
    status = run("status", "--short")
    diff = run("diff", "--binary", "HEAD")
    return {
        "head": head,
        "dirty": bool(status.strip()),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-stats", type=Path)
    parser.add_argument(
        "--causal-mode",
        choices=("interleaved", "vision_causal", "action_aggregator"),
        default="action_aggregator",
    )
    parser.add_argument("--history-blocks", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("bf16", "fp32"), default="bf16"
    )
    parser.add_argument("--seed", type=int, default=1203)
    parser.add_argument("--dataset-index", type=int)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument(
        "--require-bitwise",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--skip-checkpoint-hash", action="store_true")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.history_blocks < 0:
        raise ValueError("history-blocks must be non-negative")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    dataset_stats = args.dataset_stats
    if dataset_stats is None and checkpoint.is_file():
        candidate = checkpoint.with_name(
            f"{checkpoint.stem}_dataset_stats.json"
        )
        if candidate.is_file():
            dataset_stats = candidate
    if dataset_stats is None:
        raise ValueError(
            "provide --dataset-stats; validation must not recompute normalization"
        )
    dataset_stats = dataset_stats.expanduser().resolve()
    if not dataset_stats.is_file():
        raise FileNotFoundError(dataset_stats)
    os.environ["LEAPBOT_DATASET_STATS"] = str(dataset_stats)
    os.chdir(ROOT_DIR)

    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from torch.utils.data._utils.collate import default_collate

    from fastwam.utils.config_resolvers import register_default_resolvers

    register_default_resolvers()
    with initialize_config_dir(
        version_base=None, config_dir=str((ROOT_DIR / "configs").resolve())
    ):
        cfg = compose(
            config_name="train",
            overrides=[
                "task=libero_leapbot_2cam224",
                f"model.causal_mode={args.causal_mode}",
                "model.training_exit_depths=[30]",
                "model.history_training_mode=incremental_full_bptt",
                "model.history_vae_batch_chunk_size=1",
                "batch_size=1",
            ],
        )
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    dataset = instantiate(cfg.data.train)
    if args.dataset_index is None:
        selected_index = None
        for dataset_index, frame_index in enumerate(dataset._valid_replan_indices):
            block = int(dataset._episode_step[frame_index]) // int(
                dataset.replan_steps
            )
            if block == args.history_blocks:
                selected_index = dataset_index
                break
        if selected_index is None:
            raise ValueError(
                f"dataset contains no full-prefix sample with H={args.history_blocks}"
            )
    else:
        selected_index = int(args.dataset_index)
    sample = default_collate([dataset[selected_index]])
    actual_history = int(sample["history_valid_blocks"].sum().item())
    if actual_history != args.history_blocks:
        raise ValueError(
            f"selected sample has H={actual_history}, expected {args.history_blocks}"
        )

    model = instantiate(
        cfg.model,
        model_dtype=dtype,
        device=args.device,
    )
    model.load_checkpoint(str(checkpoint), optimizer=None)
    result = validate_incremental_action_equivalence(
        model,
        sample,
        seed=args.seed,
        tiled=args.tiled,
        atol=args.atol,
        rtol=args.rtol,
    )
    resolved_model_config = OmegaConf.to_container(cfg.model, resolve=True)
    resolved_data_config = OmegaConf.to_container(cfg.data.train, resolve=True)
    checkpoint_sha256 = (
        None if args.skip_checkpoint_hash else path_sha256(checkpoint)
    )
    result.update(
        {
            "kind": "real_6b_runtime_training_action_equivalence",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "weights_sha256": checkpoint_sha256,
            "weights_hash_scope": "checkpoint_artifact",
            "dataset_stats": str(dataset_stats),
            "dataset_stats_sha256": _file_sha256(dataset_stats),
            "model_config_sha256": _canonical_sha256(resolved_model_config),
            "data_config_sha256": _canonical_sha256(resolved_data_config),
            "validation_script_sha256": _file_sha256(Path(__file__).resolve()),
            "resolved_model_config": resolved_model_config,
            "dataset_index": selected_index,
            "dtype": str(dtype),
            "device": str(args.device),
            "seed": int(args.seed),
            "source": _git_metadata(ROOT_DIR),
        }
    )
    encoded = json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True)
    print(encoded)
    if args.output_json is not None:
        _write_json(args.output_json.expanduser().resolve(), result)
    passed = result["bitwise_pass"] if args.require_bitwise else result["tolerance_pass"]
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
