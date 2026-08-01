"""Validate LeapBot's teacher-forced and public-runtime KV contracts.

The formal invocation loads one real LIBERO prefix and the 6B checkpoint.  It
captures the current ActionDiT call made by ``model.training_loss`` and rebuilds
the same persistent real-data prefix through ``infer_action`` and
``commit_executed_actions``. Persistent K/V must be bitwise equal. The
teacher-forced GT/noised-GT condition and generated runtime condition must have
the same layer/token contract, but are intentionally not numerically equal.
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
    """Compare persistent K/V and transient conditioning contracts."""

    if sample["video"].shape[0] != 1:
        raise ValueError("equivalence validation requires batch size 1")
    action_is_pad = sample.get("action_is_pad")
    if action_is_pad is None:
        raise ValueError(
            "equivalence validation requires an explicit action_is_pad mask"
        )
    if action_is_pad.dtype != torch.bool or action_is_pad.ndim != 2:
        raise ValueError("action_is_pad must be bool [B,T]")
    if bool(action_is_pad.any().item()):
        raise ValueError(
            "runtime/training action equivalence requires a complete, unpadded "
            "action horizon; padded episode tails use a deliberately different "
            "key mask and must be validated by the padding-isolation audit"
        )
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
        runtime_action_records: list[dict[str, Any]] = []

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
            runtime_action_records.append(
                {
                    "history_kv": [
                        {"k": value["k"].detach(), "v": value["v"].detach()}
                        for value in kwargs["history_kv"]
                    ],
                    "tokens": kwargs["action_tokens"].detach(),
                }
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
                    num_video_frames=int(sample["video"].shape[2]),
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
                num_video_frames=int(sample["video"].shape[2]),
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
        if not runtime_action_records:
            raise RuntimeError("public runtime never invoked ActionDiT")
        runtime_action_prefix = runtime_action_records[-1]["history_kv"]
        persistent_tokens = int(runtime_prefix[0]["k"].shape[1])
        training_action_tokens = int(
            training_record["history_kv"][0]["k"].shape[1]
        )
        runtime_action_tokens = int(runtime_action_prefix[0]["k"].shape[1])
        if training_action_tokens <= persistent_tokens:
            raise RuntimeError("training action prefix lacks future-video K/V")
        if runtime_action_tokens <= persistent_tokens:
            raise RuntimeError("runtime action prefix lacks future-video K/V")
        training_persistent_prefix = [
            {
                "k": layer["k"][:, :persistent_tokens],
                "v": layer["v"][:, :persistent_tokens],
            }
            for layer in training_record["history_kv"]
        ]
        prefix_comparison = compare_layerwise_kv(
            training_persistent_prefix,
            runtime_prefix,
            atol=atol,
            rtol=rtol,
        )
        training_transient_tokens = training_action_tokens - persistent_tokens
        runtime_transient_tokens = runtime_action_tokens - persistent_tokens
        transient_shape_match = (
            training_transient_tokens == runtime_transient_tokens
            and all(
                training_layer["k"][:, persistent_tokens:].shape
                == runtime_layer["k"][:, persistent_tokens:].shape
                and training_layer["v"][:, persistent_tokens:].shape
                == runtime_layer["v"][:, persistent_tokens:].shape
                for training_layer, runtime_layer in zip(
                    training_record["history_kv"], runtime_action_prefix
                )
            )
        )
        transient_finite = all(
            torch.isfinite(value).all()
            for prefix in (
                training_record["history_kv"],
                runtime_action_prefix,
            )
            for layer in prefix
            for value in (
                layer["k"][:, persistent_tokens:],
                layer["v"][:, persistent_tokens:],
            )
        )

        training_kinds = [event["kind"] for event in training_events]
        runtime_kinds = [event["kind"] for event in runtime_events]
        expected_training = []
        expected_runtime = []
        for _ in range(history_count):
            expected_training.extend(("video_prefill", "action_prefill"))
            expected_runtime.extend(
                (
                    "video_prefill",
                    "video_prefill",
                    "video_prefill",
                    "action_forward",
                    "action_prefill",
                )
            )
        expected_training.extend(
            (
                "video_prefill",
                "video_prefill",
                "action_forward",
                "video_prefill",
            )
        )
        expected_runtime.extend(
            (
                "video_prefill",
                "video_prefill",
                "video_prefill",
                "action_forward",
            )
        )
        training_sequence_valid = training_kinds == expected_training
        runtime_sequence_valid = runtime_kinds == expected_runtime
        # Training teacher-forces one condition prefill; rollout first denoises
        # the imagined video and then performs the same final cache prefill.
        # Their persistent prefix must be bitwise equal, while transient values
        # are distribution-aligned rather than numerically identical.
        conditioning_contract_pass = bool(
            training_sequence_valid
            and runtime_sequence_valid
            and prefix_comparison["bitwise_equal"]
            and transient_shape_match
            and transient_finite
        )
        bitwise_pass = bool(
            conditioning_contract_pass
        )
        tolerance_pass = bool(
            training_sequence_valid
            and runtime_sequence_valid
            and prefix_comparison["allclose"]
            and transient_shape_match
            and transient_finite
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
            "transient_future_video": {
                "training_tokens": training_transient_tokens,
                "runtime_tokens": runtime_transient_tokens,
                "shape_match": transient_shape_match,
                "finite": bool(transient_finite),
                "numeric_equality_expected": False,
            },
            "sequence": {
                "training": training_events,
                "runtime": runtime_events,
                "training_valid": training_sequence_valid,
                "runtime_valid": runtime_sequence_valid,
                "persistent_path_bitwise_equal": prefix_comparison[
                    "bitwise_equal"
                ],
                "runtime_conditioning_valid": conditioning_contract_pass,
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


def validate_incremental_packed_loss_equivalence(
    model,
    sample: dict[str, Any],
    *,
    seed: int = 1203,
    tiled: bool = False,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> dict[str, Any]:
    """Compare the video-flow branch with the one-shot causal reference.

    Resetting the RNG before each program makes the video/action noise and
    sampled timesteps identical. ActionDiT now has a distinct teacher-forced
    video condition that the legacy packed oracle does not model, so this audit
    explicitly isolates the unchanged video flow objective.
    """

    from leapbot_va.training import _packed_causal_history_reference_loss

    if sample["video"].shape[0] != 1:
        raise ValueError("packed-loss equivalence validation requires batch size 1")
    action_is_pad = sample.get("action_is_pad")
    if action_is_pad is None or action_is_pad.dtype != torch.bool:
        raise ValueError("packed-loss equivalence requires bool action_is_pad")
    if bool(action_is_pad.any().item()):
        raise ValueError(
            "packed-loss equivalence requires a fully real action target; "
            "padded tails are covered by the padding-isolation audit"
        )

    device = torch.device(model.device)
    was_training = bool(model.training)
    original_action_lambda = float(model.loss_lambda_action)
    model.eval()
    try:
        model.loss_lambda_action = 0.0
        _set_seed(seed, device)
        with torch.no_grad():
            incremental_total, incremental_metrics = model.training_loss(
                sample, tiled=tiled
            )
        _set_seed(seed, device)
        with torch.no_grad():
            packed_total, packed_metrics = _packed_causal_history_reference_loss(
                model, sample, tiled=tiled
            )
    finally:
        model.loss_lambda_action = original_action_lambda
        if was_training:
            model.train()

    total = compare_tensors(
        incremental_total.detach().reshape(1),
        packed_total.detach().reshape(1),
        atol=atol,
        rtol=rtol,
    )
    metric_names = sorted(
        name
        for name in set(incremental_metrics) & set(packed_metrics)
        if name.startswith("loss_video_d")
        or name in {"history_blocks_mean", "history_blocks_max"}
    )
    if not any(name.startswith("loss_video_d") for name in metric_names):
        raise ValueError("packed video-flow audit found no common video metric")
    metrics: dict[str, Any] = {}
    for name in metric_names:
        incremental_value = torch.as_tensor(
            incremental_metrics[name], dtype=torch.float64
        ).reshape(1)
        packed_value = torch.as_tensor(
            packed_metrics[name], dtype=torch.float64
        ).reshape(1)
        metrics[name] = compare_tensors(
            incremental_value,
            packed_value,
            atol=atol,
            rtol=rtol,
        )
        metrics[name].update(
            {
                "incremental": float(incremental_value.item()),
                "packed": float(packed_value.item()),
            }
        )
    tolerance_pass = bool(
        total["allclose"]
        and all(comparison["allclose"] for comparison in metrics.values())
    )
    bitwise_pass = bool(
        total["bitwise_equal"]
        and all(
            comparison["bitwise_equal"] for comparison in metrics.values()
        )
    )
    return {
        "scope": "video_flow_only",
        "seed": int(seed),
        "atol": float(atol),
        "rtol": float(rtol),
        "total": total,
        "metrics": metrics,
        "bitwise_pass": bitwise_pass,
        "tolerance_pass": tolerance_pass,
    }


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
    parser.add_argument("--packed-atol", type=float, default=1e-3)
    parser.add_argument("--packed-rtol", type=float, default=1e-3)
    parser.add_argument(
        "--skip-packed-loss-equivalence",
        action="store_true",
        help="Skip the independent one-shot causal loss oracle.",
    )
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
            if block != args.history_blocks:
                continue
            candidate = dataset[dataset_index]
            action_is_pad = candidate.get("action_is_pad")
            if action_is_pad is None:
                raise ValueError(
                    "LeapBot dataset sample is missing action_is_pad"
                )
            if not bool(action_is_pad.any().item()):
                selected_index = dataset_index
                break
        if selected_index is None:
            raise ValueError(
                "dataset contains no complete, unpadded action target with "
                f"full-prefix H={args.history_blocks}"
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
    packed_loss_equivalence = None
    if not args.skip_packed_loss_equivalence:
        packed_loss_equivalence = validate_incremental_packed_loss_equivalence(
            model,
            sample,
            seed=args.seed,
            tiled=args.tiled,
            atol=args.packed_atol,
            rtol=args.packed_rtol,
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
            "action_target_is_fully_real": True,
            "dtype": str(dtype),
            "device": str(args.device),
            "seed": int(args.seed),
            "source": _git_metadata(ROOT_DIR),
            "packed_loss_equivalence": packed_loss_equivalence,
        }
    )
    encoded = json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True)
    print(encoded)
    if args.output_json is not None:
        _write_json(args.output_json.expanduser().resolve(), result)
    passed = result["bitwise_pass"] if args.require_bitwise else result["tolerance_pass"]
    if packed_loss_equivalence is not None:
        passed = passed and packed_loss_equivalence["tolerance_pass"]
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
