"""Paired fixed-noise loss audit across exact causal-history lengths.

This is a checkpoint-selection tool, not a training loop.  Every checkpoint is
evaluated on the same LeRobot samples with the same flow-matching noise and
timesteps.  Reporting exact history lengths makes it possible to distinguish a
healthy H=0 FastWAM-compatible path from degradation introduced by longer
causal prefixes.

The legacy CLI remains a one-draw ``correct``-history audit.  Supplying
``stratified.fixed_u_values`` enables order-independent stateless noise draws;
``stratified.history_variants=[correct,masked,shuffled]`` adds causal history
controls, and ``stratified.noise_repeats`` repeats Gaussian draws at every fixed
timestep.  Enhanced mode evaluates the native path for every checkpoint so the
incremental-prefix effect and parameter drift can be reported separately.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
import time
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data._utils.collate import default_collate

from fastwam.models.wan22.fastwam import FastWAM
from fastwam.runtime import _mixed_precision_to_model_dtype, _normalize_mixed_precision
from fastwam.utils import misc
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.logging_config import get_logger, setup_logging
from leapbot_va.eval_contract import _git_source_identity


register_default_resolvers()
logger = get_logger(__name__)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_seed(*parts: Any) -> int:
    """Return an order-independent torch-compatible seed for one audit draw."""

    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    # torch.Generator.manual_seed accepts signed 64-bit values.  Staying below
    # 2**63 also keeps the value portable through JSON and NumPy.
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _episode_ids_for_frames(dataset) -> dict[int, int]:
    """Resolve globally unique episode ids without changing the core dataset."""

    explicit = getattr(dataset, "_episode_id", None)
    if explicit is not None:
        return {int(frame): int(episode) for frame, episode in explicit.items()}

    base = getattr(dataset, "lerobot_dataset", None)
    episode_data_index = getattr(base, "episode_data_index", None)
    if episode_data_index is None:
        # Lightweight unit-test datasets may not expose LeRobot internals.  A
        # frame-unique fallback is safe for selection tests, while production
        # LeapRobotVideoDataset always takes the interval branch above.
        return {int(frame): int(frame) for frame in dataset._valid_replan_indices}

    starts = [int(value) for value in episode_data_index["from"].tolist()]
    stops = [int(value) for value in episode_data_index["to"].tolist()]
    result: dict[int, int] = {}
    episode = 0
    for frame in sorted(int(value) for value in dataset._valid_replan_indices):
        while episode + 1 < len(starts) and frame >= stops[episode]:
            episode += 1
        if not (starts[episode] <= frame < stops[episode]):
            raise ValueError(f"frame {frame} does not belong to a declared episode")
        result[frame] = episode
    return result


def select_history_samples(
    dataset,
    *,
    history_lengths: list[int],
    samples_per_history: int,
    seed: int,
) -> list[dict[str, int]]:
    """Select deterministic, paired dataset indices for exact history lengths."""

    if samples_per_history <= 0:
        raise ValueError("samples_per_history must be positive")
    if not history_lengths:
        raise ValueError("history_lengths must not be empty")
    if any(value < 0 for value in history_lengths):
        raise ValueError("history lengths must be non-negative")

    requested = set(int(value) for value in history_lengths)
    episode_ids = _episode_ids_for_frames(dataset)
    candidates: dict[int, list[int]] = defaultdict(list)
    for dataset_index, frame_index in enumerate(dataset._valid_replan_indices):
        history = int(dataset._episode_step[frame_index] // dataset.replan_steps)
        if history in requested:
            candidates[history].append(dataset_index)

    selected: list[dict[str, int]] = []
    for history in history_lengths:
        available = candidates.get(int(history), [])
        if not available:
            raise ValueError(f"dataset contains no sample with H={history}")
        count = min(samples_per_history, len(available))
        rng = np.random.default_rng(seed + int(history) * 1_000_003)
        positions = np.sort(rng.choice(len(available), size=count, replace=False))
        for replica, position in enumerate(positions.tolist()):
            selected.append(
                {
                    "history_blocks": int(history),
                    "replica": int(replica),
                    "dataset_index": int(available[position]),
                    "frame_index": int(
                        dataset._valid_replan_indices[int(available[position])]
                    ),
                    "episode_id": int(
                        episode_ids[
                            int(dataset._valid_replan_indices[int(available[position])])
                        ]
                    ),
                    "episode_step": int(history) * int(dataset.replan_steps),
                    "history_population_count": int(len(available)),
                }
            )
    return selected


def build_shuffled_history_donors(
    dataset,
    selected: list[dict[str, int]],
    *,
    seed: int,
) -> dict[int, dict[str, int] | None]:
    """Choose deterministic exact-H donors that are always from another episode.

    The mapping is keyed only by recipient dataset index and is constructed once,
    so every checkpoint, fixed timestep, and noise replica sees the same donor.
    A missing donor is represented explicitly; it is never replaced by the same
    episode (notably, the released LIBERO data has only one H=50 episode).
    """

    episode_ids = _episode_ids_for_frames(dataset)
    pools: dict[int, list[dict[str, int]]] = defaultdict(list)
    for dataset_index, frame_index in enumerate(dataset._valid_replan_indices):
        history = int(dataset._episode_step[frame_index] // dataset.replan_steps)
        pools[history].append(
            {
                "dataset_index": int(dataset_index),
                "frame_index": int(frame_index),
                "episode_id": int(episode_ids[int(frame_index)]),
                "history_blocks": history,
            }
        )

    # Construct one population-level permutation for each exact H, then look up
    # only the selected recipients.  This keeps donor identity independent of
    # selection order/checkpoint order and prevents repeated donors from
    # reducing the effective sample size of the shuffled-history control.
    permutations: dict[int, dict[int, dict[str, int]]] = {}
    for history in {int(row["history_blocks"]) for row in selected}:
        if history == 0:
            continue
        population = pools[history]
        by_episode: dict[int, list[dict[str, int]]] = defaultdict(list)
        for row in population:
            by_episode[int(row["episode_id"])].append(row)

        # A cross-episode bijection exists iff no episode owns more than half
        # the population.  In LeRobot exact-H pools each episode contributes at
        # most one boundary, but retain the general check so malformed/custom
        # datasets fail explicitly rather than self-map.
        population_size = len(population)
        max_episode_size = max(
            (len(rows) for rows in by_episode.values()), default=0
        )
        if population_size < 2 or 2 * max_episode_size > population_size:
            continue

        rng = np.random.default_rng(
            _stable_seed(seed, history, "shuffled-history-bijection")
        )
        episode_ids = sorted(by_episode)
        rng.shuffle(episode_ids)
        ordered: list[dict[str, int]] = []
        for episode_id in episode_ids:
            episode_rows = sorted(
                by_episode[episode_id], key=lambda row: int(row["dataset_index"])
            )
            rng.shuffle(episode_rows)
            ordered.extend(episode_rows)

        donors = ordered[max_episode_size:] + ordered[:max_episode_size]
        mapping: dict[int, dict[str, int]] = {}
        for recipient, donor in zip(ordered, donors):
            if int(recipient["episode_id"]) == int(donor["episode_id"]):
                raise AssertionError("cross-episode donor construction self-mapped")
            mapping[int(recipient["dataset_index"])] = dict(donor)
        if len({int(row["dataset_index"]) for row in mapping.values()}) != len(mapping):
            raise AssertionError("shuffled-history donor mapping is not bijective")
        permutations[history] = mapping

    result: dict[int, dict[str, int] | None] = {}
    for recipient in selected:
        recipient_index = int(recipient["dataset_index"])
        history = int(recipient["history_blocks"])
        result[recipient_index] = permutations.get(history, {}).get(recipient_index)
    return result


def masked_history_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Mask all history while preserving the recipient's absolute current time."""

    result = dict(sample)
    result["history_valid_blocks"] = torch.zeros_like(sample["history_valid_blocks"])
    # Packed training accepts a short/empty prefix only when this flag is false.
    # Current absolute positions remain untouched, isolating history content from
    # episode-time embeddings.
    if "full_episode_history" in sample:
        result["full_episode_history"] = torch.zeros_like(sample["full_episode_history"])
    return result


def shuffled_history_sample(
    recipient: dict[str, Any],
    donor: dict[str, Any],
    *,
    history_blocks: int,
    recipient_episode_id: int,
    donor_episode_id: int,
) -> dict[str, Any]:
    """Replace only real history tensors with an exact-length cross-episode prefix."""

    if history_blocks <= 0:
        return dict(recipient)
    if int(recipient_episode_id) == int(donor_episode_id):
        raise ValueError("shuffled history donor must come from another episode")
    donor_count = int(donor["history_valid_blocks"].sum().item())
    recipient_count = int(recipient["history_valid_blocks"].sum().item())
    if donor_count != history_blocks or recipient_count != history_blocks:
        raise ValueError(
            "shuffled history requires exact-H recipient and donor: "
            f"recipient={recipient_count} donor={donor_count} expected={history_blocks}"
        )

    result = dict(recipient)
    replacements = {
        "history_video": 2,
        "history_action": 1,
        "history_proprio": 1,
    }
    for key, history_dim in replacements.items():
        value = recipient[key].clone()
        index = [slice(None)] * value.ndim
        index[history_dim] = slice(0, history_blocks)
        value[tuple(index)] = donor[key][tuple(index)].to(
            device=value.device, dtype=value.dtype
        )
        result[key] = value
    # Validity and absolute position metadata deliberately remain those of the
    # recipient.  No donor current observation, action target, prompt, or proprio
    # is copied.
    return result


def _fixed_timestep(scheduler, u: float, *, batch_size: int, device, dtype) -> torch.Tensor:
    if not 0.0 <= float(u) < 1.0:
        raise ValueError(f"fixed flow u must lie in [0,1), got {u}")
    shift = float(scheduler.shift)
    sigma = shift * float(u) / (1.0 + (shift - 1.0) * float(u))
    timestep = sigma * float(scheduler.num_train_timesteps)
    return torch.full((batch_size,), timestep, device=device, dtype=dtype)


def _randn_like_from_seed(value: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=value.device)
    generator.manual_seed(int(seed))
    result = torch.empty_like(value, memory_format=torch.preserve_format)
    return result.normal_(generator=generator)


# Fixed per-frame flow u for the noised future-video condition. The training
# default samples u uniformly from [0.5, 1.0]; pinning to the lower bound keeps
# the audit inside that range while making the noise strength deterministic.
_FUTURE_VIDEO_FIXED_U = 0.5


@contextmanager
def fixed_flow_draw(
    model,
    *,
    fixed_u: float | None,
    video_noise_seed: int | None,
    action_noise_seed: int | None,
    future_video_noise_seed: int | None = None,
    future_video_condition: str | None = None,
):
    """Fix flow draws and capture the final action prediction in one forward.

    When explicit seeds are supplied, a local per-device generator produces the
    two noises, so draw identity does not depend on checkpoint order or global RNG
    state.  Legacy callers may leave the seeds and ``fixed_u`` unset and retain
    the original seeded-random behavior.

    ``future_video_condition`` pins the LingBot future-video condition, which the
    training path otherwise samples randomly (50% clean GT / 50% noised GT).  It
    is one of ``"noised"`` (force the noised branch and fix its noise with
    ``future_video_noise_seed``), ``"clean"`` (force the clean-GT branch, so no
    third ``randn_like`` is drawn), or ``None`` (legacy: leave the condition
    untouched).  This keeps the audit reproducible while preserving a balanced
    50/50 mix of clean and noised conditions across draws.
    """

    captured: dict[str, Any] = {}
    original_video_sampler = model.train_video_scheduler.sample_training_t
    original_action_sampler = model.train_action_scheduler.sample_training_t
    original_action_target = model.train_action_scheduler.training_target
    original_post_dit = model.action_expert.post_dit

    def sample_video_t(batch_size, device, dtype):
        timestep = (
            original_video_sampler(batch_size, device, dtype)
            if fixed_u is None
            else _fixed_timestep(
                model.train_video_scheduler,
                fixed_u,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
        )
        captured["timestep_video"] = timestep.detach()
        return timestep

    def sample_action_t(batch_size, device, dtype):
        timestep = (
            original_action_sampler(batch_size, device, dtype)
            if fixed_u is None
            else _fixed_timestep(
                model.train_action_scheduler,
                fixed_u,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
        )
        captured["timestep_action"] = timestep.detach()
        return timestep

    def capture_action_target(sample, noise, timestep):
        target = original_action_target(sample, noise, timestep)
        captured["target_action"] = target.detach()
        captured["action_noise"] = noise.detach()
        return target

    def capture_post_dit(*args, **kwargs):
        prediction = original_post_dit(*args, **kwargs)
        captured["pred_action"] = prediction.detach()
        return prediction

    explicit_noise = video_noise_seed is not None or action_noise_seed is not None
    if explicit_noise and (video_noise_seed is None or action_noise_seed is None):
        raise ValueError("video and action noise seeds must be supplied together")
    noise_call_count = 0

    def deterministic_randn_like(value, *args, **kwargs):
        nonlocal noise_call_count
        if args or kwargs:
            raise ValueError(
                "fixed-flow audit only supports the training path's plain torch.randn_like call"
            )
        if noise_call_count == 0:
            result = _randn_like_from_seed(value, int(video_noise_seed))
            captured["video_noise"] = result.detach()
        elif noise_call_count == 1:
            result = _randn_like_from_seed(value, int(action_noise_seed))
            captured["action_noise_generated"] = result.detach()
        elif noise_call_count == 2:
            if future_video_noise_seed is None:
                raise RuntimeError(
                    "fixed-flow audit observed an unexpected third torch.randn_like call"
                )
            result = _randn_like_from_seed(value, int(future_video_noise_seed))
            captured["future_video_noise"] = result.detach()
        else:
            raise RuntimeError(
                "fixed-flow audit observed an unexpected fourth torch.randn_like call"
            )
        noise_call_count += 1
        return result

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                model.train_video_scheduler,
                "sample_training_t",
                new=sample_video_t,
            )
        )
        stack.enter_context(
            patch.object(
                model.train_action_scheduler,
                "sample_training_t",
                new=sample_action_t,
            )
        )
        stack.enter_context(
            patch.object(
                model.train_action_scheduler,
                "training_target",
                new=capture_action_target,
            )
        )
        stack.enter_context(
            patch.object(model.action_expert, "post_dit", new=capture_post_dit)
        )
        if explicit_noise:
            stack.enter_context(patch.object(torch, "randn_like", new=deterministic_randn_like))
            if future_video_condition == "noised":
                # Force the noised future-video branch and pin its noise strength
                # so the per-frame u draw (training.py sample_lingbot_future_video
                # condition) is deterministic instead of random.
                stack.enter_context(
                    patch.object(
                        model, "future_video_condition_noise_probability", new=1.0
                    )
                )
                stack.enter_context(
                    patch.object(
                        model, "future_video_condition_min_u", new=_FUTURE_VIDEO_FIXED_U
                    )
                )
                stack.enter_context(
                    patch.object(
                        model, "future_video_condition_max_u", new=_FUTURE_VIDEO_FIXED_U
                    )
                )
            elif future_video_condition == "clean":
                # Force the clean-GT branch: no third randn_like is drawn.
                stack.enter_context(
                    patch.object(
                        model, "future_video_condition_noise_probability", new=0.0
                    )
                )
        yield captured

    required = {"timestep_video", "timestep_action", "target_action", "pred_action"}
    missing = sorted(required.difference(captured))
    if missing:
        raise RuntimeError(f"fixed-flow diagnostic failed to capture {missing}")
    if explicit_noise:
        expected_noise_calls = 3 if future_video_condition == "noised" else 2
        if noise_call_count != expected_noise_calls:
            raise RuntimeError(
                f"fixed-flow audit expected {expected_noise_calls} noise draws, "
                f"observed {noise_call_count}"
            )


def compute_action_diagnostics(
    *,
    pred_action: torch.Tensor,
    target_action: torch.Tensor,
    action_is_pad: torch.Tensor | None,
    scheduler_weight: torch.Tensor,
    loss_lambda_action: float,
    executed_action_steps: int,
    continuous_action_dims: int,
    gripper_action_index: int,
) -> dict[str, Any]:
    """Split raw and official weighted FM MSE without another model forward."""

    if target_action.ndim != 3:
        raise ValueError("target_action must be [B,T,D]")
    batch, horizon, action_dim = target_action.shape
    if pred_action.ndim != 3 or pred_action.shape[0] != batch:
        raise ValueError("pred_action must be [B,T,D] with the same batch size")
    if pred_action.shape[-1] != action_dim or pred_action.shape[1] < horizon:
        raise ValueError("pred_action does not contain the complete current action horizon")
    pred_action = pred_action[:, -horizon:]
    if executed_action_steps <= 0 or executed_action_steps > horizon:
        raise ValueError(
            f"executed_action_steps must lie in [1,{horizon}], got {executed_action_steps}"
        )
    if continuous_action_dims <= 0 or continuous_action_dims > action_dim:
        raise ValueError("continuous_action_dims is outside the action dimension")
    if not 0 <= gripper_action_index < action_dim:
        raise ValueError("gripper_action_index is outside the action dimension")
    if gripper_action_index < continuous_action_dims:
        raise ValueError("gripper_action_index must not overlap continuous action dimensions")

    squared_error = (pred_action.float() - target_action.float()).square()
    if action_is_pad is None:
        valid_time = torch.ones(
            (batch, horizon), dtype=torch.bool, device=squared_error.device
        )
    else:
        if tuple(action_is_pad.shape) != (batch, horizon):
            raise ValueError("action_is_pad must match [B,T]")
        valid_time = ~action_is_pad.to(squared_error.device, dtype=torch.bool)
    weight = scheduler_weight.to(squared_error.device, dtype=torch.float32).reshape(-1)
    if weight.numel() == 1 and batch != 1:
        weight = weight.expand(batch)
    if weight.numel() != batch:
        raise ValueError("scheduler_weight must be scalar or [B]")

    time_groups = {
        f"full{horizon}": (0, horizon),
        f"executed{executed_action_steps}": (0, executed_action_steps),
        f"tail{horizon - executed_action_steps}": (executed_action_steps, horizon),
    }
    dim_groups = {
        f"all{action_dim}": list(range(action_dim)),
        f"continuous{continuous_action_dims}dof": list(range(continuous_action_dims)),
        "gripper1": [gripper_action_index],
    }
    nested: dict[str, Any] = {}
    flat: dict[str, float | int | None] = {}
    for time_name, (start, stop) in time_groups.items():
        nested[time_name] = {}
        for dim_name, dimensions in dim_groups.items():
            token_valid = valid_time[:, start:stop]
            valid_count_per_sample = token_valid.sum(dim=1) * len(dimensions)
            total_valid = int(valid_count_per_sample.sum().item())
            if total_valid == 0:
                raw_mse = None
                weighted_fm = None
            else:
                error = squared_error[:, start:stop, dimensions]
                per_sample_sum = (
                    error * token_valid[:, :, None].to(error.dtype)
                ).sum(dim=(1, 2))
                per_sample_valid = valid_count_per_sample.clamp_min(1).to(error.dtype)
                per_sample_mse = per_sample_sum / per_sample_valid
                participating = valid_count_per_sample > 0
                raw_mse = float(per_sample_mse[participating].mean().item())
                weighted_fm = float(
                    (
                        per_sample_mse[participating]
                        * weight[participating]
                        * float(loss_lambda_action)
                    )
                    .mean()
                    .item()
                )
            values = {
                "raw_mse": raw_mse,
                "weighted_fm": weighted_fm,
                "valid_count": total_valid,
            }
            nested[time_name][dim_name] = values
            flat[f"action_raw_mse_{time_name}_{dim_name}"] = raw_mse
            flat[f"action_weighted_fm_{time_name}_{dim_name}"] = weighted_fm
            flat[f"action_valid_count_{time_name}_{dim_name}"] = total_valid

    return {
        "action_horizon": int(horizon),
        "action_dim": int(action_dim),
        "executed_action_steps": int(executed_action_steps),
        "continuous_action_dims": int(continuous_action_dims),
        "gripper_action_index": int(gripper_action_index),
        "segments": nested,
        "flat": flat,
    }


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize empty values")
    array = np.asarray(values, dtype=np.float64)
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": std,
        "sem": float(std / math.sqrt(array.size)),
        "min": float(array.min()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def _pair_key(record: dict[str, Any]) -> tuple[int, ...]:
    """Use a complete draw key while remaining compatible with legacy records."""

    dataset_index = int(record["dataset_index"])
    if "u_index" in record or "noise_replica" in record:
        return (
            dataset_index,
            int(record.get("u_index", 0)),
            int(record.get("noise_replica", 0)),
        )
    return (dataset_index, int(record["noise_seed"]))


def _index_unique_records(
    records: list[dict[str, Any]], *, label: str
) -> dict[tuple[int, ...], dict[str, Any]]:
    indexed: dict[tuple[int, ...], dict[str, Any]] = {}
    for row in records:
        key = _pair_key(row)
        if key in indexed:
            raise ValueError(f"duplicate paired draw key in {label}: {key}")
        indexed[key] = row
    return indexed


def _paired_rows(
    candidate: list[dict[str, Any]],
    reference: list[dict[str, Any]],
    *,
    metric_keys: list[str],
) -> list[dict[str, Any]]:
    candidate_by_key = _index_unique_records(candidate, label="candidate")
    reference_by_key = _index_unique_records(reference, label="reference")
    candidate_keys = set(candidate_by_key)
    reference_keys = set(reference_by_key)
    if candidate_keys != reference_keys:
        missing = sorted(reference_keys - candidate_keys)
        extra = sorted(candidate_keys - reference_keys)
        raise ValueError(
            "paired checkpoint draw keys differ: "
            f"missing_from_candidate={missing[:5]} extra_in_candidate={extra[:5]}"
        )

    rows: list[dict[str, Any]] = []
    for key in sorted(reference_by_key):
        baseline = reference_by_key[key]
        active = candidate_by_key[key]
        row = {
            field: baseline[field]
            for field in (
                "dataset_index",
                "history_blocks",
                "episode_id",
                "history_population_count",
                "noise_seed",
                "u_index",
                "noise_replica",
            )
            if field in baseline
        }
        for metric in metric_keys:
            candidate_value = active.get(metric)
            reference_value = baseline.get(metric)
            row[metric] = (
                None
                if candidate_value is None or reference_value is None
                else float(candidate_value) - float(reference_value)
            )
        rows.append(row)
    return rows


def _diagnostic_metric_keys(records: list[dict[str, Any]]) -> list[str]:
    preferred = {
        "loss",
        "loss_video",
        "loss_action",
        "loss_action_full_contract_abs_error",
    }
    for row in records:
        preferred.update(
            key
            for key in row
            if key.startswith("action_raw_mse_")
            or key.startswith("action_weighted_fm_")
        )
    result = []
    for key in sorted(preferred):
        if any(
            isinstance(row.get(key), (int, float))
            and math.isfinite(float(row[key]))
            for row in records
        ):
            result.append(key)
    return result


def _aggregate_metric(
    rows: list[dict[str, Any]],
    metric: str,
    *,
    mode: str,
    required_histories: tuple[int, ...] | None = None,
) -> float | None:
    valid = [
        row
        for row in rows
        if isinstance(row.get(metric), (int, float))
        and math.isfinite(float(row[metric]))
    ]
    if not valid:
        return None
    if mode == "sample_weighted":
        return float(np.mean([float(row[metric]) for row in valid]))

    grouped: dict[int, list[float]] = defaultdict(list)
    population: dict[int, int] = {}
    for row in valid:
        history = int(row["history_blocks"])
        grouped[history].append(float(row[metric]))
        if "history_population_count" in row:
            count = int(row["history_population_count"])
            previous = population.setdefault(history, count)
            if previous != count:
                raise ValueError(f"inconsistent population count for H={history}")
    for history, values in grouped.items():
        population.setdefault(history, len(values))
    expected = tuple(sorted(grouped)) if required_histories is None else required_histories
    if any(history not in grouped for history in expected):
        return None
    history_means = {history: float(np.mean(grouped[history])) for history in expected}
    if mode == "macro_h":
        return float(np.mean(list(history_means.values())))
    if mode == "history_distribution_weighted":
        denominator = sum(population[history] for history in expected)
        return float(
            sum(history_means[history] * population[history] for history in expected)
            / denominator
        )
    raise ValueError(f"unknown aggregation mode: {mode}")


def episode_cluster_bootstrap_ci(
    records: list[dict[str, Any]],
    *,
    metric: str,
    mode: str,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap complete episode clusters after sample/checkpoint pairing."""

    point = _aggregate_metric(records, metric, mode=mode)
    if point is None:
        return {
            "mean": None,
            "ci95_low": None,
            "ci95_high": None,
            "num_clusters": 0,
            "bootstrap_iterations": 0,
            "degenerate": True,
        }
    clusters: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        episode_id = int(row.get("episode_id", row["dataset_index"]))
        clusters[episode_id].append(row)
    cluster_ids = sorted(clusters)
    required_histories = tuple(
        sorted(
            {
                int(row["history_blocks"])
                for row in records
                if isinstance(row.get(metric), (int, float))
                and math.isfinite(float(row[metric]))
            }
        )
    )
    if iterations <= 0 or len(cluster_ids) <= 1:
        return {
            "mean": point,
            "ci95_low": point,
            "ci95_high": point,
            "num_clusters": len(cluster_ids),
            "bootstrap_iterations": 0,
            "degenerate": True,
        }

    rng = np.random.default_rng(_stable_seed(seed, metric, mode, "episode-bootstrap"))
    estimates: list[float] = []
    for _ in range(int(iterations)):
        sampled_ids = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        sampled_rows = [row for episode in sampled_ids for row in clusters[int(episode)]]
        estimate = _aggregate_metric(
            sampled_rows,
            metric,
            mode=mode,
            required_histories=required_histories,
        )
        if estimate is not None:
            estimates.append(estimate)
    if not estimates:
        return {
            "mean": point,
            "ci95_low": None,
            "ci95_high": None,
            "num_clusters": len(cluster_ids),
            "bootstrap_iterations": 0,
            "degenerate": True,
        }
    array = np.asarray(estimates, dtype=np.float64)
    return {
        "mean": point,
        "ci95_low": float(np.quantile(array, 0.025)),
        "ci95_high": float(np.quantile(array, 0.975)),
        "num_clusters": len(cluster_ids),
        "bootstrap_iterations": len(estimates),
        "degenerate": bool(np.all(array == array[0])),
    }


def summarize_diagnostic_records(
    records: list[dict[str, Any]],
    *,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if not records:
        return {
            "metric_keys": [],
            "sample_weighted": {},
            "macro_h": {},
            "history_distribution_weighted": {},
            "by_history": {},
        }
    metrics = _diagnostic_metric_keys(records)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[int(row["history_blocks"])].append(row)

    result = {
        "metric_keys": metrics,
        "sample_weighted": {},
        "macro_h": {},
        "history_distribution_weighted": {},
        "by_history": {},
    }
    for mode in ("sample_weighted", "macro_h", "history_distribution_weighted"):
        result[mode] = {
            metric: episode_cluster_bootstrap_ci(
                records,
                metric=metric,
                mode=mode,
                iterations=bootstrap_iterations,
                seed=bootstrap_seed,
            )
            for metric in metrics
        }
    for history, rows in sorted(grouped.items()):
        result["by_history"][str(history)] = {
            metric: episode_cluster_bootstrap_ci(
                rows,
                metric=metric,
                mode="sample_weighted",
                iterations=bootstrap_iterations,
                seed=_stable_seed(bootstrap_seed, history),
            )
            for metric in metrics
        }
    return result


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[int(record["history_blocks"])].append(record)

    def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            key: _summary([float(row[key]) for row in rows])
            for key in ("loss", "loss_video", "loss_action")
        }

    return {
        "overall": metrics(records),
        "by_history": {
            str(history): metrics(grouped[history]) for history in sorted(grouped)
        },
    }


def summarize_paired_variant_delta(
    candidate: list[dict[str, Any]],
    reference: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize candidate-reference loss deltas for identical sample/noise pairs."""

    paired = _paired_rows(
        candidate,
        reference,
        metric_keys=["loss", "loss_video", "loss_action"],
    )
    grouped: dict[int, list[dict[str, float]]] = defaultdict(list)
    all_rows: list[dict[str, float]] = []
    for row in paired:
        delta = {
            "loss": float(row["loss"]),
            "loss_video": float(row["loss_video"]),
            "loss_action": float(row["loss_action"]),
        }
        all_rows.append(delta)
        grouped[int(row["history_blocks"])].append(delta)

    def means(rows: list[dict[str, float]]) -> dict[str, float | int]:
        return {
            "count": len(rows),
            **{
                f"{key}_delta_mean": float(np.mean([row[key] for row in rows]))
                for key in ("loss", "loss_video", "loss_action")
            },
        }

    return {
        "overall": means(all_rows),
        "by_history": {
            str(history): means(rows) for history, rows in sorted(grouped.items())
        },
    }


def local_rope_history_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Keep the complete history while restoring FastWAM-local RoPE origins.

    The causal order and every historical observation/action tensor remain
    unchanged. Only the episode-absolute position fields are reset, so the
    current real frame/action chunk has the same RoPE coordinates as native
    FastWAM. Comparing this variant with native measures prefix-content and
    shared-softmax effects without cross-modal absolute-position drift.
    """

    required = (
        "history_block_positions",
        "current_block_position",
        "episode_step",
    )
    missing = [key for key in required if key not in sample]
    if missing:
        raise KeyError(f"missing temporal position fields: {missing}")
    result = dict(sample)
    for key in required:
        result[key] = torch.zeros_like(sample[key])
    return result


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _checkpoint_label(path: Path, index: int) -> str:
    if path.parent.name == "weights":
        return f"{path.parent.parent.parent.name}:{path.stem}"
    return f"checkpoint_{index}:{path.stem}"


def _evaluate_checkpoint(
    cfg: DictConfig,
    *,
    checkpoint: Path,
    label: str,
    dataset,
    selected: list[dict[str, int]],
    dtype: torch.dtype,
    precision: str,
    device: str,
    noise_seed: int,
    include_native: bool,
    include_legacy_position_variants: bool,
    fixed_u_values: list[float] | None,
    noise_repeats: int,
    history_variants: tuple[str, ...],
    shuffled_donors: dict[int, dict[str, int] | None],
    executed_action_steps: int,
    continuous_action_dims: int,
    gripper_action_index: int,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    logger.info("Loading checkpoint %s from %s", label, checkpoint)
    model = instantiate(cfg.model, model_dtype=dtype, device=device)
    payload = model.load_checkpoint(str(checkpoint), optimizer=None)
    checkpoint_step = payload.get("step") if isinstance(payload, dict) else None
    del payload
    model.eval()
    model.requires_grad_(False)

    variant_records: dict[str, list[dict[str, Any]]] = {
        variant: [] for variant in history_variants
    }
    records = variant_records["correct"]
    variant_skips: list[dict[str, Any]] = []
    local_rope_history_records: list[dict[str, Any]] = []
    absolute_no_history_records: list[dict[str, Any]] = []
    native_records: list[dict[str, Any]] = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(torch.device(device))
    started = time.perf_counter()

    for ordinal, selection in enumerate(selected):
        sample = default_collate([dataset[selection["dataset_index"]]])
        actual_history = int(sample["history_valid_blocks"].sum().item())
        if actual_history != int(selection["history_blocks"]):
            raise AssertionError(
                f"history mismatch dataset_index={selection['dataset_index']} "
                f"expected={selection['history_blocks']} actual={actual_history}"
            )
        samples_by_variant: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
            "correct": (sample, {})
        }
        if "masked" in history_variants:
            if model.history_training_mode != "incremental_full_bptt":
                raise ValueError(
                    "masked-history diagnostics require "
                    "history_training_mode=incremental_full_bptt"
                )
            samples_by_variant["masked"] = (masked_history_sample(sample), {})
        if "shuffled" in history_variants:
            history = int(selection["history_blocks"])
            donor_metadata = shuffled_donors.get(int(selection["dataset_index"]))
            if history == 0:
                samples_by_variant["shuffled"] = (
                    dict(sample),
                    {"donor_dataset_index": None, "donor_episode_id": None},
                )
            elif donor_metadata is None:
                variant_skips.append(
                    {
                        **selection,
                        "history_variant": "shuffled",
                        "reason": "no_exact_h_cross_episode_donor",
                    }
                )
            else:
                donor_sample = default_collate(
                    [dataset[int(donor_metadata["dataset_index"])]]
                )
                samples_by_variant["shuffled"] = (
                    shuffled_history_sample(
                        sample,
                        donor_sample,
                        history_blocks=history,
                        recipient_episode_id=int(selection["episode_id"]),
                        donor_episode_id=int(donor_metadata["episode_id"]),
                    ),
                    {
                        "donor_dataset_index": int(donor_metadata["dataset_index"]),
                        "donor_episode_id": int(donor_metadata["episode_id"]),
                    },
                )

        if fixed_u_values is None:
            draw_specs = [
                {
                    "u_index": 0,
                    "fixed_u": None,
                    "noise_replica": 0,
                    "noise_seed": int(noise_seed + ordinal),
                    "video_noise_seed": None,
                    "action_noise_seed": None,
                    "draw_id": f"legacy-{selection['dataset_index']}-{noise_seed + ordinal}",
                }
            ]
        else:
            draw_specs = []
            for u_index, fixed_u in enumerate(fixed_u_values):
                for noise_replica in range(noise_repeats):
                    draw_seed = _stable_seed(
                        noise_seed,
                        int(selection["dataset_index"]),
                        u_index,
                        noise_replica,
                        "flow-draw",
                    )
                    draw_specs.append(
                        {
                            "u_index": int(u_index),
                            "fixed_u": float(fixed_u),
                            "noise_replica": int(noise_replica),
                            "noise_seed": int(draw_seed),
                            "video_noise_seed": _stable_seed(draw_seed, "video"),
                            "action_noise_seed": _stable_seed(draw_seed, "action"),
                            "future_video_noise_seed": _stable_seed(
                                draw_seed, "future_video"
                            ),
                            # Deterministic 50/50 split of the LingBot future-video
                            # condition across the two noise replicas: replica 0 is
                            # noised, replica 1 is clean, so every fixed-u value
                            # covers both branches.
                            "future_video_condition": (
                                "noised" if noise_replica % 2 == 0 else "clean"
                            ),
                            "draw_id": (
                                f"sample-{selection['dataset_index']}-u-{u_index}-"
                                f"noise-{noise_replica}"
                            ),
                        }
                    )

        def evaluate(
            loss_fn,
            active_sample: dict[str, Any],
            draw: dict[str, Any],
            *,
            future_video_condition_override: str | None = "__from_draw__",
        ) -> tuple[float, dict[str, float], float, dict[str, Any]]:
            if draw["video_noise_seed"] is None:
                # Preserve the legacy one-draw contract exactly. Enhanced fixed
                # draws use local generators and do not depend on global RNG.
                torch.manual_seed(int(draw["noise_seed"]))
                torch.cuda.manual_seed_all(int(draw["noise_seed"]))
            if future_video_condition_override == "__from_draw__":
                future_video_condition = draw.get("future_video_condition")
            else:
                future_video_condition = future_video_condition_override
            before = time.perf_counter()
            with fixed_flow_draw(
                model,
                fixed_u=draw["fixed_u"],
                video_noise_seed=draw["video_noise_seed"],
                action_noise_seed=draw["action_noise_seed"],
                future_video_noise_seed=draw.get("future_video_noise_seed"),
                future_video_condition=future_video_condition,
            ) as captured:
                with torch.inference_mode(), torch.autocast(
                    device_type="cuda", dtype=dtype, enabled=precision != "no"
                ):
                    loss, raw_metrics = loss_fn(model, active_sample)
            torch.cuda.synchronize(torch.device(device))
            metrics = {
                str(key): float(value) for key, value in raw_metrics.items()
            }
            scheduler_weight = model.train_action_scheduler.training_weight(
                captured["timestep_action"]
            )
            action_diagnostics = compute_action_diagnostics(
                pred_action=captured["pred_action"],
                target_action=captured["target_action"],
                action_is_pad=active_sample.get("action_is_pad"),
                scheduler_weight=scheduler_weight,
                loss_lambda_action=float(model.loss_lambda_action),
                executed_action_steps=executed_action_steps,
                continuous_action_dims=continuous_action_dims,
                gripper_action_index=gripper_action_index,
            )
            timestep_action = float(captured["timestep_action"].float().mean().item())
            timestep_video = float(captured["timestep_video"].float().mean().item())
            diagnostic = {
                "fixed_u": draw["fixed_u"],
                "timestep_action": timestep_action,
                "timestep_video": timestep_video,
                "sigma_action": timestep_action
                / float(model.train_action_scheduler.num_train_timesteps),
                "sigma_video": timestep_video
                / float(model.train_video_scheduler.num_train_timesteps),
                "action_scheduler_weight": float(
                    scheduler_weight.float().mean().item()
                ),
                "action": action_diagnostics,
                **action_diagnostics["flat"],
            }
            return (
                float(loss.detach().float().item()),
                metrics,
                time.perf_counter() - before,
                diagnostic,
            )

        for draw in draw_specs:
            for variant in history_variants:
                if variant not in samples_by_variant:
                    continue
                active_sample, variant_metadata = samples_by_variant[variant]
                variant_loss, variant_metrics, elapsed_s, diagnostic = evaluate(
                    lambda active_model, selected_sample: active_model.training_loss(
                        selected_sample
                    ),
                    active_sample,
                    draw,
                )
                action_key = f"loss_action_d{model.mot.num_layers}"
                video_key = f"loss_video_d{model.mot.num_layers}"
                record = {
                    **selection,
                    **variant_metadata,
                    "history_variant": variant,
                    "draw_id": str(draw["draw_id"]),
                    "u_index": int(draw["u_index"]),
                    "fixed_u": draw["fixed_u"],
                    "noise_replica": int(draw["noise_replica"]),
                    "noise_seed": int(draw["noise_seed"]),
                    "video_noise_seed": draw["video_noise_seed"],
                    "action_noise_seed": draw["action_noise_seed"],
                    "future_video_condition": draw.get("future_video_condition"),
                    "future_video_noise_seed": draw.get("future_video_noise_seed"),
                    "loss": variant_loss,
                    "loss_video": float(variant_metrics[video_key]),
                    "loss_action": float(variant_metrics[action_key]),
                    "elapsed_s": float(elapsed_s),
                    "flow_diagnostics": diagnostic,
                    **diagnostic["action"]["flat"],
                }
                full_metric = (
                    f"action_weighted_fm_full{diagnostic['action']['action_horizon']}_"
                    f"all{diagnostic['action']['action_dim']}"
                )
                full_value = record.get(full_metric)
                record["loss_action_full_contract_abs_error"] = (
                    None
                    if full_value is None
                    else abs(float(full_value) - float(record["loss_action"]))
                )
                contract_error = record["loss_action_full_contract_abs_error"]
                if contract_error is not None and contract_error > 1.0e-5:
                    raise RuntimeError(
                        "action diagnostic full-horizon metric disagrees with the "
                        f"model's logged loss_action by {contract_error:.3e}"
                    )
                variant_records[variant].append(record)

            if include_legacy_position_variants:
                if model.history_training_mode != "incremental_detached_prefix":
                    raise ValueError(
                        "legacy position variants require incremental_detached_prefix; "
                        "they deliberately mutate episode metadata and are only for "
                        "auditing the retired global-RoPE implementation"
                    )
                absolute_no_history_sample = dict(sample)
                absolute_no_history_sample["history_valid_blocks"] = torch.zeros_like(
                    sample["history_valid_blocks"]
                )
                for legacy_sample, destination in (
                    (local_rope_history_sample(sample), local_rope_history_records),
                    (absolute_no_history_sample, absolute_no_history_records),
                ):
                    legacy_loss, legacy_metrics, legacy_elapsed_s, diagnostic = evaluate(
                        lambda active_model, selected_sample: active_model.training_loss(
                            selected_sample
                        ),
                        legacy_sample,
                        draw,
                    )
                    destination.append(
                        {
                            **selection,
                            "noise_seed": int(draw["noise_seed"]),
                            "u_index": int(draw["u_index"]),
                            "noise_replica": int(draw["noise_replica"]),
                            "loss": legacy_loss,
                            "loss_video": float(
                                legacy_metrics[f"loss_video_d{model.mot.num_layers}"]
                            ),
                            "loss_action": float(
                                legacy_metrics[f"loss_action_d{model.mot.num_layers}"]
                            ),
                            "elapsed_s": float(legacy_elapsed_s),
                            **diagnostic["action"]["flat"],
                        }
                    )

            if include_native:
                # The native FastWAM baseline does not consume the LingBot
                # future-video condition (its sample has no history_video and it
                # never calls sample_lingbot_future_video_condition), so it draws
                # only the two video/action noises regardless of the draw's
                # future_video_condition flag.  Pass None so the noise-count check
                # expects two draws instead of three.
                native_loss, native_metrics, native_elapsed_s, diagnostic = evaluate(
                    FastWAM.training_loss,
                    sample,
                    draw,
                    future_video_condition_override=None,
                )
                native_records.append(
                    {
                        **selection,
                        "history_variant": "native",
                        "draw_id": str(draw["draw_id"]),
                        "u_index": int(draw["u_index"]),
                        "fixed_u": draw["fixed_u"],
                        "noise_replica": int(draw["noise_replica"]),
                        "noise_seed": int(draw["noise_seed"]),
                        "video_noise_seed": draw["video_noise_seed"],
                        "action_noise_seed": draw["action_noise_seed"],
                        "future_video_condition": None,
                        "future_video_noise_seed": draw.get(
                            "future_video_noise_seed"
                        ),
                        "loss": native_loss,
                        "loss_video": float(native_metrics["loss_video"]),
                        "loss_action": float(native_metrics["loss_action"]),
                        "elapsed_s": float(native_elapsed_s),
                        "flow_diagnostics": diagnostic,
                        **diagnostic["action"]["flat"],
                    }
                )

            correct_record = variant_records["correct"][-1]
            logger.info(
                "%s H=%d replica=%d u=%s noise_replica=%d action=%.6f video=%.6f elapsed=%.2fs",
                label,
                selection["history_blocks"],
                selection["replica"],
                str(draw["fixed_u"]),
                draw["noise_replica"],
                correct_record["loss_action"],
                correct_record["loss_video"],
                correct_record["elapsed_s"],
            )

    result: dict[str, Any] = {
        "label": label,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "checkpoint_step": checkpoint_step,
        "causal_mode": str(model.causal_mode),
        "history_training_mode": str(model.history_training_mode),
        "elapsed_s": float(time.perf_counter() - started),
        "peak_gpu_allocated_gib": float(
            torch.cuda.max_memory_allocated(torch.device(device)) / 2**30
        ),
        "records": records,
        "summary": summarize_records(records),
        "diagnostic_summary": summarize_diagnostic_records(
            records,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_seed=bootstrap_seed,
        ),
        "variant_records": variant_records,
        "variant_summaries": {
            variant: summarize_diagnostic_records(
                rows,
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_seed=_stable_seed(bootstrap_seed, variant),
            )
            for variant, rows in variant_records.items()
        },
        "variant_skips": variant_skips,
    }
    correct_by_key = _index_unique_records(records, label="correct history")
    result["variant_deltas_vs_correct"] = {}
    for variant, rows in variant_records.items():
        if variant == "correct" or not rows:
            continue
        variant_by_key = _index_unique_records(rows, label=f"{variant} history")
        extra = set(variant_by_key).difference(correct_by_key)
        if extra:
            raise ValueError(
                f"{variant} history contains draws absent from correct history: {sorted(extra)[:5]}"
            )
        reference_subset = [correct_by_key[key] for key in sorted(variant_by_key)]
        result["variant_deltas_vs_correct"][variant] = {
            "excluded_correct_draws": len(correct_by_key) - len(reference_subset),
            **_paired_diagnostic_summary(
                rows,
                reference_subset,
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_seed=_stable_seed(bootstrap_seed, variant, "minus-correct"),
            ),
        }
    if include_legacy_position_variants:
        result.update(
            {
                "local_rope_history_records": local_rope_history_records,
                "local_rope_history_summary": summarize_records(
                    local_rope_history_records
                ),
                "absolute_no_history_records": absolute_no_history_records,
                "absolute_no_history_summary": summarize_records(
                    absolute_no_history_records
                ),
                "full_minus_absolute_no_history": summarize_paired_variant_delta(
                    records, absolute_no_history_records
                ),
                "full_minus_local_rope_history": summarize_paired_variant_delta(
                    records, local_rope_history_records
                ),
            }
        )
    if native_records:
        result["native_records"] = native_records
        result["native_summary"] = summarize_records(native_records)
        result["native_diagnostic_summary"] = summarize_diagnostic_records(
            native_records,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_seed=_stable_seed(bootstrap_seed, "native"),
        )
        if include_legacy_position_variants:
            result["absolute_no_history_minus_native"] = summarize_paired_variant_delta(
                absolute_no_history_records, native_records
            )
            result["local_rope_history_minus_native"] = summarize_paired_variant_delta(
                local_rope_history_records, native_records
            )

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _paired_deltas(checkpoints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(checkpoints) < 2:
        return []
    reference = checkpoints[0]
    results = []
    for candidate in checkpoints[1:]:
        paired = _paired_rows(
            candidate["records"],
            reference["records"],
            metric_keys=["loss_action", "loss_video"],
        )
        rows = [
            {
                **row,
                "loss_action_delta": float(row["loss_action"]),
                "loss_video_delta": float(row["loss_video"]),
            }
            for row in paired
        ]
        by_history: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_history[int(row["history_blocks"])].append(row)
        results.append(
            {
                "reference": reference["label"],
                "candidate": candidate["label"],
                "action_delta_mean": float(
                    np.mean([row["loss_action_delta"] for row in rows])
                ),
                "video_delta_mean": float(
                    np.mean([row["loss_video_delta"] for row in rows])
                ),
                "by_history": {
                    str(history): {
                        "count": len(group),
                        "action_delta_mean": float(
                            np.mean([row["loss_action_delta"] for row in group])
                        ),
                        "video_delta_mean": float(
                            np.mean([row["loss_video_delta"] for row in group])
                        ),
                    }
                    for history, group in sorted(by_history.items())
                },
            }
        )
    return results


def _paired_diagnostic_summary(
    candidate: list[dict[str, Any]],
    reference: list[dict[str, Any]],
    *,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    metric_keys = sorted(
        set(_diagnostic_metric_keys(candidate)).intersection(
            _diagnostic_metric_keys(reference)
        )
    )
    rows = _paired_rows(candidate, reference, metric_keys=metric_keys)
    return {
        "draw_count": len(rows),
        "summary": summarize_diagnostic_records(
            rows,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_seed=bootstrap_seed,
        ),
    }


def _checkpoint_decompositions(
    checkpoints: list[dict[str, Any]],
    *,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    if not checkpoints or "native_records" not in checkpoints[0]:
        return []
    release = checkpoints[0]
    results = []
    for candidate in checkpoints:
        if "native_records" not in candidate:
            continue
        results.append(
            {
                "release": release["label"],
                "candidate": candidate["label"],
                "incremental_minus_candidate_native": _paired_diagnostic_summary(
                    candidate["records"],
                    candidate["native_records"],
                    bootstrap_iterations=bootstrap_iterations,
                    bootstrap_seed=_stable_seed(
                        bootstrap_seed, candidate["label"], "incremental-minus-native"
                    ),
                ),
                "candidate_native_minus_release_native": _paired_diagnostic_summary(
                    candidate["native_records"],
                    release["native_records"],
                    bootstrap_iterations=bootstrap_iterations,
                    bootstrap_seed=_stable_seed(
                        bootstrap_seed, candidate["label"], "native-drift"
                    ),
                ),
                "incremental_minus_release_native": _paired_diagnostic_summary(
                    candidate["records"],
                    release["native_records"],
                    bootstrap_iterations=bootstrap_iterations,
                    bootstrap_seed=_stable_seed(
                        bootstrap_seed, candidate["label"], "total-gap"
                    ),
                ),
            }
        )
    return results


def run_audit(cfg: DictConfig) -> dict[str, Any]:
    audit = cfg.get("stratified")
    if audit is None:
        raise ValueError("missing +stratified configuration")
    checkpoints = [Path(str(value)).expanduser().resolve() for value in audit.checkpoints]
    if not checkpoints:
        raise ValueError("stratified.checkpoints must not be empty")
    missing = [str(path) for path in checkpoints if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing checkpoints: {missing}")

    output_dir = Path(str(cfg.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    misc.register_work_dir(str(output_dir))
    OmegaConf.save(cfg, output_dir / "config.yaml", resolve=True)

    precision = _normalize_mixed_precision(str(cfg.mixed_precision))
    dtype = _mixed_precision_to_model_dtype(precision)
    device = str(audit.get("device", "cuda:0"))
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("history-stratified 6B audit requires CUDA")

    dataset = instantiate(cfg.data.train)
    if not bool(getattr(dataset, "full_episode_history", False)):
        raise ValueError("stratified audit requires full_episode_history=true")
    history_lengths = [int(value) for value in audit.history_lengths]
    selected = select_history_samples(
        dataset,
        history_lengths=history_lengths,
        samples_per_history=int(audit.get("samples_per_history", 2)),
        seed=int(audit.get("selection_seed", cfg.seed)),
    )

    fixed_u_config = audit.get("fixed_u_values")
    fixed_u_values = (
        None if fixed_u_config is None else [float(value) for value in fixed_u_config]
    )
    if fixed_u_values is not None and not fixed_u_values:
        raise ValueError("stratified.fixed_u_values must not be empty when configured")
    if fixed_u_values is not None:
        for value in fixed_u_values:
            if not 0.0 <= value < 1.0:
                raise ValueError(f"fixed flow u must lie in [0,1), got {value}")
    noise_repeats = int(audit.get("noise_repeats", 1))
    if noise_repeats <= 0:
        raise ValueError("stratified.noise_repeats must be positive")
    if noise_repeats != 1 and fixed_u_values is None:
        raise ValueError("multiple noise repeats require stratified.fixed_u_values")

    history_variants = tuple(
        str(value) for value in audit.get("history_variants", ["correct"])
    )
    allowed_variants = {"correct", "masked", "shuffled"}
    invalid_variants = sorted(set(history_variants).difference(allowed_variants))
    if invalid_variants:
        raise ValueError(f"unsupported history variants: {invalid_variants}")
    if "correct" not in history_variants:
        raise ValueError("stratified.history_variants must include correct")
    if len(history_variants) != len(set(history_variants)):
        raise ValueError("stratified.history_variants contains duplicates")
    shuffled_donors = (
        build_shuffled_history_donors(
            dataset,
            selected,
            seed=int(audit.get("shuffle_seed", int(cfg.seed) + 3_000_000)),
        )
        if "shuffled" in history_variants
        else {}
    )
    enhanced = fixed_u_values is not None or history_variants != ("correct",)
    bootstrap_iterations = int(
        audit.get("bootstrap_iterations", 2000 if enhanced else 0)
    )
    if bootstrap_iterations < 0:
        raise ValueError("stratified.bootstrap_iterations must be non-negative")
    bootstrap_seed = int(audit.get("bootstrap_seed", int(cfg.seed) + 4_000_000))
    include_native_configured = audit.get("include_native")
    if enhanced and include_native_configured is not None and not bool(
        include_native_configured
    ):
        raise ValueError(
            "enhanced diagnostics require stratified.include_native=true so prefix "
            "penalty and parameter drift can be decomposed"
        )

    checkpoint_results = []
    for index, checkpoint in enumerate(checkpoints):
        include_native = (
            bool(include_native_configured)
            if include_native_configured is not None
            else bool(enhanced or index == 0)
        )
        checkpoint_results.append(
            _evaluate_checkpoint(
                cfg,
                checkpoint=checkpoint,
                label=_checkpoint_label(checkpoint, index),
                dataset=dataset,
                selected=selected,
                dtype=dtype,
                precision=precision,
                device=device,
                noise_seed=int(audit.get("noise_seed", int(cfg.seed) + 2_000_000)),
                include_native=include_native,
                include_legacy_position_variants=bool(
                    audit.get("include_legacy_position_variants", False)
                ),
                fixed_u_values=fixed_u_values,
                noise_repeats=noise_repeats,
                history_variants=history_variants,
                shuffled_donors=shuffled_donors,
                executed_action_steps=int(audit.get("executed_action_steps", 10)),
                continuous_action_dims=int(audit.get("continuous_action_dims", 6)),
                gripper_action_index=int(audit.get("gripper_action_index", 6)),
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_seed=_stable_seed(bootstrap_seed, index),
            )
        )

    result = {
        "kind": "paired_history_stratified_loss_audit",
        "source_identity": _git_source_identity(Path(__file__).parents[1]),
        "history_lengths": history_lengths,
        "samples_per_history_requested": int(audit.get("samples_per_history", 2)),
        "selected_samples": selected,
        "fixed_u_values": fixed_u_values,
        "noise_repeats": noise_repeats,
        "history_variants": list(history_variants),
        "bootstrap_iterations": bootstrap_iterations,
        "bootstrap_seed": bootstrap_seed,
        "shuffled_history_donors": {
            str(key): value for key, value in shuffled_donors.items()
        },
        "precision": precision,
        "device": device,
        "checkpoints": checkpoint_results,
        "paired_deltas_from_first": _paired_deltas(checkpoint_results),
        "checkpoint_decompositions": _checkpoint_decompositions(
            checkpoint_results,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_seed=bootstrap_seed,
        ),
    }
    output_path = output_dir / str(audit.get("output_name", "history_stratified_loss.json"))
    _write_json(output_path, result)
    logger.info("History-stratified loss audit complete: %s", output_path)
    return result


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging(log_level=logging.INFO, is_main_process=True)
    run_audit(cfg)


if __name__ == "__main__":
    main()
