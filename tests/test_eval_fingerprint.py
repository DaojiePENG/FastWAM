from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

import leapbot_va.eval_contract as eval_contract
from leapbot_va.eval_contract import (
    KV_RETENTION_SEMANTICS,
    _git_source_identity,
    build_result_contract,
    build_runtime_contract,
)
from leapbot_va.eval_fingerprint import (
    FINGERPRINT_SCHEMA_VERSION,
    atomic_write_json,
    build_evaluation_fingerprint,
    build_verified_evaluation_fingerprint,
    canonical_json_sha256,
    normalize_evaluation_fingerprint,
    result_matches_fingerprint,
    sha256_file,
)


def _runtime_contract(*, memory_enabled: bool = True):
    return {
        "config": {
            "name": "sim_leapbot_libero",
            "task_choice": "libero_leapbot_2cam224",
        },
        "source": {"revision": "a" * 40, "dirty": False},
        "seed": 42,
        "inference": {
            "num_inference_steps": 20,
            "replan_steps": 10,
            "action_horizon": 32,
            "num_steps_wait": 30,
            "sigma_shift": None,
            "text_cfg_scale": 1.0,
            "negative_prompt": "",
            "rand_device": "cpu",
            "tiled": False,
        },
        "action_execution": {
            "binarize_gripper": True,
            "use_action_ensembler": False,
        },
        "normalization": {"dataset_stats_sha256": "b" * 64},
        "precision_and_adapters": {
            "mixed_precision": "bf16",
            "merge_video_lora": False,
        },
        "memory": {
            "enabled": memory_enabled,
            "causal_mode": "action_aggregator" if memory_enabled else None,
            "exit_depth": 30 if memory_enabled else 0,
            "episode_capacity": 70 if memory_enabled else 0,
            "retained_history_blocks": None if memory_enabled else 0,
            "retention_semantics": KV_RETENTION_SEMANTICS,
            "effective_kv_retention_cap": 70 if memory_enabled else 0,
            "effective_history_cap": 70 if memory_enabled else 0,
        },
        "input": {
            "height": 224,
            "width": 448,
            "concat_multi_camera": "horizontal",
            "processor": {"_target_": "example.Processor", "num_output_cameras": 2},
        },
        "model": {"_target_": "example.Model", "causal_mode": "action_aggregator"},
    }


def _result_contract(**overrides):
    values = {
        "suite": "libero_10",
        "task": {
            "id": 0,
            "problem_folder": "libero_10",
            "init_states_file": "task0.init",
        },
        "trials": 2,
        "initial_states_sha256": "c" * 64,
    }
    values.update(overrides)
    return values


def _fingerprint(
    *,
    runtime_contract=None,
    result_contract=None,
    checkpoint_sha256=None,
):
    return build_evaluation_fingerprint(
        checkpoint_sha256=(
            checkpoint_sha256
            or hashlib.sha256(b"checkpoint-a").hexdigest()
        ),
        runtime_contract=runtime_contract or _runtime_contract(),
        result_contract=result_contract or _result_contract(),
    )


def _complete_result(fingerprint, *, memory_enabled: bool = True):
    trials = fingerprint["result_contract"]["trials"]
    metrics = []
    for _ in range(trials):
        timing = {
            "total_inference_s": 0.123,
            "input_preprocess_s": 0.003,
            "model_inference_s": 0.117,
            "action_postprocess_s": 0.003,
            "latency_residual_s": 0.0,
        }
        replan = {"timing": timing}
        if memory_enabled:
            timing.update(
                {
                    "conditioning_s": 0.002,
                    "observation_prefill_s": 0.01,
                    "future_video_setup_s": 0.001,
                    "future_video_denoise_s": 0.02,
                    "future_video_cache_s": 0.004,
                    "action_setup_s": 0.003,
                    "action_denoise_s": 0.075,
                    "causal_model_residual_s": 0.002,
                    "causal_model_s": 0.117,
                }
            )
            replan["commit"] = {"commit_s": 0.013}
        metrics.append({"enabled": memory_enabled, "replans": [replan]})
    return {
        "total_episodes": trials,
        "successes": trials - 1,
        "success_episodes": list(range(trials - 1)),
        "failure_episodes": [trials - 1],
        "evaluation_fingerprint": fingerprint,
        "completion_steps": [100] * trials,
        "memory_metrics": metrics,
    }


def test_sha256_file_streams_exact_file_identity(tmp_path):
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"checkpoint-a")
    assert sha256_file(checkpoint) == hashlib.sha256(b"checkpoint-a").hexdigest()
    checkpoint.write_bytes(b"checkpoint-b")
    assert sha256_file(checkpoint) == hashlib.sha256(b"checkpoint-b").hexdigest()


def test_git_source_identity_distinguishes_dirty_and_untracked_content(tmp_path):
    repo = tmp_path / "source"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Test"], check=True
    )
    tracked = repo / "tracked.txt"
    tracked.write_text("base")
    subprocess.run(["git", "-C", str(repo), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "base"], check=True)

    clean = _git_source_identity(repo)
    assert clean["dirty"] is False
    tracked.write_text("change-a")
    dirty_a = _git_source_identity(repo)
    tracked.write_text("change-b")
    dirty_b = _git_source_identity(repo)
    assert dirty_a["dirty"] is True
    assert dirty_a["worktree_sha256"] != dirty_b["worktree_sha256"]

    tracked.write_text("base")
    untracked = repo / "untracked.txt"
    untracked.write_text("one")
    untracked_a = _git_source_identity(repo)
    untracked.write_text("two")
    untracked_b = _git_source_identity(repo)
    assert untracked_a["worktree_sha256"] != untracked_b["worktree_sha256"]


def test_canonical_hash_is_stable_across_mapping_order_and_tuple_list():
    left = {"z": [1, 2], "a": {"y": 0.0, "x": "中文"}}
    right = {"a": {"x": "中文", "y": -0.0}, "z": (1, 2)}
    assert canonical_json_sha256(left) == canonical_json_sha256(right)
    left_fp = _fingerprint(runtime_contract={"outer": left})
    right_fp = _fingerprint(runtime_contract={"outer": right})
    assert left_fp == right_fp


@pytest.mark.parametrize(
    ("section", "field", "different"),
    [
        ("config", "name", "sim_other"),
        ("config", "task_choice", "other_task_config"),
        ("source", "revision", "d" * 40),
        ("source", "dirty", True),
        ("inference", "num_inference_steps", 10),
        ("inference", "replan_steps", 5),
        ("inference", "action_horizon", 16),
        ("inference", "num_steps_wait", 5),
        ("inference", "sigma_shift", 2.0),
        ("inference", "text_cfg_scale", 1.5),
        ("inference", "negative_prompt", "bad"),
        ("inference", "rand_device", "cuda"),
        ("inference", "tiled", True),
        ("action_execution", "binarize_gripper", False),
        ("action_execution", "use_action_ensembler", True),
        ("normalization", "dataset_stats_sha256", "e" * 64),
        ("precision_and_adapters", "mixed_precision", "fp16"),
        ("precision_and_adapters", "merge_video_lora", True),
        ("memory", "causal_mode", "vision_causal"),
        ("memory", "exit_depth", 24),
        ("memory", "episode_capacity", 80),
        ("memory", "retained_history_blocks", 32),
        ("memory", "retention_semantics", "strict_information_window"),
        ("memory", "effective_kv_retention_cap", 32),
        ("input", "width", 224),
        ("input", "concat_multi_camera", "vertical"),
    ],
)
def test_any_behavior_contract_change_rejects_reuse(section, field, different):
    expected = _fingerprint()
    changed_runtime = copy.deepcopy(expected["runtime_contract"])
    changed_runtime[section][field] = different
    stale = _fingerprint(runtime_contract=changed_runtime)
    assert not result_matches_fingerprint(_complete_result(stale), expected)


@pytest.mark.parametrize(
    ("retained_history_blocks", "effective_kv_retention_cap"),
    [(None, 70), (8, 8)],
)
def test_runtime_contract_audits_recursive_physical_kv_retention(
    tmp_path,
    monkeypatch,
    retained_history_blocks,
    effective_kv_retention_cap,
):
    dataset_stats = tmp_path / "dataset_stats.json"
    dataset_stats.write_text("{}")
    monkeypatch.setattr(
        eval_contract,
        "_git_source_identity",
        lambda source_root: {"revision": "a" * 40, "dirty": False},
    )
    monkeypatch.setattr(
        eval_contract,
        "_dependency_identity",
        lambda module_name, distribution_name: {
            "module": module_name,
            "distribution": distribution_name,
        },
    )
    monkeypatch.setattr(
        eval_contract,
        "build_wan_conditioning_identity",
        lambda **kwargs: {
            "schema_version": 1,
            "identity_sha256": "d" * 64,
        },
    )
    cfg = OmegaConf.create(
        {
            "ckpt": str(tmp_path / "checkpoint.pt"),
            "seed": 42,
            "mixed_precision": "bf16",
            "eval_num_inference_steps": 20,
            "EVALUATION": {
                "memory": {
                    "enabled": True,
                    "causal_mode": "interleaved",
                    "exit_depth": 30,
                    "max_history_blocks": 70,
                    "retained_history_blocks": retained_history_blocks,
                },
                "replan_steps": 10,
                "action_horizon": 32,
            },
            "model": {
                "causal_mode": "interleaved",
                "model_id": "Wan-AI/Wan2.2-TI2V-5B",
                "tokenizer_model_id": "Wan-AI/Wan2.1-T2V-1.3B",
                "load_text_encoder": True,
            },
            "data": {
                "train": {
                    "num_frames": 33,
                    "video_size": [224, 448],
                    "action_video_freq_ratio": 1,
                    "processor": {"_target_": "example.Processor"},
                }
            },
        }
    )

    contract = build_runtime_contract(
        cfg,
        config_name="sim_leapbot_libero",
        hydra_choices={"task": "libero_leapbot_2cam224"},
        dataset_stats_path=dataset_stats,
        source_root=tmp_path,
    )

    memory = contract["memory"]
    assert memory["retention_semantics"] == KV_RETENTION_SEMANTICS
    assert memory["effective_kv_retention_cap"] == effective_kv_retention_cap
    assert memory["effective_history_cap"] == effective_kv_retention_cap
    assert contract["conditioning_assets"]["identity_sha256"] == "d" * 64


@pytest.mark.parametrize(
    "changed_result",
    [
        {"suite": "libero_goal"},
        {
            "task": {
                "id": 1,
                "problem_folder": "libero_10",
                "init_states_file": "task1.init",
            }
        },
        {"trials": 3},
        {"initial_states_sha256": "f" * 64},
    ],
)
def test_suite_task_trials_or_initial_states_change_rejects_reuse(changed_result):
    expected = _fingerprint()
    stale_contract = _result_contract(**changed_result)
    stale = _fingerprint(result_contract=stale_contract)
    assert not result_matches_fingerprint(_complete_result(stale), expected)


def test_evaluator_rehashes_checkpoint_and_rejects_launcher_mismatch(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"actual-checkpoint")
    expected = _fingerprint(
        checkpoint_sha256=hashlib.sha256(b"different-checkpoint").hexdigest()
    )
    with pytest.raises(ValueError, match="actual_checkpoint_sha256"):
        build_verified_evaluation_fingerprint(
            checkpoint_path=checkpoint,
            runtime_contract=_runtime_contract(),
            result_contract=_result_contract(),
            expected=expected,
        )


def test_evaluator_rehashes_checkpoint_and_accepts_exact_expected(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint-a")
    expected = _fingerprint()
    assert build_verified_evaluation_fingerprint(
        checkpoint_path=checkpoint,
        runtime_contract=_runtime_contract(),
        result_contract=_result_contract(),
        expected=expected,
    ) == expected


@pytest.mark.parametrize("old_schema", [1, 2])
def test_old_fingerprint_schemas_are_rejected(old_schema):
    old = {
        "schema_version": old_schema,
        "checkpoint_sha256": "a" * 64,
        "config_name": "sim_leapbot_libero",
    }
    with pytest.raises(ValueError, match="fields mismatch|unsupported"):
        normalize_evaluation_fingerprint(old)
    result = _complete_result(_fingerprint())
    result["evaluation_fingerprint"] = old
    assert not result_matches_fingerprint(result, _fingerprint())


def test_tampered_expanded_contract_hash_is_rejected():
    fingerprint = _fingerprint()
    fingerprint["runtime_contract"]["seed"] = 7
    with pytest.raises(ValueError, match="does not match"):
        normalize_evaluation_fingerprint(fingerprint)


def test_exact_complete_memory_result_matches():
    expected = _fingerprint()
    assert result_matches_fingerprint(_complete_result(expected), expected)


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_total",
        "short_completion",
        "short_metrics",
        "incomplete_episode_partition",
        "no_replans",
        "missing_total",
        "missing_observation",
        "missing_future_video",
        "missing_action",
        "missing_commit",
    ],
)
def test_incomplete_results_never_match(mutation):
    expected = _fingerprint()
    result = _complete_result(expected)
    if mutation == "wrong_total":
        result["total_episodes"] = 1
    elif mutation == "short_completion":
        result["completion_steps"].pop()
    elif mutation == "short_metrics":
        result["memory_metrics"].pop()
    elif mutation == "incomplete_episode_partition":
        result["failure_episodes"] = []
    elif mutation == "no_replans":
        result["memory_metrics"][0]["replans"] = []
    elif mutation == "missing_total":
        del result["memory_metrics"][0]["replans"][0]["timing"]["total_inference_s"]
    elif mutation == "missing_observation":
        del result["memory_metrics"][0]["replans"][0]["timing"]["observation_prefill_s"]
    elif mutation == "missing_future_video":
        del result["memory_metrics"][0]["replans"][0]["timing"]["future_video_denoise_s"]
    elif mutation == "missing_action":
        del result["memory_metrics"][0]["replans"][0]["timing"]["action_denoise_s"]
    elif mutation == "missing_commit":
        del result["memory_metrics"][0]["replans"][0]["commit"]
    assert not result_matches_fingerprint(result, expected)


def test_non_memory_result_only_requires_total_timing_segment():
    runtime = _runtime_contract(memory_enabled=False)
    expected = _fingerprint(runtime_contract=runtime)
    result = _complete_result(expected, memory_enabled=False)
    assert result_matches_fingerprint(result, expected)


def test_allow_unprofiled_still_requires_episode_arrays():
    expected = _fingerprint()
    result = _complete_result(expected)
    result["memory_metrics"] = [{}, {}]
    assert result_matches_fingerprint(result, expected, require_profiled=False)
    result["memory_metrics"].pop()
    assert not result_matches_fingerprint(result, expected, require_profiled=False)


def test_result_contract_hashes_actual_initial_state_file(tmp_path):
    initial_states = tmp_path / "task0.init"
    initial_states.write_bytes(b"states-a")
    cfg = OmegaConf.create(
        {"EVALUATION": {"task_suite_name": "libero_10", "task_id": 0, "num_trials": 2}}
    )
    task = SimpleNamespace(problem_folder="libero_10", init_states_file="task0.init")
    contract = build_result_contract(
        cfg, task=task, initial_states_path=initial_states
    )
    assert contract["initial_states_sha256"] == hashlib.sha256(b"states-a").hexdigest()
    initial_states.write_bytes(b"states-b")
    changed = build_result_contract(
        cfg, task=task, initial_states_path=initial_states
    )
    assert changed["initial_states_sha256"] != contract["initial_states_sha256"]


def test_atomic_json_write_replaces_destination_and_leaves_no_temp(tmp_path):
    output = tmp_path / "result.json"
    output.write_text('{"old": true}', encoding="utf-8")
    atomic_write_json(output, {"new": [1, 2, 3]})
    assert json.loads(output.read_text(encoding="utf-8")) == {"new": [1, 2, 3]}
    assert list(tmp_path.glob(".result.json.*.tmp")) == []


def test_schema_version_is_three():
    assert FINGERPRINT_SCHEMA_VERSION == 3
