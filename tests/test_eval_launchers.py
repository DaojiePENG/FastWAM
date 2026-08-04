from __future__ import annotations

import json
import os
import stat
import subprocess
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = (
    "evaluate_fastwam_baseline.sh",
    "evaluate_checkpoint.sh",
    "evaluate_causal_modes.sh",
)
EXPECTED_FINGERPRINT = {"schema_version": 3, "identity": "expected"}


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_eval_launchers_are_portable_and_disable_rollout_video(launcher):
    source = (REPO_ROOT / "scripts" / launcher).read_text()
    assert 'LIBERO_ROOT="${LIBERO_ROOT:-$(cd "$ROOT_DIR/.." && pwd)/LIBERO}"' in source
    assert "/home/sheng" not in source
    assert 'PYTHONPATH="$LIBERO_ROOT:$ROOT_DIR/experiments/libero"' in source
    assert source.count("EVALUATION.save_rollout_video=false") == 2


@pytest.mark.parametrize(
    "launcher",
    (
        "evaluate_checkpoint.sh",
        "evaluate_causal_modes.sh",
    ),
)
def test_memory_eval_rejects_non_runtime_isomorphic_history_vae_chunk(launcher):
    source = (REPO_ROOT / "scripts" / launcher).read_text()
    assert "--expected-history-vae-batch-chunk-size 1" in source


def _write_executable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _write_checkpoint(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"checkpoint:{path}".encode())


def _write_run_contract(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"run_contract_sha256={'a' * 64}\ncode_commit={'b' * 40}\n",
        encoding="utf-8",
    )


def _launcher_environment(tmp_path: Path, launcher: str) -> tuple[dict[str, str], Path]:
    root = tmp_path / "fake-root"
    eval_root = tmp_path / "eval"
    python_log = tmp_path / "python-calls.jsonl"
    evaluator_marker = tmp_path / "evaluator-called"
    evaluator_env_log = tmp_path / "evaluator-env.jsonl"
    gpu_marker = tmp_path / "gpu-probed"
    fake_bin = tmp_path / "bin"
    stats = root / "checkpoints/fastwam_release/dataset_stats.json"
    release = root / "checkpoints/fastwam_release/release.pt"
    stats.parent.mkdir(parents=True, exist_ok=True)
    stats.write_text("{}\n", encoding="utf-8")
    _write_checkpoint(release)

    _write_executable(
        root / ".venv/bin/python",
        f"""
        #!/usr/bin/env python3
        import json
        import os
        import sys
        from pathlib import Path

        argv = sys.argv[1:]
        with open(os.environ["FAKE_PYTHON_LOG"], "a", encoding="utf-8") as stream:
            stream.write(json.dumps(argv) + "\\n")

        if argv and argv[0].endswith("build_eval_fingerprint.py"):
            output = Path(argv[argv.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps({{ "schema_version": 3, "identity": "expected" }}),
                encoding="utf-8",
            )
            raise SystemExit(0)

        if argv[:2] == ["-m", "leapbot_va.eval_fingerprint"]:
            result = json.loads(Path(argv[3]).read_text(encoding="utf-8"))
            fingerprint = result.get("evaluation_fingerprint", {{}})
            expected_path = Path(argv[argv.index("--expected") + 1])
            expected = json.loads(expected_path.read_text(encoding="utf-8"))
            raise SystemExit(0 if fingerprint == expected else 1)

        if argv and argv[0].endswith("eval_libero_single.py"):
            Path(os.environ["FAKE_EVALUATOR_MARKER"]).touch()
            with open(os.environ["FAKE_EVALUATOR_ENV_LOG"], "a", encoding="utf-8") as stream:
                stream.write(json.dumps({{
                    "argv": argv,
                    "cuda_visible_devices_present": "CUDA_VISIBLE_DEVICES" in os.environ,
                    "mujoco_egl_device_id": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
                }}) + "\\n")
        raise SystemExit(0)
        """,
    )
    _write_executable(
        fake_bin / "nvidia-smi",
        """
        #!/usr/bin/env bash
        : >"$FAKE_GPU_MARKER"
        printf '0\n%.0s' {{1..8}}
        """,
    )

    env = os.environ.copy()
    env.update(
        {
            "ROOT_DIR": str(root),
            "EVAL_ROOT": str(eval_root),
            "LEAPBOT_DATASET_STATS": str(stats),
            "RELEASE_CHECKPOINT": str(release),
            "GPU_IDS_CSV": "3",
            "NUM_TRIALS": "1",
            "POLL_SECONDS": "1",
            "FINAL_STEP": "1",
            "FAKE_PYTHON_LOG": str(python_log),
            "FAKE_EVALUATOR_MARKER": str(evaluator_marker),
            "FAKE_EVALUATOR_ENV_LOG": str(evaluator_env_log),
            "FAKE_GPU_MARKER": str(gpu_marker),
            "CUDA_VISIBLE_DEVICES": "7",
            "PATH": f"{fake_bin}:{env['PATH']}",
        }
    )

    if launcher == "evaluate_checkpoint.sh":
        train_root = tmp_path / "single-train"
        _write_checkpoint(
            train_root / "action_aggregator/checkpoints/weights/step_000001.pt"
        )
        _write_run_contract(train_root / "action_aggregator/run_contract.txt")
        env.update(
            {
                "TRAIN_ROOT": str(train_root),
                "MODE": "action_aggregator",
            }
        )
    elif launcher == "evaluate_causal_modes.sh":
        train_root = tmp_path / "phase1-train"
        for mode in ("interleaved", "vision_causal", "action_aggregator"):
            _write_checkpoint(train_root / mode / "checkpoints/weights/step_000001.pt")
            _write_run_contract(train_root / mode / "run_contract.txt")
        env.update(
            {
                "TRAIN_ROOT": str(train_root),
                "REQUIRE_TRAINING_COMPLETE": "false",
                "INCLUDE_BASELINE": "true",
            }
        )
    return env, eval_root


def _launcher_modes(launcher: str) -> tuple[str, ...]:
    if launcher == "evaluate_fastwam_baseline.sh":
        return ("fastwam_release",)
    if launcher == "evaluate_checkpoint.sh":
        return ("action_aggregator",)
    return ("fastwam_release", "interleaved", "vision_causal", "action_aggregator")


def _stale_result_path(eval_root: Path, launcher: str) -> Path:
    mode = {
        "evaluate_fastwam_baseline.sh": "fastwam_release",
        "evaluate_checkpoint.sh": "action_aggregator",
        "evaluate_causal_modes.sh": "action_aggregator",
    }[launcher]
    return eval_root / mode / "libero_10/gpu0_task9_results.json"


def _write_result(path: Path, fingerprint: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"evaluation_fingerprint": fingerprint}), encoding="utf-8"
    )


def _run_launcher(
    tmp_path: Path,
    launcher: str,
    *,
    result_kind: str | None,
) -> tuple[subprocess.CompletedProcess[str], dict[str, str]]:
    env, eval_root = _launcher_environment(tmp_path, launcher)
    if result_kind == "legacy":
        _write_result(_stale_result_path(eval_root, launcher), {"schema_version": 2})
    elif result_kind == "stale":
        _write_result(
            _stale_result_path(eval_root, launcher),
            {"schema_version": 3, "identity": "different"},
        )
    elif result_kind == "exact":
        for mode in _launcher_modes(launcher):
            for task_id in range(10):
                _write_result(
                    eval_root
                    / mode
                    / "libero_10"
                    / f"gpu0_task{task_id}_results.json",
                    EXPECTED_FINGERPRINT,
                )
    elif result_kind is not None:
        raise ValueError(f"unsupported result_kind: {result_kind}")
    completed = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / launcher)],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )
    return completed, env


@pytest.mark.parametrize("result_kind", ("legacy", "stale"))
@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_mismatched_result_fails_before_gpu_probe_or_evaluator(
    tmp_path, launcher, result_kind
):
    completed, env = _run_launcher(
        tmp_path, launcher, result_kind=result_kind
    )
    assert completed.returncode == 2, completed.stdout
    assert "REFUSING mixed evaluation directory" in completed.stdout
    assert not Path(env["FAKE_GPU_MARKER"]).exists()
    assert not Path(env["FAKE_EVALUATOR_MARKER"]).exists()


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_exact_results_are_reused_without_gpu_or_evaluator(tmp_path, launcher):
    completed, env = _run_launcher(tmp_path, launcher, result_kind="exact")
    assert completed.returncode == 0, completed.stdout
    assert not Path(env["FAKE_GPU_MARKER"]).exists()
    assert not Path(env["FAKE_EVALUATOR_MARKER"]).exists()


def _config_and_overrides(argv: list[str], *, builder: bool) -> tuple[str, dict[str, str]]:
    config_name = argv[argv.index("--config-name") + 1]
    tokens = argv[argv.index("--") + 1 :] if builder else argv[3:]
    overrides = dict(token.split("=", 1) for token in tokens if "=" in token)
    for operational in (
        "gpu_id",
        "EVALUATION.device",
        "EVALUATION.output_dir",
        "+EVALUATION.expected_fingerprint_path",
    ):
        overrides.pop(operational, None)
    return config_name, overrides


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_builder_and_evaluator_use_identical_behavior_overrides(tmp_path, launcher):
    completed, env = _run_launcher(tmp_path, launcher, result_kind=None)
    assert completed.returncode == 0, completed.stdout
    calls = [
        json.loads(line)
        for line in Path(env["FAKE_PYTHON_LOG"]).read_text(encoding="utf-8").splitlines()
    ]
    builders = [call for call in calls if call[0].endswith("build_eval_fingerprint.py")]
    evaluators = [call for call in calls if call[0].endswith("eval_libero_single.py")]
    assert builders
    assert len(builders) == len(evaluators)

    expected_by_run = {}
    for call in builders:
        config_name, overrides = _config_and_overrides(call, builder=True)
        run_key = (config_name, overrides["ckpt"], overrides["EVALUATION.task_id"])
        expected_by_run[run_key] = overrides

    actual_by_run = {}
    for call in evaluators:
        config_name, overrides = _config_and_overrides(call, builder=False)
        run_key = (config_name, overrides["ckpt"], overrides["EVALUATION.task_id"])
        actual_by_run[run_key] = overrides

    assert actual_by_run == expected_by_run


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_evaluator_model_and_egl_use_same_physical_gpu(tmp_path, launcher):
    completed, env = _run_launcher(tmp_path, launcher, result_kind=None)
    assert completed.returncode == 0, completed.stdout
    records = [
        json.loads(line)
        for line in Path(env["FAKE_EVALUATOR_ENV_LOG"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert records
    for record in records:
        assert not record["cuda_visible_devices_present"]
        assert record["mujoco_egl_device_id"] == "3"
        assert "gpu_id=3" in record["argv"]
        assert "EVALUATION.device=cuda:3" in record["argv"]


def test_depth_launcher_compares_exits_under_the_fixed_trained_window():
    source = (REPO_ROOT / "scripts" / "evaluate_pareto.sh").read_text()
    assert 'DEPTHS_CSV="${DEPTHS_CSV:-8,16,24,30}"' in source
    assert 'HISTORY_WINDOW_BLOCKS="${HISTORY_WINDOW_BLOCKS:-8}"' in source
    assert 'HISTORY_CAPS_CSV="${HISTORY_CAPS_CSV:-$HISTORY_WINDOW_BLOCKS}"' in source
    assert 'KV_RETENTION_CAPS_CSV="${KV_RETENTION_CAPS_CSV:-$HISTORY_CAPS_CSV}"' in source
    assert '"$kv_retention_cap" != "$HISTORY_WINDOW_BLOCKS"' in source
    assert (
        'config_root="$GRID_ROOT/configs/d${depth}_w${kv_retention_cap}"'
        in source
    )
    assert "strict-window=$kv_retention_cap" in source
    assert 'HISTORY_WINDOW_BLOCKS="$kv_retention_cap"' in source
    assert "depth/strict-window Pareto complete" in source
    assert "--expected-trained-exit-depths 8,16,24,30" in source
