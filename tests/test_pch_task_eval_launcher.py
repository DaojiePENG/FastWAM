from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/evaluate_pch_checkpoint.sh"
CONFIG = ROOT / "configs/sim_leapbot_libero_pch.yaml"
MANAGER = ROOT / "experiments/libero/run_libero_manager.py"
PARALLEL = ROOT / "experiments/libero/run_libero_parallel_test.sh"


def test_pch_launcher_only_selects_config_and_checkpoint():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "--config-name sim_leapbot_libero_pch" in source
    assert "task=libero_leapbot_pch" in source
    assert '"ckpt=$CKPT"' in source
    assert "EVALUATION." not in source
    assert "MODE=" not in source
    assert "TRIALS=" not in source


def test_pch_eval_config_owns_contract_video_and_concurrency():
    source = CONFIG.read_text(encoding="utf-8")
    for setting in (
        "history_storage_mode: strict_replay",
        "history_window_blocks: ${model.history_window_blocks}",
        "max_history_blocks: 70",
        "replan_steps: 10",
        "action_horizon: 32",
        "save_rollout_video: true",
        "num_gpus: 4",
        "max_tasks_per_gpu: 3",
    ):
        assert setting in source


def test_parallel_workers_use_immutable_resolved_config_snapshot():
    manager = MANAGER.read_text(encoding="utf-8")
    parallel = PARALLEL.read_text(encoding="utf-8")
    assert "OmegaConf.to_container(cfg, resolve=True)" in manager
    assert '"CONFIG_SNAPSHOT": str(config_snapshot.resolve())' in manager
    assert '"CONFIG_SNAPSHOT_SHA256": config_snapshot_sha256' in manager
    assert "--config-path" in parallel
    assert "verify_config_snapshot" in parallel
    assert "task=$CONFIG" not in parallel
    assert "--config-name=$SIM_CONFIG" not in parallel


def test_parallel_launcher_generates_pareto_statistics_after_summary():
    source = PARALLEL.read_text(encoding="utf-8")
    assert 'experiments/leapbot/pareto.py "${pareto_args[@]}"' in source
    assert '--expected-tasks "$total_tasks"' in source
    assert '--expected-trials-per-task "$NUM_TRIALS"' in source
    assert "--require-profiled" in source
