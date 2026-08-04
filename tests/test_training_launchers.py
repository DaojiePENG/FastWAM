from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_only_canonical_leapbot_shell_surface_remains():
    canonical = (
        "train_leapbot.sh",
        "train_causal_modes.sh",
        "train_multi_exit.sh",
        "evaluate_causal_modes.sh",
        "evaluate_checkpoint.sh",
        "evaluate_fastwam_baseline.sh",
        "evaluate_pareto.sh",
    )
    for name in canonical:
        path = ROOT / "scripts" / name
        assert path.is_file()
        assert path.stat().st_mode & 0o111

    obsolete = (
        "screen_learning_rate.sh",
        "select_learning_rate.sh",
        "audit_learning_rate.sh",
        "screen_h0_retention.sh",
        "audit_h0_retention.sh",
        "full_prefix_smoke.py",
        "history_audit_selection.py",
        "history_stratified_loss.py",
        "probe_zero2_high_history_capacity.py",
        "probe_zero2_high_history_capacity.sh",
    )
    assert not any((ROOT / "scripts" / name).exists() for name in obsolete)


def test_fastwam_legacy_wrappers_fail_closed_for_multi_node_topologies():
    for name in ("train_zero1.sh", "train_zero2.sh"):
        path = ROOT / "scripts" / "fastwam_legacy" / name
        source = path.read_text()
        assert path.stat().st_mode & 0o111
        assert '[[ "$NUM_MACHINES" != "1" ]]' in source
        assert '[[ "$MACHINE_RANK" != "0" ]]' in source
        assert "single-machine only" in source
        assert "external cluster launcher" in source
        assert 'RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"' in source
        assert "TCPStore" not in source
        assert "RUN_ID_SYNC" not in source
        assert "MASTER_ADDR" not in source
        assert "MASTER_PORT" not in source
        assert "--num_machines" not in source
        assert "--machine_rank" not in source


def test_base_launcher_encodes_the_configurable_strict_window_training_contract():
    source = (ROOT / "scripts" / "train_leapbot.sh").read_text()
    assert 'INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-$RELEASE_CHECKPOINT}"' in source
    assert 'HISTORY_WINDOW_BLOCKS="${HISTORY_WINDOW_BLOCKS:-8}"' in source
    assert '[[ "$HISTORY_WINDOW_BLOCKS" =~ ^[1-9][0-9]*$ ]]' in source
    assert '"$HISTORY_WINDOW_BLOCKS" != "8"' not in source
    assert 'LEARNING_RATE="${LEARNING_RATE:-1.0e-4}"' in source
    assert 'INITIAL_BLOCK_OVERSAMPLE="${INITIAL_BLOCK_OVERSAMPLE:-4}"' in source
    assert 'TRAINING_EXIT_DEPTHS_CSV="${TRAINING_EXIT_DEPTHS_CSV:-30}"' in source
    assert 'BATCH_SIZE="${BATCH_SIZE:-20}"' in source
    assert 'GRAD_ACCUM="${GRAD_ACCUM:-1}"' in source
    assert "model.history_training_mode=strict_replay_window_bptt" in source
    assert '"model.history_window_blocks=$HISTORY_WINDOW_BLOCKS"' in source
    assert '"data.train.history_window_blocks=$HISTORY_WINDOW_BLOCKS"' in source
    assert "data.train.history_sampling_mode=recent_window" in source
    assert "data.train.use_episode_anchor=true" in source
    assert "history_padding=left_masked" in source
    assert "episode_anchor=single_real_v0" in source
    assert "future_video_condition_clean_warmup_steps" in source
    assert "future_video_condition_noise_ramp_steps" in source
    assert "training_asset_manifest.py" in source
    assert 'flock -s "$TEXT_CACHE_LOCK_FD"' in source
    assert 'DIFFSYNTH_MODEL_BASE_PATH="$ROOT_DIR/checkpoints"' in source
    assert '"padding_attention_mask=true"' in source
    assert "LR_SELECTION_MANIFEST" not in source
    assert "H0_SELECTION_MANIFEST" not in source


def test_causal_comparison_fixes_every_factor_except_causal_mode():
    source = (ROOT / "scripts" / "train_causal_modes.sh").read_text()
    assert 'SELECTED_LR="${LEARNING_RATE:-1.0e-4}"' in source
    assert 'MAX_STEPS="${MAX_STEPS:-5000}"' in source
    assert 'SAVE_EVERY="${SAVE_EVERY:-179}"' in source
    assert 'BATCH_SIZE="${BATCH_SIZE:-20}"' in source
    assert 'GRAD_ACCUM="${GRAD_ACCUM:-1}"' in source
    assert 'NUM_PROCESSES="${NUM_PROCESSES:-8}"' in source
    assert 'MODES_CSV="${MODES_CSV:-action_aggregator,interleaved,vision_causal}"' in source
    assert "history_training_mode=strict_replay_window_bptt" in source
    assert "history_sampling_mode=recent_window" in source
    assert 'HISTORY_WINDOW_BLOCKS="${HISTORY_WINDOW_BLOCKS:-8}"' in source
    assert 'history_window_blocks=$HISTORY_WINDOW_BLOCKS' in source
    assert "padding_attention_mask=true" in source
    assert "training_exit_depths=30" in source
    assert "REQUIRE_SELF_IDENTIFYING_CHECKPOINT=true" in source
    assert "LR_SELECTION_MANIFEST" not in source
    assert "H0_SELECTION_MANIFEST" not in source


def test_multi_exit_launcher_inherits_a_validated_strict_window_winner():
    source = (ROOT / "scripts" / "train_multi_exit.sh").read_text()
    assert 'SOURCE_TRAIN_ROOT="${SOURCE_TRAIN_ROOT:?' in source
    assert 'INITIAL_CHECKPOINT="$SOURCE_CHECKPOINT"' in source
    assert "TRAINING_EXIT_DEPTHS_CSV=8,16,24,30" in source
    assert "--expected-trained-exit-depths 30" in source
    assert '--expected-history-window-blocks "$HISTORY_WINDOW_BLOCKS"' in source
    assert "history_training_mode=strict_replay_window_bptt" in source
    assert "history_sampling_mode=recent_window" in source
    assert 'history_window_blocks="$HISTORY_WINDOW_BLOCKS"' in source
    assert "history_padding=left_masked" in source
    assert "episode_anchor=single_real_v0" in source
    assert '[[ "$NUM_PROCESSES" -ne 8 ]]' in source
    assert '[[ "$BATCH_SIZE" -ne 1 ]]' in source
    assert '[[ "$GRAD_ACCUM" -ne 16 ]]' in source
    assert "REQUIRE_SELF_IDENTIFYING_CHECKPOINT=true" in source
    assert 'SOURCE_CODE_COMMIT" != "$CURRENT_CODE_COMMIT' in source
    assert 'INITIAL_BLOCK_OVERSAMPLE="$(contract_value initial_block_oversample)"' in source
    assert 'MULTI_EXIT_LR="$(contract_value learning_rate)"' in source
    assert "SOURCE_ASSET_MANIFEST_SHA256" in source
    assert "EXPECTED_TRAINING_ASSET_MANIFEST_SHA256" in source
    assert "SOURCE_LR_SELECTION_SHA256" not in source
    assert "SOURCE_H0_SELECTION_SHA256" not in source


def test_formal_launchers_use_repository_relative_environment_paths():
    for name in (
        "train_leapbot.sh",
        "train_causal_modes.sh",
        "train_multi_exit.sh",
        "evaluate_checkpoint.sh",
        "evaluate_causal_modes.sh",
        "evaluate_fastwam_baseline.sh",
        "evaluate_pareto.sh",
    ):
        source = (ROOT / "scripts" / name).read_text()
        assert 'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"' in source
        assert "/home/sheng" not in source


def test_reproduction_docs_describe_only_the_strict_window_recipe():
    docs = (ROOT / "docs" / "TRAINING_AND_REPRODUCTION.md").read_text()
    assert "strict_replay_window_bptt" in docs
    assert "history_window_blocks=8" in docs
    assert "左侧全 mask padding" in docs
    assert "srun -p i64m1tga800u" in docs
    assert "train_causal_modes.sh" in docs
    assert "incremental_full_bptt` checkpoint" in docs
