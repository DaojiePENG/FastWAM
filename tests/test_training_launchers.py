from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_canonical_shell_surface_is_executable_and_old_names_are_absent():
    canonical = (
        "train_leapbot.sh",
        "screen_learning_rate.sh",
        "audit_learning_rate.sh",
        "screen_h0_retention.sh",
        "audit_h0_retention.sh",
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
        "run_hierarchical_raw_v1_peft_5k.sh",
        "run_paired_lr_screen.sh",
        "run_paired_lr_audit.sh",
        "run_paired_h0_retention_screen.sh",
        "run_paired_h0_audit.sh",
        "run_causal_full_bptt_comparison.sh",
        "run_multi_exit_training.sh",
        "run_phase1_eval_after_training.sh",
        "run_single_mode_checkpoint_eval.sh",
        "run_fastwam_baseline_dev.sh",
        "run_depth_history_pareto.sh",
        "run_final_50_trial_comparison.sh",
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

    readme = (ROOT / "README.md").read_text()
    readme_zh = (ROOT / "README_zh.md").read_text()
    assert "compatibility wrapper above supports one machine only" in readme
    assert "64-GPU setup requires an external multi-node launcher" in readme
    assert "上述兼容启动脚本仅支持单机" in readme_zh


def test_base_launcher_contract_identifies_initial_weights_and_exit_set():
    source = (ROOT / "scripts" / "train_leapbot.sh").read_text()
    assert 'INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-$RELEASE_CHECKPOINT}"' in source
    assert '"initial_checkpoint_sha256=$INITIAL_CHECKPOINT_SHA256"' in source
    assert 'TRAINING_EXIT_DEPTHS_CSV="${TRAINING_EXIT_DEPTHS_CSV:-30}"' in source
    assert '"model.training_exit_depths=[$TRAINING_EXIT_DEPTHS_CSV]"' in source
    assert '--expected-trained-exit-depths "$expected_training_exit_depths"' in source
    assert "training_asset_manifest.py" in source
    assert '"dataset_content_sha256=$DATASET_CONTENT_SHA256"' in source
    assert '"text_embedding_cache_sha256=$TEXT_EMBEDDING_CACHE_SHA256"' in source
    assert '"text_cache_provenance_sha256=$TEXT_CACHE_PROVENANCE_SHA256"' in source
    assert '"text_cache_verification_method=$TEXT_CACHE_VERIFICATION_METHOD"' in source
    assert '"text_cache_verified_file_count=$TEXT_CACHE_VERIFIED_FILE_COUNT"' in source
    assert '"text_encoder_checkpoint_sha256=$TEXT_ENCODER_CHECKPOINT_SHA256"' in source
    assert '"tokenizer_sha256=$TOKENIZER_SHA256"' in source
    assert '"vae_checkpoint_sha256=$VAE_CHECKPOINT_SHA256"' in source
    assert 'DIFFSYNTH_MODEL_BASE_PATH="$ROOT_DIR/checkpoints"' in source
    assert 'flock -s "$TEXT_CACHE_LOCK_FD"' in source
    assert '"data.train.text_embedding_cache_dir=$TEXT_EMBEDDING_CACHE"' in source
    assert '"data.train.dataset_dirs=[$ROOT_DIR/data/' in source
    assert 'ACTUAL_RELEASE_CHECKPOINT_SHA256="$(sha256sum "$RELEASE_CHECKPOINT"' in source
    assert 'RELEASE_CHECKPOINT_SHA256="$ACTUAL_RELEASE_CHECKPOINT_SHA256"' in source
    assert '"padding_attention_mask=true"' in source
    assert "final_incremental_full_bptt_v5_mb8_ga2_" in source
    assert "final-incremental-full-bptt-v5-mb8-ga2" in source
    assert "full_bptt_v4" not in source
    assert "full-bptt-v4" not in source


def test_multi_exit_launcher_requires_winner_and_all_four_exits():
    source = (ROOT / "scripts" / "train_multi_exit.sh").read_text()
    assert 'SOURCE_TRAIN_ROOT="${SOURCE_TRAIN_ROOT:?' in source
    assert 'INITIAL_CHECKPOINT="$SOURCE_CHECKPOINT"' in source
    assert "TRAINING_EXIT_DEPTHS_CSV=8,16,24,30" in source
    assert "--expected-trained-exit-depths 30" in source
    assert "--expected-history-vae-batch-chunk-size 1" in source
    assert '[[ "$NUM_PROCESSES" -ne 8 ]]' in source
    assert '[[ "$BATCH_SIZE" -ne 1 ]]' in source
    assert '[[ "$GRAD_ACCUM" -ne 16 ]]' in source
    assert "history_training_mode=incremental_full_bptt" in source
    assert "full_episode_history=true" in source
    assert "padding_attention_mask=true" in source
    assert "training_exit_depths=30" in source
    assert "batch_size=8" in source
    assert "gradient_accumulation_steps=2" in source
    assert "^[0-9a-f]{64}$" in source
    assert "^[0-9a-f]{40}$" in source
    assert "REQUIRE_SELF_IDENTIFYING_CHECKPOINT=true" in source
    assert 'SOURCE_CODE_COMMIT" != "$CURRENT_CODE_COMMIT' in source
    assert 'INITIAL_BLOCK_OVERSAMPLE="$(contract_value initial_block_oversample)"' in source
    assert 'MULTI_EXIT_LR="$(contract_value learning_rate)"' in source
    assert "SOURCE_ASSET_MANIFEST_SHA256" in source
    assert "SOURCE_LR_SELECTION_SHA256" in source
    assert "SOURCE_H0_SELECTION_SHA256" in source
    assert "EXPECTED_TRAINING_ASSET_MANIFEST_SHA256" in source


def test_formal_causal_comparison_requires_self_identifying_checkpoints():
    source = (ROOT / "scripts" / "train_causal_modes.sh").read_text()
    assert "REQUIRE_SELF_IDENTIFYING_CHECKPOINT=true" in source
    assert 'LR_SELECTION_MANIFEST="${LR_SELECTION_MANIFEST:?' in source
    assert 'H0_SELECTION_MANIFEST="${H0_SELECTION_MANIFEST:?' in source
    assert "--expected-kind learning_rate" in source
    assert "--expected-kind initial_block_oversample" in source
    assert "lr_selection_manifest_sha256" in source
    assert "h0_selection_manifest_sha256" in source
    assert "causal_incremental_full_bptt_v5_mb8_ga2" in source
    assert "causal-incremental-full-bptt-v5-mb8-ga2" in source
    assert 'BATCH_SIZE="${BATCH_SIZE:-8}"' in source
    assert 'GRAD_ACCUM="${GRAD_ACCUM:-2}"' in source
    assert "full_bptt_v4" not in source
    assert "full-bptt-v4" not in source


def test_h0_retention_screen_consumes_the_selected_lr():
    source = (ROOT / "scripts" / "screen_h0_retention.sh").read_text()
    assert 'LR_SELECTION_MANIFEST="${LR_SELECTION_MANIFEST:?' in source
    assert "--expected-kind learning_rate" in source
    assert 'LEARNING_RATE="$SELECTED_LR"' in source
    assert "1.0e-5|1.0e-4" in source
    assert 'LEARNING_RATE="${LEARNING_RATE:-1.0e-4}"' not in source


def test_lr_audit_is_fixed_noise_paired_and_runtime_isomorphic():
    source = (ROOT / "scripts" / "audit_learning_rate.sh").read_text()
    assert "[$RELEASE_CHECKPOINT,$LOW_CHECKPOINT,$HIGH_CHECKPOINT]" in source
    assert "model.history_training_mode=incremental_full_bptt" in source
    assert "model.history_vae_batch_chunk_size=1" in source
    assert "[correct,masked,shuffled]" in source
    assert "stratified.fixed_u_values" in source
    assert "+stratified.noise_repeats=2" in source
    assert "--expected-history-vae-batch-chunk-size 1" in source
    assert "CANDIDATE_LEARNING_RATES=(1.0e-5 1.0e-4)" in source
    assert 'contract_file="$candidate_root/run_contract.txt"' in source
    assert '--expected-field "learning_rate=$learning_rate"' in source
    assert "--expected-run-contract-sha256" in source
    assert "--expected-code-commit" in source
    assert "validate_run_contract_group.py" in source


def test_h0_audit_validates_sampling_factor_and_uses_the_same_draw_contract():
    source = (ROOT / "scripts" / "audit_h0_retention.sh").read_text()
    assert 'LR_SELECTION_MANIFEST="${LR_SELECTION_MANIFEST:?' in source
    assert "initial_block_oversample" in source
    assert "[$RELEASE_CHECKPOINT,$X1_CHECKPOINT,$X4_CHECKPOINT]" in source
    assert "[correct,masked,shuffled]" in source
    assert "stratified.fixed_u_values" in source
    assert "--expected-run-contract-sha256" in source


def test_paired_screen_and_audit_defaults_use_v6_mb10_ga2_identity():
    launchers = (
        "screen_learning_rate.sh",
        "audit_learning_rate.sh",
        "screen_h0_retention.sh",
        "audit_h0_retention.sh",
    )
    for launcher in launchers:
        source = (ROOT / "scripts" / launcher).read_text()
        assert "incremental_v6_mb10_ga2" in source
        assert "incremental_v5" not in source
        assert "incremental-v5" not in source
    assert "lr-screen-incremental-v6-mb10-ga2" in (
        ROOT / "scripts" / "screen_learning_rate.sh"
    ).read_text()
    assert "h0-retention-incremental-v6-mb10-ga2" in (
        ROOT / "scripts" / "screen_h0_retention.sh"
    ).read_text()


def test_paired_screens_enforce_measured_b10_ga2_topology():
    for launcher in (
        "screen_learning_rate.sh",
        "screen_h0_retention.sh",
    ):
        source = (ROOT / "scripts" / launcher).read_text()
        assert 'BATCH_SIZE="${BATCH_SIZE:-10}"' in source
        assert 'GRAD_ACCUM="${GRAD_ACCUM:-2}"' in source
        assert "REQUIRE_SELF_IDENTIFYING_CHECKPOINT=true" in source
        assert '[[ "$BATCH_SIZE" -ne 10 ]]' in source
        assert '[[ "$GRAD_ACCUM" -ne 2 ]]' in source
        assert 'WANDB_ENABLED="${WANDB_ENABLED:-true}"' in source
        assert 'WANDB_MODE="${WANDB_MODE:-online}"' in source
        assert 'WANDB_ENABLED="$WANDB_ENABLED"' in source
        assert 'WANDB_MODE="$WANDB_MODE"' in source
