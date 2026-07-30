from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_base_launcher_contract_identifies_initial_weights_and_exit_set():
    source = (ROOT / "scripts" / "run_hierarchical_raw_v1_peft_5k.sh").read_text()
    assert 'INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-$RELEASE_CHECKPOINT}"' in source
    assert '"initial_checkpoint_sha256=$INITIAL_CHECKPOINT_SHA256"' in source
    assert 'TRAINING_EXIT_DEPTHS_CSV="${TRAINING_EXIT_DEPTHS_CSV:-30}"' in source
    assert '"model.training_exit_depths=[$TRAINING_EXIT_DEPTHS_CSV]"' in source
    assert '--expected-trained-exit-depths "$expected_training_exit_depths"' in source
    assert "training_asset_manifest.py" in source
    assert '"dataset_content_sha256=$DATASET_CONTENT_SHA256"' in source
    assert '"text_embedding_cache_sha256=$TEXT_EMBEDDING_CACHE_SHA256"' in source
    assert '"vae_checkpoint_sha256=$VAE_CHECKPOINT_SHA256"' in source
    assert 'DIFFSYNTH_MODEL_BASE_PATH="$ROOT_DIR/checkpoints"' in source
    assert 'ACTUAL_RELEASE_CHECKPOINT_SHA256="$(sha256sum "$RELEASE_CHECKPOINT"' in source
    assert 'RELEASE_CHECKPOINT_SHA256="$ACTUAL_RELEASE_CHECKPOINT_SHA256"' in source
    assert '"padding_attention_mask=true"' in source
    assert "final_incremental_full_bptt_v4_" in source
    assert "final-incremental-full-bptt-v4" in source
    assert "full_bptt_v3" not in source
    assert "full-bptt-v3" not in source


def test_multi_exit_launcher_requires_winner_and_all_four_exits():
    source = (ROOT / "scripts" / "run_multi_exit_training.sh").read_text()
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
    source = (ROOT / "scripts" / "run_causal_full_bptt_comparison.sh").read_text()
    assert "REQUIRE_SELF_IDENTIFYING_CHECKPOINT=true" in source
    assert 'LR_SELECTION_MANIFEST="${LR_SELECTION_MANIFEST:?' in source
    assert 'H0_SELECTION_MANIFEST="${H0_SELECTION_MANIFEST:?' in source
    assert "--expected-kind learning_rate" in source
    assert "--expected-kind initial_block_oversample" in source
    assert "lr_selection_manifest_sha256" in source
    assert "h0_selection_manifest_sha256" in source
    assert "causal_incremental_full_bptt_v4" in source
    assert "causal-incremental-full-bptt-v4" in source
    assert "full_bptt_v3" not in source
    assert "full-bptt-v3" not in source


def test_h0_retention_screen_consumes_the_selected_lr():
    source = (ROOT / "scripts" / "run_paired_h0_retention_screen.sh").read_text()
    assert 'LR_SELECTION_MANIFEST="${LR_SELECTION_MANIFEST:?' in source
    assert "--expected-kind learning_rate" in source
    assert 'LEARNING_RATE="$SELECTED_LR"' in source
    assert "1.0e-5|1.0e-4" in source
    assert 'LEARNING_RATE="${LEARNING_RATE:-1.0e-4}"' not in source


def test_lr_audit_is_fixed_noise_paired_and_runtime_isomorphic():
    source = (ROOT / "scripts" / "run_paired_lr_audit.sh").read_text()
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
    source = (ROOT / "scripts" / "run_paired_h0_audit.sh").read_text()
    assert 'LR_SELECTION_MANIFEST="${LR_SELECTION_MANIFEST:?' in source
    assert "initial_block_oversample" in source
    assert "[$RELEASE_CHECKPOINT,$X1_CHECKPOINT,$X4_CHECKPOINT]" in source
    assert "[correct,masked,shuffled]" in source
    assert "stratified.fixed_u_values" in source
    assert "--expected-run-contract-sha256" in source


def test_paired_screen_and_audit_defaults_use_v5_identity():
    launchers = (
        "run_paired_lr_screen.sh",
        "run_paired_lr_audit.sh",
        "run_paired_h0_retention_screen.sh",
        "run_paired_h0_audit.sh",
    )
    for launcher in launchers:
        source = (ROOT / "scripts" / launcher).read_text()
        assert "incremental_v5" in source
        assert "incremental_v4" not in source
        assert "incremental-v4" not in source
    assert "lr-screen-incremental-v5" in (
        ROOT / "scripts" / "run_paired_lr_screen.sh"
    ).read_text()
    assert "h0-retention-incremental-v5" in (
        ROOT / "scripts" / "run_paired_h0_retention_screen.sh"
    ).read_text()
