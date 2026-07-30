from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_base_launcher_contract_identifies_initial_weights_and_exit_set():
    source = (ROOT / "scripts" / "run_hierarchical_raw_v1_peft_5k.sh").read_text()
    assert 'INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-$RELEASE_CHECKPOINT}"' in source
    assert '"initial_checkpoint_sha256=$INITIAL_CHECKPOINT_SHA256"' in source
    assert 'TRAINING_EXIT_DEPTHS_CSV="${TRAINING_EXIT_DEPTHS_CSV:-30}"' in source
    assert '"model.training_exit_depths=[$TRAINING_EXIT_DEPTHS_CSV]"' in source
    assert '--expected-trained-exit-depths "$expected_training_exit_depths"' in source


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
    assert "training_exit_depths=30" in source
    assert "^[0-9a-f]{64}$" in source
    assert "^[0-9a-f]{40}$" in source
    assert "REQUIRE_SELF_IDENTIFYING_CHECKPOINT=true" in source


def test_formal_causal_comparison_requires_self_identifying_checkpoints():
    source = (ROOT / "scripts" / "run_causal_full_bptt_comparison.sh").read_text()
    assert "REQUIRE_SELF_IDENTIFYING_CHECKPOINT=true" in source
