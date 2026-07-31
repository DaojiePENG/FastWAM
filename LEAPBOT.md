# LeapBot-VA

LeapBot-VA is a FastWAM-derived world-action model whose persistent memory is
made exclusively from real observations and commands actually sent to the
controller. It keeps FastWAM's joint video/action flow-matching objective for
training, but its online action path does not create future-video latents, run
the video output head, or decode VAE video.

## Runtime contract

One episode owns one explicit `LeapMemoryState`:

```python
memory = model.create_memory(
    exit_depth=30,
    causal_mode="interleaved",
    max_history_blocks=70,
)
prediction = model.infer_action(
    prompt=prompt,
    input_image=current_real_image,
    proprio=current_proprio,
    action_horizon=32,
    memory=memory,
)

# Execute at most 10 commands after environment postprocessing. Convert those
# exact commands back to normalized model space, then commit only that slice.
model.commit_executed_actions(memory, executed_actions_model_space)
model.reset_memory(memory)  # at episode end
```

The state machine rejects a second observation before an action commit,
mid-episode prompt changes, depth changes, and history beyond the configured
capacity. A checkpoint may use only exit depths recorded as actually trained;
a D30-only checkpoint cannot silently run D8/16/24. `memory=None` retains the
original FastWAM inference behavior.

The LIBERO bridge canonicalizes each final command after any postprocessing:
it clips to `env.action_spec`, applies deterministic gripper binarization, and
passes the same array to both `env.step` and the memory commit conversion.
Cross-replan action ensembling is rejected when memory is enabled, so the 22
unexecuted predictions in a 32-step chunk cannot enter later history.

The causal modes are:

- `interleaved`: a new observation reads historical observation and action KV.
- `vision_causal`: a new observation reads only historical observation KV.
- `action_aggregator`: observations are encoded independently and ActionDiT
  aggregates the complete history.

In all modes, current action queries read historical observations/actions, the
current real observation, and the bidirectional current action block. They can
never read future-video supervision tokens.

## Environment and assets

Commands in this workspace are intended to run as user `sheng`:

```bash
cd /home/sheng/workspace/leapbot-va
uv venv --python /usr/bin/python3.10 .venv
uv sync --dev
source .venv/bin/activate
python scripts/download_leapbot_assets.py
export LEAPBOT_DATASET_STATS="$PWD/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json"
python scripts/precompute_text_embeds.py task=libero_leapbot_2cam224
```

The downloader uses the official FastWAM Hugging Face repositories
`yuanty/LIBERO-fastwam` and `yuanty/fastwam`; it does not read RLDS data.

## Training phases

The production training path is runtime-isomorphic causal attention with
`incremental_full_bptt`: every real observation/action block before the current
replan is executed chronologically and remains in the graph. There is no
history gate and no detached prefix.
Native block-local RoPE is preserved, while a learned episode clock is expressed
relative to those local coordinates; block zero is therefore an exact position
no-op even after the clock parameters train. This does not freeze the shared
DiT weights, so release-behavior retention is measured separately with paired
H0 samples.
All comparison runs use the same FastWAM release initialization, split, update
count, global batch, scheduler, and seed. The launcher refuses a dirty worktree
and binds every resumable state to a hash of the exact commit, release weights,
data statistics, topology, optimizer, and temporal/history configuration.

Formal training fixes the history-VAE chunk to 1. Every real observation uses
the same batch-one, T=1 VAE call as rollout; the earlier chunk-2 approximation
and all packed-attention runs are invalidated and are not used as results.

```bash
# Paired screens are followed by fixed-observation, fixed-timestep, fixed-noise
# audits with correct, masked, and cross-episode shuffled histories.
bash scripts/screen_learning_rate.sh
bash scripts/audit_learning_rate.sh
LR_SELECTION_MANIFEST=<lr-selection.json> \
  bash scripts/screen_h0_retention.sh
LR_SELECTION_MANIFEST=<lr-selection.json> \
  bash scripts/audit_h0_retention.sh

# Phase 1: after the paired LR audit selects a learning rate, train all three
# modes sequentially with the complete episode prefix, D30, BF16, the same
# 8-GPU topology/global batch, and a 5% warmup + cosine schedule.
LR_SELECTION_MANIFEST=<lr-selection.json> \
  H0_SELECTION_MANIFEST=<h0-selection.json> \
  bash scripts/train_causal_modes.sh

# Phase 2: initialize from the winning D30 history checkpoint and train exits.
SOURCE_TRAIN_ROOT=/path/to/d30_root MODE=<winner> SOURCE_STEP=<step> \
  MAX_STEPS=<steps> \
  bash scripts/train_multi_exit.sh
```

Short 0-8 or 0-16 windows are code-supported controlled ablations, not the main
recipe and not substitutes for complete-prefix training. The canonical
production pipeline has no short-window launcher, and the current formal
results do not run or report those ablations.
The released LIBERO training episodes provide real prefixes through H=50. The
70-block setting is a capacity and inference extrapolation bound; H=51..69 is
not represented as observed training history and must be reported as such.

The inference ablations labelled `kvret0/8/16/32/full` cap the number of
physically retained KV blocks. They are not strict information windows: a
newer causal KV was computed while attending to its older prefix and therefore
can still encode information from blocks whose tensors are later evicted.
Strict last-N-information ablations would require replaying retained raw
observations/actions after every eviction and are outside the online cache
design. The default full-episode configuration performs no eviction.

For phase 3 the objective is exactly
`L30 + (L8 + L16 + L24) / 3`; every `Ld` contains video and action
flow-matching losses.

## Evaluation

```bash
python experiments/libero/eval_libero_single.py \
  --config-name sim_leapbot_libero \
  ckpt=/path/to/leapbot.pt \
  model.training_strategy=video_lora_action_full \
  model.video_lora.enabled=true \
  EVALUATION.task_id=0 \
  EVALUATION.num_trials=10 \
  EVALUATION.memory.exit_depth=16

python experiments/leapbot/pareto.py evaluate_results/leapbot

# After training the winning four-exit checkpoint, run the complete
# D={8,16,24,30} x H={0,8,16,32,full} grid with isolated result trees.
TRAIN_ROOT=/path/to/multi_exit_run MODE=<winner> FINAL_STEP=<step> \
  bash scripts/evaluate_pareto.sh
```

Run all 10 `libero_10` tasks with 10 trials for development, then 50 trials per
task for the final table. Replan latency is a closed raw-observation-to-command
measurement: input preprocessing, context conditioning, real-observation
prefill, action-history materialization/setup, action denoising, command
postprocessing, and executed-action KV commit are retained separately. Cache
peaks include the temporary post-observation/pre-commit state. The result
fingerprint hashes the LeapBot worktree, LIBERO revision, simulator package
versions, task BDDL, initial states, runtime configuration, and checkpoint.

The Pareto tool keeps the overall non-dominated success/latency/memory frontier,
including FastWAM as a comparator. The default LeapBot configuration is chosen
only from memory-enabled LeapBot rows using the one-percentage-point plus
overlapping-confidence-interval rule; FastWAM can never be mislabeled as the
LeapBot default.

## Verification

```bash
PYTHONPATH=src python -m pytest tests -q
```

The optional real-6B acceptance tools are
`scripts/validate_real_6b_runtime_training_equivalence.py` (training/runtime KV
and fixed-noise loss equivalence) and `scripts/full_prefix_smoke.py` (real-prefix
optimizer topology and capacity protection). They are cluster preflight tools,
not alternate training entrypoints; their exact invocations and resource scope
are documented in the
[training and reproduction runbook](./docs/TRAINING_AND_REPRODUCTION.md).

Unit coverage includes causal leakage, action/observation state transitions,
rollback/reset/capacity, exact context fingerprints, hierarchical positions,
three-mode H=8 FP32/BF16 incremental-vs-one-shot KV equivalence, executed
gripper re-normalization, deterministic pre-instantiation seeding, resume-run
contracts, trained-exit enforcement, multi-depth outputs, and Pareto selection. Full 6B
H800 training and 500-episode benchmark runs require the release assets and
trained LeapBot checkpoints; unit tests do not fabricate those results.

The completed 6B single-step and 70-block H800 measurements are recorded in
[reports/SMOKE_H800.md](./reports/SMOKE_H800.md).
