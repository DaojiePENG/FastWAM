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
capacity. `memory=None` retains the original FastWAM inference behavior.

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

All comparison runs should use the same FastWAM release initialization, split,
step count, and seed.

```bash
# Phase 1: train three separate 0-8-history, 30-layer checkpoints.
accelerate launch scripts/train.py task=libero_leapbot_2cam224 \
  model.causal_mode=interleaved model.training_exit_depths='[30]'
accelerate launch scripts/train.py task=libero_leapbot_2cam224 \
  model.causal_mode=vision_causal model.training_exit_depths='[30]'
accelerate launch scripts/train.py task=libero_leapbot_2cam224 \
  model.causal_mode=action_aggregator model.training_exit_depths='[30]'

# Phase 2: repeat the winning mode with 0-16 history.
accelerate launch scripts/train.py task=libero_leapbot_2cam224 \
  data.train.max_history_blocks=16 model.causal_mode=<winner>

# Phase 3: initialize from the winning history checkpoint and train all exits.
accelerate launch scripts/train.py task=libero_leapbot_2cam224 \
  resume=<winner.pt> model.causal_mode=<winner> \
  model.training_exit_depths='[8,16,24,30]'
```

For phase 3 the objective is exactly
`L30 + (L8 + L16 + L24) / 3`; every `Ld` contains video and action
flow-matching losses.

## Evaluation

```bash
python experiments/libero/eval_libero_single.py \
  --config-name sim_leapbot_libero \
  ckpt=/path/to/leapbot.pt \
  EVALUATION.task_id=0 \
  EVALUATION.num_trials=10 \
  EVALUATION.memory.exit_depth=16

python experiments/leapbot/pareto.py evaluate_results/leapbot
```

Run all 10 `libero_10` tasks with 10 trials for development, then 50 trials per
task for the final table. The evaluator records observation prefill, action
denoising, executed-action commit, cache bytes, peak allocated GPU memory, and
control steps. The Pareto tool keeps all non-dominated success/latency/memory
configurations and applies the one-percentage-point plus overlapping-confidence
interval default-selection rule.

## Verification

```bash
PYTHONPATH=src python -m pytest tests -q

# With the release checkpoint present:
python scripts/smoke_leapbot_train_h800.py \
  --checkpoint checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  --device cuda:0
python scripts/smoke_leapbot_h800.py \
  --checkpoint checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  --dataset-stats checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  --device cuda:0 --blocks 70
```

Unit coverage includes causal leakage, action/observation state transitions,
rollback/reset/capacity, absolute RoPE positions, incremental-vs-one-shot KV
equivalence, executed gripper re-normalization, multi-depth outputs, and Pareto
selection. Full 6B H800 training and 500-episode benchmark runs require the
release assets and trained LeapBot checkpoints; unit tests do not fabricate
those results.

The completed 6B single-step and 70-block H800 measurements are recorded in
[reports/SMOKE_H800.md](./reports/SMOKE_H800.md).
