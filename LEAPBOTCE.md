# LeapBotCE

Cluster handoff and experiment guide for the CloudEdge FastWAM implementation.

LeapBotCE is a CloudEdge adaptation of FastWAM. The cloud side consumes a
possibly delayed observation with the frozen Wan video expert and transmits its
MoT video KV cache as a planning representation. The edge side consumes current
camera views with a frozen SigLIP-Base encoder and fuses its trainable projected
token with current proprioception and language in ActionDiT.

```text
delayed cloud observation -> frozen Wan -> planning KV cache
current edge views       -> frozen SigLIP -> trainable projector -> ActionDiT
current language + proprio -------------------------------------> ActionDiT
```

The public two-stage API is:

```python
planning_cache = model.encode_cloud(delayed_image, prompt=prompt, proprio=proprio)
prediction = model.infer_action_edge(planning_cache, current_views, action_horizon=32)
```

`model.infer_action(...)` remains the synchronous compatibility wrapper.

## Status

This commit supports and has been tested on the two-camera, 7D-action
LIBERO-Spatial recipe. The trainable set is ActionDiT, the SigLIP projector,
and the proprio projector. Wan and SigLIP are frozen. The inference checkpoint
is a compact delta: it requires the local Wan, ActionDiT-backbone, and SigLIP
base weights listed below.

RoboTwin CloudEdge training and delayed evaluation are **not implemented in
this commit**. RoboTwin has three cameras, a 14D action/state space, and a
different runtime policy interface; do not use the LIBERO two-view configuration
as if it were a valid RoboTwin CE recipe. The work package is specified below.

## Cluster Setup

Install the repository environment first:

```bash
conda create -n fastwam python=3.10 -y
conda activate fastwam
pip install -e ".[dev]"
```

Prepare local model assets. No automatic downloads are performed at training or
evaluation time.

- Wan2.2 TI2V 5B weights in the Hugging Face/DiffSynth cache configured by
  `DIFFSYNTH_MODEL_BASE_PATH`.
- `checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`.
- `checkpoints/siglip-base/model.safetensors` for `vit_base_patch16_siglip_224`.
- LIBERO data at `data/libero_mujoco3.3.2/`, cached text embeddings at
  `data/text_embeds_cache/libero`, and the LIBERO repository cloned at
  `third_party/LIBERO`.
- RoboTwin data at `data/robotwin2.0/`, cached text embeddings at
  `data/text_embeds_cache/robotwin`, and a separately installed official
  RoboTwin checkout at `third_party/RoboTwin`.

The local simulator checkouts and `.libero/` configuration are ignored by Git.
Follow the upstream LIBERO and RoboTwin installation instructions for simulator
assets and system dependencies. The main README covers FastWAM data and base
weight preparation in more detail.

For LIBERO evaluation, set:

```bash
export LIBERO_CONFIG_PATH="$PWD/.libero"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="$PWD/third_party/LIBERO:$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

## LIBERO CE Training

The default `libero_spatial_leapbotce` recipe is a 10,000 optimizer-step
fine-tune. With two ranks, `batch_size=2`, and accumulation 4, its effective
batch size is 16. It trains against a delay range of 1--20 steps and ramps the
stale loss coefficient from 0 to 0.5 over the first half of training.

```bash
bash scripts/train_zero1.sh 2 task=libero_spatial_leapbotce \
  output_dir=./runs/libero_spatial_leapbotce/dtrain20_seed42 \
  seed=42
```

Run the lightweight integration check before reserving a large job:

```bash
bash scripts/train_zero1.sh 1 task=libero_spatial_leapbotce_smoke
```

Train with a 40-step delay distribution by overriding the dataset range. The
dataset dynamically expands its same-episode observation window, so no manual
`num_frames` change is required.

```bash
bash scripts/train_zero1.sh 8 task=libero_spatial_leapbotce \
  data.train.max_delay_steps=40 \
  output_dir=./runs/libero_spatial_leapbotce/dtrain40_seed42 \
  seed=42
```

Use a cluster-specific Accelerate/DeepSpeed configuration as needed. The
repository default uses ZeRO-1; checkpoint retention is intentionally set to
one latest weight and one resumable state. The latter can be large, so archive
or remove it only after confirming that training will not be resumed.

## LIBERO Delay Evaluation

The delayed path changes only the cloud planning observation. Current edge
views and current proprioception always remain fresh. At every replan, delay is
sampled uniformly from `1..min(d_max, available_history)`; insufficient
history falls back to the current frame. `d_max=0` is synchronous.

For a single task, the sweep script accepts arbitrary delay windows through
`LEAPBOTCE_DELAYS`:

```bash
LEAPBOTCE_DELAYS="0 5 10 20 40 80" \
bash scripts/eval_leapbotce_delay_sweep.sh /path/to/step_010000.pt 0 \
  EVALUATION.num_trials=50 \
  EVALUATION.num_inference_steps=10 \
  EVALUATION.replan_steps=10
```

For a complete LIBERO-Spatial run, use persistent workers so each GPU loads the
5B model only once per delay window. Fix the seed and retain every generated
manager config, task JSON, summary JSON, and CSV.

```bash
CKPT=/path/to/step_010000.pt
OUT=./evaluate_results/leapbotce_spatial_dtrain40_seed42
for delay in 0 5 10 20 40 80; do
  CUDA_VISIBLE_DEVICES=0,1,2,3 python experiments/libero/run_libero_manager.py \
    task=libero_spatial_leapbotce \
    ckpt="${CKPT}" \
    seed=42 \
    EVALUATION.num_trials=50 \
    EVALUATION.max_delay_steps="${delay}" \
    EVALUATION.action_horizon=32 \
    EVALUATION.num_inference_steps=10 \
    EVALUATION.replan_steps=10 \
    EVALUATION.output_dir="${OUT}/delay_${delay}" \
    MULTIRUN.num_gpus=4 \
    model.load_text_encoder=true \
    model.skip_dit_load_from_pretrain=false
done
python scripts/summarize_leapbotce_delays.py "${OUT}"
```

The explicit `model.skip_dit_load_from_pretrain=false` is mandatory for a
LeapBotCE delta checkpoint. The delta does not serialize frozen Wan weights;
using a random Wan cloud expert invalidates the measurement.

The summary script writes `delay_sweep_summary.json` and
`delay_sweep_tasks.csv`. Record at minimum the overall and per-task success
rate, Wilson 95% interval, actual-delay histogram/mean, and retention relative
to delay zero.

## Formal LIBERO Result

The initial LIBERO-Spatial experiment used 10,000 steps, bf16, ZeRO-1,
effective batch 16, a train-time maximum delay of 20, and 10 rollouts per task
for each of the 10 Spatial tasks. This is a development benchmark, not the
final 50-rollout report:

| Eval max delay | Success | Actual mean delay | Retention vs. d=0 |
| --- | ---: | ---: | ---: |
| 0 | 46/100 | 0.00 | 100.0% |
| 5 | 45/100 | 3.09 | 97.8% |
| 10 | 43/100 | 5.30 | 93.5% |
| 20 | 45/100 | 10.06 | 97.8% |

The 95% Wilson intervals overlap. The evidence supports no obvious loss at
these windows for this initial model, but does not establish an ordering among
the nonzero delays. The absolute synchronous score is also not yet strong
enough to treat this as a final comparison against published baselines.

## Required Benchmark Program

Prioritize breadth of comparable training and evaluation over more variants of
the current single recipe.

| Priority | Work item | Required outputs |
| --- | --- | --- |
| P0 | LIBERO CE: train `d_train=20` and `d_train=40`; evaluate `d_eval={0,5,10,20,40,80}` with 3 seeds and 50 trials/task. | Checkpoints, per-seed task JSON, aggregate mean/std, Wilson intervals, actual delay histograms. |
| P1 | RoboTwin CE adaptation and training. Preserve three raw cameras; use a cloud composite plus current three-view edge encoder, 14D action/state, and an observation-history policy wrapper. Train/evaluate `d_train={20,40}`, `d_eval={0,5,10,20,40,80}` for clean and randomized protocols. | Dataset/config changes, deployment adapter tests, checkpoints, clean/random task tables and delay curves. |
| P2 | Delayed-input baselines: FastWAM first, then LingBot-VA, Motus, and any available policy. | Pinned upstream commit, adapter source, identical task lists/seeds/trials, synchronous and delayed result files. |
| P3 | Generalization: evaluate `d_eval=2*d_train` and, if episode length permits, larger windows. | Retention curve and actual sampled-delay distribution, including early-history fallback rate. |

### RoboTwin CE Work Package

RoboTwin's current FastWAM policy mosaics three cameras into one image. A valid
CE implementation must instead retain raw `head`, `left_wrist`, and
`right_wrist` images long enough to construct separate edge views. The minimum
implementation is:

1. Add a CloudEdge RoboTwin dataset/config with `edge_num_views=3`, a 14D
   action/state shape, and same-episode stale/current indexing.
2. Keep a cloud composite compatible with Wan while passing three fresh views
   directly to SigLIP; do not split the existing 384x320 mosaic by width.
3. Add a delayed-history ring buffer to
   `experiments/robotwin/fastwam_policy/deploy_policy.py`. Recompute cloud KV
   only from the sampled historical observation; edge views and proprio remain
   current at every replan.
4. Extend `eval_robotwin_single.py` and `run_robotwin_manager.py` to record
   requested/actual delay samples, clean/random success, seed, and task-level
   JSON. Validate `d=0` equivalence before evaluating nonzero delay.

### Baseline Delay Protocol

Use a single protocol across LeapBotCE, FastWAM, LingBot-VA, and Motus:

1. Pin each baseline repository and checkpoint SHA, simulator version, task
   list, instruction mode, action horizon, replan interval, seed, and trial
   count.
2. Report synchronous `d=0` before delayed results. A baseline must preserve
   its published synchronous path before its delay adapter is trusted.
3. At every replan, sample the same uniform history rule and reset history at
   every episode. Report the realized rather than requested delay distribution.
4. For policies without an edge/cloud split, delay their visual observation
   input while retaining current proprioception only when their native API
   supports it. Also report a fully stale-observation variant when possible;
   label the two conditions separately rather than treating them as identical.
5. Compare success retention, absolute success, and inference latency. Do not
   compare a CE result with fresh edge views against a baseline whose full state
   is stale without stating the observation contract.

## Verification and Storage

Run focused checks after code changes:

```bash
PYTHONPATH="$PWD/src:$PWD" pytest -q \
  tests/test_leapbotce.py \
  tests/test_cloudedge_delay.py \
  tests/test_summarize_leapbotce_delays.py
```

Before a long run, validate that the evaluator log contains a Wan pretrained
load and does not contain `Skipping pretrained video DiT`. Keep result JSON,
summaries, configs, and selected videos. Do not retain every DeepSpeed state:
the final resumable state can be tens of GB, whereas the LeapBotCE delta
checkpoint is about 2 GB.
