# LeapBotCE

LeapBotCE extends FastWAM with asynchronous cloud-edge action inference. The
Wan video expert runs on a potentially delayed observation and returns its
per-layer MoT KV cache. A frozen SigLIP-Base encoder processes the current
primary and wrist views at the edge; its trainable projection is appended to
the ActionDiT context for every denoising step.

## Training

Expected local assets:

- `data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot`
- `data/text_embeds_cache/libero`
- Wan2.2 TI2V 5B weights in the configured Hugging Face cache
- `checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`

Run the ten-step configuration check/smoke training:

```bash
bash scripts/train_zero1.sh 1 task=libero_spatial_leapbotce_smoke
```

Run the 2,000-step LIBERO-Spatial recipe:

```bash
bash scripts/train_zero1.sh 2 task=libero_spatial_leapbotce
```

Each logical sample uses a continuous same-episode window. Frame 20 is treated
as the current observation, a valid frame 1--20 environment steps earlier is
sampled for the cloud path, and the 32 actions beginning at frame 20 are the
shared fresh/stale target. The stale loss weight ramps from 0 to 0.5.

## Delay Evaluation

Evaluate one LIBERO-Spatial task at maximum delay windows 0, 5, 10, and 20:

```bash
bash scripts/eval_leapbotce_delay_sweep.sh /path/to/step_002000.pt 0
```

At every environment step, the evaluator stores the observation. At replan
time it samples the cloud observation uniformly from the available history;
the edge views and proprioception always come from the current observation.
Each result JSON includes the samples, mean, and histogram of realized delays.

The two-stage deployment API is:

```python
planning_cache = model.encode_cloud(delayed_image, prompt=prompt, proprio=proprio)
prediction = model.infer_action_edge(
    planning_cache, current_views, action_horizon=32
)
```

`model.infer_action(...)` remains a synchronous compatibility wrapper.
