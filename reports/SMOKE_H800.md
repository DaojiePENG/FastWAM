# LeapBot-VA H800 smoke report

Date: 2026-07-30
GPU: NVIDIA H800 80 GB
Checkpoint: FastWAM `libero_uncond_2cam224.pt`
Precision: BF16
Image layout: two 224×224 cameras concatenated to 224×448

## Verified results

| Test | Result | Peak allocated GPU memory |
|---|---:|---:|
| 6B causal-history training, one forward/backward step, D30 | loss 0.09710 | 23.85 GiB |
| Action inference, D30, 20 denoise steps, one block | 1.397 s total | 12.71 GiB |
| Full 70-block KV episode, D30, one denoise step per replan | 2.596 GiB persistent KV | 17.90 GiB |

The 20-step action-only latency split was:

- real-observation VAE + KV prefill: 0.395 s;
- ActionDiT denoising: 0.936 s;
- executed 10-action KV commit: 0.065 s.

The 70-block run completed all 700 committed actions. Its one-step-denoise
replan latency was P50 0.147 s and P95 0.203 s; this run is a cache/capacity
stress test and is not a production 20-step latency measurement. Block 71 was
rejected by the configured capacity guard.

For both inference runs, the video output head, all shallow video exit heads,
and VAE decode were replaced by functions that raise immediately. The run
completed, confirming these paths were not called. Only the real input image
was VAE-encoded.

The underlying JSON outputs are committed beside this report. These are smoke
results, not LIBERO success-rate results; final success/latency/memory Pareto
curves require post-trained LeapBot checkpoints and the planned 500 episodes
per configuration.
