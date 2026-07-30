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

## Complete-prefix training acceptance (`e58db5e`)

The corrected production path was additionally tested with BF16, per-GPU batch
2, D30, full BPTT, ActionDiT full training, VideoDiT rank-16 LoRA, and the real
FastWAM release checkpoint:

| Prefix/mode | Forward | Backward | Peak allocated | Parameters with finite gradient |
|---|---:|---:|---:|---:|
| real H50, `action_aggregator` | 1.531 s | 1.447 s | 54.536 GiB | 1,033,374,727 |
| real H50, `interleaved` | 1.922 s | 1.448 s | 54.536 GiB | 1,033,374,727 |
| real H50, `vision_causal` | 1.762 s | 1.445 s | 54.536 GiB | 1,033,374,727 |
| synthetic capacity H70, `action_aggregator` | 3.329 s | 2.328 s | 70.361 GiB | 1,033,374,727 |

H50 is the longest real prefix in the released training split. The H70 row
repeats H50's final real block only to exercise shapes, masks, gradients, and
OOM margin; its loss is not an effect metric. At H70/B2 the 80 GiB H800 retained
about 9.6 GiB of allocated-memory headroom.

At the release initialization, H50 video loss was 0.0568 for
`action_aggregator`, whose video path intentionally remains block-independent,
and about 1.07--1.08 for the two modes whose video expert consumes an unadapted
long prefix. This is an expected initialization effect and is why each causal
mode receives its own controlled post-training run.
