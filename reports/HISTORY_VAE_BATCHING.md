# History VAE batching acceptance

Date: 2026-07-30  
GPU: NVIDIA H800 80 GB  
Checkpoint: FastWAM `libero_uncond_2cam224.pt`  
Precision: BF16  
Sample: two real LIBERO trajectories at exact history length H=8

Each history observation remained an independent `T=1` video. The reference
used Wan's public per-observation encoder. Candidate batching changed only the
independent batch axis; no history observations were concatenated in time.

| Candidate | Latent max abs | Latent RMSE | Total loss delta | Video loss delta | Action loss delta | Decision |
|---|---:|---:|---:|---:|---:|---|
| chunk=2 | 0.0234375 | 0.0020112 | 0.0001020 | 0.0001302 | 0.0002322 | accepted |
| chunk=4 | 0.0312500 | 0.0021505 | 0.0018234 | 0.0001847 | 0.0016388 | rejected |

The fixed-noise complete-loss acceptance threshold was `1e-3`. Chunk 2 passed;
chunk 4 did not. In this diagnostic, chunk 2 reduced the full fixed-noise
forward from 0.529 s to 0.252 s while keeping the same peak operation allocation
(0.658 GiB above the loaded model). The timing is a microbenchmark, not an
end-to-end training throughput claim.

The production default is therefore chunk 2. Chunk 1 remains the strict
per-observation fallback, and chunk 4 is not permitted in the formal comparison.
