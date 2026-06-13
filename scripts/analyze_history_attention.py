#!/usr/bin/env python3
"""Unified analysis of CasWAM history attention patterns.

Reads the output of capture_history_attention_rollout.py and generates
a comprehensive analysis report with 6 modules:
  1. Timeline: frames + attention heatmap per replan
  2. Key Nodes: top-5 attended positions + stability tracking
  3. Entropy: cross/self attention entropy curves
  4. Action Alignment: phase detection + frame-attention correlation
  5. Spatial: token positions mapped to image regions
  6. KV Probing: t-SNE + cosine similarity of latent space

Usage (CPU only):
    python scripts/analyze_history_attention.py \
        --data_dir evaluate_results/attention_analysis/.../trial_00/
"""

import argparse
import csv
import json
import math
from pathlib import Path
from collections import Counter

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from PIL import Image


# ── Data loading ─────────────────────────────────────────────────────────────

def load_rollout(data_dir: str):
    """Load all replan data from a trial directory.

    Expected structure (output of capture_history_attention_rollout.py):
        trial_XX/
        ├── success.json
        ├── obs_frames/replan_XXX.png
        ├── step_frames/replan_XXX_step_YYY.png
        ├── action_chunks/replan_XXX.npy
        ├── attention/replan_XXX.pt
        ├── kv_cache/replan_XXX.pt
        └── video_pred/replan_XXX/{frame_NNN.png, pred_video.mp4}

    Returns:
        meta: dict with task_name, success, num_replans, ...
        replans: list of dicts per replan
    """
    data_dir = Path(data_dir)

    # Load meta from success.json
    success_path = data_dir / "success.json"
    if success_path.exists():
        with open(success_path) as f:
            meta = json.load(f)
    else:
        meta = {"task_name": "unknown", "success": False, "num_replans": 0}

    obs_frames_dir = data_dir / "obs_frames"
    step_frames_dir = data_dir / "step_frames"
    action_chunks_dir = data_dir / "action_chunks"
    attention_dir = data_dir / "attention"
    kv_cache_dir = data_dir / "kv_cache"
    video_pred_dir = data_dir / "video_pred"

    # Find all replan indices from obs_frames
    obs_files = sorted(obs_frames_dir.glob("replan_*.png"))
    replan_indices = [int(f.stem.split("_")[-1]) for f in obs_files]

    replans = []
    tokens_per_frame = None

    for ridx in replan_indices:
        # Observation frame
        obs_path = obs_frames_dir / f"replan_{ridx:03d}.png"
        obs_frame = np.array(Image.open(obs_path)) if obs_path.exists() else None

        # Step frames
        step_frames = sorted(step_frames_dir.glob(f"replan_{ridx:03d}_step_*.png"))

        # Action chunk
        ac_path = action_chunks_dir / f"replan_{ridx:03d}.npy"
        action_chunk = np.load(ac_path) if ac_path.exists() else np.zeros((32, 7))

        # Attention data
        attn_path = attention_dir / f"replan_{ridx:03d}.pt"
        attention = None
        if attn_path.exists():
            attention = torch.load(attn_path, map_location="cpu", weights_only=False)

        # KV cache
        kv_path = kv_cache_dir / f"replan_{ridx:03d}.pt"
        kv_cache = None
        if kv_path.exists():
            kv_cache = torch.load(kv_path, map_location="cpu", weights_only=False)

        # Video prediction
        video_pred_path = video_pred_dir / f"replan_{ridx:03d}"
        has_video_pred = video_pred_path.exists()

        # Compute hist_len from attention tensor shape
        hist_len = 0
        if attention and "cross_attn" in attention:
            # cross_attn shape: [n_layers, S_q, S_k] (new) or list[n_heads, S_q, S_k] (legacy)
            ca = attention["cross_attn"]
            hist_len = ca.shape[-1] if not isinstance(ca, list) else ca[0].shape[-1]
            if tokens_per_frame is None and ridx >= 1:
                # At replan ridx, history has exactly `ridx` frames (R0..R(ridx-1)).
                # hist_len / ridx gives tokens per frame.
                tokens_per_frame = hist_len // ridx

        replans.append({
            "replan_idx": ridx,
            "hist_len": hist_len,
            "action_chunk": action_chunk,
            "obs_frame": obs_frame,
            "step_frames": step_frames,
            "attention": attention,
            "kv_cache": kv_cache,
            "has_video_pred": has_video_pred,
        })

    meta["tokens_per_frame"] = tokens_per_frame or 130  # fallback estimate

    return meta, replans


def get_captured_replans(replans):
    """Filter to replans that have attention data (hist_len > 0)."""
    return [r for r in replans if r["attention"] is not None]


# ── Helper functions ─────────────────────────────────────────────────────────

def avg_attention(attn, over="all"):
    """Average attention from compact format: [n_layers, S_q, S_k] (float16, head-averaged).

    Supports both new format (stacked tensor) and legacy format (list of per-head tensors).

    over: "all" -> [S_q, S_k], "layers" -> [n_layers, S_q, S_k]
    """
    layers = attn["cross_attn"]  # tensor [n_layers, S_q, S_k] or legacy list
    if isinstance(layers, list):
        # Legacy format: list of [n_heads, S_q, S_k]
        if over == "all":
            return torch.stack([l.float().mean(0) for l in layers]).mean(0)
        elif over == "layers":
            return torch.stack([l.float().mean(0) for l in layers])
        elif over == "heads":
            mid = layers[len(layers) // 2]
            return mid
    else:
        # New compact format: [n_layers, S_q, S_k]
        if over == "all":
            return layers.float().mean(0)  # [S_q, S_k]
        elif over == "layers":
            return layers.float()         # [n_layers, S_q, S_k]
        elif over == "heads":
            return layers[len(layers) // 2].float()  # middle layer [S_q, S_k]
    raise ValueError(over)


def avg_self_attention(attn, over="all"):
    """Same as avg_attention but for self-attention."""
    layers = attn["self_attn"]
    if isinstance(layers, list):
        if over == "all":
            return torch.stack([l.float().mean(0) for l in layers]).mean(0)
        return torch.stack([l.float().mean(0) for l in layers])
    else:
        if over == "all":
            return layers.float().mean(0)  # [S_hist, S_hist]
        return layers.float()


def entropy(p):
    """Entropy of attention distribution, averaged over query dim."""
    if p.ndim == 1:
        p = p.unsqueeze(0)
    p = p / (p.sum(dim=-1, keepdim=True) + 1e-8)
    log_p = torch.log(p + 1e-8)
    ent = -(p * log_p).sum(dim=-1)
    return ent.mean().item()


def detect_phases(action_chunk, velocity_threshold=0.05):
    """Detect action phase (REACH/GRASP/TRANSPORT/PLACE) from action trajectory."""
    n = len(action_chunk)
    velocities = np.diff(action_chunk[:, :3], axis=0)
    speeds = np.linalg.norm(velocities, axis=1)

    phases = []
    for i in range(n):
        if i == 0:
            sp = speeds[0] if len(speeds) > 0 else 0
        elif i == n - 1:
            sp = speeds[-1] if len(speeds) > 0 else 0
        else:
            sp = (speeds[i - 1] + speeds[i]) / 2 if i < len(speeds) else 0

        grip = action_chunk[i, -1] if action_chunk.shape[1] > 6 else 0.0

        if grip > 0.5:
            phases.append("GRASP" if i < n // 2 else "PLACE")
        elif sp > velocity_threshold:
            phases.append("REACH" if i < n // 2 else "TRANSPORT")
        else:
            phases.append("REACH" if i < n // 3 else "PLACE")
    return phases


def _write_csv(output_dir, filename, rows, headers):
    """Write a list-of-dicts to CSV."""
    path = Path(output_dir) / filename
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {path}")


# ── Module 1: Timeline ──────────────────────────────────────────────────────

def plot_timeline(meta, replans, output_dir):
    """Plot attention timeline: obs frames + cross-attention heatmaps."""
    captured = get_captured_replans(replans)
    if not captured:
        print("  Skipping Timeline: no captured replans")
        return

    n = len(captured)
    n_cols = min(8, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows * 2, n_cols,
                             figsize=(3 * n_cols, 5 * n_rows))
    if n_rows == 1:
        # axes is (2, n_cols); split into list of two 1D rows so
        # axes[row*2][col] returns a single Axes, not a row-slice.
        axes = [axes[0], axes[1]]
    elif n_rows * 2 == 2:
        axes = [axes[:n_cols], axes[n_cols:]]

    for idx, r in enumerate(captured):
        row, col = divmod(idx, n_cols)

        # Top row: obs frames
        ax = axes[row * 2][col]
        if r["obs_frame"] is not None:
            ax.imshow(r["obs_frame"])
        ax.set_title(f"R{r['replan_idx']} (h={r['hist_len']})", fontsize=9)
        ax.axis("off")

        # Bottom row: attention heatmap
        ax = axes[row * 2 + 1][col]
        if r["attention"]:
            ca = avg_attention(r["attention"])
            ax.imshow(ca.numpy(), cmap="hot", aspect="auto")
            ax.set_title(f"Cross-attn R{r['replan_idx']}", fontsize=9)
        ax.axis("off")

    for idx in range(n, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row * 2][col].axis("off")
        axes[row * 2 + 1][col].axis("off")

    fig.suptitle(f"Attention Timeline — {meta.get('task_name', 'unknown')}", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_timeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Module 2: Key Nodes ─────────────────────────────────────────────────────

def plot_key_nodes(meta, replans, output_dir, top_n=5):
    """Find and track the top-N attended history positions across replans."""
    captured = get_captured_replans(replans)
    if not captured:
        print("  Skipping Key Nodes: no captured replans")
        return [], []

    top_results = []

    for r in captured:
        ca = avg_attention(r["attention"])
        mean_attn = ca.mean(0)  # [S_k] (avg over action query tokens)
        # At replan R, history has exactly R frames (replans 0..R-1).
        # Use replan_idx as ground truth, chunk tokens evenly per frame.
        n_frames = r["replan_idx"]
        if n_frames == 0:
            continue
        S_k = mean_attn.shape[0]
        chunk = S_k // n_frames
        # Truncate to avoid remainder mismatch: n_frames * chunk <= S_k
        clipped = mean_attn[:n_frames * chunk]
        frame_attn = clipped.reshape(n_frames, chunk).mean(dim=1)  # [n_frames]
        k = min(top_n, n_frames)
        topk = torch.topk(frame_attn, k)
        top_frame_idxs = topk.indices.tolist()  # unique frame indices, sorted by attn
        top_results.append({
            "replan": r["replan_idx"],
            "top_frame_idxs": top_frame_idxs,
            "top_values": topk.values.tolist(),
        })

    # Stability: fraction of top frame indices that persist between consecutive replans
    stability = []
    for i in range(1, len(top_results)):
        prev = set(top_results[i - 1]["top_frame_idxs"])
        curr = set(top_results[i]["top_frame_idxs"])
        k = min(len(prev), len(curr), top_n)
        if k > 0:
            overlap = len(prev & curr) / k
            stability.append(overlap)

    # Plot top-N positions over replan index
    fig, ax = plt.subplots(figsize=(12, 5))
    for rank in range(top_n):
        xs = [tr["replan"] for tr in top_results]
        ys = [tr["top_frame_idxs"][rank] if rank < len(tr["top_frame_idxs"]) else None for tr in top_results]
        ys_filtered = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if ys_filtered:
            ax.plot([p[0] for p in ys_filtered], [p[1] for p in ys_filtered],
                    "o-", label=f"Rank {rank + 1}", markersize=5)

    ax.set_xlabel("Replan Index (R)")
    ax.set_ylabel("History Frame Index\n(= obs_frames/replan_XXX.png)")
    ax.set_title("Key Node Tracking — Which past frames does the model attend to most?")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_key_nodes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # CSV export: per-replan top-K frame ranking
    csv_rows = []
    for tr in top_results:
        for rank, (fidx, val) in enumerate(zip(
                tr["top_frame_idxs"], tr["top_values"])):
            csv_rows.append({
                "replan_idx": tr["replan"],
                "rank": rank + 1,
                "frame_idx": fidx,
                "attention": round(val, 8),
            })
    _write_csv(output_dir, "data_key_nodes.csv", csv_rows,
               ["replan_idx", "rank", "frame_idx", "attention"])

    return top_results, stability


# ── Module 2b: Global Top-Frame Gallery ─────────────────────────────────────

def plot_global_top_frames(meta, replans, output_dir, data_dir, top_n=5):
    """Rank ALL history frames by mean attention across the ENTIRE episode.

    For each history frame f, accumulate attention from all replans R where f < R
    (frame f appears in history only from replan f+1 onward). Normalize by the
    number of replans it was visible to. Show actual images of the global top-N.
    """
    captured = get_captured_replans(replans)
    if len(captured) < 2:
        print("  Skipping Global Top-Frames: not enough replans")
        return

    # Determine max frame index (last replan's replan_idx)
    max_frame = max(r["replan_idx"] for r in captured)
    if max_frame == 0:
        return

    # Accumulators: sum of attention, count of replans frame was visible
    attn_sum = np.zeros(max_frame, dtype=np.float64)
    attn_count = np.zeros(max_frame, dtype=np.int32)

    for r in captured:
        ca = avg_attention(r["attention"])
        mean_attn = ca.mean(0)  # [S_k]
        n_frames = r["replan_idx"]
        if n_frames == 0:
            continue
        S_k = mean_attn.shape[0]
        chunk = S_k // n_frames
        # Chunk-and-average: one score per history frame
        clipped = mean_attn[:n_frames * chunk]
        frame_attn = clipped.reshape(n_frames, chunk).mean(dim=1).numpy()  # [n_frames]
        # Accumulate: frame f gets this replan's attention score
        for f_idx in range(n_frames):
            attn_sum[f_idx] += frame_attn[f_idx]
            attn_count[f_idx] += 1

    # Average attention per frame (avoid div-by-zero)
    valid_mask = attn_count > 0
    mean_attn_per_frame = np.where(valid_mask, attn_sum / attn_count, 0.0)

    # Global top-N
    k_global = min(top_n, max_frame)
    global_top_indices = np.argsort(mean_attn_per_frame)[::-1][:k_global]

    obs_frames_dir = Path(data_dir) / "obs_frames"
    n_cols = min(5, k_global)
    n_rows_img = (k_global + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(3.5 * n_cols, 4 + 2.8 * n_rows_img))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 2.8 * n_rows_img / 4],
                          hspace=0.35)

    # Top: bar chart
    ax_bar = fig.add_subplot(gs[0])
    colors = ["#FF5722" if i in global_top_indices else "#90CAF9"
              for i in range(max_frame)]
    ax_bar.bar(range(max_frame), mean_attn_per_frame, color=colors, edgecolor="white")
    ax_bar.set_xlabel("History Frame Index")
    ax_bar.set_ylabel("Mean Cross-Attention")
    ax_bar.set_title(f"Global Frame Importance (avg over {len(captured)} replans)")
    ax_bar.grid(True, alpha=0.3, axis="y")

    # Bottom: image gallery of top-N frames
    gs_img = gs[1].subgridspec(n_rows_img, n_cols, wspace=0.08, hspace=0.3)
    for rank, fidx in enumerate(global_top_indices):
        row, col = divmod(rank, n_cols)
        ax = fig.add_subplot(gs_img[row, col])

        obs_path = obs_frames_dir / f"replan_{fidx:03d}.png"
        if obs_path.exists():
            img = np.array(Image.open(obs_path))
            ax.imshow(img)
        ax.set_title(f"#{rank + 1}  R{fidx}  ({mean_attn_per_frame[fidx]:.4f})",
                     fontsize=9, color="#FF5722")
        ax.axis("off")

    fig.suptitle(f"Global Frame Attention Ranking — {meta.get('task_name', 'unknown')}",
                 fontsize=14, y=0.98)
    fig.savefig(output_dir / "fig_global_top_frames.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Print summary
    print(f"  Global top-{k_global} frames (by mean cross-attention):")
    for rank, fidx in enumerate(global_top_indices):
        print(f"    #{rank + 1}: frame R{fidx}  score={mean_attn_per_frame[fidx]:.6f}")

    # CSV: global frame ranking
    csv_rows = []
    for fidx in range(max_frame):
        rank = np.where(global_top_indices == fidx)[0]
        csv_rows.append({
            "frame_idx": fidx,
            "mean_attention": round(float(mean_attn_per_frame[fidx]), 8),
            "n_replans_visible": int(attn_count[fidx]),
            "rank": int(rank[0]) + 1 if len(rank) > 0 else -1,
            "is_top_k": int(len(rank) > 0),
        })
    _write_csv(output_dir, "data_global_top_frames.csv", csv_rows,
               ["frame_idx", "mean_attention", "n_replans_visible", "rank", "is_top_k"])

    return global_top_indices.tolist(), mean_attn_per_frame[global_top_indices].tolist()


# ── Module 2c: Relative Position Attention Profile ───────────────────────────

def plot_relative_position_attention(meta, replans, output_dir):
    """Average attention by RELATIVE position in history (0=oldest, N-1=newest).

    For each replan, extract per-frame attention, then align by relative position
    (normalizing for different history lengths). Average across all replans.
    """
    captured = get_captured_replans(replans)
    if len(captured) < 2:
        print("  Skipping Relative Position: not enough replans")
        return

    # Collect per-replan relative attention profiles
    profiles = []  # list of (n_frames,) arrays

    for r in captured:
        ca = avg_attention(r["attention"])
        mean_attn = ca.mean(0)  # [S_k]
        n_frames = r["replan_idx"]
        if n_frames <= 1:
            continue
        S_k = mean_attn.shape[0]
        chunk = S_k // n_frames
        clipped = mean_attn[:n_frames * chunk]
        frame_attn = clipped.reshape(n_frames, chunk).mean(dim=1).numpy()
        profiles.append(frame_attn)

    if not profiles:
        return

    # Pad to same length (max frames) and compute mean ± std
    max_f = max(len(p) for p in profiles)
    padded = np.full((len(profiles), max_f), np.nan)
    for i, p in enumerate(profiles):
        padded[i, :len(p)] = p

    with np.errstate(all="ignore"):
        mean_profile = np.nanmean(padded, axis=0)
        sem_profile = np.nanstd(padded, axis=0) / np.sqrt(
            np.maximum(1, np.sum(~np.isnan(padded), axis=0)))

    # Plot
    fig, ax = plt.subplots(figsize=(12, 4))
    xs = np.arange(max_f)
    ax.fill_between(xs, mean_profile - sem_profile, mean_profile + sem_profile,
                    alpha=0.2, color="#FF5722")
    ax.plot(xs, mean_profile, "o-", color="#FF5722", lw=2, markersize=6,
            label="Mean attention (across replans)")
    ax.axvline(x=max_f - 1, color="#4CAF50", linestyle="--", alpha=0.6,
               label=f"Newest frame (R{max_f - 1})")
    ax.set_xlabel("Relative Position in History  (0 = oldest → N-1 = newest)")
    ax.set_ylabel("Mean Cross-Attention")
    ax.set_title(f"Attention by Relative History Position "
                 f"(avg over {len(profiles)} replans, N={max_f} max frames)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_relative_position.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # CSV: relative position attention profile
    csv_rows = []
    for pos in range(max_f):
        n_visible = int(np.sum(~np.isnan(padded[:, pos])))
        csv_rows.append({
            "relative_position": pos,
            "mean_attention": round(float(mean_profile[pos]), 8) if not np.isnan(mean_profile[pos]) else "",
            "sem": round(float(sem_profile[pos]), 8) if not np.isnan(sem_profile[pos]) else "",
            "n_replans": n_visible,
        })
    _write_csv(output_dir, "data_relative_position.csv", csv_rows,
               ["relative_position", "mean_attention", "sem", "n_replans"])

    # Print
    newest_attn = mean_profile[-1]
    oldest_attn = mean_profile[0]
    print(f"  Oldest frame (pos 0) mean attn: {oldest_attn:.6f}")
    print(f"  Newest frame (pos {max_f - 1}) mean attn: {newest_attn:.6f}")
    print(f"  Newest / oldest ratio: {newest_attn / oldest_attn:.2f}x")


# ── Module 2d: Per-Action-Position Frame-Level Attention ────────────────────

def plot_action_frame_attention(replans, output_dir, data_dir, top_n=5):
    """Per-replan heatmaps: for each action position, which HISTORY frames matter most.

    Samples 3 replans (early, mid, late) and shows per-replan action×frame
    heatmaps + aggregate across all replans.
    """
    captured = get_captured_replans(replans)
    if len(captured) < 2:
        print("  Skipping Action-Frame: not enough replans")
        return

    obs_frames_dir = Path(data_dir) / "obs_frames"
    action_horizon = None
    max_frame = max(r["replan_idx"] for r in captured)

    # ── Aggregate accumulator (for global mean) ──
    acc_sum = None
    acc_cnt = None
    # ── Per-replan snapshots ──
    snapshots = {}  # replan_idx → [action_horizon, n_frames]

    for r in captured:
        attn = r["attention"]
        if attn is None or "cross_attn" not in attn:
            continue
        ca = attn["cross_attn"]
        if isinstance(ca, list):
            ca = torch.stack([t.mean(0) for t in ca])
        A = ca.float().mean(0)  # [S_q, S_k]

        if action_horizon is None:
            action_horizon = A.shape[0]
            acc_sum = np.zeros((action_horizon, max_frame), dtype=np.float64)
            acc_cnt = np.zeros((action_horizon, max_frame), dtype=np.int32)

        if A.shape[0] != action_horizon:
            continue

        n_frames = r["replan_idx"]
        if n_frames == 0:
            continue
        S_k = A.shape[1]
        chunk = S_k // n_frames
        if chunk == 0:
            continue

        mat = np.zeros((action_horizon, n_frames), dtype=np.float64)
        for p in range(action_horizon):
            attn_p = A[p, :n_frames * chunk].numpy()
            frame_attn = attn_p.reshape(n_frames, chunk).mean(axis=1)
            mat[p] = frame_attn
            acc_sum[p, :n_frames] += frame_attn
            acc_cnt[p, :n_frames] += 1

        # Save per-replan snapshot (only for sampled replans)
        snapshots[r["replan_idx"]] = mat

    if action_horizon is None or acc_sum is None:
        print("  Skipping Action-Frame: no valid data")
        return

    # Mean attention per frame per action position (aggregate)
    with np.errstate(all="ignore"):
        agg_mean = acc_sum / np.maximum(acc_cnt, 1)

    # ── Select sample replans: early, mid, late ──
    all_ridx = sorted(snapshots.keys())
    n_sample = min(3, len(all_ridx))
    sample_ridx = [all_ridx[0], all_ridx[len(all_ridx) // 2], all_ridx[-1]][:n_sample]
    sample_ridx = list(dict.fromkeys(sample_ridx))  # dedup

    # ── Plot ──
    n_panels = 1 + n_sample  # aggregate + per-replan
    fig = plt.figure(figsize=(16, 3 + 0.35 * action_horizon * n_panels))

    def _plot_heatmap(mat, ax, title_prefix, n_frames_display):
        """Plot a single action×frame heatmap with proper labels."""
        im = ax.imshow(mat, aspect="auto", cmap="hot", interpolation="nearest")
        ax.set_xlabel("History Frame  (0=oldest → newest)")
        ax.set_ylabel("Action Pos")
        ax.set_title(title_prefix)
        ax.set_yticks(range(action_horizon))
        ax.set_yticklabels([f"A{p}" for p in range(action_horizon)], fontsize=6)
        max_label = max(1, n_frames_display - 1)
        step_x = max(1, n_frames_display // 8)
        ax.set_xticks(range(0, n_frames_display, step_x))
        return im

    # Panel 1: Aggregate
    ax_agg = fig.add_subplot(n_panels, 1, 1)
    _plot_heatmap(agg_mean, ax_agg,
                  f"AGGREGATE  (avg over {len(captured)} replans)", max_frame)
    plt.colorbar(plt.cm.ScalarMappable(cmap="hot"), ax=ax_agg, shrink=0.6,
                 label="Mean Cross-Attention")

    # Panels 2..N: Per-replan
    for pi, ridx in enumerate(sample_ridx):
        mat = snapshots[ridx]
        nf = mat.shape[1]
        ax = fig.add_subplot(n_panels, 1, pi + 2)
        _plot_heatmap(mat, ax, f"R{ridx}  (history: {nf} frames)", nf)

    fig.tight_layout()
    fig.savefig(output_dir / "fig_action_frame.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # CSV: aggregate action_position × history_frame matrix
    csv_rows = []
    for p in range(action_horizon):
        row = {"action_pos": p}
        for f in range(max_frame):
            row[f"frame_{f}"] = round(float(agg_mean[p, f]), 8)
        csv_rows.append(row)
    headers = ["action_pos"] + [f"frame_{f}" for f in range(max_frame)]
    _write_csv(output_dir, "data_action_frame_aggregate.csv", csv_rows, headers)

    # CSV: per-replan action × frame matrices
    for ridx in sample_ridx:
        mat = snapshots[ridx]
        nf = mat.shape[1]
        csv_rows = []
        for p in range(action_horizon):
            row = {"action_pos": p}
            for f in range(nf):
                row[f"frame_{f}"] = round(float(mat[p, f]), 8)
            csv_rows.append(row)
        headers = ["action_pos"] + [f"frame_{f}" for f in range(nf)]
        _write_csv(output_dir, f"data_action_frame_replan_{ridx:03d}.csv",
                   csv_rows, headers)

    # Print per-replan top frames
    k = min(top_n, max_frame)
    print(f"  Aggregate top-{k} history frames per action position:")
    for p in range(min(5, action_horizon)):
        order = np.argsort(agg_mean[p])[::-1][:k]
        print(f"    A{p}: {order.tolist()}")
    if action_horizon > 5:
        print(f"    ... (showing first 5 of {action_horizon} positions)")

    for ridx in sample_ridx:
        mat = snapshots[ridx]
        k_r = min(top_n, mat.shape[1])
        print(f"  R{ridx} top-{k_r} history frames per action position:")
        for p in range(min(5, action_horizon)):
            order = np.argsort(mat[p])[::-1][:k_r]
            print(f"    A{p}: {order.tolist()}")
        if action_horizon > 5:
            print(f"    ... (showing first 5 of {action_horizon} positions)")


# ── Module 3: Entropy ────────────────────────────────────────────────────────

def plot_entropy(meta, replans, output_dir):
    """Plot cross-attention and self-attention entropy over replans."""
    captured = get_captured_replans(replans)
    if len(captured) < 2:
        print("  Skipping Entropy: need >= 2 replans with attention")
        return

    xs = [r["replan_idx"] for r in captured]
    cross_ent = [entropy(avg_attention(r["attention"])) for r in captured]
    self_ent = [entropy(avg_self_attention(r["attention"])) for r in captured]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(xs, cross_ent, "o-", color="#1976D2", label="Cross-attention")
    ax1.set_xlabel("Replan Index")
    ax1.set_ylabel("Entropy (nats)")
    ax1.set_title("Cross-Attention Entropy")
    ax1.grid(True, alpha=0.3)

    ax2.plot(xs, self_ent, "o-", color="#E91E63", label="Self-attention")
    ax2.set_xlabel("Replan Index")
    ax2.set_ylabel("Entropy (nats)")
    ax2.set_title("Self-Attention Entropy")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Attention Entropy Over Replans", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_entropy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # CSV: entropy per replan
    csv_rows = []
    for x, ce, se in zip(xs, cross_ent, self_ent):
        csv_rows.append({
            "replan_idx": x,
            "cross_entropy": round(ce, 6),
            "self_entropy": round(se, 6),
        })
    _write_csv(output_dir, "data_entropy.csv", csv_rows,
               ["replan_idx", "cross_entropy", "self_entropy"])


# ── Module 4: Action Alignment ──────────────────────────────────────────────

def plot_action_alignment(meta, replans, output_dir):
    """Correlate attention with action phases."""
    captured = get_captured_replans(replans)
    if not captured:
        print("  Skipping Action Alignment: no captured replans")
        return

    tpf = meta["tokens_per_frame"]
    fig, axes = plt.subplots(len(captured), 1, figsize=(14, 3 * len(captured)), sharex=False)
    if len(captured) == 1:
        axes = [axes]

    phase_colors = {
        "REACH": "#2196F3", "GRASP": "#FF9800",
        "TRANSPORT": "#4CAF50", "PLACE": "#9C27B0",
    }

    for idx, r in enumerate(captured):
        ax = axes[idx]
        ca = avg_attention(r["attention"])
        hist_attn = ca.mean(0).numpy()  # [S_k] (avg over action query tokens)
        mean_hist = hist_attn            # already [S_k]

        phases = detect_phases(r["action_chunk"])
        n_tokens = len(mean_hist)
        n_frames = n_tokens // tpf

        # Plot attention over frames
        frame_attn = [mean_hist[f * tpf: (f + 1) * tpf].mean() for f in range(n_frames)]
        ax.plot(range(n_frames), frame_attn, "k-", lw=1.5, label="Mean attn")

        # Color by phase
        for step_i, phase in enumerate(phases[:n_frames]):
            ax.axvspan(step_i - 0.5, step_i + 0.5, alpha=0.15, color=phase_colors[phase])

        ax.set_ylabel("Attention")
        ax.set_title(f"R{r['replan_idx']} (h={r['hist_len']})")
        ax.set_xlabel("Frame")
        ax.annotate(
            f"Action: {phases[:4]}...",
            xy=(0.02, 0.9), xycoords="axes fraction",
            fontsize=9, va="top", family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    phase_patches = [mpatches.Patch(color=phase_colors[p], alpha=0.4, label=p)
                     for p in ["REACH", "GRASP", "TRANSPORT", "PLACE"]]
    fig.legend(handles=phase_patches, loc="upper right", fontsize=8)
    fig.suptitle(f"Action-Attention Alignment — {meta.get('task_name', 'unknown')}", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_action_alignment.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # CSV: per-frame attention with action phases
    csv_rows = []
    tpf = meta["tokens_per_frame"]
    for r in captured:
        ca = avg_attention(r["attention"])
        hist_attn = ca.mean(0).numpy()
        n_tokens = len(hist_attn)
        n_frames = n_tokens // tpf
        phases = detect_phases(r["action_chunk"])
        for f in range(n_frames):
            frame_attn = hist_attn[f * tpf: (f + 1) * tpf].mean()
            phase = phases[f] if f < len(phases) else "UNKNOWN"
            csv_rows.append({
                "replan_idx": r["replan_idx"],
                "frame_idx": f,
                "attention": round(float(frame_attn), 8),
                "phase": phase,
            })
    _write_csv(output_dir, "data_action_alignment.csv", csv_rows,
               ["replan_idx", "frame_idx", "attention", "phase"])


# ── Module 5: Spatial ───────────────────────────────────────────────────────

def discover_grid(tokens_per_frame):
    """Find (h, w) spatial grid dimensions from token count."""
    h = int(np.sqrt(tokens_per_frame))
    if h * h == tokens_per_frame:
        return h, h
    best = (1, tokens_per_frame)
    best_diff = float("inf")
    for h in range(1, tokens_per_frame + 1):
        if tokens_per_frame % h == 0:
            w = tokens_per_frame // h
            if abs(h - w) < best_diff:
                best_diff = abs(h - w)
                best = (h, w)
    return best


def plot_spatial(meta, replans, output_dir):
    """Plot spatial attention maps (which image regions are attended to)."""
    captured = get_captured_replans(replans)
    tpf = meta["tokens_per_frame"]
    gh, gw = discover_grid(tpf)

    # Use last replan (most history)
    last = captured[-1]
    ca = avg_attention(last["attention"])
    hist_attn = ca.mean(0).numpy()

    # Per-frame spatial maps
    nf = len(hist_attn) // tpf
    if nf == 0:
        return

    n_cols = min(6, nf)
    n_rows = (nf + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = [axes]

    # Global range for consistent coloring
    all_maps = []
    for f in range(nf):
        sm = hist_attn[f * tpf: f * tpf + gh * gw].reshape(gh, gw)
        all_maps.append(sm)
    vmin = min(m.min() for m in all_maps)
    vmax = max(m.max() for m in all_maps)

    for f in range(nf):
        row, col = divmod(f, n_cols)
        ax = axes[row][col] if n_rows > 1 else axes[col]
        im = ax.imshow(all_maps[f], cmap="hot", interpolation="bilinear", vmin=vmin, vmax=vmax)
        ax.set_title(f"Frame {f}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)
    for f in range(nf, n_rows * n_cols):
        row, col = divmod(f, n_cols)
        ax = axes[row][col] if n_rows > 1 else axes[col]
        ax.axis("off")

    fig.suptitle(f"Spatial Attention (Replan {last['replan_idx']}, Grid {gh}×{gw})", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_spatial.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # CSV: per-token spatial attention (last replan only)
    csv_rows = []
    for f in range(nf):
        sm = all_maps[f]
        for r in range(gh):
            for c in range(gw):
                csv_rows.append({
                    "frame_idx": f,
                    "row": r,
                    "col": c,
                    "attention": round(float(sm[r, c]), 8),
                })
    _write_csv(output_dir, "data_spatial.csv", csv_rows,
               ["frame_idx", "row", "col", "attention"])


# ── Module 6: KV Probing ────────────────────────────────────────────────────

def plot_kv_probing(meta, replans, output_dir):
    """t-SNE + cosine similarity analysis of KV cache latent space."""
    captured = get_captured_replans(replans)
    # Find last replan with KV data
    kv_replans = [r for r in captured if r["kv_cache"] is not None]
    if not kv_replans:
        print("  Skipping KV probing: no kv_cache.pt found")
        return

    tpf = meta["tokens_per_frame"]
    last = kv_replans[-1]

    # kv_cache is list of (K, V) per layer. Use last layer.
    last_layer_kv = last["kv_cache"][-1]
    raw_k = last_layer_kv[0]  # [S_hist, hidden_dim]
    n_tokens = raw_k.shape[0]
    n_frames = n_tokens // tpf if tpf > 0 else n_tokens

    try:
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
    except ImportError:
        print("  Skipping KV probing: scikit-learn not installed (pip install scikit-learn)")
        return

    # t-SNE
    perplexity = min(30, max(5, n_tokens // 4))
    print(f"  Running t-SNE on {n_tokens} tokens (perplexity={perplexity})...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    tsne_2d = tsne.fit_transform(raw_k.numpy())

    # Attention weights for coloring
    ca = avg_attention(last["attention"])
    mean_attn = ca.mean(0).numpy()
    # Pad or truncate to match n_tokens
    if len(mean_attn) < n_tokens:
        mean_attn = np.concatenate([mean_attn, np.zeros(n_tokens - len(mean_attn))])
    elif len(mean_attn) > n_tokens:
        mean_attn = mean_attn[:n_tokens]

    frame_ids = np.array([i // tpf for i in range(n_tokens)])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: colored by frame
    sc1 = ax1.scatter(tsne_2d[:, 0], tsne_2d[:, 1], c=frame_ids, cmap="viridis",
                      s=20, alpha=0.7, edgecolors="black", linewidths=0.3)
    ax1.set_title(f"t-SNE: Frame Index (R{last['replan_idx']})")
    plt.colorbar(sc1, ax=ax1, label="Frame")

    # Right: colored by attention
    sc2 = ax2.scatter(tsne_2d[:, 0], tsne_2d[:, 1], c=mean_attn, cmap="hot",
                      s=20, alpha=0.7, edgecolors="black", linewidths=0.3)
    ax2.set_title(f"t-SNE: Attention Weight (R{last['replan_idx']})")
    plt.colorbar(sc2, ax=ax2, label="Mean attn")

    # Top-5 markers
    top5 = np.argsort(mean_attn)[-5:][::-1]
    ax2.scatter(tsne_2d[top5, 0], tsne_2d[top5, 1], s=200,
                facecolors="none", edgecolors="cyan", linewidths=3)

    fig.suptitle("KV Cache Latent Space (t-SNE)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_kv_probing.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Cosine similarity heatmap
    raw_norm = raw_k / (raw_k.norm(dim=-1, keepdim=True) + 1e-8)
    sim = (raw_norm @ raw_norm.T).numpy()

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title(f"History Token Cosine Similarity (R{last['replan_idx']})")
    for f in range(1, n_frames):
        ax.axhline(y=f * tpf, color="white", lw=0.5, alpha=0.5)
        ax.axvline(x=f * tpf, color="white", lw=0.5, alpha=0.5)
    plt.colorbar(im, ax=ax, label="Cosine similarity")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_cosine_similarity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # CSV: t-SNE coordinates with attention and frame labels
    csv_rows = []
    for i in range(n_tokens):
        csv_rows.append({
            "token_idx": i,
            "tsne_x": round(float(tsne_2d[i, 0]), 6),
            "tsne_y": round(float(tsne_2d[i, 1]), 6),
            "frame_idx": int(frame_ids[i]),
            "attention": round(float(mean_attn[i]), 8),
        })
    _write_csv(output_dir, "data_kv_tsne.csv", csv_rows,
               ["token_idx", "tsne_x", "tsne_y", "frame_idx", "attention"])

    # CSV: cosine similarity matrix (compact: keep only off-diagonal for large n_tokens)
    if n_tokens <= 5000:
        csv_rows = []
        for i in range(n_tokens):
            for j in range(i + 1, n_tokens):
                csv_rows.append({
                    "token_i": i,
                    "token_j": j,
                    "cosine_sim": round(float(sim[i, j]), 6),
                    "frame_i": int(frame_ids[i]),
                    "frame_j": int(frame_ids[j]),
                })
        _write_csv(output_dir, "data_kv_cosine_sim.csv", csv_rows,
                   ["token_i", "token_j", "cosine_sim", "frame_i", "frame_j"])


# ── Text Report ─────────────────────────────────────────────────────────────

def generate_report(meta, replans, top_results, stability, output_dir):
    """Generate text summary of all analysis findings."""
    captured = get_captured_replans(replans)
    tpf = meta["tokens_per_frame"]
    lines = []
    lines.append("=" * 70)
    lines.append("History Attention Analysis Report")
    lines.append("=" * 70)
    lines.append(f"Task: {meta.get('task_name', 'unknown')}")
    lines.append(f"Success: {meta.get('success', 'N/A')}")
    lines.append(f"Replans: {meta.get('num_replans', len(replans))} ({len(captured)} with attention)")
    lines.append(f"Tokens/frame: {tpf}")
    lines.append("")

    # Key nodes
    if top_results:
        lines.append("--- Key Nodes ---")
        lines.append("  (frame index N = obs_frames/replan_NNN.png, as stored in history KV cache)")
        for tr in top_results:
            top_frames = tr["top_frame_idxs"]
            lines.append(f"  R{tr['replan']}: top frames = {top_frames}")
        if stability:
            avg_stab = np.mean(stability)
            lines.append(f"  Average stability: {avg_stab:.1%}")
        lines.append("")

    # Entropy summary
    if len(captured) >= 2:
        lines.append("--- Entropy ---")
        ent_first = entropy(avg_attention(captured[0]["attention"]))
        ent_last = entropy(avg_attention(captured[-1]["attention"]))
        lines.append(f"  First replan: {ent_first:.2f} nats")
        lines.append(f"  Last replan:  {ent_last:.2f} nats")
        lines.append(f"  Change: {ent_last - ent_first:+.2f} nats")
        lines.append("")

    # Action alignment summary
    lines.append("--- Action Alignment ---")
    for r in captured[:8]:
        phases = detect_phases(r["action_chunk"])
        dominant = Counter(phases).most_common(1)[0][0]
        lines.append(f"  R{r['replan_idx']}: {dominant} ({Counter(phases)})")
    lines.append("")

    # Video prediction availability
    video_replans = [r for r in captured if r.get("has_video_pred")]
    if video_replans:
        lines.append("--- Video Predictions ---")
        lines.append(f"  {len(video_replans)} replans have decoded video predictions")
        lines.append(f"  Located in: video_pred/replan_XXX/")
        lines.append("")

    lines.append("=" * 70)
    report = "\n".join(lines)
    (output_dir / "analysis_report.txt").write_text(report)
    print(report)


# ── Main ─────────────────────────────────────────────────────────────────────

def run_analysis(args):
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.data_dir) / "analysis_top{}".format(args.top_n)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {args.data_dir}...")
    meta, replans = load_rollout(args.data_dir)
    captured = get_captured_replans(replans)

    print(f"Task: {meta.get('task_name', 'unknown')}")
    print(f"Success: {meta.get('success', 'N/A')}")
    print(f"Replans: {len(replans)} total, {len(captured)} with attention")
    print(f"Tokens/frame: {meta['tokens_per_frame']}")
    print(f"Output: {output_dir}")
    print()

    if not captured:
        print("No replans with attention data found!")
        return

    # Module 1: Timeline
    print("Module 1: Timeline...")
    plot_timeline(meta, replans, output_dir)

    # Module 2: Key Nodes
    print("Module 2: Key Nodes...")
    top_results, stability = plot_key_nodes(meta, replans, output_dir, top_n=args.top_n)

    # Module 2b: Global Top-Frames (most important frames across entire episode)
    print("Module 2b: Global Top-Frames...")
    plot_global_top_frames(meta, replans, output_dir, args.data_dir, top_n=args.top_n)

    # Module 2c: Relative Position Attention (oldest → newest position profile)
    print("Module 2c: Relative Position...")
    plot_relative_position_attention(meta, replans, output_dir)

    # Module 2d: Per-Action-Frame Attention (which action frame attends most)
    print("Module 2d: Action-Frame Attention...")
    plot_action_frame_attention(replans, output_dir, data_dir=args.data_dir, top_n=args.top_n)

    # Module 3: Entropy
    print("Module 3: Entropy...")
    plot_entropy(meta, replans, output_dir)

    # Module 4: Action Alignment
    print("Module 4: Action Alignment...")
    plot_action_alignment(meta, replans, output_dir)

    # Module 5: Spatial
    print("Module 5: Spatial...")
    plot_spatial(meta, replans, output_dir)

    # Module 6: KV Probing
    print("Module 6: KV Probing...")
    plot_kv_probing(meta, replans, output_dir)

    # Text report
    print("Generating report...")
    generate_report(meta, replans, top_results, stability, output_dir)

    print(f"\n=== Done! All analysis saved to {output_dir}/ ===")
    for f in sorted(output_dir.iterdir()):
        print(f"  {f.name}")


def main():
    parser = argparse.ArgumentParser(description="Unified CasWAM attention analysis")
    parser.add_argument("--data_dir", required=True, help="Trial directory (e.g., trial_00/)")
    parser.add_argument("--output_dir", default=None, help="Analysis output dir (default: data_dir/analysis)")
    parser.add_argument("--top_n", type=int, default=5, help="Number of top key positions")
    args = parser.parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
