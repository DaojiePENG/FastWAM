"""Capture full episode rollouts with attention weights, KV cache, and full-denoised predictions.

Follows the pattern of eval_libero_single.py:
  - Runs full episodes (until done or max steps)
  - Supports --num_trials for repeated runs on same task
  - Saves everything for offline analysis (similar to evaluation result structure)

Output structure:
    evaluate_results/attention_analysis/{config}/{timestamp}/
    ├── videos/
    │   ├── trial_00.mp4          (full episode video)
    │   └── trial_01.mp4
    ├── trial_00/
    │   ├── success.json          {"success": true/false, "num_replans": N}
    │   ├── obs_frames/           observation at each replan (before model prediction)
    │   │   ├── replan_000.png
    │   │   └── ...
    │   ├── step_frames/          observation after each env.step() (during action execution)
    │   │   ├── replan_000_step_000.png ...
    │   │   └── ...
    │   ├── action_chunks/        predicted action chunks
    │   │   ├── replan_000.npy
    │   │   └── ...
    │   ├── attention/            attention weights per replan
    │   │   ├── replan_000.pt
    │   │   └── ...
    │   ├── kv_cache/             first-step KV cache per replan
    │   │   ├── replan_000.pt
    │   │   └── ...
    │   └── video_pred/           fully-denoised video predictions (decoded PIL frames)
    │       ├── replan_000/
    │       │   ├── frame_000.png ... frame_NNN.png
    │       │   └── pred_video.mp4
    │       └── ...
    ├── trial_01/
    │   └── ...
    ├── results.json              {"task_name": ..., "success_rate": 0.8, ...}
    └── rollout_meta.json

Usage:
    # Phase 1: capture on GPU
    python scripts/capture_history_attention_rollout.py \
        --ckpt <checkpoint_path> --task <task_config_name> \
        --task_idx 0 --num_trials 10

    # Phase 2: analyze on CPU
    python scripts/analyze_history_attention.py \
        --data_dir evaluate_results/attention_analysis/<config>/<timestamp>/trial_00/
"""

import argparse
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import pathlib
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Monkey-patch robosuite to disable file logging before any robosuite import.
# On shared HPC clusters, /tmp is not writable. Robosuite hardcodes
# FileHandler("/tmp/robosuite.log") in log_utils.py.
import importlib.util as _importlib_util
_macros_path = None
for _p in sys.path:
    _candidate = os.path.join(_p, "robosuite", "macros.py")
    if os.path.isfile(_candidate):
        _macros_path = _candidate
        break
if _macros_path is None:
    raise ImportError("Cannot locate robosuite/macros.py on sys.path")
_robosuite_macros_spec = _importlib_util.spec_from_file_location("robosuite.macros", _macros_path)
_robosuite_macros = _importlib_util.module_from_spec(_robosuite_macros_spec)
_robosuite_macros_spec.loader.exec_module(_robosuite_macros)
_robosuite_macros.FILE_LOGGING_LEVEL = None
sys.modules["robosuite.macros"] = _robosuite_macros

import imageio

from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("max", lambda x: max(x))
OmegaConf.register_new_resolver("split", lambda s, idx: s.split("/")[int(idx)])

project_root = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Monkey-patch for attention capture
# ---------------------------------------------------------------------------
def _patch_history_attention(model, attn_data: dict):
    """Replace HistoryAttention.forward to capture attention weights and KV cache.

    The actual HistoryAttention.forward signature is:
        def forward(self, x, history_kv) -> tensor
    where flash_attention doesn't expose attention weights.
    We call the original forward for the correct output, then manually
    recompute attention weights from the same Q/K/V projections.
    """
    import torch.nn.functional as F
    from einops import rearrange
    from fastwam.models.wan22.action_dit import HistoryAttention

    _orig_forward = HistoryAttention.forward
    _layer_idx = [0]

    def _capturing_forward(self, x, history_kv):
        # Get the correct output from original forward
        result = _orig_forward(self, x, history_kv)

        # Manually recompute attention weights (flash_attention doesn't expose them)
        with torch.no_grad():
            history_k, history_v = history_kv
            seq_len = history_k.shape[1]
            pos = self.temporal_pos_emb[:, :seq_len, :].to(dtype=history_k.dtype)
            h_k = history_k + pos
            h_v = history_v + pos

            # Step 1: Self-attention weights
            sq = self.norm_q(self.self_q(h_k))
            sk = self.norm_k(self.self_k(h_k))
            n = self.num_heads
            sq_h = rearrange(sq, "b s (n d) -> b n s d", n=n)
            sk_h = rearrange(sk, "b s (n d) -> b n s d", n=n)
            scale = sq_h.shape[-1] ** -0.5
            self_attn_w = (sq_h @ sk_h.transpose(-2, -1) * scale).softmax(dim=-1)

            sv = self.self_v(h_v)
            sv_h = rearrange(sv, "b s (n d) -> b n s d", n=n)
            h_attn = F.scaled_dot_product_attention(sq_h, sk_h, sv_h)
            h_attn = rearrange(h_attn, "b n s d -> b s (n d)", n=n)
            h_refined = h_k + self.self_o(h_attn)

            # Step 2: Cross-attention weights (action -> refined history)
            cq = self.norm_q(self.cross_q(x))
            ck = self.norm_k(self.cross_k(h_refined))
            cq_h = rearrange(cq, "b s (n d) -> b n s d", n=n)
            ck_h = rearrange(ck, "b s (n d) -> b n s d", n=n)
            scale_c = cq_h.shape[-1] ** -0.5
            cross_attn_w = (cq_h @ ck_h.transpose(-2, -1) * scale_c).softmax(dim=-1)

            # Save to attn_data (head-averaged, float16 for compact storage)
            if "cross_attn" not in attn_data:
                attn_data["cross_attn"] = []
                attn_data["self_attn"] = []
            # cross_attn_w: [n_heads, S_action, S_history] → avg → [S_action, S_history]
            attn_data["cross_attn"].append(cross_attn_w[0].mean(dim=0).half().cpu())
            # self_attn_w: [n_heads, S_history, S_history] → avg → [S_history, S_history]
            attn_data["self_attn"].append(self_attn_w[0].mean(dim=0).half().cpu())
            _layer_idx[0] += 1

        return result

    HistoryAttention.forward = _capturing_forward

    def restore():
        HistoryAttention.forward = _orig_forward

    return restore


# ---------------------------------------------------------------------------
# Eval helpers (from eval_libero_single.py)
# ---------------------------------------------------------------------------
from experiments.libero.libero_utils import (
    get_libero_image,
    get_libero_env,
    get_libero_dummy_action,
    invert_gripper_action,
    quat2axisangle,
)
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json


def _center_crop_resize(img: np.ndarray, *, width: int, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w != h:
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img = img[top : top + side, left : left + side]
    return np.array(PILImage.fromarray(img).resize((width, height), PILImage.BILINEAR))


def _extract_sim_state(obs: dict) -> np.ndarray:
    return np.concatenate(
        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
    ).astype(np.float32)


def _normalize_proprio(proprio: np.ndarray, processor: FastWAMProcessor) -> torch.Tensor:
    state_meta = processor.shape_meta["state"]
    state_key = state_meta[0]["key"]
    state_batch = {"state": {state_key: torch.as_tensor(proprio, dtype=torch.float32).unsqueeze(0)}}
    state_batch = processor.action_state_transform(state_batch)
    state_batch = processor.normalizer.forward(state_batch)
    return state_batch["state"][state_key]


def _denormalize_action(action: torch.Tensor, processor: FastWAMProcessor) -> np.ndarray:
    if action.ndim == 2:
        action = action.unsqueeze(0)
    action_key = processor.shape_meta["action"][0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    action = action.to(dtype=torch.float32, device="cpu")
    return normalizer.backward(action).numpy()


def _obs_to_model_input(obs, processor, input_w, input_h, device, dtype, cfg):
    imgs = get_libero_image(obs)
    image_meta = processor.shape_meta["images"]
    num_cameras = processor.num_output_cameras
    concatenation = cfg.data.train.get("concat_multi_camera", "horizontal")

    if num_cameras == 1:
        shape = image_meta[0]["shape"]
        rgb = _center_crop_resize(imgs["image"], width=int(shape[2]), height=int(shape[1]))
    elif num_cameras == 2:
        s0, s1 = image_meta[0]["shape"], image_meta[1]["shape"]
        primary = _center_crop_resize(imgs["image"], width=int(s0[2]), height=int(s0[1]))
        wrist = _center_crop_resize(imgs["wrist_image"], width=int(s1[2]), height=int(s1[1]))
        if concatenation == "horizontal":
            rgb = np.concatenate([primary, wrist], axis=1)
        else:
            rgb = np.concatenate([primary, wrist], axis=0)
    else:
        raise ValueError(f"Unsupported num_output_cameras={num_cameras}")

    x = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    x = x * (2.0 / 255.0) - 1.0
    proprio = _normalize_proprio(_extract_sim_state(obs), processor)
    return x, proprio, imgs


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------
def run_episode(
    model, processor, env, task_description, cfg,
    action_horizon, input_w, input_h, num_video_frames,
    attn_data: dict, save_dir: str, save_video_pred: bool = True,
    num_steps_wait: int = 5, max_steps: int = 600,
    replan_steps: int = 10, binarize_gripper: bool = True,
):
    """Run one full episode until done or max steps."""

    device = str(model.device)
    model_dtype = model.torch_dtype

    # Encode prompt
    prompt = DEFAULT_PROMPT.format(task=task_description)
    context, context_mask = model.encode_prompt(prompt)

    # Warm-up: step dummy actions (env already reset + init_state set by caller)
    for _ in range(num_steps_wait):
        obs, _, _, _ = env.step(get_libero_dummy_action())

    # Reset history cache for CasWAM models
    if hasattr(model, "reset_history"):
        model.reset_history()

    # Create output directories
    for subdir in ["obs_frames", "step_frames", "action_chunks", "attention", "kv_cache"]:
        os.makedirs(os.path.join(save_dir, subdir), exist_ok=True)
    if save_video_pred:
        os.makedirs(os.path.join(save_dir, "video_pred"), exist_ok=True)

    # Rollout state
    done, success = False, False
    step_count, replan_id = 0, 0
    all_video_frames = []
    pending_actions = []
    t_episode_start = time.time()
    num_inference_steps = int(cfg.EVALUATION.get("num_inference_steps", cfg.get("eval_num_inference_steps", 20)))

    print(f"  Running episode (max {max_steps} steps)...")
    pbar = tqdm(desc="Steps", total=max_steps + num_steps_wait)

    # Count initial wait steps
    for _ in range(num_steps_wait):
        pbar.update(1)

    while not done and step_count < max_steps:
        t_replan = time.time()

        # === Predict ===
        if len(pending_actions) == 0:
            image_tensor, proprio_tensor, imgs = _obs_to_model_input(
                obs, processor, input_w, input_h, device, model_dtype, cfg
            )

            # Save observation frame (agentview)
            obs_img = imgs["image"]
            obs_pil = PILImage.fromarray(obs_img)
            obs_pil.save(os.path.join(save_dir, "obs_frames", f"replan_{replan_id:03d}.png"))

            action = torch.zeros(
                (action_horizon, model.action_expert.action_dim),
                device=device, dtype=model_dtype,
            )

            # Capture attention
            attn_data.clear()
            restore_attn = _patch_history_attention(model, attn_data)

            seed = int(torch.randint(0, 2**16, (1,)).item())
            sigma_shift = cfg.EVALUATION.get("sigma_shift", None)

            try:
                # Use infer_action for actions (matches eval_libero_single.py)
                action_out = model.infer_action(
                    prompt=None,
                    input_image=image_tensor.clone(),
                    action_horizon=action_horizon,
                    context=context.clone(),
                    context_mask=context_mask.clone(),
                    num_inference_steps=num_inference_steps,
                    sigma_shift=sigma_shift,
                    seed=seed,
                    proprio=proprio_tensor,
                )
            finally:
                restore_attn()

            # Denormalize action
            action_pred_np = _denormalize_action(action_out["action"], processor)  # [1, T, D]
            action_buffer = action_pred_np[0]  # [T, D]

            # Gripper processing (matching eval_libero_single.py):
            # Map from [0,1] to [-1,1], then invert sign for LIBERO
            action_buffer[..., -1] = action_buffer[..., -1] * 2 - 1
            action_buffer = invert_gripper_action(action_buffer)
            if binarize_gripper:
                action_buffer[..., -1] = np.sign(action_buffer[..., -1])

            pending_actions = list(action_buffer[:replan_steps])

            # Save action chunk (before gripper inversion, raw denormalized)
            np.save(
                os.path.join(save_dir, "action_chunks", f"replan_{replan_id:03d}.npy"),
                action_pred_np[0].astype(np.float32),
            )

            # Save attention data (compact: head-averaged, float16, per-layer stacked)
            if attn_data.get("cross_attn"):
                torch.save(
                    {
                        "cross_attn": torch.stack(attn_data["cross_attn"]),  # [n_layers, S_action, S_hist]
                        "self_attn": torch.stack(attn_data["self_attn"]),    # [n_layers, S_hist, S_hist]
                    },
                    os.path.join(save_dir, "attention", f"replan_{replan_id:03d}.pt"),
                )

            # Save KV cache
            if model._history_kv_cache is not None:
                kv_save = []
                for k, v in model._history_kv_cache:
                    kv_save.append((
                        k.detach().float().cpu(), v.detach().float().cpu(),
                    ))
                torch.save(kv_save, os.path.join(save_dir, "kv_cache", f"replan_{replan_id:03d}.pt"))

            # Save fully-denoised video prediction (separate infer_joint call)
            # infer_action already appended to history; save/restore to avoid
            # double-append from infer_joint.
            if save_video_pred:
                saved_hist = None
                if model._history_kv_cache is not None:
                    saved_hist = [(k.clone(), v.clone()) for k, v in model._history_kv_cache]
                    saved_count = model._history_step_count

                joint_out = model.infer_joint(
                    prompt=None,
                    input_image=image_tensor.clone(),
                    num_video_frames=num_video_frames,
                    action_horizon=action_horizon,
                    action=action,
                    context=context.clone(),
                    context_mask=context_mask.clone(),
                    num_inference_steps=num_inference_steps,
                    sigma_shift=sigma_shift,
                    seed=seed,
                    proprio=proprio_tensor,
                    test_action_with_infer_action=False,
                )

                # Restore history to state after infer_action
                if saved_hist is not None:
                    model._history_kv_cache = saved_hist
                    model._history_step_count = saved_count

                pred_frames = joint_out.get("video")
                if pred_frames is not None:
                    frame_dir = os.path.join(save_dir, "video_pred", f"replan_{replan_id:03d}")
                    os.makedirs(frame_dir, exist_ok=True)
                    writer = imageio.get_writer(
                        os.path.join(frame_dir, "pred_video.mp4"), fps=10, macro_block_size=1,
                    )
                    for fi, frame in enumerate(pred_frames):
                        frame.save(os.path.join(frame_dir, f"frame_{fi:03d}.png"))
                        writer.append_data(np.array(frame))
                    writer.close()

            replan_id += 1

        # === Execute ===
        action_to_execute = pending_actions.pop(0)
        obs, reward, done, info = env.step(action_to_execute)
        step_count += 1
        dt = time.time() - t_replan

        # Capture step frame (observation AFTER executing this action step)
        step_img = obs["agentview_image"][::-1, ::-1]
        all_video_frames.append(step_img.copy())
        PILImage.fromarray(step_img).save(
            os.path.join(save_dir, "step_frames",
                         f"replan_{replan_id - 1:03d}_step_{step_count:03d}.png")
        )

        success = bool(success or info.get("is_success", False) or done)
        pbar.update(1)
        pbar.set_postfix({"replan": replan_id, "success": success, "dt": f"{dt:.3f}s"})

    pbar.close()
    episode_time = time.time() - t_episode_start
    print(f"  Episode done: success={success}, replans={replan_id}, steps={step_count}, "
          f"time={episode_time:.1f}s")

    # Save episode video
    if all_video_frames:
        trial_name = os.path.basename(save_dir)
        video_path = os.path.join(os.path.dirname(save_dir), "videos", f"{trial_name}.mp4")
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        writer = imageio.get_writer(video_path, fps=20, macro_block_size=1)
        for frame in all_video_frames:
            writer.append_data(frame)
        writer.close()

    # Save success info
    with open(os.path.join(save_dir, "success.json"), "w") as f:
        json.dump({
            "success": bool(success),
            "num_replans": replan_id,
            "num_steps": step_count,
            "episode_time": episode_time,
            "task_name": task_description,
        }, f, indent=2)

    return {
        "success": success,
        "num_replans": replan_id,
        "num_steps": step_count,
        "episode_time": episode_time,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture rollouts with attention, KV cache, and video predictions"
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint file (.pt)")
    parser.add_argument("--task", type=str, default=None,
                        help="Task config name (e.g. libero_caswam_acthist_uncond_2cam224_5e-5). "
                             "Auto-derived from --ckpt path if omitted.")
    parser.add_argument("--task_suite", type=str, default="libero_10",
                        help="LIBERO task suite name")
    parser.add_argument("--task_idx", type=int, default=0)
    parser.add_argument("--num_trials", type=int, default=1, help="Number of episode trials")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--no_video_pred", action="store_true",
                        help="Skip full-denoised video predictions (faster)")
    return parser.parse_args()


def main():
    args = parse_args()
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    # --- Checkpoint validation ---
    ckpt_path = os.path.abspath(args.ckpt)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Derive run_name from checkpoint path (or use explicit --task)
    if args.task:
        run_name = args.task
    else:
        ckpt_dir = os.path.dirname(ckpt_path)  # .../checkpoints/weights
        run_dir = os.path.dirname(os.path.dirname(os.path.dirname(ckpt_dir)))
        run_name = os.path.basename(run_dir)

    # Output directory
    if args.output_dir:
        output_base = args.output_dir
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_base = os.path.join("evaluate_results", "attention_analysis", run_name, args.task_suite, str(args.task_idx), timestamp)
    os.makedirs(output_base, exist_ok=True)

    print("=" * 70)
    print("CAPTURE HISTORY ATTENTION ROLLOUT")
    print("=" * 70)
    print(f"  checkpoint: {ckpt_path}")
    print(f"  run_name: {run_name}")
    print(f"  task_suite: {args.task_suite}")
    print(f"  task_idx: {args.task_idx}")
    print(f"  num_trials: {args.num_trials}")
    print(f"  save_video_pred: {not args.no_video_pred}")
    print(f"  output_dir: {output_base}")
    print("=" * 70)

    # --- Load config ---
    config_dir = str(project_root / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(
            config_name="sim_libero.yaml",
            overrides=[
                f"task={run_name}",
                f"ckpt={ckpt_path}",
                "model.load_text_encoder=true",
                "model.skip_dit_load_from_pretrain=false",
                f"EVALUATION.task_suite_name={args.task_suite}",
                f"EVALUATION.task_id={args.task_idx}",
                f"EVALUATION.num_trials={args.num_trials}",
            ],
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dtype = torch.bfloat16

    # --- Load model ---
    model = instantiate(cfg.model, model_dtype=model_dtype, device=device)
    model.load_checkpoint(ckpt_path)
    model = model.to(device).eval()

    # --- Load processor ---
    dataset_stats_path = "checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json"
    dataset_stats = load_dataset_stats_from_json(dataset_stats_path)
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)

    # --- Compute dimensions ---
    action_horizon = int(cfg.data.train.num_frames) - 1
    video_size = cfg.data.train.get("video_size", [224, 224])
    input_h, input_w = int(video_size[0]), int(video_size[1])
    num_video_frames = (int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1

    # --- LIBERO environment ---
    from libero.libero import benchmark as libero_benchmark
    benchmark_dict = libero_benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    task = task_suite.get_task(args.task_idx)
    task_description = task.language
    init_states = task_suite.get_task_init_states(args.task_idx)

    # Extend init_states for num_trials
    while len(init_states) < args.num_trials:
        init_states.extend(init_states[: args.num_trials - len(init_states)])

    env, _ = get_libero_env(task, 256, seed=args.seed)

    print(f"  task: {task_description}")
    print(f"  action_horizon: {action_horizon}")
    print(f"  num_video_frames: {num_video_frames}")
    print(f"  input_size: {input_h}x{input_w}")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Eval params
    num_steps_wait = int(cfg.EVALUATION.get("num_steps_wait", 5))
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 10))
    binarize_gripper = bool(cfg.EVALUATION.get("binarize_gripper", True))
    max_steps = 600  # LIBERO standard

    # --- Save analysis metadata (model class, token layout) ---
    model_class = type(model).__name__
    is_acthist = "ActHist" in model_class
    action_tokens_per_frame = 0
    if is_acthist:
        _ah = int(cfg.data.train.num_frames) - 1  # action_horizon
        _nf = _ah // int(cfg.data.train.action_video_freq_ratio) + 1  # num_obs_frames
        _nr = _nf - 1  # num_replans
        action_tokens_per_frame = _ah // _nr if _nr > 0 else 0

    analysis_meta = {
        "model_class": model_class,
        "is_acthist": is_acthist,
        "action_tokens_per_frame": action_tokens_per_frame,
        "action_horizon": action_horizon,
    }
    with open(os.path.join(output_base, "analysis_meta.json"), "w") as f:
        json.dump(analysis_meta, f, indent=2)
    print(f"  analysis_meta: model_class={model_class}, is_acthist={is_acthist}, "
          f"action_tokens_per_frame={action_tokens_per_frame}")

    # --- Run trials ---
    all_results = []
    num_success = 0

    for trial in range(args.num_trials):
        trial_dir = os.path.join(output_base, f"trial_{trial:02d}")
        os.makedirs(trial_dir, exist_ok=True)

        print(f"\n{'=' * 50}")
        print(f"Trial {trial}/{args.num_trials - 1} | Task: {task_description}")
        print(f"{'=' * 50}")

        # Set initial state for this trial
        env.reset()
        init_state = init_states[trial % len(init_states)]
        env.set_init_state(init_state)

        attn_data = {}
        result = run_episode(
            model=model,
            processor=processor,
            env=env,
            task_description=task_description,
            cfg=cfg,
            action_horizon=action_horizon,
            input_w=input_w,
            input_h=input_h,
            num_video_frames=num_video_frames,
            attn_data=attn_data,
            save_dir=trial_dir,
            save_video_pred=not args.no_video_pred,
            num_steps_wait=num_steps_wait,
            max_steps=max_steps,
            replan_steps=replan_steps,
            binarize_gripper=binarize_gripper,
        )

        result["trial"] = trial
        result["task_name"] = task_description
        all_results.append(result)
        if result["success"]:
            num_success += 1

    # --- Save summary ---
    success_rate = num_success / args.num_trials
    summary = {
        "task_name": task_description,
        "task_idx": args.task_idx,
        "task_suite": args.task_suite,
        "num_trials": args.num_trials,
        "num_success": num_success,
        "success_rate": success_rate,
        "checkpoint": ckpt_path,
        "trials": all_results,
    }

    for fname in ["results.json", "rollout_meta.json"]:
        with open(os.path.join(output_base, fname), "w") as f:
            json.dump(summary, f, indent=2)

    env.close()

    print(f"\n{'=' * 70}")
    print(f"CAPTURE COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Task: {task_description}")
    print(f"  Success rate: {num_success}/{args.num_trials} = {success_rate:.1%}")
    print(f"  Output: {output_base}")
    print(f"  Per-trial data: {output_base}/trial_XX/")
    print(f"  Videos: {output_base}/videos/")
    print()
    print(f"  Analyze with:")
    print(f"    python scripts/analyze_history_attention.py \\")
    print(f"        --data_dir {output_base}/trial_00/")


if __name__ == "__main__":
    main()
