import json
import inspect
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import hydra
import numpy as np
import torch
from accelerate import PartialState
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

# try:
#     import rootutils

#     rootutils.setup_root(__file__, indicator=".python-version", pythonpath=True)
# except ModuleNotFoundError:
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.libero.libero_utils import (
    LIBERO_ENV_RESOLUTION,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    invert_gripper_action,
    quat2axisangle,
    save_prediction_video,
    save_rollout_video,
)
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.pytorch_utils import set_global_seed
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from leapbot_va.eval_contract import (
    build_result_contract,
    build_runtime_contract,
    resolve_dataset_stats_path,
    resolve_libero_task_and_initial_states,
)
from leapbot_va.eval_fingerprint import (
    atomic_write_json,
    build_verified_evaluation_fingerprint,
    load_evaluation_fingerprint,
)
from leapbot_va.image_preprocessing import preprocess_uint8_libero_cameras
from leapbot_va.libero import (
    canonicalize_libero_env_action,
    executed_env_actions_to_model_space,
)
from action_ensembler import ActionEnsembler

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("max", lambda x: max(x))
OmegaConf.register_new_resolver("split", lambda s, idx: s.split("/")[int(idx)])

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _build_result_fingerprint(
    cfg: DictConfig,
    *,
    configured_causal_mode: Optional[str],
    dataset_stats_path: Path,
    task: Any,
    initial_states_path: Path,
) -> dict[str, Any]:
    """Build identity from actual files and strictly verify preflight output."""
    hydra_choices = OmegaConf.to_container(
        HydraConfig.get().runtime.choices, resolve=True
    )
    config_name = str(HydraConfig.get().job.config_name)
    runtime_contract = build_runtime_contract(
        cfg,
        config_name=config_name,
        hydra_choices=hydra_choices,
        dataset_stats_path=dataset_stats_path,
        source_root=project_root,
        configured_causal_mode=configured_causal_mode,
    )
    result_contract = build_result_contract(
        cfg,
        task=task,
        initial_states_path=initial_states_path,
    )
    expected_path_cfg = cfg.EVALUATION.get("expected_fingerprint_path", None)
    expected = None
    if expected_path_cfg is not None:
        expected_path = Path(
            os.path.expanduser(os.path.expandvars(str(expected_path_cfg)))
        ).resolve()
        expected = load_evaluation_fingerprint(expected_path)
    elif cfg.EVALUATION.get("result_fingerprint", None) is not None:
        raise ValueError(
            "EVALUATION.result_fingerprint is legacy and is not trusted by schema 3; "
            "use EVALUATION.expected_fingerprint_path for strict preflight matching."
        )
    return build_verified_evaluation_fingerprint(
        checkpoint_path=str(cfg.ckpt),
        runtime_contract=runtime_contract,
        result_contract=result_contract,
        expected=expected,
    )


def _resolve_eval_device(cfg: DictConfig) -> str:
    eval_device = cfg.EVALUATION.get("device")
    if eval_device is not None:
        return str(eval_device)
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_dataset_stats_path(cfg: DictConfig) -> Path:
    return resolve_dataset_stats_path(cfg)


def _load_model_checkpoint(model: torch.nn.Module, ckpt: str) -> dict:
    payload = model.load_checkpoint(ckpt)
    logging.info("Loaded checkpoint via model.load_checkpoint: %s", ckpt)
    return payload if isinstance(payload, dict) else {}

    # deprecated legacy checkpoint loading
    payload = torch.load(ckpt, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Legacy checkpoint payload must be dict, got: {type(payload)}")

    if "mot" in payload and hasattr(model, "mot"):
        missing, unexpected = model.mot.load_state_dict(payload["mot"], strict=False)
        logging.warning(
            "Loaded fallback `mot` state_dict with strict=False. Missing=%d Unexpected=%d",
            len(missing),
            len(unexpected),
        )
        return

    state_dict = None
    for key in ("model_state_dict", "state_dict", "model"):
        value = payload.get(key)
        if isinstance(value, dict):
            state_dict = value
            break
    if state_dict is None and all(torch.is_tensor(v) for v in payload.values()):
        state_dict = payload
    if state_dict is None:
        raise ValueError(f"Cannot parse legacy checkpoint keys from: {ckpt}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logging.warning(
        "Loaded fallback model state_dict with strict=False. Missing=%d Unexpected=%d",
        len(missing),
        len(unexpected),
    )


def _normalize_proprio(
    proprio: np.ndarray,
    processor: FastWAMProcessor,
) -> torch.Tensor:
    state_meta = processor.shape_meta["state"]
    if len(state_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged state key in shape_meta['state']."
        )
    state_key = state_meta[0]["key"]

    state_batch = {"state": {state_key: torch.as_tensor(proprio, dtype=torch.float32).unsqueeze(0)}}
    state_batch = processor.action_state_transform(state_batch)
    state_batch = processor.normalizer.forward(state_batch)
    return state_batch["state"][state_key]


def _obs_to_model_input(
    obs: dict,
    cfg: DictConfig,
    processor: FastWAMProcessor,
    width: int,
    height: int,
    device: str,
    dtype: torch.dtype,
):
    imgs = get_libero_image(obs)
    concatenation = cfg.data.train.get("concat_multi_camera", "horizontal")
    x = preprocess_uint8_libero_cameras(
        imgs,
        processor=processor,
        concat_multi_camera=str(concatenation),
        video_size=[height, width],
    ).to(device=device, dtype=dtype)

    proprio = _normalize_proprio(_extract_sim_state(obs), processor)

    return x, proprio, imgs


def _extract_sim_state(obs: dict) -> np.ndarray:
    """Build simulator state from current observation.

    This is used as proprio input for model inference.
    """
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)
    return state


def _denormalize_action(action: torch.Tensor, processor: FastWAMProcessor) -> np.ndarray:
    if action.ndim == 2:
        action = action.unsqueeze(0)
    if action.ndim != 3:
        raise ValueError(f"Expected action tensor [B, T, D], got {tuple(action.shape)}")

    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged action key in shape_meta['action']."
        )

    action_key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    action = action.to(dtype=torch.float32, device="cpu")
    denorm = normalizer.backward(action)
    return denorm.numpy()


def _get_num_video_frames(cfg: DictConfig) -> int:
    return (int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1


def _validate_visualize_future_video_cfg(cfg: DictConfig) -> None:
    if not bool(cfg.EVALUATION.get("visualize_future_video", False)):
        return

    action_conditioned = cfg.model.video_dit_config.get("action_conditioned", None)
    if action_conditioned is not False:
        raise ValueError(
            "EVALUATION.visualize_future_video=true requires "
            "model.video_dit_config.action_conditioned=false."
        )


def _select_predicted_future_frames(pred_video: list[Image.Image], cfg: DictConfig) -> list[Image.Image]:
    if len(pred_video) == 0:
        raise ValueError("`infer_joint` returned an empty predicted video.")

    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    action_video_freq_ratio = int(cfg.data.train.action_video_freq_ratio)
    num_future_frames = replan_steps // action_video_freq_ratio
    keep_frames = 1 + num_future_frames
    return list(pred_video[:keep_frames])


def _get_future_frame_capture_steps(cfg: DictConfig) -> list[int]:
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    action_video_freq_ratio = int(cfg.data.train.action_video_freq_ratio)
    num_future_frames = replan_steps // action_video_freq_ratio
    return [step_idx * action_video_freq_ratio for step_idx in range(num_future_frames + 1)]


def _frame_to_rgb_array(frame: Any) -> np.ndarray:
    if isinstance(frame, dict):
        images = []
        for value in frame.values():
            value_array = np.array(value) if isinstance(value, Image.Image) else np.array(value, copy=True)
            images.append(value_array)
        return np.concatenate(images, axis=1)
    if isinstance(frame, Image.Image):
        return np.array(frame.convert("RGB"))
    return np.array(frame, copy=True)


def _compute_clip_mean_psnr(
    gt_frames: list[Any],
    pred_frames: list[Any],
    eps: float = 1e-8,
) -> Optional[float]:
    if len(gt_frames) == 0 or len(pred_frames) == 0:
        return None
    assert len(gt_frames) == len(pred_frames), (
        "GT/pred frame count mismatch for PSNR: "
        f"len(gt_frames)={len(gt_frames)} len(pred_frames)={len(pred_frames)}. "
        "This indicates temporal misalignment in future-video capture."
    )
    num_frames = len(gt_frames)

    frame_psnr_values = []
    for gt_frame, pred_frame in zip(gt_frames[:num_frames], pred_frames[:num_frames]):
        gt_image = _frame_to_rgb_array(gt_frame)
        pred_image = _frame_to_rgb_array(pred_frame)
        target_h, target_w = pred_image.shape[:2]
        if gt_image.shape[:2] != (target_h, target_w):
            gt_image = np.array(
                Image.fromarray(gt_image).resize((target_w, target_h), resample=Image.BILINEAR)
            )

        gt_f32 = gt_image.astype(np.float32)
        pred_f32 = pred_image.astype(np.float32)
        mse = float(np.mean((pred_f32 - gt_f32) ** 2))
        psnr = 10.0 * np.log10((255.0 * 255.0) / max(mse, eps))
        frame_psnr_values.append(float(psnr))

    if len(frame_psnr_values) == 0:
        return None
    return float(np.mean(frame_psnr_values))


def _predict_action_chunk(
    obs: dict,
    task_description: str,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
    memory=None,
) -> tuple[np.ndarray, dict, Optional[list[Image.Image]], dict[str, Any]]:
    num_inference_steps_cfg = cfg.EVALUATION.get("num_inference_steps", None)
    if num_inference_steps_cfg is None:
        num_inference_steps = int(cfg.get("eval_num_inference_steps", 20))
    else:
        num_inference_steps = int(num_inference_steps_cfg)
    prompt_template = DEFAULT_PROMPT
    prompt = prompt_template.format(task=task_description)

    image, proprio, imgs = _obs_to_model_input(
        obs,
        cfg=cfg,
        processor=processor,
        width=input_w,
        height=input_h,
        device=model_device,
        dtype=model.torch_dtype,
    )

    infer_kwargs = {
        "prompt": prompt,
        "input_image": image,
        "action_horizon": action_horizon,
        "negative_prompt": str(cfg.EVALUATION.get("negative_prompt", "")),
        "text_cfg_scale": float(cfg.EVALUATION.get("text_cfg_scale", 1.0)),
        "num_inference_steps": num_inference_steps,
        "proprio": proprio,
        "sigma_shift": (
            None
            if cfg.EVALUATION.get("sigma_shift") is None
            else float(cfg.EVALUATION.get("sigma_shift"))
        ),
        "seed": None if cfg.get("seed") is None else int(cfg.seed),
        "rand_device": str(cfg.EVALUATION.get("rand_device", "cpu")),
        "tiled": bool(cfg.EVALUATION.get("tiled", False)),
    }
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    predicted_future_frames = None
    if memory is not None:
        if visualize_future_video:
            raise ValueError("LeapBot memory inference never predicts/decodes future video")
        infer_kwargs["memory"] = memory
        infer_kwargs["profile"] = True
    if visualize_future_video:
        infer_kwargs["num_video_frames"] = _get_num_video_frames(cfg)
    elif "num_video_frames" in inspect.signature(model.infer_action).parameters:
        infer_kwargs["num_video_frames"] = _get_num_video_frames(cfg)

    if str(model_device).startswith("cuda"):
        torch.cuda.synchronize(torch.device(model_device))
    inference_start = time.perf_counter()
    with torch.no_grad():
        if visualize_future_video:
            pred = model.infer_joint(**infer_kwargs)
            predicted_future_frames = _select_predicted_future_frames(pred["video"], cfg)
        else:
            pred = model.infer_action(**infer_kwargs)
    if str(model_device).startswith("cuda"):
        torch.cuda.synchronize(torch.device(model_device))
    total_inference_s = time.perf_counter() - inference_start
    action = pred["action"]  # [T, D]

    action = _denormalize_action(action, processor)[0]  # [T, D]

    # The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    action[..., -1] = action[..., -1] * 2 - 1
    action = invert_gripper_action(action)
    if bool(cfg.EVALUATION.get("binarize_gripper", False)):
        action[..., -1] = np.sign(action[..., -1])
    timing = dict(pred.get("timing", {}))
    timing["total_inference_s"] = total_inference_s
    return action, imgs, predicted_future_frames, {
        "timing": timing,
        "memory": pred.get("memory", {}),
    }


def _get_max_steps(task_suite_name: str) -> int:
    suite_steps = {
        "libero_spatial": 400,
        "libero_object": 400,
        "libero_goal": 400,
        "libero_10": 700,
        "libero_90": 700,
    }
    if task_suite_name not in suite_steps:
        raise ValueError(f"Unknown task suite: {task_suite_name}")
    return suite_steps[task_suite_name]


def run_single_episode(
    env,
    initial_state,
    task_description: str,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    episode_idx: int,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
) -> tuple[bool, list, list[dict[str, Any]], Optional[float], dict[str, Any]]:
    max_steps = _get_max_steps(cfg.EVALUATION.task_suite_name)
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    num_steps_wait = int(cfg.EVALUATION.get("num_steps_wait", 5))
    use_action_ensembler = bool(cfg.EVALUATION.get("use_action_ensembler", False))
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    record_rollout_video = bool(cfg.EVALUATION.get("save_rollout_video", True))
    capture_steps = set(_get_future_frame_capture_steps(cfg)[1:])

    memory_cfg = cfg.EVALUATION.get("memory", {})
    memory_enabled = bool(memory_cfg.get("enabled", False))
    if memory_enabled and use_action_ensembler:
        raise ValueError(
            "LeapBot memory evaluation discards every unexecuted action prediction; "
            "action ensembling across replans is therefore unsupported"
        )
    memory = None
    if memory_enabled:
        if not hasattr(model, "create_memory") or not hasattr(model, "commit_executed_actions"):
            raise TypeError("EVALUATION.memory.enabled requires a LeapBotVA model")
        memory = model.create_memory(
            exit_depth=int(memory_cfg.get("exit_depth", 30)),
            causal_mode=str(memory_cfg.get("causal_mode", "interleaved")),
            max_history_blocks=int(memory_cfg.get("max_history_blocks", 70)),
            retained_history_blocks=memory_cfg.get(
                "retained_history_blocks", None
            ),
            action_horizon=action_horizon,
            replan_steps=replan_steps,
        )
        if visualize_future_video:
            raise ValueError("future-video visualization is incompatible with LeapBot memory")

    env.reset()
    obs = env.set_init_state(initial_state)
    if use_action_ensembler:
        ensembler = ActionEnsembler()
        ensembler.reset()

    replay_images = []
    predicted_future_video_clips: list[dict[str, Any]] = []
    episode_future_clip_psnr: list[float] = []
    pending_actions: list[list[float]] = []
    current_predicted_future_clip: Optional[dict[str, Any]] = None
    current_replan_step = 0
    current_replan_idx = -1
    current_executed_actions: list[list[float]] = []
    executed_control_steps = 0
    memory_metrics: dict[str, Any] = {
        "enabled": memory_enabled,
        "replans": [],
        "peak_cache_bytes": 0,
    }
    if str(model_device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(model_device))

    t = 0
    done = False
    pbar = tqdm(total=max_steps + num_steps_wait, desc=f"Episode {episode_idx + 1}")
    while t < max_steps + num_steps_wait:
        pbar.update(1)
        if t < num_steps_wait:
            obs, _, done, _ = env.step(get_libero_dummy_action())
            t += 1
            continue

        if len(pending_actions) == 0:
            action_chunk, imgs, predicted_future_frames, inference_metrics = _predict_action_chunk(
                obs=obs,
                task_description=task_description,
                model=model,
                processor=processor,
                cfg=cfg,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
                memory=memory,
            )
            current_executed_actions = []
            memory_metrics["replans"].append(inference_metrics)
            if predicted_future_frames is not None:
                current_replan_idx += 1
                current_predicted_future_clip = {
                    "replan_idx": current_replan_idx,
                    "gt_frames": [imgs.copy()],
                    "pred_frames": predicted_future_frames,
                }
            else:
                current_predicted_future_clip = None
            current_replan_step = 0
            if use_action_ensembler:
                ensembler.add_actions(action_chunk, t)
                pending_actions = [ensembler.get_action(ts).tolist() for ts in range(t, t + replan_steps)]
            else:
                pending_actions = action_chunk[:replan_steps].tolist()
            if record_rollout_video:
                replay_images.append(imgs.copy())
        else:
            if record_rollout_video:
                imgs = get_libero_image(obs)
                replay_images.append(imgs.copy())

        executed_action = canonicalize_libero_env_action(
            pending_actions.pop(0),
            env.action_spec,
            binarize_gripper=bool(cfg.EVALUATION.get("binarize_gripper", False)),
        )
        committed_action = executed_action.copy()
        obs, _, done, _ = env.step(executed_action)
        executed_control_steps += 1
        if memory_enabled:
            current_executed_actions.append(committed_action)
        if visualize_future_video and current_predicted_future_clip is not None:
            current_replan_step += 1
            if current_replan_step in capture_steps:
                current_predicted_future_clip["gt_frames"].append(get_libero_image(obs))
            if done or len(pending_actions) == 0:
                expected_frame_count = 1 + sum(
                    1 for capture_step in capture_steps if capture_step <= current_replan_step
                )
                gt_len = len(current_predicted_future_clip["gt_frames"])
                pred_len = len(current_predicted_future_clip["pred_frames"])
                assert gt_len == expected_frame_count, (
                    "GT future frames do not match expected capture count: "
                    f"gt_len={gt_len} expected={expected_frame_count} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']} "
                    f"current_replan_step={current_replan_step} capture_steps={sorted(capture_steps)}."
                )
                assert pred_len >= expected_frame_count, (
                    "Predicted future frames shorter than expected capture count: "
                    f"pred_len={pred_len} expected={expected_frame_count} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']}."
                )
                if pred_len != expected_frame_count:
                    logging.info(
                        "Align predicted clip length to executed steps: "
                        "episode=%s replan=%s done=%s expected=%s pred_full=%s",
                        episode_idx,
                        current_predicted_future_clip["replan_idx"],
                        done,
                        expected_frame_count,
                        pred_len,
                    )
                current_predicted_future_clip["pred_frames"] = current_predicted_future_clip["pred_frames"][
                    :expected_frame_count
                ]
                assert len(current_predicted_future_clip["gt_frames"]) == len(
                    current_predicted_future_clip["pred_frames"]
                ), (
                    "GT/pred frame count mismatch after alignment: "
                    f"len(gt_frames)={len(current_predicted_future_clip['gt_frames'])} "
                    f"len(pred_frames)={len(current_predicted_future_clip['pred_frames'])} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']}."
                )
                clip_psnr = _compute_clip_mean_psnr(
                    current_predicted_future_clip["gt_frames"],
                    current_predicted_future_clip["pred_frames"],
                )
                if clip_psnr is not None:
                    episode_future_clip_psnr.append(clip_psnr)
                predicted_future_video_clips.append(current_predicted_future_clip)
                current_predicted_future_clip = None
        if memory_enabled and (done or len(pending_actions) == 0):
            model_actions = executed_env_actions_to_model_space(
                np.asarray(current_executed_actions, dtype=np.float32),
                processor,
            )
            commit_metrics = model.commit_executed_actions(
                memory,
                model_actions,
                profile=True,
            )
            memory_metrics["replans"][-1]["commit"] = commit_metrics
            memory_metrics["peak_cache_bytes"] = max(
                int(memory_metrics["peak_cache_bytes"]),
                int(commit_metrics["cache_bytes"]),
            )
            current_executed_actions = []
        if done:
            break
        t += 1
    if memory_enabled and current_executed_actions:
        model_actions = executed_env_actions_to_model_space(
            np.asarray(current_executed_actions, dtype=np.float32), processor
        )
        commit_metrics = model.commit_executed_actions(memory, model_actions, profile=True)
        memory_metrics["replans"][-1]["commit"] = commit_metrics
        memory_metrics["peak_cache_bytes"] = max(
            int(memory_metrics["peak_cache_bytes"]), int(commit_metrics["cache_bytes"])
        )
        current_executed_actions = []
    pbar.close()
    memory_metrics["control_steps"] = executed_control_steps

    if memory is not None:
        memory_metrics["completed_blocks"] = memory.completed_blocks
        memory_metrics["retained_history_blocks"] = memory.retained_completed_blocks
        memory_metrics["final_cache_bytes"] = memory.cache_nbytes
        committed_steps = sum(
            int(replan.get("commit", {}).get("executed_actions", 0))
            for replan in memory_metrics["replans"]
        )
        if committed_steps != executed_control_steps:
            raise AssertionError(
                f"executed/committed action mismatch: {executed_control_steps} vs {committed_steps}"
            )
        model.reset_memory(memory)
    if str(model_device).startswith("cuda"):
        memory_metrics["peak_gpu_bytes"] = int(
            torch.cuda.max_memory_allocated(torch.device(model_device))
        )

    episode_mean_psnr = (
        float(np.mean(episode_future_clip_psnr)) if len(episode_future_clip_psnr) > 0 else None
    )
    return bool(done), replay_images, predicted_future_video_clips, episode_mean_psnr, memory_metrics


def run_single_task(
    task,
    initial_states,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    video_dir: Path,
    predicted_video_dir: Path,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
) -> dict:
    env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, cfg.get("seed"))
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    record_rollout_video = bool(cfg.EVALUATION.get("save_rollout_video", True))
    results = {
        "successes": 0,
        "failure_episodes": [],
        "success_episodes": [],
        "task_description": task_description,
        "completion_steps": [],
    }
    if visualize_future_video:
        results["episode_future_video_psnr"] = []
        results["future_video_psnr_mean"] = None
    results["memory_metrics"] = []

    for trial_idx in range(int(cfg.EVALUATION.num_trials)):
        success, replay_images, predicted_future_video_clips, episode_mean_psnr, memory_metrics = run_single_episode(
            env=env,
            initial_state=initial_states[trial_idx],
            task_description=task_description,
            model=model,
            processor=processor,
            cfg=cfg,
            episode_idx=trial_idx,
            action_horizon=action_horizon,
            input_w=input_w,
            input_h=input_h,
            model_device=model_device,
        )
        if success:
            results["successes"] += 1
            results["success_episodes"].append(trial_idx)
        else:
            results["failure_episodes"].append(trial_idx)
        results["completion_steps"].append(memory_metrics["control_steps"])
        if visualize_future_video:
            results["episode_future_video_psnr"].append(episode_mean_psnr)
        results["memory_metrics"].append(memory_metrics)

        if record_rollout_video:
            save_rollout_video(
                video_dir,
                replay_images,
                f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                success=success,
                task_description=task_description,
            )
        if visualize_future_video:
            if len(predicted_future_video_clips) == 0:
                logging.warning(
                    "No predicted future frames collected for task %s trial %s.",
                    cfg.EVALUATION.task_id,
                    trial_idx,
                )
            else:
                all_gt_frames = []
                all_pred_frames = []
                for clip in predicted_future_video_clips:
                    all_gt_frames.extend(clip["gt_frames"])
                    all_pred_frames.extend(clip["pred_frames"])
                    save_prediction_video(
                        predicted_video_dir,
                        clip["gt_frames"],
                        clip["pred_frames"],
                        f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                        clip["replan_idx"],
                        success=success,
                        task_description=task_description,
                    )
                save_prediction_video(
                    predicted_video_dir,
                    all_gt_frames,
                    all_pred_frames,
                    f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                    "all",
                    success=success,
                    task_description=task_description,
                )

    if visualize_future_video:
        valid_episode_psnr = [x for x in results["episode_future_video_psnr"] if x is not None]
        if len(valid_episode_psnr) > 0:
            results["future_video_psnr_mean"] = float(np.mean(valid_episode_psnr))
    results["mean_completion_steps"] = float(np.mean(results["completion_steps"]))
    return results


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def eval_single_process(cfg: DictConfig):
    start_time = time.time()
    partial_state = PartialState()
    partial_state.config = cfg

    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)

    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None.")
    _validate_visualize_future_video_cfg(cfg)

    env_num = int(cfg.EVALUATION.get("env_num", 1))
    if env_num != 1:
        raise ValueError(
            "Only env_num=1 is supported in eval_libero_single.py. "
            "Use run_libero_manager/run_libero_parallel_test.sh for multi-GPU task parallelism."
        )

    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    checkpoint_payload = _load_model_checkpoint(model, str(cfg.ckpt))
    checkpoint_training_strategy = checkpoint_payload.get("training_strategy")
    configured_training_strategy = getattr(model, "training_strategy", None)
    if (
        checkpoint_training_strategy is not None
        and configured_training_strategy is not None
        and str(checkpoint_training_strategy) != str(configured_training_strategy)
    ):
        raise ValueError(
            "checkpoint/evaluation training strategy mismatch: "
            f"checkpoint={checkpoint_training_strategy} "
            f"configured={configured_training_strategy}"
        )
    checkpoint_causal_mode = checkpoint_payload.get("causal_mode")
    configured_causal_mode = getattr(model, "causal_mode", None)
    if (
        checkpoint_causal_mode is not None
        and configured_causal_mode is not None
        and str(checkpoint_causal_mode) != str(configured_causal_mode)
    ):
        raise ValueError(
            "checkpoint/evaluation causal mode mismatch: "
            f"checkpoint={checkpoint_causal_mode} configured={configured_causal_mode}"
        )
    model = model.to(model_device).eval()
    if bool(cfg.EVALUATION.get("merge_video_lora", False)):
        merge = getattr(model, "merge_video_lora_", None)
        if merge is None:
            raise ValueError("merge_video_lora=true but model has no LoRA merge support")
        merged_projections = int(merge())
        if merged_projections <= 0:
            raise ValueError("merge_video_lora=true but no LoRA projections were merged")
        logging.info("Merged %d video LoRA projections for inference", merged_projections)

    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    logging.info("Using dataset stats: %s", dataset_stats_path)

    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    if action_horizon_cfg is None:
        action_horizon = int(cfg.data.train.num_frames) - 1
    else:
        action_horizon = int(action_horizon_cfg)
    if action_horizon <= 0:
        raise ValueError(f"EVALUATION.action_horizon must be positive, got {action_horizon}")

    video_size = cfg.data.train.get("video_size", [224, 224])
    if len(video_size) != 2:
        raise ValueError(f"data.train.video_size must be [H, W], got {video_size}")
    input_h = int(video_size[0])
    input_w = int(video_size[1])
    _, task, initial_states_path = resolve_libero_task_and_initial_states(cfg)
    evaluation_fingerprint = _build_result_fingerprint(
        cfg,
        configured_causal_mode=configured_causal_mode,
        dataset_stats_path=dataset_stats_path,
        task=task,
        initial_states_path=initial_states_path,
    )

    local_log_dir = Path(cfg.EVALUATION.output_dir)
    local_log_dir.mkdir(parents=True, exist_ok=True)
    video_dir = local_log_dir / cfg.EVALUATION.task_suite_name / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    predicted_video_dir = local_log_dir / cfg.EVALUATION.task_suite_name / "predicted_videos"
    if bool(cfg.EVALUATION.get("visualize_future_video", False)):
        predicted_video_dir.mkdir(parents=True, exist_ok=True)

    # LIBERO's helper calls ``torch.load`` without an explicit ``weights_only``
    # argument. PyTorch >=2.6 defaults that argument to True, while the trusted
    # official LIBERO init-state files contain NumPy arrays. Load the same file
    # explicitly so evaluation works without a process-wide unsafe-load flag.
    initial_states = torch.load(initial_states_path, weights_only=False)

    while len(initial_states) < int(cfg.EVALUATION.num_trials):
        initial_states.extend(initial_states[: (int(cfg.EVALUATION.num_trials) - len(initial_states))])

    memory_cfg = cfg.EVALUATION.get("memory", None)
    memory_config = (
        OmegaConf.to_container(memory_cfg, resolve=True)
        if OmegaConf.is_config(memory_cfg)
        else dict(memory_cfg or {})
    )
    results = {
        "task_suite": cfg.EVALUATION.task_suite_name,
        "task_id": cfg.EVALUATION.task_id,
        "task_description": None,
        "successes": 0,
        "total_episodes": int(cfg.EVALUATION.num_trials),
        "gpu_id": int(cfg.gpu_id),
        "success_episodes": [],
        "failure_episodes": [],
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration": 0,
        "checkpoint": str(cfg.ckpt),
        "checkpoint_step": checkpoint_payload.get("step"),
        "model_config": {
            "causal_mode": configured_causal_mode,
            "history_training_mode": getattr(model, "history_training_mode", None),
            "training_strategy": getattr(model, "training_strategy", None),
        },
        "memory_config": memory_config,
        "evaluation_fingerprint": evaluation_fingerprint,
    }

    logging.info("Running LIBERO evaluation with env_num=1")
    task_results = run_single_task(
        task=task,
        initial_states=initial_states,
        model=model,
        processor=processor,
        cfg=cfg,
        video_dir=video_dir,
        predicted_video_dir=predicted_video_dir,
        action_horizon=action_horizon,
        input_w=input_w,
        input_h=input_h,
        model_device=model_device,
    )
    results.update(task_results)

    results["duration"] = time.time() - start_time
    output_dir = Path(cfg.EVALUATION.output_dir) / cfg.EVALUATION.task_suite_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"gpu{cfg.gpu_id}_task{cfg.EVALUATION.task_id}_results.json"

    atomic_write_json(output_file, results, indent=4, encoder_cls=NumpyEncoder)

    print(
        f"Task {cfg.EVALUATION.task_id} completed: "
        f"{results['successes']}/{cfg.EVALUATION.num_trials} successes"
    )
    if results.get("future_video_psnr_mean") is not None:
        print(f"Task {cfg.EVALUATION.task_id} future-video PSNR mean: {results['future_video_psnr_mean']:.4f}")
    print(f"Time taken: {results['duration']:.2f} seconds")
    return results


if __name__ == "__main__":
    eval_single_process()
