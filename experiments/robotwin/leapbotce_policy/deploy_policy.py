"""LeapBotCE policy adapter for the official RoboTwin eval loop.

Same `policy.<name>` contract as fastwam_policy (get_model / eval /
reset_model at module level), but inference follows the LeapBotCE cloud-edge
protocol from experiments/libero/eval_libero_cloudedge.py:

- cloud stream: the DELAYED composite observation (+ its proprio) is encoded
  once per replan into a MoT planning cache (`encode_cloud`);
- edge stream: the CURRENT composite is split into the three RoboTwin views
  (head / left wrist / right wrist, each resized to 224) exactly like
  CloudEdgeRobotVideoDataset does at training time, and actions are denoised
  from the cache with fresh edge vision (`infer_action_edge`).

Delay only ever applies to the cloud stream — edge vision and edge proprio are
always fresh. The delay machinery (env-gated ObservationDelayHistory with a
per-episode RandomState) is identical to fastwam_policy, with LEAPBOTCE_*
env names:
  LEAPBOTCE_MAX_DELAY_STEPS  (0 disables delay -> cloud == edge == current)
  LEAPBOTCE_DELAY_SEED       (default 1234; rng = RandomState(seed + episode))
  LEAPBOTCE_DELAY_LOG_DIR    (writes <task>_delay_log.jsonl per replan)
  LEAPBOTCE_TASK_NAME        (log label)
"""

import copy
import importlib.util
import json
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
# Keep deployment independent of the training dataset/video decoder stack.
DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

logger = logging.getLogger(__name__)


def _load_observation_delay_history() -> type:
    """Return the ObservationDelayHistory class shared with the LIBERO entry.

    Imported through the module file path so the robotwin policy works both
    under hydra (arbitrary sys.path) and direct runs.
    """
    delay_module_path = (
        PROJECT_ROOT / "experiments" / "libero" / "cloudedge_delay.py"
    )
    module_spec = importlib.util.spec_from_file_location(
        "fastwam_robotwin_cloudedge_delay", delay_module_path
    )
    delay_module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(delay_module)
    return delay_module.ObservationDelayHistory


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool value: {value}")


def _parse_optional_int(value: Any) -> Optional[int]:
    if _is_none_like(value):
        return None
    return int(value)


def _parse_optional_float(value: Any) -> Optional[float]:
    if _is_none_like(value):
        return None
    return float(value)


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


def _resolve_sim_cfg_name(sim_cfg_path: Optional[str], sim_cfg_name: Optional[str]) -> str:
    configs_root = (PROJECT_ROOT / "configs").resolve()
    if not _is_none_like(sim_cfg_path):
        cfg_path = Path(str(sim_cfg_path)).expanduser().resolve()
        try:
            relative = cfg_path.relative_to(configs_root)
        except ValueError as exc:
            raise ValueError(
                f"`sim_cfg_path` must be under {configs_root}, got: {cfg_path}"
            ) from exc
        return relative.as_posix()

    if _is_none_like(sim_cfg_name):
        return "sim_robotwin_leapbotce.yaml"
    return str(sim_cfg_name)


def _compose_sim_cfg(
    sim_cfg_path: Optional[str],
    sim_cfg_name: Optional[str],
    sim_task: Optional[str],
) -> DictConfig:
    config_name = _resolve_sim_cfg_name(sim_cfg_path=sim_cfg_path, sim_cfg_name=sim_cfg_name)
    configs_root = (PROJECT_ROOT / "configs").resolve()
    overrides = []
    if not _is_none_like(sim_task):
        overrides.append(f"task={str(sim_task)}")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base="1.3", config_dir=str(configs_root)):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def _resolve_dataset_stats_path(dataset_stats_path: Optional[str]) -> Path:
    if _is_none_like(dataset_stats_path):
        raise FileNotFoundError(
            "`dataset_stats_path` is required. "
            "Please pass it from eval entrypoint overrides."
        )
    resolved = Path(str(dataset_stats_path)).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset stats path not found: {resolved}")
    return resolved


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    resized = pil_image.resize(size_wh, resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


class WorldActionLeapBotCEPolicy:
    def __init__(
        self,
        model_cfg: DictConfig,
        processor_cfg: DictConfig,
        checkpoint_path: str,
        dataset_stats_path: Path,
        device: str,
        model_dtype: torch.dtype,
        action_horizon: int,
        replan_steps: int,
        num_inference_steps: int,
        sigma_shift: Optional[float],
        seed: Optional[int],
        rand_device: str,
        timing_enabled: bool,
    ) -> None:
        model_cfg_copy = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=True))
        model_cfg_copy.load_text_encoder = True

        self.model = instantiate(model_cfg_copy, model_dtype=model_dtype, device=device)
        self.model.load_checkpoint(checkpoint_path)
        self.model = self.model.to(device).eval()

        self.processor: FastWAMProcessor = instantiate(processor_cfg).eval()
        dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
        self.processor.set_normalizer_from_stats(dataset_stats)

        self.action_horizon = int(action_horizon)
        self.replan_steps = int(max(1, min(replan_steps, action_horizon)))
        self.num_inference_steps = int(num_inference_steps)
        self.sigma_shift = sigma_shift
        self.seed = seed
        self.rand_device = str(rand_device)
        self.timing_enabled = bool(timing_enabled)

        self.pending_actions: deque[np.ndarray] = deque()
        self.episode_count = 0
        self.step_count = 0
        self._timing_rollout = {"infer_s": 0.0, "sim_s": 0.0}

        # LeapBotCE prompts are the 10 seen task instructions (constant within
        # an episode): encode each once with the in-process T5 and reuse the
        # (context, context_mask) tensors across replans.
        self._prompt_cache: Dict[str, tuple] = {}

        # CloudEdgeVLA-style cloud-stream frame delay (opt-in via env vars).
        # Semantics mirror fastwam_policy exactly: per-step deep-copied
        # observations in a bounded deque; at every replan the cloud
        # observation is hist[-1 - d] with d ~ U{1..min(D, len-1)} from a
        # per-episode RandomState(delay_seed + episode_index). Only the cloud
        # stream is delayed; edge vision/proprio always see the current step.
        self._delay_cls = _load_observation_delay_history()
        self._max_delay_steps = int(os.environ.get("LEAPBOTCE_MAX_DELAY_STEPS", "0") or 0)
        self._delay_seed = int(os.environ.get("LEAPBOTCE_DELAY_SEED", "1234") or 0)
        self._delay_log_dir = os.environ.get("LEAPBOTCE_DELAY_LOG_DIR") or None
        self._delay_task = os.environ.get("LEAPBOTCE_TASK_NAME", "task")
        self._delay_history: Optional[deque] = None
        self._delay_rng = None
        self._delay_replan_idx = 0
        if self._max_delay_steps > 0 and self._delay_log_dir:
            os.makedirs(self._delay_log_dir, exist_ok=True)
        if self._max_delay_steps > 0:
            logger.info(
                "LeapBotCE cloud-stream delay enabled | max_delay_steps=%d | "
                "delay_seed=%d | log_dir=%s",
                self._max_delay_steps,
                self._delay_seed,
                self._delay_log_dir,
            )

        logger.info(
            "Initialized WorldActionLeapBotCEPolicy | ckpt=%s | stats=%s | "
            "horizon=%d | replan=%d | edge_views=%d",
            checkpoint_path,
            dataset_stats_path,
            self.action_horizon,
            self.replan_steps,
            int(self.model.edge_num_views),
        )

    def _get_prompt_context(self, instruction: str):
        prompt = DEFAULT_PROMPT.format(task=instruction)
        cached = self._prompt_cache.get(prompt)
        if cached is None:
            with torch.no_grad():
                cached = self.model.encode_prompt(prompt)
            self._prompt_cache[prompt] = cached
        return cached

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("Expected exactly one merged state key in shape_meta['state'].")
        state_key = state_meta[0]["key"]

        state_batch = {"state": {state_key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}}
        state_batch = self.processor.action_state_transform(state_batch)
        state_batch = self.processor.normalizer.forward(state_batch)
        return state_batch["state"][state_key]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"Expected action tensor [B,T,D], got {tuple(action.shape)}")

        action_meta = self.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("Expected exactly one merged action key in shape_meta['action'].")

        action_key = action_meta[0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][action_key]
        denorm = normalizer.backward(action.to(dtype=torch.float32, device="cpu"))
        return denorm.numpy()

    def _build_robotwin_image_tensor(self, observation: Dict[str, Any]) -> torch.Tensor:
        obs_data = observation["observation"]
        head = _resize_rgb(obs_data["head_camera"]["rgb"], (320, 256))
        left = _resize_rgb(obs_data["left_camera"]["rgb"], (160, 128))
        right = _resize_rgb(obs_data["right_camera"]["rgb"], (160, 128))
        bottom = np.concatenate([left, right], axis=1)
        image = np.concatenate([head, bottom], axis=0)  # [384, 320, 3]

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(
            device=self.model.device,
            dtype=self.model.torch_dtype,
        )
        image_tensor = image_tensor * (2.0 / 255.0) - 1.0
        return image_tensor

    def _split_robotwin_views(self, composite: torch.Tensor) -> torch.Tensor:
        """Replicate CloudEdgeRobotVideoDataset's RoboTwin view split.

        Training builds edge views from the finished [384, 320] composite:
        the top ``h - h//3`` rows are the head camera and the bottom row is
        the left/right wrist pair. Each view is bilinearly resized to
        SigLIP's 224x224 input in float32 exactly as the dataset does on its
        dataloader workers (the EdgeVisionEncoder repeats the same resize
        internally, where it is a no-op on already-224 inputs), then the
        views are stacked to [B, 3, 3, 224, 224]. Splitting the normalized
        composite is exact because every op involved is element-wise.
        """
        # Dataset operates on a [C, H, W] cloud frame; drop the batch dim and
        # mirror its slicing 1:1 (row-slices of dim1 would otherwise cut the
        # 3-channel axis and yield empty views).
        image = composite.detach().float()
        batched = image.ndim == 4
        if batched:
            image = image.squeeze(0)
        height, width = image.shape[-2], image.shape[-1]
        top_h = height - height // 3
        views = [
            image[:, :top_h],
            image[:, top_h:, : width // 2],
            image[:, top_h:, width // 2 :],
        ]
        current_views = torch.stack(
            [
                F.interpolate(
                    view.unsqueeze(0), size=(224, 224),
                    mode="bilinear", align_corners=False,
                ).squeeze(0)
                for view in views
            ],
            dim=0,
        )
        if batched:
            current_views = current_views.unsqueeze(0)
        return current_views.to(self.model.device, self.model.torch_dtype)

    def _infer_action_chunk(
        self, obs_for_plan: Dict[str, Any], observation: Dict[str, Any], instruction: str
    ) -> np.ndarray:
        """One replan: delayed cloud obs -> cache; current obs -> edge views."""
        cloud_image = self._build_robotwin_image_tensor(obs_for_plan)
        current_image = self._build_robotwin_image_tensor(observation)
        cloud_proprio = self._normalize_state(
            np.asarray(obs_for_plan["joint_action"]["vector"], dtype=np.float32)
        )
        edge_proprio = self._normalize_state(
            np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        )
        context, context_mask = self._get_prompt_context(instruction)
        current_views = self._split_robotwin_views(current_image)

        infer_t0 = time.perf_counter() if self.timing_enabled else 0.0
        with torch.no_grad():
            cache = self.model.encode_cloud(
                cloud_image,
                context=context,
                context_mask=context_mask,
                cloud_proprio=cloud_proprio,
            )
            pred = self.model.infer_action_edge(
                cache,
                current_views,
                self.action_horizon,
                edge_proprio=edge_proprio,
                num_inference_steps=self.num_inference_steps,
                sigma_shift=self.sigma_shift,
                seed=self.seed,
                rand_device=self.rand_device,
            )
        if self.timing_enabled:
            self._timing_rollout["infer_s"] += time.perf_counter() - infer_t0

        action_tensor = pred["action"]  # [T, D]
        action_chunk = self._denormalize_action(action_tensor)[0]  # [T, D]
        return action_chunk

    def _fill_action_queue(
        self,
        obs_for_plan: Dict[str, Any],
        observation: Dict[str, Any],
        instruction: str,
    ) -> None:
        action_chunk = self._infer_action_chunk(
            obs_for_plan=obs_for_plan, observation=observation, instruction=instruction
        )
        n_exec = min(self.replan_steps, action_chunk.shape[0])
        for i in range(n_exec):
            self.pending_actions.append(np.asarray(action_chunk[i], dtype=np.float32))

    def should_request_observation(self) -> bool:
        return not self.pending_actions

    def _delay_log(self, episode_idx: int, delay: int) -> None:
        if not self._delay_log_dir:
            return
        record = {
            "task": self._delay_task,
            "episode": int(episode_idx),
            "replan_idx": int(self._delay_replan_idx),
            "delay": int(delay),
            "max_delay_steps": int(self._max_delay_steps),
            "delay_seed": int(self._delay_seed),
        }
        path = os.path.join(self._delay_log_dir, f"{self._delay_task}_delay_log.jsonl")
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def step(self, task_env, observation: Optional[Dict[str, Any]]) -> None:
        if observation is not None and self._delay_history is not None:
            self._delay_history.append(copy.deepcopy(observation))

        if not self.pending_actions:
            if observation is None:
                raise ValueError(
                    "Observation is required when action queue is empty "
                    "(replan step for leapbotce)."
                )
            obs_for_plan = observation
            if self._delay_history is not None:
                obs_for_plan, delay = self._delay_history.sample(self._delay_rng)
                self._delay_log(self.episode_count - 1, delay)
                self._delay_replan_idx += 1
            instruction = task_env.get_instruction()
            # Dual-stream inference: obs_for_plan (possibly delayed) feeds the
            # cloud cache; the current observation feeds the edge stream.
            self._fill_action_queue(
                obs_for_plan=obs_for_plan, observation=observation, instruction=instruction
            )

        if not self.pending_actions:
            logger.warning("No action generated; skip current eval step.")
            return

        action = self.pending_actions.popleft()
        sim_t0 = time.perf_counter() if self.timing_enabled else 0.0
        task_env.take_action(action, action_type="qpos")
        if self.timing_enabled:
            self._timing_rollout["sim_s"] += time.perf_counter() - sim_t0
        self.step_count += 1

    def reset_timing_rollout(self) -> None:
        self._timing_rollout["infer_s"] = 0.0
        self._timing_rollout["sim_s"] = 0.0

    def get_timing_rollout(self) -> Dict[str, float]:
        return {
            "infer_s": float(self._timing_rollout["infer_s"]),
            "sim_s": float(self._timing_rollout["sim_s"]),
        }

    def reset(self) -> None:
        self.pending_actions.clear()
        episode_idx = self.episode_count
        self.episode_count += 1
        self.step_count = 0
        self.reset_timing_rollout()
        if self._max_delay_steps > 0:
            self._delay_history = self._delay_cls(self._max_delay_steps)
            self._delay_rng = np.random.RandomState(self._delay_seed + episode_idx)
            self._delay_replan_idx = 0


def encode_obs(observation: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return observation


def get_model(usr_args: Dict[str, Any]):
    sim_cfg_path = usr_args.get("sim_cfg_path")
    sim_cfg_name = usr_args.get("sim_cfg_name")
    sim_task = usr_args.get("sim_task")
    cfg = _compose_sim_cfg(
        sim_cfg_path=sim_cfg_path,
        sim_cfg_name=sim_cfg_name,
        sim_task=sim_task,
    )

    checkpoint_path = usr_args.get("ckpt_setting")
    if _is_none_like(checkpoint_path):
        raise ValueError("`ckpt_setting` is required and must be a valid checkpoint path.")

    device = str(usr_args.get("device") or cfg.EVALUATION.get("device") or "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; fallback device to cpu.")
        device = "cpu"

    mixed_precision = str(usr_args.get("mixed_precision") or cfg.get("mixed_precision", "bf16"))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)

    dataset_stats_path = _resolve_dataset_stats_path(
        dataset_stats_path=usr_args.get("dataset_stats_path"),
    )

    action_horizon = _parse_optional_int(usr_args.get("action_horizon"))
    if action_horizon is None:
        eval_horizon = _parse_optional_int(cfg.EVALUATION.get("action_horizon"))
        action_horizon = eval_horizon if eval_horizon is not None else int(cfg.data.train.num_frames) - 1
    if action_horizon <= 0:
        raise ValueError(f"`action_horizon` must be positive, got {action_horizon}")

    replan_steps = _parse_optional_int(usr_args.get("replan_steps"))
    if replan_steps is None:
        replan_steps = int(cfg.EVALUATION.get("replan_steps", 8))

    num_inference_steps = _parse_optional_int(usr_args.get("num_inference_steps"))
    if num_inference_steps is None:
        num_inference_steps = int(cfg.EVALUATION.get("num_inference_steps", cfg.eval_num_inference_steps))

    sigma_shift = _parse_optional_float(usr_args.get("sigma_shift"))
    if sigma_shift is None:
        sigma_shift = _parse_optional_float(cfg.EVALUATION.get("sigma_shift"))

    seed = _parse_optional_int(usr_args.get("seed"))
    rand_device = str(usr_args.get("rand_device", cfg.EVALUATION.get("rand_device", "cpu")))
    timing_enabled = _parse_bool(
        usr_args.get("timing_enabled", cfg.EVALUATION.get("timing_enabled", False))
    )

    policy = WorldActionLeapBotCEPolicy(
        model_cfg=cfg.model,
        processor_cfg=cfg.data.train.processor,
        checkpoint_path=str(checkpoint_path),
        dataset_stats_path=dataset_stats_path,
        device=device,
        model_dtype=model_dtype,
        action_horizon=action_horizon,
        replan_steps=replan_steps,
        num_inference_steps=num_inference_steps,
        sigma_shift=sigma_shift,
        seed=seed,
        rand_device=rand_device,
        timing_enabled=timing_enabled,
    )
    return policy


def eval(TASK_ENV, model, observation: Optional[Dict[str, Any]]):
    obs = encode_obs(observation)
    model.step(TASK_ENV, obs)


def reset_model(model):
    model.reset()
