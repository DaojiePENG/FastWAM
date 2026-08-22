"""LIBERO evaluation with delayed cloud observations and current edge vision."""

import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.libero import eval_libero_single as base
from experiments.libero.cloudedge_delay import ObservationDelayHistory
from experiments.libero.libero_utils import get_libero_dummy_action, get_libero_image


DELAY_SAMPLES = []


def _predict(obs_cloud, obs_current, task_description, model, processor, cfg,
             action_horizon, input_w, input_h, model_device):
    prompt = base.DEFAULT_PROMPT.format(task=task_description)
    cloud_image, cloud_proprio, _ = base._obs_to_model_input(
        obs_cloud, cfg, processor, input_w, input_h, model_device, model.torch_dtype)
    current_image, edge_proprio, imgs = base._obs_to_model_input(
        obs_current, cfg, processor, input_w, input_h, model_device, model.torch_dtype)
    current_views = model._split_views(current_image, model.edge_num_views)
    steps = int(cfg.EVALUATION.get("num_inference_steps", cfg.eval_num_inference_steps))
    with torch.no_grad():
        cache = model.encode_cloud(
            cloud_image, prompt=prompt, cloud_proprio=cloud_proprio
        )
        pred = model.infer_action_edge(
            cache, current_views, action_horizon,
            edge_proprio=edge_proprio,
            num_inference_steps=steps,
            sigma_shift=cfg.EVALUATION.get("sigma_shift"),
            seed=None if cfg.get("seed") is None else int(cfg.seed),
            rand_device=str(cfg.EVALUATION.get("rand_device", "cpu")),
        )
    action = base._denormalize_action(pred["action"], processor)[0]
    action[..., -1] = action[..., -1] * 2 - 1
    action = base.invert_gripper_action(action)
    if bool(cfg.EVALUATION.get("binarize_gripper", False)):
        action[..., -1] = np.sign(action[..., -1])
    return action, imgs


def run_single_episode(env, initial_state, task_description, model, processor, cfg,
                       episode_idx, *, action_horizon, input_w, input_h, model_device):
    max_steps = base._get_max_steps(cfg.EVALUATION.task_suite_name)
    wait_steps = int(cfg.EVALUATION.get("num_steps_wait", 5))
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    max_delay = int(cfg.EVALUATION.get("max_delay_steps", 0))
    rng = np.random.RandomState(int(cfg.get("seed", 0) or 0) + episode_idx)
    history = ObservationDelayHistory(max_delay)
    env.reset()
    obs = env.set_init_state(initial_state)
    replay, pending = [], []
    done = False
    t = 0
    pbar = tqdm(total=max_steps + wait_steps, desc=f"Episode {episode_idx + 1}")
    while t < max_steps + wait_steps:
        pbar.update(1)
        history.append(obs)
        if t < wait_steps:
            obs, _, done, _ = env.step(get_libero_dummy_action())
            t += 1
            continue
        if not pending:
            delayed_obs, delay = history.sample(rng)
            DELAY_SAMPLES.append(delay)
            chunk, imgs = _predict(
                delayed_obs, obs, task_description, model, processor, cfg,
                action_horizon, input_w, input_h, model_device)
            pending = chunk[:replan_steps].tolist()
            replay.append(imgs.copy())
        else:
            replay.append(get_libero_image(obs).copy())
        obs, _, done, _ = env.step(pending.pop(0))
        if done:
            break
        t += 1
    pbar.close()
    return bool(done), replay, [], None


_base_run_single_task = base.run_single_task


def run_single_task(*args, **kwargs):
    DELAY_SAMPLES.clear()
    results = _base_run_single_task(*args, **kwargs)
    values = np.asarray(DELAY_SAMPLES, dtype=np.int64)
    results["delay_samples"] = values.tolist()
    results["delay_mean"] = float(values.mean()) if values.size else 0.0
    results["delay_histogram"] = {
        str(delay): int((values == delay).sum()) for delay in sorted(set(values.tolist()))
    }
    return results


base.run_single_episode = run_single_episode
base.run_single_task = run_single_task


if __name__ == "__main__":
    base.eval_single_process()
