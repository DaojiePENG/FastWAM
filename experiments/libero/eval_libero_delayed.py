"""LIBERO evaluation for a single-stream policy with delayed observations.

Unlike ``eval_libero_cloudedge.py``, this baseline delays the complete model
input (both camera views and proprioception).  It is therefore the appropriate
comparison for an ordinary FastWAM checkpoint that has no fresh edge branch.
"""

import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.libero import eval_libero_single as base
from experiments.libero.cloudedge_delay import ObservationDelayHistory
from experiments.libero.libero_utils import get_libero_dummy_action, get_libero_image


DELAY_SAMPLES: list[int] = []


def run_single_episode(
    env,
    initial_state,
    task_description,
    model,
    processor,
    cfg,
    episode_idx,
    *,
    action_horizon,
    input_w,
    input_h,
    model_device,
):
    if bool(cfg.EVALUATION.get("use_action_ensembler", False)):
        raise ValueError(
            "eval_libero_delayed.py currently requires "
            "EVALUATION.use_action_ensembler=false."
        )
    if bool(cfg.EVALUATION.get("visualize_future_video", False)):
        raise ValueError(
            "eval_libero_delayed.py currently requires "
            "EVALUATION.visualize_future_video=false."
        )

    max_steps = int(
        cfg.EVALUATION.get("max_episode_steps")
        or base._get_max_steps(cfg.EVALUATION.task_suite_name)
    )
    wait_steps = int(cfg.EVALUATION.get("num_steps_wait", 5))
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    max_delay = int(cfg.EVALUATION.get("max_delay_steps", 0))
    rng = np.random.RandomState(int(cfg.get("seed", 0) or 0) + episode_idx)
    history = ObservationDelayHistory(max_delay)

    env.reset()
    obs = env.set_init_state(initial_state)
    replay_images = []
    pending_actions = []
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

        if not pending_actions:
            delayed_obs, delay = history.sample(rng)
            DELAY_SAMPLES.append(delay)
            action_chunk, imgs, _ = base._predict_action_chunk(
                obs=delayed_obs,
                task_description=task_description,
                model=model,
                processor=processor,
                cfg=cfg,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
            )
            pending_actions = action_chunk[:replan_steps].tolist()
            replay_images.append(imgs.copy())
        else:
            replay_images.append(get_libero_image(obs).copy())

        obs, _, done, _ = env.step(pending_actions.pop(0))
        if done:
            break
        t += 1

    pbar.close()
    return bool(done), replay_images, [], None


_base_run_single_task = base.run_single_task


def run_single_task(*args, **kwargs):
    DELAY_SAMPLES.clear()
    results = _base_run_single_task(*args, **kwargs)
    cfg = kwargs.get("cfg")
    if cfg is None:
        cfg = args[4]
    values = np.asarray(DELAY_SAMPLES, dtype=np.int64)
    results["delay_protocol"] = "single_stream_full_observation"
    results["delay_max"] = int(cfg.EVALUATION.get("max_delay_steps", 0))
    results["delay_samples"] = values.tolist()
    results["delay_mean"] = float(values.mean()) if values.size else 0.0
    results["delay_histogram"] = {
        str(delay): int((values == delay).sum())
        for delay in sorted(set(values.tolist()))
    }
    return results


base.run_single_episode = run_single_episode
base.run_single_task = run_single_task


if __name__ == "__main__":
    base.eval_single_process()
