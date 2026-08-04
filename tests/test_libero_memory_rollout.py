import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(_ROOT / "experiments" / "libero"))
_LIBERO_ROOT = _ROOT.parent / "LIBERO"
if _LIBERO_ROOT.is_dir():
    sys.path.insert(0, str(_LIBERO_ROOT))

from experiments.libero import eval_libero_single as evaluator  # noqa: E402


class _Memory:
    completed_blocks = 0
    retained_completed_blocks = 0
    cache_nbytes = 100


class _Model:
    torch_dtype = torch.float32

    def __init__(self):
        self.infer_calls = 0
        self.committed = []
        self.reset_called = False

    def create_memory(self, **kwargs):
        self.memory_kwargs = kwargs
        return _Memory()

    def infer_action(self, **kwargs):
        self.infer_calls += 1
        action = torch.arange(32 * 3, dtype=torch.float32).view(32, 3)
        return {
            "action": action,
            "memory": {
                "completed_blocks": 0,
                "retained_history_blocks": 0,
                "cache_bytes": 100,
                "transient_future_video_cache_bytes": 24,
            },
            "timing": {
                "conditioning_s": 0.01,
                "observation_prefill_s": 0.02,
                "action_setup_s": 0.03,
                "action_denoise_s": 0.04,
            },
        }

    def infer_joint(self, **kwargs):
        raise AssertionError("memory rollout must not run future-video inference")

    def commit_executed_actions(self, memory, actions, *, profile):
        assert profile is True
        self.committed.append(actions.clone())
        memory.completed_blocks += 1
        memory.retained_completed_blocks += 1
        memory.cache_nbytes = 20
        return {
            "executed_actions": int(actions.shape[0]),
            "completed_blocks": memory.completed_blocks,
            "retained_history_blocks": memory.retained_completed_blocks,
            "cache_bytes": memory.cache_nbytes,
            "commit_s": 0.05,
        }

    def reset_memory(self, memory):
        self.reset_called = True
        memory.cache_nbytes = 0


class _Env:
    action_spec = (
        np.array([-1.0, -0.5, -1.0], dtype=np.float32),
        np.array([1.0, 0.5, 1.0], dtype=np.float32),
    )

    def __init__(self, done_after):
        self.done_after = done_after
        self.actions = []

    def reset(self):
        return None

    def set_init_state(self, initial_state):
        return {"frame": 0}

    def step(self, action):
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        done = len(self.actions) >= self.done_after
        return {"frame": len(self.actions)}, 0.0, done, {}


class _NoopProgress:
    def __init__(self, *args, **kwargs):
        pass

    def update(self, amount):
        pass

    def close(self):
        pass


def test_memory_rollout_commits_only_executed_prefix_and_counts_observation_peak(
    monkeypatch,
):
    monkeypatch.setattr(evaluator, "tqdm", _NoopProgress)
    monkeypatch.setattr(
        evaluator,
        "_obs_to_model_input",
        lambda *args, **kwargs: (
            torch.zeros(1, 3, 16, 16),
            torch.zeros(2),
            {"frame": 0},
        ),
    )
    predicted_env_space = np.zeros((1, 32, 3), dtype=np.float32)
    predicted_env_space[0, :, 0] = np.linspace(-2.0, 2.0, 32)
    predicted_env_space[0, :, 1] = np.linspace(1.0, -1.0, 32)
    predicted_env_space[0, :, 2] = np.linspace(0.0, 1.0, 32)
    monkeypatch.setattr(
        evaluator,
        "_denormalize_action",
        lambda action, processor: predicted_env_space.copy(),
    )
    monkeypatch.setattr(
        evaluator,
        "executed_env_actions_to_model_space",
        lambda actions, processor: torch.as_tensor(actions).clone(),
    )

    cfg = OmegaConf.create(
        {
            "seed": 7,
            "eval_num_inference_steps": 2,
            "data": {"train": {"action_video_freq_ratio": 5}},
            "EVALUATION": {
                "task_suite_name": "libero_10",
                "replan_steps": 10,
                "num_steps_wait": 0,
                "use_action_ensembler": False,
                "visualize_future_video": False,
                "save_rollout_video": False,
                "binarize_gripper": True,
                "memory": {
                    "enabled": True,
                    "exit_depth": 30,
                    "causal_mode": "action_aggregator",
                    "max_history_blocks": 70,
                    "retained_history_blocks": None,
                },
            },
        }
    )
    env = _Env(done_after=3)
    model = _Model()

    success, _, future_clips, _, metrics = evaluator.run_single_episode(
        env,
        initial_state=np.zeros(1),
        task_description="test task",
        model=model,
        processor=object(),
        cfg=cfg,
        episode_idx=0,
        action_horizon=32,
        input_w=16,
        input_h=16,
        model_device="cpu",
    )

    assert success is True
    assert future_clips == []
    assert model.infer_calls == 1
    assert model.reset_called is True
    assert len(model.committed) == 1
    assert model.committed[0].shape == (3, 3)
    np.testing.assert_array_equal(model.committed[0].numpy(), np.stack(env.actions))
    assert metrics["control_steps"] == 3
    assert metrics["completed_blocks"] == 1
    assert metrics["peak_cache_bytes"] == 100
    assert metrics["peak_transient_future_video_cache_bytes"] == 24
    assert metrics["final_cache_bytes"] == 20

    replan = metrics["replans"][0]
    assert replan["future_video_condition"] == {}
    assert replan["commit"]["executed_actions"] == 3
    timing = replan["timing"]
    assert timing["total_inference_s"] >= 0
    assert abs(
        timing["total_inference_s"]
        - timing["input_preprocess_s"]
        - timing["model_inference_s"]
        - timing["action_postprocess_s"]
        - timing["latency_residual_s"]
    ) < 1e-9
