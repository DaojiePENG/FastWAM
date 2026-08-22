import math

import numpy as np
import torch

from .robot_video_dataset import RobotVideoDataset


def sample_stale_index(current_index: int, max_delay: int, is_pad: torch.Tensor,
                       rng=np.random) -> tuple[int, int]:
    """Sample a valid same-window history index and its delay."""
    if current_index < 0 or max_delay < 0:
        raise ValueError("current_index and max_delay must be non-negative")
    earliest = max(0, current_index - max_delay)
    valid = [index for index in range(earliest, current_index)
             if not bool(is_pad[index].item())]
    if not valid:
        return current_index, 0
    stale_index = int(valid[int(rng.randint(0, len(valid)))])
    return stale_index, current_index - stale_index


class CloudEdgeRobotVideoDataset(torch.utils.data.Dataset):
    """Re-anchor a contiguous LeRobot window for stale/current supervision."""

    def __init__(self, max_delay_steps: int = 20, action_horizon: int = 32, **kwargs):
        self.max_delay_steps = int(max_delay_steps)
        self.action_horizon = int(action_horizon)
        if self.max_delay_steps < 1 or self.action_horizon < 1:
            raise ValueError("max_delay_steps and action_horizon must be positive")

        # RobotVideoDataset requires its number of transitions to be divisible
        # by four. Keep every environment frame so delays remain step-exact.
        transitions = int(math.ceil(
            (self.max_delay_steps + self.action_horizon) / 4.0
        ) * 4)
        kwargs["num_frames"] = transitions + 1
        kwargs["action_video_freq_ratio"] = 1
        processor = kwargs.get("processor")
        if processor is not None:
            processor.num_obs_steps = transitions + 1
            processor.num_image_steps = transitions + 1
        self.dataset = RobotVideoDataset(**kwargs)

    @property
    def lerobot_dataset(self):
        return self.dataset.lerobot_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        current = self.max_delay_steps
        end = current + self.action_horizon
        if sample["video"].shape[1] <= current:
            raise ValueError("Underlying video window is shorter than the current anchor")
        stale, delay = sample_stale_index(
            current, self.max_delay_steps, sample["image_is_pad"])

        current_cloud = sample["video"][:, current]
        stale_cloud = sample["video"][:, stale]
        if current_cloud.shape[-1] % 2:
            raise ValueError("LeapBotCE expects a horizontal two-camera image")
        current_views = torch.stack(current_cloud.chunk(2, dim=-1), dim=0)
        output = {
            "cloud_current_image": current_cloud,
            "cloud_stale_image": stale_cloud,
            # The cloud observation is a temporally consistent snapshot: its
            # visual input and state come from the same history index.
            "cloud_current_proprio": sample["proprio"][current:current + 1],
            "cloud_stale_proprio": sample["proprio"][stale:stale + 1],
            "edge_current_views": current_views,
            "edge_current_proprio": sample["proprio"][current:current + 1],
            "cloud_delay_steps": torch.tensor(delay, dtype=torch.long),
            "action": sample["action"][current:end],
            "action_is_pad": sample["action_is_pad"][current:end],
            # Retain the established key for callers outside LeapBotCE.
            "proprio": sample["proprio"][current:current + 1],
            "prompt": sample["prompt"],
        }
        if "context" in sample:
            output["context"] = sample["context"]
            output["context_mask"] = sample["context_mask"]
        return output
