"""LeRobot episode-window sampling for LeapBot causal-history training."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torchvision.transforms.functional as transforms_F
from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from fastwam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset
from fastwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset
from leapbot_va.training import history_window_indices


def full_episode_sparse_offsets(
    *,
    max_history_blocks: int,
    replan_steps: int,
    current_action_horizon: int,
    current_video_offsets: list[int],
) -> tuple[list[int], list[int]]:
    """Return sparse observation and dense-action offsets for a full prefix.

    Historical images/proprio are needed only at replanning boundaries, while
    every historical executed action is required.  Keeping those schedules
    separate avoids decoding hundreds of unused intermediate video frames.
    """

    if max_history_blocks <= 0 or replan_steps <= 0 or current_action_horizon <= 0:
        raise ValueError("history/replan/horizon sizes must be positive")
    if not current_video_offsets or current_video_offsets[0] != 0:
        raise ValueError("current_video_offsets must start at zero")
    if current_video_offsets != sorted(set(current_video_offsets)):
        raise ValueError("current_video_offsets must be sorted and unique")
    history_steps = max_history_blocks * replan_steps
    observation_offsets = list(range(-history_steps, 0, replan_steps))
    observation_offsets.extend(int(value) for value in current_video_offsets)
    action_offsets = list(range(-history_steps, current_action_horizon))
    return observation_offsets, action_offsets


class LeapRobotVideoDataset(RobotVideoDataset):
    """Return a fixed-size causal prefix plus the original FastWAM target.

    Samples are restricted to true 10-step replanning boundaries.  Every item
    remains inside one LeRobot episode and contains zero-padded fixed-size
    history tensors plus ``history_valid_blocks`` for collation.  The current
    ``video`` and ``action`` fields retain FastWAM's 33-frame/32-action target.
    """

    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames=33,
        video_size=(384, 640),
        camera_key=None,
        processor=None,
        text_embedding_cache_dir=None,
        context_len=128,
        pretrained_norm_stats=None,
        val_set_proportion=0.05,
        is_training_set=False,
        global_sample_stride=1,
        action_video_freq_ratio: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal",
        override_instruction: Optional[str] = None,
        max_history_blocks: int = 8,
        min_history_blocks: int = 0,
        replan_steps: int = 10,
        history_seed: int = 42,
        full_episode_history: bool = False,
    ):
        if max_history_blocks < 0:
            raise ValueError("max_history_blocks must be non-negative")
        if min_history_blocks < 0 or min_history_blocks > max_history_blocks:
            raise ValueError("min_history_blocks must be in [0,max_history_blocks]")
        if replan_steps <= 0:
            raise ValueError("replan_steps must be positive")

        # Let the base class build transforms, text cache behavior, and the
        # release-compatible normalizer before replacing its temporal window.
        super().__init__(
            dataset_dirs=dataset_dirs,
            shape_meta=shape_meta,
            num_frames=num_frames,
            video_size=list(video_size),
            camera_key=camera_key,
            processor=processor,
            text_embedding_cache_dir=text_embedding_cache_dir,
            context_len=context_len,
            pretrained_norm_stats=pretrained_norm_stats,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
            action_video_freq_ratio=action_video_freq_ratio,
            skip_padding_as_possible=skip_padding_as_possible,
            max_padding_retry=max_padding_retry,
            concat_multi_camera=concat_multi_camera,
            override_instruction=override_instruction,
        )
        prepared_processor = self.lerobot_dataset.processor
        if prepared_processor is None:
            raise ValueError("LeapRobotVideoDataset requires a processor")

        self.max_history_blocks = int(max_history_blocks)
        self.min_history_blocks = int(min_history_blocks)
        self.replan_steps = int(replan_steps)
        self.history_seed = int(history_seed)
        self.full_episode_history = bool(full_episode_history)
        self.history_action_steps = self.max_history_blocks * self.replan_steps
        self.window_frames = self.history_action_steps + int(num_frames)

        resolved_shape_meta = OmegaConf.to_container(shape_meta, resolve=True)
        if self.full_episode_history:
            observation_offsets, action_offsets = full_episode_sparse_offsets(
                max_history_blocks=self.max_history_blocks,
                replan_steps=self.replan_steps,
                current_action_horizon=int(num_frames) - 1,
                current_video_offsets=list(self.video_sample_indices),
            )
            self.lerobot_dataset = BaseLerobotDataset(
                dataset_dirs=dataset_dirs,
                shape_meta=resolved_shape_meta,
                observation_offsets=observation_offsets,
                action_offsets=action_offsets,
                val_set_proportion=val_set_proportion,
                is_training_set=is_training_set,
                global_sample_stride=global_sample_stride,
                seed=history_seed,
            )
            prepared_processor.num_obs_steps = len(observation_offsets)
        else:
            self.lerobot_dataset = BaseLerobotDataset(
                dataset_dirs=dataset_dirs,
                shape_meta=resolved_shape_meta,
                obs_size=self.window_frames,
                action_size=self.window_frames - 1,
                past_obs_size=self.history_action_steps,
                past_action_size=self.history_action_steps,
                val_set_proportion=val_set_proportion,
                is_training_set=is_training_set,
                global_sample_stride=global_sample_stride,
                seed=history_seed,
            )
            prepared_processor.num_obs_steps = self.window_frames
        self.lerobot_dataset._set_return_images(True)
        self.lerobot_dataset.set_processor(prepared_processor)

        self._valid_replan_indices: list[int] = []
        self._episode_step: dict[int, int] = {}
        starts = self.lerobot_dataset.episode_data_index["from"].tolist()
        stops = self.lerobot_dataset.episode_data_index["to"].tolist()
        for start, stop in zip(starts, stops):
            for relative_step in range(0, int(stop - start), self.replan_steps):
                if (
                    self.full_episode_history
                    and relative_step // self.replan_steps >= self.max_history_blocks
                ):
                    raise ValueError(
                        "episode exceeds configured total replanning-block capacity: "
                        f"block={relative_step // self.replan_steps} "
                        f"capacity={self.max_history_blocks}"
                    )
                index = int(start + relative_step)
                self._valid_replan_indices.append(index)
                self._episode_step[index] = relative_step
        if not self._valid_replan_indices:
            raise ValueError("no episode replanning boundaries found")

    def __len__(self):
        return len(self._valid_replan_indices)

    def sampler_grouping_lengths(self) -> tuple[int, ...] | None:
        """Return exact prefix lengths for distributed compute-cost grouping.

        Full-episode history has a deterministic length for every dataset item,
        so grouping only changes permutation order and never the training
        distribution.  Random short-window ablations deliberately opt out because
        their length is sampled in ``_get`` and is not known to the sampler.
        """

        if not self.full_episode_history:
            return None
        return tuple(
            self._episode_step[index] // self.replan_steps
            for index in self._valid_replan_indices
        )

    def _choose_history_blocks(self, sample_index: int) -> int:
        current_block = self._episode_step[sample_index] // self.replan_steps
        if self.full_episode_history:
            if current_block >= self.max_history_blocks:
                raise ValueError(
                    "episode exceeds configured total replanning-block capacity: "
                    f"block={current_block} capacity={self.max_history_blocks}"
                )
            return current_block
        upper = min(self.max_history_blocks, current_block)
        lower = min(self.min_history_blocks, upper)
        # numpy's worker seeding keeps this reproducible across dataloader runs.
        return int(np.random.randint(lower, upper + 1))

    def _format_camera_video(self, full_video: torch.Tensor, indices: list[int]) -> torch.Tensor:
        if full_video.ndim == 5:
            video = full_video[:, indices]
            num_cameras, steps, channels, height, width = video.shape
        elif full_video.ndim == 4:
            video = full_video[indices].unsqueeze(0)
            num_cameras, steps, channels, height, width = video.shape
        else:
            raise ValueError(f"unexpected pixel_values shape: {tuple(full_video.shape)}")

        if self.concat_multi_camera == "robotwin":
            if num_cameras != 3:
                raise ValueError("robotwin camera layout requires exactly 3 cameras")
            top = transforms_F.resize(video[0], [256, 320], antialias=True)
            left = transforms_F.resize(video[1], [128, 160], antialias=True)
            right = transforms_F.resize(video[2], [128, 160], antialias=True)
            video = torch.cat([top, torch.cat([left, right], dim=-1)], dim=-2)
        elif num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)
            elif self.concat_multi_camera == "vertical":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)
            else:
                raise ValueError(f"invalid concat_multi_camera: {self.concat_multi_camera}")
        else:
            video = video.squeeze(0)

        video = self.resize_transform(video)
        video = self.crop_transform(video)
        video = self.normalize_transform(video)
        return video.permute(1, 0, 2, 3)  # [C,T,H,W]

    def _get(self, idx):
        mapped_idx = self._valid_replan_indices[int(idx)]
        sample = self.lerobot_dataset[mapped_idx]
        loaded_idx = int(sample.get("idx", -1))
        if loaded_idx != mapped_idx:
            raise RuntimeError(
                "underlying LeRobot loader substituted a different frame after an I/O "
                f"failure: requested={mapped_idx} loaded={loaded_idx}; refusing to attach "
                "causal metadata from another trajectory position"
            )
        history_blocks = self._choose_history_blocks(mapped_idx)
        current_block = self._episode_step[mapped_idx] // self.replan_steps
        if self.full_episode_history:
            observation_history_slots = self.max_history_blocks
            current_indices = list(
                range(
                    observation_history_slots,
                    observation_history_slots + len(self.video_sample_indices),
                )
            )
        else:
            offset = self.history_action_steps
            current_indices = [offset + value for value in self.video_sample_indices]
        video = self._format_camera_video(sample["pixel_values"], current_indices)
        current_image_is_pad = sample["image_is_pad"][current_indices]

        if self.full_episode_history:
            history_start = self.max_history_blocks - history_blocks
            history_indices = list(range(history_start, self.max_history_blocks))
            history_action_end = self.history_action_steps
            history_action_slice = slice(
                history_action_end - history_blocks * self.replan_steps,
                history_action_end,
            )
            absolute_history_positions = list(range(history_blocks))
        else:
            history_indices, history_action_slice, absolute_history_positions = history_window_indices(
                current_episode_step=self._episode_step[mapped_idx],
                history_blocks=history_blocks,
                replan_steps=self.replan_steps,
                current_window_offset=offset,
            )
        channels, _, height, width = video.shape
        history_video = torch.zeros(
            (channels, self.max_history_blocks, height, width), dtype=video.dtype
        )
        if history_indices:
            history_video[:, :history_blocks] = self._format_camera_video(
                sample["pixel_values"], history_indices
            )

        action_dim = int(sample["action"].shape[-1])
        proprio_dim = int(sample["proprio"].shape[-1])
        history_action = torch.zeros(
            (self.max_history_blocks, self.replan_steps, action_dim),
            dtype=sample["action"].dtype,
        )
        history_proprio = torch.zeros(
            (self.max_history_blocks, proprio_dim), dtype=sample["proprio"].dtype
        )
        history_valid = torch.zeros(self.max_history_blocks, dtype=torch.bool)
        history_positions = torch.full((self.max_history_blocks,), -1, dtype=torch.long)
        if history_blocks:
            history_action[:history_blocks] = sample["action"][history_action_slice].reshape(
                history_blocks, self.replan_steps, action_dim
            )
            history_proprio[:history_blocks] = sample["proprio"][history_indices]
            history_valid[:history_blocks] = True
            history_positions[:history_blocks] = torch.tensor(
                absolute_history_positions, dtype=torch.long
            )

        action_start = self.history_action_steps
        action_end = action_start + self.num_frames - 1
        action = sample["action"][action_start:action_end]
        action_is_pad = sample["action_is_pad"][action_start:action_end]
        if self.full_episode_history:
            # Only the current proprio is consumed by LeapBot training.  Keep
            # the public FastWAM-shaped field aligned with the action horizon.
            current_proprio = sample["proprio"][self.max_history_blocks]
            proprio = current_proprio.unsqueeze(0).expand(self.num_frames - 1, -1).clone()
            proprio_is_pad = torch.zeros(
                self.num_frames - 1, dtype=torch.bool, device=proprio.device
            )
        else:
            proprio = sample["proprio"][action_start:action_end]
            proprio_is_pad = sample["proprio_is_pad"][action_start:action_end]

        task = self.override_instruction or sample["instruction"]
        from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT

        instruction = DEFAULT_PROMPT.format(task=task)
        context, context_mask = self._get_cached_text_context(instruction)
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)
        return {
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": current_image_is_pad,
            "action_is_pad": action_is_pad,
            "proprio_is_pad": proprio_is_pad,
            "history_video": history_video,
            "history_action": history_action,
            "history_proprio": history_proprio,
            "history_valid_blocks": history_valid,
            "history_block_positions": history_positions,
            "current_block_position": torch.tensor(current_block, dtype=torch.long),
            "episode_step": torch.tensor(self._episode_step[mapped_idx], dtype=torch.long),
            "full_episode_history": torch.tensor(self.full_episode_history, dtype=torch.bool),
        }

    def __getitem__(self, idx):
        # RobotVideoDataset's generic fallback silently replaces failed items
        # with random samples.  That is unacceptable for causal episode prefixes:
        # an I/O error must fail fast instead of changing the trajectory/index.
        if int(idx) < 0 or int(idx) >= len(self):
            raise IndexError(f"Index {idx} out of bounds {len(self)}")
        return self._get(int(idx))
