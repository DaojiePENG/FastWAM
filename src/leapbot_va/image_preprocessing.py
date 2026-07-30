"""Shared training/evaluation image preprocessing for LeapBot LIBERO inputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as transforms_F

from fastwam.datasets.dataset_utils import (
    CenterCrop,
    Normalize,
    ResizeSmallestSideAspectPreserving,
)


def format_processed_camera_video(
    video: torch.Tensor,
    *,
    concat_multi_camera: str,
    video_size: Sequence[int],
) -> torch.Tensor:
    """Apply FastWAM's camera layout and final video transform exactly once.

    Args:
        video: Processor-transformed camera tensors shaped ``[N,T,C,H,W]`` or
            a single camera shaped ``[T,C,H,W]`` in the ``[0,1]`` range.
        concat_multi_camera: FastWAM camera layout name.
        video_size: Final ``[H,W]`` model input size.

    Returns:
        Normalized ``[C,T,H,W]`` tensor in the ``[-1,1]`` range.
    """

    if len(video_size) != 2:
        raise ValueError(f"video_size must be [H,W], got {video_size}")
    target_height, target_width = (int(video_size[0]), int(video_size[1]))
    if target_height <= 0 or target_width <= 0:
        raise ValueError("video_size dimensions must be positive")

    if video.ndim == 4:
        video = video.unsqueeze(0)
    if video.ndim != 5:
        raise ValueError(
            "processed camera video must be [N,T,C,H,W] or [T,C,H,W], "
            f"got {tuple(video.shape)}"
        )
    num_cameras, _, channels, _, _ = video.shape
    if channels != 3:
        raise ValueError(f"camera tensors must have three channels, got {channels}")
    if not bool(torch.isfinite(video).all().item()):
        raise ValueError("processed camera video contains non-finite values")

    if concat_multi_camera == "robotwin":
        if num_cameras != 3:
            raise ValueError("robotwin camera layout requires exactly 3 cameras")
        top = transforms_F.resize(video[0], [256, 320], antialias=True)
        left = transforms_F.resize(video[1], [128, 160], antialias=True)
        right = transforms_F.resize(video[2], [128, 160], antialias=True)
        video = torch.cat([top, torch.cat([left, right], dim=-1)], dim=-2)
    elif num_cameras > 1:
        if concat_multi_camera == "horizontal":
            video = torch.cat([video[index] for index in range(num_cameras)], dim=-1)
        elif concat_multi_camera == "vertical":
            video = torch.cat([video[index] for index in range(num_cameras)], dim=-2)
        else:
            raise ValueError(f"invalid concat_multi_camera: {concat_multi_camera}")
    else:
        video = video.squeeze(0)

    video = ResizeSmallestSideAspectPreserving(
        args={"img_w": target_width, "img_h": target_height}
    )(video)
    video = CenterCrop(args={"img_w": target_width, "img_h": target_height})(video)
    video = Normalize(args={"mean": 0.5, "std": 0.5})(video)
    expected = (int(video.shape[0]), 3, target_height, target_width)
    if tuple(video.shape) != expected:
        raise ValueError(
            f"formatted video shape mismatch: expected {expected}, got {tuple(video.shape)}"
        )
    return video.permute(1, 0, 2, 3).contiguous()


def preprocess_uint8_libero_cameras(
    camera_images: Mapping[str, np.ndarray],
    *,
    processor: Any,
    concat_multi_camera: str,
    video_size: Sequence[int],
) -> torch.Tensor:
    """Run the configured FastWAM processor and shared final video transform.

    This is the closed-loop counterpart of ``BaseLerobotDataset`` followed by
    ``LeapRobotVideoDataset._format_camera_video``.  It deliberately uses the
    processor's configured validation transforms rather than a parallel PIL
    implementation.
    """

    image_meta = list(processor.shape_meta["images"])
    num_cameras = int(processor.num_output_cameras)
    if num_cameras <= 0 or num_cameras > len(image_meta):
        raise ValueError(
            f"num_output_cameras={num_cameras} is incompatible with "
            f"{len(image_meta)} image metadata entries"
        )
    transforms = (
        processor.train_transforms if processor.is_train else processor.val_transforms
    )
    processed = []
    for camera_index, meta in enumerate(image_meta[:num_cameras]):
        key = str(meta["key"])
        if key not in camera_images:
            raise KeyError(f"missing LIBERO camera image for processor key {key!r}")
        array = np.asarray(camera_images[key])
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(
                f"camera {key!r} must be uint8 [H,W,3], got "
                f"dtype={array.dtype} shape={array.shape}"
            )
        image = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).unsqueeze(0)
        current_transforms = transforms[key] if isinstance(transforms, Mapping) else transforms
        for transform in current_transforms:
            image = transform(image)
        expected_shape = (1, *tuple(int(value) for value in meta["shape"]))
        if tuple(image.shape) != expected_shape:
            raise ValueError(
                f"processor output for camera {camera_index}/{key!r} must be "
                f"{expected_shape}, got {tuple(image.shape)}"
            )
        processed.append(image)

    stacked = torch.stack(processed, dim=0)
    formatted = format_processed_camera_video(
        stacked,
        concat_multi_camera=concat_multi_camera,
        video_size=video_size,
    )
    if int(formatted.shape[1]) != 1:
        raise RuntimeError("closed-loop preprocessing must produce exactly one frame")
    return formatted[:, 0].unsqueeze(0)
