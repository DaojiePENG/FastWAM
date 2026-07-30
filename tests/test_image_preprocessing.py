from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from torchvision.transforms import Resize

from fastwam.datasets.lerobot.transforms.image import ToTensor
from leapbot_va.data import LeapRobotVideoDataset
from leapbot_va.image_preprocessing import preprocess_uint8_libero_cameras


def _processor() -> SimpleNamespace:
    transforms = [ToTensor(), Resize(size=[224, 224])]
    return SimpleNamespace(
        shape_meta={
            "images": [
                {"key": "image", "shape": [3, 224, 224]},
                {"key": "wrist_image", "shape": [3, 224, 224]},
            ]
        },
        num_output_cameras=2,
        train_transforms=transforms,
        val_transforms=transforms,
        is_train=False,
    )


def test_closed_loop_image_path_is_elementwise_identical_to_training_path():
    generator = np.random.default_rng(42)
    cameras = {
        "image": generator.integers(0, 256, (512, 512, 3), dtype=np.uint8),
        "wrist_image": generator.integers(0, 256, (512, 512, 3), dtype=np.uint8),
    }
    processor = _processor()

    evaluation = preprocess_uint8_libero_cameras(
        cameras,
        processor=processor,
        concat_multi_camera="horizontal",
        video_size=[224, 448],
    )

    processed = []
    for key in ("image", "wrist_image"):
        image = torch.from_numpy(cameras[key]).permute(2, 0, 1).unsqueeze(0)
        for transform in processor.train_transforms:
            image = transform(image)
        processed.append(image)
    dataset = object.__new__(LeapRobotVideoDataset)
    dataset.concat_multi_camera = "horizontal"
    dataset.video_size = [224, 448]
    training = dataset._format_camera_video(torch.stack(processed), [0])

    assert evaluation.shape == (1, 3, 224, 448)
    assert torch.equal(evaluation[0], training[:, 0])


def test_closed_loop_preprocessing_rejects_wrong_raw_camera_contract():
    processor = _processor()
    cameras = {
        "image": np.zeros((512, 512, 3), dtype=np.float32),
        "wrist_image": np.zeros((512, 512, 3), dtype=np.uint8),
    }
    try:
        preprocess_uint8_libero_cameras(
            cameras,
            processor=processor,
            concat_multi_camera="horizontal",
            video_size=[224, 448],
        )
    except ValueError as error:
        assert "must be uint8" in str(error)
    else:
        raise AssertionError("non-uint8 LIBERO camera input unexpectedly passed")
