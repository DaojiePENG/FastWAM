#!/usr/bin/env python3
"""Re-encode every formal prompt and persist exact text-cache provenance."""

from __future__ import annotations

import fcntl
import logging
import os
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer
from fastwam.datasets.lerobot.text_cache import wan_text_cache_filename
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.logging_config import get_logger, setup_logging
from leapbot_va.conditioning_assets import (
    build_text_cache_provenance,
    write_text_cache_provenance,
)
from scripts.precompute_text_embeds import (
    DEFAULT_BATCH_SIZE,
    _acquire_cache_locks,
    _collect_dataset_settings,
    _read_unique_prompts,
    _resolve_context_len,
)


register_default_resolvers()
logger = get_logger(__name__)


def assert_cached_payload_exact(
    payload,
    *,
    expected_context: torch.Tensor,
    expected_mask: torch.Tensor,
    cache_path: Path,
) -> None:
    if not isinstance(payload, dict) or set(payload) != {"context", "mask"}:
        raise ValueError(f"unexpected text-cache payload fields: {cache_path}")
    context = payload["context"]
    mask = payload["mask"]
    if not isinstance(context, torch.Tensor):
        raise ValueError(f"text-cache context is not a tensor: {cache_path}")
    if not isinstance(mask, torch.Tensor):
        raise ValueError(f"text-cache mask is not a tensor: {cache_path}")
    if context.dtype is not torch.bfloat16:
        raise ValueError(
            f"text-cache context dtype mismatch at {cache_path}: {context.dtype}"
        )
    if mask.dtype is not torch.bool:
        raise ValueError(f"text-cache mask dtype mismatch at {cache_path}: {mask.dtype}")
    if context.shape != expected_context.shape:
        raise ValueError(
            f"text-cache context shape mismatch at {cache_path}: "
            f"actual={tuple(context.shape)} expected={tuple(expected_context.shape)}"
        )
    if mask.shape != expected_mask.shape:
        raise ValueError(
            f"text-cache mask shape mismatch at {cache_path}: "
            f"actual={tuple(mask.shape)} expected={tuple(expected_mask.shape)}"
        )
    if not torch.equal(context, expected_context):
        difference = (
            context.float() - expected_context.float()
        ).abs().max()
        raise ValueError(
            f"text-cache context mismatch at {cache_path}; "
            f"max_abs={float(difference)}"
        )
    if not torch.equal(mask, expected_mask):
        raise ValueError(f"text-cache mask mismatch at {cache_path}")


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging(log_level=logging.INFO)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("text-cache provenance verification must run as one process")

    dataset_dirs, cache_dirs, context_lens = _collect_dataset_settings(cfg.data)
    dataset_dirs = [
        str(
            (PROJECT_ROOT / Path(path)).resolve()
            if not Path(path).expanduser().is_absolute()
            else Path(path).expanduser().resolve()
        )
        for path in dataset_dirs
    ]
    cache_dirs = [
        (
            (PROJECT_ROOT / path).resolve()
            if not path.expanduser().is_absolute()
            else path.expanduser().resolve()
        )
        for path in cache_dirs
    ]
    context_len = _resolve_context_len(context_lens)
    prompts = _read_unique_prompts(dataset_dirs)
    if not prompts or not cache_dirs:
        raise ValueError("text-cache verification requires prompts and cache directories")

    model_cfg = cfg.model
    model_id = str(model_cfg.get("model_id", "Wan-AI/Wan2.2-TI2V-5B"))
    tokenizer_model_id = str(
        model_cfg.get("tokenizer_model_id", "Wan-AI/Wan2.1-T2V-1.3B")
    )
    redirect_common_files = bool(model_cfg.get("redirect_common_files", True))
    device = str(cfg.get("text_cache_verification_device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by text_cache_verification_device")

    locks = _acquire_cache_locks(cache_dirs)
    try:
        _, text_config, _, tokenizer_config = _resolve_configs(
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            redirect_common_files=redirect_common_files,
        )
        text_config.download_if_necessary()
        tokenizer_config.download_if_necessary()
        text_encoder = _load_registered_model(
            text_config.path,
            "wan_video_text_encoder",
            torch_dtype=torch.bfloat16,
            device=device,
        ).eval()
        tokenizer = HuggingfaceTokenizer(
            name=tokenizer_config.path,
            seq_len=context_len,
            clean="whitespace",
        )

        cache_file_names = [
            wan_text_cache_filename(
                prompt, context_len=context_len, model_id=model_id
            )
            for prompt in prompts
        ]
        verified = {str(cache_dir): 0 for cache_dir in cache_dirs}
        with torch.no_grad():
            for start in tqdm(
                range(0, len(prompts), DEFAULT_BATCH_SIZE),
                desc="Verifying text cache",
                unit="batch",
            ):
                batch_prompts = prompts[start : start + DEFAULT_BATCH_SIZE]
                ids, mask = tokenizer(
                    batch_prompts, return_mask=True, add_special_tokens=True
                )
                ids = ids.to(device)
                mask = mask.to(device=device, dtype=torch.bool)
                context = text_encoder(ids, mask)
                for offset, prompt in enumerate(batch_prompts):
                    filename = wan_text_cache_filename(
                        prompt, context_len=context_len, model_id=model_id
                    )
                    expected_context = (
                        context[offset]
                        .detach()
                        .to(device="cpu", dtype=torch.bfloat16)
                        .contiguous()
                    )
                    expected_mask = (
                        mask[offset]
                        .detach()
                        .to(device="cpu", dtype=torch.bool)
                        .contiguous()
                    )
                    for cache_dir in cache_dirs:
                        cache_path = cache_dir / filename
                        payload = torch.load(
                            cache_path, map_location="cpu", weights_only=True
                        )
                        assert_cached_payload_exact(
                            payload,
                            expected_context=expected_context,
                            expected_mask=expected_mask,
                            cache_path=cache_path,
                        )
                        verified[str(cache_dir)] += 1

        for cache_dir in cache_dirs:
            count = verified[str(cache_dir)]
            provenance = build_text_cache_provenance(
                cache_dir=cache_dir,
                cache_file_names=cache_file_names,
                model_id=model_id,
                tokenizer_model_id=tokenizer_model_id,
                redirect_common_files=redirect_common_files,
                context_len=context_len,
                text_encoder_path=text_config.path,
                tokenizer_path=tokenizer_config.path,
                verified_file_count=count,
                verification_method="online_source_forward_cache_tensor_exact",
            )
            output = write_text_cache_provenance(cache_dir, provenance)
            logger.info(
                "Verified %d/%d cache files; provenance=%s sha256=%s",
                count,
                len(cache_file_names),
                output,
                provenance["provenance_sha256"],
            )
    finally:
        for stream in locks:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            stream.close()


if __name__ == "__main__":
    main()
