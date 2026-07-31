"""Shared text prompt and cache-key contract for Wan-based datasets."""

from __future__ import annotations

import hashlib
import re


DEFAULT_PROMPT = (
    "A video recorded from a robot's point of view executing the following "
    "instruction: {task}"
)


def wan_text_encoder_id(model_id: str) -> str:
    base = str(model_id).split("/")[-1]
    encoder_id = re.sub(r"[^a-z0-9]+", "", base.lower())
    return encoder_id or "textenc"


def wan_text_cache_filename(
    prompt: str, *, context_len: int, model_id: str
) -> str:
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return (
        f"{prompt_sha256}.t5_len{int(context_len)}."
        f"{wan_text_encoder_id(model_id)}.pt"
    )
