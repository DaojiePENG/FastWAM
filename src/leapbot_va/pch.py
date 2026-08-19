"""Fixed-padding Packed Causal History layout and attention masks."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import torch

from leapbot_va.memory import VALID_CAUSAL_MODES


def encode_pch_slot_validity_metadata(
    history_valid_blocks: torch.Tensor,
    anchor_valid: torch.Tensor,
) -> bytes:
    """Encode slot validity on CPU before Accelerate moves a batch to CUDA."""
    if history_valid_blocks.device.type != "cpu" or anchor_valid.device.type != "cpu":
        raise ValueError("PCH slot-validity metadata must be encoded on CPU")
    if history_valid_blocks.dtype != torch.bool or history_valid_blocks.ndim != 2:
        raise ValueError("history_valid_blocks must be CPU bool [B,W]")
    batch = int(history_valid_blocks.shape[0])
    if anchor_valid.dtype != torch.bool or tuple(anchor_valid.shape) != (batch,):
        raise ValueError("anchor_valid must be CPU bool [B]")
    return torch.cat([anchor_valid[:, None], history_valid_blocks], dim=1).contiguous().numpy().tobytes()


def build_pch_validity_signature(
    slot_validity_metadata: bytes | list[bytes] | tuple[bytes, ...],
    *,
    batch_size: int,
    window_blocks: int,
    video_tokens_per_slot: int,
    action_tokens_per_slot: int,
) -> bytes:
    """Expand CPU slot metadata to the former key-validity byte layout."""
    row_bytes = window_blocks + 1
    if isinstance(slot_validity_metadata, bytes):
        if len(slot_validity_metadata) != batch_size * row_bytes:
            raise ValueError("PCH slot-validity metadata length mismatch")
        rows = [
            slot_validity_metadata[index * row_bytes : (index + 1) * row_bytes]
            for index in range(batch_size)
        ]
    else:
        rows = list(slot_validity_metadata)
    if len(rows) != batch_size or any(not isinstance(row, bytes) for row in rows):
        raise ValueError("PCH slot-validity metadata must contain B byte strings")
    if any(len(row) != row_bytes for row in rows):
        raise ValueError("PCH slot-validity metadata length mismatch")
    if video_tokens_per_slot <= 0 or action_tokens_per_slot <= 0:
        raise ValueError("PCH token geometry must be positive")
    expanded = bytearray()
    for row in rows:
        if any(value not in (0, 1) for value in row):
            raise ValueError("PCH slot-validity metadata must contain zero/one bytes")
        expanded.extend([row[0]] * video_tokens_per_slot)
        for value in row[1:]:
            expanded.extend([value] * video_tokens_per_slot)
        for value in row[1:]:
            expanded.extend([value] * action_tokens_per_slot)
    return bytes(expanded)


@dataclass
class PCHLayout:
    """Token metadata for one fixed-padding packed-history batch.

    Video slots are ``[anchor, history_0, ..., history_W-1]`` and action
    slots are ``[history_0, ..., history_W-1]``. History is right aligned.
    """

    history_valid_blocks: torch.Tensor
    anchor_valid: torch.Tensor
    history_block_positions: torch.Tensor
    anchor_block_positions: torch.Tensor
    video_tokens_per_slot: int
    action_tokens_per_slot: int
    validity_signature: bytes | None = None

    def __post_init__(self) -> None:
        valid = self.history_valid_blocks
        if valid.ndim != 2 or valid.dtype != torch.bool:
            raise ValueError("history_valid_blocks must be bool [B,W]")
        batch, window = valid.shape
        if self.anchor_valid.dtype != torch.bool or self.anchor_valid.shape != (batch,):
            raise ValueError("anchor_valid must be bool [B]")
        if self.history_block_positions.shape != (batch, window):
            raise ValueError("history_block_positions must match [B,W]")
        if self.anchor_block_positions.shape != (batch,):
            raise ValueError("anchor_block_positions must be [B]")
        if self.video_tokens_per_slot <= 0 or self.action_tokens_per_slot <= 0:
            raise ValueError("PCH token geometry must be positive")
        if self.validity_signature is not None:
            expected = batch * ((window + 1) * self.video_tokens_per_slot + window * self.action_tokens_per_slot)
            if not isinstance(self.validity_signature, bytes) or len(self.validity_signature) != expected:
                raise ValueError("PCH validity signature has invalid type or length")
        devices = {
            valid.device,
            self.anchor_valid.device,
            self.history_block_positions.device,
            self.anchor_block_positions.device,
        }
        if len(devices) != 1:
            raise ValueError("all PCHLayout tensors must share a device")

    @property
    def batch_size(self) -> int:
        return int(self.history_valid_blocks.shape[0])

    @property
    def window_blocks(self) -> int:
        return int(self.history_valid_blocks.shape[1])

    @property
    def device(self) -> torch.device:
        return self.history_valid_blocks.device

    @property
    def video_slot_valid(self) -> torch.Tensor:
        return torch.cat(
            [self.anchor_valid[:, None], self.history_valid_blocks], dim=1
        )

    @property
    def action_slot_valid(self) -> torch.Tensor:
        return self.history_valid_blocks

    @property
    def video_slot_indices(self) -> torch.Tensor:
        return torch.arange(
            self.window_blocks + 1, device=self.device
        ).repeat_interleave(self.video_tokens_per_slot)

    @property
    def action_slot_indices(self) -> torch.Tensor:
        # Action slot i belongs to video slot i+1; the anchor has no action.
        return (
            torch.arange(self.window_blocks, device=self.device) + 1
        ).repeat_interleave(self.action_tokens_per_slot)

    @property
    def video_key_valid_mask(self) -> torch.Tensor:
        return self.video_slot_valid[:, self.video_slot_indices]

    @property
    def action_key_valid_mask(self) -> torch.Tensor:
        return self.action_slot_valid[:, self.action_slot_indices - 1]

    @property
    def key_valid_mask(self) -> torch.Tensor:
        return torch.cat(
            [self.video_key_valid_mask, self.action_key_valid_mask], dim=1
        )

    @property
    def packed_tokens(self) -> int:
        return int(self.key_valid_mask.shape[1])

    @property
    def valid_blocks(self) -> torch.Tensor:
        return self.history_valid_blocks.sum(dim=1) + self.anchor_valid.long()

    def structure_signature(self, causal_mode: str) -> tuple[Any, ...]:
        validity = self.validity_signature
        if validity is None:
            if self.device.type != "cpu":
                raise RuntimeError("CUDA Flex PCH requires a CPU-generated validity signature")
            validity = build_pch_validity_signature(
                encode_pch_slot_validity_metadata(self.history_valid_blocks, self.anchor_valid),
                batch_size=self.batch_size,
                window_blocks=self.window_blocks,
                video_tokens_per_slot=self.video_tokens_per_slot,
                action_tokens_per_slot=self.action_tokens_per_slot,
            )
        return (
            self.device.type,
            self.device.index,
            self.batch_size,
            self.window_blocks,
            self.video_tokens_per_slot,
            self.action_tokens_per_slot,
            causal_mode,
            validity,
        )


def build_pch_dense_attention_mask(
    layout: PCHLayout,
    causal_mode: str,
    *,
    prefix_video_tokens: int = 0,
) -> torch.Tensor:
    """Build the exact block-causal PCH mask as bool ``[B,S,S]``."""

    if causal_mode not in VALID_CAUSAL_MODES:
        raise ValueError(f"unsupported causal mode: {causal_mode}")

    video_slots = layout.video_slot_indices
    action_slots = layout.action_slot_indices
    video_len = int(video_slots.numel())
    action_len = int(action_slots.numel())
    slot = torch.cat([video_slots, action_slots])
    is_action = torch.cat(
        [
            torch.zeros(video_len, dtype=torch.bool, device=layout.device),
            torch.ones(action_len, dtype=torch.bool, device=layout.device),
        ]
    )

    q_slot = slot[:, None]
    k_slot = slot[None, :]
    q_action = is_action[:, None]
    k_action = is_action[None, :]
    earlier = k_slot < q_slot
    same = k_slot == q_slot

    # Every action segment reads its own action tokens, same-block video, and
    # all earlier valid video/action segments.
    action_allowed = earlier | same

    same_video = same & ~k_action
    if causal_mode == "interleaved":
        earlier_video = earlier
    elif causal_mode == "vision_causal":
        earlier_video = earlier & ~k_action
    else:  # action_aggregator
        earlier_video = torch.zeros_like(earlier)
    video_allowed = same_video | earlier_video
    allowed = torch.where(q_action, action_allowed, video_allowed)

    valid = layout.key_valid_mask
    mask = allowed.unsqueeze(0) & valid.unsqueeze(2) & valid.unsqueeze(1)
    invalid_diagonal = (~valid).unsqueeze(2) & torch.eye(
        layout.packed_tokens, dtype=torch.bool, device=layout.device
    ).unsqueeze(0)
    mask = mask | invalid_diagonal
    prefix_video_tokens = int(prefix_video_tokens)
    if prefix_video_tokens < 0:
        raise ValueError("prefix_video_tokens must be non-negative")
    if prefix_video_tokens:
        video_reads_prefix = causal_mode != "action_aggregator"
        prefix_allowed = is_action | video_reads_prefix
        prefix_mask = (
            valid[:, :, None] & prefix_allowed[None, :, None]
        ).expand(-1, -1, prefix_video_tokens)
        mask = torch.cat([prefix_mask, mask], dim=2)
    return mask


def build_pch_context_masks(
    base_context_mask: torch.Tensor,
    layout: PCHLayout,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Language plus exactly the proprio token owned by each valid slot."""

    if base_context_mask.dtype != torch.bool or base_context_mask.ndim != 2:
        raise ValueError("base_context_mask must be bool [B,L]")
    if int(base_context_mask.shape[0]) != layout.batch_size:
        raise ValueError("base_context_mask batch does not match PCH layout")
    text_len = int(base_context_mask.shape[1])
    slot_valid = layout.video_slot_valid
    slot_count = layout.window_blocks + 1

    def build(slot_ids: torch.Tensor, query_valid: torch.Tensor) -> torch.Tensor:
        result = torch.zeros(
            (layout.batch_size, int(slot_ids.numel()), text_len + slot_count),
            dtype=torch.bool,
            device=layout.device,
        )
        result[:, :, :text_len] = (
            base_context_mask[:, None, :] & query_valid[:, :, None]
        )
        proprio_columns = text_len + slot_ids
        result.scatter_(
            2,
            proprio_columns.view(1, -1, 1).expand(layout.batch_size, -1, -1),
            query_valid[:, :, None],
        )
        return result

    return (
        build(layout.video_slot_indices, layout.video_key_valid_mask),
        build(layout.action_slot_indices, layout.action_key_valid_mask),
    )


class _FlexBlockMaskCache:
    def __init__(self, capacity: int = 64):
        self.capacity = int(capacity)
        self._cache: OrderedDict[tuple[Any, ...], Any] = OrderedDict()

    @staticmethod
    def _key(
        layout: PCHLayout, causal_mode: str, prefix_video_tokens: int
    ) -> tuple[Any, ...]:
        return (*layout.structure_signature(causal_mode), int(prefix_video_tokens))

    def get(self, layout: PCHLayout, causal_mode: str, prefix_video_tokens: int = 0) -> Any | None:
        key = self._key(layout, causal_mode, prefix_video_tokens)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
        return cached

    def get_or_create(
        self,
        layout: PCHLayout,
        causal_mode: str,
        dense_mask: torch.Tensor,
        prefix_video_tokens: int = 0,
    ) -> Any:
        key = self._key(layout, causal_mode, prefix_video_tokens)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        try:
            from torch.nn.attention.flex_attention import create_block_mask
        except Exception as exc:  # pragma: no cover - depends on torch build
            raise RuntimeError(
                "PCH FlexAttention is unavailable in this PyTorch build; "
                "set model.packed_history_attention_backend=dense for debugging"
            ) from exc

        def mask_mod(b, _h, q_idx, kv_idx):
            return dense_mask[b, q_idx, kv_idx]

        try:
            block_mask = create_block_mask(
                mask_mod,
                B=layout.batch_size,
                H=None,
                Q_LEN=layout.packed_tokens,
                KV_LEN=int(prefix_video_tokens) + layout.packed_tokens,
                device=str(layout.device),
            )
        except Exception as exc:
            raise RuntimeError(
                "PCH FlexAttention BlockMask construction failed; "
                "set model.packed_history_attention_backend=dense to run the "
                "explicit dense reference backend"
            ) from exc
        self._cache[key] = block_mask
        self._cache.move_to_end(key)
        while len(self._cache) > self.capacity:
            self._cache.popitem(last=False)
        return block_mask


_FLEX_MASK_CACHE = _FlexBlockMaskCache(capacity=64)


def build_pch_attention_mask(
    layout: PCHLayout,
    causal_mode: str,
    backend: str,
    *,
    prefix_video_tokens: int = 0,
) -> torch.Tensor | Any:
    """Create a Dense SDPA mask or a semantically identical Flex BlockMask."""

    backend = str(backend).lower()
    if backend not in {"dense", "flex"}:
        raise ValueError("packed_history_attention_backend must be flex or dense")
    if backend == "flex":
        cached = _FLEX_MASK_CACHE.get(
            layout, causal_mode, prefix_video_tokens
        )
        if cached is not None:
            return cached
    dense = build_pch_dense_attention_mask(
        layout,
        causal_mode,
        prefix_video_tokens=prefix_video_tokens,
    )
    if backend == "dense":
        return dense
    return _FLEX_MASK_CACHE.get_or_create(
        layout,
        causal_mode,
        dense,
        prefix_video_tokens,
    )


def combine_packed_history_kv(
    cache: Any,
    *,
    causal_mode: str,
    query_kind: str,
) -> tuple[list[dict[str, torch.Tensor]] | None, torch.Tensor | None]:
    """Select the PCH prefix required by a downstream video/action query."""

    if causal_mode not in VALID_CAUSAL_MODES:
        raise ValueError(f"unsupported causal mode: {causal_mode}")
    if query_kind == "action":
        include_video, include_action = True, True
    elif query_kind == "video":
        include_video = causal_mode != "action_aggregator"
        include_action = causal_mode == "interleaved"
    else:
        raise ValueError("query_kind must be video or action")
    if not include_video and not include_action:
        return None, None

    masks: list[torch.Tensor] = []
    if include_video:
        masks.append(cache.video_valid_mask)
    if include_action:
        masks.append(cache.action_valid_mask)
    valid = torch.cat(masks, dim=1)
    layers: list[dict[str, torch.Tensor]] = []
    for video_layer, action_layer in zip(cache.video_kv, cache.action_kv):
        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        if include_video:
            keys.append(video_layer["k"])
            values.append(video_layer["v"])
        if include_action:
            keys.append(action_layer["k"])
            values.append(action_layer["v"])
        layers.append({"k": torch.cat(keys, dim=1), "v": torch.cat(values, dim=1)})
    return layers, valid
