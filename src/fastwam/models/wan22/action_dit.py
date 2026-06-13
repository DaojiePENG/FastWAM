import os
import math
import torch
import torch.nn as nn
from typing import Any, Dict, Optional

from fastwam.utils.logging_config import get_logger

from .helpers.gradient import gradient_checkpoint_forward
from .wan_video_dit import (
    DiTBlock,
    sinusoidal_embedding_1d,
    precompute_freqs_cis,
    flash_attention,
    modulate,
    RMSNorm,
    GateModule,
)

logger = get_logger(__name__)


class HistoryAttention(nn.Module):
    """Full-history self-attention + action→history cross-attention.

    Step 1 — Self-attention on history KVs to learn temporal dependencies
    between frames (Q/K from cached video K, V from cached video V).

    Step 2 — Cross-attention from action tokens to the temporally-refined
    history representation.  Output is added as a standard residual at the
    block level (same pattern as ``cross_attn`` in ActionDiTBlock).

    Args:
        hidden_dim: Dimension of action tokens (e.g. 1024 for ActionDiT).
        attn_head_dim: Per-head dimension.
        num_heads: Number of attention heads.
        max_history_len: Maximum number of history positions.
        history_hidden_dim: Dimension of history video KV (e.g. 3072 for VideoDiT).
            Defaults to ``hidden_dim`` for backward compatibility.
    """

    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int,
                 eps: float = 1e-6, max_history_len: int = 256,
                 history_hidden_dim: int | None = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.history_hidden_dim = history_hidden_dim or hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.attn_hidden_dim = num_heads * attn_head_dim

        # --- QK normalization (shared across both attention ops) ---
        self.norm_q = RMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = RMSNorm(self.attn_hidden_dim, eps=eps)

        # --- Learnable temporal positional embedding ---
        self.temporal_pos_emb = nn.Parameter(
            torch.zeros(1, max_history_len, self.history_hidden_dim)
        )
        nn.init.trunc_normal_(self.temporal_pos_emb, std=0.02)

        # ---- Step 1: Self-attention on history (learns temporal dependencies) ---
        # Q/K from cached video K; V from cached video V
        self.self_q = nn.Linear(self.history_hidden_dim, self.attn_hidden_dim)
        self.self_k = nn.Linear(self.history_hidden_dim, self.attn_hidden_dim)
        self.self_v = nn.Linear(self.history_hidden_dim, self.attn_hidden_dim)
        self.self_o = nn.Linear(self.attn_hidden_dim, self.history_hidden_dim)

        # ---- Step 2: Cross-attention from action → refined history ---
        self.cross_q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.cross_k = nn.Linear(self.history_hidden_dim, self.attn_hidden_dim)
        self.cross_v = nn.Linear(self.history_hidden_dim, self.attn_hidden_dim)
        self.cross_o = nn.Linear(self.attn_hidden_dim, hidden_dim)

        # Zero-initialize output projections so History Attention contributes
        # nothing at initialization (same pattern as GPT-2 residual layers).
        # The optimizer will learn non-zero weights as training progresses.
        nn.init.zeros_(self.self_o.weight)
        nn.init.zeros_(self.self_o.bias)
        nn.init.zeros_(self.cross_o.weight)
        nn.init.zeros_(self.cross_o.bias)

    def forward(self, x: torch.Tensor,
                history_kv: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x: Action tokens, shape ``[B, S_action, hidden_dim]``.
            history_kv: Tuple of (history_k, history_v), each
                ``[B, S_history, history_hidden_dim]``.

        Returns:
            History attention output, shape ``[B, S_action, hidden_dim]``.
        """
        history_k, history_v = history_kv
        seq_len = history_k.shape[1]
        max_len = self.temporal_pos_emb.shape[1]
        if seq_len > max_len:
            raise ValueError(
                f"History length {seq_len} exceeds max_history_len {max_len}"
            )

        # Add temporal positional encoding to both K and V
        pos = self.temporal_pos_emb[:, :seq_len, :].to(dtype=history_k.dtype)
        h_k = history_k + pos
        h_v = history_v + pos

        # Step 1: Self-attention — learn temporal dependencies between history frames
        sq = self.norm_q(self.self_q(h_k))
        sk = self.norm_k(self.self_k(h_k))
        sv = self.self_v(h_v)
        h_attn = flash_attention(q=sq, k=sk, v=sv, num_heads=self.num_heads)
        h_refined = h_k + self.self_o(h_attn)

        # Step 2: Cross-attention — action tokens attend to refined history
        cq = self.norm_q(self.cross_q(x))
        ck = self.norm_k(self.cross_k(h_refined))
        cv = self.cross_v(h_refined)
        out = flash_attention(q=cq, k=ck, v=cv, num_heads=self.num_heads)

        return self.cross_o(out)


class ActionDiTBlock(nn.Module):
    """DiTBlock variant for ActionDiT with optional history attention.

    When ``use_history_cross_attention=True``, each block gains a
    ``history_cross_attn`` module that (1) runs self-attention on the
    full-history video (K, V) cache to learn temporal dependencies, then
    (2) runs cross-attention from action tokens to the refined history.
    The output is added as a standard residual (same pattern as
    ``cross_attn``).
    """

    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int,
                 ffn_dim: int, eps: float = 1e-6,
                 use_history_cross_attention: bool = False,
                 max_history_len: int = 256,
                 history_hidden_dim: int | None = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_head_dim = attn_head_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.use_history_cross_attention = use_history_cross_attention

        from .wan_video_dit import SelfAttention, CrossAttention

        self.self_attn = SelfAttention(hidden_dim, attn_head_dim, num_heads, eps)
        self.cross_attn = CrossAttention(hidden_dim, attn_head_dim, num_heads, eps)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, hidden_dim) / hidden_dim**0.5)
        self.gate = GateModule()

        if use_history_cross_attention:
            self.history_cross_attn = HistoryAttention(
                hidden_dim=hidden_dim,
                attn_head_dim=attn_head_dim,
                num_heads=num_heads,
                eps=eps,
                max_history_len=max_history_len,
                history_hidden_dim=history_hidden_dim,
            )

    def forward(self, x, context, t_mod, freqs, context_mask=None,
                self_attn_mask=None, history_kv=None):
        """
        Args:
            history_kv: Optional tuple ``(history_k, history_v)``, each
                ``[B, S_history, history_hidden_dim]``.
        """
        if context_mask is not None and context_mask.dim() == 3:
            context_mask = context_mask.unsqueeze(1)
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs, self_attn_mask=self_attn_mask))
        x = x + self.cross_attn(self.norm3(x), context, ctx_mask=context_mask)

        # History attention: self-attn on history + cross-attn action→history
        if self.use_history_cross_attention and history_kv is not None:
            x = x + self.history_cross_attn(self.norm3(x), history_kv)

        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


class ActionHead(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int, eps: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.proj = nn.Linear(hidden_dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, hidden_dim) / hidden_dim**0.5)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        shift, scale = (self.modulation.to(dtype=t.dtype, device=t.device) + t.unsqueeze(1)).chunk(2, dim=1)
        shift = shift.squeeze(1)
        scale = scale.squeeze(1)
        return self.proj(self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1))


class ActionDiT(nn.Module):
    ACTION_BACKBONE_SKIP_PREFIXES = ("action_encoder.", "head.")
    ACTION_BACKBONE_META_KEYS = (
        "hidden_dim",
        "ffn_dim",
        "num_layers",
        "num_heads",
        "attn_head_dim",
        "text_dim",
        "freq_dim",
        "eps",
    )

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        ffn_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        use_gradient_checkpointing: bool = False,
        use_history_cross_attention: bool = False,
        max_history_len: int = 256,
        history_hidden_dim: int | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.ffn_dim = ffn_dim
        self.text_dim = text_dim
        self.freq_dim = freq_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.use_history_cross_attention = use_history_cross_attention
        self.max_history_len = max_history_len
        self.history_hidden_dim = history_hidden_dim

        if num_heads <= 0:
            raise ValueError(f"`num_heads` must be > 0, got {num_heads}")
        if attn_head_dim <= 0:
            raise ValueError(f"`attn_head_dim` must be > 0, got {attn_head_dim}")
        if attn_head_dim % 2 != 0:
            raise ValueError(f"`attn_head_dim` must be even for RoPE, got {attn_head_dim}")

        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))

        if use_history_cross_attention:
            self.blocks = nn.ModuleList(
                [
                    ActionDiTBlock(
                        hidden_dim=hidden_dim,
                        attn_head_dim=attn_head_dim,
                        num_heads=num_heads,
                        ffn_dim=ffn_dim,
                        eps=eps,
                        use_history_cross_attention=True,
                        max_history_len=max_history_len,
                        history_hidden_dim=history_hidden_dim,
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            self.blocks = nn.ModuleList(
                [
                    DiTBlock(
                        hidden_dim=hidden_dim,
                        attn_head_dim=attn_head_dim,
                        num_heads=num_heads,
                        ffn_dim=ffn_dim,
                        eps=eps,
                    )
                    for _ in range(num_layers)
                ]
            )
        self.head = nn.Linear(hidden_dim, action_dim)
        self.freqs = precompute_freqs_cis(attn_head_dim, end=1024)

        self.use_gradient_checkpointing = use_gradient_checkpointing

    @classmethod
    def backbone_key_set(cls, keys) -> set[str]:
        return {
            key
            for key in keys
            if not any(key.startswith(prefix) for prefix in cls.ACTION_BACKBONE_SKIP_PREFIXES)
        }

    @classmethod
    def from_pretrained(
        cls,
        action_dit_config: dict[str, Any],
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "ActionDiT":
        if action_dit_config is None:
            raise ValueError("`action_dit_config` is required for ActionDiT.from_pretrained().")
        if skip_dit_load_from_pretrain:
            logger.info(
                "Skipping ActionDiT pretrained load (`skip_dit_load_from_pretrain=True`); "
                "initializing action expert randomly and expecting checkpoint override."
            )
            return cls(**action_dit_config).to(device=device, dtype=torch_dtype)
        if not action_dit_pretrained_path:
            logger.info("No `action_dit_pretrained_path` provided, initializing ActionDiT with random weights.")
            return cls(**action_dit_config).to(device=device, dtype=torch_dtype)
        from pathlib import Path
        p = Path(action_dit_pretrained_path)
        if not p.is_absolute():
            p = Path(__file__).resolve().parents[4] / p
        action_dit_pretrained_path = str(p)
        if not os.path.isfile(action_dit_pretrained_path):
            raise FileNotFoundError(
                f"`action_dit_pretrained_path` does not exist: {action_dit_pretrained_path}"
            )

        action_cfg = dict(action_dit_config)
        action_expert = cls(**action_cfg).to(device=device, dtype=torch_dtype)
        action_state = action_expert.state_dict()
        expected_backbone_keys = cls.backbone_key_set(action_state.keys())

        payload = torch.load(action_dit_pretrained_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(
                f"Invalid action backbone payload type from {action_dit_pretrained_path}: {type(payload)}"
            )
        
        policy = payload.get("policy", {})
        if policy:
            logger.info(f"ActionDiT backbone payload policy: {policy}")

        meta = payload.get("meta")
        expected_meta = {
            "hidden_dim": int(action_cfg["hidden_dim"]),
            "ffn_dim": int(action_cfg["ffn_dim"]),
            "num_layers": int(action_cfg["num_layers"]),
            "num_heads": int(action_cfg["num_heads"]),
            "attn_head_dim": int(action_cfg["attn_head_dim"]),
            "text_dim": int(action_cfg["text_dim"]),
            "freq_dim": int(action_cfg["freq_dim"]),
            "eps": float(action_cfg["eps"]),
        }
        for key in cls.ACTION_BACKBONE_META_KEYS:
            if key not in meta:
                raise ValueError(f"`meta.{key}` missing in {action_dit_pretrained_path}")
            expected_value = expected_meta[key]
            got_value = meta[key]
            if key == "eps":
                if abs(float(got_value) - float(expected_value)) > 1e-12:
                    raise ValueError(
                        f"`meta.{key}` mismatch in {action_dit_pretrained_path}: "
                        f"expected {expected_value}, got {got_value}"
                    )
            elif int(got_value) != int(expected_value):
                raise ValueError(
                    f"`meta.{key}` mismatch in {action_dit_pretrained_path}: "
                    f"expected {expected_value}, got {got_value}"
                )

        backbone_state_dict = payload.get("backbone_state_dict")
        if not isinstance(backbone_state_dict, dict):
            raise ValueError(
                f"`backbone_state_dict` must be a dict in {action_dit_pretrained_path}, "
                f"got {type(backbone_state_dict)}"
            )

        provided_keys = set(backbone_state_dict.keys())

        use_history = bool(action_cfg.get("use_history_cross_attention", False))

        if use_history:
            # History cross-attention params are not in the pretrained backbone;
            # load only matching keys and keep the rest randomly initialised.
            common_keys = expected_backbone_keys & provided_keys
            extra_model_keys = expected_backbone_keys - provided_keys
            unexpected_keys = sorted(provided_keys - expected_backbone_keys)
            if unexpected_keys:
                raise ValueError(
                    "Unexpected keys in pretrained backbone: "
                    f"{unexpected_keys[:10]}{'...' if len(unexpected_keys) > 10 else ''}"
                )

            merged_state = dict(action_state)
            for key in common_keys:
                value = backbone_state_dict[key]
                if not isinstance(value, torch.Tensor):
                    raise ValueError(
                        f"`backbone_state_dict[{key}]` must be torch.Tensor, got {type(value)}"
                    )
                target = merged_state[key]
                if tuple(value.shape) != tuple(target.shape):
                    raise ValueError(
                        f"Shape mismatch for `{key}` in {action_dit_pretrained_path}: "
                        f"expected {tuple(target.shape)}, got {tuple(value.shape)}"
                    )
                merged_state[key] = value.to(device=target.device, dtype=target.dtype)

            action_expert.load_state_dict(merged_state, strict=True)
            logger.info(
                "Loaded ActionDiT backbone from %s (loaded=%d; history_new=%d; random_kept_prefixes=%s).",
                action_dit_pretrained_path,
                len(common_keys),
                len(extra_model_keys),
                list(cls.ACTION_BACKBONE_SKIP_PREFIXES),
            )
        else:
            missing_keys = sorted(expected_backbone_keys - provided_keys)
            unexpected_keys = sorted(provided_keys - expected_backbone_keys)
            if missing_keys or unexpected_keys:
                raise ValueError(
                    "Action backbone key mismatch in preprocessed payload. "
                    f"missing={missing_keys[:10]}{'...' if len(missing_keys) > 10 else ''}, "
                    f"unexpected={unexpected_keys[:10]}{'...' if len(unexpected_keys) > 10 else ''}"
                )

            merged_state = dict(action_state)
            for key in expected_backbone_keys:
                value = backbone_state_dict[key]
                if not isinstance(value, torch.Tensor):
                    raise ValueError(
                        f"`backbone_state_dict[{key}]` must be torch.Tensor in {action_dit_pretrained_path}, "
                        f"got {type(value)}"
                    )
                target = merged_state[key]
                if tuple(value.shape) != tuple(target.shape):
                    raise ValueError(
                        f"Shape mismatch for `{key}` in {action_dit_pretrained_path}: "
                        f"expected {tuple(target.shape)}, got {tuple(value.shape)}"
                    )
                merged_state[key] = value.to(device=target.device, dtype=target.dtype)

            action_expert.load_state_dict(merged_state, strict=True)
            logger.info(
                "Loaded ActionDiT backbone from %s (keys=%d; random_kept_prefixes=%s).",
                action_dit_pretrained_path,
                len(expected_backbone_keys),
                list(cls.ACTION_BACKBONE_SKIP_PREFIXES),
            )
        return action_expert.to(device=device, dtype=torch_dtype)

    def pre_dit(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if action_tokens.ndim != 3:
            raise ValueError(
                f"`action_tokens` must be 3D [B, T, action_dim], got shape {tuple(action_tokens.shape)}"
            )
        if action_tokens.shape[2] != self.action_dim:
            raise ValueError(
                f"`action_tokens` last dim must be {self.action_dim}, got {action_tokens.shape[2]}"
            )
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be 1D [B] or [1], got shape {tuple(timestep.shape)}")
        if context.ndim != 3:
            raise ValueError(
                f"`context` must be 3D [B, L, D], got shape {tuple(context.shape)}"
            )

        batch_size = action_tokens.shape[0]
        if context.shape[0] != batch_size:
            raise ValueError(
                f"Batch mismatch between action tokens and text context: {batch_size} vs {context.shape[0]}"
            )
        if timestep.shape[0] not in (1, batch_size):
            raise ValueError(
                f"`timestep` length must be 1 or batch_size({batch_size}), got {timestep.shape[0]}"
            )
        if timestep.shape[0] == 1 and batch_size > 1:
            if self.training:
                raise ValueError("During training, action timestep length must match batch_size.")
            timestep = timestep.expand(batch_size)

        if context_mask is None:
            context_mask = torch.ones(
                (batch_size, context.shape[1]), dtype=torch.bool, device=context.device
            )
        else:
            if context_mask.ndim != 2:
                raise ValueError(f"`context_mask` must be 2D [B, L], got shape {tuple(context_mask.shape)}")
            if context_mask.shape[0] != batch_size or context_mask.shape[1] != context.shape[1]:
                raise ValueError(
                    f"`context_mask` shape must match `context` shape [B, L], got {tuple(context_mask.shape)} vs {tuple(context.shape)}"
                )

        seq_len = action_tokens.shape[1]
        if seq_len > self.freqs.shape[0]:
            raise ValueError(
                f"Action token length {seq_len} exceeds RoPE cache {self.freqs.shape[0]}."
            )

        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))

        tokens = self.action_encoder(action_tokens)
        context_emb = self.text_embedding(context)
        context_attn_mask = context_mask.unsqueeze(1).expand(-1, seq_len, -1)
        freqs = self.freqs[:seq_len].view(seq_len, 1, -1).to(tokens.device)

        return {
            "tokens": tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": context_emb,
            "context_mask": context_attn_mask,
            "meta": {
                "batch_size": batch_size,
                "seq_len": seq_len,
            },
        }

    def post_dit(self, tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        return self.head(tokens)

    def forward(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        history_kv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pre_state = self.pre_dit(
            action_tokens=action_tokens,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
        )
        x = pre_state["tokens"]
        context = pre_state["context"]
        t_mod = pre_state["t_mod"]
        freqs = pre_state["freqs"]
        context_mask = pre_state["context_mask"]

        for block in self.blocks:
            if self.use_history_cross_attention:
                if self.use_gradient_checkpointing:
                    x = gradient_checkpoint_forward(
                        block,
                        self.use_gradient_checkpointing,
                        x,
                        context,
                        t_mod,
                        freqs,
                        context_mask=context_mask,
                        history_kv=history_kv,
                    )
                else:
                    x = block(x, context, t_mod, freqs,
                              context_mask=context_mask, history_kv=history_kv)
            else:
                if self.use_gradient_checkpointing:
                    x = gradient_checkpoint_forward(
                        block,
                        self.use_gradient_checkpointing,
                        x,
                        context,
                        t_mod,
                        freqs,
                        context_mask=context_mask,
                    )
                else:
                    x = block(x, context, t_mod, freqs, context_mask=context_mask)

        return self.post_dit(x, pre_state)
