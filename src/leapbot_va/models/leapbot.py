"""LeapBot-VA model: FastWAM with real-observation causal KV memory."""

from __future__ import annotations

import copy
import hashlib
import os
import time
from collections.abc import Mapping
from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn

from fastwam.models.wan22.fastwam import FastWAM
from leapbot_va.lora import (
    VideoLoRAConfig,
    inject_video_self_attention_lora,
    lora_parameters,
    merge_video_self_attention_lora,
)

from leapbot_va.memory import (
    KVSegment,
    LeapMemoryConfig,
    LeapMemoryState,
    VALID_CAUSAL_MODES,
)
from leapbot_va.positions import (
    TEMPORAL_POSITION_SCHEME,
    HierarchicalTemporalPositionEmbedding,
)


class LeapBotVA(FastWAM):
    """FastWAM with real-data memory and transient world-model planning.

    A call to :meth:`infer_action` commits one real observation segment
    transactionally, imagines the current block's future video, and lets
    ActionDiT attend to that video's layer-wise K/V.  Imagined-video K/V is
    local to the call and is never written to :class:`LeapMemoryState`; the
    caller later commits only commands actually sent to the robot.
    """

    def __init__(self, *args, exit_depths: Sequence[int] = (8, 16, 24, 30), **kwargs):
        super().__init__(*args, **kwargs)
        num_layers = int(self.mot.num_layers)
        requested = tuple(sorted({int(depth) for depth in exit_depths}))
        if not requested or requested[-1] != num_layers:
            raise ValueError(
                f"exit_depths must include the final model depth {num_layers}, got {requested}"
            )
        if requested[0] <= 0 or requested[-1] > num_layers:
            raise ValueError(f"invalid exit depths: {requested}")
        self.exit_depths = requested

        shallow_depths = [depth for depth in requested if depth != num_layers]
        self.action_exit_heads = nn.ModuleDict(
            {
                str(depth): copy.deepcopy(self.action_expert.head)
                for depth in shallow_depths
            }
        )
        self.video_exit_heads = nn.ModuleDict(
            {
                str(depth): copy.deepcopy(self.video_expert.head)
                for depth in shallow_depths
            }
        )
        self.action_exit_heads.to(device=self.device, dtype=self.torch_dtype)
        self.video_exit_heads.to(device=self.device, dtype=self.torch_dtype)
        # Preserve FastWAM's pretrained, block-local RoPE coordinates.  The
        # episode-global block/control progress is injected separately through
        # exact-zero projections, so the extension leaves a freshly loaded
        # release checkpoint's token inputs unchanged at initialization.
        self.temporal_positions = HierarchicalTemporalPositionEmbedding(
            video_dim=int(self.video_expert.hidden_dim),
            action_dim=int(self.action_expert.hidden_dim),
            device=self.device,
        )
        self.causal_mode = "interleaved"
        self.training_exit_depths = (num_layers,)
        # Architecture support alone does not mean that a shallow head has
        # received training. This is replaced from checkpoint metadata on load.
        self.trained_exit_depths = (num_layers,)
        self.training_strategy = "full_dit"
        self.history_training_mode = "incremental_full_bptt"
        self.video_lora_config = VideoLoRAConfig()
        self.video_lora_merged = False
        self.training_replan_steps: int | None = None
        self.training_action_horizon: int | None = None
        # LingBot-VA-style inverse-dynamics conditioning.  The persistent
        # episode memory remains real-data-only; this contract governs the
        # transient same-block video condition seen by ActionDiT.
        self.future_video_conditioning = "lingbot_teacher_forced_v1"
        self.future_video_condition_noise_probability = 0.5
        self.future_video_condition_min_u = 0.5
        self.future_video_condition_max_u = 1.0
        self.training_num_video_frames: int | None = None
        # -1 means fully denoise on the schedule supplied to infer_action.
        # Positive values are an explicit speed/quality inference ablation.
        self.future_video_denoise_steps = -1
        self.history_vae_batch_chunk_size = 1
        # The outer segment checkpoint rematerializes full-prefix concatenations
        # during backward.  Keep its default aligned with the existing MoT
        # checkpoint switch so production training enables both consistently.
        self.history_segment_activation_checkpointing = bool(
            self.mot.mot_checkpoint_mixed_attn
        )

    def configure_finetuning(
        self,
        *,
        training_strategy: str = "full_dit",
        video_lora_config: VideoLoRAConfig | None = None,
    ) -> None:
        valid_strategies = {"full_dit", "video_lora_action_full"}
        if training_strategy not in valid_strategies:
            raise ValueError(
                f"training_strategy must be one of {sorted(valid_strategies)}, "
                f"got {training_strategy}"
            )
        config = video_lora_config or VideoLoRAConfig()
        if training_strategy == "video_lora_action_full" and not config.enabled:
            raise ValueError("hybrid fine-tuning requires video LoRA to be enabled")
        if config.enabled:
            inject_video_self_attention_lora(self.video_expert, config)
        self.training_strategy = training_strategy
        self.video_lora_config = config
        self.video_lora_merged = False

    def merge_video_lora_(self) -> int:
        if not self.video_lora_config.enabled:
            return 0
        merged = merge_video_self_attention_lora(self.video_expert)
        if not merged:
            raise RuntimeError("video LoRA was enabled but no adapters were available to merge")
        self.video_lora_merged = True
        return len(merged)

    def configure_trainable_parameters(self) -> None:
        """Select full-DiT or video-LoRA/action-full trainable parameters."""
        self.mot.train()
        self.temporal_positions.train()
        self.temporal_positions.requires_grad_(True)
        if self.training_strategy == "full_dit":
            self.mot.requires_grad_(True)
            return
        self.action_expert.requires_grad_(True)
        video_lora_params = lora_parameters(self.video_expert)
        if not video_lora_params:
            raise RuntimeError("hybrid fine-tuning selected but video LoRA is absent")
        for parameter in video_lora_params:
            parameter.requires_grad_(True)

    def optimizer_parameter_groups(
        self,
        *,
        learning_rate: float,
        weight_decay: float,
    ) -> list[dict]:
        trainable = [parameter for parameter in self.parameters() if parameter.requires_grad]
        if self.training_strategy == "full_dit":
            return [
                {
                    "params": trainable,
                    "lr": learning_rate,
                    "weight_decay": weight_decay,
                    "group_name": "full_dit",
                }
            ]
        video_lora = lora_parameters(self.video_expert)
        video_lora_ids = {id(parameter) for parameter in video_lora}
        action_and_aux = [
            parameter for parameter in trainable if id(parameter) not in video_lora_ids
        ]
        return [
            {
                "params": action_and_aux,
                "lr": learning_rate,
                "weight_decay": weight_decay,
                "group_name": "action_and_aux",
            },
            {
                "params": video_lora,
                "lr": learning_rate * self.video_lora_config.learning_rate_multiplier,
                "weight_decay": 0.0,
                "group_name": "video_lora",
            },
        ]

    def configure_causal_training(
        self,
        *,
        causal_mode: str = "interleaved",
        training_exit_depths: Sequence[int] = (30,),
        history_training_mode: str = "incremental_full_bptt",
        replan_steps: int | None = None,
        action_horizon: int | None = None,
        num_video_frames: int | None = None,
        future_video_condition_noise_probability: float = 0.5,
        future_video_condition_min_u: float = 0.5,
        future_video_condition_max_u: float = 1.0,
        future_video_denoise_steps: int = -1,
    ) -> None:
        if causal_mode not in VALID_CAUSAL_MODES:
            raise ValueError(f"unsupported causal mode: {causal_mode}")
        depths = tuple(sorted({int(depth) for depth in training_exit_depths}))
        if not depths or depths[-1] != self.mot.num_layers:
            raise ValueError("training_exit_depths must include the final model depth")
        if any(depth not in self.exit_depths for depth in depths):
            raise ValueError(
                f"training exits {depths} are not available in model exits {self.exit_depths}"
            )
        self.causal_mode = causal_mode
        self.training_exit_depths = depths
        if history_training_mode != "incremental_full_bptt":
            raise ValueError(
                "LeapBot causal training requires incremental_full_bptt; packed and "
                "detached-prefix programs do not match runtime BF16 execution, got "
                f"{history_training_mode}"
            )
        self.history_training_mode = history_training_mode
        noise_probability = float(future_video_condition_noise_probability)
        min_u = float(future_video_condition_min_u)
        max_u = float(future_video_condition_max_u)
        if not 0.0 <= noise_probability <= 1.0:
            raise ValueError(
                "future-video condition noise probability must be in [0,1]"
            )
        if not 0.0 <= min_u <= max_u <= 1.0:
            raise ValueError(
                "future-video condition noise bounds must satisfy "
                "0 <= min_u <= max_u <= 1"
            )
        denoise_steps = int(future_video_denoise_steps)
        if denoise_steps == 0 or denoise_steps < -1:
            raise ValueError("future_video_denoise_steps must be -1 or positive")
        self.future_video_condition_noise_probability = noise_probability
        self.future_video_condition_min_u = min_u
        self.future_video_condition_max_u = max_u
        self.future_video_denoise_steps = denoise_steps
        if (replan_steps is None) != (action_horizon is None):
            raise ValueError(
                "replan_steps and action_horizon must be configured together"
            )
        if replan_steps is not None:
            resolved_replan_steps = int(replan_steps)
            resolved_action_horizon = int(action_horizon)
            if resolved_replan_steps <= 0:
                raise ValueError("replan_steps must be positive")
            if resolved_action_horizon < resolved_replan_steps:
                raise ValueError(
                    "action_horizon must be greater than or equal to replan_steps"
                )
            self.training_replan_steps = resolved_replan_steps
            self.training_action_horizon = resolved_action_horizon
        if num_video_frames is not None:
            resolved_video_frames = int(num_video_frames)
            temporal_factor = int(
                getattr(self.vae, "temporal_downsample_factor", 1)
            )
            if resolved_video_frames <= 1:
                raise ValueError("num_video_frames must be greater than one")
            if (resolved_video_frames - 1) % temporal_factor:
                raise ValueError(
                    "num_video_frames-1 must be divisible by the VAE temporal "
                    f"downsample factor {temporal_factor}"
                )
            if action_horizon is not None and int(action_horizon) % (
                resolved_video_frames - 1
            ):
                raise ValueError(
                    "action_horizon must be divisible by num_video_frames-1"
                )
            self.training_num_video_frames = resolved_video_frames

    def validate_temporal_contract(
        self, *, replan_steps: int, action_horizon: int
    ) -> None:
        """Reject training/evaluation clocks that differ from the checkpoint."""

        if self.training_replan_steps is not None and int(replan_steps) != int(
            self.training_replan_steps
        ):
            raise ValueError(
                "replan_steps differs from the model temporal contract: "
                f"expected={self.training_replan_steps} got={replan_steps}"
            )
        if self.training_action_horizon is not None and int(action_horizon) != int(
            self.training_action_horizon
        ):
            raise ValueError(
                "action_horizon differs from the model temporal contract: "
                f"expected={self.training_action_horizon} got={action_horizon}"
            )

    def training_loss(self, sample, tiled: bool = False):
        if "history_video" not in sample:
            return super().training_loss(sample, tiled=tiled)
        from leapbot_va.training import causal_history_training_loss

        return causal_history_training_loss(self, sample, tiled=tiled)

    def auxiliary_trainable_modules(self) -> tuple[nn.Module, ...]:
        if all(depth == self.mot.num_layers for depth in self.training_exit_depths):
            return ()
        return (self.action_exit_heads, self.video_exit_heads)

    def create_memory(
        self,
        *,
        exit_depth: int = 30,
        causal_mode: str | None = None,
        max_history_blocks: int = 70,
        retained_history_blocks: int | None = None,
        action_horizon: int | None = None,
        replan_steps: int | None = None,
    ) -> LeapMemoryState:
        if exit_depth not in self.exit_depths:
            raise ValueError(
                f"model supports exit depths {self.exit_depths}, got {exit_depth}"
            )
        if exit_depth not in self.trained_exit_depths:
            raise ValueError(
                f"exit depth {exit_depth} was not trained in the loaded checkpoint; "
                f"available trained exits are {self.trained_exit_depths}"
            )
        resolved_causal_mode = self.causal_mode if causal_mode is None else causal_mode
        if resolved_causal_mode != self.causal_mode:
            raise ValueError(
                "memory/model causal mode mismatch: "
                f"memory={resolved_causal_mode} model={self.causal_mode}"
            )
        resolved_action_horizon = (
            self.training_action_horizon
            if action_horizon is None
            else int(action_horizon)
        )
        resolved_replan_steps = (
            self.training_replan_steps if replan_steps is None else int(replan_steps)
        )
        # Directly constructed research/toy models predating the explicit
        # contract retain the public FastWAM defaults. Production LeapBot models
        # always configure and checkpoint these values.
        if resolved_action_horizon is None:
            resolved_action_horizon = 32
        if resolved_replan_steps is None:
            resolved_replan_steps = 10
        self.validate_temporal_contract(
            replan_steps=resolved_replan_steps,
            action_horizon=resolved_action_horizon,
        )
        return LeapMemoryState(
            config=LeapMemoryConfig(
                exit_depth=exit_depth,
                causal_mode=resolved_causal_mode,
                max_history_blocks=max_history_blocks,
                retained_history_blocks=retained_history_blocks,
                action_horizon=resolved_action_horizon,
                replan_steps=resolved_replan_steps,
            )
        )

    @staticmethod
    def reset_memory(memory: LeapMemoryState) -> None:
        memory.reset()

    def _validate_memory_compatibility(self, memory: LeapMemoryState) -> None:
        if memory.config.exit_depth not in self.exit_depths:
            raise ValueError(
                "memory exit depth is unsupported by this model: "
                f"memory={memory.config.exit_depth} model={self.exit_depths}"
            )
        if memory.config.exit_depth not in self.trained_exit_depths:
            raise ValueError(
                "memory exit depth was not trained in the loaded checkpoint: "
                f"memory={memory.config.exit_depth} trained={self.trained_exit_depths}"
            )
        if memory.config.causal_mode != self.causal_mode:
            raise ValueError(
                "memory/model causal mode mismatch: "
                f"memory={memory.config.causal_mode} model={self.causal_mode}; "
                "reset and create memory from this model"
            )
        self.validate_temporal_contract(
            replan_steps=memory.config.replan_steps,
            action_horizon=memory.config.action_horizon,
        )

    @staticmethod
    def _prompt_fingerprint(
        prompt: Optional[str],
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
    ) -> str:
        if prompt is not None:
            if context is not None or context_mask is not None:
                raise ValueError("prompt and context/context_mask are mutually exclusive")
            return hashlib.sha256(f"prompt:{prompt}".encode("utf-8")).hexdigest()
        if context is None or context_mask is None:
            raise ValueError("either prompt or both context/context_mask are required")

        # Hash the complete caller-provided language state.  Moment summaries
        # are not injective (for example, a token permutation has the same sum
        # and squared sum) and could silently let a context switch reuse an
        # incompatible episode cache.  Proprio is deliberately excluded: it is
        # expected to change at every replanning boundary.
        digest = hashlib.sha256()
        for label, tensor in (("context", context), ("context_mask", context_mask)):
            value = tensor.detach().to(device="cpu").contiguous()
            digest.update(
                f"{label}:{tuple(value.shape)}:{value.dtype}:".encode("utf-8")
            )
            digest.update(value.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def _prepare_inference_context(
        self,
        *,
        prompt: Optional[str],
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("prompt and context/context_mask are mutually exclusive")
        if not use_prompt and not use_context:
            raise ValueError("either prompt or both context/context_mask must be provided")
        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("context and context_mask must be provided together")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError("context/context_mask must be [B,L,D]/[B,L]")
            context = context.to(device=self.device, dtype=self.torch_dtype)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool)

        if not bool(torch.isfinite(context).all().item()):
            raise ValueError("language context contains non-finite values")
        if self.proprio_encoder is not None and proprio is None:
            raise ValueError(
                "proprio is required for memory inference because the loaded "
                "model was trained with proprio conditioning"
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("proprio was provided but proprio_encoder is disabled")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 2 or proprio.shape != (1, self.proprio_dim):
                raise ValueError(
                    f"proprio must be [D] or [1,D] with D={self.proprio_dim}"
                )
            if not bool(torch.isfinite(proprio).all().item()):
                raise ValueError("proprio contains non-finite values")
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
            )
        return context, context_mask

    def _action_head_at_depth(self, depth: int) -> nn.Module:
        if depth == self.mot.num_layers:
            return self.action_expert.head
        return self.action_exit_heads[str(depth)]

    def _prepare_real_observation_pre_dit(
        self,
        *,
        latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        block_index: int,
    ) -> dict[str, Any]:
        """Build the shared runtime/training pre-DiT state for one real frame."""

        if latents.ndim != 5 or tuple(latents.shape[:1]) != (1,):
            raise ValueError("real observation latents must be [1,C,1,H,W]")
        num_latent_frames = int(latents.shape[2])
        if num_latent_frames != 1:
            raise ValueError(
                "one real observation must encode to exactly one latent frame"
            )
        if int(block_index) < 0:
            raise ValueError("observation block_index must be non-negative")
        timestep = torch.zeros(
            (1,), device=self.device, dtype=latents.dtype
        )
        pre_state = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=bool(
                getattr(self.video_expert, "fuse_vae_embedding_in_latents", False)
            ),
            frame_position_ids=self.temporal_positions.local_video_rope_ids(
                num_latent_frames, device=self.device
            ),
        )
        return self.temporal_positions.apply_video_pre_dit(
            pre_state,
            torch.full(
                (num_latent_frames,),
                int(block_index),
                dtype=torch.long,
                device=self.device,
            ),
        )

    def _prepare_action_segment_pre_dit(
        self,
        *,
        actions: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        absolute_start: int,
        block_index: int,
    ) -> dict[str, Any]:
        """Build one clean or noisy action block with rollout-identical clocks."""

        if actions.ndim != 3 or int(actions.shape[0]) != 1:
            raise ValueError("action segment must be [1,T,D]")
        if timestep.shape != (1,):
            raise ValueError("action segment timestep must have shape [1]")
        action_length = int(actions.shape[1])
        if action_length <= 0:
            raise ValueError("action segment must be non-empty")
        if int(absolute_start) < 0 or int(block_index) < 0:
            raise ValueError("absolute action clocks must be non-negative")
        pre_state = self.action_expert.pre_dit(
            action_tokens=actions,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            position_ids=self.temporal_positions.local_action_rope_ids(
                action_length, device=self.device
            ),
        )
        return self.temporal_positions.apply_action_pre_dit(
            pre_state,
            torch.arange(
                int(absolute_start),
                int(absolute_start) + action_length,
                device=self.device,
            ),
            torch.full(
                (action_length,),
                int(block_index),
                dtype=torch.long,
                device=self.device,
            ),
        )

    def _prepare_future_video_pre_dit(
        self,
        *,
        latents: torch.Tensor,
        frame_timesteps: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        block_index: int,
    ) -> dict[str, Any]:
        """Build one transient imagined-video segment after the real frame."""

        if latents.ndim != 5 or int(latents.shape[0]) != 1:
            raise ValueError("future video latents must be [1,C,F,H,W]")
        num_frames = int(latents.shape[2])
        if num_frames <= 0:
            raise ValueError("future video must contain at least one latent frame")
        if tuple(frame_timesteps.shape) != (1, num_frames):
            raise ValueError(
                "future video frame_timesteps must have shape "
                f"(1,{num_frames}), got {tuple(frame_timesteps.shape)}"
            )
        pre_state = self.video_expert.pre_dit(
            x=latents,
            timestep=frame_timesteps[:, 0],
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=bool(
                getattr(self.video_expert, "fuse_vae_embedding_in_latents", False)
            ),
            frame_position_ids=torch.arange(
                1,
                num_frames + 1,
                device=self.device,
                dtype=torch.long,
            ),
            frame_timesteps=frame_timesteps,
        )
        return self.temporal_positions.apply_video_pre_dit(
            pre_state,
            torch.full(
                (num_frames,),
                int(block_index),
                dtype=torch.long,
                device=self.device,
            ),
        )

    def _video_head_at_depth(
        self,
        *,
        depth: int,
        hidden: torch.Tensor,
        pre_state: dict[str, Any],
    ) -> torch.Tensor:
        if depth == self.mot.num_layers:
            return self.video_expert.post_dit(hidden, pre_state)
        pred_tokens = self.video_exit_heads[str(depth)](hidden, pre_state["t"])
        return self.video_expert.unpatchify(
            pred_tokens, pre_state["meta"]["grid_size"]
        )

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        num_video_frames: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        memory: Optional[LeapMemoryState] = None,
        profile: bool = False,
        future_video_denoise_steps: Optional[int] = None,
    ) -> dict[str, Any]:
        if memory is None:
            return super().infer_action(
                prompt=prompt,
                input_image=input_image,
                action_horizon=action_horizon,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                negative_prompt=negative_prompt,
                text_cfg_scale=text_cfg_scale,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
            )
        del negative_prompt, text_cfg_scale  # FastWAM release inference uses scale 1.
        self.eval()
        self._validate_memory_compatibility(memory)
        if action_horizon != memory.config.action_horizon:
            raise ValueError(
                f"action_horizon must remain {memory.config.action_horizon} for this episode"
            )
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError("input_image must have shape [1,3,H,W] or [3,H,W]")
        if input_image.shape[-2] % 16 or input_image.shape[-1] % 16:
            raise ValueError("input_image height and width must be multiples of 16")
        if not bool(torch.isfinite(input_image).all().item()):
            raise ValueError("input_image contains non-finite values")

        if profile and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        model_start = time.perf_counter()
        snapshot = memory.snapshot()
        timings: dict[str, float] = {}
        try:
            conditioning_start = time.perf_counter()
            memory.bind_prompt(
                self._prompt_fingerprint(prompt, context, context_mask)
            )
            block_index = memory.begin_observation()
            context, context_mask = self._prepare_inference_context(
                prompt=prompt,
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )
            if profile and self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            timings["conditioning_s"] = time.perf_counter() - conditioning_start

            t0 = time.perf_counter()
            input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
            first_frame_latents = self._encode_input_image_latents_tensor(
                input_image=input_image,
                tiled=tiled,
            )
            video_pre = self._prepare_real_observation_pre_dit(
                latents=first_frame_latents,
                context=context,
                context_mask=context_mask,
                block_index=block_index,
            )
            video_history = memory.materialize(memory.selected_segments_for_video())
            _, video_kv = self.mot.prefill_expert_segment(
                expert_name="video",
                tokens=video_pre["tokens"],
                freqs=video_pre["freqs"],
                t_mod=video_pre["t_mod"],
                context_payload={
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                history_kv=video_history,
                max_layers=memory.config.exit_depth,
            )
            del video_history
            video_seq_len = int(video_pre["tokens"].shape[1])
            video_segment = KVSegment(
                modality="video",
                block_index=block_index,
                positions=torch.full(
                    (video_seq_len,), block_index, dtype=torch.long, device=self.device
                ),
                keys=[item["k"] for item in video_kv],
                values=[item["v"] for item in video_kv],
            )
            memory.append_observation(
                video_segment,
                context=context,
                context_mask=context_mask,
            )
            if profile and self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            timings["observation_prefill_s"] = time.perf_counter() - t0

            future_setup_start = time.perf_counter()
            resolved_num_video_frames = (
                (
                    action_horizon + 1
                    if self.training_num_video_frames is None
                    else self.training_num_video_frames
                )
                if num_video_frames is None
                else int(num_video_frames)
            )
            if (
                self.training_num_video_frames is not None
                and resolved_num_video_frames != self.training_num_video_frames
            ):
                raise ValueError(
                    "num_video_frames differs from the training contract: "
                    f"expected={self.training_num_video_frames} "
                    f"got={resolved_num_video_frames}"
                )
            temporal_factor = int(
                getattr(self.vae, "temporal_downsample_factor", 1)
            )
            if resolved_num_video_frames <= 1 or (
                resolved_num_video_frames - 1
            ) % temporal_factor:
                raise ValueError(
                    "num_video_frames-1 must be positive and divisible by the "
                    f"VAE temporal factor {temporal_factor}"
                )
            if action_horizon % (resolved_num_video_frames - 1):
                raise ValueError(
                    "action_horizon must be divisible by num_video_frames-1"
                )
            num_future_latent_frames = (
                resolved_num_video_frames - 1
            ) // temporal_factor
            video_generator = (
                None
                if seed is None
                else torch.Generator(device=rand_device).manual_seed(seed)
            )
            future_video_latents = torch.randn(
                (
                    1,
                    int(first_frame_latents.shape[1]),
                    num_future_latent_frames,
                    int(first_frame_latents.shape[3]),
                    int(first_frame_latents.shape[4]),
                ),
                generator=video_generator,
                device=rand_device,
                dtype=torch.float32,
            ).to(device=self.device, dtype=self.torch_dtype)
            future_history_kv = memory.materialize(
                memory.selected_segments_for_future_video()
            )
            if future_history_kv is None:
                raise RuntimeError("imagined video has no current real observation")
            infer_timesteps_video, infer_deltas_video = (
                self.infer_video_scheduler.build_inference_schedule(
                    num_inference_steps=num_inference_steps,
                    device=self.device,
                    dtype=future_video_latents.dtype,
                    shift_override=sigma_shift,
                )
            )
            configured_video_steps = (
                self.future_video_denoise_steps
                if future_video_denoise_steps is None
                else int(future_video_denoise_steps)
            )
            video_steps = (
                num_inference_steps
                if configured_video_steps == -1
                else configured_video_steps
            )
            if video_steps <= 0 or video_steps > num_inference_steps:
                raise ValueError(
                    "future video denoise steps must be -1 or in "
                    f"[1,{num_inference_steps}], got {configured_video_steps}"
                )
            if profile and self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            timings["future_video_setup_s"] = (
                time.perf_counter() - future_setup_start
            )

            video_denoise_start = time.perf_counter()
            for step_index in range(video_steps):
                step_t_video = infer_timesteps_video[step_index]
                step_delta_video = infer_deltas_video[step_index]
                frame_timesteps = step_t_video.reshape(1, 1).expand(
                    -1, num_future_latent_frames
                )
                future_video_pre = self._prepare_future_video_pre_dit(
                    latents=future_video_latents,
                    frame_timesteps=frame_timesteps,
                    context=context,
                    context_mask=context_mask,
                    block_index=block_index,
                )
                video_hidden_by_depth, _ = self.mot.prefill_expert_segment(
                    expert_name="video",
                    tokens=future_video_pre["tokens"],
                    freqs=future_video_pre["freqs"],
                    t_mod=future_video_pre["t_mod"],
                    context_payload={
                        "context": future_video_pre["context"],
                        "mask": future_video_pre["context_mask"],
                    },
                    history_kv=future_history_kv,
                    max_layers=memory.config.exit_depth,
                    exit_depths=(memory.config.exit_depth,),
                )
                if not isinstance(video_hidden_by_depth, dict):
                    raise RuntimeError("video denoising did not return depth output")
                pred_video = self._video_head_at_depth(
                    depth=memory.config.exit_depth,
                    hidden=video_hidden_by_depth[memory.config.exit_depth],
                    pre_state=future_video_pre,
                )
                future_video_latents = self.infer_video_scheduler.step(
                    pred_video,
                    step_delta_video,
                    future_video_latents,
                )
            if profile and self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            timings["future_video_denoise_s"] = (
                time.perf_counter() - video_denoise_start
            )

            # LingBot performs one final video forward at the attained sigma
            # and caches that representation for the action denoiser.  This KV
            # stays local to the current call.
            future_cache_start = time.perf_counter()
            attained_timestep = (
                infer_timesteps_video[video_steps]
                if video_steps < num_inference_steps
                else torch.zeros(
                    (), device=self.device, dtype=future_video_latents.dtype
                )
            )
            attained_frame_timesteps = attained_timestep.reshape(1, 1).expand(
                -1, num_future_latent_frames
            )
            future_condition_pre = self._prepare_future_video_pre_dit(
                latents=future_video_latents,
                frame_timesteps=attained_frame_timesteps,
                context=context,
                context_mask=context_mask,
                block_index=block_index,
            )
            _, future_condition_kv = self.mot.prefill_expert_segment(
                expert_name="video",
                tokens=future_condition_pre["tokens"],
                freqs=future_condition_pre["freqs"],
                t_mod=future_condition_pre["t_mod"],
                context_payload={
                    "context": future_condition_pre["context"],
                    "mask": future_condition_pre["context_mask"],
                },
                history_kv=future_history_kv,
                max_layers=memory.config.exit_depth,
            )
            transient_future_cache_bytes = sum(
                tensor.numel() * tensor.element_size()
                for layer in future_condition_kv
                for tensor in (layer["k"], layer["v"])
            )
            if profile and self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            timings["future_video_cache_s"] = (
                time.perf_counter() - future_cache_start
            )

            action_setup_start = time.perf_counter()
            generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
            latents_action = torch.randn(
                (1, action_horizon, self.action_expert.action_dim),
                generator=generator,
                device=rand_device,
                dtype=torch.float32,
            ).to(device=self.device, dtype=self.torch_dtype)
            history_kv = memory.materialize(memory.selected_segments_for_action())
            if history_kv is None:
                raise RuntimeError("current real observation was not committed to memory")
            history_kv = [
                {
                    "k": torch.cat(
                        [persistent["k"], transient["k"]], dim=1
                    ),
                    "v": torch.cat(
                        [persistent["v"], transient["v"]], dim=1
                    ),
                }
                for persistent, transient in zip(
                    history_kv, future_condition_kv
                )
            ]
            if profile and self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            timings["action_setup_s"] = time.perf_counter() - action_setup_start

            t1 = time.perf_counter()
            infer_timesteps, infer_deltas = self.infer_action_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=self.device,
                dtype=latents_action.dtype,
                shift_override=sigma_shift,
            )
            for step_t, step_delta in zip(infer_timesteps, infer_deltas):
                timestep_action = step_t.unsqueeze(0).to(
                    dtype=latents_action.dtype, device=self.device
                )
                action_pre = self._prepare_action_segment_pre_dit(
                    actions=latents_action,
                    timestep=timestep_action,
                    context=context,
                    context_mask=context_mask,
                    absolute_start=memory.next_action_position,
                    block_index=block_index,
                )
                action_hidden = self.mot.forward_action_with_history(
                    action_tokens=action_pre["tokens"],
                    action_freqs=action_pre["freqs"],
                    action_t_mod=action_pre["t_mod"],
                    action_context_payload={
                        "context": action_pre["context"],
                        "mask": action_pre["context_mask"],
                    },
                    history_kv=history_kv,
                    max_layers=memory.config.exit_depth,
                )
                pred_action = self._action_head_at_depth(memory.config.exit_depth)(action_hidden)
                latents_action = self.infer_action_scheduler.step(
                    pred_action, step_delta, latents_action
                )
            if profile and self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            timings["action_denoise_s"] = time.perf_counter() - t1
            if not bool(torch.isfinite(latents_action).all().item()):
                raise RuntimeError("action denoising produced non-finite commands")
        except Exception:
            memory.rollback(snapshot)
            raise

        timings["causal_model_s"] = time.perf_counter() - model_start
        timings["causal_model_residual_s"] = max(
            0.0,
            timings["causal_model_s"]
            - sum(
                timings[name]
                for name in (
                    "conditioning_s",
                    "observation_prefill_s",
                    "future_video_setup_s",
                    "future_video_denoise_s",
                    "future_video_cache_s",
                    "action_setup_s",
                    "action_denoise_s",
                )
            ),
        )

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
            "memory": {
                "completed_blocks": memory.completed_blocks,
                "retained_history_blocks": memory.retained_completed_blocks,
                "cache_bytes": memory.cache_nbytes,
                "transient_future_video_cache_bytes": transient_future_cache_bytes,
                "token_counts": memory.token_counts,
                "phase": memory.phase.value,
            },
            "timing": timings,
            "future_video_condition": {
                "physical_frames": resolved_num_video_frames,
                "latent_frames": num_future_latent_frames,
                "denoise_steps": video_steps,
                "schedule_steps": num_inference_steps,
                "attained_timestep": float(attained_timestep.float().item()),
            },
        }

    @torch.no_grad()
    def commit_executed_actions(
        self,
        memory: LeapMemoryState,
        actions_model_space: torch.Tensor,
        *,
        profile: bool = False,
    ) -> dict[str, Any]:
        """Commit only actions actually sent to the controller.

        Callers are responsible for converting clipped/binarized environment
        commands back into the normalized model action space before this call.
        """

        self.eval()
        self._validate_memory_compatibility(memory)
        if actions_model_space.ndim == 2:
            actions_model_space = actions_model_space.unsqueeze(0)
        if actions_model_space.ndim != 3 or actions_model_space.shape[0] != 1:
            raise ValueError("actions_model_space must be [T,D] or [1,T,D]")
        if actions_model_space.shape[2] != self.action_expert.action_dim:
            raise ValueError(
                f"action dimension must be {self.action_expert.action_dim}"
            )
        if not bool(torch.isfinite(actions_model_space).all().item()):
            raise ValueError("actions_model_space contains non-finite values")
        executed = int(actions_model_space.shape[1])
        if executed <= 0 or executed > memory.config.replan_steps:
            raise ValueError(
                f"commit accepts 1..{memory.config.replan_steps} executed actions, got {executed}"
            )
        if memory.pending_context is None or memory.pending_context_mask is None:
            raise RuntimeError("no pending observation context; call infer_action first")

        snapshot = memory.snapshot()
        t0 = time.perf_counter()
        try:
            actions = actions_model_space.to(device=self.device, dtype=self.torch_dtype)
            timestep = torch.zeros((1,), device=self.device, dtype=actions.dtype)
            absolute_start = memory.next_action_position
            block_index = memory.completed_blocks
            action_pre = self._prepare_action_segment_pre_dit(
                actions=actions,
                timestep=timestep,
                context=memory.pending_context,
                context_mask=memory.pending_context_mask,
                absolute_start=absolute_start,
                block_index=block_index,
            )
            history_kv = memory.materialize(memory.selected_segments_for_action())
            if history_kv is None:
                raise RuntimeError("cannot commit actions without observation history")
            _, action_kv = self.mot.prefill_expert_segment(
                expert_name="action",
                tokens=action_pre["tokens"],
                freqs=action_pre["freqs"],
                t_mod=action_pre["t_mod"],
                context_payload={
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
                history_kv=history_kv,
                max_layers=memory.config.exit_depth,
            )
            memory.append_actions(
                KVSegment(
                    modality="action",
                    block_index=block_index,
                    positions=torch.arange(
                        absolute_start,
                        absolute_start + executed,
                        device=self.device,
                    ),
                    keys=[item["k"] for item in action_kv],
                    values=[item["v"] for item in action_kv],
                )
            )
            if profile and self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        except Exception:
            memory.rollback(snapshot)
            raise

        return {
            "executed_actions": executed,
            "completed_blocks": memory.completed_blocks,
            "retained_history_blocks": memory.retained_completed_blocks,
            "cache_bytes": memory.cache_nbytes,
            "commit_s": time.perf_counter() - t0,
        }

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "action_exit_heads": self.action_exit_heads.state_dict(),
            "video_exit_heads": self.video_exit_heads.state_dict(),
            "temporal_positions": self.temporal_positions.state_dict(),
            "temporal_position_scheme": TEMPORAL_POSITION_SCHEME,
            "exit_depths": self.exit_depths,
            "training_exit_depths": self.training_exit_depths,
            "trained_exit_depths": self.training_exit_depths,
            "causal_mode": self.causal_mode,
            "training_strategy": self.training_strategy,
            "history_training_mode": self.history_training_mode,
            "training_replan_steps": self.training_replan_steps,
            "training_action_horizon": self.training_action_horizon,
            "training_num_video_frames": self.training_num_video_frames,
            "future_video_conditioning": self.future_video_conditioning,
            "future_video_condition_noise_probability": (
                self.future_video_condition_noise_probability
            ),
            "future_video_condition_min_u": self.future_video_condition_min_u,
            "future_video_condition_max_u": self.future_video_condition_max_u,
            "history_vae_batch_chunk_size": self.history_vae_batch_chunk_size,
            "video_lora_config": self.video_lora_config.__dict__,
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        run_contract = os.environ.get("LEAPBOT_RUN_CONTRACT_SHA256")
        code_commit = os.environ.get("LEAPBOT_CODE_COMMIT")
        if run_contract:
            payload["run_contract_sha256"] = run_contract
        if code_commit:
            payload["code_commit"] = code_commit
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    @staticmethod
    def _validate_checkpoint_state_dict(
        module: nn.Module,
        state_dict: Any,
        *,
        label: str,
        strict: bool,
    ) -> None:
        """Validate a module state dict without writing into ``module``."""

        if not isinstance(state_dict, Mapping):
            raise ValueError(f"{label} must be a state_dict mapping")
        expected = module.state_dict()
        expected_keys = set(expected)
        checkpoint_keys = set(state_dict)
        missing_keys = sorted(expected_keys - checkpoint_keys)
        unexpected_keys = sorted(checkpoint_keys - expected_keys)
        if strict and (missing_keys or unexpected_keys):
            raise ValueError(
                f"{label} key mismatch: missing={missing_keys[:8]} "
                f"unexpected={unexpected_keys[:8]}"
            )

        for key in sorted(expected_keys & checkpoint_keys):
            expected_value = expected[key]
            checkpoint_value = state_dict[key]
            if isinstance(expected_value, torch.Tensor):
                if not isinstance(checkpoint_value, torch.Tensor):
                    raise ValueError(
                        f"{label} value type mismatch for {key!r}: "
                        f"checkpoint={type(checkpoint_value).__name__} model=Tensor"
                    )
                if checkpoint_value.shape != expected_value.shape:
                    raise ValueError(
                        f"{label} shape mismatch for {key!r}: "
                        f"checkpoint={tuple(checkpoint_value.shape)} "
                        f"model={tuple(expected_value.shape)}"
                    )
            elif type(checkpoint_value) is not type(expected_value):
                raise ValueError(
                    f"{label} value type mismatch for {key!r}: "
                    f"checkpoint={type(checkpoint_value).__name__} "
                    f"model={type(expected_value).__name__}"
                )

    @staticmethod
    def _validate_optimizer_checkpoint(optimizer, optimizer_state: Any) -> None:
        """Mirror Optimizer's structural checks before any model mutation."""

        if not isinstance(optimizer_state, Mapping):
            raise ValueError("checkpoint optimizer state must be a mapping")
        if "state" not in optimizer_state or "param_groups" not in optimizer_state:
            raise ValueError(
                "checkpoint optimizer state must contain state and param_groups"
            )
        saved_groups = optimizer_state["param_groups"]
        if not isinstance(saved_groups, (list, tuple)):
            raise ValueError("checkpoint optimizer param_groups must be a sequence")
        if len(saved_groups) != len(optimizer.param_groups):
            raise ValueError(
                "loaded optimizer state has a different number of parameter groups"
            )
        for index, (saved_group, current_group) in enumerate(
            zip(saved_groups, optimizer.param_groups)
        ):
            if not isinstance(saved_group, Mapping) or "params" not in saved_group:
                raise ValueError(
                    f"checkpoint optimizer param group {index} is missing params"
                )
            if len(saved_group["params"]) != len(current_group["params"]):
                raise ValueError(
                    "loaded optimizer state contains a parameter group that "
                    "doesn't match the size of optimizer's group"
                )

    def load_checkpoint(self, path, optimizer=None):
        # Loading is deliberately split into a read-only preflight and a commit.
        # A native checkpoint mismatch must never leave a partially loaded MoT,
        # auxiliary head, temporal clock, or optimizer-visible model behind.
        payload = torch.load(path, map_location="cpu")
        checkpoint_causal_mode = payload.get("causal_mode")
        native_marker_fields = {
            "action_exit_heads",
            "video_exit_heads",
            "temporal_positions",
            "temporal_position_scheme",
            "exit_depths",
            "training_exit_depths",
            "trained_exit_depths",
            "causal_mode",
            "training_strategy",
            "history_training_mode",
            "training_replan_steps",
            "training_action_horizon",
            "training_num_video_frames",
            "future_video_conditioning",
            "future_video_condition_noise_probability",
            "future_video_condition_min_u",
            "future_video_condition_max_u",
            "history_vae_batch_chunk_size",
            "video_lora_config",
        }
        is_native_leapbot = bool(native_marker_fields & set(payload))
        if is_native_leapbot:
            required_native_fields = {
                "mot",
                "action_exit_heads",
                "video_exit_heads",
                "temporal_positions",
                "temporal_position_scheme",
                "exit_depths",
                "training_exit_depths",
                "trained_exit_depths",
                "causal_mode",
                "training_strategy",
                "history_training_mode",
                "training_replan_steps",
                "training_action_horizon",
                "training_num_video_frames",
                "future_video_conditioning",
                "future_video_condition_noise_probability",
                "future_video_condition_min_u",
                "future_video_condition_max_u",
                "history_vae_batch_chunk_size",
                "video_lora_config",
            }
            missing_native_fields = sorted(required_native_fields - set(payload))
            if self.proprio_encoder is not None and "proprio_encoder" not in payload:
                missing_native_fields.append("proprio_encoder")
            if self.proprio_encoder is None and "proprio_encoder" in payload:
                raise ValueError(
                    "native LeapBot checkpoint contains proprio_encoder but the "
                    "configured model has proprio conditioning disabled"
                )
            if missing_native_fields:
                raise ValueError(
                    "native LeapBot checkpoint is incomplete; missing fields="
                    f"{sorted(set(missing_native_fields))}"
                )
            checkpoint_exit_depths = tuple(
                int(depth) for depth in payload["exit_depths"]
            )
            if checkpoint_exit_depths != self.exit_depths:
                raise ValueError(
                    "checkpoint/model exit architecture mismatch: "
                    f"checkpoint={checkpoint_exit_depths} model={self.exit_depths}"
                )
            checkpoint_position_scheme = str(payload["temporal_position_scheme"])
            if checkpoint_position_scheme != TEMPORAL_POSITION_SCHEME:
                raise ValueError(
                    "checkpoint temporal-position scheme mismatch: "
                    f"checkpoint={checkpoint_position_scheme!r} "
                    f"model={TEMPORAL_POSITION_SCHEME!r}"
                )
            checkpoint_training_depths = tuple(
                sorted({int(depth) for depth in payload["training_exit_depths"]})
            )
            checkpoint_trained_depths = tuple(
                sorted({int(depth) for depth in payload["trained_exit_depths"]})
            )
            if (
                not checkpoint_training_depths
                or self.mot.num_layers not in checkpoint_training_depths
                or any(
                    depth not in checkpoint_exit_depths
                    for depth in checkpoint_training_depths
                )
            ):
                raise ValueError(
                    "checkpoint training exits are incompatible with its architecture: "
                    f"training={checkpoint_training_depths} "
                    f"architecture={checkpoint_exit_depths}"
                )
            if (
                not checkpoint_trained_depths
                or self.mot.num_layers not in checkpoint_trained_depths
                or any(
                    depth not in checkpoint_exit_depths
                    for depth in checkpoint_trained_depths
                )
            ):
                raise ValueError(
                    "checkpoint trained exits are incompatible with this model: "
                    f"checkpoint={checkpoint_trained_depths} "
                    f"model={checkpoint_exit_depths}"
                )
            if checkpoint_trained_depths != checkpoint_training_depths:
                raise ValueError(
                    "checkpoint training/trained exit metadata mismatch: "
                    f"training={checkpoint_training_depths} "
                    f"trained={checkpoint_trained_depths}"
                )
        checkpoint_training_strategy = payload.get("training_strategy")
        if (
            checkpoint_training_strategy is not None
            and str(checkpoint_training_strategy) != self.training_strategy
        ):
            raise ValueError(
                "checkpoint/model training strategy mismatch: "
                f"checkpoint={checkpoint_training_strategy} "
                f"model={self.training_strategy}"
            )
        if (
            checkpoint_causal_mode is not None
            and str(checkpoint_causal_mode) != self.causal_mode
        ):
            raise ValueError(
                "checkpoint/model causal mode mismatch: "
                f"checkpoint={checkpoint_causal_mode} model={self.causal_mode}"
            )
        checkpoint_history_mode = payload.get("history_training_mode")
        if (
            checkpoint_history_mode is not None
            and str(checkpoint_history_mode) != self.history_training_mode
        ):
            raise ValueError(
                "checkpoint/model history training mode mismatch: "
                f"checkpoint={checkpoint_history_mode} model={self.history_training_mode}"
            )
        checkpoint_replan_steps = payload.get("training_replan_steps")
        checkpoint_action_horizon = payload.get("training_action_horizon")
        if (checkpoint_replan_steps is None) != (checkpoint_action_horizon is None):
            raise ValueError(
                "checkpoint temporal contract must contain both replan_steps and action_horizon"
            )
        resolved_checkpoint_replan_steps = None
        resolved_checkpoint_action_horizon = None
        if checkpoint_replan_steps is not None:
            resolved_checkpoint_replan_steps = int(checkpoint_replan_steps)
            resolved_checkpoint_action_horizon = int(checkpoint_action_horizon)
            if resolved_checkpoint_replan_steps <= 0:
                raise ValueError("checkpoint replan_steps must be positive")
            if resolved_checkpoint_action_horizon < resolved_checkpoint_replan_steps:
                raise ValueError(
                    "checkpoint action_horizon must be greater than or equal to "
                    "replan_steps"
                )
            if (
                self.training_replan_steps is not None
                and resolved_checkpoint_replan_steps
                != int(self.training_replan_steps)
            ):
                raise ValueError(
                    "replan_steps differs from the model temporal contract: "
                    f"expected={self.training_replan_steps} "
                    f"got={resolved_checkpoint_replan_steps}"
                )
            if (
                self.training_action_horizon is not None
                and resolved_checkpoint_action_horizon
                != int(self.training_action_horizon)
            ):
                raise ValueError(
                    "action_horizon differs from the model temporal contract: "
                    f"expected={self.training_action_horizon} "
                    f"got={resolved_checkpoint_action_horizon}"
                )
        checkpoint_video_frames = payload.get("training_num_video_frames")
        if checkpoint_video_frames is not None:
            resolved_checkpoint_video_frames = int(checkpoint_video_frames)
            if resolved_checkpoint_video_frames <= 1:
                raise ValueError("checkpoint num_video_frames must be greater than one")
            if (
                self.training_num_video_frames is not None
                and resolved_checkpoint_video_frames
                != int(self.training_num_video_frames)
            ):
                raise ValueError(
                    "num_video_frames differs from the model contract: "
                    f"expected={self.training_num_video_frames} "
                    f"got={resolved_checkpoint_video_frames}"
                )
        checkpoint_conditioning = payload.get("future_video_conditioning")
        if (
            checkpoint_conditioning is not None
            and str(checkpoint_conditioning) != self.future_video_conditioning
        ):
            raise ValueError(
                "future-video conditioning contract mismatch: "
                f"checkpoint={checkpoint_conditioning} "
                f"model={self.future_video_conditioning}"
            )
        for field, configured in (
            (
                "future_video_condition_noise_probability",
                self.future_video_condition_noise_probability,
            ),
            ("future_video_condition_min_u", self.future_video_condition_min_u),
            ("future_video_condition_max_u", self.future_video_condition_max_u),
        ):
            checkpoint_value = payload.get(field)
            if checkpoint_value is not None and float(checkpoint_value) != float(
                configured
            ):
                raise ValueError(
                    f"{field} differs from the model contract: "
                    f"checkpoint={checkpoint_value} model={configured}"
                )
        checkpoint_vae_chunk = payload.get("history_vae_batch_chunk_size")
        if (
            checkpoint_vae_chunk is not None
            and int(checkpoint_vae_chunk) != int(self.history_vae_batch_chunk_size)
        ):
            raise ValueError(
                "checkpoint/model history VAE batch chunk mismatch: "
                f"checkpoint={checkpoint_vae_chunk} "
                f"model={self.history_vae_batch_chunk_size}"
            )
        checkpoint_lora = payload.get("video_lora_config")
        if checkpoint_lora is not None:
            expected_lora = self.video_lora_config.__dict__
            normalized_checkpoint_lora = {
                "enabled": bool(checkpoint_lora.get("enabled", False)),
                "rank": int(checkpoint_lora.get("rank", -1)),
                "alpha": float(checkpoint_lora.get("alpha", float("nan"))),
                "dropout": float(checkpoint_lora.get("dropout", float("nan"))),
                "learning_rate_multiplier": float(
                    checkpoint_lora.get("learning_rate_multiplier", float("nan"))
                ),
            }
            if normalized_checkpoint_lora != expected_lora:
                raise ValueError(
                    "checkpoint/model video LoRA configuration mismatch: "
                    f"checkpoint={normalized_checkpoint_lora}, model={expected_lora}"
                )
        # Original FastWAM releases intentionally lack LeapBot metadata and
        # adapters, so their permissive load remains the initialization path.
        # A native LeapBot checkpoint, however, must have an exact MoT key set:
        # silently accepting missing LoRA/action weights would invalidate every
        # downstream comparison even if causal metadata happened to match.
        if is_native_leapbot:
            checkpoint_mot = payload.get("mot")
            if not isinstance(checkpoint_mot, dict):
                raise ValueError("native LeapBot checkpoint is missing a mot state_dict")
            self._validate_checkpoint_state_dict(
                self.mot,
                checkpoint_mot,
                label="native LeapBot checkpoint MoT",
                strict=True,
            )
            self._validate_checkpoint_state_dict(
                self.action_exit_heads,
                payload["action_exit_heads"],
                label="native LeapBot checkpoint action exit heads",
                strict=True,
            )
            self._validate_checkpoint_state_dict(
                self.video_exit_heads,
                payload["video_exit_heads"],
                label="native LeapBot checkpoint video exit heads",
                strict=True,
            )
            self._validate_checkpoint_state_dict(
                self.temporal_positions,
                payload["temporal_positions"],
                label="native LeapBot checkpoint temporal positions",
                strict=True,
            )
            if self.proprio_encoder is not None:
                self._validate_checkpoint_state_dict(
                    self.proprio_encoder,
                    payload["proprio_encoder"],
                    label="native LeapBot checkpoint proprio encoder",
                    strict=True,
                )
        else:
            if "mot" in payload:
                self._validate_checkpoint_state_dict(
                    self.mot,
                    payload["mot"],
                    label="FastWAM checkpoint MoT",
                    strict=False,
                )
            elif "dit" in payload:
                self._validate_checkpoint_state_dict(
                    self.video_expert,
                    payload["dit"],
                    label="legacy FastWAM checkpoint video DiT",
                    strict=False,
                )
            else:
                raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {path}")
            if self.proprio_encoder is not None and "proprio_encoder" in payload:
                self._validate_checkpoint_state_dict(
                    self.proprio_encoder,
                    payload["proprio_encoder"],
                    label="FastWAM checkpoint proprio encoder",
                    strict=True,
                )
        trained_exits = tuple(
            sorted(
                {
                    int(depth)
                    for depth in payload.get(
                        "trained_exit_depths",
                        payload.get("training_exit_depths", (self.mot.num_layers,)),
                    )
                }
            )
        )
        if not trained_exits:
            trained_exits = (self.mot.num_layers,)
        if self.mot.num_layers not in trained_exits or any(
            depth not in self.exit_depths for depth in trained_exits
        ):
            raise ValueError(
                "checkpoint trained exits are incompatible with this model: "
                f"checkpoint={trained_exits} model={self.exit_depths}"
            )
        has_trained_shallow_exits = any(depth != self.mot.num_layers for depth in trained_exits)

        if optimizer is not None and "optimizer" in payload:
            self._validate_optimizer_checkpoint(optimizer, payload["optimizer"])

        # Commit only after every metadata and tensor key/shape check above has
        # succeeded. Module loading below cannot discover a checkpoint mismatch.
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        if "mot" in payload:
            self.mot.load_state_dict(payload["mot"], strict=is_native_leapbot)
        else:
            self.video_expert.load_state_dict(payload["dit"], strict=False)
        if self.proprio_encoder is not None and "proprio_encoder" in payload:
            self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
        if is_native_leapbot:
            self.temporal_positions.load_state_dict(
                payload["temporal_positions"], strict=True
            )
        else:
            # Loading an original FastWAM checkpoint must always restore the
            # identity extension, even if this model instance was previously
            # used with trained temporal-position weights.
            self.temporal_positions.reset_parameters()
        if is_native_leapbot and has_trained_shallow_exits:
            self.action_exit_heads.load_state_dict(payload["action_exit_heads"], strict=True)
            self.video_exit_heads.load_state_dict(payload["video_exit_heads"], strict=True)
        else:
            # A FastWAM release or 30-layer-only LeapBot checkpoint has no
            # trained shallow heads. Initialize every exit from the now-loaded
            # final head, not from the random pre-checkpoint constructor state.
            for head in self.action_exit_heads.values():
                head.load_state_dict(self.action_expert.head.state_dict(), strict=True)
            for head in self.video_exit_heads.values():
                head.load_state_dict(self.video_expert.head.state_dict(), strict=True)
        self.trained_exit_depths = trained_exits
        if (
            resolved_checkpoint_replan_steps is not None
            and self.training_replan_steps is None
        ):
            self.training_replan_steps = resolved_checkpoint_replan_steps
            self.training_action_horizon = resolved_checkpoint_action_horizon
        if (
            checkpoint_video_frames is not None
            and self.training_num_video_frames is None
        ):
            self.training_num_video_frames = int(checkpoint_video_frames)
        return payload
