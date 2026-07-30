"""LeapBot-VA model: FastWAM with real-observation causal KV memory."""

from __future__ import annotations

import copy
import hashlib
import time
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
from leapbot_va.positions import HierarchicalTemporalPositionEmbedding


class LeapBotVA(FastWAM):
    """FastWAM inference with explicit persistent real-data-only memory.

    Future video latents are never instantiated by the memory path.  A call to
    :meth:`infer_action` commits one real observation segment transactionally;
    the caller must later commit only the commands actually sent to the robot.
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
        # exact-zero projections, so a release checkpoint remains an identity
        # at initialization.
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
        self.history_training_mode = "packed_full_bptt"
        self.video_lora_config = VideoLoRAConfig()
        self.video_lora_merged = False
        self.training_replan_steps: int | None = None
        self.training_action_horizon: int | None = None
        self.history_vae_batch_chunk_size = 2

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
        history_training_mode: str = "packed_full_bptt",
        replan_steps: int | None = None,
        action_horizon: int | None = None,
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
        if history_training_mode != "packed_full_bptt":
            raise ValueError(
                "LeapBot causal training requires packed_full_bptt; detached-prefix "
                f"training is retired because it drops historical gradients, got "
                f"{history_training_mode}"
            )
        self.history_training_mode = history_training_mode
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
                action_horizon=resolved_action_horizon,
                replan_steps=resolved_replan_steps,
            )
        )

    @staticmethod
    def reset_memory(memory: LeapMemoryState) -> None:
        memory.reset()

    def _validate_memory_compatibility(self, memory: LeapMemoryState) -> None:
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

        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("proprio was provided but proprio_encoder is disabled")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 2 or proprio.shape != (1, self.proprio_dim):
                raise ValueError(
                    f"proprio must be [D] or [1,D] with D={self.proprio_dim}"
                )
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

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
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

        snapshot = memory.snapshot()
        timings: dict[str, float] = {}
        try:
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

            t0 = time.perf_counter()
            input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
            first_frame_latents = self._encode_input_image_latents_tensor(
                input_image=input_image,
                tiled=tiled,
            )
            timestep_video = torch.zeros(
                (1,), device=self.device, dtype=first_frame_latents.dtype
            )
            num_latent_frames = int(first_frame_latents.shape[2])
            local_video_positions = self.temporal_positions.local_video_rope_ids(
                num_latent_frames,
                device=self.device,
            )
            absolute_video_blocks = torch.full(
                (num_latent_frames,),
                block_index,
                dtype=torch.long,
                device=self.device,
            )
            video_pre = self.video_expert.pre_dit(
                x=first_frame_latents,
                timestep=timestep_video,
                context=context,
                context_mask=context_mask,
                action=None,
                fuse_vae_embedding_in_latents=bool(
                    getattr(self.video_expert, "fuse_vae_embedding_in_latents", False)
                ),
                frame_position_ids=local_video_positions,
            )
            video_pre = self.temporal_positions.apply_video_pre_dit(
                video_pre,
                absolute_video_blocks,
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

            generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
            latents_action = torch.randn(
                (1, action_horizon, self.action_expert.action_dim),
                generator=generator,
                device=rand_device,
                dtype=torch.float32,
            ).to(device=self.device, dtype=self.torch_dtype)
            absolute_action_positions = torch.arange(
                memory.next_action_position,
                memory.next_action_position + action_horizon,
                device=self.device,
            )
            local_action_positions = self.temporal_positions.local_action_rope_ids(
                action_horizon,
                device=self.device,
            )
            absolute_action_blocks = torch.full(
                (action_horizon,),
                block_index,
                dtype=torch.long,
                device=self.device,
            )
            history_kv = memory.materialize(memory.selected_segments_for_action())
            if history_kv is None:
                raise RuntimeError("current real observation was not committed to memory")

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
                action_pre = self.action_expert.pre_dit(
                    action_tokens=latents_action,
                    timestep=timestep_action,
                    context=context,
                    context_mask=context_mask,
                    position_ids=local_action_positions,
                )
                action_pre = self.temporal_positions.apply_action_pre_dit(
                    action_pre,
                    absolute_action_positions,
                    absolute_action_blocks,
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
        except Exception:
            memory.rollback(snapshot)
            raise

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
            "memory": {
                "completed_blocks": memory.completed_blocks,
                "cache_bytes": memory.cache_nbytes,
                "token_counts": memory.token_counts,
                "phase": memory.phase.value,
            },
            "timing": timings,
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
            absolute_positions = torch.arange(
                memory.next_action_position,
                memory.next_action_position + executed,
                device=self.device,
            )
            local_positions = self.temporal_positions.local_action_rope_ids(
                executed,
                device=self.device,
            )
            absolute_blocks = torch.full(
                (executed,),
                memory.completed_blocks,
                dtype=torch.long,
                device=self.device,
            )
            timestep = torch.zeros((1,), device=self.device, dtype=actions.dtype)
            action_pre = self.action_expert.pre_dit(
                action_tokens=actions,
                timestep=timestep,
                context=memory.pending_context,
                context_mask=memory.pending_context_mask,
                position_ids=local_positions,
            )
            action_pre = self.temporal_positions.apply_action_pre_dit(
                action_pre,
                absolute_positions,
                absolute_blocks,
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
                    block_index=memory.completed_blocks,
                    positions=absolute_positions,
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
            "cache_bytes": memory.cache_nbytes,
            "commit_s": time.perf_counter() - t0,
        }

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "action_exit_heads": self.action_exit_heads.state_dict(),
            "video_exit_heads": self.video_exit_heads.state_dict(),
            "temporal_positions": self.temporal_positions.state_dict(),
            "exit_depths": self.exit_depths,
            "training_exit_depths": self.training_exit_depths,
            "trained_exit_depths": self.training_exit_depths,
            "causal_mode": self.causal_mode,
            "training_strategy": self.training_strategy,
            "history_training_mode": self.history_training_mode,
            "training_replan_steps": self.training_replan_steps,
            "training_action_horizon": self.training_action_horizon,
            "history_vae_batch_chunk_size": self.history_vae_batch_chunk_size,
            "video_lora_config": self.video_lora_config.__dict__,
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = super().load_checkpoint(path, optimizer=optimizer)
        checkpoint_causal_mode = payload.get("causal_mode")
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
        if checkpoint_replan_steps is not None:
            if self.training_replan_steps is None:
                self.training_replan_steps = int(checkpoint_replan_steps)
                self.training_action_horizon = int(checkpoint_action_horizon)
            self.validate_temporal_contract(
                replan_steps=int(checkpoint_replan_steps),
                action_horizon=int(checkpoint_action_horizon),
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
            checkpoint_lora_enabled = bool(checkpoint_lora.get("enabled", False))
            if checkpoint_lora_enabled != self.video_lora_config.enabled:
                raise ValueError(
                    "checkpoint/model video LoRA mismatch: "
                    f"checkpoint enabled={checkpoint_lora_enabled}, "
                    f"model enabled={self.video_lora_config.enabled}"
                )
        if "temporal_positions" in payload:
            self.temporal_positions.load_state_dict(
                payload["temporal_positions"],
                strict=True,
            )
        else:
            # Loading an original FastWAM checkpoint must always restore the
            # identity extension, even if this model instance was previously
            # used with trained temporal-position weights.
            self.temporal_positions.reset_parameters()
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
        self.trained_exit_depths = trained_exits
        has_trained_shallow_exits = any(depth != self.mot.num_layers for depth in trained_exits)
        if "action_exit_heads" in payload and has_trained_shallow_exits:
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
        return payload
