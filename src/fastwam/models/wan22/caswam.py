"""CasWAM: Speed-centric Full-history Memory Grounded World Action Model.

Adds full-history KV cache with temporal positional encoding and history
cross-attention to the ActionDiT, enabling the action expert to attend to
all past video observations for improved long-horizon robotic control.

VideoDiT remains fully frozen.  ActionDiT is fully trainable (~3B params
with history_cross_attn): self-attention, text cross-attention, FFN,
modulation, encoder, head, and the new history_cross_attn layers.
Memory-safe with ZeRO-1 on 4 GPUs (~12 GB/GPU optimizer states).
"""

from typing import Any, Optional

import torch
from PIL import Image

from fastwam.utils.logging_config import get_logger

from .fastwam import FastWAM

logger = get_logger(__name__)


class CasWAM(FastWAM):
    """FastWAM variant with full-history KV cache and history cross-attention."""

    def __init__(self, *args, max_history_len: int = 256, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_history_len = int(max_history_len)
        # Runtime history cache – populated during ``infer_action``.
        self._history_kv_cache: list[tuple] | None = None
        self._history_step_count: int = 0

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_wan22_pretrained(cls, max_history_len: int = 256, **kwargs):
        # Extract CasWAM-specific kwargs before passing to parent.
        fastwam_pretrained_path = kwargs.pop("fastwam_pretrained_path", None)

        # Inject history cross-attention into ActionDiT config.
        action_dit_config = dict(kwargs.get("action_dit_config", {}) or {})
        action_dit_config["use_history_cross_attention"] = True
        action_dit_config["max_history_len"] = int(max_history_len)

        # Inject history_hidden_dim from VideoDiT config (history KV is 3072-dim)
        video_dit_config = kwargs.get("video_dit_config", {})
        if isinstance(video_dit_config, dict):
            action_dit_config["history_hidden_dim"] = int(
                video_dit_config.get("hidden_dim", action_dit_config.get("hidden_dim", 3072))
            )

        kwargs["action_dit_config"] = action_dit_config
        model = super().from_wan22_pretrained(**kwargs)
        model.max_history_len = int(max_history_len)
        model._history_kv_cache = None
        model._history_step_count = 0

        # Load jointly-trained MoT weights from the FastWAM release checkpoint.
        # This provides ActionDiT self-attn/FFN weights from joint video+action
        # training, which is a better starting point than the linear-interp backbone.
        if fastwam_pretrained_path is not None:
            logger.info("Loading FastWAM pretrained checkpoint: %s", fastwam_pretrained_path)
            model.load_checkpoint(fastwam_pretrained_path)

        # Freeze VideoDiT only.  ActionDiT is fully trainable.
        # Must happen AFTER super().from_wan22_pretrained() and checkpoint load
        # because both construct/overwrite MoT parameters.
        model._freeze_non_history_params()
        return model

    def get_trainable_parameters(self):
        """Return all ActionDiT params + VideoDiT frozen.

        ActionDiT is fully trainable (self-attn, cross-attn, FFN, modulation,
        encoder, head, history_cross_attn).  With 30 layers, hidden_dim=1024,
        history_hidden_dim=3072: ~3B trainable params.
        AdamW optimizer states: ~48 GB (fp32 momentum + variance).
        ZeRO-1 on 4 GPUs: ~12 GB/GPU for optimizer states.
        DDP on 4x80GB: ~48 GB/GPU optimizer + ~16 GB model + gradients,
        leaving ~16 GB for activations.

        VideoDiT (all video expert params) remains frozen.
        """
        params = []
        for name, param in self.mot.named_parameters():
            if 'mixtures.action' in name:
                params.append(param)
        return params

    def _freeze_non_history_params(self):
        """Freeze all MoT parameters except those returned by get_trainable_parameters."""
        trainable_set = set(id(p) for p in self.get_trainable_parameters())
        for param in self.mot.parameters():
            if id(param) not in trainable_set:
                param.requires_grad_(False)

    # ------------------------------------------------------------------
    # History cache management
    # ------------------------------------------------------------------

    def reset_history(self) -> None:
        """Clear the accumulated history KV cache (call at episode start)."""
        self._history_kv_cache = None
        self._history_step_count = 0

    def _append_to_history(self, new_kv: list[tuple]) -> None:
        """Append a single-step video (K, V) pair to the running history cache.

        Args:
            new_kv: Per-layer ``(k, v)`` tuples from ``prefill_video_cache``,
                each tensor shape ``[B, Sv, H*Dh]``.
        """
        if self._history_kv_cache is None:
            self._history_kv_cache = [(k.detach().clone(), v.detach().clone()) for k, v in new_kv]
        else:
            if len(new_kv) != len(self._history_kv_cache):
                raise ValueError(
                    f"History layer count mismatch: expected "
                    f"{len(self._history_kv_cache)}, got {len(new_kv)}"
                )
            for i, (k, v) in enumerate(new_kv):
                ek, ev = self._history_kv_cache[i]
                k_cat = torch.cat([ek, k.detach()], dim=1)
                v_cat = torch.cat([ev, v.detach()], dim=1)
                if k_cat.shape[1] > self.max_history_len:
                    k_cat = k_cat[:, -self.max_history_len:]
                    v_cat = v_cat[:, -self.max_history_len:]
                self._history_kv_cache[i] = (k_cat, v_cat)
        self._history_step_count += 1

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_loss(self, sample, tiled: bool = False):
        """Memory-efficient training loss for CasWAM.

        Uses per-step ``.backward()`` to free each step's computation graph
        immediately, avoiding OOM from accumulating all 8 steps' activations.

        **Strategy** (VideoDiT frozen, ActionDiT fully trainable):

        1. Under ``torch.no_grad()``: run joint video+action MoT forward
           with noised latents to compute video loss (monitoring only).
        2. For each replan step i (under no_grad): video-only prefill with
           ``[real_frame_i, noise_1, ..., noise_8]`` → full 9-frame KV
           as current context.  Slice frame i's KV for history.
        3. **With gradients**: autoregressive loop over ``num_replans`` steps.
           Each step: current = full 9-frame KV (real + noise future),
           history = real frames [0..i-1].
        4. History KV (video-only, detached, real frame only) accumulates.
        5. Returns: action MSE loss (with gradients).
           Video MSE loss (no-grad, monitoring only).
        """
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        # -- noise / targets ------------------------------------------------
        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size, device=self.device, dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        fuse_flag = inputs["fuse_vae_embedding_in_latents"]
        first_frame_latents = inputs["first_frame_latents"]
        if first_frame_latents is None:
            first_frame_latents = input_latents[:, :, 0:1]

        import torch.nn.functional as F

        def _action_loss(pred_a, target_a, pad_mask):
            token_loss = F.mse_loss(pred_a.float(), target_a.float(), reduction="none").mean(dim=2)
            if pad_mask is not None:
                valid = (~pad_mask).to(device=token_loss.device, dtype=token_loss.dtype)
                valid_sum = valid.sum(dim=1).clamp(min=1.0)
                return (token_loss * valid).sum(dim=1) / valid_sum
            return token_loss.mean(dim=1)

        # ==================================================================
        # STEP 1 — Joint MoT forward (no_grad): video loss monitoring
        # ==================================================================
        num_latent_frames = int(input_latents.shape[2])
        tokens_per_frame = None

        with torch.no_grad():
            noise_video = torch.randn_like(input_latents)
            timestep_video = self.train_video_scheduler.sample_training_t(
                batch_size=batch_size, device=self.device, dtype=input_latents.dtype,
            )
            latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
            if inputs["first_frame_latents"] is not None:
                latents[:, :, 0:1] = inputs["first_frame_latents"]

            video_pre_joint = self.video_expert.pre_dit(
                x=latents, timestep=timestep_video,
                context=context, context_mask=context_mask,
                action=action, fuse_vae_embedding_in_latents=fuse_flag,
            )
            action_pre_ng = self.action_expert.pre_dit(
                action_tokens=noisy_action, timestep=timestep_action,
                context=context, context_mask=context_mask,
            )
            attn_mask_joint = self._build_mot_attention_mask(
                video_seq_len=video_pre_joint["tokens"].shape[1],
                action_seq_len=action_pre_ng["tokens"].shape[1],
                video_tokens_per_frame=int(video_pre_joint["meta"]["tokens_per_frame"]),
                device=video_pre_joint["tokens"].device,
            )
            tokens_out_joint = self.mot(
                embeds_all={"video": video_pre_joint["tokens"], "action": action_pre_ng["tokens"]},
                attention_mask=attn_mask_joint,
                freqs_all={"video": video_pre_joint["freqs"], "action": action_pre_ng["freqs"]},
                context_all={
                    "video": {"context": video_pre_joint["context"], "mask": video_pre_joint["context_mask"]},
                    "action": {"context": action_pre_ng["context"], "mask": action_pre_ng["context_mask"]},
                },
                t_mod_all={"video": video_pre_joint["t_mod"], "action": action_pre_ng["t_mod"]},
                return_video_kv_cache=False,
            )
            target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)
            pred_video = self.video_expert.post_dit(tokens_out_joint["video"], video_pre_joint)
            if inputs["first_frame_latents"] is not None:
                pred_video = pred_video[:, :, 1:]
                target_video = target_video[:, :, 1:]
            loss_vid = self._compute_video_loss_per_sample(
                pred_video, target_video, image_is_pad,
                include_initial_video_step=(inputs["first_frame_latents"] is None),
            )
            vw = self.train_video_scheduler.training_weight(timestep_video).to(
                device=loss_vid.device, dtype=loss_vid.dtype)
            loss_video_monitor = float((loss_vid * vw).mean().item())
            del video_pre_joint, action_pre_ng, attn_mask_joint, latents, noise_video
            del timestep_video, pred_video, target_video, loss_vid, vw, tokens_out_joint

            # Pre-compute tokens_per_frame for mask building
            _dummy = input_latents[:, :, 0:1]
            _dummy_pre = self.video_expert.pre_dit(
                x=_dummy, timestep=torch.zeros(batch_size, device=self.device, dtype=input_latents.dtype),
                context=context, context_mask=context_mask,
                action=action, fuse_vae_embedding_in_latents=fuse_flag,
            )
            tokens_per_frame = int(_dummy_pre["meta"]["tokens_per_frame"])
            del _dummy, _dummy_pre

        # ==================================================================
        # STEP 2 — Autoregressive action training loop
        # Each step i: video-only prefill [real_frame_i, noise_1, ..., noise_8]
        # → full 9-frame KV = current context (with predicted future).
        # History accumulates ONLY real frame i's KV (sliced from the 9-frame KV).
        # ==================================================================
        action_horizon = int(action.shape[1])
        num_replans = num_latent_frames - 1
        actions_per_replan = action_horizon // num_replans

        history_kv = None
        loss_values = []
        _per_step_backward = self.training
        # DeepSpeed: raw .backward() interacts badly with DS gradient hooks.
        if _per_step_backward and getattr(self, '_using_deepspeed', False):
            _per_step_backward = False

        for step_i in range(num_replans):
            a_start = step_i * actions_per_replan
            a_end = (step_i + 1) * actions_per_replan
            noisy_chunk = noisy_action[:, a_start:a_end]
            target_chunk = target_action[:, a_start:a_end]
            pad_chunk = action_is_pad[:, a_start:a_end] if action_is_pad is not None else None

            # — Video-only prefill: [real_frame_i, noise_1, ..., noise_8] —
            with torch.no_grad():
                noise_video = torch.randn_like(input_latents)
                timestep_video = self.train_video_scheduler.sample_training_t(
                    batch_size=batch_size, device=self.device, dtype=input_latents.dtype,
                )
                latents_pf = self.train_video_scheduler.add_noise(
                    input_latents, noise_video, timestep_video,
                )
                # Set frame i to real, all others stay noisy
                latents_pf[:, :, step_i:step_i+1] = input_latents[:, :, step_i:step_i+1]

                video_pre = self.video_expert.pre_dit(
                    x=latents_pf, timestep=timestep_video,
                    context=context, context_mask=context_mask,
                    action=action, fuse_vae_embedding_in_latents=fuse_flag,
                )
                video_svl = int(video_pre["tokens"].shape[1])
                joint_mask_tmp = self._build_mot_attention_mask(
                    video_seq_len=video_svl,
                    action_seq_len=actions_per_replan,
                    video_tokens_per_frame=tokens_per_frame,
                    device=video_pre["tokens"].device,
                )
                video_attn_mask = joint_mask_tmp[:video_svl, :video_svl]
                full_video_kv = self.mot.prefill_video_cache(
                    video_tokens=video_pre["tokens"],
                    video_freqs=video_pre["freqs"],
                    video_t_mod=video_pre["t_mod"],
                    video_context_payload={
                        "context": video_pre["context"],
                        "mask": video_pre["context_mask"],
                    },
                    video_attention_mask=video_attn_mask,
                )
                # Slice frame i's KV for history (real frame only)
                start_tok = step_i * tokens_per_frame
                end_tok = start_tok + tokens_per_frame
                frame_i_kv = [
                    (kv["k"][:, start_tok:end_tok].detach().clone(),
                     kv["v"][:, start_tok:end_tok].detach().clone())
                    for kv in full_video_kv
                ]
                del latents_pf, noise_video, video_pre, joint_mask_tmp, video_attn_mask

            # — Attention mask (9 video frames + action chunk) —
            attn_mask = self._build_mot_attention_mask(
                video_seq_len=video_svl,
                action_seq_len=actions_per_replan,
                video_tokens_per_frame=tokens_per_frame,
                device=full_video_kv[0]["k"].device,
            )

            # ── WITH GRADIENTS (every step) ──────────────────────────
            action_pre = self.action_expert.pre_dit(
                action_tokens=noisy_chunk, timestep=timestep_action,
                context=context, context_mask=context_mask,
            )
            act_out = self.mot.forward_action_with_history_cache(
                action_tokens=action_pre["tokens"],
                action_freqs=action_pre["freqs"],
                action_t_mod=action_pre["t_mod"],
                action_context_payload={
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
                video_kv_cache=full_video_kv,  # full 9-frame KV as current context
                attention_mask=attn_mask,
                video_seq_len=video_svl,
                history_kv_cache=history_kv,
                return_action_kv=True,
            )
            act_tokens, act_kv_grad = act_out
            pred_a = self.action_expert.post_dit(act_tokens, action_pre)
            aw = self.train_action_scheduler.training_weight(timestep_action).to(
                device=pred_a.device, dtype=pred_a.dtype)
            li = (_action_loss(pred_a, target_chunk, pad_chunk) * aw).mean()

            # Per-step backward: free graph immediately to avoid OOM
            if _per_step_backward:
                (li / num_replans).backward()
                loss_values.append(float(li.detach().item()))
                del action_pre, act_out, act_tokens, act_kv_grad, pred_a, aw, li, attn_mask
            else:
                loss_values.append(li)

            # — Append frame i's video KV to history (real frame only) —
            with torch.no_grad():
                if history_kv is None:
                    history_kv = [(k, v) for k, v in frame_i_kv]
                else:
                    new_hist = []
                    for (ek, ev), (nk, nv) in zip(history_kv, frame_i_kv):
                        ck = torch.cat([ek, nk], dim=1)
                        cv = torch.cat([ev, nv], dim=1)
                        if ck.shape[1] > self.max_history_len:
                            ck = ck[:, -self.max_history_len:]
                            cv = cv[:, -self.max_history_len:]
                        new_hist.append((ck, cv))
                    history_kv = new_hist

        if _per_step_backward:
            loss_dict = {
                "loss_action": sum(loss_values) / max(len(loss_values), 1),
                "loss_video": loss_video_monitor,
            }
            return torch.tensor(0.0, device=self.device), loss_dict

        loss_action = sum(loss_values) / max(len(loss_values), 1)
        loss_total = self.loss_lambda_action * loss_action
        loss_dict = {
            "loss_action": float(loss_action.detach().item()),
            "loss_video": loss_video_monitor,
        }
        return loss_total, loss_dict

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
    ) -> dict[str, Any]:
        """Action inference with full-history KV accumulation.

        Call ``reset_history()`` at the beginning of each episode.  Each
        subsequent call to ``infer_action`` appends the current observation's
        video KV to the running history cache.
        """
        self.eval()
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action` requires `video_attention_mask_mode='first_frame_causal'`."
            )

        # -- validate / prepare input image ---------------------------------
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be multiples of 16, got HxW=({height},{width})"
            )

        # -- proprio --------------------------------------------------------
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None`.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}.")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        # -- action noise ---------------------------------------------------
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator, device=rand_device, dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        # -- encode current image & build video KV cache --------------------
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        # -- text context ---------------------------------------------------
        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")
        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context, context_mask=context_mask, proprio=proprio,
            )

        # -- VideoDiT prefill: [real_f0, noise_1, ..., noise_8] -----------
        # Matches training: 9-frame prefill with noise-based "predicted future".
        # Only frame 0's KV goes into history; full 9-frame KV is current context.
        num_latent_frames = 9  # matches VAE temporal factor (33 px → 9 latent)
        B = first_frame_latents.shape[0]
        latents_9 = torch.randn(
            B, *first_frame_latents.shape[1:2], num_latent_frames,
            *first_frame_latents.shape[3:],
            device=self.device, dtype=first_frame_latents.dtype,
        )
        latents_9[:, :, 0:1] = first_frame_latents  # frame 0 = real observation
        timestep_video = torch.full(
            (B,), self.train_video_scheduler.num_train_timesteps - 1,
            device=self.device, dtype=first_frame_latents.dtype,
        )  # max noise for future frames, matching training semantics

        video_pre = self.video_expert.pre_dit(
            x=latents_9,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=tokens_per_frame,
            device=video_pre["tokens"].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"], "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )

        # -- action denoising with history ----------------------------------
        # NOTE: Do NOT append current frame to history before denoising!
        # The current frame is already in video_kv_cache; appending it to
        # history would cause it to appear TWICE in the attention.
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device, dtype=latents_action.dtype, shift_override=sigma_shift,
        )
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(
                dtype=latents_action.dtype, device=self.device,
            )
            pred_action = self._predict_action_noise_with_history(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
                history_kv_cache=self._history_kv_cache,
            )
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        # -- accumulate history AFTER denoising (only frame 0's KV) ---------
        # Slice frame 0's KV from the 9-frame cache for history.
        f0_start = 0
        f0_end = tokens_per_frame
        frame0_kv = [
            (layer_cache["k"][:, f0_start:f0_end],
             layer_cache["v"][:, f0_start:f0_end])
            for layer_cache in video_kv_cache
        ]
        self._append_to_history(frame0_kv)

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }

    @torch.no_grad()
    def _predict_action_noise_with_history(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        history_kv_cache: Optional[list[torch.Tensor]] = None,
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self.mot.forward_action_with_history_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"], "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
            history_kv_cache=history_kv_cache,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    # ------------------------------------------------------------------
    # Trainer-compatible inference (with history)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ):
        """Trainer-compatible inference with history-aware action prediction.

        CasWAM focuses on action prediction, not video generation.
        This method:
        1. Uses ``infer_action()`` for history-aware action prediction.
        2. Returns a placeholder video (first frame repeated) for trainer
           compatibility.

        The trainer will compute video metrics (PSNR, SSIM) on the placeholder,
        which are not meaningful for CasWAM. The action metrics (L1, L2)
        are the relevant evaluation signals.
        """
        # Use history-aware action prediction
        action_out = self.infer_action(
            prompt=prompt,
            input_image=input_image,
            action_horizon=action_horizon or action.shape[1] if action is not None else 16,
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
        )["action"]

        # Generate placeholder video in pixel space (bypass VAE temporal upsampling).
        # The trainer expects a list of PIL frames matching GT video shape.
        # Video metrics on this placeholder are not meaningful for CasWAM.
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        img = input_image.to(device="cpu", dtype=torch.float32).clamp(-1, 1)
        img = ((img + 1.0) * 127.5).to(torch.uint8)[0]  # [3, H, W]
        pil_frame = Image.fromarray(img.permute(1, 2, 0).numpy())
        placeholder_video = [pil_frame] * num_frames

        return {
            "video": placeholder_video,
            "action": action_out,
        }

    # ------------------------------------------------------------------
    # Multi-step history evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def infer_action_sequence(
        self,
        video: torch.Tensor,
        action_chunk_size: int,
        num_obs_frames: int,
        action_video_freq_ratio: int = 1,
        prompt: Optional[str] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        tiled: bool = False,
    ) -> torch.Tensor:
        """Predict actions frame-by-frame with accumulating history.

        Simulates autoregressive evaluation: at each observation frame,
        the model encodes the frame, appends its video KV to history,
        then predicts the next ``action_chunk_size`` actions.

        Args:
            video: [C, T_pixel, H, W] pixel-space video tensor.
            action_chunk_size: Number of actions to predict per frame.
            num_obs_frames: Number of observation frames to iterate over.
            action_video_freq_ratio: Pixel-frame stride between obs frames.
            prompt: Text prompt (used if context is None).
            context: Pre-encoded text context [L, D].
            context_mask: Context mask [L].
            proprio: [T_proprio, d] proprio sequence (one per action step).
            num_inference_steps: Denoising steps per chunk.
            sigma_shift: Optional sigma shift override.
            seed: Random seed for reproducibility.
            tiled: Use tiled VAE encoding.

        Returns:
            Predicted actions [num_obs_frames * action_chunk_size, D].
        """
        self.reset_history()
        all_actions = []

        for i in range(num_obs_frames):
            # Extract the i-th observation frame from the pixel video.
            pixel_idx = i * action_video_freq_ratio
            frame_i = video[:, pixel_idx:pixel_idx + 1, :, :]  # [C, 1, H, W]
            input_image_i = frame_i.permute(1, 0, 2, 3)  # [1, C, H, W]

            # Select proprio for this frame.
            # Proprio is [T_proprio, d] aligned with action steps.
            # Observation frame i corresponds to action index i * action_chunk_size.
            proprio_i = None
            if proprio is not None:
                proprio_idx = min(i * action_chunk_size, proprio.shape[0] - 1)
                proprio_i = proprio[proprio_idx]  # [d]

            result = self.infer_action(
                prompt=prompt,
                input_image=input_image_i,
                action_horizon=action_chunk_size,
                proprio=proprio_i,
                context=context,
                context_mask=context_mask,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                tiled=tiled,
            )
            all_actions.append(result["action"])  # [action_chunk_size, D]

        return torch.cat(all_actions, dim=0)  # [num_obs_frames * action_chunk_size, D]

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
            "max_history_len": self.max_history_len,
            "model_class": "CasWAM",
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        if "mot" in payload:
            missing, unexpected = self.mot.load_state_dict(payload["mot"], strict=False)
            if missing:
                logger.info("CasWAM load_checkpoint – missing MoT keys: %d (likely history_cross_attn)", len(missing))
            if unexpected:
                logger.warning("CasWAM load_checkpoint – unexpected MoT keys: %d", len(unexpected))
        elif "dit" in payload:
            logger.warning("Loading legacy `dit` checkpoint into video expert only.")
            self.video_expert.load_state_dict(payload["dit"], strict=False)
        else:
            raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {path}")

        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
            else:
                logger.warning("Checkpoint has no `proprio_encoder` weights.")
        elif "proprio_encoder" in payload:
            logger.warning("Checkpoint has `proprio_encoder` but model has proprio_dim=None.")

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload
