"""CasWAM: Speed-centric Full-history Memory Grounded World Action Model.

Adds full-history KV cache with temporal positional encoding and history
cross-attention to the ActionDiT, enabling the action expert to attend to
all past video observations for improved long-horizon robotic control.

VideoDiT remains fully frozen.  ActionDiT is partially fine-tuned:
the new ``history_cross_attn`` layers, per-block AdaLN modulation
(all layers), plus the action encoder/head are trained by default.
ActionDiT FFN can optionally be unfrozen via ``_n_unfrozen_ffn_layers``
(0 = safe, modulation-only; 6-12 = more capacity, needs GPU headroom).
ActionDiT self-attention and text cross-attention stay frozen.
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
        n_unfrozen_ffn_layers = kwargs.pop("n_unfrozen_ffn_layers", 0)

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

        # Freeze everything except history_cross_attn + action encoder/head.
        # Must happen AFTER super().from_wan22_pretrained() and checkpoint load
        # because both construct/overwrite MoT parameters.
        model._n_unfrozen_ffn_layers = int(n_unfrozen_ffn_layers)
        model._freeze_non_history_params()
        return model

    def get_trainable_parameters(self):
        """Return history_cross_attn + action encoder/head + selected ActionDiT params.

        The FFN and modulation in the last ``n_unfrozen_ffn_layers`` ActionDiT
        blocks are unfrozen because the history_cross_attn output flows through
        them at every block.  Keeping all FFNs frozen prevents learning;
        unfreezing all 30 exceeds 80 GB Adam state.

        The last layers are closest to the output head and most impactful for
        transforming history-enriched representations.

        VideoDiT (self-attn, FFN, modulation) and ActionDiT self-attn /
        text cross-attn remain frozen.
        """
        n_unfrozen = getattr(self, '_n_unfrozen_ffn_layers', 0)  # 0 = modulation only, safe default
        params = []
        for name, param in self.mot.named_parameters():
            if 'mixtures.action' not in name:
                continue
            if ('history_cross_attn' in name
                    or name.startswith('mixtures.action.action_encoder')
                    or name.startswith('mixtures.action.head')):
                params.append(param)
                continue
            # Unfreeze modulation for ALL action blocks (negligible ~180K params)
            if '.modulation' in name and '.blocks.' in name:
                params.append(param)
                continue
            # Unfreeze FFN only for the last N blocks (memory-constrained)
            if '.ffn.' in name and '.blocks.' in name:
                try:
                    layer_idx = int(name.split('.blocks.')[1].split('.')[0])
                except (ValueError, IndexError):
                    continue
                if layer_idx >= 30 - n_unfrozen:  # e.g. layers 18-29 for n=12
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

        **Strategy** (VideoDiT frozen, only history_cross_attn trained):

        1. Under ``torch.no_grad()``: run standard MoT forward to collect
           per-layer video K tensors (used as history signal).
        2. Under ``torch.no_grad()``: run video cache prefill on the first
           frame to populate ``video_kv_cache`` for the current step.
        3. **With gradients**: run ``forward_action_with_history_cache``
           so that history_cross_attn layers receive training signal.
        4. Loss = history-augmented action MSE loss (trained).
           Video MSE loss is also computed (no-grad, monitoring only).

        This keeps peak memory close to a single MoT forward with gradient
        tracking, plus two no-grad passes that only store forward activations
        (no backward graph).
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

        # ==================================================================
        # STEP 1 — Collect history video KV (no gradients)
        # ==================================================================
        with torch.no_grad():
            noise_video = torch.randn_like(input_latents)
            timestep_video = self.train_video_scheduler.sample_training_t(
                batch_size=batch_size, device=self.device, dtype=input_latents.dtype,
            )
            latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
            if inputs["first_frame_latents"] is not None:
                latents[:, :, 0:1] = inputs["first_frame_latents"]

            video_pre = self.video_expert.pre_dit(
                x=latents,
                timestep=timestep_video,
                context=context,
                context_mask=context_mask,
                action=action,
                fuse_vae_embedding_in_latents=fuse_flag,
            )
            action_pre_no_grad = self.action_expert.pre_dit(
                action_tokens=noisy_action,
                timestep=timestep_action,
                context=context,
                context_mask=context_mask,
            )
            attention_mask = self._build_mot_attention_mask(
                video_seq_len=video_pre["tokens"].shape[1],
                action_seq_len=action_pre_no_grad["tokens"].shape[1],
                video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                device=video_pre["tokens"].device,
            )
            tokens_out, video_kv_per_layer = self.mot(
                embeds_all={"video": video_pre["tokens"], "action": action_pre_no_grad["tokens"]},
                attention_mask=attention_mask,
                freqs_all={"video": video_pre["freqs"], "action": action_pre_no_grad["freqs"]},
                context_all={
                    "video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
                    "action": {"context": action_pre_no_grad["context"], "mask": action_pre_no_grad["context_mask"]},
                },
                t_mod_all={"video": video_pre["t_mod"], "action": action_pre_no_grad["t_mod"]},
                return_video_kv_cache=True,
            )
            # Detach and trim history KV to max_history_len tokens.
            # ``video_kv_per_layer`` is a list of ``(k, v)`` tuples, one per
            # layer.  During training, the full multi-frame sequence may
            # exceed ``max_history_len``; we keep only the most recent tokens.
            history_kv_list = []
            for k, v in video_kv_per_layer:
                k_det, v_det = k.detach(), v.detach()
                if k_det.shape[1] > self.max_history_len:
                    k_det = k_det[:, -self.max_history_len:]
                    v_det = v_det[:, -self.max_history_len:]
                history_kv_list.append((k_det, v_det))

            # ---- video loss monitoring (no-grad, VideoDiT frozen) -------
            target_video = self.train_video_scheduler.training_target(
                input_latents, noise_video, timestep_video,
            )
            pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)

            include_initial_video_step = inputs["first_frame_latents"] is None
            if inputs["first_frame_latents"] is not None:
                pred_video = pred_video[:, :, 1:]
                target_video = target_video[:, :, 1:]

            loss_video_per_sample = self._compute_video_loss_per_sample(
                pred_video=pred_video,
                target_video=target_video,
                image_is_pad=image_is_pad,
                include_initial_video_step=include_initial_video_step,
            )
            video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
                device=loss_video_per_sample.device, dtype=loss_video_per_sample.dtype,
            )
            loss_video_monitor = float((loss_video_per_sample * video_weight).mean().item())

            del video_pre, action_pre_no_grad, attention_mask, latents
            del noise_video, timestep_video, video_kv_per_layer, tokens_out
            del pred_video, target_video, loss_video_per_sample, video_weight

        # ==================================================================
        # STEP 2 — Video cache prefill (no gradients, first frame only)
        # ==================================================================
        with torch.no_grad():
            timestep_video_zero = torch.zeros(
                batch_size, dtype=first_frame_latents.dtype, device=self.device,
            )
            video_pre_cache = self.video_expert.pre_dit(
                x=first_frame_latents,
                timestep=timestep_video_zero,
                context=context,
                context_mask=context_mask,
                action=None,
                fuse_vae_embedding_in_latents=fuse_flag,
            )
            cache_video_seq_len = int(video_pre_cache["tokens"].shape[1])

        # Action expert pre_dit (WITH gradients — feeds into history forward)
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        cache_attn_mask = self._build_mot_attention_mask(
            video_seq_len=cache_video_seq_len,
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre_cache["meta"]["tokens_per_frame"]),
            device=video_pre_cache["tokens"].device,
        )

        with torch.no_grad():
            video_kv_cache = self.mot.prefill_video_cache(
                video_tokens=video_pre_cache["tokens"],
                video_freqs=video_pre_cache["freqs"],
                video_t_mod=video_pre_cache["t_mod"],
                video_context_payload={
                    "context": video_pre_cache["context"],
                    "mask": video_pre_cache["context_mask"],
                },
                video_attention_mask=cache_attn_mask[:cache_video_seq_len, :cache_video_seq_len],
            )
            del video_pre_cache

        # ==================================================================
        # STEP 3 — Action forward with history cache (WITH gradients)
        # ==================================================================
        action_tokens_hist = self.mot.forward_action_with_history_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=cache_attn_mask,
            video_seq_len=cache_video_seq_len,
            history_kv_cache=history_kv_list,
        )
        pred_action_history = self.action_expert.post_dit(action_tokens_hist, action_pre)

        # ==================================================================
        # STEP 4 — Loss (action only — VideoDiT is frozen)
        # ==================================================================
        import torch.nn.functional as F

        def _action_loss(pred_a, target_a, pad_mask):
            token_loss = F.mse_loss(pred_a.float(), target_a.float(), reduction="none").mean(dim=2)
            if pad_mask is not None:
                valid = (~pad_mask).to(device=token_loss.device, dtype=token_loss.dtype)
                valid_sum = valid.sum(dim=1).clamp(min=1.0)
                return (token_loss * valid).sum(dim=1) / valid_sum
            return token_loss.mean(dim=1)

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            device=pred_action_history.device, dtype=pred_action_history.dtype,
        )
        loss_action = (_action_loss(pred_action_history, target_action, action_is_pad) * action_weight).mean()

        loss_total = self.loss_lambda_action * loss_action
        loss_dict = {
            "loss_action": float(loss_action.detach().item()),
            "loss_video": loss_video_monitor,
        }
        return loss_total, loss_dict

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

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

        # -- VideoDiT prefill (current frame only) --------------------------
        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],), dtype=first_frame_latents.dtype, device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
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

        # -- accumulate history ---------------------------------------------
        # Extract per-layer (k, v) tensors from the prefill cache.
        new_kv = [(layer_cache["k"], layer_cache["v"]) for layer_cache in video_kv_cache]
        self._append_to_history(new_kv)

        # -- action denoising with history ----------------------------------
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
