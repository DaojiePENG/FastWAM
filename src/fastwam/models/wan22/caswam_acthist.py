"""CasWAM-ActHist: Full-history KV cache with action history.

Extends CasWAM by also accumulating action K/V in the history cache.
This enables true autoregressive action-to-action conditioning: the
ActionDiT can attend to both past observations AND past actions when
predicting future actions (similar to an LLM attending to all previous
tokens).

Kept as a separate class from CasWAM for clean ablation comparison:
  - CasWAM:       video-only history (observation memory)
  - CasWAMActHist: video + action history (full autoregressive memory)
"""

from typing import Any, Optional

import torch

from fastwam.utils.logging_config import get_logger

from .caswam import CasWAM

logger = get_logger(__name__)


class CasWAMActHist(CasWAM):
    """CasWAM with action tokens stored in the history KV cache."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_wan22_pretrained(cls, max_history_len: int = 256, **kwargs):
        # Same construction as CasWAM — injects history_cross_attn,
        # loads FastWAM release checkpoint, freezes non-history params.
        # The only difference is which class is instantiated.
        return super().from_wan22_pretrained(
            max_history_len=int(max_history_len), **kwargs,
        )

    # ------------------------------------------------------------------
    # History cache management (joint video + action)
    # ------------------------------------------------------------------

    def _append_to_history(
        self,
        new_video_kv: list[tuple],
        new_action_kv: Optional[list[tuple]] = None,
    ) -> None:
        """Append joint (video + action) KV to the running history cache.

        Video and action K/V are concatenated per layer, with video tokens
        placed before action tokens at each step.  The temporal positional
        encoding assigns positions in concatenation order, so the model
        sees::

            [v1, a1, v2, a2, ..., vn, an]

        Args:
            new_video_kv: Per-layer ``(k, v)`` tuples from
                ``prefill_video_cache`` (current step).
            new_action_kv: Per-layer ``(k, v)`` tuples from the action
                forward (denoised action tokens, current step).  When
                ``None`` (e.g. first step), only video KV is stored.
        """
        if new_action_kv is not None and len(new_video_kv) != len(new_action_kv):
            raise ValueError(
                f"Layer count mismatch: video={len(new_video_kv)} vs action={len(new_action_kv)}"
            )

        # Build joint per-layer (k, v) for this step.
        if new_action_kv is not None:
            joint_kv = [
                (torch.cat([vk, ak], dim=1), torch.cat([vv, av], dim=1))
                for (vk, vv), (ak, av) in zip(new_video_kv, new_action_kv)
            ]
        else:
            # First step: no action history yet, use video-only.
            joint_kv = [(k.detach().clone(), v.detach().clone()) for k, v in new_video_kv]

        if self._history_kv_cache is None:
            self._history_kv_cache = [
                (k.detach().clone(), v.detach().clone()) for k, v in joint_kv
            ]
        else:
            if len(joint_kv) != len(self._history_kv_cache):
                raise ValueError(
                    f"History layer count mismatch: expected "
                    f"{len(self._history_kv_cache)}, got {len(joint_kv)}"
                )
            for i, (k_new, v_new) in enumerate(joint_kv):
                ek, ev = self._history_kv_cache[i]
                k_cat = torch.cat([ek, k_new.detach()], dim=1)
                v_cat = torch.cat([ev, v_new.detach()], dim=1)
                if k_cat.shape[1] > self.max_history_len:
                    k_cat = k_cat[:, -self.max_history_len:]
                    v_cat = v_cat[:, -self.max_history_len:]
                self._history_kv_cache[i] = (k_cat, v_cat)

        self._history_step_count += 1

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_loss(self, sample, tiled: bool = False):
        """Training loss with joint (video + action) history.

        Same three-step strategy as CasWAM, but Step 1 also collects
        action K/V from the MoT forward and creates a joint history.
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
        # STEP 1 — Collect joint (video + action) history KV (no gradients)
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
                action_tokens=action,  # clean GT actions → clean action KV for history
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
            tokens_out, video_kv_per_layer, action_kv_per_layer = self.mot(
                embeds_all={"video": video_pre["tokens"], "action": action_pre_no_grad["tokens"]},
                attention_mask=attention_mask,
                freqs_all={"video": video_pre["freqs"], "action": action_pre_no_grad["freqs"]},
                context_all={
                    "video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
                    "action": {"context": action_pre_no_grad["context"], "mask": action_pre_no_grad["context_mask"]},
                },
                t_mod_all={"video": video_pre["t_mod"], "action": action_pre_no_grad["t_mod"]},
                return_video_kv_cache=True,
                return_action_kv_cache=True,
            )
            # Build joint history: concatenate video + action KV per layer.
            history_kv_list = []
            for (vk, vv), (ak, av) in zip(video_kv_per_layer, action_kv_per_layer):
                vk_det, vv_det = vk.detach(), vv.detach()
                ak_det, av_det = ak.detach(), av.detach()
                joint_k = torch.cat([vk_det, ak_det], dim=1)
                joint_v = torch.cat([vv_det, av_det], dim=1)
                if joint_k.shape[1] > self.max_history_len:
                    joint_k = joint_k[:, -self.max_history_len:]
                    joint_v = joint_v[:, -self.max_history_len:]
                history_kv_list.append((joint_k, joint_v))

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
            del noise_video, timestep_video, video_kv_per_layer, action_kv_per_layer, tokens_out
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

        # Action expert pre_dit (WITH gradients)
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
        # STEP 3 — Action forward with joint history cache (WITH gradients)
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
        # STEP 4 — Loss
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
        """Action inference with joint (video + action) history accumulation.

        Same as CasWAM.infer_action, but after each step's denoising
        loop, the final action tokens' K/V are collected and appended to
        the history alongside the video KV.
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

        # -- action denoising with history ----------------------------------
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device, dtype=latents_action.dtype, shift_override=sigma_shift,
        )
        num_steps = len(infer_timesteps_action)
        action_kv = None
        for step_idx, (step_t_action, step_delta_action) in enumerate(
            zip(infer_timesteps_action, infer_deltas_action)
        ):
            timestep_action = step_t_action.unsqueeze(0).to(
                dtype=latents_action.dtype, device=self.device,
            )
            # On the last denoising step, collect action K/V for history.
            is_last_step = (step_idx == num_steps - 1)
            result = self._predict_action_noise_with_history(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
                history_kv_cache=self._history_kv_cache,
                return_action_kv=is_last_step,
            )
            if is_last_step:
                pred_action, action_kv = result
            else:
                pred_action = result
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        # -- accumulate joint (video + action) history ----------------------
        new_video_kv = [(layer_cache["k"], layer_cache["v"]) for layer_cache in video_kv_cache]
        self._append_to_history(new_video_kv=new_video_kv, new_action_kv=action_kv)

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
        return_action_kv: bool = False,
    ):
        """Predict action noise, optionally returning action K/V for history."""
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        result = self.mot.forward_action_with_history_cache(
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
            return_action_kv=return_action_kv,
        )
        if return_action_kv:
            action_tokens, action_kv = result
            pred = self.action_expert.post_dit(action_tokens, action_pre)
            return pred, action_kv
        return self.action_expert.post_dit(result, action_pre)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
            "max_history_len": self.max_history_len,
            "model_class": "CasWAMActHist",
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
                logger.info(
                    "CasWAMActHist load_checkpoint – missing MoT keys: %d (likely history_cross_attn)",
                    len(missing),
                )
            if unexpected:
                logger.warning(
                    "CasWAMActHist load_checkpoint – unexpected MoT keys: %d",
                    len(unexpected),
                )
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
