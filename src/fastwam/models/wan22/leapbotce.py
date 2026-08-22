from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fastwam import FastWAM


@dataclass
class CloudPlanningCache:
    video_cache_k: list[torch.Tensor]
    video_cache_v: list[torch.Tensor]
    context: torch.Tensor
    context_mask: torch.Tensor
    video_seq_len: int
    video_tokens_per_frame: int


class EdgeVisionEncoder(nn.Module):
    """Frozen per-view SigLIP encoder with a trainable context projector."""

    def __init__(self, text_dim: int, num_views: int = 2,
                 model_name: str = "vit_base_patch16_siglip_224",
                 pretrained: bool = True, freeze: bool = True,
                 encoder: Optional[nn.Module] = None, embed_dim: Optional[int] = None):
        super().__init__()
        if encoder is None:
            try:
                import timm
            except ImportError as exc:
                raise ImportError("LeapBotCE requires `timm` for the edge SigLIP encoder.") from exc
            encoder = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.encoder = encoder
        self.num_views = int(num_views)
        self.embed_dim = int(embed_dim or getattr(encoder, "embed_dim"))
        self.projector = nn.Sequential(
            nn.Linear(self.num_views * self.embed_dim, int(text_dim)),
            nn.GELU(), nn.LayerNorm(int(text_dim)),
        )
        self.register_buffer("image_mean", torch.tensor([0.5] * 3).view(1, 1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.5] * 3).view(1, 1, 3, 1, 1))
        self.freeze = bool(freeze)
        if self.freeze:
            self.encoder.requires_grad_(False)
            self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.encoder.eval()
        return self

    def forward(self, views: torch.Tensor) -> torch.Tensor:
        if views.ndim != 5 or views.shape[1] != self.num_views or views.shape[2] != 3:
            raise ValueError(f"Expected [B,{self.num_views},3,H,W], got {tuple(views.shape)}")
        views = views.to(device=self.image_mean.device)
        if views.min().detach() < 0:
            views = (views + 1.0) * 0.5
        batch_size = views.shape[0]
        views = F.interpolate(views.flatten(0, 1).float(), size=(224, 224), mode="bilinear", align_corners=False)
        views = views.unflatten(0, (batch_size, self.num_views))
        views = (views - self.image_mean) / self.image_std
        encoder_dtype = next(self.encoder.parameters()).dtype
        flat = views.flatten(0, 1).to(dtype=encoder_dtype)
        if self.freeze:
            with torch.no_grad():
                features = self.encoder(flat)
        else:
            features = self.encoder(flat)
        if features.ndim == 3:
            features = features.mean(dim=1)
        features = features.reshape(batch_size, self.num_views * self.embed_dim)
        return self.projector(features.to(dtype=self.projector[0].weight.dtype)).unsqueeze(1)


def stale_loss_weight(step: int, warmup_steps: int, maximum: float) -> float:
    if not 0.0 <= maximum <= 1.0:
        raise ValueError(f"stale loss maximum must be in [0,1], got {maximum}")
    if warmup_steps <= 0:
        return float(maximum)
    return float(maximum) * min(max(int(step), 0) / float(warmup_steps), 1.0)


class LeapBotCE(FastWAM):
    """Cloud-edge WAM with stale cloud planning and fresh edge vision."""

    def __init__(self, *args, edge_vision_model_name="vit_base_patch16_siglip_224",
                 edge_num_views=2, edge_vision_pretrained=True, edge_vision_freeze=True,
                 stale_loss_lambda_max=0.5, stale_loss_warmup_steps=1,
                 edge_vision_encoder=None, edge_vision_embed_dim=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.edge_vision = EdgeVisionEncoder(
            self.text_dim, edge_num_views, edge_vision_model_name,
            edge_vision_pretrained, edge_vision_freeze,
            edge_vision_encoder, edge_vision_embed_dim,
        ).to(device=self.device, dtype=self.torch_dtype)
        self.edge_num_views = int(edge_num_views)
        self.stale_loss_lambda_max = float(stale_loss_lambda_max)
        self.stale_loss_warmup_steps = int(stale_loss_warmup_steps)
        self.training_step = 0

    def set_training_step(self, step: int, total_steps: Optional[int] = None):
        self.training_step = int(step)
        if self.stale_loss_warmup_steps < 0:
            if total_steps is None:
                raise ValueError("Auto stale-loss warmup requires total_steps.")
            self.stale_loss_warmup_steps = max(int(total_steps) // 2, 1)

    def _prepare_context(self, sample):
        """Build the static language context shared by cloud and edge paths."""
        context, context_mask = sample.get("context"), sample.get("context_mask")
        if context is None or context_mask is None:
            context, context_mask = self.encode_prompt(sample["prompt"])
        context = context.to(self.device, self.torch_dtype)
        context_mask = context_mask.to(self.device, torch.bool)
        return context, context_mask

    def _with_proprio(self, context, context_mask, proprio, source):
        if getattr(self, "proprio_encoder", None) is not None:
            if proprio is None:
                raise ValueError(f"LeapBotCE requires {source} proprio.")
            if proprio.ndim == 3:
                if proprio.shape[1] != 1:
                    raise ValueError(
                        f"LeapBotCE expects one {source} proprio state, got {tuple(proprio.shape)}"
                    )
                proprio = proprio[:, 0]
            context, context_mask = self._append_proprio_to_context(
                context, context_mask, proprio.to(self.device, self.torch_dtype))
        return context, context_mask

    def _edge_context(self, context, context_mask, current_views):
        token = self.edge_vision(current_views.to(self.device, self.torch_dtype)).to(context.dtype)
        mask = torch.ones((context.shape[0], 1), device=context.device, dtype=torch.bool)
        return torch.cat([context, token], 1), torch.cat([context_mask, mask], 1)

    def _encode_cloud_tensor(self, input_image, cloud_context, cloud_context_mask,
                             edge_context=None, edge_context_mask=None):
        if input_image.ndim != 4:
            raise ValueError(f"cloud image must be [B,3,H,W], got {tuple(input_image.shape)}")
        latents = self._encode_video_latents(input_image.unsqueeze(2).to(self.device, self.torch_dtype))
        timestep = torch.zeros((latents.shape[0],), device=self.device, dtype=latents.dtype)
        fuse = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))
        values = self.video_expert.prepare(
            latents, timestep, cloud_context, cloud_context_mask, None, fuse
        )
        video_tokens, _, video_t_mod, video_context, video_context_mask, video_freqs, _, _, _, tokens_per_frame = values
        video_mask = self.video_expert.build_video_to_video_mask(
            video_tokens.shape[1], tokens_per_frame, video_tokens.device)
        cache_k, cache_v = self.mot.prefill_video_cache_tensor(
            video_tokens, video_freqs, video_t_mod, video_context, video_context_mask, video_mask)
        if edge_context is None:
            edge_context, edge_context_mask = cloud_context, cloud_context_mask
        return CloudPlanningCache(cache_k, cache_v, edge_context, edge_context_mask,
                                  int(video_tokens.shape[1]), int(tokens_per_frame))

    @torch.no_grad()
    def encode_cloud(self, input_image, prompt=None, context=None, context_mask=None,
                     cloud_proprio=None, proprio=None):
        """Encode one temporally consistent cloud observation into a MoT cache.

        ``context`` is language-only. ``cloud_proprio`` must belong to the same
        time step as ``input_image``; ``proprio`` is a backwards-compatible alias.
        The cache retains the language context only so edge inference can append
        its current state without reusing the stale cloud state.
        """
        if cloud_proprio is not None and proprio is not None:
            raise ValueError("Provide either cloud_proprio or proprio, not both.")
        cloud_proprio = cloud_proprio if cloud_proprio is not None else proprio
        if context is None or context_mask is None:
            if prompt is None:
                raise ValueError("Provide prompt or context/context_mask to encode_cloud.")
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context.ndim == 2:
                context, context_mask = context.unsqueeze(0), context_mask.unsqueeze(0)
            context = context.to(self.device, self.torch_dtype)
            context_mask = context_mask.to(self.device, torch.bool)
        if cloud_proprio is not None and cloud_proprio.ndim == 1:
            cloud_proprio = cloud_proprio.unsqueeze(0)
        cloud_context, cloud_context_mask = self._with_proprio(
            context, context_mask, cloud_proprio, "cloud"
        )
        return self._encode_cloud_tensor(
            input_image.to(self.device, self.torch_dtype),
            cloud_context, cloud_context_mask, context, context_mask
        )

    def _predict_from_cache(self, cache, noisy_action, timestep, edge_context, edge_mask):
        action_mask = torch.ones((noisy_action.shape[1], cache.video_seq_len + noisy_action.shape[1]),
                                 device=noisy_action.device, dtype=torch.bool)
        return self._denoise_action_with_video_cache(
            noisy_action, timestep, edge_context, edge_mask,
            cache.video_cache_k, cache.video_cache_v, action_mask)

    def training_loss(self, sample, tiled=False):
        del tiled
        action = sample["action"].to(self.device, self.torch_dtype)
        context, context_mask = self._prepare_context(sample)
        edge_proprio = sample.get("edge_current_proprio", sample.get("proprio"))
        edge_context, edge_mask = self._with_proprio(
            context, context_mask, edge_proprio, "edge"
        )
        edge_context, edge_mask = self._edge_context(
            edge_context, edge_mask, sample["edge_current_views"]
        )
        noise = torch.randn_like(action)
        timestep = self.train_action_scheduler.sample_training_t(action.shape[0], self.device, action.dtype)
        noisy = self.train_action_scheduler.add_noise(action, noise, timestep)
        target = self.train_action_scheduler.training_target(action, noise, timestep)
        cloud_current_proprio = sample.get("cloud_current_proprio", edge_proprio)
        cloud_stale_proprio = sample.get("cloud_stale_proprio")
        if self.proprio_encoder is not None and cloud_stale_proprio is None:
            raise ValueError(
                "LeapBotCE requires cloud_stale_proprio so the stale cloud cache "
                "cannot be conditioned on current edge state."
            )
        fresh_context, fresh_context_mask = self._with_proprio(
            context, context_mask, cloud_current_proprio, "cloud current"
        )
        stale_context, stale_context_mask = self._with_proprio(
            context, context_mask, cloud_stale_proprio, "cloud stale"
        )
        fresh = self._encode_cloud_tensor(
            sample["cloud_current_image"], fresh_context, fresh_context_mask, context, context_mask
        )
        stale = self._encode_cloud_tensor(
            sample["cloud_stale_image"], stale_context, stale_context_mask, context, context_mask
        )
        pred_fresh = self._predict_from_cache(fresh, noisy, timestep, edge_context, edge_mask)
        pred_stale = self._predict_from_cache(stale, noisy, timestep, edge_context, edge_mask)
        valid = None
        if sample.get("action_is_pad") is not None:
            valid = (~sample["action_is_pad"].to(self.device, torch.bool)).float()

        def compute(pred):
            token = F.mse_loss(pred.float(), target.float(), reduction="none").mean(2)
            per_sample = token.mean(1) if valid is None else (token * valid).sum(1) / valid.sum(1).clamp(min=1)
            return (per_sample * self.train_action_scheduler.training_weight(timestep).to(per_sample)).mean()

        fresh_loss, stale_loss = compute(pred_fresh), compute(pred_stale)
        lam = stale_loss_weight(self.training_step, self.stale_loss_warmup_steps,
                                self.stale_loss_lambda_max)
        loss = (1 - lam) * fresh_loss + lam * stale_loss
        if self.training:
            self.training_step += 1
        delay = sample.get("cloud_delay_steps")
        return loss, {"loss_fresh": float(fresh_loss.detach()),
                      "loss_stale": float(stale_loss.detach()),
                      "stale_loss_weight": lam,
                      "delay_steps": float(delay.float().mean()) if delay is not None else 0.0}

    @torch.no_grad()
    def infer_action_edge(self, planning_cache, current_views, action_horizon,
                          num_inference_steps=20, sigma_shift=None, seed=None,
                          rand_device="cpu", edge_proprio=None, proprio=None, **_):
        """Denoise actions from a cloud cache and current edge-only inputs."""
        if edge_proprio is not None and proprio is not None:
            raise ValueError("Provide either edge_proprio or proprio, not both.")
        edge_proprio = edge_proprio if edge_proprio is not None else proprio
        if current_views.ndim == 4:
            current_views = current_views.unsqueeze(0)
        edge_context, edge_mask = self._with_proprio(
            planning_cache.context, planning_cache.context_mask, edge_proprio, "edge"
        )
        edge_context, edge_mask = self._edge_context(
            edge_context, edge_mask, current_views)
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action = torch.randn((planning_cache.context.shape[0], action_horizon, self.action_expert.action_dim),
                             generator=generator, device=rand_device).to(self.device, self.torch_dtype)
        timesteps, deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps, self.device, action.dtype, shift_override=sigma_shift)
        for step_t, delta in zip(timesteps, deltas):
            timestep = step_t.expand(action.shape[0]).to(self.device, action.dtype)
            prediction = self._predict_from_cache(planning_cache, action, timestep, edge_context, edge_mask)
            action = self.infer_action_scheduler.step(prediction, delta, action)
        return {"action": action[0].float().cpu()}

    @staticmethod
    def _split_views(input_image, num_views):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.shape[-1] % num_views:
            raise ValueError("Cannot split cloud image into equal edge views.")
        return torch.stack(input_image.chunk(num_views, dim=-1), dim=1)

    @torch.no_grad()
    def infer_action(self, prompt, input_image, action_horizon, proprio=None,
                     context=None, context_mask=None, current_views=None, **kwargs):
        current_views = current_views if current_views is not None else self._split_views(input_image, self.edge_num_views)
        cache = self.encode_cloud(
            input_image, prompt, context, context_mask, cloud_proprio=proprio
        )
        return self.infer_action_edge(
            cache, current_views, action_horizon, edge_proprio=proprio, **kwargs
        )

    def save_checkpoint(self, path, optimizer=None, step=None):
        # Frozen Wan/SigLIP base weights come from the configured local paths.
        # Export only the fine-tuned action and edge adapter parameters.
        payload = {"leapbotce_delta": True,
                   "action_expert": self.action_expert.state_dict(),
                   "edge_vision_projector": self.edge_vision.projector.state_dict(),
                   "method": "LeapBotCE", "step": step, "torch_dtype": str(self.torch_dtype),
                   "leapbotce_meta": {"edge_num_views": self.edge_num_views,
                                       "stale_loss_lambda_max": self.stale_loss_lambda_max,
                                       "stale_loss_warmup_steps": self.stale_loss_warmup_steps}}
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        if payload.get("leapbotce_delta", False):
            self.action_expert.load_state_dict(payload["action_expert"], strict=True)
            self.edge_vision.projector.load_state_dict(
                payload["edge_vision_projector"], strict=True)
            if self.proprio_encoder is not None and "proprio_encoder" in payload:
                self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
            if optimizer is not None and "optimizer" in payload:
                optimizer.load_state_dict(payload["optimizer"])
            return payload
        # v1 full-MoT checkpoint compatibility.
        if "mot" in payload:
            self.mot.load_state_dict(payload["mot"], strict=False)
        elif "dit" in payload:
            self.video_expert.load_state_dict(payload["dit"], strict=False)
        else:
            raise ValueError(f"Checkpoint missing LeapBotCE weights: {path}")
        if self.proprio_encoder is not None and "proprio_encoder" in payload:
            self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
        if "edge_vision" in payload:
            self.edge_vision.load_state_dict(payload["edge_vision"], strict=True)
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload
