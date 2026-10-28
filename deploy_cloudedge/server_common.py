"""Shared bootstrap and timed inference stages for the cloud/edge services.

The services rebuild the exact eval-time LeapBot-CE stack (hydra compose of
``sim_libero.yaml`` with the deployed task recipe, delta-checkpoint load,
processor + normalizer) by reusing ``WorldActionLeapBotCEPolicy`` — the same
construction path the LIBERO/RoboTwin sweep entries run.  The inference
stages below replicate ``LeapBotCE._encode_cloud_tensor`` /
``infer_action_edge`` line-for-line with CUDA-event marks inserted, so the
sub-step latencies (VAE, prefill, codec / SigLIP, denoise) attribute to the
same math the evaluation executes; ``cloud_server`` self-tests at startup
that the split path reproduces ``model.encode_cloud`` + ``infer_action_edge``
bit-for-bit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for _path in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.core.global_hydra import GlobalHydra  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from experiments.robotwin.leapbotce_policy.deploy_policy import (  # noqa: E402
    WorldActionLeapBotCEPolicy,
)
from fastwam.models.wan22.leapbotce import CloudPlanningCache  # noqa: E402

SIM_CFG_NAME = "sim_libero.yaml"


def register_resolvers() -> None:
    """Same resolvers eval_libero_single registers before composing."""
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver("max", lambda x: max(x), replace=True)
    OmegaConf.register_new_resolver(
        "split", lambda s, idx: s.split("/")[int(idx)], replace=True
    )


def compose_deploy_cfg(task: str, extra_overrides: list[str] | None = None):
    """Compose sim_libero.yaml with the deployed task recipe + eval overrides."""
    register_resolvers()
    configs_root = (PROJECT_ROOT / "configs").resolve()
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(configs_root)):
        return compose(
            config_name=SIM_CFG_NAME,
            overrides=[f"task={task}"] + list(extra_overrides or []),
        )


def load_policy(cfg, checkpoint: str, stats: str, device: str) -> WorldActionLeapBotCEPolicy:
    """Build the full LeapBot-CE policy exactly like the eval entries do."""
    return WorldActionLeapBotCEPolicy(
        model_cfg=cfg.model,
        processor_cfg=cfg.data.train.processor,
        checkpoint_path=str(checkpoint),
        dataset_stats_path=Path(stats),
        device=device,
        model_dtype=torch.bfloat16,
        action_horizon=32,
        replan_steps=10,
        num_inference_steps=10,
        sigma_shift=None,
        seed=None,
        rand_device="cpu",
        timing_enabled=False,
    )


class GpuTimeline:
    """CUDA-event marks; ``elapsed`` returns per-section milliseconds."""

    def __init__(self) -> None:
        self._marks: list[tuple[str, torch.cuda.Event]] = []

    def mark(self, name: str) -> None:
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self._marks.append((name, event))

    def elapsed(self) -> dict[str, float]:
        torch.cuda.synchronize()
        out: dict[str, float] = {}
        for (name_a, ev_a), (name_b, ev_b) in zip(self._marks, self._marks[1:]):
            out[name_b] = out.get(name_b, 0.0) + ev_a.elapsed_time(ev_b)
        return out


def center_crop_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """eval_libero_single._center_crop_resize, verbatim."""
    pil_image = Image.fromarray(image)
    src_w, src_h = pil_image.size
    scale = max(width / src_w, height / src_h)
    resized = pil_image.resize(
        (round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR
    )
    left = max((resized.size[0] - width) // 2, 0)
    top = max((resized.size[1] - height) // 2, 0)
    cropped = resized.crop((left, top, left + width, top + height))
    return np.asarray(cropped, dtype=np.uint8)


def compose_image(policy: WorldActionLeapBotCEPolicy, primary: np.ndarray,
                  wrist: np.ndarray) -> np.ndarray:
    """Two-camera horizontal composite, mirrors eval_libero_single._obs_to_model_input."""
    image_meta = policy.processor.shape_meta["images"]
    if len(image_meta) < 2:
        raise ValueError("deploy protocol expects the two-camera LIBERO recipe")
    primary_h, primary_w = (int(v) for v in image_meta[0]["shape"][1:])
    wrist_h, wrist_w = (int(v) for v in image_meta[1]["shape"][1:])
    return np.concatenate(
        [
            center_crop_resize(primary, primary_w, primary_h),
            center_crop_resize(wrist, wrist_w, wrist_h),
        ],
        axis=1,
    )


def composite_to_tensor(model, composite: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(composite)).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(device=model.device, dtype=model.torch_dtype)
    return tensor * (2.0 / 255.0) - 1.0


def views_from_composite(model, composite: np.ndarray) -> torch.Tensor:
    """eval path: model._split_views on the normalized composite."""
    return model._split_views(composite_to_tensor(model, composite), model.edge_num_views)


def encode_prompt_cached(policy: WorldActionLeapBotCEPolicy, prompt: str):
    cached = policy._prompt_cache.get(prompt)
    if cached is None:
        with torch.no_grad():
            cached = policy.model.encode_prompt(prompt)
        policy._prompt_cache[prompt] = cached
    return cached


def codec_z(model, cache_k: list[torch.Tensor], cache_v: list[torch.Tensor],
            token_grid: tuple[int, int]) -> torch.Tensor:
    """The transmitted payload: CloudPlanningCodec's bottleneck output z.

    Replicates codec.forward up to (and excluding) the per-layer scaling, so
    the wire carries the pre-scale z and the edge re-instantiates per-layer
    K/V with the checkpoint's own scale vectors (zero transmission cost).
    """
    codec = model.cloud_kv_codec
    first = cache_v[0]
    parameter = codec.in_proj.weight
    value_stack = torch.stack(list(cache_v), dim=0).to(
        device=parameter.device, dtype=parameter.dtype
    )
    pooled = codec._spatial_pool(value_stack, token_grid)
    weights = torch.softmax(codec.layer_logits.to(parameter.dtype), dim=0)
    mixed = torch.einsum("l,lbmc->bmc", weights, pooled)
    return codec.out_proj(codec.dropout(F.gelu(codec.in_proj(codec.norm(mixed))))) .to(first.dtype)


@torch.no_grad()
def encode_cloud_parts(model, image: torch.Tensor, context: torch.Tensor,
                       context_mask: torch.Tensor, cloud_proprio: torch.Tensor):
    """VAE encode + Wan prefill + codec with CUDA-event marks.

    Line-for-line ``LeapBotCE._encode_cloud_tensor`` (codec branch), returning
    the wire payload z plus (seq_len, tokens_per_frame) and sub-step timings.
    """
    timeline = GpuTimeline()
    timeline.mark("start")
    cloud_context, cloud_context_mask = model._with_proprio(
        context, context_mask, cloud_proprio, "cloud")
    latents = model._encode_video_latents(
        image.unsqueeze(2).to(model.device, model.torch_dtype))
    timeline.mark("vae")
    timestep = torch.zeros((latents.shape[0],), device=model.device, dtype=latents.dtype)
    fuse = bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False))
    values = model.video_expert.prepare(
        latents, timestep, cloud_context, cloud_context_mask, None, fuse
    )
    video_tokens, _, video_t_mod, video_context, video_context_mask, video_freqs, \
        _, _, _, tokens_per_frame = values
    video_mask = model.video_expert.build_video_to_video_mask(
        video_tokens.shape[1], tokens_per_frame, video_tokens.device)
    cache_k, cache_v = model.mot.prefill_video_cache_tensor(
        video_tokens, video_freqs, video_t_mod, video_context,
        video_context_mask, video_mask)
    timeline.mark("prefill")
    z = codec_z(model, cache_k, cache_v, model._video_token_grid(latents))
    timeline.mark("codec")
    timings = timeline.elapsed()
    return z, int(model.cloud_kv_codec.num_tokens), int(tokens_per_frame), timings


@torch.no_grad()
def encode_cloud_raw_parts(model, image: torch.Tensor, context: torch.Tensor,
                           context_mask: torch.Tensor, cloud_proprio: torch.Tensor):
    """No-codec interface: the full per-layer K/V cache crosses the network.

    ``LeapBotCE.encode_cloud`` without the adapter/codec tail — the
    uncompressed-interface counterfactual (``tab:ablations`` "Full cache"
    row): same VAE + prefill, but ``planning_seq_len`` stays at the full
    video token count and the 30-layer K/V lists themselves are the payload.
    """
    timeline = GpuTimeline()
    timeline.mark("start")
    cloud_context, cloud_context_mask = model._with_proprio(
        context, context_mask, cloud_proprio, "cloud")
    latents = model._encode_video_latents(
        image.unsqueeze(2).to(model.device, model.torch_dtype))
    timeline.mark("vae")
    timestep = torch.zeros((latents.shape[0],), device=model.device, dtype=latents.dtype)
    fuse = bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False))
    values = model.video_expert.prepare(
        latents, timestep, cloud_context, cloud_context_mask, None, fuse
    )
    video_tokens, _, video_t_mod, video_context, video_context_mask, video_freqs, \
        _, _, _, tokens_per_frame = values
    video_mask = model.video_expert.build_video_to_video_mask(
        video_tokens.shape[1], tokens_per_frame, video_tokens.device)
    cache_k, cache_v = model.mot.prefill_video_cache_tensor(
        video_tokens, video_freqs, video_t_mod, video_context,
        video_context_mask, video_mask)
    timeline.mark("prefill")
    timings = timeline.elapsed()
    seq_len = int(cache_k[0].shape[1])
    return cache_k, cache_v, seq_len, int(tokens_per_frame), timings


def rebuild_cache(z: torch.Tensor, scale_k: torch.Tensor, scale_v: torch.Tensor,
                  context: torch.Tensor, context_mask: torch.Tensor,
                  seq_len: int, tokens_per_frame: int) -> CloudPlanningCache:
    """Edge-side per-layer K/V re-instantiation (codec.forward's scaling tail)."""
    keys = [(z * scale_k[i].to(z.dtype)).to(z.dtype) for i in range(len(scale_k))]
    values = [(z * scale_v[i].to(z.dtype)).to(z.dtype) for i in range(len(scale_v))]
    return CloudPlanningCache(keys, values, context, context_mask, seq_len, tokens_per_frame)


@torch.no_grad()
def edge_denoise_parts(model, cache: CloudPlanningCache, current_views: torch.Tensor,
                       action_horizon: int, edge_proprio: torch.Tensor,
                       num_inference_steps: int, sigma_shift, seed,
                       rand_device: str = "cpu"):
    """SigLIP prefix + 10-step denoise with CUDA-event marks.

    Line-for-line ``LeapBotCE.infer_action_edge`` with the timeline inserted.
    """
    timeline = GpuTimeline()
    timeline.mark("start")
    edge_context, edge_mask = model._with_proprio(
        cache.context, cache.context_mask, edge_proprio, "edge"
    )
    edge_prefix = model._encode_edge_prefix(current_views)
    timeline.mark("siglip")
    if edge_prefix is None:
        edge_context, edge_mask = model._edge_context(edge_context, edge_mask, current_views)
    generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
    action = torch.randn(
        (cache.context.shape[0], action_horizon, model.action_expert.action_dim),
        generator=generator, device=rand_device,
    ).to(model.device, model.torch_dtype)
    timesteps, deltas = model.infer_action_scheduler.build_inference_schedule(
        num_inference_steps, model.device, action.dtype, shift_override=sigma_shift)
    for step_t, delta in zip(timesteps, deltas):
        timestep = step_t.expand(action.shape[0]).to(model.device, action.dtype)
        prediction = model._predict_from_cache(
            cache, action, timestep, edge_context, edge_mask, edge_prefix)
        action = model.infer_action_scheduler.step(prediction, delta, action)
    timeline.mark("denoise")
    timings = timeline.elapsed()
    return {"action": action[0].float().cpu()}, timings


def strip_edge_weights(model) -> None:
    """Cloud host: free the 1B action expert + edge vision after the self-test."""
    del model.mot.mixtures["action"]
    model.action_expert = None
    if getattr(model.mot, "_modules", None) is not None and \
            "edge_vision_projector" in model.mot._modules:
        del model.mot._modules["edge_vision_projector"]
    model.edge_vision = None
    model.mot.mixtures["video"]  # touch to fail loudly if the video expert is gone
    import gc

    gc.collect()
    torch.cuda.empty_cache()


def strip_cloud_weights(model) -> None:
    """Edge host: free Wan + VAE + T5 + codec; keep action stream + edge vision.

    ``model.mot`` stays (the action-stream blocks live in
    ``mot.mixtures["action"]``); ``model.dit`` keeps pointing at it.
    """
    del model.mot.mixtures["video"]
    model.video_expert = None
    model.vae = None
    model.text_encoder = None
    model.tokenizer = None
    model.cloud_kv_codec = None
    import gc

    gc.collect()
    torch.cuda.empty_cache()
