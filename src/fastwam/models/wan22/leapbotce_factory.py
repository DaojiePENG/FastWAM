from omegaconf import DictConfig, OmegaConf

from fastwam.runtime import create_fastwam

from .leapbotce import LeapBotCE


def _as_dict(value):
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    return {} if value is None else dict(value)


def create_leapbotce(edge_vision=None, cloudedge=None, **kwargs):
    """Hydra factory that reuses the FastWAM pretrained loader."""
    edge_vision = _as_dict(edge_vision)
    cloudedge = _as_dict(cloudedge)
    kv_adapter = _as_dict(cloudedge.get("kv_adapter"))
    kv_adapter_rank = kv_adapter.get("rank", kv_adapter.get("bottleneck_dim", 64))
    kv_codec = _as_dict(cloudedge.get("kv_codec"))
    pool_grid = kv_codec.get("pool_grid")
    base = create_fastwam(**kwargs)
    model = LeapBotCE(
        video_expert=base.video_expert,
        action_expert=base.action_expert,
        mot=base.mot,
        vae=base.vae,
        text_encoder=base.text_encoder,
        tokenizer=base.tokenizer,
        text_dim=base.text_dim,
        proprio_dim=base.proprio_dim,
        device=str(base.device),
        torch_dtype=base.torch_dtype,
        video_train_shift=base.train_video_scheduler.shift,
        video_infer_shift=base.infer_video_scheduler.shift,
        video_num_train_timesteps=base.train_video_scheduler.num_train_timesteps,
        action_train_shift=base.train_action_scheduler.shift,
        action_infer_shift=base.infer_action_scheduler.shift,
        action_num_train_timesteps=base.train_action_scheduler.num_train_timesteps,
        loss_lambda_video=0.0,
        loss_lambda_action=base.loss_lambda_action,
        edge_vision_model_name=edge_vision.get("model_name", "vit_base_patch16_siglip_224"),
        edge_num_views=int(edge_vision.get("num_views", 2)),
        edge_vision_pretrained=bool(edge_vision.get("pretrained", True)),
        edge_vision_freeze=bool(edge_vision.get("freeze", True)),
        edge_vision_mode=str(edge_vision.get("mode", "pooled")),
        edge_vision_tokens_per_view=int(edge_vision.get("tokens_per_view", 5)),
        edge_vision_selection=str(edge_vision.get("selection", "topk")),
        stale_loss_lambda_max=float(cloudedge.get("stale_loss_lambda_max", 0.5)),
        stale_loss_warmup_steps=int(cloudedge.get("stale_loss_warmup_steps", 1000)),
        cloud_kv_adapter_enabled=bool(kv_adapter.get("enabled", False)),
        cloud_kv_adapter_rank=int(kv_adapter_rank),
        cloud_kv_adapter_dropout=float(kv_adapter.get("dropout", 0.0)),
        cloud_kv_codec_enabled=bool(kv_codec.get("enabled", False)),
        cloud_kv_codec_pool_grid=list(pool_grid) if pool_grid else None,
        cloud_kv_codec_bottleneck_dim=int(kv_codec.get("bottleneck_dim", 1024)),
        cloud_kv_codec_use_decoder=bool(kv_codec.get("use_kv_decoder", False)),
        cloud_kv_codec_decoder_rank=int(kv_codec.get("decoder_rank", 256)),
        cloud_kv_codec_dropout=float(kv_codec.get("dropout", 0.0)),
        edge_prefix_enabled=bool(cloudedge.get("edge_prefix_enabled", True)),
    )
    # Wan22Trainer optimizes model.dit; make the edge projector part of it.
    model.mot.add_module("edge_vision_projector", model.edge_vision.projector)
    model.model_paths = dict(getattr(base, "model_paths", {}))
    return model
