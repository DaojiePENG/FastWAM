import torch.nn as nn

from .leapbotce_local_factory import create_leapbotce as _create_local_leapbotce
from .video_lora import inject_video_lora, lora_parameter_count


def create_leapbotce(**kwargs):
    """Build the memory-feasible LeapBotCE trainable module set."""
    # a4 producer-side adaptation: optional LoRA on the frozen video expert's
    # attention projections.  Popped before the local factory so the extra
    # config block never reaches its signature.
    video_lora_cfg = kwargs.pop("video_lora", None) or {}
    model = _create_local_leapbotce(**kwargs)
    trainable_modules = {
        "action_expert": model.action_expert,
        "edge_vision_projector": model.edge_vision.projector,
    }
    cloud_kv_adapter = getattr(model, "cloud_kv_adapter", None)
    if cloud_kv_adapter is not None:
        trainable_modules["cloud_kv_adapter"] = cloud_kv_adapter
    if getattr(model, "cloud_kv_codec", None) is not None:
        trainable_modules["cloud_kv_codec"] = model.cloud_kv_codec
    # Wan22Trainer freezes the complete model, then re-enables model.dit.  Keep
    # the cloud adapter in this registry without exposing the 5B video expert.
    if bool(video_lora_cfg.get("enabled", False)):
        wrappers = inject_video_lora(
            model.video_expert,
            rank=int(video_lora_cfg.get("rank", 16)),
            alpha=float(video_lora_cfg.get("alpha", 32.0)),
            dropout=float(video_lora_cfg.get("dropout", 0.0)),
            targets=tuple(video_lora_cfg.get("targets", ["q", "k", "v", "o"])),
        )
        trainable_modules["video_lora"] = nn.ModuleList(wrappers)
        model.video_lora_wrappers = wrappers
        # LoRA backward now flows through the frozen 5B tower, so activations
        # must be kept for autograd.  MoT forbids per-expert checkpointing at
        # __init__ (compiled-tensor-core guard), but its runtime checkpoint
        # path (_apply_post_with_optional_checkpoint) is fully implemented and
        # compile_training_layers is hard-wired False, so enabling the flag
        # here — after MoT.__init__ has already passed — is safe.  MoT.mixtures
        # holds the very same video expert object, so the runtime reads it.
        model.video_expert.use_gradient_checkpointing = True
        print(
            f"[edge-train-factory] video LoRA injected: {len(wrappers)} wrappers, "
            f"{lora_parameter_count(wrappers) / 1e6:.2f}M trainable LoRA params "
            f"(rank={video_lora_cfg.get('rank', 16)}, targets={video_lora_cfg.get('targets', ['q', 'k', 'v', 'o'])}); "
            "video-expert gradient checkpointing enabled post-init"
        )
    model.dit = nn.ModuleDict(trainable_modules)
    return model
