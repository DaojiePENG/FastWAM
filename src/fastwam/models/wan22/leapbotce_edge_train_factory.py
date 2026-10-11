import torch.nn as nn

from .leapbotce_local_factory import create_leapbotce as _create_local_leapbotce


def create_leapbotce(**kwargs):
    """Build the memory-feasible LeapBotCE trainable module set."""
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
    model.dit = nn.ModuleDict(trainable_modules)
    return model
