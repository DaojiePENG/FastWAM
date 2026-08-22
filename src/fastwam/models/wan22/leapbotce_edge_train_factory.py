import torch.nn as nn

from .leapbotce_local_factory import create_leapbotce as _create_local_leapbotce


def create_leapbotce(**kwargs):
    """Memory-feasible v1: frozen cloud expert, trainable action expert and edge adapter."""
    model = _create_local_leapbotce(**kwargs)
    model.dit = nn.ModuleDict(
        {
            "action_expert": model.action_expert,
            "edge_vision_projector": model.edge_vision.projector,
        }
    )
    return model
