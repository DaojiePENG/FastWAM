import timm

from .leapbotce_factory import _as_dict, create_leapbotce as _create_leapbotce


def create_leapbotce(edge_vision=None, **kwargs):
    """Create LeapBotCE with SigLIP loaded from an explicit local checkpoint."""
    edge_vision = _as_dict(edge_vision)
    checkpoint_path = edge_vision.pop(
        "checkpoint_path", "checkpoints/siglip-base/model.safetensors"
    )
    encoder = timm.create_model(
        edge_vision.get("model_name", "vit_base_patch16_siglip_224"),
        pretrained=False,
        checkpoint_path=checkpoint_path,
        num_classes=0,
    )
    model = _create_leapbotce(edge_vision={**edge_vision, "pretrained": False}, **kwargs)
    # Replace the randomly initialized encoder while preserving the projector.
    model.edge_vision.encoder = encoder.to(device=model.device, dtype=model.torch_dtype)
    model.edge_vision.embed_dim = int(encoder.embed_dim)
    model.edge_vision.encoder.requires_grad_(False)
    model.edge_vision.encoder.eval()
    return model
