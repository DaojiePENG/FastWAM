"""
Action loss with speed-based weighting for FastWAM.

Core idea: Slow-speed actions (fine-grained manipulation) are usually more critical
and should receive higher weights during training.
"""

import torch
import torch.nn.functional as F


def compute_weighted_action_loss(
    ground_truth_actions: torch.Tensor,
    predicted_actions: torch.Tensor,
    target_actions: torch.Tensor,
    weight_strategy: str = "inverse",
    clip_max_weight: float = 2.0,
    epsilon: float = 1e-3,
    alpha: float = 2.0,
    normalize_weights: bool = True,
    use_l1: bool = True,
    action_is_pad: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute weighted action loss with speed-based attention.

    This loss function assigns higher weights to slow-speed actions (critical for
    fine-grained manipulation) and lower weights to fast-speed actions.

    Args:
        ground_truth_actions: Original normalized actions (B, T, D), D=7 (first 6 dims are joint velocities)
        predicted_actions: Model predictions (B, T, D)
        target_actions: Training targets from flow matching (B, T, D)
        weight_strategy: Weight computation strategy
            - "inverse": 1/speed (basic inverse proportional)
            - "inverse_squared": 1/speed^2 (amplify weight differences)
            - "exp_decay": exp(-alpha*speed) (exponential decay)
            - "log": 1/log(1+speed) (logarithmic smoothing)
        clip_max_weight: Maximum weight to avoid extreme values
        epsilon: Small constant to avoid division by zero
        alpha: Exponential decay parameter (only for "exp_decay" strategy)
        normalize_weights: Whether to normalize weights to have mean=1
        use_l1: If True, use L1 loss; otherwise use MSE loss
        action_is_pad: Optional padding mask (B, T), True for padded positions

    Returns:
        weighted_loss_per_sample: Weighted loss per sample (B,), can be further aggregated
    """
    # 1. Extract first 6 joint dimensions to compute speed (ignore 7th dim: gripper)
    joint_gt = ground_truth_actions[..., :6]  # (B, T, 6)
    speed = torch.norm(joint_gt, dim=-1, keepdim=True)  # Joint speed magnitude (B, T, 1)
    speed = torch.clamp(speed, min=epsilon)  # Avoid division by zero

    # 2. Compute weights based on strategy (all are inverse mappings: low speed -> high weight)
    if weight_strategy == "inverse":
        # Basic inverse proportional: 1/speed
        weights = 1.0 / speed
    elif weight_strategy == "inverse_squared":
        # Squared inverse proportional: amplify weight differences for low-speed actions
        weights = 1.0 / (speed ** 2)
    elif weight_strategy == "exp_decay":
        # Exponential decay: weights decay rapidly as speed increases (alpha controls decay rate)
        weights = torch.exp(-alpha * speed)
    elif weight_strategy == "log":
        # Logarithmic mapping: smooth out extreme speed fluctuations
        weights = 1.0 / torch.log1p(speed)  # log1p = log(1+speed)
    else:
        raise ValueError(f"Unsupported weight strategy: {weight_strategy}")

    # 3. Weight clipping: limit maximum value before normalization to avoid extreme weights
    weights = torch.clamp(weights, max=clip_max_weight)

    # 3.2 Weight clipping: limit minimum value to avoid overly small weights
    min_weight = 1.0 / clip_max_weight
    weights = torch.clamp(weights, min=min_weight)

    # 4. Weight normalization: ensure average weight ~1 to stabilize training scale (optional)
    if normalize_weights:
        weights = weights / clip_max_weight * 2.0  # Normalize to average ~1

    # 5. Compute weighted loss (apply same weight to all 7 action dimensions)
    if use_l1:
        # L1 loss (more robust to outliers)
        errors = torch.abs(predicted_actions - target_actions)  # (B, T, D)
    else:
        # MSE loss (original FastWAM behavior)
        errors = (predicted_actions - target_actions) ** 2  # (B, T, D)

    weighted_errors = errors * weights  # (B, T, D) * (B, T, 1) -> (B, T, D)

    # 6. Aggregate over action dimensions: mean over D
    loss_per_timestep = weighted_errors.mean(dim=2)  # (B, T)

    # 7. Handle padding mask if provided
    if action_is_pad is not None:
        valid = (~action_is_pad).to(device=loss_per_timestep.device, dtype=loss_per_timestep.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)  # (B,)
        loss_per_sample = (loss_per_timestep * valid).sum(dim=1) / valid_sum  # (B,)
    else:
        loss_per_sample = loss_per_timestep.mean(dim=1)  # (B,)

    return loss_per_sample


def compute_action_loss_token(
    predicted_actions: torch.Tensor,
    target_actions: torch.Tensor,
    use_mse: bool = True,
) -> torch.Tensor:
    """
    Compute basic action loss per token (FastWAM's original implementation).

    Args:
        predicted_actions: Model predictions (B, T, D)
        target_actions: Training targets (B, T, D)
        use_mse: If True, use MSE; otherwise use L1

    Returns:
        loss_per_token: Loss per token (B, T)
    """
    if use_mse:
        loss_per_token = F.mse_loss(
            predicted_actions.float(),
            target_actions.float(),
            reduction="none"
        ).mean(dim=2)  # (B, T)
    else:
        loss_per_token = F.l1_loss(
            predicted_actions.float(),
            target_actions.float(),
            reduction="none"
        ).mean(dim=2)  # (B, T)

    return loss_per_token
