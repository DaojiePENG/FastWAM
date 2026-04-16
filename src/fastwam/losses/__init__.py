"""Loss functions for FastWAM training."""

from .action_loss import compute_weighted_action_loss

__all__ = ["compute_weighted_action_loss"]
