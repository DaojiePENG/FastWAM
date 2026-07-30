"""LeapBot-VA public API."""

from .memory import (
    KVSegment,
    LeapMemoryConfig,
    LeapMemoryState,
    MemoryPhase,
    build_block_causal_mask,
)
__all__ = [
    "KVSegment",
    "LeapBotVA",
    "LeapMemoryConfig",
    "LeapMemoryState",
    "MemoryPhase",
    "build_block_causal_mask",
]


def __getattr__(name: str):
    """Keep the lightweight memory API usable without loading the 6B stack."""
    if name == "LeapBotVA":
        from .models.leapbot import LeapBotVA

        return LeapBotVA
    raise AttributeError(name)
