"""Attention mechanisms: multi-head, flash, and rotary position embeddings."""

from rawformer.attention.flash import (
    FlashForwardResult,
    flash_attention_backward,
    flash_attention_forward,
)
from rawformer.attention.multi_head import MultiHeadAttention
from rawformer.attention.rope import (
    apply_rope,
    apply_rope_backward,
    build_rope_frequencies,
)

__all__ = [
    "FlashForwardResult",
    "MultiHeadAttention",
    "apply_rope",
    "apply_rope_backward",
    "build_rope_frequencies",
    "flash_attention_backward",
    "flash_attention_forward",
]
