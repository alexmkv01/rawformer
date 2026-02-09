"""Transformer encoder (Vaswani et al., 2017).

Each encoder block: self-attention -> add & norm -> FFN -> add & norm.
The Encoder stacks N identical blocks.
"""

import numpy as np
import numpy.typing as npt

from rawformer.attention.multi_head import MultiHeadAttention
from rawformer.exceptions import ForwardNotCalledError
from rawformer.layers.norm import LayerNorm
from rawformer.transformer.feed_forward import PositionWiseFeedForward


class EncoderBlock:
    """Single transformer encoder block.

    Args:
        d_model: Model embedding dimension.
        n_heads: Number of attention heads.
        d_ff: Feed-forward inner dimension.
        rng: NumPy random generator for weight initialization.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, rng: np.random.Generator) -> None:
        self.self_attn = MultiHeadAttention(d_model, n_heads, rng)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, rng)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        self._residual1_cache: npt.NDArray[np.float64] | None = None
        self._residual2_cache: npt.NDArray[np.float64] | None = None

    def forward(
        self,
        x: npt.NDArray[np.float64],
        mask: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Forward pass through one encoder block.

        Args:
            x: Input of shape (batch, seq_len, d_model).
            mask: Optional padding mask for self-attention.
        """
        self._residual1_cache = x
        attn_out = self.self_attn.forward(x, x, x, mask)
        x1 = self.norm1.forward(x + attn_out)

        self._residual2_cache = x1
        ffn_out = self.ffn.forward(x1)
        return self.norm2.forward(x1 + ffn_out)

    def backward(self, grad_z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if self._residual1_cache is None or self._residual2_cache is None:
            raise ForwardNotCalledError("EncoderBlock")

        # backward through norm2 and FFN residual
        grad_norm2 = self.norm2.backward(grad_z)
        grad_ffn = self.ffn.backward(grad_norm2)
        grad_x1 = grad_norm2 + grad_ffn

        # backward through norm1 and self-attention residual
        grad_norm1 = self.norm1.backward(grad_x1)
        grad_q, grad_k, grad_v = self.self_attn.backward(grad_norm1)
        grad_x: npt.NDArray[np.float64] = grad_norm1 + grad_q + grad_k + grad_v
        return grad_x

    def update_params(self, learning_rate: float) -> None:
        self.self_attn.update_params(learning_rate)
        self.ffn.update_params(learning_rate)
        self.norm1.update_params(learning_rate)
        self.norm2.update_params(learning_rate)


class Encoder:
    """Stack of N encoder blocks.

    Args:
        n_layers: Number of encoder blocks.
        d_model: Model embedding dimension.
        n_heads: Number of attention heads.
        d_ff: Feed-forward inner dimension.
        rng: NumPy random generator for weight initialization.
    """

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        rng: np.random.Generator,
    ) -> None:
        self.blocks = [EncoderBlock(d_model, n_heads, d_ff, rng) for _ in range(n_layers)]

    def forward(
        self,
        x: npt.NDArray[np.float64],
        mask: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Forward pass through all encoder blocks sequentially."""
        output = x
        for block in self.blocks:
            output = block.forward(output, mask)
        return output

    def backward(self, grad_z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        grad = grad_z
        for block in reversed(self.blocks):
            grad = block.backward(grad)
        return grad

    def update_params(self, learning_rate: float) -> None:
        for block in self.blocks:
            block.update_params(learning_rate)
