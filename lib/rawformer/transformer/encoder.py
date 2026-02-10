"""Transformer encoder (Vaswani et al., 2017).

Each encoder block: self-attention -> dropout -> add & norm -> FFN -> dropout -> add & norm.
The Encoder stacks N identical blocks.
"""

import numpy as np
import numpy.typing as npt

from rawformer.attention.multi_head import MultiHeadAttention
from rawformer.core.base import Layer
from rawformer.core.exceptions import ForwardNotCalledError
from rawformer.layers.dropout import Dropout
from rawformer.layers.norm import LayerNorm
from rawformer.transformer.feed_forward import PositionWiseFeedForward


class EncoderBlock(Layer):
    """Single transformer encoder block.

    Args:
        d_model: Model embedding dimension.
        n_heads: Number of attention heads.
        d_ff: Feed-forward inner dimension.
        rng: NumPy random generator for weight initialization.
        dropout_rate: Dropout probability for sub-layer outputs and
            attention weights.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        rng: np.random.Generator,
        dropout_rate: float = 0.1,
    ) -> None:
        self.self_attn = MultiHeadAttention(d_model, n_heads, rng, dropout_rate)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, rng, dropout_rate)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout_rate, rng)
        self.dropout2 = Dropout(dropout_rate, rng)

        self._forward_called: bool = False

    @property
    def dropouts(self) -> list[Dropout]:
        return [self.dropout1, self.dropout2, *self.self_attn.dropouts, *self.ffn.dropouts]

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
        # post-norm: residual + dropout then layer norm (Vaswani et al., 2017)
        attn_out = self.self_attn.forward(x, x, x, mask)
        x1 = self.norm1.forward(x + self.dropout1.forward(attn_out))

        ffn_out = self.ffn.forward(x1)
        out = self.norm2.forward(x1 + self.dropout2.forward(ffn_out))

        self._forward_called = True
        return out

    def backward(self, grad_z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if not self._forward_called:
            raise ForwardNotCalledError("EncoderBlock")

        # backward through norm2 and FFN residual
        grad_norm2 = self.norm2.backward(grad_z)
        grad_ffn = self.dropout2.backward(self.ffn.backward(grad_norm2))
        grad_x1 = grad_norm2 + grad_ffn

        # backward through norm1 and self-attention residual
        grad_norm1 = self.norm1.backward(grad_x1)
        grad_q, grad_k, grad_v = self.self_attn.backward(self.dropout1.backward(grad_norm1))
        grad_x: npt.NDArray[np.float64] = grad_norm1 + grad_q + grad_k + grad_v
        return grad_x

    def update_params(self, learning_rate: float) -> None:
        self.self_attn.update_params(learning_rate)
        self.ffn.update_params(learning_rate)
        self.norm1.update_params(learning_rate)
        self.norm2.update_params(learning_rate)


class Encoder(Layer):
    """Stack of N encoder blocks.

    Args:
        n_layers: Number of encoder blocks.
        d_model: Model embedding dimension.
        n_heads: Number of attention heads.
        d_ff: Feed-forward inner dimension.
        rng: NumPy random generator for weight initialization.
        dropout_rate: Dropout probability passed to each encoder block.
    """

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        rng: np.random.Generator,
        dropout_rate: float = 0.1,
    ) -> None:
        self.blocks = [
            EncoderBlock(d_model, n_heads, d_ff, rng, dropout_rate) for _ in range(n_layers)
        ]

    @property
    def dropouts(self) -> list[Dropout]:
        return [d for block in self.blocks for d in block.dropouts]

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
