"""Transformer decoder (Vaswani et al., 2017).

Each decoder block: masked self-attention -> dropout -> add & norm ->
    cross-attention -> dropout -> add & norm -> FFN -> dropout -> add & norm.
The Decoder stacks N identical blocks.
"""

import numpy as np
import numpy.typing as npt

from rawformer.attention.multi_head import MultiHeadAttention
from rawformer.base import Layer
from rawformer.exceptions import ForwardNotCalledError
from rawformer.layers.dropout import Dropout
from rawformer.layers.norm import LayerNorm
from rawformer.transformer.feed_forward import PositionWiseFeedForward


def causal_mask(seq_len: int) -> npt.NDArray[np.float64]:
    """Create a causal (look-ahead) mask for autoregressive decoding.

    Returns an additive mask of shape (1, 1, seq_len, seq_len) where future
    positions are -inf and allowed positions are 0.

    Args:
        seq_len: Sequence length.
    """
    mask = np.triu(np.full((seq_len, seq_len), -np.inf), k=1)
    result: npt.NDArray[np.float64] = mask[np.newaxis, np.newaxis, :, :]
    return result


class DecoderBlock(Layer):
    """Single transformer decoder block with masked self-attention and cross-attention.

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
        self.cross_attn = MultiHeadAttention(d_model, n_heads, rng, dropout_rate)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, rng, dropout_rate)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout_rate, rng)
        self.dropout2 = Dropout(dropout_rate, rng)
        self.dropout3 = Dropout(dropout_rate, rng)

        self._forward_called: bool = False

    @property
    def dropouts(self) -> list[Dropout]:
        return [
            self.dropout1,
            self.dropout2,
            self.dropout3,
            *self.self_attn.dropouts,
            *self.cross_attn.dropouts,
            *self.ffn.dropouts,
        ]

    def forward(
        self,
        x: npt.NDArray[np.float64],
        encoder_output: npt.NDArray[np.float64],
        self_attn_mask: npt.NDArray[np.float64] | None = None,
        cross_attn_mask: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Forward pass through one decoder block.

        Args:
            x: Decoder input of shape (batch, seq_tgt, d_model).
            encoder_output: Encoder output of shape (batch, seq_src, d_model).
            self_attn_mask: Causal mask for masked self-attention.
            cross_attn_mask: Optional padding mask for cross-attention.
        """
        # post-norm: residual + dropout then layer norm (Vaswani et al., 2017)
        attn_out = self.self_attn.forward(x, x, x, self_attn_mask)
        x1 = self.norm1.forward(x + self.dropout1.forward(attn_out))

        # cross-attention + residual + norm
        cross_out = self.cross_attn.forward(x1, encoder_output, encoder_output, cross_attn_mask)
        x2 = self.norm2.forward(x1 + self.dropout2.forward(cross_out))

        # FFN + residual + norm
        ffn_out = self.ffn.forward(x2)
        out = self.norm3.forward(x2 + self.dropout3.forward(ffn_out))

        self._forward_called = True
        return out

    def backward(
        self, grad_z: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Backward pass through one decoder block.

        Args:
            grad_z: Upstream gradient of shape (batch, seq_tgt, d_model).

        Returns:
            Tuple of (grad_x, grad_encoder_output).
        """
        if not self._forward_called:
            raise ForwardNotCalledError("DecoderBlock")

        # backward through norm3 + FFN residual
        grad_norm3 = self.norm3.backward(grad_z)
        grad_ffn = self.dropout3.backward(self.ffn.backward(grad_norm3))
        grad_x2 = grad_norm3 + grad_ffn

        # backward through norm2 + cross-attention residual
        # cross-attention gradient routing: q grad -> decoder input,
        # k/v grads -> encoder output
        grad_norm2 = self.norm2.backward(grad_x2)
        grad_cross_q, grad_cross_k, grad_cross_v = self.cross_attn.backward(
            self.dropout2.backward(grad_norm2)
        )
        grad_x1 = grad_norm2 + grad_cross_q
        grad_encoder: npt.NDArray[np.float64] = grad_cross_k + grad_cross_v

        # backward through norm1 + self-attention residual
        grad_norm1 = self.norm1.backward(grad_x1)
        grad_self_q, grad_self_k, grad_self_v = self.self_attn.backward(
            self.dropout1.backward(grad_norm1)
        )
        grad_x: npt.NDArray[np.float64] = grad_norm1 + grad_self_q + grad_self_k + grad_self_v

        return grad_x, grad_encoder

    def update_params(self, learning_rate: float) -> None:
        self.self_attn.update_params(learning_rate)
        self.cross_attn.update_params(learning_rate)
        self.ffn.update_params(learning_rate)
        self.norm1.update_params(learning_rate)
        self.norm2.update_params(learning_rate)
        self.norm3.update_params(learning_rate)


class Decoder(Layer):
    """Stack of N decoder blocks.

    Args:
        n_layers: Number of decoder blocks.
        d_model: Model embedding dimension.
        n_heads: Number of attention heads.
        d_ff: Feed-forward inner dimension.
        rng: NumPy random generator for weight initialization.
        dropout_rate: Dropout probability passed to each decoder block.
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
            DecoderBlock(d_model, n_heads, d_ff, rng, dropout_rate) for _ in range(n_layers)
        ]

    @property
    def dropouts(self) -> list[Dropout]:
        return [d for block in self.blocks for d in block.dropouts]

    def forward(
        self,
        x: npt.NDArray[np.float64],
        encoder_output: npt.NDArray[np.float64],
        self_attn_mask: npt.NDArray[np.float64] | None = None,
        cross_attn_mask: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Forward pass through all decoder blocks sequentially."""
        output = x
        for block in self.blocks:
            output = block.forward(output, encoder_output, self_attn_mask, cross_attn_mask)
        return output

    def backward(
        self, grad_z: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Backward pass accumulating encoder gradients across all blocks."""
        grad = grad_z
        enc_grads: list[npt.NDArray[np.float64]] = []
        for block in reversed(self.blocks):
            grad, grad_enc = block.backward(grad)
            enc_grads.append(grad_enc)
        total_grad_enc: npt.NDArray[np.float64] = np.sum(enc_grads, axis=0)
        return grad, total_grad_enc

    def update_params(self, learning_rate: float) -> None:
        for block in self.blocks:
            block.update_params(learning_rate)
