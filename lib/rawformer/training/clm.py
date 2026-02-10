"""Decoder-only causal language model (Radford et al., 2018).

Composes token embedding, positional encoding, a stack of decoder blocks
(self-attention only, no cross-attention), and an output projection to
vocabulary logits.  Used for causal language modelling (next-token
prediction) and supervised fine-tuning.
"""

import numpy as np
import numpy.typing as npt

from rawformer.attention.multi_head import MultiHeadAttention
from rawformer.core.base import Layer
from rawformer.core.exceptions import ForwardNotCalledError
from rawformer.layers.dropout import Dropout
from rawformer.layers.embedding import PositionalEncoding, TokenEmbedding
from rawformer.layers.linear import LinearLayer
from rawformer.layers.norm import LayerNorm
from rawformer.transformer.decoder import causal_mask
from rawformer.transformer.feed_forward import PositionWiseFeedForward


class DecoderOnlyBlock(Layer):
    """Single decoder-only block: self-attention -> add & norm -> FFN -> add & norm.

    Unlike the encoder-decoder DecoderBlock, this has no cross-attention.
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
        return [
            self.dropout1,
            self.dropout2,
            *self.self_attn.dropouts,
            *self.ffn.dropouts,
        ]

    def forward(
        self,
        x: npt.NDArray[np.float64],
        mask: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Forward pass through one decoder-only block."""
        attn_out = self.self_attn.forward(x, x, x, mask)
        x1 = self.norm1.forward(x + self.dropout1.forward(attn_out))

        ffn_out = self.ffn.forward(x1)
        out = self.norm2.forward(x1 + self.dropout2.forward(ffn_out))

        self._forward_called = True
        return out

    def backward(self, grad_z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Backward pass through one decoder-only block."""
        if not self._forward_called:
            raise ForwardNotCalledError("DecoderOnlyBlock")

        grad_norm2 = self.norm2.backward(grad_z)
        grad_ffn = self.dropout2.backward(self.ffn.backward(grad_norm2))
        grad_x1 = grad_norm2 + grad_ffn

        grad_norm1 = self.norm1.backward(grad_x1)
        grad_q, grad_k, grad_v = self.self_attn.backward(self.dropout1.backward(grad_norm1))
        grad_x: npt.NDArray[np.float64] = grad_norm1 + grad_q + grad_k + grad_v

        return grad_x

    def update_params(self, learning_rate: float) -> None:
        self.self_attn.update_params(learning_rate)
        self.ffn.update_params(learning_rate)
        self.norm1.update_params(learning_rate)
        self.norm2.update_params(learning_rate)


class DecoderOnlyModel(Layer):
    """Decoder-only transformer for causal language modelling.

    Follows the GPT architecture (Radford et al., 2018) with sinusoidal
    positional encoding.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_len: int,
        rng: np.random.Generator,
        dropout_rate: float = 0.1,
    ) -> None:
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.token_embed = TokenEmbedding(vocab_size, d_model, rng)
        self.pos_enc = PositionalEncoding(max_len, d_model)
        self.embed_dropout = Dropout(dropout_rate, rng)
        self.blocks = [
            DecoderOnlyBlock(d_model, n_heads, d_ff, rng, dropout_rate) for _ in range(n_layers)
        ]
        self.output_proj = LinearLayer(d_model, vocab_size, rng)

        self._forward_called: bool = False

    @property
    def dropouts(self) -> list[Dropout]:
        return [self.embed_dropout] + [d for block in self.blocks for d in block.dropouts]

    def train(self) -> None:
        """Set all dropout layers to training mode."""
        for d in self.dropouts:
            d.training = True

    def eval(self) -> None:
        """Set all dropout layers to inference mode."""
        for d in self.dropouts:
            d.training = False

    def forward(self, token_ids: npt.NDArray[np.intp]) -> npt.NDArray[np.float64]:
        """Forward pass: token IDs (batch, seq_len) -> logits (batch, seq_len, vocab_size)."""
        x = self.token_embed.forward(token_ids)
        x = self.embed_dropout.forward(self.pos_enc.forward(x))

        seq_len = token_ids.shape[1]
        mask = causal_mask(seq_len)

        for block in self.blocks:
            x = block.forward(x, mask)

        self._forward_called = True
        return self.output_proj.forward(x)

    def backward(self, grad_z: npt.NDArray[np.float64]) -> None:
        """Backward pass through the entire decoder-only model.

        Returns None because this is the top-level model; there is no
        upstream layer to propagate gradients to.
        """
        if not self._forward_called:
            raise ForwardNotCalledError("DecoderOnlyModel")

        grad = self.output_proj.backward(grad_z)

        for block in reversed(self.blocks):
            grad = block.backward(grad)

        grad = self.pos_enc.backward(self.embed_dropout.backward(grad))
        self.token_embed.backward(grad)

    def update_params(self, learning_rate: float) -> None:
        self.token_embed.update_params(learning_rate)
        for block in self.blocks:
            block.update_params(learning_rate)
        self.output_proj.update_params(learning_rate)
