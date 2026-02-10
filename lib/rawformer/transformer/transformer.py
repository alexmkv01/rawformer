"""Full encoder-decoder transformer (Vaswani et al., 2017).

Combines source/target embeddings, positional encoding, encoder stack,
decoder stack, and an output linear projection to vocabulary logits.
"""

import numpy as np
import numpy.typing as npt

from rawformer.core.base import Layer
from rawformer.core.exceptions import ForwardNotCalledError
from rawformer.layers.dropout import Dropout
from rawformer.layers.embedding import PositionalEncoding, TokenEmbedding
from rawformer.layers.linear import LinearLayer
from rawformer.transformer.decoder import Decoder, causal_mask
from rawformer.transformer.encoder import Encoder


class Transformer(Layer):
    """Full encoder-decoder transformer model.

    Args:
        src_vocab_size: Source vocabulary size.
        tgt_vocab_size: Target vocabulary size.
        d_model: Model embedding dimension.
        n_heads: Number of attention heads.
        n_encoder_layers: Number of encoder blocks.
        n_decoder_layers: Number of decoder blocks.
        d_ff: Feed-forward inner dimension.
        max_len: Maximum sequence length for positional encoding.
        rng: NumPy random generator for weight initialization.
        dropout_rate: Dropout probability applied throughout the model
            (embeddings, sub-layers, attention weights, FFN).
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        n_heads: int,
        n_encoder_layers: int,
        n_decoder_layers: int,
        d_ff: int,
        max_len: int,
        rng: np.random.Generator,
        dropout_rate: float = 0.1,
    ) -> None:
        self.src_embed = TokenEmbedding(src_vocab_size, d_model, rng)
        self.tgt_embed = TokenEmbedding(tgt_vocab_size, d_model, rng)
        self.pos_enc = PositionalEncoding(max_len, d_model)
        self.src_dropout = Dropout(dropout_rate, rng)
        self.tgt_dropout = Dropout(dropout_rate, rng)
        self.encoder = Encoder(n_encoder_layers, d_model, n_heads, d_ff, rng, dropout_rate)
        self.decoder = Decoder(n_decoder_layers, d_model, n_heads, d_ff, rng, dropout_rate)
        self.output_proj = LinearLayer(d_model, tgt_vocab_size, rng)

        self._forward_called: bool = False

    @property
    def dropouts(self) -> list[Dropout]:
        return [
            self.src_dropout,
            self.tgt_dropout,
            *self.encoder.dropouts,
            *self.decoder.dropouts,
        ]

    def train(self) -> None:
        """Set all dropout layers to training mode."""
        for d in self.dropouts:
            d.training = True

    def eval(self) -> None:
        """Set all dropout layers to inference mode (dropout disabled)."""
        for d in self.dropouts:
            d.training = False

    def forward(
        self,
        src: npt.NDArray[np.intp],
        tgt: npt.NDArray[np.intp],
        src_padding_mask: npt.NDArray[np.float64] | None = None,
        tgt_padding_mask: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Forward pass through the full encoder-decoder transformer.

        Args:
            src: Source token indices of shape (batch, src_seq_len).
            tgt: Target token indices of shape (batch, tgt_seq_len).
            src_padding_mask: Optional padding mask for encoder self-attention.
            tgt_padding_mask: Optional padding mask for cross-attention.

        Returns:
            Logits of shape (batch, tgt_seq_len, tgt_vocab_size).
        """
        # source embedding + positional encoding + dropout -> encoder
        src_tokens = self.src_embed.forward(src)
        src_emb = self.src_dropout.forward(self.pos_enc.forward(src_tokens))
        enc_out = self.encoder.forward(src_emb, src_padding_mask)

        # target embedding + positional encoding + dropout -> decoder
        tgt_tokens = self.tgt_embed.forward(tgt)
        tgt_emb = self.tgt_dropout.forward(self.pos_enc.forward(tgt_tokens))
        self_attn_mask = causal_mask(tgt.shape[1])
        dec_out = self.decoder.forward(tgt_emb, enc_out, self_attn_mask, tgt_padding_mask)

        self._forward_called = True

        return self.output_proj.forward(dec_out)

    def backward(self, grad_z: npt.NDArray[np.float64]) -> None:
        """Backward pass through the full transformer.

        Args:
            grad_z: Upstream gradient on the logits,
                shape (batch, tgt_seq_len, tgt_vocab_size).
        """
        if not self._forward_called:
            raise ForwardNotCalledError("Transformer")

        # backward through output projection
        grad_dec = self.output_proj.backward(grad_z)

        # backward through decoder
        grad_tgt_emb, grad_enc_out = self.decoder.backward(grad_dec)

        # backward through encoder
        grad_src_emb = self.encoder.backward(grad_enc_out)

        # backward through dropout + positional encoding (PE is pass-through)
        grad_tgt_emb = self.pos_enc.backward(self.tgt_dropout.backward(grad_tgt_emb))
        grad_src_emb = self.pos_enc.backward(self.src_dropout.backward(grad_src_emb))

        # backward through embeddings
        self.tgt_embed.backward(grad_tgt_emb)
        self.src_embed.backward(grad_src_emb)

    def update_params(self, learning_rate: float) -> None:
        self.src_embed.update_params(learning_rate)
        self.tgt_embed.update_params(learning_rate)
        self.encoder.update_params(learning_rate)
        self.decoder.update_params(learning_rate)
        self.output_proj.update_params(learning_rate)
