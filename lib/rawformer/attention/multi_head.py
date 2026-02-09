"""Multi-head attention (Vaswani et al., 2017).

Splits Q, K, V into multiple heads, applies scaled dot-product attention
to each head independently, concatenates the results, and projects back
to d_model.
"""

from typing import TypedDict

import numpy as np
import numpy.typing as npt

from rawformer.attention.scaled_dot_product import (
    scaled_dot_product_attention,
    scaled_dot_product_attention_backward,
)
from rawformer.base import Layer
from rawformer.exceptions import ForwardNotCalledError
from rawformer.layers.dropout import Dropout
from rawformer.layers.linear import LinearLayer


class _MHACache(TypedDict):
    q: npt.NDArray[np.float64]
    k: npt.NDArray[np.float64]
    v: npt.NDArray[np.float64]
    weights: npt.NDArray[np.float64]


class MultiHeadAttention(Layer):
    """Multi-head attention mechanism.

    Args:
        d_model: Model embedding dimension.
        n_heads: Number of attention heads. Must evenly divide d_model.
        rng: NumPy random generator for weight initialization.
        dropout_rate: Dropout probability applied to attention weights
            after softmax.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        rng: np.random.Generator,
        dropout_rate: float = 0.1,
    ) -> None:
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = LinearLayer(d_model, d_model, rng)
        self.w_k = LinearLayer(d_model, d_model, rng)
        self.w_v = LinearLayer(d_model, d_model, rng)
        self.w_o = LinearLayer(d_model, d_model, rng)

        self.attn_dropout = Dropout(dropout_rate, rng)
        self._cache: _MHACache | None = None

    @property
    def dropouts(self) -> list[Dropout]:
        return [self.attn_dropout]

    def _split_heads(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Reshape (batch, seq, d_model) -> (batch, n_heads, seq, d_k)."""
        batch, seq_len, _ = x.shape
        reshaped = x.reshape(batch, seq_len, self.n_heads, self.d_k)
        result: npt.NDArray[np.float64] = reshaped.transpose(0, 2, 1, 3)
        return result

    def _merge_heads(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Reshape (batch, n_heads, seq, d_k) -> (batch, seq, d_model)."""
        batch, _, seq_len, _ = x.shape
        transposed = x.transpose(0, 2, 1, 3)
        result: npt.NDArray[np.float64] = transposed.reshape(batch, seq_len, self.d_model)
        return result

    def forward(
        self,
        query: npt.NDArray[np.float64],
        key: npt.NDArray[np.float64],
        value: npt.NDArray[np.float64],
        mask: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Compute multi-head attention.

        For self-attention, pass the same tensor for query, key, value.
        For cross-attention, query comes from the decoder and key/value
        from the encoder.

        Args:
            query: Shape (batch, seq_q, d_model).
            key: Shape (batch, seq_k, d_model).
            value: Shape (batch, seq_k, d_model).
            mask: Optional additive mask broadcastable to
                (batch, n_heads, seq_q, seq_k).

        Returns:
            Output of shape (batch, seq_q, d_model).
        """
        q = self._split_heads(self.w_q.forward(query))
        k = self._split_heads(self.w_k.forward(key))
        v = self._split_heads(self.w_v.forward(value))

        attn_out, weights = scaled_dot_product_attention(q, k, v, mask, dropout=self.attn_dropout)

        self._cache = _MHACache(q=q, k=k, v=v, weights=weights)

        merged = self._merge_heads(attn_out)
        return self.w_o.forward(merged)

    def backward(
        self, grad_z: npt.NDArray[np.float64]
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """Backward pass through multi-head attention.

        For self-attention where query=key=value, the caller should sum
        the three returned gradients.

        Args:
            grad_z: Upstream gradient of shape (batch, seq_q, d_model).

        Returns:
            Tuple of (grad_query, grad_key, grad_value), each with the
            same shape as the corresponding forward input.
        """
        if self._cache is None:
            raise ForwardNotCalledError("MultiHeadAttention")
        q = self._cache["q"]
        k = self._cache["k"]
        v = self._cache["v"]
        weights = self._cache["weights"]

        # backward through output projection
        grad_merged = self.w_o.backward(grad_z)

        # un-merge heads: (batch, seq, d_model) -> (batch, n_heads, seq, d_k)
        batch, seq_q, _ = grad_merged.shape
        grad_attn_out = grad_merged.reshape(batch, seq_q, self.n_heads, self.d_k).transpose(
            0, 2, 1, 3
        )

        # backward through scaled dot-product attention
        grad_q, grad_k, grad_v = scaled_dot_product_attention_backward(
            grad_attn_out, q, k, v, weights, dropout=self.attn_dropout
        )

        # merge heads back for the linear backward passes
        grad_q_merged = self._merge_heads(grad_q)
        grad_k_merged = self._merge_heads(grad_k)
        grad_v_merged = self._merge_heads(grad_v)

        # backward through Q, K, V projections
        grad_query = self.w_q.backward(grad_q_merged)
        grad_key = self.w_k.backward(grad_k_merged)
        grad_value = self.w_v.backward(grad_v_merged)

        return grad_query, grad_key, grad_value

    def update_params(self, learning_rate: float) -> None:
        self.w_q.update_params(learning_rate)
        self.w_k.update_params(learning_rate)
        self.w_v.update_params(learning_rate)
        self.w_o.update_params(learning_rate)
