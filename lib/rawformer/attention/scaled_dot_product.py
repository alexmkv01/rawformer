"""Scaled dot-product attention (Vaswani et al., 2017).

Attention(Q, K, V) = softmax(QK^T / sqrt(d_k) + mask) V

When a Dropout instance is provided, dropout is applied to the attention
weights after softmax.  The Dropout object caches the mask internally
for use in the backward pass.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from rawformer.layers.dropout import Dropout


def _softmax(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Row-wise softmax with numerical stability (subtract max per row)."""
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    result: npt.NDArray[np.float64] = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    return result


def scaled_dot_product_attention(
    q: npt.NDArray[np.float64],
    k: npt.NDArray[np.float64],
    v: npt.NDArray[np.float64],
    mask: npt.NDArray[np.float64] | None = None,
    dropout: Dropout | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute scaled dot-product attention.

    Args:
        q: Queries of shape (..., seq_q, d_k).
        k: Keys of shape (..., seq_k, d_k).
        v: Values of shape (..., seq_k, d_v).
        mask: Optional additive mask broadcastable to (..., seq_q, seq_k).
            Use -inf (or large negative) to mask out positions.
        dropout: Optional Dropout instance applied to the attention
            weights after softmax.

    Returns:
        Tuple of (output, attention_weights):
            output: shape (..., seq_q, d_v)
            attention_weights: shape (..., seq_q, seq_k), after dropout
                when provided.
    """
    d_k = q.shape[-1]
    scale = np.sqrt(np.float64(d_k))

    scores: npt.NDArray[np.float64] = np.einsum("...qi,...ki->...qk", q, k) / scale

    if mask is not None:
        scores = scores + mask

    weights = _softmax(scores)

    if dropout is not None:
        weights = dropout.forward(weights)

    output: npt.NDArray[np.float64] = np.einsum("...qk,...kv->...qv", weights, v)
    return output, weights


def scaled_dot_product_attention_backward(
    grad_output: npt.NDArray[np.float64],
    q: npt.NDArray[np.float64],
    k: npt.NDArray[np.float64],
    v: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    dropout: Dropout | None = None,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Backward pass for scaled dot-product attention.

    Args:
        grad_output: Upstream gradient, shape (..., seq_q, d_v).
        q: Queries cached from forward, shape (..., seq_q, d_k).
        k: Keys cached from forward, shape (..., seq_k, d_k).
        v: Values cached from forward, shape (..., seq_k, d_v).
        weights: Attention weights cached from forward, shape
            (..., seq_q, seq_k).
        dropout: Optional Dropout instance (must be the same one used
            in the forward pass so that it holds the cached mask).

    Returns:
        Tuple of (grad_q, grad_k, grad_v).
    """
    d_k = q.shape[-1]
    scale = np.sqrt(np.float64(d_k))

    # grad through weights @ v
    grad_weights: npt.NDArray[np.float64] = np.einsum("...qv,...kv->...qk", grad_output, v)
    grad_v: npt.NDArray[np.float64] = np.einsum("...qk,...qv->...kv", weights, grad_output)

    # grad through dropout (if used in forward)
    if dropout is not None:
        grad_weights = dropout.backward(grad_weights)

    # grad through softmax: d_softmax = weights * (grad_weights - sum(grad_weights * weights))
    sum_term = np.sum(grad_weights * weights, axis=-1, keepdims=True)
    grad_scores: npt.NDArray[np.float64] = weights * (grad_weights - sum_term)

    # grad through score computation (QK^T / sqrt(d_k))
    grad_scores_scaled = grad_scores / scale
    grad_q: npt.NDArray[np.float64] = np.einsum("...qk,...ki->...qi", grad_scores_scaled, k)
    grad_k: npt.NDArray[np.float64] = np.einsum("...qk,...qi->...ki", grad_scores_scaled, q)

    return grad_q, grad_k, grad_v
