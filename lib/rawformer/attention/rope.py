"""Rotary Position Embeddings — RoFormer (Su et al., 2021).

Encodes absolute position by rotating Q and K vectors in paired 2-D
subspaces.  The key property: after rotation the dot product
``rope(q, pos=m) · rope(k, pos=n)`` depends only on ``m - n``,
giving the model relative-position awareness without explicit relative
position encodings.

Convention
----------
This implementation uses the **original RoFormer even/odd interleave**:
dimension pairs are ``(x[..., 0::2], x[..., 1::2])``.  An alternative
convention used by Meta's Llama splits the vector in half
``(x[..., :d//2], x[..., d//2:])``.  Both are mathematically equivalent
but produce different numerical outputs for the same input.

Usage
-----
The three public functions are stateless — no classes, no caches::

    cos, sin = build_rope_frequencies(seq_len, d_k)
    q_rot    = apply_rope(q, cos, sin)
    grad_q   = apply_rope_backward(grad_q_rot, cos, sin)
"""

import numpy as np
import numpy.typing as npt

from rawformer.core.exceptions import ShapeMismatchError


def build_rope_frequencies(
    seq_len: int,
    d_k: int,
    base: float = 10_000.0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Precompute cosine and sine tables for rotary embeddings.

    Args:
        seq_len: Maximum sequence length.
        d_k: Head dimension.  Must be even.
        base: Frequency base (default 10 000, following the paper).

    Returns:
        Tuple ``(cos_table, sin_table)`` each of shape
        ``(seq_len, d_k // 2)``.

    Raises:
        ValueError: If *d_k* is odd.
    """
    if d_k % 2 != 0:
        raise ValueError(f"d_k must be even for RoPE, got {d_k}")
    half = d_k // 2
    theta: npt.NDArray[np.float64] = 1.0 / np.power(base, (2.0 * np.arange(half)) / d_k)
    positions = np.arange(seq_len, dtype=np.float64)
    angles: npt.NDArray[np.float64] = np.outer(positions, theta)
    return np.cos(angles), np.sin(angles)


def apply_rope(
    x: npt.NDArray[np.float64],
    cos: npt.NDArray[np.float64],
    sin: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Apply rotary position embedding to *x*.

    The rotation is applied pair-wise using the **even/odd interleave**
    convention (RoFormer original).

    Args:
        x: Input of shape ``(..., seq_len, d_k)`` where the last
            dimension is even.
        cos: Cosine table of shape ``(seq_len, d_k // 2)`` from
            :func:`build_rope_frequencies`.
        sin: Sine table, same shape as *cos*.

    Returns:
        Rotated tensor, same shape as *x*.

    Raises:
        ShapeMismatchError: If *x* has an odd last dimension, or if
            the cos/sin tables don't match the expected shape.
    """
    d_k = x.shape[-1]
    if d_k % 2 != 0:
        raise ShapeMismatchError(f"Last dimension of x must be even for RoPE, got {d_k}")
    seq_len = x.shape[-2]
    expected_shape = (seq_len, d_k // 2)
    if cos.shape != expected_shape or sin.shape != expected_shape:
        raise ShapeMismatchError(
            f"cos/sin shape {cos.shape} does not match expected {expected_shape}"
        )
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out_even: npt.NDArray[np.float64] = x_even * cos - x_odd * sin
    out_odd: npt.NDArray[np.float64] = x_even * sin + x_odd * cos
    out = np.empty_like(x)
    out[..., 0::2] = out_even
    out[..., 1::2] = out_odd
    return out


def apply_rope_backward(
    grad_output: npt.NDArray[np.float64],
    cos: npt.NDArray[np.float64],
    sin: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Backward pass for :func:`apply_rope`.

    Because each 2-D rotation matrix is orthogonal, the gradient is
    simply the inverse rotation (rotate by -angle), which equals
    transposing the rotation matrix.

    Args:
        grad_output: Upstream gradient, same shape as the forward output.
        cos: Cosine table used in the forward pass.
        sin: Sine table used in the forward pass.

    Returns:
        Gradient with respect to the input *x*, same shape as
        *grad_output*.

    Raises:
        ShapeMismatchError: If *grad_output* has an odd last dimension,
            or if the cos/sin tables don't match the expected shape.
    """
    d_k = grad_output.shape[-1]
    if d_k % 2 != 0:
        raise ShapeMismatchError(f"Last dimension of grad_output must be even for RoPE, got {d_k}")
    seq_len = grad_output.shape[-2]
    expected_shape = (seq_len, d_k // 2)
    if cos.shape != expected_shape or sin.shape != expected_shape:
        raise ShapeMismatchError(
            f"cos/sin shape {cos.shape} does not match expected {expected_shape}"
        )
    g_even = grad_output[..., 0::2]
    g_odd = grad_output[..., 1::2]
    # Inverse rotation: transpose of [[cos, -sin], [sin, cos]]
    #                              = [[cos,  sin], [-sin, cos]]
    dx_even: npt.NDArray[np.float64] = g_even * cos + g_odd * sin
    dx_odd: npt.NDArray[np.float64] = -g_even * sin + g_odd * cos
    dx = np.empty_like(grad_output)
    dx[..., 0::2] = dx_even
    dx[..., 1::2] = dx_odd
    return dx
