"""Flash Attention: tiled attention with online softmax.

Implements the tiled attention algorithm from FlashAttention (Dao et al.,
2022) with the outer-Q/inner-KV loop order and deferred output rescaling
from FlashAttention-2 (Dao, 2023).  The backward pass recomputes the
attention matrix block-by-block from stored logsumexp statistics, avoiding
materialisation of the full (seq_q, seq_k) attention matrix.

FlashAttention-3 (Shah et al., 2024) introduces Hopper-specific
optimisations — warp specialisation, pingpong scheduling, WGMMA, TMA,
and FP8 with incoherent processing — that are outside the scope of a
pure NumPy implementation.

References:
    - Dao et al., "FlashAttention: Fast and Memory-Efficient Exact
      Attention with IO-Awareness", NeurIPS 2022.
    - Dao, "FlashAttention-2: Faster Attention with Better Parallelism
      and Work Partitioning", ICLR 2024.
    - Shah et al., "FlashAttention-3: Fast and Accurate Attention with
      Asynchrony and Low-precision", 2024.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from rawformer.core.exceptions import ShapeMismatchError


@dataclass
class FlashForwardResult:
    """Return value of :func:`flash_attention_forward`.

    Using a dataclass makes call-site unpacking self-documenting and
    avoids positional confusion (especially for ``base_seed``).
    """

    output: npt.NDArray[np.float64]
    """Attention output, shape ``(..., seq_q, d_v)``."""

    logsumexp: npt.NDArray[np.float64]
    """``m + log(ell)`` per Q-row, shape ``(..., seq_q)``.

    Needed by the backward pass to recover attention weights without
    storing the full attention matrix.
    """

    base_seed: int | None
    """Seed used to derive per-block dropout masks.

    ``None`` when dropout is inactive.  Pass to
    :func:`flash_attention_backward` to regenerate the identical masks.
    """


def _block_ranges(total: int, block_size: int) -> list[tuple[int, int]]:
    """Return (start, end) pairs that tile ``[0, total)`` into blocks.

    Args:
        total: Length of the range to tile.
        block_size: Maximum size of each block.

    Raises:
        ValueError: If *block_size* is not positive.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    return [(i, min(i + block_size, total)) for i in range(0, total, block_size)]


def _make_block_rng(
    base_seed: int,
    block_i: int,
    block_j: int,
) -> np.random.Generator:
    """Create a deterministic RNG for a specific block pair.

    Derives a per-block seed so that the identical dropout mask can be
    regenerated in the backward pass without storing it.

    Args:
        base_seed: Seed drawn from the caller-supplied RNG at the start
            of the forward pass.
        block_i: Index of the Q-block.
        block_j: Index of the KV-block.

    Returns:
        A fresh ``Generator`` seeded deterministically from the inputs.
    """
    # SeedSequence mixes the three integers into a high-quality seed.
    ss = np.random.SeedSequence([base_seed, block_i, block_j])
    return np.random.Generator(np.random.PCG64(ss))


def _apply_block_dropout(
    weights: npt.NDArray[np.float64],
    rate: float,
    block_rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Apply inverted dropout to a block of attention weights.

    Args:
        weights: Attention weights for this block.
        rate: Dropout probability.
        block_rng: Deterministic RNG for this block.

    Returns:
        Masked and scaled weights.
    """
    keep = 1.0 - rate
    # Mask covers the full broadcasted shape (..., Bq, Bk) so that
    # each element in the batch gets an independent dropout decision.
    mask = (block_rng.binomial(1, keep, weights.shape) / keep).astype(
        np.float64,
    )
    return weights * mask


# ── Validation helpers ───────────────────────────────────────────────


def _validate_qkv_shapes(
    q: npt.NDArray[np.float64],
    k: npt.NDArray[np.float64],
    v: npt.NDArray[np.float64],
) -> None:
    """Check that Q, K, V have compatible shapes.

    Raises:
        ShapeMismatchError: On incompatible dimensions.
    """
    if q.shape[-1] != k.shape[-1]:
        raise ShapeMismatchError(
            f"q and k must have the same d_k: q.shape={q.shape}, k.shape={k.shape}"
        )
    if k.shape[-2] != v.shape[-2]:
        raise ShapeMismatchError(
            f"k and v must have the same seq_k: k.shape={k.shape}, v.shape={v.shape}"
        )
    # Validate that all batch dimensions broadcast.
    try:
        np.broadcast_shapes(q.shape[:-2], k.shape[:-2], v.shape[:-2])
    except ValueError as exc:
        raise ShapeMismatchError(
            f"q, k, v batch dimensions are not broadcastable: "
            f"q.shape={q.shape}, k.shape={k.shape}, v.shape={v.shape}"
        ) from exc


# ── Forward ──────────────────────────────────────────────────────────


def flash_attention_forward(
    q: npt.NDArray[np.float64],
    k: npt.NDArray[np.float64],
    v: npt.NDArray[np.float64],
    mask: npt.NDArray[np.float64] | None = None,
    block_size: int = 64,
    dropout_rate: float = 0.0,
    rng: np.random.Generator | None = None,
    training: bool = True,
) -> FlashForwardResult:
    """Tiled forward pass for scaled dot-product attention.

    Uses the FA2 loop order: outer loop over Q-blocks, inner loop over
    KV-blocks.  Running softmax statistics (row-wise max ``m`` and
    sum-of-exponentials ``ell``) are maintained per Q-row; the output
    accumulator is rescaled once at the end of the inner loop.

    Args:
        q: Queries, shape ``(..., seq_q, d_k)``.
        k: Keys, shape ``(..., seq_k, d_k)``.
        v: Values, shape ``(..., seq_k, d_v)``.
        mask: Optional additive mask broadcastable to
            ``(..., seq_q, seq_k)``.  Use ``-inf`` to block positions.
        block_size: Tile size for Q and KV blocks.
        dropout_rate: Probability of zeroing attention weights.
        rng: Random generator, required when *dropout_rate* > 0 and
            *training* is True.
        training: When False, dropout is skipped.

    Returns:
        A :class:`FlashForwardResult` containing *output*, *logsumexp*,
        and *base_seed*.

    Raises:
        ShapeMismatchError: If Q/K/V shapes are incompatible.
        ValueError: If *block_size* is not positive, or *rng* is missing
            when dropout is active.
    """
    _validate_qkv_shapes(q, k, v)

    seq_q = q.shape[-2]
    seq_k = k.shape[-2]
    d_k = q.shape[-1]
    d_v = v.shape[-1]
    batch_shape = np.broadcast_shapes(q.shape[:-2], k.shape[:-2], v.shape[:-2])
    scale = np.sqrt(np.float64(d_k))

    # Output accumulator and softmax statistics.
    output = np.zeros((*batch_shape, seq_q, d_v), dtype=np.float64)
    # m_i: running row-wise max of scores (for numerical stability).
    m = np.full((*batch_shape, seq_q), -np.inf, dtype=np.float64)
    # ell_i: running sum of exp(scores - m_i).
    ell = np.zeros((*batch_shape, seq_q), dtype=np.float64)

    q_blocks = _block_ranges(seq_q, block_size)
    kv_blocks = _block_ranges(seq_k, block_size)

    apply_dropout = dropout_rate > 0.0 and training
    if apply_dropout:
        if rng is None:
            raise ValueError("rng is required when dropout_rate > 0 and training=True")
        # int() narrows the numpy scalar to a plain int for mypy.
        base_seed = int(rng.integers(0, 2**63))
        # Local binding narrows int | None → int so mypy can see it
        # inside the nested loop without type: ignore comments.
        _seed: int = base_seed
    else:
        base_seed = None

    for bi, (q_start, q_end) in enumerate(q_blocks):
        q_block = q[..., q_start:q_end, :]  # (..., Bq, d_k)

        for bj, (k_start, k_end) in enumerate(kv_blocks):
            k_block = k[..., k_start:k_end, :]  # (..., Bk, d_k)
            v_block = v[..., k_start:k_end, :]  # (..., Bk, d_v)

            # S_ij = Q_i K_j^T / sqrt(d_k)
            s_ij: npt.NDArray[np.float64] = (
                np.einsum("...qi,...ki->...qk", q_block, k_block) / scale
            )

            # Apply additive mask for this block.
            if mask is not None:
                mask_block = mask[..., q_start:q_end, k_start:k_end]
                s_ij = s_ij + mask_block

            # Online softmax: update running max and sum.
            # m_ij: row-wise max within this KV block.
            m_ij = np.max(s_ij, axis=-1)  # (..., Bq)

            m_old = m[..., q_start:q_end]
            m_new = np.maximum(m_old, m_ij)

            # Rescale existing accumulators for the new max.
            correction = np.exp(m_old - m_new)  # (..., Bq)
            ell_old = ell[..., q_start:q_end]
            ell_new = correction * ell_old + np.sum(
                np.exp(s_ij - m_new[..., :, np.newaxis]),
                axis=-1,
            )

            # Attention weights for this block (unnormalised by ell).
            p_ij = np.exp(s_ij - m_new[..., :, np.newaxis])  # (..., Bq, Bk)

            # Dropout.
            if apply_dropout:
                block_rng = _make_block_rng(_seed, bi, bj)
                p_ij = _apply_block_dropout(p_ij, dropout_rate, block_rng)

            # Update output accumulator.
            # O_i <- correction * O_i + P_ij @ V_j
            output[..., q_start:q_end, :] = correction[..., :, np.newaxis] * output[
                ..., q_start:q_end, :
            ] + np.einsum("...qk,...kv->...qv", p_ij, v_block)

            m[..., q_start:q_end] = m_new
            ell[..., q_start:q_end] = ell_new

    # Final normalisation: O_i /= ell_i.
    output = output / ell[..., :, np.newaxis]

    logsumexp = m + np.log(ell)

    return FlashForwardResult(output=output, logsumexp=logsumexp, base_seed=base_seed)


# ── Backward ─────────────────────────────────────────────────────────


def flash_attention_backward(
    grad_output: npt.NDArray[np.float64],
    q: npt.NDArray[np.float64],
    k: npt.NDArray[np.float64],
    v: npt.NDArray[np.float64],
    output: npt.NDArray[np.float64],
    logsumexp: npt.NDArray[np.float64],
    mask: npt.NDArray[np.float64] | None = None,
    block_size: int = 64,
    dropout_rate: float = 0.0,
    base_seed: int | None = None,
    training: bool = True,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Tiled backward pass for flash attention.

    Recomputes the attention matrix block-by-block using the stored
    ``logsumexp`` statistics, never materialising the full
    ``(seq_q, seq_k)`` weight matrix.

    Args:
        grad_output: Upstream gradient, shape ``(..., seq_q, d_v)``.
        q: Queries cached from forward.
        k: Keys cached from forward.
        v: Values cached from forward.
        output: Output cached from forward.
        logsumexp: Logsumexp from forward, shape ``(..., seq_q)``.
        mask: Same additive mask used in the forward pass.
        block_size: Must match the forward pass.
        dropout_rate: Must match the forward pass.
        base_seed: Base seed from the forward pass (``None`` when
            dropout was inactive).
        training: Must match the forward pass.

    Returns:
        Tuple of ``(grad_q, grad_k, grad_v)``.

    Raises:
        ValueError: If dropout is active but *base_seed* is ``None``.
    """
    seq_q = q.shape[-2]
    seq_k = k.shape[-2]
    d_k = q.shape[-1]
    scale = np.sqrt(np.float64(d_k))

    grad_q = np.zeros_like(q)
    grad_k = np.zeros_like(k)
    grad_v = np.zeros_like(v)

    q_blocks = _block_ranges(seq_q, block_size)
    kv_blocks = _block_ranges(seq_k, block_size)

    apply_dropout = dropout_rate > 0.0 and training
    if apply_dropout:
        if base_seed is None:
            raise ValueError("base_seed is required for backward with dropout")
        # Local binding narrows int | None → int for the nested loop.
        _seed: int = base_seed

    # D_i = rowsum(dO * O) — the "D vector" from FlashAttention.
    d_vec = np.sum(grad_output * output, axis=-1)  # (..., seq_q)

    for bi, (q_start, q_end) in enumerate(q_blocks):
        q_block = q[..., q_start:q_end, :]
        grad_out_block = grad_output[..., q_start:q_end, :]
        lse_block = logsumexp[..., q_start:q_end]
        d_block = d_vec[..., q_start:q_end]

        for bj, (k_start, k_end) in enumerate(kv_blocks):
            k_block = k[..., k_start:k_end, :]
            v_block = v[..., k_start:k_end, :]

            # Recompute S_ij.
            s_ij: npt.NDArray[np.float64] = (
                np.einsum("...qi,...ki->...qk", q_block, k_block) / scale
            )
            if mask is not None:
                s_ij = s_ij + mask[..., q_start:q_end, k_start:k_end]

            # Recover normalised attention weights from logsumexp.
            p_ij: npt.NDArray[np.float64] = np.exp(
                s_ij - lse_block[..., :, np.newaxis],
            )

            # Apply and later back-prop through dropout.  Both calls
            # regenerate the identical mask from the same (base_seed,
            # bi, bj) triple, mirroring the forward pass.
            if apply_dropout:
                block_rng = _make_block_rng(_seed, bi, bj)
                p_ij_dropped = _apply_block_dropout(p_ij, dropout_rate, block_rng)
            else:
                p_ij_dropped = p_ij

            # dV_j += P_ij^T @ dO_i  (using post-dropout weights).
            grad_v[..., k_start:k_end, :] += np.einsum(
                "...qk,...qv->...kv",
                p_ij_dropped,
                grad_out_block,
            )

            # dP_ij = dO_i @ V_j^T.
            dp_ij: npt.NDArray[np.float64] = np.einsum(
                "...qv,...kv->...qk",
                grad_out_block,
                v_block,
            )

            # Back-prop through dropout: regenerate identical mask for dP.
            if apply_dropout:
                block_rng = _make_block_rng(_seed, bi, bj)
                dp_ij = _apply_block_dropout(dp_ij, dropout_rate, block_rng)

            # dS_ij = P_ij * (dP_ij - D_i)  — softmax backward.
            ds_ij: npt.NDArray[np.float64] = p_ij * (dp_ij - d_block[..., :, np.newaxis])

            # Scale back.
            ds_ij_scaled = ds_ij / scale

            # dQ_i += dS_ij @ K_j.
            grad_q[..., q_start:q_end, :] += np.einsum(
                "...qk,...ki->...qi",
                ds_ij_scaled,
                k_block,
            )

            # dK_j += dS_ij^T @ Q_i.
            grad_k[..., k_start:k_end, :] += np.einsum(
                "...qk,...qi->...ki",
                ds_ij_scaled,
                q_block,
            )

    return grad_q, grad_k, grad_v
