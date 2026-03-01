"""Tests for flash attention tiled implementation.

Cross-verifies flash_attention_forward / flash_attention_backward
against the standard scaled_dot_product_attention to ensure the tiled
online-softmax algorithm produces numerically identical results.
"""

import numpy as np
import numpy.typing as npt
import pytest
import torch
import torch.nn.functional as F

from rawformer.attention.flash import (
    FlashForwardResult,
    flash_attention_backward,
    flash_attention_forward,
)
from rawformer.attention.scaled_dot_product import (
    scaled_dot_product_attention,
    scaled_dot_product_attention_backward,
)
from rawformer.core.exceptions import ShapeMismatchError
from rawformer.transformer.decoder import causal_mask

# =====================================================================
# Forward: correctness
# =====================================================================


class TestFlashAttentionForward:
    def test_output_matches_standard(self) -> None:
        """Flash forward must match standard attention within tight tolerance."""
        rng = np.random.default_rng(42)
        q = rng.standard_normal((2, 4, 16, 8))
        k = rng.standard_normal((2, 4, 16, 8))
        v = rng.standard_normal((2, 4, 16, 8))

        expected, _, _ = scaled_dot_product_attention(q, k, v)
        result = flash_attention_forward(q, k, v, block_size=4)

        assert isinstance(result, FlashForwardResult)
        np.testing.assert_allclose(result.output, expected, atol=1e-10)

    def test_output_shape(self) -> None:
        """Output shape must be (..., seq_q, d_v)."""
        rng = np.random.default_rng(0)
        q = rng.standard_normal((2, 4, 5, 8))
        k = rng.standard_normal((2, 4, 6, 8))
        v = rng.standard_normal((2, 4, 6, 10))

        result = flash_attention_forward(q, k, v, block_size=3)
        assert result.output.shape == (2, 4, 5, 10)
        assert result.logsumexp.shape == (2, 4, 5)

    def test_logsumexp_values(self) -> None:
        """Logsumexp must equal log(sum(exp(scores))) computed via standard softmax."""
        rng = np.random.default_rng(42)
        q = rng.standard_normal((1, 1, 4, 8))
        k = rng.standard_normal((1, 1, 4, 8))
        v = rng.standard_normal((1, 1, 4, 8))

        result = flash_attention_forward(q, k, v, block_size=2)

        # Compute reference logsumexp from raw scores.
        d_k = q.shape[-1]
        scale = np.sqrt(float(d_k))
        scores = np.einsum("...qi,...ki->...qk", q, k) / scale
        ref_lse = np.log(np.sum(np.exp(scores - np.max(scores, axis=-1, keepdims=True)), axis=-1))
        ref_lse += np.max(scores, axis=-1)

        np.testing.assert_allclose(result.logsumexp, ref_lse, atol=1e-10)

    def test_causal_mask(self) -> None:
        """Flash attention with causal mask must match standard."""
        rng = np.random.default_rng(42)
        seq_len = 12
        q = rng.standard_normal((2, 4, seq_len, 8))
        k = rng.standard_normal((2, 4, seq_len, 8))
        v = rng.standard_normal((2, 4, seq_len, 8))
        mask = causal_mask(seq_len)

        expected, _, _ = scaled_dot_product_attention(q, k, v, mask)
        result = flash_attention_forward(q, k, v, mask=mask, block_size=4)

        np.testing.assert_allclose(result.output, expected, atol=1e-10)

    def test_non_square_seq_lengths(self) -> None:
        """seq_q != seq_k must produce correct results."""
        rng = np.random.default_rng(42)
        q = rng.standard_normal((2, 4, 5, 8))
        k = rng.standard_normal((2, 4, 9, 8))
        v = rng.standard_normal((2, 4, 9, 8))

        expected, _, _ = scaled_dot_product_attention(q, k, v)
        result = flash_attention_forward(q, k, v, block_size=3)

        np.testing.assert_allclose(result.output, expected, atol=1e-10)

    def test_forward_matches_pytorch(self) -> None:
        """Cross-verify against PyTorch's scaled_dot_product_attention."""
        rng = np.random.default_rng(42)
        q = rng.standard_normal((2, 4, 8, 16))
        k = rng.standard_normal((2, 4, 8, 16))
        v = rng.standard_normal((2, 4, 8, 16))

        result = flash_attention_forward(q, k, v, block_size=4)

        q_t = torch.from_numpy(q)
        k_t = torch.from_numpy(k)
        v_t = torch.from_numpy(v)
        output_torch = F.scaled_dot_product_attention(q_t, k_t, v_t).numpy()

        np.testing.assert_allclose(result.output, output_torch, atol=1e-6)


# =====================================================================
# Forward: block size invariance
# =====================================================================


class TestBlockSizeInvariance:
    def test_different_block_sizes_match(self) -> None:
        """Results must be identical across block sizes."""
        rng = np.random.default_rng(42)
        q = rng.standard_normal((2, 4, 12, 8))
        k = rng.standard_normal((2, 4, 12, 8))
        v = rng.standard_normal((2, 4, 12, 8))

        ref = flash_attention_forward(q, k, v, block_size=1)

        for bs in [2, 3, 4, 6, 12]:
            actual = flash_attention_forward(q, k, v, block_size=bs)
            np.testing.assert_allclose(
                actual.output, ref.output, atol=1e-10, err_msg=f"block_size={bs}"
            )

    def test_block_size_larger_than_seq(self) -> None:
        """block_size > sequence length must degenerate to standard attention."""
        rng = np.random.default_rng(42)
        q = rng.standard_normal((2, 4, 5, 8))
        k = rng.standard_normal((2, 4, 5, 8))
        v = rng.standard_normal((2, 4, 5, 8))

        expected, _, _ = scaled_dot_product_attention(q, k, v)
        result = flash_attention_forward(q, k, v, block_size=100)

        np.testing.assert_allclose(result.output, expected, atol=1e-10)

    def test_block_size_one(self) -> None:
        """block_size=1 (maximally tiled) must still be correct."""
        rng = np.random.default_rng(42)
        q = rng.standard_normal((1, 2, 6, 4))
        k = rng.standard_normal((1, 2, 6, 4))
        v = rng.standard_normal((1, 2, 6, 4))

        expected, _, _ = scaled_dot_product_attention(q, k, v)
        result = flash_attention_forward(q, k, v, block_size=1)

        np.testing.assert_allclose(result.output, expected, atol=1e-10)

    def test_seq_length_one(self) -> None:
        """Edge case: single-token sequence."""
        rng = np.random.default_rng(42)
        q = rng.standard_normal((2, 4, 1, 8))
        k = rng.standard_normal((2, 4, 1, 8))
        v = rng.standard_normal((2, 4, 1, 8))

        expected, _, _ = scaled_dot_product_attention(q, k, v)
        result = flash_attention_forward(q, k, v, block_size=4)

        np.testing.assert_allclose(result.output, expected, atol=1e-10)


# =====================================================================
# Forward: dropout
# =====================================================================


class TestFlashAttentionDropout:
    def test_eval_mode_matches_standard(self) -> None:
        """With training=False, dropout is skipped and output matches standard."""
        rng = np.random.default_rng(42)
        q = rng.standard_normal((2, 4, 8, 8))
        k = rng.standard_normal((2, 4, 8, 8))
        v = rng.standard_normal((2, 4, 8, 8))

        expected, _, _ = scaled_dot_product_attention(q, k, v)
        result = flash_attention_forward(
            q,
            k,
            v,
            block_size=4,
            dropout_rate=0.3,
            training=False,
        )
        np.testing.assert_allclose(result.output, expected, atol=1e-10)

    def test_training_dropout_is_deterministic(self) -> None:
        """Same rng seed must produce identical dropout output."""
        rng = np.random.default_rng(42)
        q = rng.standard_normal((2, 4, 8, 8))
        k = rng.standard_normal((2, 4, 8, 8))
        v = rng.standard_normal((2, 4, 8, 8))

        r1 = flash_attention_forward(
            q,
            k,
            v,
            block_size=4,
            dropout_rate=0.3,
            rng=np.random.default_rng(99),
            training=True,
        )
        r2 = flash_attention_forward(
            q,
            k,
            v,
            block_size=4,
            dropout_rate=0.3,
            rng=np.random.default_rng(99),
            training=True,
        )
        np.testing.assert_array_equal(r1.output, r2.output)

    def test_training_dropout_changes_output(self) -> None:
        """With training=True and non-zero dropout, output must differ from no-dropout."""
        rng = np.random.default_rng(42)
        q = rng.standard_normal((2, 4, 8, 8))
        k = rng.standard_normal((2, 4, 8, 8))
        v = rng.standard_normal((2, 4, 8, 8))

        no_drop = flash_attention_forward(q, k, v, block_size=4)
        with_drop = flash_attention_forward(
            q,
            k,
            v,
            block_size=4,
            dropout_rate=0.3,
            rng=np.random.default_rng(99),
            training=True,
        )
        assert not np.allclose(no_drop.output, with_drop.output)

    def test_dropout_requires_rng(self) -> None:
        """Must raise ValueError when dropout_rate > 0, training=True, but no rng."""
        rng = np.random.default_rng(42)
        q = rng.standard_normal((2, 4, 8, 8))
        k = rng.standard_normal((2, 4, 8, 8))
        v = rng.standard_normal((2, 4, 8, 8))

        with pytest.raises(ValueError, match="rng is required"):
            flash_attention_forward(
                q,
                k,
                v,
                block_size=4,
                dropout_rate=0.3,
                training=True,
            )


# =====================================================================
# Backward: correctness against standard backward
# =====================================================================


class TestFlashAttentionBackward:
    def test_backward_matches_standard(self) -> None:
        """Flash backward grads must match standard backward."""
        rng = np.random.default_rng(42)
        q = rng.standard_normal((2, 4, 8, 8))
        k = rng.standard_normal((2, 4, 8, 8))
        v = rng.standard_normal((2, 4, 8, 8))

        # Standard forward + backward.
        out_std, w_pre, w_post = scaled_dot_product_attention(q, k, v)
        grad_output = rng.standard_normal(out_std.shape)
        gq_std, gk_std, gv_std = scaled_dot_product_attention_backward(
            grad_output,
            q,
            k,
            v,
            w_pre,
            w_post,
        )

        # Flash forward + backward.
        fwd = flash_attention_forward(q, k, v, block_size=4)
        gq_flash, gk_flash, gv_flash = flash_attention_backward(
            grad_output,
            q,
            k,
            v,
            fwd.output,
            fwd.logsumexp,
            block_size=4,
        )

        np.testing.assert_allclose(gq_flash, gq_std, atol=1e-9)
        np.testing.assert_allclose(gk_flash, gk_std, atol=1e-9)
        np.testing.assert_allclose(gv_flash, gv_std, atol=1e-9)

    def test_backward_with_causal_mask(self) -> None:
        """Flash backward with causal mask must match standard backward."""
        rng = np.random.default_rng(42)
        seq_len = 8
        q = rng.standard_normal((2, 4, seq_len, 8))
        k = rng.standard_normal((2, 4, seq_len, 8))
        v = rng.standard_normal((2, 4, seq_len, 8))
        mask = causal_mask(seq_len)

        out_std, w_pre, w_post = scaled_dot_product_attention(q, k, v, mask)
        grad_output = rng.standard_normal(out_std.shape)
        gq_std, gk_std, gv_std = scaled_dot_product_attention_backward(
            grad_output,
            q,
            k,
            v,
            w_pre,
            w_post,
        )

        fwd = flash_attention_forward(q, k, v, mask=mask, block_size=4)
        gq_flash, gk_flash, gv_flash = flash_attention_backward(
            grad_output,
            q,
            k,
            v,
            fwd.output,
            fwd.logsumexp,
            mask=mask,
            block_size=4,
        )

        np.testing.assert_allclose(gq_flash, gq_std, atol=1e-9)
        np.testing.assert_allclose(gk_flash, gk_std, atol=1e-9)
        np.testing.assert_allclose(gv_flash, gv_std, atol=1e-9)

    def test_backward_non_square(self) -> None:
        """Flash backward with seq_q != seq_k."""
        rng = np.random.default_rng(42)
        q = rng.standard_normal((2, 4, 5, 8))
        k = rng.standard_normal((2, 4, 9, 8))
        v = rng.standard_normal((2, 4, 9, 8))

        out_std, w_pre, w_post = scaled_dot_product_attention(q, k, v)
        grad_output = rng.standard_normal(out_std.shape)
        gq_std, gk_std, gv_std = scaled_dot_product_attention_backward(
            grad_output,
            q,
            k,
            v,
            w_pre,
            w_post,
        )

        fwd = flash_attention_forward(q, k, v, block_size=3)
        gq_flash, gk_flash, gv_flash = flash_attention_backward(
            grad_output,
            q,
            k,
            v,
            fwd.output,
            fwd.logsumexp,
            block_size=3,
        )

        np.testing.assert_allclose(gq_flash, gq_std, atol=1e-9)
        np.testing.assert_allclose(gk_flash, gk_std, atol=1e-9)
        np.testing.assert_allclose(gv_flash, gv_std, atol=1e-9)

    def test_backward_block_size_one(self) -> None:
        """Maximally tiled backward must still be correct."""
        rng = np.random.default_rng(42)
        q = rng.standard_normal((1, 2, 4, 4))
        k = rng.standard_normal((1, 2, 4, 4))
        v = rng.standard_normal((1, 2, 4, 4))

        out_std, w_pre, w_post = scaled_dot_product_attention(q, k, v)
        grad_output = rng.standard_normal(out_std.shape)
        gq_std, gk_std, gv_std = scaled_dot_product_attention_backward(
            grad_output,
            q,
            k,
            v,
            w_pre,
            w_post,
        )

        fwd = flash_attention_forward(q, k, v, block_size=1)
        gq_flash, gk_flash, gv_flash = flash_attention_backward(
            grad_output,
            q,
            k,
            v,
            fwd.output,
            fwd.logsumexp,
            block_size=1,
        )

        np.testing.assert_allclose(gq_flash, gq_std, atol=1e-9)
        np.testing.assert_allclose(gk_flash, gk_std, atol=1e-9)
        np.testing.assert_allclose(gv_flash, gv_std, atol=1e-9)


# =====================================================================
# Backward: numerical gradient checks
# =====================================================================


class TestFlashAttentionNumericalGradient:
    """Verify flash backward via finite differences.

    Shapes are intentionally small — finite-difference loops are O(n)
    per element, so keeping tensors tiny keeps the test suite fast.
    """

    def _forward_sum(
        self,
        q: npt.NDArray[np.float64],
        k: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
        block_size: int = 2,
    ) -> float:
        result = flash_attention_forward(q, k, v, block_size=block_size)
        return float(np.sum(result.output))

    def test_backward_numerical_gradient_q(self) -> None:
        rng = np.random.default_rng(42)
        q = rng.standard_normal((1, 2, 3, 4))
        k = rng.standard_normal((1, 2, 3, 4))
        v = rng.standard_normal((1, 2, 3, 4))

        fwd = flash_attention_forward(q, k, v, block_size=2)
        grad_output = np.ones_like(fwd.output)
        grad_q, _, _ = flash_attention_backward(
            grad_output,
            q,
            k,
            v,
            fwd.output,
            fwd.logsumexp,
            block_size=2,
        )

        eps = 1e-5
        num_grad = np.zeros_like(q)
        for idx in np.ndindex(q.shape):
            q_plus = q.copy()
            q_minus = q.copy()
            q_plus[idx] += eps
            q_minus[idx] -= eps
            num_grad[idx] = (self._forward_sum(q_plus, k, v) - self._forward_sum(q_minus, k, v)) / (
                2 * eps
            )

        np.testing.assert_allclose(grad_q, num_grad, atol=1e-4)

    def test_backward_numerical_gradient_k(self) -> None:
        rng = np.random.default_rng(42)
        q = rng.standard_normal((1, 2, 3, 4))
        k = rng.standard_normal((1, 2, 3, 4))
        v = rng.standard_normal((1, 2, 3, 4))

        fwd = flash_attention_forward(q, k, v, block_size=2)
        grad_output = np.ones_like(fwd.output)
        _, grad_k, _ = flash_attention_backward(
            grad_output,
            q,
            k,
            v,
            fwd.output,
            fwd.logsumexp,
            block_size=2,
        )

        eps = 1e-5
        num_grad = np.zeros_like(k)
        for idx in np.ndindex(k.shape):
            k_plus = k.copy()
            k_minus = k.copy()
            k_plus[idx] += eps
            k_minus[idx] -= eps
            num_grad[idx] = (self._forward_sum(q, k_plus, v) - self._forward_sum(q, k_minus, v)) / (
                2 * eps
            )

        np.testing.assert_allclose(grad_k, num_grad, atol=1e-4)

    def test_backward_numerical_gradient_v(self) -> None:
        rng = np.random.default_rng(42)
        q = rng.standard_normal((1, 2, 3, 4))
        k = rng.standard_normal((1, 2, 3, 4))
        v = rng.standard_normal((1, 2, 3, 4))

        fwd = flash_attention_forward(q, k, v, block_size=2)
        grad_output = np.ones_like(fwd.output)
        _, _, grad_v = flash_attention_backward(
            grad_output,
            q,
            k,
            v,
            fwd.output,
            fwd.logsumexp,
            block_size=2,
        )

        eps = 1e-5
        num_grad = np.zeros_like(v)
        for idx in np.ndindex(v.shape):
            v_plus = v.copy()
            v_minus = v.copy()
            v_plus[idx] += eps
            v_minus[idx] -= eps
            num_grad[idx] = (self._forward_sum(q, k, v_plus) - self._forward_sum(q, k, v_minus)) / (
                2 * eps
            )

        np.testing.assert_allclose(grad_v, num_grad, atol=1e-4)


# =====================================================================
# Backward: dropout
# =====================================================================


class TestFlashAttentionBackwardDropout:
    def test_backward_with_dropout_eval(self) -> None:
        """Backward with dropout in eval mode must match standard (no dropout)."""
        rng = np.random.default_rng(42)
        q = rng.standard_normal((2, 4, 8, 8))
        k = rng.standard_normal((2, 4, 8, 8))
        v = rng.standard_normal((2, 4, 8, 8))

        out_std, w_pre, w_post = scaled_dot_product_attention(q, k, v)
        grad_output = rng.standard_normal(out_std.shape)
        gq_std, gk_std, gv_std = scaled_dot_product_attention_backward(
            grad_output,
            q,
            k,
            v,
            w_pre,
            w_post,
        )

        fwd = flash_attention_forward(
            q,
            k,
            v,
            block_size=4,
            dropout_rate=0.3,
            training=False,
        )
        gq_flash, gk_flash, gv_flash = flash_attention_backward(
            grad_output,
            q,
            k,
            v,
            fwd.output,
            fwd.logsumexp,
            block_size=4,
            dropout_rate=0.3,
            training=False,
        )

        np.testing.assert_allclose(gq_flash, gq_std, atol=1e-9)
        np.testing.assert_allclose(gk_flash, gk_std, atol=1e-9)
        np.testing.assert_allclose(gv_flash, gv_std, atol=1e-9)

    def _dropout_forward_sum(
        self,
        q: npt.NDArray[np.float64],
        k: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
    ) -> float:
        """Deterministic forward sum with dropout for finite-difference checks.

        Uses ``default_rng(99)`` so that ``base_seed`` — the first
        ``rng.integers`` call — is identical across perturbations.
        """
        result = flash_attention_forward(
            q,
            k,
            v,
            block_size=2,
            dropout_rate=0.3,
            rng=np.random.default_rng(99),
            training=True,
        )
        return float(np.sum(result.output))

    def test_backward_with_dropout_numerical_q(self) -> None:
        """Numerical gradient check for grad_q with dropout (training=True)."""
        rng = np.random.default_rng(42)
        # Shapes intentionally small for finite-difference speed.
        q = rng.standard_normal((1, 2, 3, 4))
        k = rng.standard_normal((1, 2, 3, 4))
        v = rng.standard_normal((1, 2, 3, 4))

        fwd = flash_attention_forward(
            q,
            k,
            v,
            block_size=2,
            dropout_rate=0.3,
            rng=np.random.default_rng(99),
            training=True,
        )
        grad_output = np.ones_like(fwd.output)
        grad_q, _, _ = flash_attention_backward(
            grad_output,
            q,
            k,
            v,
            fwd.output,
            fwd.logsumexp,
            block_size=2,
            dropout_rate=0.3,
            base_seed=fwd.base_seed,
            training=True,
        )

        eps = 1e-5
        num_grad = np.zeros_like(q)
        for idx in np.ndindex(q.shape):
            q_plus = q.copy()
            q_minus = q.copy()
            q_plus[idx] += eps
            q_minus[idx] -= eps
            num_grad[idx] = (
                self._dropout_forward_sum(q_plus, k, v) - self._dropout_forward_sum(q_minus, k, v)
            ) / (2 * eps)

        np.testing.assert_allclose(grad_q, num_grad, atol=1e-4)

    def test_backward_with_dropout_numerical_k(self) -> None:
        """Numerical gradient check for grad_k with dropout (training=True)."""
        rng = np.random.default_rng(42)
        q = rng.standard_normal((1, 2, 3, 4))
        k = rng.standard_normal((1, 2, 3, 4))
        v = rng.standard_normal((1, 2, 3, 4))

        fwd = flash_attention_forward(
            q,
            k,
            v,
            block_size=2,
            dropout_rate=0.3,
            rng=np.random.default_rng(99),
            training=True,
        )
        grad_output = np.ones_like(fwd.output)
        _, grad_k, _ = flash_attention_backward(
            grad_output,
            q,
            k,
            v,
            fwd.output,
            fwd.logsumexp,
            block_size=2,
            dropout_rate=0.3,
            base_seed=fwd.base_seed,
            training=True,
        )

        eps = 1e-5
        num_grad = np.zeros_like(k)
        for idx in np.ndindex(k.shape):
            k_plus = k.copy()
            k_minus = k.copy()
            k_plus[idx] += eps
            k_minus[idx] -= eps
            num_grad[idx] = (
                self._dropout_forward_sum(q, k_plus, v) - self._dropout_forward_sum(q, k_minus, v)
            ) / (2 * eps)

        np.testing.assert_allclose(grad_k, num_grad, atol=1e-4)

    def test_backward_with_dropout_numerical_v(self) -> None:
        """Numerical gradient check for grad_v with dropout (training=True)."""
        rng = np.random.default_rng(42)
        q = rng.standard_normal((1, 2, 3, 4))
        k = rng.standard_normal((1, 2, 3, 4))
        v = rng.standard_normal((1, 2, 3, 4))

        fwd = flash_attention_forward(
            q,
            k,
            v,
            block_size=2,
            dropout_rate=0.3,
            rng=np.random.default_rng(99),
            training=True,
        )
        grad_output = np.ones_like(fwd.output)
        _, _, grad_v = flash_attention_backward(
            grad_output,
            q,
            k,
            v,
            fwd.output,
            fwd.logsumexp,
            block_size=2,
            dropout_rate=0.3,
            base_seed=fwd.base_seed,
            training=True,
        )

        eps = 1e-5
        num_grad = np.zeros_like(v)
        for idx in np.ndindex(v.shape):
            v_plus = v.copy()
            v_minus = v.copy()
            v_plus[idx] += eps
            v_minus[idx] -= eps
            num_grad[idx] = (
                self._dropout_forward_sum(q, k, v_plus) - self._dropout_forward_sum(q, k, v_minus)
            ) / (2 * eps)

        np.testing.assert_allclose(grad_v, num_grad, atol=1e-4)


# =====================================================================
# Validation errors
# =====================================================================


class TestFlashAttentionValidation:
    def test_block_size_zero_raises(self) -> None:
        """block_size=0 must raise ValueError."""
        rng = np.random.default_rng(42)
        q = rng.standard_normal((1, 1, 4, 8))
        k = rng.standard_normal((1, 1, 4, 8))
        v = rng.standard_normal((1, 1, 4, 8))

        with pytest.raises(ValueError, match="block_size must be positive"):
            flash_attention_forward(q, k, v, block_size=0)

    def test_block_size_negative_raises(self) -> None:
        """Negative block_size must raise ValueError."""
        rng = np.random.default_rng(42)
        q = rng.standard_normal((1, 1, 4, 8))
        k = rng.standard_normal((1, 1, 4, 8))
        v = rng.standard_normal((1, 1, 4, 8))

        with pytest.raises(ValueError, match="block_size must be positive"):
            flash_attention_forward(q, k, v, block_size=-1)

    def test_mismatched_dk_raises(self) -> None:
        """q and k with different d_k must raise ShapeMismatchError."""
        rng = np.random.default_rng(42)
        q = rng.standard_normal((1, 1, 4, 8))
        k = rng.standard_normal((1, 1, 4, 16))
        v = rng.standard_normal((1, 1, 4, 8))

        with pytest.raises(ShapeMismatchError, match="same d_k"):
            flash_attention_forward(q, k, v, block_size=2)

    def test_mismatched_seq_k_raises(self) -> None:
        """k and v with different seq_k must raise ShapeMismatchError."""
        rng = np.random.default_rng(42)
        q = rng.standard_normal((1, 1, 4, 8))
        k = rng.standard_normal((1, 1, 6, 8))
        v = rng.standard_normal((1, 1, 4, 8))

        with pytest.raises(ShapeMismatchError, match="same seq_k"):
            flash_attention_forward(q, k, v, block_size=2)

    def test_non_broadcastable_batch_raises(self) -> None:
        """Incompatible batch dimensions must raise ShapeMismatchError."""
        rng = np.random.default_rng(42)
        q = rng.standard_normal((2, 4, 4, 8))
        k = rng.standard_normal((3, 4, 4, 8))
        v = rng.standard_normal((3, 4, 4, 8))

        with pytest.raises(ShapeMismatchError, match="not broadcastable"):
            flash_attention_forward(q, k, v, block_size=2)
