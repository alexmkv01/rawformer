"""Tests for Rotary Position Embeddings (RoPE) with PyTorch cross-verification."""

import numpy as np
import pytest
import torch

from rawformer.attention.rope import (
    apply_rope,
    apply_rope_backward,
    build_rope_frequencies,
)


class TestBuildRopeFrequencies:
    def test_output_shapes(self) -> None:
        cos, sin = build_rope_frequencies(seq_len=32, d_k=16)
        assert cos.shape == (32, 8)
        assert sin.shape == (32, 8)

    def test_position_zero_cos_is_one(self) -> None:
        cos, _ = build_rope_frequencies(seq_len=10, d_k=16)
        np.testing.assert_allclose(cos[0], 1.0, atol=1e-10)

    def test_position_zero_sin_is_zero(self) -> None:
        _, sin = build_rope_frequencies(seq_len=10, d_k=16)
        np.testing.assert_allclose(sin[0], 0.0, atol=1e-10)

    def test_odd_d_k_raises(self) -> None:
        with pytest.raises(ValueError, match="even"):
            build_rope_frequencies(seq_len=10, d_k=7)


class TestApplyRope:
    def test_output_shape(self) -> None:
        d_k = 16
        seq_len = 10
        cos, sin = build_rope_frequencies(seq_len, d_k)
        x = np.random.default_rng(0).standard_normal((2, 4, seq_len, d_k))
        result = apply_rope(x, cos, sin)
        assert result.shape == (2, 4, seq_len, d_k)

    def test_position_zero_is_identity(self) -> None:
        """At position 0 (cos=1, sin=0), rotation should be identity."""
        d_k = 16
        cos, sin = build_rope_frequencies(seq_len=1, d_k=d_k)
        x = np.random.default_rng(42).standard_normal((2, 4, 1, d_k))
        result = apply_rope(x, cos, sin)
        np.testing.assert_allclose(result, x, atol=1e-12)

    def test_rotation_preserves_norm(self) -> None:
        """Rotation is orthogonal, so norms must be preserved."""
        d_k = 16
        seq_len = 8
        cos, sin = build_rope_frequencies(seq_len, d_k)
        x = np.random.default_rng(0).standard_normal((2, 4, seq_len, d_k))
        result = apply_rope(x, cos, sin)
        norms_before = np.linalg.norm(x, axis=-1)
        norms_after = np.linalg.norm(result, axis=-1)
        np.testing.assert_allclose(norms_after, norms_before, atol=1e-10)

    def test_forward_matches_pytorch(self) -> None:
        """Cross-verify against a reference PyTorch RoPE (even/odd interleave)."""
        d_k = 16
        seq_len = 8
        cos_np, sin_np = build_rope_frequencies(seq_len, d_k)
        x_np = np.random.default_rng(42).standard_normal((2, 4, seq_len, d_k))

        result_np = apply_rope(x_np, cos_np, sin_np)

        # PyTorch reference: same even/odd convention
        x_t = torch.from_numpy(x_np)
        cos_t = torch.from_numpy(cos_np)
        sin_t = torch.from_numpy(sin_np)
        x_even = x_t[..., 0::2]
        x_odd = x_t[..., 1::2]
        out_even = x_even * cos_t - x_odd * sin_t
        out_odd = x_even * sin_t + x_odd * cos_t
        result_t = torch.empty_like(x_t)
        result_t[..., 0::2] = out_even
        result_t[..., 1::2] = out_odd

        np.testing.assert_allclose(result_np, result_t.numpy(), atol=1e-12)

    def test_relative_position_dot_product(self) -> None:
        """The dot product rope(q, pos=m) · rope(k, pos=n) depends only on m-n.

        This is the core property that makes RoPE useful: it gives
        relative-position awareness from absolute-position rotations.
        """
        d_k = 16
        max_len = 32
        cos, sin = build_rope_frequencies(max_len, d_k)
        rng = np.random.default_rng(42)

        q = rng.standard_normal((1, 1, 1, d_k))
        k = rng.standard_normal((1, 1, 1, d_k))

        # Compute dot(rope(q, m), rope(k, n)) for several (m, n) pairs
        # with the same relative offset m - n.
        offset = 5
        dots = []
        for m in range(0, 20, 4):
            n = m + offset
            q_rot = apply_rope(q, cos[m : m + 1], sin[m : m + 1])
            k_rot = apply_rope(k, cos[n : n + 1], sin[n : n + 1])
            dots.append(float(np.sum(q_rot * k_rot)))

        # All dot products should be identical since m - n is constant.
        np.testing.assert_allclose(dots, dots[0], atol=1e-10)


class TestApplyRopeBackward:
    def test_backward_shape(self) -> None:
        d_k = 16
        seq_len = 8
        cos, sin = build_rope_frequencies(seq_len, d_k)
        grad = np.random.default_rng(0).standard_normal((2, 4, seq_len, d_k))
        result = apply_rope_backward(grad, cos, sin)
        assert result.shape == (2, 4, seq_len, d_k)

    def test_roundtrip_is_identity(self) -> None:
        """apply_rope_backward(apply_rope(x)) == x (inverse rotation)."""
        d_k = 16
        seq_len = 8
        cos, sin = build_rope_frequencies(seq_len, d_k)
        x = np.random.default_rng(42).standard_normal((2, 4, seq_len, d_k))
        rotated = apply_rope(x, cos, sin)
        recovered = apply_rope_backward(rotated, cos, sin)
        np.testing.assert_allclose(recovered, x, atol=1e-12)

    def test_backward_numerical_gradient(self) -> None:
        """Finite-difference check on apply_rope."""
        # Intentionally small shape — brute-force finite diff is O(n) forward passes.
        d_k = 8
        seq_len = 4
        cos, sin = build_rope_frequencies(seq_len, d_k)
        x = np.random.default_rng(42).standard_normal((1, 2, seq_len, d_k))

        eps = 1e-5
        grad_output = np.random.default_rng(0).standard_normal(x.shape)

        analytic = apply_rope_backward(grad_output, cos, sin)

        numerical = np.zeros_like(x)
        for idx in np.ndindex(x.shape):
            x_plus = x.copy()
            x_plus[idx] += eps
            x_minus = x.copy()
            x_minus[idx] -= eps
            f_plus = apply_rope(x_plus, cos, sin)
            f_minus = apply_rope(x_minus, cos, sin)
            numerical[idx] = np.sum((f_plus - f_minus) / (2 * eps) * grad_output)

        np.testing.assert_allclose(analytic, numerical, atol=1e-7)

    def test_backward_matches_pytorch(self) -> None:
        """Cross-verify backward pass against PyTorch autograd."""
        d_k = 16
        seq_len = 8
        cos_np, sin_np = build_rope_frequencies(seq_len, d_k)
        x_np = np.random.default_rng(42).standard_normal((2, 4, seq_len, d_k))
        grad_np = np.random.default_rng(0).standard_normal((2, 4, seq_len, d_k))

        grad_x_np = apply_rope_backward(grad_np, cos_np, sin_np)

        # PyTorch autograd reference
        x_t = torch.from_numpy(x_np).requires_grad_(True)
        cos_t = torch.from_numpy(cos_np)
        sin_t = torch.from_numpy(sin_np)
        x_even = x_t[..., 0::2]
        x_odd = x_t[..., 1::2]
        out_even = x_even * cos_t - x_odd * sin_t
        out_odd = x_even * sin_t + x_odd * cos_t
        out_t = torch.empty_like(x_t)
        out_t[..., 0::2] = out_even
        out_t[..., 1::2] = out_odd
        out_t.backward(torch.from_numpy(grad_np))  # type: ignore[no-untyped-call]
        assert x_t.grad is not None

        np.testing.assert_allclose(grad_x_np, x_t.grad.numpy(), atol=1e-10)
