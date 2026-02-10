"""Tests for scaled dot-product attention and multi-head attention."""

import numpy as np
import torch
import torch.nn.functional as F

from rawformer.attention.multi_head import MultiHeadAttention
from rawformer.attention.scaled_dot_product import (
    scaled_dot_product_attention,
    scaled_dot_product_attention_backward,
)
from rawformer.layers.dropout import Dropout
from rawformer.transformer.decoder import causal_mask


class TestScaledDotProductAttention:
    def test_output_shape(self) -> None:
        rng = np.random.default_rng(0)
        q = rng.standard_normal((2, 4, 5, 8))
        k = rng.standard_normal((2, 4, 6, 8))
        v = rng.standard_normal((2, 4, 6, 10))
        output, weights_pre, weights_post = scaled_dot_product_attention(q, k, v)
        assert output.shape == (2, 4, 5, 10)
        assert weights_pre.shape == (2, 4, 5, 6)
        assert weights_post.shape == (2, 4, 5, 6)

    def test_weights_sum_to_one(self) -> None:
        rng = np.random.default_rng(0)
        q = rng.standard_normal((2, 4, 5, 8))
        k = rng.standard_normal((2, 4, 6, 8))
        v = rng.standard_normal((2, 4, 6, 10))
        _, weights_pre, _ = scaled_dot_product_attention(q, k, v)
        row_sums = np.sum(weights_pre, axis=-1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_no_dropout_weights_equal(self) -> None:
        """Without dropout, pre and post weights should be identical."""
        rng = np.random.default_rng(0)
        q = rng.standard_normal((2, 4, 5, 8))
        k = rng.standard_normal((2, 4, 5, 8))
        v = rng.standard_normal((2, 4, 5, 8))
        _, weights_pre, weights_post = scaled_dot_product_attention(q, k, v)
        np.testing.assert_array_equal(weights_pre, weights_post)

    def test_causal_mask_blocks_future(self) -> None:
        """With a causal mask, attention weights for future positions should be ~0."""
        rng = np.random.default_rng(0)
        seq_len = 5
        q = rng.standard_normal((1, 1, seq_len, 8))
        k = rng.standard_normal((1, 1, seq_len, 8))
        v = rng.standard_normal((1, 1, seq_len, 8))
        mask = causal_mask(seq_len)
        _, weights_pre, _ = scaled_dot_product_attention(q, k, v, mask)
        # upper triangle (future positions) should be ~0
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert weights_pre[0, 0, i, j] < 1e-6

    def test_forward_matches_pytorch(self) -> None:
        """Cross-verify against PyTorch's scaled_dot_product_attention."""
        rng = np.random.default_rng(42)
        q = rng.standard_normal((2, 4, 5, 8))
        k = rng.standard_normal((2, 4, 5, 8))
        v = rng.standard_normal((2, 4, 5, 8))

        output_np, _, _ = scaled_dot_product_attention(q, k, v)

        q_t = torch.from_numpy(q)
        k_t = torch.from_numpy(k)
        v_t = torch.from_numpy(v)
        output_torch = F.scaled_dot_product_attention(q_t, k_t, v_t).numpy()

        np.testing.assert_allclose(output_np, output_torch, atol=1e-6)

    def test_backward_numerical_gradient_q(self) -> None:
        """Verify grad_q via finite differences."""
        rng = np.random.default_rng(42)
        q = rng.standard_normal((1, 2, 3, 4))
        k = rng.standard_normal((1, 2, 3, 4))
        v = rng.standard_normal((1, 2, 3, 4))

        output, weights_pre, weights_post = scaled_dot_product_attention(q, k, v)
        grad_output = np.ones_like(output)
        grad_q, _, _ = scaled_dot_product_attention_backward(
            grad_output, q, k, v, weights_pre, weights_post
        )

        eps = 1e-5
        num_grad_q = np.zeros_like(q)
        for idx in np.ndindex(q.shape):
            q_plus = q.copy()
            q_minus = q.copy()
            q_plus[idx] += eps
            q_minus[idx] -= eps
            out_plus, _, _ = scaled_dot_product_attention(q_plus, k, v)
            out_minus, _, _ = scaled_dot_product_attention(q_minus, k, v)
            num_grad_q[idx] = (np.sum(out_plus) - np.sum(out_minus)) / (2 * eps)

        np.testing.assert_allclose(grad_q, num_grad_q, atol=1e-4)

    def test_backward_numerical_gradient_k(self) -> None:
        """Verify grad_k via finite differences."""
        rng = np.random.default_rng(42)
        q = rng.standard_normal((1, 2, 3, 4))
        k = rng.standard_normal((1, 2, 3, 4))
        v = rng.standard_normal((1, 2, 3, 4))

        output, weights_pre, weights_post = scaled_dot_product_attention(q, k, v)
        grad_output = np.ones_like(output)
        _, grad_k, _ = scaled_dot_product_attention_backward(
            grad_output, q, k, v, weights_pre, weights_post
        )

        eps = 1e-5
        num_grad_k = np.zeros_like(k)
        for idx in np.ndindex(k.shape):
            k_plus = k.copy()
            k_minus = k.copy()
            k_plus[idx] += eps
            k_minus[idx] -= eps
            out_plus, _, _ = scaled_dot_product_attention(q, k_plus, v)
            out_minus, _, _ = scaled_dot_product_attention(q, k_minus, v)
            num_grad_k[idx] = (np.sum(out_plus) - np.sum(out_minus)) / (2 * eps)

        np.testing.assert_allclose(grad_k, num_grad_k, atol=1e-4)

    def test_backward_numerical_gradient_v(self) -> None:
        """Verify grad_v via finite differences."""
        rng = np.random.default_rng(42)
        q = rng.standard_normal((1, 2, 3, 4))
        k = rng.standard_normal((1, 2, 3, 4))
        v = rng.standard_normal((1, 2, 3, 4))

        output, weights_pre, weights_post = scaled_dot_product_attention(q, k, v)
        grad_output = np.ones_like(output)
        _, _, grad_v = scaled_dot_product_attention_backward(
            grad_output, q, k, v, weights_pre, weights_post
        )

        eps = 1e-5
        num_grad_v = np.zeros_like(v)
        for idx in np.ndindex(v.shape):
            v_plus = v.copy()
            v_minus = v.copy()
            v_plus[idx] += eps
            v_minus[idx] -= eps
            out_plus, _, _ = scaled_dot_product_attention(q, k, v_plus)
            out_minus, _, _ = scaled_dot_product_attention(q, k, v_minus)
            num_grad_v[idx] = (np.sum(out_plus) - np.sum(out_minus)) / (2 * eps)

        np.testing.assert_allclose(grad_v, num_grad_v, atol=1e-4)

    def test_backward_with_dropout_numerical_gradient(self) -> None:
        """Verify backward correctness when dropout is applied.

        Uses eval mode (dropout disabled) so finite differences are
        deterministic, but exercises the full pre/post weight path.
        Then verifies that with training mode on, the pre-dropout
        weights (softmax output) still sum to 1 while post-dropout
        weights do not.
        """
        rng = np.random.default_rng(42)
        q = rng.standard_normal((1, 2, 3, 4))
        k = rng.standard_normal((1, 2, 3, 4))
        v = rng.standard_normal((1, 2, 3, 4))

        dropout = Dropout(0.3, np.random.default_rng(99))

        # eval mode: dropout is identity, but exercises the code path
        dropout.training = False
        output, weights_pre, weights_post = scaled_dot_product_attention(q, k, v, dropout=dropout)
        np.testing.assert_array_equal(weights_pre, weights_post)

        grad_output = np.ones_like(output)
        grad_q, _grad_k, _grad_v = scaled_dot_product_attention_backward(
            grad_output, q, k, v, weights_pre, weights_post, dropout=dropout
        )

        eps = 1e-5
        num_grad_q = np.zeros_like(q)
        for idx in np.ndindex(q.shape):
            q_plus = q.copy()
            q_minus = q.copy()
            q_plus[idx] += eps
            q_minus[idx] -= eps
            out_plus, _, _ = scaled_dot_product_attention(q_plus, k, v, dropout=dropout)
            out_minus, _, _ = scaled_dot_product_attention(q_minus, k, v, dropout=dropout)
            num_grad_q[idx] = (np.sum(out_plus) - np.sum(out_minus)) / (2 * eps)

        np.testing.assert_allclose(grad_q, num_grad_q, atol=1e-4)

        # training mode: pre-dropout weights sum to 1, post-dropout differ
        dropout.training = True
        _, weights_pre_train, weights_post_train = scaled_dot_product_attention(
            q, k, v, dropout=dropout
        )
        row_sums_pre = np.sum(weights_pre_train, axis=-1)
        np.testing.assert_allclose(row_sums_pre, 1.0, atol=1e-6)
        assert not np.allclose(weights_pre_train, weights_post_train)


class TestMultiHeadAttention:
    def test_forward_output_shape(self) -> None:
        mha = MultiHeadAttention(16, 4, rng=np.random.default_rng(0), dropout_rate=0.0)
        x = np.random.default_rng(0).standard_normal((2, 5, 16))
        output = mha.forward(x, x, x)
        assert output.shape == (2, 5, 16)

    def test_forward_matches_pytorch(self) -> None:
        """Cross-verify forward pass against torch.nn.MultiheadAttention."""
        d_model, n_heads = 16, 4
        rng = np.random.default_rng(42)
        mha = MultiHeadAttention(d_model, n_heads, rng=rng, dropout_rate=0.0)

        torch_mha = torch.nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dtype=torch.float64
        )
        # copy weights: PyTorch packs Q,K,V into one in_proj_weight
        wq = mha.w_q.weights.T
        wk = mha.w_k.weights.T
        wv = mha.w_v.weights.T
        torch_mha.in_proj_weight.data = torch.from_numpy(
            np.concatenate([wq, wk, wv], axis=0).copy()
        )
        bq = mha.w_q.biases
        bk = mha.w_k.biases
        bv = mha.w_v.biases
        torch_mha.in_proj_bias.data = torch.from_numpy(np.concatenate([bq, bk, bv]).copy())
        torch_mha.out_proj.weight.data = torch.from_numpy(mha.w_o.weights.T.copy())
        torch_mha.out_proj.bias.data = torch.from_numpy(mha.w_o.biases.copy())

        x_np = np.random.default_rng(0).standard_normal((2, 5, d_model))
        result_np = mha.forward(x_np, x_np, x_np)

        x_torch = torch.from_numpy(x_np)
        result_torch, _ = torch_mha(x_torch, x_torch, x_torch, need_weights=False)

        np.testing.assert_allclose(result_np, result_torch.detach().numpy(), atol=1e-6)

    def test_cross_attention_output_shape(self) -> None:
        mha = MultiHeadAttention(16, 4, rng=np.random.default_rng(0), dropout_rate=0.0)
        q = np.random.default_rng(0).standard_normal((2, 5, 16))
        kv = np.random.default_rng(1).standard_normal((2, 8, 16))
        output = mha.forward(q, kv, kv)
        assert output.shape == (2, 5, 16)

    def test_backward_output_shapes(self) -> None:
        mha = MultiHeadAttention(16, 4, rng=np.random.default_rng(0), dropout_rate=0.0)
        x = np.random.default_rng(0).standard_normal((2, 5, 16))
        mha.forward(x, x, x)
        grad = np.ones((2, 5, 16))
        grad_q, grad_k, grad_v = mha.backward(grad)
        assert grad_q.shape == (2, 5, 16)
        assert grad_k.shape == (2, 5, 16)
        assert grad_v.shape == (2, 5, 16)
