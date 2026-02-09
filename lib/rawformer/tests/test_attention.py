"""Tests for scaled dot-product attention and multi-head attention."""

import numpy as np
import torch
import torch.nn.functional as F

from rawformer.attention.multi_head import MultiHeadAttention
from rawformer.attention.scaled_dot_product import (
    scaled_dot_product_attention,
    scaled_dot_product_attention_backward,
)
from rawformer.transformer.decoder import causal_mask


class TestScaledDotProductAttention:
    def test_output_shape(self) -> None:
        rng = np.random.default_rng(0)
        q = rng.standard_normal((2, 4, 5, 8))
        k = rng.standard_normal((2, 4, 6, 8))
        v = rng.standard_normal((2, 4, 6, 10))
        output, weights = scaled_dot_product_attention(q, k, v)
        assert output.shape == (2, 4, 5, 10)
        assert weights.shape == (2, 4, 5, 6)

    def test_weights_sum_to_one(self) -> None:
        rng = np.random.default_rng(0)
        q = rng.standard_normal((2, 4, 5, 8))
        k = rng.standard_normal((2, 4, 6, 8))
        v = rng.standard_normal((2, 4, 6, 10))
        _, weights = scaled_dot_product_attention(q, k, v)
        row_sums = np.sum(weights, axis=-1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_causal_mask_blocks_future(self) -> None:
        """With a causal mask, attention weights for future positions should be ~0."""
        rng = np.random.default_rng(0)
        seq_len = 5
        q = rng.standard_normal((1, 1, seq_len, 8))
        k = rng.standard_normal((1, 1, seq_len, 8))
        v = rng.standard_normal((1, 1, seq_len, 8))
        mask = causal_mask(seq_len)
        _, weights = scaled_dot_product_attention(q, k, v, mask)
        # upper triangle (future positions) should be ~0
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert weights[0, 0, i, j] < 1e-6

    def test_forward_matches_pytorch(self) -> None:
        """Cross-verify against PyTorch's scaled_dot_product_attention."""
        rng = np.random.default_rng(42)
        q = rng.standard_normal((2, 4, 5, 8))
        k = rng.standard_normal((2, 4, 5, 8))
        v = rng.standard_normal((2, 4, 5, 8))

        output_np, _ = scaled_dot_product_attention(q, k, v)

        q_t = torch.from_numpy(q)
        k_t = torch.from_numpy(k)
        v_t = torch.from_numpy(v)
        output_torch = F.scaled_dot_product_attention(q_t, k_t, v_t).numpy()

        np.testing.assert_allclose(output_np, output_torch, atol=1e-6)

    def test_backward_numerical_gradient(self) -> None:
        """Verify backward pass via finite differences on the query."""
        rng = np.random.default_rng(42)
        q = rng.standard_normal((1, 2, 3, 4))
        k = rng.standard_normal((1, 2, 3, 4))
        v = rng.standard_normal((1, 2, 3, 4))

        output, weights = scaled_dot_product_attention(q, k, v)
        grad_output = np.ones_like(output)
        grad_q, _, _ = scaled_dot_product_attention_backward(grad_output, q, k, v, weights)

        eps = 1e-5
        num_grad_q = np.zeros_like(q)
        for idx in np.ndindex(q.shape):
            q_plus = q.copy()
            q_minus = q.copy()
            q_plus[idx] += eps
            q_minus[idx] -= eps
            out_plus, _ = scaled_dot_product_attention(q_plus, k, v)
            out_minus, _ = scaled_dot_product_attention(q_minus, k, v)
            num_grad_q[idx] = (np.sum(out_plus) - np.sum(out_minus)) / (2 * eps)

        np.testing.assert_allclose(grad_q, num_grad_q, atol=1e-4)


class TestMultiHeadAttention:
    def test_forward_output_shape(self) -> None:
        mha = MultiHeadAttention(16, 4, rng=np.random.default_rng(0))
        x = np.random.default_rng(0).standard_normal((2, 5, 16))
        output = mha.forward(x, x, x)
        assert output.shape == (2, 5, 16)

    def test_forward_matches_pytorch(self) -> None:
        """Cross-verify forward pass against torch.nn.MultiheadAttention."""
        d_model, n_heads = 16, 4
        rng = np.random.default_rng(42)
        mha = MultiHeadAttention(d_model, n_heads, rng=rng)

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
        mha = MultiHeadAttention(16, 4, rng=np.random.default_rng(0))
        q = np.random.default_rng(0).standard_normal((2, 5, 16))
        kv = np.random.default_rng(1).standard_normal((2, 8, 16))
        output = mha.forward(q, kv, kv)
        assert output.shape == (2, 5, 16)

    def test_backward_output_shapes(self) -> None:
        mha = MultiHeadAttention(16, 4, rng=np.random.default_rng(0))
        x = np.random.default_rng(0).standard_normal((2, 5, 16))
        mha.forward(x, x, x)
        grad = np.ones((2, 5, 16))
        grad_q, grad_k, grad_v = mha.backward(grad)
        assert grad_q.shape == (2, 5, 16)
        assert grad_k.shape == (2, 5, 16)
        assert grad_v.shape == (2, 5, 16)
