"""Tests for LayerNorm with PyTorch cross-verification."""

import numpy as np
import pytest
import torch

from rawformer.exceptions import ForwardNotCalledError
from rawformer.layers.norm import LayerNorm


class TestLayerNorm:
    def test_forward_output_shape(self) -> None:
        norm = LayerNorm(8)
        x = np.random.default_rng(0).standard_normal((2, 5, 8))
        result = norm.forward(x)
        assert result.shape == (2, 5, 8)

    def test_forward_zero_mean_unit_var(self) -> None:
        """After normalization (with gamma=1, beta=0), each position should
        have approximately zero mean and unit variance."""
        norm = LayerNorm(16)
        x = np.random.default_rng(0).standard_normal((4, 10, 16))
        result = norm.forward(x)
        np.testing.assert_allclose(np.mean(result, axis=-1), 0.0, atol=1e-6)
        np.testing.assert_allclose(np.var(result, axis=-1), 1.0, atol=1e-3)

    def test_backward_raises_without_forward(self) -> None:
        norm = LayerNorm(8)
        with pytest.raises(ForwardNotCalledError):
            norm.backward(np.ones((2, 5, 8)))

    def test_forward_matches_pytorch(self) -> None:
        """Cross-verify forward pass against torch.nn.LayerNorm."""
        d_model = 8
        norm = LayerNorm(d_model)
        x_np = np.random.default_rng(42).standard_normal((2, 5, d_model))

        torch_norm = torch.nn.LayerNorm(d_model, dtype=torch.float64)
        torch_norm.weight.data = torch.from_numpy(norm.gamma.copy())
        torch_norm.bias.data = torch.from_numpy(norm.beta.copy())

        result_np = norm.forward(x_np)
        x_torch = torch.from_numpy(x_np)
        result_torch = torch_norm(x_torch).detach().numpy()

        np.testing.assert_allclose(result_np, result_torch, atol=1e-6)

    def test_backward_matches_pytorch(self) -> None:
        """Cross-verify backward pass against torch.nn.LayerNorm."""
        d_model = 8
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((2, 5, d_model))
        grad_np = rng.standard_normal((2, 5, d_model))

        norm = LayerNorm(d_model)
        norm.forward(x_np)
        grad_input = norm.backward(grad_np)

        torch_norm = torch.nn.LayerNorm(d_model, dtype=torch.float64)
        torch_norm.weight.data = torch.from_numpy(norm.gamma.copy())
        torch_norm.bias.data = torch.from_numpy(norm.beta.copy())
        x_torch = torch.from_numpy(x_np).requires_grad_(True)
        out_torch = torch_norm(x_torch)
        out_torch.backward(torch.from_numpy(grad_np))
        assert x_torch.grad is not None
        grad_torch = x_torch.grad.numpy()

        np.testing.assert_allclose(grad_input, grad_torch, atol=1e-5)
