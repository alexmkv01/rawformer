"""Tests for LayerNorm and RMSNorm with PyTorch cross-verification."""

import numpy as np
import pytest
import torch

from rawformer.core.exceptions import ForwardNotCalledError
from rawformer.layers.norm import LayerNorm, RMSNorm


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


class TestRMSNorm:
    def test_forward_output_shape(self) -> None:
        norm = RMSNorm(8)
        x = np.random.default_rng(0).standard_normal((2, 5, 8))
        result = norm.forward(x)
        assert result.shape == (2, 5, 8)

    def test_forward_unit_rms(self) -> None:
        """After normalization (with gamma=1), each position should have RMS ~ 1."""
        norm = RMSNorm(16)
        x = np.random.default_rng(0).standard_normal((4, 10, 16))
        result = norm.forward(x)
        rms = np.sqrt(np.mean(result**2, axis=-1))
        np.testing.assert_allclose(rms, 1.0, atol=1e-4)

    def test_no_beta_parameter(self) -> None:
        """RMSNorm has no learned bias, unlike LayerNorm."""
        norm = RMSNorm(8)
        assert not hasattr(norm, "beta")
        assert not hasattr(norm, "_beta")

    def test_backward_raises_without_forward(self) -> None:
        norm = RMSNorm(8)
        with pytest.raises(ForwardNotCalledError):
            norm.backward(np.ones((2, 5, 8)))

    def test_forward_matches_pytorch(self) -> None:
        """Cross-verify forward pass against a manual PyTorch RMSNorm."""
        d_model = 8
        eps = 1e-5
        norm = RMSNorm(d_model, eps=eps)
        x_np = np.random.default_rng(42).standard_normal((2, 5, d_model))

        result_np = norm.forward(x_np)

        x_torch = torch.from_numpy(x_np)
        rms = torch.sqrt(x_torch.pow(2).mean(-1, keepdim=True) + eps)
        gamma_torch = torch.from_numpy(norm.gamma.copy())
        result_torch = (gamma_torch * x_torch / rms).detach().numpy()

        np.testing.assert_allclose(result_np, result_torch, atol=1e-6)

    def test_backward_matches_pytorch(self) -> None:
        """Cross-verify backward pass against PyTorch autograd."""
        d_model = 8
        eps = 1e-5
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((2, 5, d_model))
        grad_np = rng.standard_normal((2, 5, d_model))

        norm = RMSNorm(d_model, eps=eps)
        norm.forward(x_np)
        grad_input = norm.backward(grad_np)

        gamma_torch = torch.from_numpy(norm.gamma.copy())
        x_torch = torch.from_numpy(x_np).requires_grad_(True)
        rms = torch.sqrt(x_torch.pow(2).mean(-1, keepdim=True) + eps)
        out_torch = gamma_torch * x_torch / rms
        out_torch.backward(torch.from_numpy(grad_np))  # type: ignore[no-untyped-call]
        assert x_torch.grad is not None
        grad_torch = x_torch.grad.numpy()

        np.testing.assert_allclose(grad_input, grad_torch, atol=1e-5)
