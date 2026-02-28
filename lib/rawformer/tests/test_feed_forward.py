"""Tests for PositionWiseFeedForward and SwiGLUFeedForward."""

from typing import NamedTuple

import numpy as np
import pytest
import torch

from rawformer.core.exceptions import ForwardNotCalledError
from rawformer.transformer.feed_forward import PositionWiseFeedForward, SwiGLUFeedForward


class _TorchSwiGLU(NamedTuple):
    w_gate: torch.nn.Linear
    w_up: torch.nn.Linear
    w_down: torch.nn.Linear


def _build_torch_swiglu(ffn: SwiGLUFeedForward) -> _TorchSwiGLU:
    """Build three torch.nn.Linear layers with weights copied from a SwiGLUFeedForward.

    Args:
        ffn: Source SwiGLUFeedForward whose weights are copied.

    Returns:
        Named tuple of (w_gate, w_up, w_down) torch.nn.Linear layers.
    """
    d_model = ffn.w_gate.weights.shape[0]
    d_ff = ffn.w_gate.weights.shape[1]

    w_gate_t = torch.nn.Linear(d_model, d_ff, dtype=torch.float64)
    w_up_t = torch.nn.Linear(d_model, d_ff, dtype=torch.float64)
    w_down_t = torch.nn.Linear(d_ff, d_model, dtype=torch.float64)

    w_gate_t.weight.data = torch.from_numpy(ffn.w_gate.weights.T.copy())
    w_gate_t.bias.data = torch.from_numpy(ffn.w_gate.biases.copy())
    w_up_t.weight.data = torch.from_numpy(ffn.w_up.weights.T.copy())
    w_up_t.bias.data = torch.from_numpy(ffn.w_up.biases.copy())
    w_down_t.weight.data = torch.from_numpy(ffn.w_down.weights.T.copy())
    w_down_t.bias.data = torch.from_numpy(ffn.w_down.biases.copy())

    return _TorchSwiGLU(w_gate=w_gate_t, w_up=w_up_t, w_down=w_down_t)


class TestPositionWiseFeedForward:
    def test_forward_output_shape(self) -> None:
        ffn = PositionWiseFeedForward(16, 64, rng=np.random.default_rng(0), dropout_rate=0.0)
        x = np.random.default_rng(0).standard_normal((2, 5, 16))
        result = ffn.forward(x)
        assert result.shape == (2, 5, 16)

    def test_backward_output_shape(self) -> None:
        ffn = PositionWiseFeedForward(16, 64, rng=np.random.default_rng(0), dropout_rate=0.0)
        x = np.random.default_rng(0).standard_normal((2, 5, 16))
        ffn.forward(x)
        grad = np.ones((2, 5, 16))
        grad_input = ffn.backward(grad)
        assert grad_input.shape == (2, 5, 16)

    def test_backward_raises_without_forward(self) -> None:
        ffn = PositionWiseFeedForward(16, 64, rng=np.random.default_rng(0), dropout_rate=0.0)
        with pytest.raises(ForwardNotCalledError):
            ffn.backward(np.ones((2, 5, 16)))

    def test_relu_cache_contains_positive_and_negative(self) -> None:
        """Pre-ReLU activations should contain both signs, proving ReLU is active."""
        ffn = PositionWiseFeedForward(16, 64, rng=np.random.default_rng(0), dropout_rate=0.0)
        x = np.random.default_rng(0).standard_normal((2, 5, 16))
        ffn.forward(x)
        assert ffn.relu._cache is not None
        assert np.any(ffn.relu._cache > 0)
        assert np.any(ffn.relu._cache < 0)


class TestSwiGLUFeedForward:
    def test_forward_output_shape(self) -> None:
        ffn = SwiGLUFeedForward(16, 32, rng=np.random.default_rng(0))
        x = np.random.default_rng(0).standard_normal((2, 5, 16))
        result = ffn.forward(x)
        assert result.shape == (2, 5, 16)

    def test_backward_output_shape(self) -> None:
        ffn = SwiGLUFeedForward(16, 32, rng=np.random.default_rng(0))
        x = np.random.default_rng(0).standard_normal((2, 5, 16))
        ffn.forward(x)
        grad = np.ones((2, 5, 16))
        grad_input = ffn.backward(grad)
        assert grad_input.shape == (2, 5, 16)

    def test_backward_raises_without_forward(self) -> None:
        ffn = SwiGLUFeedForward(16, 32, rng=np.random.default_rng(0))
        with pytest.raises(ForwardNotCalledError):
            ffn.backward(np.ones((2, 5, 16)))

    def test_forward_matches_pytorch(self) -> None:
        """Cross-verify forward pass against an equivalent PyTorch module."""
        d_model, d_ff = 16, 32
        rng = np.random.default_rng(42)
        ffn = SwiGLUFeedForward(d_model, d_ff, rng=rng)
        x_np = np.random.default_rng(0).standard_normal((2, 5, d_model))

        result_np = ffn.forward(x_np)

        pt = _build_torch_swiglu(ffn)
        x_t = torch.from_numpy(x_np)
        gate_t = torch.nn.functional.silu(pt.w_gate(x_t))
        result_t = pt.w_down(gate_t * pt.w_up(x_t)).detach().numpy()

        np.testing.assert_allclose(result_np, result_t, atol=1e-6)

    def test_backward_matches_pytorch(self) -> None:
        """Cross-verify backward pass against PyTorch autograd."""
        d_model, d_ff = 8, 16
        rng = np.random.default_rng(42)
        ffn = SwiGLUFeedForward(d_model, d_ff, rng=rng)

        x_np = np.random.default_rng(0).standard_normal((2, 3, d_model))
        grad_np = np.random.default_rng(1).standard_normal((2, 3, d_model))

        ffn.forward(x_np)
        grad_input_np = ffn.backward(grad_np)

        pt = _build_torch_swiglu(ffn)
        x_t = torch.from_numpy(x_np).requires_grad_(True)
        gate_t = torch.nn.functional.silu(pt.w_gate(x_t))
        out_t = pt.w_down(gate_t * pt.w_up(x_t))
        out_t.backward(torch.from_numpy(grad_np))
        assert x_t.grad is not None
        grad_input_torch = x_t.grad.numpy()

        np.testing.assert_allclose(grad_input_np, grad_input_torch, atol=1e-5)

    def test_update_params_changes_output(self) -> None:
        rng = np.random.default_rng(42)
        ffn = SwiGLUFeedForward(8, 16, rng=rng)
        x = np.random.default_rng(0).standard_normal((2, 3, 8))

        out_before = ffn.forward(x).copy()
        grad = np.ones((2, 3, 8))
        ffn.backward(grad)
        ffn.update_params(0.01)
        out_after = ffn.forward(x)

        assert not np.allclose(out_before, out_after)
