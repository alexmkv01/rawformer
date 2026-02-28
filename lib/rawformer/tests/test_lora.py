"""Tests for LoRALinear with PyTorch cross-verification.

Covers 2D and 3D inputs, zero-init invariant, freeze semantics,
numerical gradient checks, and PyTorch autograd cross-verification.
"""

from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import pytest
import torch

from rawformer.core.exceptions import ForwardNotCalledError, ShapeMismatchError
from rawformer.layers.linear import LinearLayer
from rawformer.layers.lora import LoRALinear
from rawformer.tests.conftest import numerical_gradient

_RNG_SEED = 0
_NONZERO_B_SEED = 99


def _make_lora(
    n_in: int = 4,
    n_out: int = 3,
    rank: int = 2,
    alpha: float = 4.0,
    seed: int = _RNG_SEED,
    *,
    nonzero_b: bool = False,
) -> LoRALinear:
    """Create a LoRALinear with deterministic seeds.

    When *nonzero_b* is True the B matrix is replaced with random
    values so the LoRA path produces a non-trivial contribution.
    """
    rng_base = np.random.default_rng(seed)
    base = LinearLayer(n_in, n_out, rng=rng_base)
    rng_lora = np.random.default_rng(seed + 1)
    lora = LoRALinear(base, rank=rank, alpha=alpha, rng=rng_lora)
    if nonzero_b:
        lora._lora_b = np.random.default_rng(_NONZERO_B_SEED).standard_normal((rank, n_out))
    return lora


class _TorchLoRAResult(NamedTuple):
    x: torch.Tensor
    lora_a: torch.Tensor
    lora_b: torch.Tensor
    out: torch.Tensor


def _torch_lora_forward(
    lora: LoRALinear,
    x_np: npt.NDArray[np.float64],
    grad_np: npt.NDArray[np.float64] | None = None,
) -> _TorchLoRAResult:
    """Build a PyTorch equivalent of a LoRALinear and optionally run backward.

    When *grad_np* is provided the graph is back-propagated so that
    ``.grad`` attributes are populated on the returned tensors.
    """
    base = lora.base
    scale = lora.scale

    # Base linear: xW + b  (W and b frozen, no grad)
    w_torch = torch.from_numpy(base.weights.copy())
    b_torch = torch.from_numpy(base.biases.copy())

    x_torch = torch.from_numpy(x_np.copy()).requires_grad_(True)
    lora_a_torch = torch.from_numpy(lora.lora_a.copy()).requires_grad_(True)
    lora_b_torch = torch.from_numpy(lora.lora_b.copy()).requires_grad_(True)

    base_out = torch.einsum("...i,ij->...j", x_torch, w_torch) + b_torch
    lora_out = (
        torch.einsum(
            "...i,ij->...j",
            torch.einsum("...i,ij->...j", x_torch, lora_a_torch),
            lora_b_torch,
        )
        * scale
    )
    out_torch = base_out + lora_out

    if grad_np is not None:
        # Arithmetic on tensors produces an untyped .backward() in the
        # torch stubs shipped with this project's pinned version.
        out_torch.backward(torch.from_numpy(grad_np))  # type: ignore[no-untyped-call]

    return _TorchLoRAResult(x=x_torch, lora_a=lora_a_torch, lora_b=lora_b_torch, out=out_torch)


class TestLoRALinear2D:
    """Tests with 2D input (batch, features)."""

    def test_forward_output_shape(self) -> None:
        lora = _make_lora(n_in=4, n_out=3)
        x = np.random.default_rng(42).standard_normal((8, 4))
        output = lora.forward(x)
        assert output.shape == (8, 3)

    def test_backward_output_shape(self) -> None:
        lora = _make_lora(n_in=4, n_out=3)
        x = np.random.default_rng(42).standard_normal((8, 4))
        lora.forward(x)
        grad = lora.backward(np.ones((8, 3)))
        assert grad.shape == (8, 4)

    def test_zero_init_matches_base(self) -> None:
        """B is zero-initialized, so LoRA output should exactly match base."""
        lora = _make_lora()
        # Verify B is zeros
        np.testing.assert_array_equal(lora.lora_b, np.zeros_like(lora.lora_b))

        x = np.random.default_rng(42).standard_normal((8, 4))
        lora_out = lora.forward(x)
        base_out = np.einsum("...i,ij->...j", x, lora.base.weights) + lora.base.biases
        np.testing.assert_allclose(lora_out, base_out, atol=1e-12)

    def test_scaling_applied(self) -> None:
        """Verify that changing alpha/rank scales the LoRA contribution."""
        rng_base = np.random.default_rng(0)
        base = LinearLayer(4, 3, rng=rng_base)

        rng1 = np.random.default_rng(1)
        lora_a = LoRALinear(base, rank=2, alpha=2.0, rng=rng1)
        rng2 = np.random.default_rng(1)
        lora_b = LoRALinear(base, rank=2, alpha=4.0, rng=rng2)

        # Both instances need the same nonzero B so we can compare scaling
        shared_b = np.random.default_rng(_NONZERO_B_SEED).standard_normal((2, 3))
        lora_a._lora_b = shared_b.copy()
        lora_b._lora_b = shared_b.copy()

        x = np.random.default_rng(42).standard_normal((4, 4))
        out_a = lora_a.forward(x)
        out_b = lora_b.forward(x)

        # base output is the same; LoRA contribution of b should be 2x that of a
        base_out = np.einsum("...i,ij->...j", x, base.weights) + base.biases
        diff_a = out_a - base_out
        diff_b = out_b - base_out
        np.testing.assert_allclose(diff_b, 2.0 * diff_a, atol=1e-12)

    def test_forward_matches_pytorch(self) -> None:
        """Cross-verify forward pass against PyTorch."""
        lora = _make_lora(nonzero_b=True)

        x_np = np.random.default_rng(42).standard_normal((8, 4))
        result_np = lora.forward(x_np)

        ref = _torch_lora_forward(lora, x_np)
        result_torch = ref.out.detach().numpy()

        np.testing.assert_allclose(result_np, result_torch, atol=1e-6)

    def test_backward_matches_pytorch(self) -> None:
        """Cross-verify backward pass (input gradient) against PyTorch."""
        lora = _make_lora(nonzero_b=True)

        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((8, 4))
        grad_np = rng.standard_normal((8, 3))

        lora.forward(x_np)
        grad_input = lora.backward(grad_np)

        ref = _torch_lora_forward(lora, x_np, grad_np)
        assert ref.x.grad is not None
        grad_torch = ref.x.grad.numpy()

        np.testing.assert_allclose(grad_input, grad_torch, atol=1e-6)

    def test_backward_grad_a_matches_pytorch(self) -> None:
        """Cross-verify gradient of A against PyTorch."""
        lora = _make_lora(nonzero_b=True)

        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((8, 4))
        grad_np = rng.standard_normal((8, 3))

        lora.forward(x_np)
        lora.backward(grad_np)

        ref = _torch_lora_forward(lora, x_np, grad_np)
        assert ref.lora_a.grad is not None
        assert lora._grad_a is not None

        np.testing.assert_allclose(lora._grad_a, ref.lora_a.grad.numpy(), atol=1e-6)

    def test_backward_grad_b_matches_pytorch(self) -> None:
        """Cross-verify gradient of B against PyTorch."""
        lora = _make_lora(nonzero_b=True)

        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((8, 4))
        grad_np = rng.standard_normal((8, 3))

        lora.forward(x_np)
        lora.backward(grad_np)

        ref = _torch_lora_forward(lora, x_np, grad_np)
        assert ref.lora_b.grad is not None
        assert lora._grad_b is not None

        np.testing.assert_allclose(lora._grad_b, ref.lora_b.grad.numpy(), atol=1e-6)

    def test_numerical_gradient_check(self) -> None:
        """Verify backward pass via central-difference numerical gradient."""
        lora = _make_lora(n_in=3, n_out=2, rank=2, nonzero_b=True)

        x = np.array([[1.0, 0.5, -0.3], [0.2, -0.1, 0.7]])
        num_grad = numerical_gradient(lora, x)

        lora.forward(x)
        analytic_grad = lora.backward(np.ones((2, 2)))
        np.testing.assert_allclose(analytic_grad, num_grad, atol=1e-5)

    def test_backward_raises_without_forward(self) -> None:
        lora = _make_lora()
        with pytest.raises(ForwardNotCalledError):
            lora.backward(np.ones((1, 3)))

    def test_update_params_raises_without_backward(self) -> None:
        lora = _make_lora()
        with pytest.raises(ForwardNotCalledError):
            lora.update_params(0.01)

    def test_update_params_only_changes_lora_weights(self) -> None:
        """update_params must modify A and B but NOT base W and b."""
        lora = _make_lora(nonzero_b=True)

        x = np.random.default_rng(42).standard_normal((4, 4))
        lora.forward(x)
        lora.backward(np.ones((4, 3)))

        base_w_before = lora.base.weights.copy()
        base_b_before = lora.base.biases.copy()
        a_before = lora.lora_a.copy()
        b_before = lora.lora_b.copy()

        lora.update_params(learning_rate=0.01)

        # Base weights frozen
        np.testing.assert_array_equal(lora.base.weights, base_w_before)
        np.testing.assert_array_equal(lora.base.biases, base_b_before)
        # LoRA weights updated
        assert not np.array_equal(lora.lora_a, a_before)
        assert not np.array_equal(lora.lora_b, b_before)

    def test_callable_interface(self) -> None:
        lora = _make_lora()
        x = np.random.default_rng(42).standard_normal((2, 4))
        # __call__ and forward() are called on the same input; the second
        # call overwrites the internal cache, but we only compare outputs.
        result_call = lora(x)
        result_forward = lora.forward(x)
        np.testing.assert_array_equal(result_call, result_forward)

    def test_lora_a_shape(self) -> None:
        lora = _make_lora(n_in=8, n_out=4, rank=2)
        assert lora.lora_a.shape == (8, 2)

    def test_lora_b_shape(self) -> None:
        lora = _make_lora(n_in=8, n_out=4, rank=2)
        assert lora.lora_b.shape == (2, 4)

    def test_rejects_1d_input(self) -> None:
        lora = _make_lora(n_in=4, n_out=3)
        with pytest.raises(ShapeMismatchError):
            lora.forward(np.ones(4))

    def test_rejects_4d_input(self) -> None:
        lora = _make_lora(n_in=4, n_out=3)
        with pytest.raises(ShapeMismatchError):
            lora.forward(np.ones((2, 3, 4, 5)))

    def test_rejects_wrong_last_dim(self) -> None:
        lora = _make_lora(n_in=4, n_out=3)
        with pytest.raises(ShapeMismatchError):
            lora.forward(np.ones((2, 7)))


class TestLoRALinear3D:
    """Tests with 3D input (batch, seq_len, features)."""

    def test_forward_output_shape(self) -> None:
        lora = _make_lora(n_in=8, n_out=4)
        x = np.random.default_rng(42).standard_normal((2, 5, 8))
        output = lora.forward(x)
        assert output.shape == (2, 5, 4)

    def test_backward_output_shape(self) -> None:
        lora = _make_lora(n_in=8, n_out=4)
        x = np.random.default_rng(42).standard_normal((2, 5, 8))
        lora.forward(x)
        grad = lora.backward(np.ones((2, 5, 4)))
        assert grad.shape == (2, 5, 8)

    def test_forward_matches_pytorch(self) -> None:
        """Cross-verify 3D forward pass against PyTorch."""
        lora = _make_lora(n_in=8, n_out=4, nonzero_b=True)

        x_np = np.random.default_rng(42).standard_normal((2, 5, 8))
        result_np = lora.forward(x_np)

        ref = _torch_lora_forward(lora, x_np)
        result_torch = ref.out.detach().numpy()

        np.testing.assert_allclose(result_np, result_torch, atol=1e-6)

    def test_backward_matches_pytorch(self) -> None:
        """Cross-verify 3D backward pass against PyTorch autograd."""
        lora = _make_lora(n_in=8, n_out=4, nonzero_b=True)

        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((2, 5, 8))
        grad_np = rng.standard_normal((2, 5, 4))

        lora.forward(x_np)
        grad_input = lora.backward(grad_np)

        ref = _torch_lora_forward(lora, x_np, grad_np)
        assert ref.x.grad is not None
        grad_torch = ref.x.grad.numpy()

        np.testing.assert_allclose(grad_input, grad_torch, atol=1e-6)

    def test_3d_and_2d_produce_same_lora_grads(self) -> None:
        """Flattening a 3D input to 2D should give the same A/B gradients."""
        lora_3d = _make_lora(seed=42, nonzero_b=True)
        lora_2d = _make_lora(seed=42, nonzero_b=True)

        x_3d = np.random.default_rng(0).standard_normal((2, 5, 4))
        x_2d = x_3d.reshape(-1, 4)

        lora_3d.forward(x_3d)
        lora_2d.forward(x_2d)

        grad_3d = np.ones((2, 5, 3))
        grad_2d = np.ones((10, 3))

        lora_3d.backward(grad_3d)
        lora_2d.backward(grad_2d)

        assert lora_3d._grad_a is not None
        assert lora_2d._grad_a is not None
        assert lora_3d._grad_b is not None
        assert lora_2d._grad_b is not None
        np.testing.assert_allclose(lora_3d._grad_a, lora_2d._grad_a, atol=1e-12)
        np.testing.assert_allclose(lora_3d._grad_b, lora_2d._grad_b, atol=1e-12)


class TestLoRALinearProperties:
    """Property and invariant tests."""

    def test_trainable_param_count(self) -> None:
        """LoRA trainable params = rank * (n_in + n_out)."""
        n_in, n_out, rank = 64, 32, 4
        lora = _make_lora(n_in=n_in, n_out=n_out, rank=rank)
        expected = rank * (n_in + n_out)
        actual = lora.lora_a.size + lora.lora_b.size
        assert actual == expected

    def test_rank_property(self) -> None:
        lora = _make_lora(rank=8)
        assert lora.rank == 8

    def test_alpha_property(self) -> None:
        lora = _make_lora(alpha=16.0)
        assert lora.alpha == 16.0

    def test_scale_property(self) -> None:
        lora = _make_lora(rank=4, alpha=8.0)
        assert lora.scale == 2.0

    def test_base_property_returns_original(self) -> None:
        rng_base = np.random.default_rng(0)
        base = LinearLayer(4, 3, rng=rng_base)
        rng_lora = np.random.default_rng(1)
        lora = LoRALinear(base, rank=2, alpha=4.0, rng=rng_lora)
        assert lora.base is base
