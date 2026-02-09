"""Tests for LinearLayer (2D and 3D support)."""

import numpy as np
import pytest

from rawformer.exceptions import ForwardNotCalledError, ShapeMismatchError
from rawformer.layers.linear import LinearLayer
from rawformer.tests.conftest import numerical_gradient

_RNG = np.random.default_rng(0)


class TestLinearLayer2D:
    def test_forward_output_shape(self) -> None:
        layer = LinearLayer(4, 3, rng=_RNG)
        x = np.random.default_rng(0).standard_normal((8, 4))
        output = layer.forward(x)
        assert output.shape == (8, 3)

    def test_forward_computes_xw_plus_b(self) -> None:
        layer = LinearLayer(2, 2, rng=_RNG)
        x = np.array([[1.0, 2.0]])
        expected = x @ layer.weights + layer.biases
        np.testing.assert_allclose(layer.forward(x), expected)

    def test_backward_output_shape(self) -> None:
        layer = LinearLayer(4, 3, rng=_RNG)
        x = np.random.default_rng(0).standard_normal((8, 4))
        layer.forward(x)
        grad_output = np.ones((8, 3))
        grad_input = layer.backward(grad_output)
        assert grad_input.shape == (8, 4)

    def test_backward_gradient_numerical_check(self) -> None:
        """Verify backward pass via numerical gradient on the full layer."""
        layer = LinearLayer(3, 2, rng=_RNG)
        x = np.array([[1.0, 0.5, -0.3], [0.2, -0.1, 0.7]])

        num_grad = numerical_gradient(layer, x)

        layer.forward(x)
        analytic_grad = layer.backward(np.ones((2, 2)))
        np.testing.assert_allclose(analytic_grad, num_grad, atol=1e-5)

    def test_update_params_changes_weights(self) -> None:
        layer = LinearLayer(3, 2, rng=_RNG)
        x = np.array([[1.0, 0.5, -0.3]])
        layer.forward(x)
        layer.backward(np.ones((1, 2)))

        weights_before = layer.weights.copy()
        layer.update_params(learning_rate=0.01)
        assert not np.array_equal(layer.weights, weights_before)

    def test_backward_raises_without_forward(self) -> None:
        layer = LinearLayer(3, 2, rng=_RNG)
        with pytest.raises(ForwardNotCalledError):
            layer.backward(np.ones((1, 2)))

    def test_update_params_raises_without_backward(self) -> None:
        layer = LinearLayer(3, 2, rng=_RNG)
        with pytest.raises(ForwardNotCalledError):
            layer.update_params(0.01)

    def test_callable_interface(self) -> None:
        layer = LinearLayer(4, 3, rng=_RNG)
        x = np.random.default_rng(0).standard_normal((2, 4))
        result_call = layer(x)
        result_forward = layer.forward(x)
        np.testing.assert_array_equal(result_call, result_forward)

    def test_biases_initialized_to_zero(self) -> None:
        layer = LinearLayer(10, 5, rng=_RNG)
        np.testing.assert_array_equal(layer.biases, np.zeros(5))


class TestLinearLayer3D:
    def test_forward_output_shape(self) -> None:
        layer = LinearLayer(8, 4, rng=_RNG)
        x = np.random.default_rng(0).standard_normal((2, 5, 8))
        output = layer.forward(x)
        assert output.shape == (2, 5, 4)

    def test_forward_matches_manual_einsum(self) -> None:
        rng = np.random.default_rng(1)
        layer = LinearLayer(4, 3, rng=rng)
        x = np.random.default_rng(2).standard_normal((2, 5, 4))
        result = layer.forward(x)
        expected = np.einsum("bsi,ij->bsj", x, layer.weights) + layer.biases
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_backward_output_shape(self) -> None:
        layer = LinearLayer(8, 4, rng=_RNG)
        x = np.random.default_rng(0).standard_normal((2, 5, 8))
        layer.forward(x)
        grad = np.ones((2, 5, 4))
        grad_input = layer.backward(grad)
        assert grad_input.shape == (2, 5, 8)

    def test_3d_and_2d_produce_same_weights_grad(self) -> None:
        """Flattening a 3D input to 2D should give the same weight gradients."""
        rng = np.random.default_rng(42)
        layer_3d = LinearLayer(4, 3, rng=rng)
        rng = np.random.default_rng(42)
        layer_2d = LinearLayer(4, 3, rng=rng)

        x_3d = np.random.default_rng(0).standard_normal((2, 5, 4))
        x_2d = x_3d.reshape(-1, 4)

        layer_3d.forward(x_3d)
        layer_2d.forward(x_2d)

        grad_3d = np.ones((2, 5, 3))
        grad_2d = np.ones((10, 3))

        layer_3d.backward(grad_3d)
        layer_2d.backward(grad_2d)

        assert layer_3d._grad_weights is not None
        assert layer_2d._grad_weights is not None
        assert layer_3d._grad_biases is not None
        assert layer_2d._grad_biases is not None
        np.testing.assert_allclose(layer_3d._grad_weights, layer_2d._grad_weights, atol=1e-12)
        np.testing.assert_allclose(layer_3d._grad_biases, layer_2d._grad_biases, atol=1e-12)

    def test_rejects_1d_input(self) -> None:
        layer = LinearLayer(4, 3, rng=_RNG)
        with pytest.raises(ShapeMismatchError):
            layer.forward(np.ones(4))

    def test_rejects_4d_input(self) -> None:
        layer = LinearLayer(4, 3, rng=_RNG)
        with pytest.raises(ShapeMismatchError):
            layer.forward(np.ones((2, 3, 4, 5)))

    def test_rejects_wrong_last_dim(self) -> None:
        layer = LinearLayer(4, 3, rng=_RNG)
        with pytest.raises(ShapeMismatchError):
            layer.forward(np.ones((2, 5, 7)))
