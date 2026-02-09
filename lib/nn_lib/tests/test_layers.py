"""Tests for LinearLayer."""

import numpy as np
import pytest

from nn_lib.exceptions import ForwardNotCalledError
from nn_lib.layers import LinearLayer
from nn_lib.tests.conftest import numerical_gradient

_RNG = np.random.default_rng(0)


class TestLinearLayer:
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
