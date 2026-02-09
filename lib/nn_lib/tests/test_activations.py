"""Tests for activation function layers."""

import numpy as np
import numpy.typing as npt
import pytest

from nn_lib.activations import IdentityLayer, ReluLayer, SigmoidLayer, TanhLayer
from nn_lib.exceptions import ForwardNotCalledError
from nn_lib.tests.conftest import numerical_gradient


class TestSigmoidLayer:
    def test_forward_known_values(self) -> None:
        layer = SigmoidLayer()
        x = np.array([[0.0, 1.0, -1.0]])
        result = layer.forward(x)
        expected = np.array([[0.5, 1.0 / (1.0 + np.exp(-1.0)), 1.0 / (1.0 + np.exp(1.0))]])
        np.testing.assert_allclose(result, expected, rtol=1e-7)

    def test_forward_output_range(self, sample_input: npt.NDArray[np.float64]) -> None:
        layer = SigmoidLayer()
        result = layer.forward(sample_input)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_backward_matches_numerical(self) -> None:
        layer = SigmoidLayer()
        x = np.array([[0.5, -0.3, 1.2], [0.1, -0.7, 0.8]])
        layer.forward(x)
        grad_output = np.ones_like(x)
        analytic_grad = layer.backward(grad_output)
        numerical_grad = numerical_gradient(layer, x)
        np.testing.assert_allclose(analytic_grad, numerical_grad, atol=1e-5)

    def test_backward_raises_without_forward(self) -> None:
        layer = SigmoidLayer()
        with pytest.raises(ForwardNotCalledError):
            layer.backward(np.ones((2, 3)))


class TestReluLayer:
    def test_forward_known_values(self) -> None:
        layer = ReluLayer()
        x = np.array([[-1.0, 0.0, 1.0, 2.0]])
        result = layer.forward(x)
        expected = np.array([[0.0, 0.0, 1.0, 2.0]])
        np.testing.assert_array_equal(result, expected)

    def test_backward_matches_numerical(self) -> None:
        layer = ReluLayer()
        x = np.array([[0.5, -0.3, 1.2], [0.1, -0.7, 0.8]])
        layer.forward(x)
        grad_output = np.ones_like(x)
        analytic_grad = layer.backward(grad_output)
        numerical_grad = numerical_gradient(layer, x)
        np.testing.assert_allclose(analytic_grad, numerical_grad, atol=1e-5)

    def test_backward_zeros_for_negative_input(self) -> None:
        layer = ReluLayer()
        x = np.array([[-1.0, -2.0, -0.5]])
        layer.forward(x)
        grad = layer.backward(np.ones_like(x))
        np.testing.assert_array_equal(grad, np.zeros_like(x))

    def test_backward_raises_without_forward(self) -> None:
        layer = ReluLayer()
        with pytest.raises(ForwardNotCalledError):
            layer.backward(np.ones((2, 3)))


class TestTanhLayer:
    def test_forward_known_values(self) -> None:
        layer = TanhLayer()
        x = np.array([[0.0, 1.0, -1.0]])
        result = layer.forward(x)
        expected = np.tanh(x)
        np.testing.assert_allclose(result, expected, rtol=1e-7)

    def test_forward_output_range(self, sample_input: npt.NDArray[np.float64]) -> None:
        layer = TanhLayer()
        result = layer.forward(sample_input)
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_backward_matches_numerical(self) -> None:
        layer = TanhLayer()
        x = np.array([[0.5, -0.3, 1.2], [0.1, -0.7, 0.8]])
        layer.forward(x)
        grad_output = np.ones_like(x)
        analytic_grad = layer.backward(grad_output)
        numerical_grad = numerical_gradient(layer, x)
        np.testing.assert_allclose(analytic_grad, numerical_grad, atol=1e-5)

    def test_backward_raises_without_forward(self) -> None:
        layer = TanhLayer()
        with pytest.raises(ForwardNotCalledError):
            layer.backward(np.ones((2, 3)))


class TestIdentityLayer:
    def test_forward_passthrough(self, sample_input: npt.NDArray[np.float64]) -> None:
        layer = IdentityLayer()
        result = layer.forward(sample_input)
        np.testing.assert_array_equal(result, sample_input)

    def test_backward_passthrough(self) -> None:
        layer = IdentityLayer()
        grad = np.array([[1.0, 2.0, 3.0]])
        result = layer.backward(grad)
        np.testing.assert_array_equal(result, grad)
