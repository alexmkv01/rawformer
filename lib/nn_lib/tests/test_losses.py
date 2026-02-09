"""Tests for loss functions."""

import numpy as np
import numpy.typing as npt
import pytest

from nn_lib.exceptions import ForwardNotCalledError, ShapeMismatchError
from nn_lib.losses import CrossEntropyLoss, MSELoss


class TestMSELoss:
    def test_forward_perfect_prediction(self) -> None:
        loss = MSELoss()
        y = np.array([[1.0, 0.0], [0.0, 1.0]])
        assert loss.forward(y, y) == 0.0

    def test_forward_known_value(self) -> None:
        loss = MSELoss()
        y_pred = np.array([[1.0, 0.0]])
        y_target = np.array([[0.0, 1.0]])
        expected = (1.0 + 1.0) / 2  # mean over all elements (n=1, d=2)
        assert loss.forward(y_pred, y_target) == pytest.approx(expected)

    def test_backward_numerical_gradient(self) -> None:
        loss = MSELoss()
        y_pred = np.array([[0.5, 0.3], [0.2, 0.8]])
        y_target = np.array([[1.0, 0.0], [0.0, 1.0]])

        loss.forward(y_pred, y_target)
        analytic_grad = loss.backward()

        eps = 1e-5
        numerical_grad = np.zeros_like(y_pred)
        for idx in np.ndindex(y_pred.shape):
            y_plus = y_pred.copy()
            y_minus = y_pred.copy()
            y_plus[idx] += eps
            y_minus[idx] -= eps
            loss_plus = loss.forward(y_plus, y_target)
            loss_minus = loss.forward(y_minus, y_target)
            numerical_grad[idx] = (loss_plus - loss_minus) / (2 * eps)

        np.testing.assert_allclose(analytic_grad, numerical_grad, atol=1e-5)

    def test_backward_raises_without_forward(self) -> None:
        loss = MSELoss()
        with pytest.raises(ForwardNotCalledError):
            loss.backward()


class TestCrossEntropyLoss:
    def test_forward_returns_positive(self, sample_targets_onehot: npt.NDArray[np.float64]) -> None:
        loss = CrossEntropyLoss()
        rng = np.random.default_rng(0)
        y_pred = rng.standard_normal((8, 3))
        result = loss.forward(y_pred, sample_targets_onehot)
        assert result > 0.0

    def test_forward_low_loss_for_correct_predictions(self) -> None:
        loss = CrossEntropyLoss()
        y_target = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        y_pred = np.array([[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0]])
        result = loss.forward(y_pred, y_target)
        assert result < 0.001

    def test_forward_batch_size_mismatch_raises(self) -> None:
        loss = CrossEntropyLoss()
        y_pred = np.ones((3, 2))
        y_target = np.ones((4, 2))
        with pytest.raises(ShapeMismatchError):
            loss.forward(y_pred, y_target)

    def test_backward_numerical_gradient(self) -> None:
        loss = CrossEntropyLoss()
        y_pred = np.array([[1.0, 2.0, 0.5], [0.3, -0.5, 1.2]])
        y_target = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

        loss.forward(y_pred, y_target)
        analytic_grad = loss.backward()

        eps = 1e-5
        numerical_grad = np.zeros_like(y_pred)
        for idx in np.ndindex(y_pred.shape):
            y_plus = y_pred.copy()
            y_minus = y_pred.copy()
            y_plus[idx] += eps
            y_minus[idx] -= eps
            loss_plus = loss.forward(y_plus, y_target)
            loss_minus = loss.forward(y_minus, y_target)
            numerical_grad[idx] = (loss_plus - loss_minus) / (2 * eps)

        np.testing.assert_allclose(analytic_grad, numerical_grad, atol=1e-5)

    def test_backward_raises_without_forward(self) -> None:
        loss = CrossEntropyLoss()
        with pytest.raises(ForwardNotCalledError):
            loss.backward()
