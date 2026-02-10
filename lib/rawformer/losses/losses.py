"""Loss function implementations for neural network training."""

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from rawformer.core.exceptions import ForwardNotCalledError, ShapeMismatchError

# Small constant added inside log() to prevent log(0).
# Note: backward() does not account for this clamp — an intentional approximation
# since the gradient at near-zero probabilities is already dominated by the 1/p term.
_LOG_EPS: float = 1e-8


class Loss(ABC):
    """Abstract base class for all loss functions."""

    @abstractmethod
    def forward(
        self,
        y_pred: npt.NDArray[np.float64],
        y_target: npt.NDArray[np.float64],
    ) -> float:
        """Compute the loss value."""

    @abstractmethod
    def backward(self) -> npt.NDArray[np.float64]:
        """Compute the gradient of the loss with respect to predictions."""


class MSELoss(Loss):
    """Mean Squared Error loss for regression tasks.

    L = mean((y_pred - y_target)^2)

    Averages over both batch and output dimensions (n*d), giving true
    element-wise MSE. Note: CrossEntropyLoss normalizes by batch size (n)
    only, so switching loss functions changes gradient scale.
    """

    def __init__(self) -> None:
        self._cache: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None = None

    def forward(
        self,
        y_pred: npt.NDArray[np.float64],
        y_target: npt.NDArray[np.float64],
    ) -> float:
        if y_pred.shape[0] != y_target.shape[0]:
            raise ShapeMismatchError(
                f"Batch size mismatch: predictions {y_pred.shape[0]}, targets {y_target.shape[0]}"
            )
        self._cache = (y_pred, y_target)
        return float(np.mean((y_pred - y_target) ** 2))

    def backward(self) -> npt.NDArray[np.float64]:
        if self._cache is None:
            raise ForwardNotCalledError("MSELoss")
        y_pred, y_target = self._cache
        n_elements = y_pred.shape[0] * y_pred.shape[1]
        result: npt.NDArray[np.float64] = 2.0 * (y_pred - y_target) / n_elements
        return result


class CrossEntropyLoss(Loss):
    """Cross-entropy loss with built-in softmax for multi-class classification.

    Applies softmax to predictions, then computes:
    L = -(1/n) * sum(y_target * log(softmax(y_pred)))
    """

    def __init__(self) -> None:
        self._cache: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None = None

    @staticmethod
    def _softmax(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Numerically stable softmax: subtract max per row before exp."""
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        result: npt.NDArray[np.float64] = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return result

    def forward(
        self,
        y_pred: npt.NDArray[np.float64],
        y_target: npt.NDArray[np.float64],
    ) -> float:
        if y_pred.shape[0] != y_target.shape[0]:
            raise ShapeMismatchError(
                f"Batch size mismatch: predictions {y_pred.shape[0]}, targets {y_target.shape[0]}"
            )
        probs = self._softmax(y_pred)
        self._cache = (probs, y_target)
        n = y_pred.shape[0]
        return float(-np.sum(y_target * np.log(probs + _LOG_EPS)) / n)

    def backward(self) -> npt.NDArray[np.float64]:
        if self._cache is None:
            raise ForwardNotCalledError("CrossEntropyLoss")
        probs, y_target = self._cache
        n = probs.shape[0]
        result: npt.NDArray[np.float64] = (probs - y_target) / n
        return result
