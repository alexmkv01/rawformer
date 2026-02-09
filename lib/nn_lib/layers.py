"""Fully connected (linear) layer implementation."""

import numpy as np
import numpy.typing as npt

from nn_lib.base import Layer
from nn_lib.exceptions import ForwardNotCalledError, ShapeMismatchError
from nn_lib.initializers import xavier_init, zeros_init


class LinearLayer(Layer):
    """Fully connected layer: y = xW + b.

    Weights are always Xavier-initialized regardless of downstream activation.
    To use a different initializer (e.g. He for ReLU), construct weights
    externally or extend this class. Biases are zero-initialized.
    """

    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator) -> None:
        self.n_in = n_in
        self.n_out = n_out
        self._weights: npt.NDArray[np.float64] = xavier_init(n_in, n_out, rng)
        self._biases: npt.NDArray[np.float64] = zeros_init(n_out)

        self._input_cache: npt.NDArray[np.float64] | None = None
        self._grad_weights: npt.NDArray[np.float64] | None = None
        self._grad_biases: npt.NDArray[np.float64] | None = None

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        return self._weights

    @property
    def biases(self) -> npt.NDArray[np.float64]:
        return self._biases

    def forward(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if x.ndim != 2:
            raise ShapeMismatchError(f"Expected 2D input, got {x.ndim}D")
        self._input_cache = x
        return x @ self._weights + self._biases

    def backward(self, grad_z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if self._input_cache is None:
            raise ForwardNotCalledError("LinearLayer")
        self._grad_weights = self._input_cache.T @ grad_z
        self._grad_biases = np.sum(grad_z, axis=0)
        return grad_z @ self._weights.T

    def update_params(self, learning_rate: float) -> None:
        if self._grad_weights is None or self._grad_biases is None:
            raise ForwardNotCalledError("LinearLayer")
        self._weights -= learning_rate * self._grad_weights
        self._biases -= learning_rate * self._grad_biases
