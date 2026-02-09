"""Fully connected (linear) layer implementation supporting 2D and 3D tensors."""

import numpy as np
import numpy.typing as npt

from rawformer.base import Layer
from rawformer.exceptions import ForwardNotCalledError, ShapeMismatchError
from rawformer.layers.initializers import xavier_init, zeros_init


class LinearLayer(Layer):
    """Fully connected layer: y = xW + b.

    Supports both 2D (batch, features) and 3D (batch, seq_len, features)
    inputs. Uses np.einsum for dimension-agnostic matrix multiplication.

    Weights are Xavier-initialized, biases are zero-initialized.
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
        if x.ndim not in (2, 3):
            raise ShapeMismatchError(f"Expected 2D or 3D input, got {x.ndim}D")
        if x.shape[-1] != self.n_in:
            raise ShapeMismatchError(f"Last dim {x.shape[-1]} != n_in {self.n_in}")
        self._input_cache = x
        result: npt.NDArray[np.float64] = (
            np.einsum("...i,ij->...j", x, self._weights) + self._biases
        )
        return result

    def backward(self, grad_z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if self._input_cache is None:
            raise ForwardNotCalledError("LinearLayer")
        # flatten batch dims for weight gradient: (*, n_in) and (*, n_out) -> (n_in, n_out)
        x_flat = self._input_cache.reshape(-1, self.n_in)
        grad_flat = grad_z.reshape(-1, self.n_out)
        self._grad_weights = x_flat.T @ grad_flat
        self._grad_biases = grad_flat.sum(axis=0)
        result: npt.NDArray[np.float64] = np.einsum("...j,ij->...i", grad_z, self._weights)
        return result

    def update_params(self, learning_rate: float) -> None:
        if self._grad_weights is None or self._grad_biases is None:
            raise ForwardNotCalledError("LinearLayer")
        self._weights -= learning_rate * self._grad_weights
        self._biases -= learning_rate * self._grad_biases
