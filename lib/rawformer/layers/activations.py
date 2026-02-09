"""Activation function layers for neural networks."""

import numpy as np
import numpy.typing as npt

from rawformer.base import SimpleLayer
from rawformer.exceptions import ForwardNotCalledError


class SigmoidLayer(SimpleLayer):
    """Sigmoid activation: f(x) = 1 / (1 + exp(-x))."""

    def __init__(self) -> None:
        self._cache: npt.NDArray[np.float64] | None = None

    def forward(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        output: npt.NDArray[np.float64] = 1.0 / (1.0 + np.exp(-x))
        self._cache = output
        return output

    def backward(self, grad_z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if self._cache is None:
            raise ForwardNotCalledError("SigmoidLayer")
        return grad_z * self._cache * (1.0 - self._cache)


class ReluLayer(SimpleLayer):
    """ReLU activation: f(x) = max(0, x)."""

    def __init__(self) -> None:
        self._cache: npt.NDArray[np.float64] | None = None

    def forward(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self._cache = x
        return np.maximum(0, x)

    def backward(self, grad_z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if self._cache is None:
            raise ForwardNotCalledError("ReluLayer")
        return grad_z * (self._cache > 0).astype(np.float64)


class TanhLayer(SimpleLayer):
    """Tanh activation: f(x) = tanh(x)."""

    def __init__(self) -> None:
        self._cache: npt.NDArray[np.float64] | None = None

    def forward(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        output = np.tanh(x)
        self._cache = output
        return output

    def backward(self, grad_z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if self._cache is None:
            raise ForwardNotCalledError("TanhLayer")
        return grad_z * (1.0 - self._cache**2)


class IdentityLayer(SimpleLayer):
    """Identity activation: f(x) = x. Pass-through layer."""

    def forward(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return x

    def backward(self, grad_z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return grad_z
