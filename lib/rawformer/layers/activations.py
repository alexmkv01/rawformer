"""Activation function layers for neural networks.

Includes standard activations (Sigmoid, ReLU, Tanh, Identity) and
SiLU/Swish (Elfwing et al., 2018; Ramachandran et al., 2017),
used as a component of SwiGLU (Shazeer, 2020).
"""

from typing import TypedDict

import numpy as np
import numpy.typing as npt

from rawformer.core.base import SimpleLayer
from rawformer.core.exceptions import ForwardNotCalledError


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


class _SiLUCache(TypedDict):
    x: npt.NDArray[np.float64]
    sigmoid: npt.NDArray[np.float64]


class SiLULayer(SimpleLayer):
    """SiLU (Sigmoid Linear Unit) activation, also known as Swish.

    f(x) = x * sigmoid(x)

    Used as the gating activation inside SwiGLU (Shazeer, 2020).
    """

    def __init__(self) -> None:
        self._cache: _SiLUCache | None = None

    def forward(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        sigmoid: npt.NDArray[np.float64] = 1.0 / (1.0 + np.exp(-x))
        self._cache = _SiLUCache(x=x, sigmoid=sigmoid)
        result: npt.NDArray[np.float64] = x * sigmoid
        return result

    def backward(self, grad_z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """SiLU gradient: f'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))."""
        if self._cache is None:
            raise ForwardNotCalledError("SiLULayer")
        x = self._cache["x"]
        s = self._cache["sigmoid"]
        result: npt.NDArray[np.float64] = grad_z * (s + x * s * (1.0 - s))
        return result
