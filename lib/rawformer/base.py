"""Abstract base class for all neural network layers."""

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class Layer(ABC):
    """Abstract base layer that all layers must inherit from."""

    @abstractmethod
    def forward(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute the forward pass."""

    @abstractmethod
    def backward(self, grad_z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute the backward pass given upstream gradients."""

    def update_params(self, learning_rate: float) -> None:
        """Update learnable parameters. No-op for layers without parameters."""

    def __call__(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self.forward(x)
