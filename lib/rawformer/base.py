"""Abstract base classes for all neural network components.

Layer is the minimal base for any component with learnable parameters
(or that participates in a computational graph).  SimpleLayer extends it
with the standard single-tensor forward/backward contract used by most
primitive layers.
"""

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class Layer(ABC):
    """Base class for all neural network components.

    Every component in the transformer (and the feedforward network)
    inherits from this class.  It guarantees a uniform
    ``update_params`` interface while leaving ``forward`` and
    ``backward`` signatures to subclasses, since composite components
    like MultiHeadAttention and Decoder accept multiple inputs and
    may return tuples of gradients.
    """

    def update_params(self, learning_rate: float) -> None:
        """Update learnable parameters. No-op for layers without parameters."""


class SimpleLayer(Layer):
    """Layer with the standard single-tensor forward/backward contract.

    Subclass this when your layer takes a single NDArray input and
    returns a single NDArray output (e.g. linear, activation,
    normalization, dropout).
    """

    @abstractmethod
    def forward(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute the forward pass."""

    @abstractmethod
    def backward(self, grad_z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute the backward pass given upstream gradients."""

    def __call__(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self.forward(x)
