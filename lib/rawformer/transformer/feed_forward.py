"""Position-wise feed-forward network (Vaswani et al., 2017).

FFN(x) = ReLU(x W_1 + b_1) W_2 + b_2

Applied identically to each position in the sequence.
"""

import numpy as np
import numpy.typing as npt

from rawformer.exceptions import ForwardNotCalledError
from rawformer.layers.linear import LinearLayer


class PositionWiseFeedForward:
    """Two-layer MLP with ReLU, applied independently to each position.

    Args:
        d_model: Input and output dimension.
        d_ff: Inner (hidden) dimension, typically 4 * d_model.
        rng: NumPy random generator for weight initialization.
    """

    def __init__(self, d_model: int, d_ff: int, rng: np.random.Generator) -> None:
        self.linear1 = LinearLayer(d_model, d_ff, rng)
        self.linear2 = LinearLayer(d_ff, d_model, rng)
        self._relu_cache: npt.NDArray[np.float64] | None = None

    def forward(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Forward pass: Linear -> ReLU -> Linear.

        Args:
            x: Input of shape (batch, seq_len, d_model).
        """
        hidden = self.linear1.forward(x)
        self._relu_cache = hidden
        relu_out: npt.NDArray[np.float64] = np.maximum(0, hidden)
        return self.linear2.forward(relu_out)

    def backward(self, grad_z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Backward pass through the FFN.

        Args:
            grad_z: Upstream gradient of shape (batch, seq_len, d_model).
        """
        if self._relu_cache is None:
            raise ForwardNotCalledError("PositionWiseFeedForward")
        grad_relu = self.linear2.backward(grad_z)
        grad_relu = grad_relu * (self._relu_cache > 0).astype(np.float64)
        return self.linear1.backward(grad_relu)

    def update_params(self, learning_rate: float) -> None:
        self.linear1.update_params(learning_rate)
        self.linear2.update_params(learning_rate)
