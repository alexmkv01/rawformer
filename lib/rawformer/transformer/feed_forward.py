"""Position-wise feed-forward network (Vaswani et al., 2017).

FFN(x) = Dropout(ReLU(x W_1 + b_1)) W_2 + b_2

Applied identically to each position in the sequence.
"""

import numpy as np
import numpy.typing as npt

from rawformer.base import SimpleLayer
from rawformer.layers.activations import ReluLayer
from rawformer.layers.dropout import Dropout
from rawformer.layers.linear import LinearLayer


class PositionWiseFeedForward(SimpleLayer):
    """Two-layer MLP with ReLU and dropout, applied independently to each position.

    Args:
        d_model: Input and output dimension.
        d_ff: Inner (hidden) dimension, typically 4 * d_model.
        rng: NumPy random generator for weight initialization.
        dropout_rate: Dropout probability applied after ReLU.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        rng: np.random.Generator,
        dropout_rate: float = 0.1,
    ) -> None:
        self.linear1 = LinearLayer(d_model, d_ff, rng)
        self.linear2 = LinearLayer(d_ff, d_model, rng)
        self.relu = ReluLayer()
        self.dropout = Dropout(dropout_rate, rng)

    @property
    def dropouts(self) -> list[Dropout]:
        return [self.dropout]

    def forward(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Forward pass: Linear -> ReLU -> Dropout -> Linear.

        Args:
            x: Input of shape (batch, seq_len, d_model).
        """
        relu_out = self.relu.forward(self.linear1.forward(x))
        dropped = self.dropout.forward(relu_out)
        return self.linear2.forward(dropped)

    def backward(self, grad_z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Backward pass through the FFN.

        Args:
            grad_z: Upstream gradient of shape (batch, seq_len, d_model).
        """
        grad_dropped = self.linear2.backward(grad_z)
        grad_relu = self.dropout.backward(grad_dropped)
        return self.linear1.backward(self.relu.backward(grad_relu))

    def update_params(self, learning_rate: float) -> None:
        self.linear1.update_params(learning_rate)
        self.linear2.update_params(learning_rate)
