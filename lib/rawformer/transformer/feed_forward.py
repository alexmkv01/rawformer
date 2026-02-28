"""Position-wise feed-forward networks.

PositionWiseFeedForward — Vaswani et al., 2017
    FFN(x) = Dropout(ReLU(x W_1 + b_1)) W_2 + b_2

SwiGLUFeedForward — Shazeer, 2020
    SwiGLU(x) = (SiLU(x W_gate) * (x W_up)) W_down
"""

from typing import TypedDict

import numpy as np
import numpy.typing as npt

from rawformer.core.base import SimpleLayer
from rawformer.core.exceptions import ForwardNotCalledError
from rawformer.layers.activations import ReluLayer, SiLULayer
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


class _SwiGLUCache(TypedDict):
    gate: npt.NDArray[np.float64]
    up: npt.NDArray[np.float64]


class SwiGLUFeedForward(SimpleLayer):
    """Gated feed-forward network with SiLU activation (Shazeer, 2020).

    SwiGLU(x) = (SiLU(x W_gate) * (x W_up)) W_down

    Uses three linear projections instead of two. To keep total parameter
    count comparable to a standard FFN with inner dim 4*d_model, callers
    typically set d_ff = int((2/3) * 4 * d_model).

    Args:
        d_model: Input and output dimension.
        d_ff: Inner (hidden) dimension for gate and up projections.
        rng: NumPy random generator for weight initialization.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        rng: np.random.Generator,
    ) -> None:
        self.w_gate = LinearLayer(d_model, d_ff, rng)
        self.w_up = LinearLayer(d_model, d_ff, rng)
        self.w_down = LinearLayer(d_ff, d_model, rng)
        self.silu = SiLULayer()
        # No dropout — following the Llama convention where SwiGLU layers omit it.

        self._cache: _SwiGLUCache | None = None

    def forward(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Forward pass: SiLU(x W_gate) * (x W_up) then W_down.

        Args:
            x: Input of shape (batch, seq_len, d_model).

        Returns:
            Output of shape (batch, seq_len, d_model).
        """
        gate = self.silu.forward(self.w_gate.forward(x))
        up = self.w_up.forward(x)
        self._cache = _SwiGLUCache(gate=gate, up=up)
        hidden: npt.NDArray[np.float64] = gate * up
        return self.w_down.forward(hidden)

    def backward(self, grad_z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Backward pass through the SwiGLU FFN.

        Args:
            grad_z: Upstream gradient of shape (batch, seq_len, d_model).

        Returns:
            Gradient with respect to the input x, shape (batch, seq_len, d_model).
        """
        if self._cache is None:
            raise ForwardNotCalledError("SwiGLUFeedForward")
        gate = self._cache["gate"]
        up = self._cache["up"]

        # backward through W_down
        grad_hidden = self.w_down.backward(grad_z)

        # backward through element-wise product: hidden = gate * up
        grad_gate: npt.NDArray[np.float64] = grad_hidden * up
        grad_up: npt.NDArray[np.float64] = grad_hidden * gate

        # backward through SiLU and W_gate
        grad_x_gate = self.w_gate.backward(self.silu.backward(grad_gate))

        # backward through W_up
        grad_x_up = self.w_up.backward(grad_up)

        # both paths originate from the same input x
        result: npt.NDArray[np.float64] = grad_x_gate + grad_x_up
        return result

    def update_params(self, learning_rate: float) -> None:
        self.w_gate.update_params(learning_rate)
        self.w_up.update_params(learning_rate)
        self.w_down.update_params(learning_rate)
