"""Multi-layer feedforward neural network."""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

from rawformer.core.base import SimpleLayer
from rawformer.layers.activations import IdentityLayer, ReluLayer, SigmoidLayer, TanhLayer
from rawformer.layers.linear import LinearLayer


@dataclass
class LayerGroup:
    """A linear layer paired with its activation."""

    linear: LinearLayer
    activation: SimpleLayer


ActivationType = Literal["relu", "sigmoid", "tanh", "identity"]

_ACTIVATION_MAP: dict[str, type[SimpleLayer]] = {
    "relu": ReluLayer,
    "sigmoid": SigmoidLayer,
    "tanh": TanhLayer,
    "identity": IdentityLayer,
}


class MultiLayerNetwork(SimpleLayer):
    """A feedforward neural network composed of linear layers and activations.

    Each entry in `neurons` and `activations` defines one layer group:
    a LinearLayer followed by an activation layer.

    Example:
        >>> net = MultiLayerNetwork(4, [16, 3], ["relu", "identity"])
        >>> output = net(input_data)
    """

    def __init__(
        self,
        input_dim: int,
        neurons: list[int],
        activations: list[ActivationType],
        rng: np.random.Generator | None = None,
    ) -> None:
        if len(neurons) != len(activations):
            raise ValueError(
                f"neurons ({len(neurons)}) and activations ({len(activations)}) "
                f"must have the same length"
            )
        rng = rng if rng is not None else np.random.default_rng()
        self._layers = _build_layers(input_dim, neurons, activations, rng)

    def forward(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Forward pass through all layers sequentially."""
        output = x
        for group in self._layers:
            output = group.linear.forward(output)
            output = group.activation.forward(output)
        return output

    def backward(self, grad_z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Backward pass through all layers in reverse order."""
        grad = grad_z
        for group in reversed(self._layers):
            grad = group.activation.backward(grad)
            grad = group.linear.backward(grad)
        return grad

    def update_params(self, learning_rate: float) -> None:
        """Update parameters of all learnable layers."""
        for group in self._layers:
            group.linear.update_params(learning_rate)

    def save(self, path: Path) -> None:
        """Serialize the network to a pickle file."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "MultiLayerNetwork":
        """Deserialize a network from a pickle file."""
        with open(path, "rb") as f:
            network = pickle.load(f)
        if not isinstance(network, MultiLayerNetwork):
            raise TypeError(f"Expected MultiLayerNetwork, got {type(network).__name__} from {path}")
        return network


def _build_layers(
    input_dim: int,
    neurons: list[int],
    activations: list[ActivationType],
    rng: np.random.Generator,
) -> list[LayerGroup]:
    """Construct layer groups from architecture specification."""
    layers: list[LayerGroup] = []
    prev_dim = input_dim
    for n_out, act_name in zip(neurons, activations, strict=True):
        linear = LinearLayer(prev_dim, n_out, rng)
        activation = _create_activation(act_name)
        layers.append(LayerGroup(linear=linear, activation=activation))
        prev_dim = n_out
    return layers


def _create_activation(name: ActivationType) -> SimpleLayer:
    """Instantiate an activation layer by name."""
    cls = _ACTIVATION_MAP.get(name)
    if cls is None:
        raise ValueError(f"Unknown activation: {name!r}. Choose from {list(_ACTIVATION_MAP)}")
    return cls()
