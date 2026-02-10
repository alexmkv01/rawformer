"""Tests for MultiLayerNetwork."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from rawformer.models.network import MultiLayerNetwork


class TestMultiLayerNetwork:
    def test_forward_output_shape(self) -> None:
        net = MultiLayerNetwork(4, [16, 3], ["relu", "identity"])
        x = np.random.default_rng(0).standard_normal((8, 4))
        output = net.forward(x)
        assert output.shape == (8, 3)

    def test_callable_interface(self) -> None:
        net = MultiLayerNetwork(4, [16, 3], ["relu", "identity"])
        x = np.random.default_rng(0).standard_normal((2, 4))
        np.testing.assert_array_equal(net(x), net.forward(x))

    def test_backward_output_shape(self) -> None:
        net = MultiLayerNetwork(4, [16, 3], ["relu", "identity"])
        x = np.random.default_rng(0).standard_normal((8, 4))
        net.forward(x)
        grad = net.backward(np.ones((8, 3)))
        assert grad.shape == (8, 4)

    def test_mismatched_neurons_activations_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            MultiLayerNetwork(4, [16, 3], ["relu"])

    def test_unknown_activation_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown activation"):
            MultiLayerNetwork(4, [16], ["bad_activation"])  # type: ignore[list-item]

    def test_all_activation_types(self) -> None:
        for act in ["relu", "sigmoid", "tanh", "identity"]:
            net = MultiLayerNetwork(4, [8], [act])  # type: ignore[list-item]
            x = np.random.default_rng(0).standard_normal((2, 4))
            output = net.forward(x)
            assert output.shape == (2, 8)

    def test_deep_network(self) -> None:
        net = MultiLayerNetwork(4, [32, 16, 8, 3], ["relu", "relu", "sigmoid", "identity"])
        x = np.random.default_rng(0).standard_normal((4, 4))
        output = net.forward(x)
        assert output.shape == (4, 3)

    def test_save_and_load_roundtrip(self) -> None:
        net = MultiLayerNetwork(4, [16, 3], ["relu", "identity"])
        x = np.random.default_rng(42).standard_normal((4, 4))
        original_output = net.forward(x)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pkl"
            net.save(path)
            loaded = MultiLayerNetwork.load(path)

        loaded_output = loaded.forward(x)
        np.testing.assert_array_equal(original_output, loaded_output)

    def test_update_params_changes_output(self) -> None:
        net = MultiLayerNetwork(4, [16, 3], ["relu", "identity"])
        x = np.random.default_rng(0).standard_normal((4, 4))

        output_before = net.forward(x).copy()
        net.backward(np.ones((4, 3)))
        net.update_params(learning_rate=0.1)
        output_after = net.forward(x)

        assert not np.array_equal(output_before, output_after)
