"""Smoke tests for the training pipeline stages.

These tests validate the pipeline logic using synthetic data,
without requiring the actual iris.dat file or DVC.
"""

import tempfile
from pathlib import Path

import numpy as np

from rawformer import MultiLayerNetwork
from rawformer_train.evaluate import compute_accuracy, evaluate
from rawformer_train.prepare import load_and_split


class TestPrepare:
    def test_load_and_split_shapes(self) -> None:
        """Verify the 3-way split produces correct shapes."""
        with tempfile.NamedTemporaryFile(suffix=".dat", mode="w", delete=False) as f:
            data = np.random.default_rng(0).standard_normal((100, 7))
            data[:, 4:] = 0.0
            classes = np.random.default_rng(0).integers(0, 3, 100)
            data[np.arange(100), 4 + classes] = 1.0
            np.savetxt(f.name, data)
            path = Path(f.name)

        x_train, x_val, x_test, y_train, y_val, y_test = load_and_split(
            path, val_split=0.15, test_split=0.15, random_seed=42
        )

        assert x_train.shape[0] == 70
        assert x_val.shape[0] == 15
        assert x_test.shape[0] == 15
        assert x_train.shape[1] == 4
        assert y_train.shape[1] == 3
        assert y_val.shape[1] == 3
        assert y_test.shape[1] == 3

        path.unlink()

    def test_load_and_split_is_deterministic(self) -> None:
        """Same seed should produce identical splits."""
        with tempfile.NamedTemporaryFile(suffix=".dat", mode="w", delete=False) as f:
            data = np.random.default_rng(0).standard_normal((30, 7))
            np.savetxt(f.name, data)
            path = Path(f.name)

        result_1 = load_and_split(path, val_split=0.2, test_split=0.1, random_seed=99)
        result_2 = load_and_split(path, val_split=0.2, test_split=0.1, random_seed=99)

        for arr1, arr2 in zip(result_1, result_2, strict=True):
            np.testing.assert_array_equal(arr1, arr2)

        path.unlink()


class TestEvaluate:
    def test_compute_accuracy_perfect(self) -> None:
        predictions = np.array([[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0]])
        targets = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        assert compute_accuracy(predictions, targets) == 1.0

    def test_compute_accuracy_zero(self) -> None:
        predictions = np.array([[10.0, -10.0, -10.0], [-10.0, -10.0, 10.0]])
        targets = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        assert compute_accuracy(predictions, targets) == 0.0

    def test_evaluate_returns_expected_keys(self) -> None:
        model = MultiLayerNetwork(4, [16, 3], ["relu", "identity"], rng=np.random.default_rng(42))
        x = np.random.default_rng(0).standard_normal((10, 4))
        y = np.zeros((10, 3))
        y[np.arange(10), np.random.default_rng(0).integers(0, 3, 10)] = 1.0

        metrics = evaluate(model, x, y, "val")

        assert "val_loss" in metrics
        assert "val_accuracy" in metrics
        assert 0.0 <= metrics["val_accuracy"] <= 1.0
        assert metrics["val_loss"] > 0.0
