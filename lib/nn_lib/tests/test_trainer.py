"""Tests for the Trainer class."""

import numpy as np
import pytest

from nn_lib.network import MultiLayerNetwork
from nn_lib.trainer import Trainer, TrainerHyperparams


class TestTrainer:
    def test_training_reduces_loss(self) -> None:
        """Training for multiple epochs should reduce the loss."""
        net = MultiLayerNetwork(4, [16, 3], ["relu", "identity"], rng=np.random.default_rng(42))
        x = np.random.default_rng(0).standard_normal((20, 4))
        y = np.zeros((20, 3))
        y[np.arange(20), np.random.default_rng(0).integers(0, 3, 20)] = 1.0

        hyperparams: TrainerHyperparams = {
            "batch_size": 8,
            "epochs": 1,
            "learning_rate": 0.01,
            "loss": "cross_entropy",
            "shuffle": False,
        }
        trainer = Trainer(network=net, hyperparams=hyperparams)
        loss_before = trainer.eval_loss(x, y)

        trainer.epochs = 200
        trainer.train(x, y)
        loss_after = trainer.eval_loss(x, y)

        assert loss_after < loss_before

    def test_mse_loss_training(self) -> None:
        """Verify that MSE loss mode works and reduces loss."""
        net = MultiLayerNetwork(2, [8, 1], ["sigmoid", "identity"], rng=np.random.default_rng(42))
        x = np.random.default_rng(0).standard_normal((16, 2))
        y = np.random.default_rng(0).standard_normal((16, 1))

        hyperparams: TrainerHyperparams = {
            "batch_size": 4,
            "epochs": 100,
            "learning_rate": 0.01,
            "loss": "mse",
            "shuffle": True,
        }
        trainer = Trainer(network=net, hyperparams=hyperparams)
        loss_before = trainer.eval_loss(x, y)
        trainer.train(x, y)
        loss_after = trainer.eval_loss(x, y)

        assert loss_after < loss_before

    def test_unknown_loss_raises(self) -> None:
        net = MultiLayerNetwork(4, [3], ["identity"])
        hyperparams: TrainerHyperparams = {
            "batch_size": 8,
            "epochs": 1,
            "learning_rate": 0.01,
            "loss": "bad_loss",  # type: ignore[typeddict-item]
            "shuffle": False,
        }
        with pytest.raises(ValueError, match="Unknown loss"):
            Trainer(network=net, hyperparams=hyperparams)

    def test_eval_loss_does_not_change_weights(self) -> None:
        net = MultiLayerNetwork(4, [3], ["identity"], rng=np.random.default_rng(42))
        x = np.random.default_rng(0).standard_normal((4, 4))
        y = np.zeros((4, 3))
        y[np.arange(4), [0, 1, 2, 0]] = 1.0

        hyperparams: TrainerHyperparams = {
            "batch_size": 4,
            "epochs": 1,
            "learning_rate": 0.01,
            "loss": "cross_entropy",
            "shuffle": False,
        }
        trainer = Trainer(network=net, hyperparams=hyperparams)

        output_before = net.forward(x).copy()
        trainer.eval_loss(x, y)
        output_after = net.forward(x)

        np.testing.assert_array_equal(output_before, output_after)
