"""Mini-batch gradient descent trainer for neural networks."""

from typing import Literal, NotRequired, TypedDict

import numpy as np
import numpy.typing as npt

from nn_lib.losses import CrossEntropyLoss, Loss, MSELoss
from nn_lib.network import MultiLayerNetwork

LossType = Literal["mse", "cross_entropy"]

_LOSS_MAP: dict[str, type[Loss]] = {
    "mse": MSELoss,
    "cross_entropy": CrossEntropyLoss,
}


class TrainerHyperparams(TypedDict):
    """Typed hyperparameters for the Trainer."""

    batch_size: int
    epochs: int
    learning_rate: float
    loss: LossType
    shuffle: bool
    seed: NotRequired[int]


class Trainer:
    """Mini-batch stochastic gradient descent trainer.

    Supports MSE and cross-entropy loss functions with optional
    data shuffling between epochs.
    """

    def __init__(
        self,
        network: MultiLayerNetwork,
        hyperparams: TrainerHyperparams,
    ) -> None:
        loss = hyperparams["loss"]
        if loss not in _LOSS_MAP:
            raise ValueError(f"Unknown loss: {loss!r}. Choose from {list(_LOSS_MAP)}")
        self.network = network
        self.batch_size = hyperparams["batch_size"]
        self.epochs = hyperparams["epochs"]
        self.learning_rate = hyperparams["learning_rate"]
        self.shuffle = hyperparams["shuffle"]
        self._loss_layer = _LOSS_MAP[loss]()
        self._rng = np.random.default_rng(hyperparams.get("seed"))

    def train(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
    ) -> None:
        """Train the network using mini-batch gradient descent.

        Args:
            x: Training input features.
            y: Training target labels.
        """
        n_samples = x.shape[0]
        for _ in range(self.epochs):
            x_epoch, y_epoch = _shuffle(self._rng, x, y) if self.shuffle else (x, y)
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                x_batch = x_epoch[start:end]
                y_batch = y_epoch[start:end]

                predictions = self.network.forward(x_batch)
                self._loss_layer.forward(predictions, y_batch)
                grad = self._loss_layer.backward()
                self.network.backward(grad)
                self.network.update_params(self.learning_rate)

    def eval_loss(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
    ) -> float:
        """Compute loss on a dataset without updating parameters.

        Warning:
            Mutates internal forward-pass caches on the network and loss
            layer. Do not interleave with training without re-forwarding.
        """
        predictions = self.network.forward(x)
        return self._loss_layer.forward(predictions, y)


def _shuffle(
    rng: np.random.Generator,
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Shuffle x and y with the same random permutation."""
    indices = rng.permutation(x.shape[0])
    return x[indices], y[indices]
