"""Training stage: build network from params, train, save model artifact."""

import json
import logging
from typing import TypedDict

import numpy as np
import numpy.typing as npt
import yaml

from nn_train._paths import ARTIFACTS_DIR, PARAMS_PATH
from rawformer import MultiLayerNetwork, Trainer, TrainerHyperparams
from rawformer.network import ActivationType
from rawformer.trainer import LossType

logger = logging.getLogger(__name__)


class TrainParams(TypedDict):
    """Combined network architecture and trainer params from params.yaml."""

    neurons: list[int]
    activations: list[str]
    batch_size: int
    epochs: int
    learning_rate: float
    loss: str
    shuffle: bool


def _load_params() -> TrainParams:
    """Load the train stage parameters from params.yaml."""
    with open(PARAMS_PATH) as f:
        all_params: dict[str, TrainParams] = yaml.safe_load(f)
    stage = "train"
    if stage not in all_params:
        raise ValueError(f"Missing {stage!r} section in {PARAMS_PATH}")
    return all_params[stage]


def _load_training_data() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Load the preprocessed training arrays from artifacts/."""
    logger.info("Loading training data from %s", ARTIFACTS_DIR)
    x_train: npt.NDArray[np.float64] = np.load(ARTIFACTS_DIR / "train_x.npy")
    y_train: npt.NDArray[np.float64] = np.load(ARTIFACTS_DIR / "train_y.npy")
    return x_train, y_train


# Identity dicts that let mypy narrow str -> Literal without type: ignore.
_ACTIVATION_LITERALS: dict[str, ActivationType] = {
    "relu": "relu",
    "sigmoid": "sigmoid",
    "tanh": "tanh",
    "identity": "identity",
}

_LOSS_LITERALS: dict[str, LossType] = {
    "mse": "mse",
    "cross_entropy": "cross_entropy",
}


def _validate_activations(raw: list[str]) -> list[ActivationType]:
    """Validate activation names from params.yaml against known literals."""
    validated: list[ActivationType] = []
    for name in raw:
        literal = _ACTIVATION_LITERALS.get(name)
        if literal is None:
            raise ValueError(
                f"Unknown activation {name!r} in params.yaml. "
                f"Choose from {sorted(_ACTIVATION_LITERALS)}"
            )
        validated.append(literal)
    return validated


def _validate_loss(raw: str) -> LossType:
    """Validate a loss name from params.yaml against known literals."""
    literal = _LOSS_LITERALS.get(raw)
    if literal is None:
        raise ValueError(
            f"Unknown loss {raw!r} in params.yaml. Choose from {sorted(_LOSS_LITERALS)}"
        )
    return literal


def _build_and_train(
    x_train: npt.NDArray[np.float64],
    y_train: npt.NDArray[np.float64],
    params: TrainParams,
) -> tuple[MultiLayerNetwork, Trainer]:
    """Construct the network, train it, and return both.

    Args:
        x_train: Preprocessed training features.
        y_train: Training target labels.
        params: Train stage parameters from params.yaml.

    Returns:
        Tuple of the trained network and its trainer.
    """
    input_dim = x_train.shape[1]
    activations = _validate_activations(params["activations"])
    loss = _validate_loss(params["loss"])

    network = MultiLayerNetwork(
        input_dim=input_dim,
        neurons=params["neurons"],
        activations=activations,
    )

    hyperparams: TrainerHyperparams = {
        "batch_size": params["batch_size"],
        "epochs": params["epochs"],
        "learning_rate": params["learning_rate"],
        "loss": loss,
        "shuffle": params["shuffle"],
    }

    trainer = Trainer(network=network, hyperparams=hyperparams)

    logger.info(
        "Training: %d epochs, batch_size=%d, lr=%s",
        params["epochs"],
        params["batch_size"],
        params["learning_rate"],
    )
    trainer.train(x_train, y_train)
    return network, trainer


def _save_model(network: MultiLayerNetwork) -> None:
    """Serialize the trained model to artifacts/."""
    model_path = ARTIFACTS_DIR / "model.pkl"
    network.save(model_path)
    logger.info("Model saved to %s", model_path)


def _save_metrics(train_loss: float, params: TrainParams) -> None:
    """Write training metrics to artifacts/train-metrics.json."""
    metrics = {
        "final_train_loss": round(train_loss, 6),
        "n_epochs": params["epochs"],
        "batch_size": params["batch_size"],
        "learning_rate": params["learning_rate"],
    }
    metrics_path = ARTIFACTS_DIR / "train-metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Train metrics written to %s", metrics_path)


def main() -> None:
    """Orchestrate the train stage."""
    # Setup
    params = _load_params()
    x_train, y_train = _load_training_data()

    # Train
    network, trainer = _build_and_train(x_train, y_train, params)

    # Evaluate
    train_loss = trainer.eval_loss(x_train, y_train)
    logger.info("Final training loss: %.6f", train_loss)

    # Save
    _save_model(network)
    _save_metrics(train_loss, params)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
