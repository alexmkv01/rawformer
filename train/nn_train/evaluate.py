"""Evaluation stage: load model, compute metrics, write evaluate-metrics.json."""

import json
import logging

import numpy as np
import numpy.typing as npt

from nn_train._paths import ARTIFACTS_DIR
from rawformer import MultiLayerNetwork
from rawformer.losses import CrossEntropyLoss

logger = logging.getLogger(__name__)


def _load_model() -> MultiLayerNetwork:
    """Load the trained model from artifacts/."""
    logger.info("Loading model from %s", ARTIFACTS_DIR)
    return MultiLayerNetwork.load(ARTIFACTS_DIR / "model.pkl")


def _load_split(name: str) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Load a preprocessed data split (val or test) from artifacts/."""
    x: npt.NDArray[np.float64] = np.load(ARTIFACTS_DIR / f"{name}_x.npy")
    y: npt.NDArray[np.float64] = np.load(ARTIFACTS_DIR / f"{name}_y.npy")
    return x, y


def compute_accuracy(
    predictions: npt.NDArray[np.float64],
    targets: npt.NDArray[np.float64],
) -> float:
    """Compute classification accuracy from one-hot predictions and targets."""
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(targets, axis=1)
    return float(np.mean(pred_classes == true_classes))


def evaluate(
    model: MultiLayerNetwork,
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    split_name: str,
) -> dict[str, float]:
    """Run evaluation on a single data split and return prefixed metrics.

    Args:
        model: Trained neural network.
        x: Preprocessed features.
        y: Target labels (one-hot).
        split_name: Label for the split (e.g. "val", "test"), used as metric key prefix.

    Returns:
        Dictionary with {split_name}_loss and {split_name}_accuracy.
    """
    predictions = model.forward(x)

    loss_fn = CrossEntropyLoss()
    loss = loss_fn.forward(predictions, y)
    accuracy = compute_accuracy(predictions, y)

    logger.info("%s loss: %.6f", split_name, loss)
    logger.info("%s accuracy: %.2f%%", split_name, accuracy * 100)

    return {
        f"{split_name}_loss": round(loss, 6),
        f"{split_name}_accuracy": round(accuracy, 4),
    }


def _save_metrics(metrics: dict[str, float]) -> None:
    """Write metrics dict to artifacts/evaluate-metrics.json."""
    metrics_path = ARTIFACTS_DIR / "evaluate-metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics written to %s", metrics_path)


def main() -> None:
    """Orchestrate the evaluate stage."""
    model = _load_model()

    x_val, y_val = _load_split("val")
    x_test, y_test = _load_split("test")

    metrics: dict[str, float] = {}
    metrics.update(evaluate(model, x_val, y_val, "val"))
    metrics.update(evaluate(model, x_test, y_test, "test"))

    _save_metrics(metrics)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
