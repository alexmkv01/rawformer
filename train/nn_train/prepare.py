"""Data preparation stage: load, split, preprocess, and save artifacts."""

import json
import logging
import pickle
from pathlib import Path
from typing import TypedDict

import numpy as np
import numpy.typing as npt
import yaml

from nn_lib import Preprocessor
from nn_train._paths import ARTIFACTS_DIR, DATA_PATH, PARAMS_PATH

logger = logging.getLogger(__name__)

_N_FEATURES = 4


class PrepareParams(TypedDict):
    """Typed parameters for the prepare stage from params.yaml."""

    val_split: float
    test_split: float
    random_seed: int


def _load_params() -> PrepareParams:
    """Load the prepare stage parameters from params.yaml."""
    with open(PARAMS_PATH) as f:
        all_params: dict[str, PrepareParams] = yaml.safe_load(f)
    stage = "prepare"
    if stage not in all_params:
        raise ValueError(f"Missing {stage!r} section in {PARAMS_PATH}")
    return all_params[stage]


def load_and_split(
    data_path: Path,
    val_split: float,
    test_split: float,
    random_seed: int,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Load iris dataset and split into train/val/test sets.

    Args:
        data_path: Path to the iris.dat file (4 features + 3 one-hot labels).
        val_split: Fraction of data to reserve for validation.
        test_split: Fraction of data to reserve for testing.
        random_seed: Seed for reproducible shuffling.

    Returns:
        Tuple of (x_train, x_val, x_test, y_train, y_val, y_test).
    """
    logger.info("Loading data from %s", data_path)
    data: npt.NDArray[np.float64] = np.loadtxt(data_path)
    rng = np.random.default_rng(seed=random_seed)
    indices = rng.permutation(data.shape[0])
    data = data[indices]

    n = data.shape[0]
    n_test = int(n * test_split)
    n_val = int(n * val_split)

    test_data = data[:n_test]
    val_data = data[n_test : n_test + n_val]
    train_data = data[n_test + n_val :]

    x_train: npt.NDArray[np.float64] = train_data[:, :_N_FEATURES]
    y_train: npt.NDArray[np.float64] = train_data[:, _N_FEATURES:]
    x_val: npt.NDArray[np.float64] = val_data[:, :_N_FEATURES]
    y_val: npt.NDArray[np.float64] = val_data[:, _N_FEATURES:]
    x_test: npt.NDArray[np.float64] = test_data[:, :_N_FEATURES]
    y_test: npt.NDArray[np.float64] = test_data[:, _N_FEATURES:]

    logger.info(
        "Split: %d train, %d val, %d test", x_train.shape[0], x_val.shape[0], x_test.shape[0]
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def _fit_and_apply_preprocessor(
    x_train: npt.NDArray[np.float64],
    x_val: npt.NDArray[np.float64],
    x_test: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], Preprocessor]:
    """Fit preprocessor on training data and apply to all splits."""
    logger.info("Fitting preprocessor on %d training samples", x_train.shape[0])
    preprocessor = Preprocessor(x_train)
    return (
        preprocessor.apply(x_train),
        preprocessor.apply(x_val),
        preprocessor.apply(x_test),
        preprocessor,
    )


def _save_artifacts(
    x_train: npt.NDArray[np.float64],
    x_val: npt.NDArray[np.float64],
    x_test: npt.NDArray[np.float64],
    y_train: npt.NDArray[np.float64],
    y_val: npt.NDArray[np.float64],
    y_test: npt.NDArray[np.float64],
    preprocessor: Preprocessor,
) -> None:
    """Save prepared data arrays and fitted preprocessor to artifacts/."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(ARTIFACTS_DIR / "train_x.npy", x_train)
    np.save(ARTIFACTS_DIR / "val_x.npy", x_val)
    np.save(ARTIFACTS_DIR / "test_x.npy", x_test)
    np.save(ARTIFACTS_DIR / "train_y.npy", y_train)
    np.save(ARTIFACTS_DIR / "val_y.npy", y_val)
    np.save(ARTIFACTS_DIR / "test_y.npy", y_test)
    with open(ARTIFACTS_DIR / "preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)
    logger.info("Saved artifacts to %s", ARTIFACTS_DIR)


def _save_metrics(
    x_train: npt.NDArray[np.float64],
    x_val: npt.NDArray[np.float64],
    x_test: npt.NDArray[np.float64],
    y_train: npt.NDArray[np.float64],
) -> None:
    """Write data split metrics to artifacts/prepare-metrics.json."""
    total = x_train.shape[0] + x_val.shape[0] + x_test.shape[0]
    metrics = {
        "total_samples": total,
        "train_samples": x_train.shape[0],
        "val_samples": x_val.shape[0],
        "test_samples": x_test.shape[0],
        "train_proportion": round(x_train.shape[0] / total, 4),
        "val_proportion": round(x_val.shape[0] / total, 4),
        "test_proportion": round(x_test.shape[0] / total, 4),
        "n_features": x_train.shape[1],
        "n_classes": y_train.shape[1],
    }
    metrics_path = ARTIFACTS_DIR / "prepare-metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Prepare metrics written to %s", metrics_path)


def main() -> None:
    """Orchestrate the prepare stage."""
    # Setup
    params = _load_params()

    # Process
    x_train, x_val, x_test, y_train, y_val, y_test = load_and_split(
        DATA_PATH, params["val_split"], params["test_split"], params["random_seed"]
    )
    x_train, x_val, x_test, preprocessor = _fit_and_apply_preprocessor(x_train, x_val, x_test)

    # Save
    _save_artifacts(x_train, x_val, x_test, y_train, y_val, y_test, preprocessor)
    _save_metrics(x_train, x_val, x_test, y_train)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
