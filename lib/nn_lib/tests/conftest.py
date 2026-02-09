"""Shared test fixtures and helpers for nn_lib tests."""

import numpy as np
import numpy.typing as npt
import pytest

from nn_lib.base import Layer

NUMERICAL_EPS: float = 1e-5


# Loss tests use their own inline numerical gradient loop because Loss.forward
# takes two args (predictions, targets) and returns a scalar directly — a
# different signature from Layer.forward.
def numerical_gradient(
    layer: Layer,
    x: npt.NDArray[np.float64],
    eps: float = NUMERICAL_EPS,
) -> npt.NDArray[np.float64]:
    """Compute numerical gradient via central difference for a layer.

    Sums the layer output to produce a scalar objective, then computes
    the gradient of that scalar with respect to each element of x.

    Args:
        layer: Any Layer whose forward() maps (batch, in) -> (batch, out).
        x: Input array to differentiate with respect to.
        eps: Finite difference step size.
    """
    grad = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[idx] += eps
        x_minus[idx] -= eps
        grad[idx] = (np.sum(layer.forward(x_plus)) - np.sum(layer.forward(x_minus))) / (2 * eps)
    return grad


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def sample_input(rng: np.random.Generator) -> npt.NDArray[np.float64]:
    """Batch of 8 samples with 4 features."""
    return rng.standard_normal((8, 4))


@pytest.fixture
def sample_targets_onehot(rng: np.random.Generator) -> npt.NDArray[np.float64]:
    """One-hot encoded targets for 3 classes, batch size 8."""
    targets = np.zeros((8, 3))
    classes = rng.integers(0, 3, size=8)
    targets[np.arange(8), classes] = 1.0
    return targets
