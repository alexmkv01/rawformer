"""Weight initialization strategies for neural network layers."""

import numpy as np
import numpy.typing as npt


def xavier_init(n_in: int, n_out: int, rng: np.random.Generator) -> npt.NDArray[np.float64]:
    """Xavier/Glorot uniform initialization.

    Appropriate for sigmoid and tanh activations. Draws weights from
    U(-limit, limit) where limit = sqrt(6 / (n_in + n_out)).

    Args:
        n_in: Number of input neurons.
        n_out: Number of output neurons.
        rng: NumPy random generator instance.

    Returns:
        Weight matrix of shape (n_in, n_out).
    """
    limit = np.sqrt(6.0 / (n_in + n_out))
    return rng.uniform(-limit, limit, size=(n_in, n_out))


def he_init(n_in: int, n_out: int, rng: np.random.Generator) -> npt.NDArray[np.float64]:
    """He/Kaiming normal initialization.

    Appropriate for ReLU activations. Draws weights from
    N(0, sqrt(2 / n_in)).

    Args:
        n_in: Number of input neurons.
        n_out: Number of output neurons (unused in the formula, kept for
            signature symmetry with xavier_init).
        rng: NumPy random generator instance.

    Returns:
        Weight matrix of shape (n_in, n_out).
    """
    std = np.sqrt(2.0 / n_in)
    result: npt.NDArray[np.float64] = rng.standard_normal((n_in, n_out)) * std
    return result


def zeros_init(n_out: int) -> npt.NDArray[np.float64]:
    """Zero initialization, typically used for biases.

    Args:
        n_out: Number of output neurons.

    Returns:
        Zero vector of shape (n_out,).
    """
    return np.zeros(n_out)
