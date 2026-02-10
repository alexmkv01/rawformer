"""Layer normalization (Ba, Kiros & Hinton, 2016)."""

from typing import TypedDict

import numpy as np
import numpy.typing as npt

from rawformer.base import SimpleLayer
from rawformer.exceptions import ForwardNotCalledError


class _LayerNormCache(TypedDict):
    x_hat: npt.NDArray[np.float64]
    inv_std: npt.NDArray[np.float64]


class LayerNorm(SimpleLayer):
    """Layer normalization over the last dimension.

    Normalizes activations to zero mean and unit variance, then applies
    a learnable affine transform: y = gamma * x_hat + beta.

    Args:
        d_model: Size of the last dimension to normalize over.
        eps: Small constant for numerical stability in the variance
            denominator.
    """

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        self.d_model = d_model
        self._eps = eps
        self._gamma: npt.NDArray[np.float64] = np.ones(d_model)
        self._beta: npt.NDArray[np.float64] = np.zeros(d_model)

        self._cache: _LayerNormCache | None = None
        self._grad_gamma: npt.NDArray[np.float64] | None = None
        self._grad_beta: npt.NDArray[np.float64] | None = None

    @property
    def gamma(self) -> npt.NDArray[np.float64]:
        return self._gamma

    @property
    def beta(self) -> npt.NDArray[np.float64]:
        return self._beta

    def forward(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Normalize over the last dimension.

        Args:
            x: Input of shape (..., d_model).
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        inv_std: npt.NDArray[np.float64] = 1.0 / np.sqrt(var + self._eps)
        x_hat: npt.NDArray[np.float64] = (x - mean) * inv_std
        self._cache = _LayerNormCache(x_hat=x_hat, inv_std=inv_std)
        result: npt.NDArray[np.float64] = self._gamma * x_hat + self._beta
        return result

    def backward(self, grad_z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute gradients for input, gamma, and beta.

        Args:
            grad_z: Upstream gradient, same shape as forward output.
        """
        if self._cache is None:
            raise ForwardNotCalledError("LayerNorm")
        x_hat = self._cache["x_hat"]
        inv_std = self._cache["inv_std"]
        d = self.d_model

        self._grad_gamma = np.sum((grad_z * x_hat).reshape(-1, d), axis=0)
        self._grad_beta = np.sum(grad_z.reshape(-1, d), axis=0)

        # LayerNorm gradient (Ba et al., 2016, S3.1)
        dx_hat = grad_z * self._gamma
        dx: npt.NDArray[np.float64] = (
            inv_std
            / d
            * (
                d * dx_hat
                - np.sum(dx_hat, axis=-1, keepdims=True)
                - x_hat * np.sum(dx_hat * x_hat, axis=-1, keepdims=True)
            )
        )
        return dx

    def update_params(self, learning_rate: float) -> None:
        if self._grad_gamma is None or self._grad_beta is None:
            raise ForwardNotCalledError("LayerNorm")
        self._gamma -= learning_rate * self._grad_gamma
        self._beta -= learning_rate * self._grad_beta
