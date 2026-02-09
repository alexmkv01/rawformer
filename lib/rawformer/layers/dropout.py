"""Dropout regularization (Srivastava et al., 2014).

During training, randomly zeroes elements with probability `rate` and scales
the remaining elements by 1/(1-rate) (inverted dropout).  During inference
the input is returned unchanged.
"""

import numpy as np
import numpy.typing as npt


class Dropout:
    """Inverted dropout layer.

    Args:
        rate: Probability of zeroing each element (0 = no dropout).
        rng: NumPy random generator for mask sampling.
    """

    def __init__(self, rate: float, rng: np.random.Generator) -> None:
        self.rate = rate
        self.training = True
        self._rng = rng
        self._mask_cache: npt.NDArray[np.float64] | None = None

    def forward(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Apply dropout to the input.

        Args:
            x: Input array of arbitrary shape.

        Returns:
            Masked and scaled array during training, unchanged input
            during inference.
        """
        if not self.training or self.rate == 0.0:
            self._mask_cache = None
            return x
        keep = 1.0 - self.rate
        self._mask_cache = (self._rng.binomial(1, keep, x.shape) / keep).astype(np.float64)
        result: npt.NDArray[np.float64] = x * self._mask_cache
        return result

    def backward(self, grad_z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Backward pass through dropout.

        Applies the same mask that was used in the forward pass.

        Args:
            grad_z: Upstream gradient, same shape as forward input.
        """
        if self._mask_cache is None:
            return grad_z
        result: npt.NDArray[np.float64] = grad_z * self._mask_cache
        return result
