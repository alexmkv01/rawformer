"""Data preprocessing with min-max normalization and NaN imputation."""

import numpy as np
import numpy.typing as npt


class Preprocessor:
    """Min-max normalization to [0, 1] with NaN-safe median imputation.

    Fit on training data, then apply to both train and validation sets.
    Handles constant features (zero range) without producing NaN/inf —
    constant features map to 0.0 after normalization.

    Example:
        >>> prep = Preprocessor(x_train)
        >>> x_train_norm = prep.apply(x_train)
        >>> x_val_norm = prep.apply(x_val)
        >>> x_original = prep.revert(x_train_norm)
    """

    def __init__(self, data: npt.NDArray[np.float64]) -> None:
        self._min: npt.NDArray[np.float64] = np.nanmin(data, axis=0)
        self._max: npt.NDArray[np.float64] = np.nanmax(data, axis=0)
        raw_range = self._max - self._min
        self._range: npt.NDArray[np.float64] = np.where(raw_range == 0, 1.0, raw_range)
        self._medians: npt.NDArray[np.float64] = np.nanmedian(data, axis=0)

    def apply(self, data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Normalize data to [0, 1] range, imputing NaN with column medians."""
        # medians computed at fit time
        result = np.where(np.isnan(data), self._medians, data)
        return (result - self._min) / self._range

    def revert(self, data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Reverse the normalization back to the original scale."""
        return data * self._range + self._min
