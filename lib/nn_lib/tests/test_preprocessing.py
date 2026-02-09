"""Tests for the Preprocessor class."""

import numpy as np

from nn_lib.preprocessing import Preprocessor


class TestPreprocessor:
    def test_apply_normalizes_to_zero_one(self) -> None:
        data = np.array([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]])
        prep = Preprocessor(data)
        result = prep.apply(data)
        np.testing.assert_allclose(np.min(result, axis=0), [0.0, 0.0])
        np.testing.assert_allclose(np.max(result, axis=0), [1.0, 1.0])

    def test_revert_roundtrip(self) -> None:
        data = np.array([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]])
        prep = Preprocessor(data)
        normalized = prep.apply(data)
        reverted = prep.revert(normalized)
        np.testing.assert_allclose(reverted, data)

    def test_apply_does_not_mutate_input(self) -> None:
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        original = data.copy()
        prep = Preprocessor(data)
        prep.apply(data)
        np.testing.assert_array_equal(data, original)

    def test_constant_feature_no_nan(self) -> None:
        """A column with zero range should not produce NaN/inf."""
        data = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
        prep = Preprocessor(data)
        result = prep.apply(data)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_nan_imputation(self) -> None:
        """NaN values should be replaced with the column median."""
        data = np.array([[1.0, 2.0], [np.nan, 4.0], [3.0, np.nan]])
        prep = Preprocessor(data)
        result = prep.apply(data)
        assert not np.any(np.isnan(result))

    def test_apply_on_unseen_data(self) -> None:
        """Preprocessing should work on data not seen during fitting."""
        train = np.array([[0.0, 0.0], [10.0, 10.0]])
        test = np.array([[5.0, 5.0]])
        prep = Preprocessor(train)
        result = prep.apply(test)
        np.testing.assert_allclose(result, [[0.5, 0.5]])

    def test_single_sample(self) -> None:
        """Single-row data should not crash (all ranges are zero)."""
        data = np.array([[3.0, 7.0]])
        prep = Preprocessor(data)
        result = prep.apply(data)
        assert not np.any(np.isnan(result))
