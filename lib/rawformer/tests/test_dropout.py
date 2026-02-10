"""Tests for Dropout."""

import numpy as np
import pytest

from rawformer.exceptions import ForwardNotCalledError
from rawformer.layers.dropout import Dropout


class TestDropout:
    def test_training_zeros_some_elements(self) -> None:
        dropout = Dropout(0.5, rng=np.random.default_rng(0))
        x = np.ones((4, 8))
        out = dropout.forward(x)
        assert np.any(out == 0.0)
        assert np.any(out != 0.0)

    def test_training_scales_kept_elements(self) -> None:
        """Kept elements should be scaled by 1/(1-rate) to preserve expectation."""
        dropout = Dropout(0.5, rng=np.random.default_rng(0))
        x = np.ones((4, 8))
        out = dropout.forward(x)
        nonzero_values = out[out != 0.0]
        np.testing.assert_allclose(nonzero_values, 1.0 / 0.5)

    def test_eval_mode_is_passthrough(self) -> None:
        dropout = Dropout(0.5, rng=np.random.default_rng(0))
        dropout.training = False
        x = np.random.default_rng(0).standard_normal((4, 8))
        out = dropout.forward(x)
        np.testing.assert_array_equal(out, x)

    def test_backward_applies_same_mask(self) -> None:
        dropout = Dropout(0.5, rng=np.random.default_rng(0))
        x = np.ones((4, 8))
        out = dropout.forward(x)
        grad = np.ones_like(x)
        grad_out = dropout.backward(grad)
        np.testing.assert_array_equal(out, grad_out)

    def test_backward_passthrough_without_mask(self) -> None:
        """Backward should pass through when no mask was cached (eval mode)."""
        dropout = Dropout(0.5, rng=np.random.default_rng(0))
        dropout.training = False
        x = np.ones((4, 8))
        dropout.forward(x)
        grad = np.random.default_rng(0).standard_normal((4, 8))
        grad_out = dropout.backward(grad)
        np.testing.assert_array_equal(grad_out, grad)

    def test_rate_zero_is_passthrough(self) -> None:
        dropout = Dropout(0.0, rng=np.random.default_rng(0))
        x = np.random.default_rng(0).standard_normal((4, 8))
        out = dropout.forward(x)
        np.testing.assert_array_equal(out, x)

    def test_deterministic_with_same_seed(self) -> None:
        x = np.ones((4, 8))
        out1 = Dropout(0.5, rng=np.random.default_rng(42)).forward(x)
        out2 = Dropout(0.5, rng=np.random.default_rng(42)).forward(x)
        np.testing.assert_array_equal(out1, out2)

    def test_backward_raises_without_forward(self) -> None:
        dropout = Dropout(0.5, rng=np.random.default_rng(0))
        with pytest.raises(ForwardNotCalledError):
            dropout.backward(np.ones((4, 8)))
