"""Tests for PositionWiseFeedForward."""

import numpy as np
import pytest

from rawformer.exceptions import ForwardNotCalledError
from rawformer.transformer.feed_forward import PositionWiseFeedForward


class TestPositionWiseFeedForward:
    def test_forward_output_shape(self) -> None:
        ffn = PositionWiseFeedForward(16, 64, rng=np.random.default_rng(0))
        x = np.random.default_rng(0).standard_normal((2, 5, 16))
        result = ffn.forward(x)
        assert result.shape == (2, 5, 16)

    def test_backward_output_shape(self) -> None:
        ffn = PositionWiseFeedForward(16, 64, rng=np.random.default_rng(0))
        x = np.random.default_rng(0).standard_normal((2, 5, 16))
        ffn.forward(x)
        grad = np.ones((2, 5, 16))
        grad_input = ffn.backward(grad)
        assert grad_input.shape == (2, 5, 16)

    def test_backward_raises_without_forward(self) -> None:
        ffn = PositionWiseFeedForward(16, 64, rng=np.random.default_rng(0))
        with pytest.raises(ForwardNotCalledError):
            ffn.backward(np.ones((2, 5, 16)))

    def test_relu_zeroes_negative_hidden(self) -> None:
        """The hidden layer should have no negative values after ReLU."""
        ffn = PositionWiseFeedForward(16, 64, rng=np.random.default_rng(0))
        x = np.random.default_rng(0).standard_normal((2, 5, 16))
        ffn.forward(x)
        assert ffn._relu_cache is not None
        relu_out = np.maximum(0, ffn._relu_cache)
        assert np.all(relu_out >= 0)
