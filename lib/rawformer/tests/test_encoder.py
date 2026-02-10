"""Tests for EncoderBlock and Encoder."""

import numpy as np

from rawformer.transformer.encoder import Encoder, EncoderBlock


class TestEncoderBlock:
    def test_forward_output_shape(self) -> None:
        block = EncoderBlock(16, 4, 64, rng=np.random.default_rng(0), dropout_rate=0.0)
        x = np.random.default_rng(0).standard_normal((2, 5, 16))
        result = block.forward(x)
        assert result.shape == (2, 5, 16)

    def test_backward_output_shape(self) -> None:
        block = EncoderBlock(16, 4, 64, rng=np.random.default_rng(0), dropout_rate=0.0)
        x = np.random.default_rng(0).standard_normal((2, 5, 16))
        block.forward(x)
        grad = np.ones((2, 5, 16))
        grad_input = block.backward(grad)
        assert grad_input.shape == (2, 5, 16)


class TestEncoder:
    def test_forward_output_shape(self) -> None:
        encoder = Encoder(3, 16, 4, 64, rng=np.random.default_rng(0), dropout_rate=0.0)
        x = np.random.default_rng(0).standard_normal((2, 5, 16))
        result = encoder.forward(x)
        assert result.shape == (2, 5, 16)

    def test_backward_output_shape(self) -> None:
        encoder = Encoder(3, 16, 4, 64, rng=np.random.default_rng(0), dropout_rate=0.0)
        x = np.random.default_rng(0).standard_normal((2, 5, 16))
        encoder.forward(x)
        grad = np.ones((2, 5, 16))
        grad_input = encoder.backward(grad)
        assert grad_input.shape == (2, 5, 16)

    def test_output_changes_with_different_input(self) -> None:
        encoder = Encoder(2, 16, 4, 64, rng=np.random.default_rng(0), dropout_rate=0.0)
        x1 = np.random.default_rng(0).standard_normal((2, 5, 16))
        x2 = np.random.default_rng(1).standard_normal((2, 5, 16))
        out1 = encoder.forward(x1)
        out2 = encoder.forward(x2)
        assert not np.array_equal(out1, out2)
