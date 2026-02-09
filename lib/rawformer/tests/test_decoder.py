"""Tests for DecoderBlock, Decoder, and causal mask."""

import numpy as np

from rawformer.transformer.decoder import Decoder, DecoderBlock, causal_mask


class TestCausalMask:
    def test_shape(self) -> None:
        mask = causal_mask(5)
        assert mask.shape == (1, 1, 5, 5)

    def test_diagonal_and_below_are_zero(self) -> None:
        mask = causal_mask(5)
        for i in range(5):
            for j in range(i + 1):
                assert mask[0, 0, i, j] == 0.0

    def test_above_diagonal_is_neg_inf(self) -> None:
        mask = causal_mask(5)
        for i in range(5):
            for j in range(i + 1, 5):
                assert mask[0, 0, i, j] == -np.inf


class TestDecoderBlock:
    def test_forward_output_shape(self) -> None:
        block = DecoderBlock(16, 4, 64, rng=np.random.default_rng(0))
        x = np.random.default_rng(0).standard_normal((2, 5, 16))
        enc_out = np.random.default_rng(1).standard_normal((2, 8, 16))
        mask = causal_mask(5)
        result = block.forward(x, enc_out, self_attn_mask=mask)
        assert result.shape == (2, 5, 16)

    def test_backward_output_shapes(self) -> None:
        block = DecoderBlock(16, 4, 64, rng=np.random.default_rng(0))
        x = np.random.default_rng(0).standard_normal((2, 5, 16))
        enc_out = np.random.default_rng(1).standard_normal((2, 8, 16))
        mask = causal_mask(5)
        block.forward(x, enc_out, self_attn_mask=mask)
        grad = np.ones((2, 5, 16))
        grad_x, grad_enc = block.backward(grad)
        assert grad_x.shape == (2, 5, 16)
        assert grad_enc.shape == (2, 8, 16)


class TestDecoder:
    def test_forward_output_shape(self) -> None:
        decoder = Decoder(3, 16, 4, 64, rng=np.random.default_rng(0))
        x = np.random.default_rng(0).standard_normal((2, 5, 16))
        enc_out = np.random.default_rng(1).standard_normal((2, 8, 16))
        mask = causal_mask(5)
        result = decoder.forward(x, enc_out, self_attn_mask=mask)
        assert result.shape == (2, 5, 16)

    def test_backward_output_shapes(self) -> None:
        decoder = Decoder(3, 16, 4, 64, rng=np.random.default_rng(0))
        x = np.random.default_rng(0).standard_normal((2, 5, 16))
        enc_out = np.random.default_rng(1).standard_normal((2, 8, 16))
        mask = causal_mask(5)
        decoder.forward(x, enc_out, self_attn_mask=mask)
        grad = np.ones((2, 5, 16))
        grad_x, grad_enc = decoder.backward(grad)
        assert grad_x.shape == (2, 5, 16)
        assert grad_enc.shape == (2, 8, 16)
