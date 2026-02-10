"""Tests for TokenEmbedding and PositionalEncoding."""

import numpy as np
import pytest
import torch

from rawformer.core.exceptions import ForwardNotCalledError
from rawformer.layers.embedding import (
    PositionalEncoding,
    TokenEmbedding,
    sinusoidal_positional_encoding,
)


class TestTokenEmbedding:
    def test_forward_output_shape(self) -> None:
        emb = TokenEmbedding(100, 16, rng=np.random.default_rng(0))
        ids = np.array([[1, 5, 10], [0, 3, 7]], dtype=np.intp)
        result = emb.forward(ids)
        assert result.shape == (2, 3, 16)

    def test_forward_matches_pytorch(self) -> None:
        """Cross-verify forward pass against torch.nn.Embedding."""
        vocab, d_model = 50, 16
        rng = np.random.default_rng(42)
        emb = TokenEmbedding(vocab, d_model, rng=rng)

        torch_emb = torch.nn.Embedding(vocab, d_model, dtype=torch.float64)
        torch_emb.weight.data = torch.from_numpy(emb.weight.copy())

        ids = np.array([[1, 5, 10], [0, 3, 7]], dtype=np.intp)
        result_np = emb.forward(ids)

        ids_torch = torch.tensor([[1, 5, 10], [0, 3, 7]], dtype=torch.long)
        result_torch = torch_emb(ids_torch).detach().numpy() * np.sqrt(np.float64(d_model))

        np.testing.assert_allclose(result_np, result_torch, atol=1e-10)

    def test_backward_accumulates_gradients(self) -> None:
        emb = TokenEmbedding(10, 4, rng=np.random.default_rng(0))
        ids = np.array([[1, 1, 2]], dtype=np.intp)
        emb.forward(ids)
        grad = np.ones((1, 3, 4))
        emb.backward(grad)
        assert emb._grad_weight is not None
        assert emb._grad_weight[1].sum() != 0.0

    def test_backward_raises_without_forward(self) -> None:
        emb = TokenEmbedding(10, 4, rng=np.random.default_rng(0))
        with pytest.raises(ForwardNotCalledError):
            emb.backward(np.ones((1, 3, 4)))


class TestSinusoidalPositionalEncoding:
    def test_output_shape(self) -> None:
        pe = sinusoidal_positional_encoding(100, 16)
        assert pe.shape == (100, 16)

    def test_values_bounded(self) -> None:
        pe = sinusoidal_positional_encoding(100, 32)
        assert np.all(pe >= -1.0)
        assert np.all(pe <= 1.0)

    def test_position_zero_sin_is_zero(self) -> None:
        """At position 0, all sin terms should be 0."""
        pe = sinusoidal_positional_encoding(10, 16)
        np.testing.assert_allclose(pe[0, 0::2], 0.0, atol=1e-10)

    def test_position_zero_cos_is_one(self) -> None:
        """At position 0, all cos terms should be 1."""
        pe = sinusoidal_positional_encoding(10, 16)
        np.testing.assert_allclose(pe[0, 1::2], 1.0, atol=1e-10)


class TestPositionalEncoding:
    def test_forward_output_shape(self) -> None:
        pos_enc = PositionalEncoding(100, 16)
        x = np.random.default_rng(0).standard_normal((2, 10, 16))
        result = pos_enc.forward(x)
        assert result.shape == (2, 10, 16)

    def test_forward_adds_encoding(self) -> None:
        """Output should differ from input by exactly the PE table."""
        pos_enc = PositionalEncoding(100, 16)
        x = np.random.default_rng(0).standard_normal((2, 10, 16))
        result = pos_enc.forward(x)
        diff = result - x
        expected_pe = pos_enc.encoding_table[:10]
        np.testing.assert_allclose(diff[0], expected_pe, atol=1e-12)
        np.testing.assert_allclose(diff[1], expected_pe, atol=1e-12)

    def test_backward_passthrough(self) -> None:
        pos_enc = PositionalEncoding(100, 16)
        grad = np.random.default_rng(0).standard_normal((2, 10, 16))
        result = pos_enc.backward(grad)
        np.testing.assert_array_equal(result, grad)
