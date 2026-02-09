"""Tests for the full Transformer model."""

import numpy as np
import pytest

from rawformer.exceptions import ForwardNotCalledError
from rawformer.transformer.transformer import Transformer


class TestTransformer:
    def test_forward_output_shape(self) -> None:
        model = Transformer(
            src_vocab_size=50,
            tgt_vocab_size=50,
            d_model=16,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            d_ff=64,
            max_len=100,
            rng=np.random.default_rng(0),
        )
        src = np.array([[1, 5, 10, 3], [2, 7, 0, 4]], dtype=np.float64)
        tgt = np.array([[1, 3, 5], [2, 4, 6]], dtype=np.float64)
        logits = model.forward(src, tgt)
        assert logits.shape == (2, 3, 50)

    def test_backward_runs_without_error(self) -> None:
        model = Transformer(
            src_vocab_size=50,
            tgt_vocab_size=50,
            d_model=16,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            d_ff=64,
            max_len=100,
            rng=np.random.default_rng(0),
        )
        src = np.array([[1, 5, 10, 3]], dtype=np.float64)
        tgt = np.array([[1, 3, 5]], dtype=np.float64)
        logits = model.forward(src, tgt)
        grad = np.ones_like(logits)
        model.backward(grad)

    def test_backward_raises_without_forward(self) -> None:
        model = Transformer(
            src_vocab_size=50,
            tgt_vocab_size=50,
            d_model=16,
            n_heads=4,
            n_encoder_layers=1,
            n_decoder_layers=1,
            d_ff=32,
            max_len=50,
            rng=np.random.default_rng(0),
        )
        with pytest.raises(ForwardNotCalledError):
            model.backward(np.ones((1, 3, 50)))

    def test_update_params_changes_output(self) -> None:
        model = Transformer(
            src_vocab_size=50,
            tgt_vocab_size=50,
            d_model=16,
            n_heads=4,
            n_encoder_layers=1,
            n_decoder_layers=1,
            d_ff=32,
            max_len=50,
            rng=np.random.default_rng(0),
        )
        src = np.array([[1, 5, 10]], dtype=np.float64)
        tgt = np.array([[1, 3]], dtype=np.float64)
        logits_before = model.forward(src, tgt).copy()
        model.backward(np.ones_like(logits_before))
        model.update_params(learning_rate=0.01)
        logits_after = model.forward(src, tgt)
        assert not np.array_equal(logits_before, logits_after)

    def test_different_src_tgt_vocab_sizes(self) -> None:
        model = Transformer(
            src_vocab_size=30,
            tgt_vocab_size=40,
            d_model=16,
            n_heads=4,
            n_encoder_layers=1,
            n_decoder_layers=1,
            d_ff=32,
            max_len=50,
            rng=np.random.default_rng(0),
        )
        src = np.array([[1, 5, 10]], dtype=np.float64)
        tgt = np.array([[1, 3]], dtype=np.float64)
        logits = model.forward(src, tgt)
        assert logits.shape == (1, 2, 40)
