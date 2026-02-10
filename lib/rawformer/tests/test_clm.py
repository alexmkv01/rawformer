"""Tests for the decoder-only causal language model and LM trainer."""

import numpy as np

from rawformer.training.clm import DecoderOnlyBlock, DecoderOnlyModel
from rawformer.training.lm_trainer import LMTrainer, clm_loss_and_grad


class TestDecoderOnlyBlock:
    def test_forward_shape(self) -> None:
        rng = np.random.default_rng(42)
        block = DecoderOnlyBlock(d_model=32, n_heads=4, d_ff=64, rng=rng, dropout_rate=0.0)
        x = rng.standard_normal((2, 8, 32))
        out = block.forward(x)
        assert out.shape == (2, 8, 32)

    def test_backward_shape(self) -> None:
        rng = np.random.default_rng(42)
        block = DecoderOnlyBlock(d_model=32, n_heads=4, d_ff=64, rng=rng, dropout_rate=0.0)
        x = rng.standard_normal((2, 8, 32))
        out = block.forward(x)
        grad = np.ones_like(out)
        grad_x = block.backward(grad)
        assert grad_x.shape == (2, 8, 32)


class TestDecoderOnlyModel:
    def test_forward_shape(self) -> None:
        rng = np.random.default_rng(42)
        model = DecoderOnlyModel(
            vocab_size=50,
            d_model=32,
            n_heads=4,
            n_layers=2,
            d_ff=64,
            max_len=16,
            rng=rng,
            dropout_rate=0.0,
        )
        tokens = rng.integers(0, 50, size=(2, 8)).astype(np.intp)
        model.eval()
        logits = model.forward(tokens)
        assert logits.shape == (2, 8, 50)

    def test_backward_runs(self) -> None:
        rng = np.random.default_rng(42)
        model = DecoderOnlyModel(
            vocab_size=50,
            d_model=32,
            n_heads=4,
            n_layers=2,
            d_ff=64,
            max_len=16,
            rng=rng,
            dropout_rate=0.0,
        )
        tokens = rng.integers(0, 50, size=(2, 8)).astype(np.intp)
        model.eval()
        logits = model.forward(tokens)
        grad = np.ones_like(logits) * 0.01
        model.backward(grad)
        model.update_params(1e-4)

    def test_train_eval_mode(self) -> None:
        rng = np.random.default_rng(42)
        model = DecoderOnlyModel(
            vocab_size=50,
            d_model=32,
            n_heads=4,
            n_layers=1,
            d_ff=64,
            max_len=16,
            rng=rng,
            dropout_rate=0.1,
        )
        model.train()
        for d in model.dropouts:
            assert d.training is True
        model.eval()
        for d in model.dropouts:
            assert d.training is False

    def test_generate_returns_longer_sequence(self) -> None:
        rng = np.random.default_rng(42)
        model = DecoderOnlyModel(
            vocab_size=50,
            d_model=16,
            n_heads=2,
            n_layers=1,
            d_ff=32,
            max_len=32,
            rng=rng,
            dropout_rate=0.0,
        )
        prompt = np.array([1, 2, 3], dtype=np.intp)
        generated = model.generate(prompt, max_tokens=5, eos_token_id=99)
        # Should have prompt + at least 1 generated token (unless it hits EOS immediately).
        assert len(generated) >= len(prompt)
        # Should not exceed prompt + max_tokens.
        assert len(generated) <= len(prompt) + 5
        # First tokens should match prompt.
        np.testing.assert_array_equal(generated[: len(prompt)], prompt)

    def test_generate_stops_at_eos(self) -> None:
        rng = np.random.default_rng(42)
        model = DecoderOnlyModel(
            vocab_size=10,
            d_model=8,
            n_heads=2,
            n_layers=1,
            d_ff=16,
            max_len=32,
            rng=rng,
            dropout_rate=0.0,
        )
        # Use a very small vocab so EOS token has a high chance of being sampled.
        eos_token_id = 5
        prompt = np.array([1, 2], dtype=np.intp)
        generated = model.generate(prompt, max_tokens=20, eos_token_id=eos_token_id)
        # If EOS was generated, sequence should end with it.
        if eos_token_id in generated[len(prompt) :]:
            eos_idx = np.where(generated == eos_token_id)[0][0]
            # Should stop right after first EOS.
            assert len(generated) == eos_idx + 1

    def test_generate_stops_at_max_tokens(self) -> None:
        rng = np.random.default_rng(42)
        model = DecoderOnlyModel(
            vocab_size=50,
            d_model=16,
            n_heads=2,
            n_layers=1,
            d_ff=32,
            max_len=32,
            rng=rng,
            dropout_rate=0.0,
        )
        prompt = np.array([1, 2], dtype=np.intp)
        max_tokens = 3
        # Use an EOS token that's unlikely to be generated.
        generated = model.generate(prompt, max_tokens=max_tokens, eos_token_id=999)
        # Should generate exactly max_tokens new tokens.
        assert len(generated) == len(prompt) + max_tokens

    def test_generate_preserves_training_mode(self) -> None:
        rng = np.random.default_rng(42)
        model = DecoderOnlyModel(
            vocab_size=50,
            d_model=16,
            n_heads=2,
            n_layers=1,
            d_ff=32,
            max_len=16,
            rng=rng,
            dropout_rate=0.1,
        )
        # Start in training mode.
        model.train()
        assert model.dropouts[0].training is True
        # Generate should temporarily switch to eval but restore training mode.
        prompt = np.array([1, 2, 3], dtype=np.intp)
        model.generate(prompt, max_tokens=2, eos_token_id=99)
        assert model.dropouts[0].training is True


class TestCLMLoss:
    def test_loss_is_positive(self) -> None:
        rng = np.random.default_rng(42)
        logits = rng.standard_normal((2, 4, 50))
        targets = rng.integers(0, 50, size=(2, 4)).astype(np.intp)
        loss, _grad = clm_loss_and_grad(logits, targets)
        assert loss > 0

    def test_grad_shape_matches_logits(self) -> None:
        rng = np.random.default_rng(42)
        logits = rng.standard_normal((2, 4, 50))
        targets = rng.integers(0, 50, size=(2, 4)).astype(np.intp)
        _, grad = clm_loss_and_grad(logits, targets)
        assert grad.shape == logits.shape

    def test_ignored_positions_have_zero_grad(self) -> None:
        rng = np.random.default_rng(42)
        logits = rng.standard_normal((2, 4, 50))
        targets = np.full((2, 4), -1, dtype=np.intp)
        # Only first position is valid.
        targets[:, 0] = rng.integers(0, 50, size=2).astype(np.intp)
        _, grad = clm_loss_and_grad(logits, targets)
        # Positions 1-3 should have zero gradient.
        np.testing.assert_array_equal(grad[:, 1:, :], 0.0)

    def test_all_ignored_returns_zero_loss(self) -> None:
        rng = np.random.default_rng(42)
        logits = rng.standard_normal((2, 4, 50))
        targets = np.full((2, 4), -1, dtype=np.intp)
        loss, grad = clm_loss_and_grad(logits, targets)
        assert loss == 0.0
        np.testing.assert_array_equal(grad, 0.0)


class TestLMTrainer:
    def test_loss_decreases(self) -> None:
        """Training for a few epochs should decrease the loss."""
        rng = np.random.default_rng(42)
        model = DecoderOnlyModel(
            vocab_size=20,
            d_model=16,
            n_heads=2,
            n_layers=1,
            d_ff=32,
            max_len=16,
            rng=rng,
            dropout_rate=0.0,
        )
        trainer = LMTrainer(
            model=model,
            learning_rate=0.01,
            batch_size=4,
            pad_token_id=0,
        )

        # Tiny synthetic dataset: 8 sequences of length 8.
        data = rng.integers(1, 20, size=(8, 8)).astype(np.intp)

        loss_before = trainer.eval_loss(data)
        for _ in range(5):
            trainer.train_epoch(data, rng)
        loss_after = trainer.eval_loss(data)

        assert loss_after < loss_before

    def test_eval_loss_does_not_change_model(self) -> None:
        rng = np.random.default_rng(42)
        model = DecoderOnlyModel(
            vocab_size=20,
            d_model=16,
            n_heads=2,
            n_layers=1,
            d_ff=32,
            max_len=16,
            rng=rng,
            dropout_rate=0.0,
        )
        trainer = LMTrainer(
            model=model,
            learning_rate=0.01,
            batch_size=4,
            pad_token_id=0,
        )
        data = rng.integers(1, 20, size=(4, 8)).astype(np.intp)

        loss1 = trainer.eval_loss(data)
        loss2 = trainer.eval_loss(data)
        assert loss1 == loss2
