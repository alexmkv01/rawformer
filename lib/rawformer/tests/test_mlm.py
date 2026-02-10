"""Tests for masked language modelling utilities."""

import numpy as np

from rawformer.training.mlm import MLMHead, mask_tokens


class TestMaskTokens:
    def test_output_shapes(self) -> None:
        rng = np.random.default_rng(42)
        token_ids = rng.integers(0, 100, size=(4, 16)).astype(np.intp)
        masked, labels, mask = mask_tokens(
            token_ids, mask_prob=0.15, mask_token_id=999, vocab_size=100, rng=rng
        )
        assert masked.shape == token_ids.shape
        assert labels.shape == token_ids.shape
        assert mask.shape == token_ids.shape
        assert mask.dtype == np.bool_

    def test_labels_negative_one_at_unmasked(self) -> None:
        rng = np.random.default_rng(42)
        token_ids = rng.integers(0, 100, size=(4, 16)).astype(np.intp)
        _, labels, mask = mask_tokens(
            token_ids, mask_prob=0.15, mask_token_id=999, vocab_size=100, rng=rng
        )
        # Unmasked positions should have label -1.
        assert np.all(labels[~mask] == -1)

    def test_labels_match_original_at_masked(self) -> None:
        rng = np.random.default_rng(42)
        token_ids = rng.integers(0, 100, size=(4, 16)).astype(np.intp)
        _, labels, mask = mask_tokens(
            token_ids, mask_prob=0.15, mask_token_id=999, vocab_size=100, rng=rng
        )
        # Masked positions should have the original token ID.
        np.testing.assert_array_equal(labels[mask], token_ids[mask])

    def test_special_tokens_never_masked(self) -> None:
        rng = np.random.default_rng(42)
        # First and last tokens are special.
        token_ids = rng.integers(10, 100, size=(4, 16)).astype(np.intp)
        token_ids[:, 0] = 1  # CLS
        token_ids[:, -1] = 2  # SEP
        _, _, mask = mask_tokens(
            token_ids,
            mask_prob=0.5,
            mask_token_id=999,
            vocab_size=100,
            rng=rng,
            special_token_ids={1, 2},
        )
        assert not np.any(mask[:, 0])
        assert not np.any(mask[:, -1])

    def test_approximate_mask_rate(self) -> None:
        rng = np.random.default_rng(42)
        token_ids = rng.integers(0, 100, size=(100, 64)).astype(np.intp)
        _, _, mask = mask_tokens(
            token_ids, mask_prob=0.15, mask_token_id=999, vocab_size=100, rng=rng
        )
        actual_rate = np.mean(mask)
        assert 0.10 < actual_rate < 0.20

    def test_mask_token_replacement_distribution(self) -> None:
        """Check ~80% of masked positions get the [MASK] token.

        Uses a fixed seed so the result is deterministic despite being statistical.
        """
        rng = np.random.default_rng(42)
        token_ids = rng.integers(10, 100, size=(200, 64)).astype(np.intp)
        masked, _, mask = mask_tokens(
            token_ids, mask_prob=0.15, mask_token_id=999, vocab_size=100, rng=rng
        )
        n_masked = int(np.sum(mask))
        n_replaced_with_mask = int(np.sum(masked[mask] == 999))
        ratio = n_replaced_with_mask / n_masked
        # Should be approximately 80%.
        assert 0.70 < ratio < 0.90


class TestMLMHead:
    def test_forward_shape(self) -> None:
        rng = np.random.default_rng(42)
        head = MLMHead(d_model=64, vocab_size=100, rng=rng)
        hidden = rng.standard_normal((2, 8, 64))
        logits = head.forward(hidden)
        assert logits.shape == (2, 8, 100)

    def test_backward_shape(self) -> None:
        rng = np.random.default_rng(42)
        head = MLMHead(d_model=64, vocab_size=100, rng=rng)
        hidden = rng.standard_normal((2, 8, 64))
        logits = head.forward(hidden)
        grad = np.ones_like(logits)
        grad_hidden = head.backward(grad)
        assert grad_hidden.shape == (2, 8, 64)
