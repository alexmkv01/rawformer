"""Tests for DPO loss, log-prob utilities, and DPOTrainer.

Covers log-softmax, sequence log-probs, DPO loss with gradient
(PyTorch cross-verification and numerical gradient check),
and trainer loss-decrease / accuracy-improvement.
"""

from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F

from rawformer.training.clm import DecoderOnlyModel
from rawformer.training.dpo import (
    DPOTrainer,
    _log_softmax,
    dpo_loss_and_grad,
    sequence_log_probs,
)

_RNG_SEED = 0


# ----- Helpers -----


class _DPOSetup(NamedTuple):
    """Holds all arrays needed for a single DPO loss computation."""

    policy_logits: npt.NDArray[np.float64]
    ref_logits: npt.NDArray[np.float64]
    chosen_targets: npt.NDArray[np.intp]
    rejected_targets: npt.NDArray[np.intp]
    chosen_mask: npt.NDArray[np.bool_]
    rejected_mask: npt.NDArray[np.bool_]
    beta: float


def _make_dpo_setup(
    batch: int = 2,
    seq_len: int = 4,
    vocab_size: int = 8,
    beta: float = 0.1,
    seed: int = _RNG_SEED,
) -> _DPOSetup:
    """Create a reproducible DPO test setup with random logits and targets."""
    rng = np.random.default_rng(seed)

    # Concatenated logits: first `batch` rows are chosen, next `batch` are rejected
    policy_logits = rng.standard_normal((2 * batch, seq_len, vocab_size))
    ref_logits = rng.standard_normal((2 * batch, seq_len, vocab_size))

    chosen_targets = rng.integers(0, vocab_size, size=(batch, seq_len)).astype(np.intp)
    rejected_targets = rng.integers(0, vocab_size, size=(batch, seq_len)).astype(np.intp)

    # All positions valid (no padding)
    chosen_mask = np.ones((batch, seq_len), dtype=np.bool_)
    rejected_mask = np.ones((batch, seq_len), dtype=np.bool_)

    return _DPOSetup(
        policy_logits=policy_logits,
        ref_logits=ref_logits,
        chosen_targets=chosen_targets,
        rejected_targets=rejected_targets,
        chosen_mask=chosen_mask,
        rejected_mask=rejected_mask,
        beta=beta,
    )


def _make_dpo_setup_with_padding(
    batch: int = 2,
    seq_len: int = 6,
    vocab_size: int = 8,
    beta: float = 0.1,
    seed: int = _RNG_SEED,
) -> _DPOSetup:
    """Create a DPO setup where the last 2 positions are padding."""
    rng = np.random.default_rng(seed)

    policy_logits = rng.standard_normal((2 * batch, seq_len, vocab_size))
    ref_logits = rng.standard_normal((2 * batch, seq_len, vocab_size))

    chosen_targets = rng.integers(0, vocab_size, size=(batch, seq_len)).astype(np.intp)
    rejected_targets = rng.integers(0, vocab_size, size=(batch, seq_len)).astype(np.intp)

    # Last 2 positions are padding
    chosen_mask = np.ones((batch, seq_len), dtype=np.bool_)
    chosen_mask[:, -2:] = False
    rejected_mask = np.ones((batch, seq_len), dtype=np.bool_)
    rejected_mask[:, -2:] = False

    return _DPOSetup(
        policy_logits=policy_logits,
        ref_logits=ref_logits,
        chosen_targets=chosen_targets,
        rejected_targets=rejected_targets,
        chosen_mask=chosen_mask,
        rejected_mask=rejected_mask,
        beta=beta,
    )


def _torch_dpo_loss(setup: _DPOSetup) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute DPO loss and gradient using PyTorch autograd.

    Returns:
        Tuple of (scalar loss, gradient w.r.t. policy logits).
    """
    batch = setup.chosen_targets.shape[0]

    policy_t = torch.from_numpy(setup.policy_logits.copy()).requires_grad_(True)
    ref_t = torch.from_numpy(setup.ref_logits.copy())

    chosen_targets_t = torch.from_numpy(setup.chosen_targets.copy())
    rejected_targets_t = torch.from_numpy(setup.rejected_targets.copy())
    chosen_mask_t = torch.from_numpy(setup.chosen_mask.copy())
    rejected_mask_t = torch.from_numpy(setup.rejected_mask.copy())

    # Log-softmax over vocab dimension
    policy_log_probs = F.log_softmax(policy_t, dim=-1)
    ref_log_probs = F.log_softmax(ref_t, dim=-1)

    # Gather and mask — chosen half
    pi_lp_chosen = policy_log_probs[:batch].gather(2, chosen_targets_t.unsqueeze(-1)).squeeze(-1)
    pi_lp_chosen = (pi_lp_chosen * chosen_mask_t).sum(dim=1)

    ref_lp_chosen = ref_log_probs[:batch].gather(2, chosen_targets_t.unsqueeze(-1)).squeeze(-1)
    ref_lp_chosen = (ref_lp_chosen * chosen_mask_t).sum(dim=1)

    # Gather and mask — rejected half
    pi_lp_rejected = (
        policy_log_probs[batch:].gather(2, rejected_targets_t.unsqueeze(-1)).squeeze(-1)
    )
    pi_lp_rejected = (pi_lp_rejected * rejected_mask_t).sum(dim=1)

    ref_lp_rejected = ref_log_probs[batch:].gather(2, rejected_targets_t.unsqueeze(-1)).squeeze(-1)
    ref_lp_rejected = (ref_lp_rejected * rejected_mask_t).sum(dim=1)

    log_ratio_w = pi_lp_chosen - ref_lp_chosen
    log_ratio_l = pi_lp_rejected - ref_lp_rejected
    h = setup.beta * (log_ratio_w - log_ratio_l)

    loss = -F.logsigmoid(h).mean()

    loss.backward()  # type: ignore[no-untyped-call]
    assert policy_t.grad is not None
    return loss, policy_t.grad


# =====================================================================
# log_softmax
# =====================================================================


class TestLogSoftmax:
    def test_matches_pytorch(self) -> None:
        """Cross-verify log-softmax against PyTorch."""
        rng = np.random.default_rng(42)
        logits = rng.standard_normal((3, 5, 10))

        result_np = _log_softmax(logits)
        result_torch = F.log_softmax(torch.from_numpy(logits), dim=-1).numpy()

        np.testing.assert_allclose(result_np, result_torch, atol=1e-10)

    def test_output_shape_preserved(self) -> None:
        logits = np.random.default_rng(0).standard_normal((2, 4, 8))
        result = _log_softmax(logits)
        assert result.shape == logits.shape

    def test_values_are_nonpositive(self) -> None:
        """Log-probabilities must be <= 0."""
        logits = np.random.default_rng(0).standard_normal((3, 5))
        result = _log_softmax(logits)
        assert np.all(result <= 1e-10)

    def test_exp_sums_to_one(self) -> None:
        """exp(log_softmax) should sum to 1 along the last axis."""
        logits = np.random.default_rng(0).standard_normal((4, 6))
        probs = np.exp(_log_softmax(logits))
        np.testing.assert_allclose(np.sum(probs, axis=-1), 1.0, atol=1e-10)


# =====================================================================
# sequence_log_probs
# =====================================================================


class TestSequenceLogProbs:
    def test_matches_pytorch(self) -> None:
        """Cross-verify sequence_log_probs against PyTorch gather + sum."""
        rng = np.random.default_rng(42)
        batch, seq_len, vocab = 3, 5, 10
        logits = rng.standard_normal((batch, seq_len, vocab))
        targets = rng.integers(0, vocab, size=(batch, seq_len)).astype(np.intp)
        mask = np.ones((batch, seq_len), dtype=np.bool_)

        result_np = sequence_log_probs(logits, targets, mask)

        logits_t = torch.from_numpy(logits)
        lp = F.log_softmax(logits_t, dim=-1)
        gathered = lp.gather(2, torch.from_numpy(targets).unsqueeze(-1)).squeeze(-1)
        result_torch = gathered.sum(dim=1).numpy()

        np.testing.assert_allclose(result_np, result_torch, atol=1e-10)

    def test_mask_zeros_padding(self) -> None:
        """Masked positions should not contribute to the sum."""
        rng = np.random.default_rng(42)
        logits = rng.standard_normal((2, 6, 8))
        targets = rng.integers(0, 8, size=(2, 6)).astype(np.intp)

        full_mask = np.ones((2, 6), dtype=np.bool_)
        partial_mask = np.ones((2, 6), dtype=np.bool_)
        partial_mask[:, -2:] = False

        full_result = sequence_log_probs(logits, targets, full_mask)
        partial_result = sequence_log_probs(logits, targets, partial_mask)

        # With fewer valid positions, the partial sum should differ
        assert not np.allclose(full_result, partial_result)
        # Partial should be closer to zero (fewer negative terms summed)
        assert np.all(np.abs(partial_result) < np.abs(full_result))

    def test_output_shape(self) -> None:
        rng = np.random.default_rng(0)
        logits = rng.standard_normal((4, 5, 10))
        targets = rng.integers(0, 10, size=(4, 5)).astype(np.intp)
        mask = np.ones((4, 5), dtype=np.bool_)

        result = sequence_log_probs(logits, targets, mask)
        assert result.shape == (4,)


# =====================================================================
# dpo_loss_and_grad
# =====================================================================


class TestDPOLossAndGrad:
    def test_loss_matches_pytorch(self) -> None:
        """Cross-verify DPO loss value against PyTorch."""
        setup = _make_dpo_setup()
        result = dpo_loss_and_grad(
            setup.policy_logits,
            setup.ref_logits,
            setup.chosen_targets,
            setup.rejected_targets,
            setup.chosen_mask,
            setup.rejected_mask,
            setup.beta,
        )
        loss_torch, _ = _torch_dpo_loss(setup)

        np.testing.assert_allclose(result.loss, loss_torch.item(), atol=1e-10)

    def test_grad_matches_pytorch(self) -> None:
        """Cross-verify DPO gradient against PyTorch autograd."""
        setup = _make_dpo_setup()
        result = dpo_loss_and_grad(
            setup.policy_logits,
            setup.ref_logits,
            setup.chosen_targets,
            setup.rejected_targets,
            setup.chosen_mask,
            setup.rejected_mask,
            setup.beta,
        )
        _, grad_torch = _torch_dpo_loss(setup)

        np.testing.assert_allclose(result.grad, grad_torch.numpy(), atol=1e-8)

    def test_loss_with_padding_matches_pytorch(self) -> None:
        """Cross-verify DPO loss with padded sequences against PyTorch."""
        setup = _make_dpo_setup_with_padding()
        result = dpo_loss_and_grad(
            setup.policy_logits,
            setup.ref_logits,
            setup.chosen_targets,
            setup.rejected_targets,
            setup.chosen_mask,
            setup.rejected_mask,
            setup.beta,
        )
        loss_torch, _ = _torch_dpo_loss(setup)

        np.testing.assert_allclose(result.loss, loss_torch.item(), atol=1e-10)

    def test_grad_with_padding_matches_pytorch(self) -> None:
        """Cross-verify gradient with padding against PyTorch autograd."""
        setup = _make_dpo_setup_with_padding()
        result = dpo_loss_and_grad(
            setup.policy_logits,
            setup.ref_logits,
            setup.chosen_targets,
            setup.rejected_targets,
            setup.chosen_mask,
            setup.rejected_mask,
            setup.beta,
        )
        _, grad_torch = _torch_dpo_loss(setup)

        np.testing.assert_allclose(result.grad, grad_torch.numpy(), atol=1e-8)

    def test_grad_zero_at_masked_positions(self) -> None:
        """Gradient should be zero at padding positions."""
        setup = _make_dpo_setup_with_padding()
        result = dpo_loss_and_grad(
            setup.policy_logits,
            setup.ref_logits,
            setup.chosen_targets,
            setup.rejected_targets,
            setup.chosen_mask,
            setup.rejected_mask,
            setup.beta,
        )
        batch = setup.chosen_targets.shape[0]
        # Chosen half: last 2 positions should have zero grad
        np.testing.assert_array_equal(result.grad[:batch, -2:, :], 0.0)
        # Rejected half: last 2 positions should have zero grad
        np.testing.assert_array_equal(result.grad[batch:, -2:, :], 0.0)

    def test_numerical_gradient_check(self) -> None:
        """Verify gradient via central-difference finite differences.

        Uses intentionally small dimensions (batch=1, seq_len=3, vocab=4)
        so the loop over all 2*1*3*4=24 elements stays fast.
        """
        setup = _make_dpo_setup(batch=1, seq_len=3, vocab_size=4)
        result = dpo_loss_and_grad(
            setup.policy_logits,
            setup.ref_logits,
            setup.chosen_targets,
            setup.rejected_targets,
            setup.chosen_mask,
            setup.rejected_mask,
            setup.beta,
        )

        eps = 1e-5
        numerical_grad = np.zeros_like(setup.policy_logits)

        for idx in np.ndindex(setup.policy_logits.shape):
            logits_plus = setup.policy_logits.copy()
            logits_minus = setup.policy_logits.copy()
            logits_plus[idx] += eps
            logits_minus[idx] -= eps

            loss_plus = dpo_loss_and_grad(
                logits_plus,
                setup.ref_logits,
                setup.chosen_targets,
                setup.rejected_targets,
                setup.chosen_mask,
                setup.rejected_mask,
                setup.beta,
            ).loss
            loss_minus = dpo_loss_and_grad(
                logits_minus,
                setup.ref_logits,
                setup.chosen_targets,
                setup.rejected_targets,
                setup.chosen_mask,
                setup.rejected_mask,
                setup.beta,
            ).loss
            numerical_grad[idx] = (loss_plus - loss_minus) / (2 * eps)

        np.testing.assert_allclose(result.grad, numerical_grad, atol=1e-5)

    def test_loss_is_nonnegative(self) -> None:
        """DPO loss = -mean(log sigmoid(h)) is always >= 0 since log sigmoid <= 0."""
        setup = _make_dpo_setup()
        result = dpo_loss_and_grad(
            setup.policy_logits,
            setup.ref_logits,
            setup.chosen_targets,
            setup.rejected_targets,
            setup.chosen_mask,
            setup.rejected_mask,
            setup.beta,
        )
        assert result.loss >= 0.0

    def test_grad_shape_matches_logits(self) -> None:
        setup = _make_dpo_setup()
        result = dpo_loss_and_grad(
            setup.policy_logits,
            setup.ref_logits,
            setup.chosen_targets,
            setup.rejected_targets,
            setup.chosen_mask,
            setup.rejected_mask,
            setup.beta,
        )
        assert result.grad.shape == setup.policy_logits.shape

    def test_different_beta_scales_loss(self) -> None:
        """Larger beta should change the loss magnitude."""
        setup_low = _make_dpo_setup(beta=0.05)
        setup_high = _make_dpo_setup(beta=0.5)

        loss_low = dpo_loss_and_grad(
            setup_low.policy_logits,
            setup_low.ref_logits,
            setup_low.chosen_targets,
            setup_low.rejected_targets,
            setup_low.chosen_mask,
            setup_low.rejected_mask,
            setup_low.beta,
        ).loss
        loss_high = dpo_loss_and_grad(
            setup_high.policy_logits,
            setup_high.ref_logits,
            setup_high.chosen_targets,
            setup_high.rejected_targets,
            setup_high.chosen_mask,
            setup_high.rejected_mask,
            setup_high.beta,
        ).loss
        assert loss_low != loss_high


# =====================================================================
# DPOTrainer
# =====================================================================


def _make_tiny_model(
    vocab_size: int = 20,
    d_model: int = 16,
    n_heads: int = 2,
    n_layers: int = 1,
    d_ff: int = 32,
    max_len: int = 16,
    seed: int = _RNG_SEED,
) -> DecoderOnlyModel:
    """Build a tiny DecoderOnlyModel for testing."""
    rng = np.random.default_rng(seed)
    return DecoderOnlyModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_len=max_len,
        rng=rng,
        dropout_rate=0.0,
    )


class TestDPOTrainer:
    def test_loss_decreases(self) -> None:
        """DPO training should decrease the loss over multiple steps.

        Both _RNG_SEED (model init) and seed=42 (data/shuffling) are
        fixed so the test is fully deterministic.
        """
        model = _make_tiny_model()
        trainer = DPOTrainer.from_sft_model(
            sft_model=model,
            learning_rate=0.01,
            beta=0.1,
            pad_token_id=0,
        )

        rng = np.random.default_rng(42)
        # Create toy preference pairs: chosen has lower token IDs (a pattern)
        n_pairs = 4
        seq_len = 8
        chosen_ids = rng.integers(1, 10, size=(n_pairs, seq_len)).astype(np.intp)
        rejected_ids = rng.integers(10, 20, size=(n_pairs, seq_len)).astype(np.intp)

        first = trainer.train_epoch(chosen_ids, rejected_ids, batch_size=2, rng=rng)

        last = first
        for _ in range(19):
            last = trainer.train_epoch(chosen_ids, rejected_ids, batch_size=2, rng=rng)

        assert last.loss < first.loss

    def test_ref_model_frozen(self) -> None:
        """Reference model parameters should not change during training."""
        model = _make_tiny_model()
        trainer = DPOTrainer.from_sft_model(
            sft_model=model,
            learning_rate=0.01,
            beta=0.1,
            pad_token_id=0,
        )

        # Snapshot reference model weights
        ref_weights_before = trainer.ref_model.output_proj.weights.copy()

        rng = np.random.default_rng(42)
        chosen_ids = rng.integers(1, 10, size=(2, 6)).astype(np.intp)
        rejected_ids = rng.integers(10, 20, size=(2, 6)).astype(np.intp)

        trainer.train_step(chosen_ids, rejected_ids)

        np.testing.assert_array_equal(
            trainer.ref_model.output_proj.weights,
            ref_weights_before,
        )

    def test_policy_model_updated(self) -> None:
        """Policy model parameters should change after a training step."""
        model = _make_tiny_model()
        trainer = DPOTrainer.from_sft_model(
            sft_model=model,
            learning_rate=0.01,
            beta=0.1,
            pad_token_id=0,
        )

        policy_weights_before = trainer.model.output_proj.weights.copy()

        rng = np.random.default_rng(42)
        chosen_ids = rng.integers(1, 10, size=(2, 6)).astype(np.intp)
        rejected_ids = rng.integers(10, 20, size=(2, 6)).astype(np.intp)

        trainer.train_step(chosen_ids, rejected_ids)

        assert not np.array_equal(
            trainer.model.output_proj.weights,
            policy_weights_before,
        )

    def test_train_step_returns_named_metrics(self) -> None:
        """train_step should return a DPOMetrics namedtuple."""
        model = _make_tiny_model()
        trainer = DPOTrainer.from_sft_model(
            sft_model=model,
            learning_rate=0.01,
            beta=0.1,
            pad_token_id=0,
        )

        rng = np.random.default_rng(42)
        chosen_ids = rng.integers(1, 10, size=(2, 6)).astype(np.intp)
        rejected_ids = rng.integers(10, 20, size=(2, 6)).astype(np.intp)

        metrics = trainer.train_step(chosen_ids, rejected_ids)

        assert isinstance(metrics.loss, float)
        assert isinstance(metrics.reward_margin, float)
        assert isinstance(metrics.accuracy, float)
        assert metrics.loss >= 0.0
        assert 0.0 <= metrics.accuracy <= 1.0

    def test_accuracy_improves(self) -> None:
        """After training, the model should prefer chosen over rejected more often."""
        model = _make_tiny_model()
        trainer = DPOTrainer.from_sft_model(
            sft_model=model,
            learning_rate=0.01,
            beta=0.1,
            pad_token_id=0,
        )

        rng = np.random.default_rng(42)
        n_pairs = 4
        seq_len = 8
        chosen_ids = rng.integers(1, 10, size=(n_pairs, seq_len)).astype(np.intp)
        rejected_ids = rng.integers(10, 20, size=(n_pairs, seq_len)).astype(np.intp)

        first = trainer.train_epoch(chosen_ids, rejected_ids, batch_size=2, rng=rng)

        last = first
        for _ in range(29):
            last = trainer.train_epoch(chosen_ids, rejected_ids, batch_size=2, rng=rng)

        assert last.accuracy >= first.accuracy

    def test_from_sft_model_does_not_mutate_original(self) -> None:
        """from_sft_model deep-copies both policy and ref, leaving the original intact."""
        model = _make_tiny_model()
        original_weights = model.output_proj.weights.copy()

        trainer = DPOTrainer.from_sft_model(
            sft_model=model,
            learning_rate=0.01,
            beta=0.1,
            pad_token_id=0,
        )

        rng = np.random.default_rng(42)
        chosen_ids = rng.integers(1, 10, size=(2, 6)).astype(np.intp)
        rejected_ids = rng.integers(10, 20, size=(2, 6)).astype(np.intp)

        trainer.train_step(chosen_ids, rejected_ids)

        # The original model should be completely untouched
        np.testing.assert_array_equal(model.output_proj.weights, original_weights)
