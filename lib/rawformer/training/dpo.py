"""Direct Preference Optimization (Rafailov et al., 2023).

Implements the DPO loss, gradient computation, and a trainer that
composes a policy model with a frozen reference model.  All gradients
are hand-derived — no autograd.

The core identity::

    L = -mean(log sigmoid(beta * (log_ratio_w - log_ratio_l)))

where ``log_ratio_w = sum_t [log pi_theta(w_t | w_{<t}) - log pi_ref(w_t | w_{<t})]``
and likewise for the rejected sequence.
"""

import copy
import logging
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from rawformer.training.clm import DecoderOnlyModel

logger = logging.getLogger(__name__)

_SIGMOID_CLIP: float = 500.0
"""Clip bound for the sigmoid exponent to avoid overflow."""


class DPOMetrics(NamedTuple):
    """Metrics returned by a single DPO training step or epoch."""

    loss: float
    reward_margin: float
    accuracy: float


class DPOLossResult(NamedTuple):
    """Return value of :func:`dpo_loss_and_grad`."""

    loss: float
    grad: npt.NDArray[np.float64]
    log_pi_w: npt.NDArray[np.float64]
    log_pi_l: npt.NDArray[np.float64]
    log_ref_w: npt.NDArray[np.float64]
    log_ref_l: npt.NDArray[np.float64]


def _log_softmax(logits: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Numerically stable log-softmax over the last axis.

    Args:
        logits: Array of shape (..., vocab_size).

    Returns:
        Log-probabilities with the same shape as *logits*.
    """
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    result: npt.NDArray[np.float64] = shifted - np.log(
        np.sum(np.exp(shifted), axis=-1, keepdims=True)
    )
    return result


def sequence_log_probs(
    logits: npt.NDArray[np.float64],
    targets: npt.NDArray[np.intp],
    mask: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float64]:
    """Compute per-sequence *unnormalised* sum of log-probabilities.

    Returns the raw sum (not length-normalised) because the DPO loss
    operates on total log-probability ratios, not per-token averages.

    Args:
        logits: Shape (batch, seq_len, vocab_size).
        targets: Shape (batch, seq_len) with token IDs.
        mask: Shape (batch, seq_len), True for valid (non-padding) positions.

    Returns:
        Array of shape (batch,) with the sum of log-probs for each sequence.
    """
    batch, seq_len, _ = logits.shape
    log_probs = _log_softmax(logits)

    # Gather log-probs at target positions: (batch, seq_len)
    gathered = log_probs[np.arange(batch)[:, None], np.arange(seq_len)[None, :], targets]

    # Zero out padding positions and sum per sequence
    result: npt.NDArray[np.float64] = np.sum(gathered * mask, axis=1)
    return result


def dpo_loss_and_grad(
    policy_logits: npt.NDArray[np.float64],
    ref_logits: npt.NDArray[np.float64],
    chosen_targets: npt.NDArray[np.intp],
    rejected_targets: npt.NDArray[np.intp],
    chosen_mask: npt.NDArray[np.bool_],
    rejected_mask: npt.NDArray[np.bool_],
    beta: float,
) -> DPOLossResult:
    """Compute the DPO loss and gradient w.r.t. policy logits.

    Uses the concatenated-batch layout: the first half of the batch
    dimension holds chosen sequences, the second half holds rejected.

    Args:
        policy_logits: Shape (2 * batch, seq_len, vocab_size) from the policy model.
        ref_logits: Same shape, from the frozen reference model.
        chosen_targets: Shape (batch, seq_len) target token IDs for chosen.
        rejected_targets: Shape (batch, seq_len) target token IDs for rejected.
        chosen_mask: Shape (batch, seq_len), True for valid positions.
        rejected_mask: Shape (batch, seq_len), True for valid positions.
        beta: Temperature parameter controlling deviation from reference.

    Returns:
        A :class:`DPOLossResult` containing the scalar loss, gradient,
        and per-sequence log-probability sums (reusable for metrics).
    """
    total_batch = policy_logits.shape[0]
    batch = total_batch // 2
    seq_len = policy_logits.shape[1]

    # Split concatenated logits into chosen / rejected halves
    policy_chosen = policy_logits[:batch]
    policy_rejected = policy_logits[batch:]
    ref_chosen = ref_logits[:batch]
    ref_rejected = ref_logits[batch:]

    # Per-sequence log-probs: shape (batch,)
    log_pi_w = sequence_log_probs(policy_chosen, chosen_targets, chosen_mask)
    log_pi_l = sequence_log_probs(policy_rejected, rejected_targets, rejected_mask)
    log_ref_w = sequence_log_probs(ref_chosen, chosen_targets, chosen_mask)
    log_ref_l = sequence_log_probs(ref_rejected, rejected_targets, rejected_mask)

    # Log-ratios and implicit reward difference
    log_ratio_w = log_pi_w - log_ref_w  # (batch,)
    log_ratio_l = log_pi_l - log_ref_l  # (batch,)
    h = beta * (log_ratio_w - log_ratio_l)  # (batch,)

    # Loss: L = -mean(log sigmoid(h))
    # Use log sigmoid(h) = -log(1 + exp(-h)) = -softplus(-h) for stability
    # softplus(x) = log(1 + exp(x)), stable version: max(x,0) + log(1 + exp(-|x|))
    neg_h = -h
    softplus_neg_h = np.maximum(neg_h, 0.0) + np.log1p(np.exp(-np.abs(neg_h)))
    loss = float(np.mean(softplus_neg_h))

    # sigmoid(-h) = 1 / (1 + exp(h)), computed stably
    sigma_neg_h_vec = 1.0 / (1.0 + np.exp(np.clip(h, -_SIGMOID_CLIP, _SIGMOID_CLIP)))

    # --- Gradient w.r.t. policy_logits ---
    # softmax probabilities for the full concatenated batch
    shifted = policy_logits - np.max(policy_logits, axis=-1, keepdims=True)
    exp_logits = np.exp(shifted)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    grad = np.zeros_like(policy_logits)

    # sigma_neg_h_vec: (batch,) -> (batch, 1, 1) for broadcasting
    snh = sigma_neg_h_vec[:, None, None]

    # Chosen half gradient:
    # dL/dlogits_chosen[b,t,k] =
    #   -sigmoid(-h[b]) * beta * (1[k=w_t] - probs[b,t,k]) * mask[b,t] / batch
    probs_chosen = probs[:batch]
    one_hot_chosen = np.zeros_like(probs_chosen)
    one_hot_chosen[
        np.arange(batch)[:, None],
        np.arange(seq_len)[None, :],
        chosen_targets,
    ] = 1.0
    mask_w = chosen_mask[:, :, None].astype(np.float64)
    grad[:batch] = -snh * beta * (one_hot_chosen - probs_chosen) * mask_w / batch

    # Rejected half gradient:
    # dL/dlogits_rejected[b,t,k] =
    #   +sigmoid(-h[b]) * beta * (1[k=l_t] - probs[b,t,k]) * mask[b,t] / batch
    probs_rejected = probs[batch:]
    one_hot_rejected = np.zeros_like(probs_rejected)
    one_hot_rejected[
        np.arange(batch)[:, None],
        np.arange(seq_len)[None, :],
        rejected_targets,
    ] = 1.0
    mask_l = rejected_mask[:, :, None].astype(np.float64)
    grad[batch:] = snh * beta * (one_hot_rejected - probs_rejected) * mask_l / batch

    return DPOLossResult(
        loss=loss,
        grad=grad,
        log_pi_w=log_pi_w,
        log_pi_l=log_pi_l,
        log_ref_w=log_ref_w,
        log_ref_l=log_ref_l,
    )


class DPOTrainer:
    """Mini-batch DPO trainer composing a policy model and a frozen reference.

    Uses the concatenated-batch strategy: chosen and rejected sequences
    are padded to the same length and concatenated along the batch axis
    for a single forward/backward pass.
    """

    def __init__(
        self,
        model: DecoderOnlyModel,
        ref_model: DecoderOnlyModel,
        learning_rate: float,
        beta: float,
        pad_token_id: int,
    ) -> None:
        self.model = model
        self.ref_model = ref_model
        self.learning_rate = learning_rate
        self.beta = beta
        self.pad_token_id = pad_token_id

        self.model.train()
        self.ref_model.eval()

    @classmethod
    def from_sft_model(
        cls,
        sft_model: DecoderOnlyModel,
        learning_rate: float,
        beta: float,
        pad_token_id: int,
    ) -> "DPOTrainer":
        """Create a DPOTrainer by deep-copying the SFT model.

        Both the policy and reference are independent copies, so the
        caller's *sft_model* is left untouched.

        Args:
            sft_model: The supervised fine-tuned model to copy.
            learning_rate: SGD step size.
            beta: DPO temperature parameter.
            pad_token_id: Token ID used for padding.
        """
        policy_model: DecoderOnlyModel = copy.deepcopy(sft_model)
        ref_model: DecoderOnlyModel = copy.deepcopy(sft_model)
        return cls(
            model=policy_model,
            ref_model=ref_model,
            learning_rate=learning_rate,
            beta=beta,
            pad_token_id=pad_token_id,
        )

    def _prepare_batch(
        self,
        chosen_ids: npt.NDArray[np.intp],
        rejected_ids: npt.NDArray[np.intp],
    ) -> tuple[
        npt.NDArray[np.intp],
        npt.NDArray[np.intp],
        npt.NDArray[np.intp],
        npt.NDArray[np.bool_],
        npt.NDArray[np.bool_],
    ]:
        """Shift into input/target pairs and build masks.

        Returns:
            Tuple of (concatenated_inputs, chosen_targets, rejected_targets,
            chosen_mask, rejected_mask).
        """
        # Shift: input is all-but-last, target is all-but-first
        chosen_input = chosen_ids[:, :-1].astype(np.intp)
        chosen_target = chosen_ids[:, 1:].astype(np.intp)
        rejected_input = rejected_ids[:, :-1].astype(np.intp)
        rejected_target = rejected_ids[:, 1:].astype(np.intp)

        # Mask: True where target is NOT padding
        chosen_mask: npt.NDArray[np.bool_] = chosen_target != self.pad_token_id
        rejected_mask: npt.NDArray[np.bool_] = rejected_target != self.pad_token_id

        # Concatenate inputs along batch dimension for single forward pass
        concat_input = np.concatenate([chosen_input, rejected_input], axis=0)

        return concat_input, chosen_target, rejected_target, chosen_mask, rejected_mask

    def train_step(
        self,
        chosen_ids: npt.NDArray[np.intp],
        rejected_ids: npt.NDArray[np.intp],
    ) -> DPOMetrics:
        """Run one DPO training step on a batch of preference pairs.

        Args:
            chosen_ids: Shape (batch, seq_len) token IDs for chosen sequences.
            rejected_ids: Shape (batch, seq_len) token IDs for rejected sequences.

        Returns:
            A :class:`DPOMetrics` with loss, reward margin, and accuracy.
            Reward margin is mean(beta * (log_ratio_w - log_ratio_l)).
            Accuracy is the fraction of pairs where the model prefers chosen.
        """
        concat_input, chosen_target, rejected_target, chosen_mask, rejected_mask = (
            self._prepare_batch(chosen_ids, rejected_ids)
        )

        # Forward pass: policy and reference on the concatenated batch
        policy_logits = self.model.forward(concat_input)
        ref_logits = self.ref_model.forward(concat_input)

        result = dpo_loss_and_grad(
            policy_logits,
            ref_logits,
            chosen_target,
            rejected_target,
            chosen_mask,
            rejected_mask,
            self.beta,
        )

        # Backward and update (policy only)
        self.model.backward(result.grad)
        self.model.update_params(self.learning_rate)

        # Compute metrics from cached log-prob sums (no recomputation)
        reward_diff = self.beta * (
            (result.log_pi_w - result.log_ref_w) - (result.log_pi_l - result.log_ref_l)
        )
        reward_margin = float(np.mean(reward_diff))
        accuracy = float(np.mean((reward_diff > 0).astype(np.float64)))

        return DPOMetrics(loss=result.loss, reward_margin=reward_margin, accuracy=accuracy)

    def train_epoch(
        self,
        chosen_ids: npt.NDArray[np.intp],
        rejected_ids: npt.NDArray[np.intp],
        batch_size: int,
        rng: np.random.Generator,
    ) -> DPOMetrics:
        """Train for one epoch over all preference pairs.

        Args:
            chosen_ids: Shape (n_pairs, seq_len).
            rejected_ids: Shape (n_pairs, seq_len).
            batch_size: Number of preference pairs per mini-batch.
            rng: Random generator for shuffling.

        Returns:
            A :class:`DPOMetrics` with mean loss, reward margin, and
            accuracy across all batches in the epoch.
        """
        n_pairs = chosen_ids.shape[0]
        indices = rng.permutation(n_pairs)

        total_loss = 0.0
        total_margin = 0.0
        total_acc = 0.0
        n_batches = 0

        for start in range(0, n_pairs, batch_size):
            end = min(start + batch_size, n_pairs)
            batch_idx = indices[start:end]

            metrics = self.train_step(
                chosen_ids[batch_idx],
                rejected_ids[batch_idx],
            )
            total_loss += metrics.loss
            total_margin += metrics.reward_margin
            total_acc += metrics.accuracy
            n_batches += 1

        denom = max(n_batches, 1)
        return DPOMetrics(
            loss=total_loss / denom,
            reward_margin=total_margin / denom,
            accuracy=total_acc / denom,
        )
