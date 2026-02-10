"""Language model trainer for causal (next-token) language modelling.

Handles mini-batch iteration over token sequences, computes cross-entropy
loss on the shifted prediction targets, and runs backward + update_params.
"""

import logging

import numpy as np
import numpy.typing as npt

from rawformer.training.clm import DecoderOnlyModel

logger = logging.getLogger(__name__)

_LOG_EPS: float = 1e-8


def _softmax(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Numerically stable softmax over the last axis."""
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    result: npt.NDArray[np.float64] = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    return result


def clm_loss_and_grad(
    logits: npt.NDArray[np.float64],
    targets: npt.NDArray[np.intp],
    ignore_index: int = -1,
) -> tuple[float, npt.NDArray[np.float64]]:
    """Compute cross-entropy loss and gradient for causal language modelling.

    Positions where targets == ignore_index are excluded from the loss.
    """
    batch, seq_len, vocab_size = logits.shape

    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)

    valid_mask = targets_flat != ignore_index
    n_valid = int(np.sum(valid_mask))

    if n_valid == 0:
        return 0.0, np.zeros_like(logits)

    probs = _softmax(logits_flat)

    valid_indices = np.where(valid_mask)[0]
    valid_targets = targets_flat[valid_indices]
    log_probs = np.log(probs[valid_indices, valid_targets] + _LOG_EPS)
    loss = float(-np.sum(log_probs) / n_valid)

    grad_flat = probs.copy()
    grad_flat[valid_indices, valid_targets] -= 1.0
    grad_flat[~valid_mask] = 0.0
    grad_flat /= n_valid

    grad: npt.NDArray[np.float64] = grad_flat.reshape(batch, seq_len, vocab_size)
    return loss, grad


class LMTrainer:
    """Mini-batch SGD trainer for causal language models.

    Uses next-token prediction: for input [t0, t1, ..., tn],
    the target at position i is t(i+1).
    """

    def __init__(
        self,
        model: DecoderOnlyModel,
        learning_rate: float,
        batch_size: int,
        pad_token_id: int,
    ) -> None:
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.pad_token_id = pad_token_id

    def _prepare_batch(
        self, batch: npt.NDArray[np.intp]
    ) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        """Shift token sequences into input/target pairs and mask padding."""
        input_ids = batch[:, :-1].astype(np.intp)
        # .astype() returns a copy, so the in-place mask below is safe.
        target_ids = batch[:, 1:].astype(np.intp)
        target_ids[target_ids == self.pad_token_id] = -1
        return input_ids, target_ids

    def train_epoch(
        self,
        token_ids: npt.NDArray[np.intp],
        rng: np.random.Generator,
    ) -> float:
        """Train for one epoch, returning average loss."""
        self.model.train()

        n_sequences = token_ids.shape[0]
        indices = rng.permutation(n_sequences)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n_sequences, self.batch_size):
            end = min(start + self.batch_size, n_sequences)
            batch = token_ids[indices[start:end]]
            input_ids, target_ids = self._prepare_batch(batch)

            logits = self.model.forward(input_ids)
            loss, grad = clm_loss_and_grad(logits, target_ids)

            self.model.backward(grad)
            self.model.update_params(self.learning_rate)

            total_loss += loss
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def eval_loss(self, token_ids: npt.NDArray[np.intp]) -> float:
        """Compute average loss on a dataset without updating parameters."""
        self.model.eval()

        n_sequences = token_ids.shape[0]
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n_sequences, self.batch_size):
            end = min(start + self.batch_size, n_sequences)
            input_ids, target_ids = self._prepare_batch(token_ids[start:end])

            logits = self.model.forward(input_ids)
            loss, _ = clm_loss_and_grad(logits, target_ids)

            total_loss += loss
            n_batches += 1

        return total_loss / max(n_batches, 1)
