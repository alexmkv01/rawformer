"""Masked Language Modelling utilities (Devlin et al., 2018).

Implements BERT-style token masking and an MLM prediction head.
- 80% of selected tokens are replaced with [MASK]
- 10% are replaced with a random token
- 10% are kept unchanged
The model must predict the original token at each masked position.
"""

import numpy as np
import numpy.typing as npt

from rawformer.core.base import Layer
from rawformer.core.exceptions import ForwardNotCalledError
from rawformer.layers.linear import LinearLayer


def mask_tokens(
    token_ids: npt.NDArray[np.intp],
    mask_prob: float,
    mask_token_id: int,
    vocab_size: int,
    rng: np.random.Generator,
    special_token_ids: set[int] | None = None,
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], npt.NDArray[np.bool_]]:
    """Apply BERT-style 80/10/10 random masking to a batch of token sequences.

    Returns (masked_ids, labels, mask) where labels holds the original
    token at masked positions and -1 elsewhere.
    """
    if special_token_ids is None:
        special_token_ids = set()

    masked_ids: npt.NDArray[np.intp] = token_ids.copy()
    labels: npt.NDArray[np.intp] = np.full_like(token_ids, -1)

    prob_matrix = rng.random(token_ids.shape)
    for special_id in special_token_ids:
        prob_matrix[token_ids == special_id] = 1.0

    mask: npt.NDArray[np.bool_] = prob_matrix < mask_prob
    labels[mask] = token_ids[mask]

    # 80% -> [MASK], 10% -> random token, 10% -> keep original
    replace_mask_prob = rng.random(token_ids.shape)
    indices_mask = mask & (replace_mask_prob < 0.8)
    masked_ids[indices_mask] = np.intp(mask_token_id)

    indices_random = mask & (replace_mask_prob >= 0.8) & (replace_mask_prob < 0.9)
    n_random = int(np.sum(indices_random))
    if n_random > 0:
        random_tokens = rng.integers(0, vocab_size, size=n_random).astype(np.intp)
        masked_ids[indices_random] = random_tokens

    return masked_ids, labels, mask


class MLMHead(Layer):
    """Masked Language Modelling prediction head.

    Projects hidden states to vocabulary logits.
    """

    def __init__(self, d_model: int, vocab_size: int, rng: np.random.Generator) -> None:
        self.proj = LinearLayer(d_model, vocab_size, rng)
        self._forward_called: bool = False

    def forward(self, hidden_states: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Project hidden states to vocabulary logits."""
        self._forward_called = True
        return self.proj.forward(hidden_states)

    def backward(self, grad_z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Backward through the projection."""
        if not self._forward_called:
            raise ForwardNotCalledError("MLMHead")
        return self.proj.backward(grad_z)

    def update_params(self, learning_rate: float) -> None:
        self.proj.update_params(learning_rate)
