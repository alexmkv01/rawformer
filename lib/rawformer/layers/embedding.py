"""Token embedding and sinusoidal positional encoding (Vaswani et al., 2017)."""

import numpy as np
import numpy.typing as npt

from rawformer.exceptions import ForwardNotCalledError


class TokenEmbedding:
    """Learnable lookup-table embedding that maps token indices to dense vectors.

    The embedding matrix is scaled by sqrt(d_model) following the convention
    in "Attention Is All You Need" (Vaswani et al., 2017).

    Args:
        vocab_size: Number of tokens in the vocabulary.
        d_model: Embedding dimension.
        rng: NumPy random generator for weight initialization.
    """

    def __init__(self, vocab_size: int, d_model: int, rng: np.random.Generator) -> None:
        self.vocab_size = vocab_size
        self.d_model = d_model
        self._scale = np.sqrt(np.float64(d_model))
        self._weight: npt.NDArray[np.float64] = rng.standard_normal((vocab_size, d_model)) * 0.02
        self._input_cache: npt.NDArray[np.float64] | None = None
        self._grad_weight: npt.NDArray[np.float64] | None = None

    @property
    def weight(self) -> npt.NDArray[np.float64]:
        return self._weight

    def forward(self, token_ids: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Look up embeddings for token indices.

        Args:
            token_ids: Integer token indices of shape (batch, seq_len).

        Returns:
            Embeddings of shape (batch, seq_len, d_model), scaled by sqrt(d_model).
        """
        self._input_cache = token_ids
        result: npt.NDArray[np.float64] = self._weight[token_ids.astype(np.intp)] * self._scale
        return result

    def backward(self, grad_z: npt.NDArray[np.float64]) -> None:
        """Accumulate gradients for the embedding weight matrix.

        Args:
            grad_z: Upstream gradient of shape (batch, seq_len, d_model).
        """
        if self._input_cache is None:
            raise ForwardNotCalledError("TokenEmbedding")
        ids = self._input_cache.astype(np.intp).ravel()
        grad_flat = (grad_z * self._scale).reshape(-1, self.d_model)
        self._grad_weight = np.zeros_like(self._weight)
        np.add.at(self._grad_weight, ids, grad_flat)

    def update_params(self, learning_rate: float) -> None:
        if self._grad_weight is None:
            raise ForwardNotCalledError("TokenEmbedding")
        self._weight -= learning_rate * self._grad_weight


def sinusoidal_positional_encoding(max_len: int, d_model: int) -> npt.NDArray[np.float64]:
    """Generate fixed sinusoidal positional encoding table.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        max_len: Maximum sequence length.
        d_model: Model embedding dimension.

    Returns:
        Encoding table of shape (max_len, d_model).
    """
    positions = np.arange(max_len)[:, np.newaxis]
    dim_indices = np.arange(d_model)[np.newaxis, :]
    angles: npt.NDArray[np.float64] = positions / np.power(
        10000.0, (2.0 * (dim_indices // 2)) / d_model
    )
    pe = np.empty((max_len, d_model))
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return pe


class PositionalEncoding:
    """Adds fixed sinusoidal positional encoding to input embeddings.

    The encoding is precomputed at construction time and not learnable.
    No backward pass is needed since the PE is a constant additive term
    (gradient passes through unchanged).

    Args:
        max_len: Maximum sequence length supported.
        d_model: Model embedding dimension.
    """

    def __init__(self, max_len: int, d_model: int) -> None:
        self.max_len = max_len
        self.d_model = d_model
        self._pe: npt.NDArray[np.float64] = sinusoidal_positional_encoding(max_len, d_model)

    @property
    def encoding_table(self) -> npt.NDArray[np.float64]:
        return self._pe

    def forward(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Add positional encoding to input embeddings.

        Args:
            x: Input of shape (batch, seq_len, d_model).

        Returns:
            x + PE[:seq_len], same shape as input.
        """
        seq_len = x.shape[1]
        result: npt.NDArray[np.float64] = x + self._pe[:seq_len]
        return result

    def backward(self, grad_z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Pass gradient through unchanged (PE is constant)."""
        return grad_z
