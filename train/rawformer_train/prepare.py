"""Tokenize stage: train BPE tokenizer on corpus and produce tokenized arrays.

Reads raw text, trains a BPE tokenizer, encodes the corpus into integer
sequences, splits into train/val sets, and saves all artifacts.
"""

from logging import getLogger
from pathlib import Path

import numpy as np
import numpy.typing as npt

from rawformer.tokenizers.bpe import BPETokenizer
from rawformer_train._metrics import TokenizeMetrics
from rawformer_train._params import TokenizeParams, load_params
from rawformer_train._utils import pad_and_truncate, save_json_metrics

logger = getLogger(__name__)


def load_corpus(path: Path) -> list[str]:
    """Load pretrain corpus as a list of text lines.

    Args:
        path: Path to the corpus text file.

    Returns:
        Non-empty lines from the file.
    """
    logger.info("Loading corpus from %s", path)
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]
    logger.info("Loaded %d lines", len(lines))
    return lines


def _train_tokenizer(corpus: list[str], vocab_size: int) -> BPETokenizer:
    """Train a BPE tokenizer on the corpus.

    Args:
        corpus: List of text lines.
        vocab_size: Target vocabulary size.

    Returns:
        A trained BPE tokenizer.
    """
    tokenizer = BPETokenizer()
    logger.info("Training BPE tokenizer with vocab_size=%d", vocab_size)
    tokenizer.train(corpus, vocab_size=vocab_size)
    return tokenizer


def tokenize_corpus(
    corpus: list[str],
    tokenizer: BPETokenizer,
    max_seq_len: int,
) -> npt.NDArray[np.intp]:
    """Encode corpus lines and pad/truncate to fixed length.

    Returns array of shape ``(n_lines, max_seq_len)``.
    """
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    all_ids: list[list[int]] = []

    for line in corpus:
        ids = tokenizer.encode(line, add_special_tokens=True)
        all_ids.append(pad_and_truncate(ids, max_seq_len, pad_id, eos_id))

    return np.array(all_ids, dtype=np.intp)


def split_data(
    token_ids: npt.NDArray[np.intp],
    val_split: float,
    random_seed: int,
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Split tokenized data into train and validation sets."""
    rng = np.random.default_rng(seed=random_seed)
    indices = rng.permutation(token_ids.shape[0])
    token_ids = token_ids[indices]

    n_val = max(1, int(token_ids.shape[0] * val_split))
    val_ids: npt.NDArray[np.intp] = token_ids[:n_val]
    train_ids: npt.NDArray[np.intp] = token_ids[n_val:]

    logger.info("Split: %d train, %d val", train_ids.shape[0], val_ids.shape[0])
    return train_ids, val_ids


def _save_artifacts(
    tokenizer_dir: Path,
    tokenizer: BPETokenizer,
    train_ids: npt.NDArray[np.intp],
    val_ids: npt.NDArray[np.intp],
) -> None:
    """Save tokenizer and tokenized arrays to the tokenizer directory."""
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(tokenizer_dir / "tokenizer.json")
    np.save(tokenizer_dir / "train_ids.npy", train_ids)
    np.save(tokenizer_dir / "val_ids.npy", val_ids)
    logger.info("Saved tokenizer artifacts to %s", tokenizer_dir)


def _compile_metrics(
    tokenizer: BPETokenizer,
    train_ids: npt.NDArray[np.intp],
    val_ids: npt.NDArray[np.intp],
) -> TokenizeMetrics:
    """Compile tokenization summary metrics.

    Args:
        tokenizer: The trained tokenizer.
        train_ids: Training token ID array.
        val_ids: Validation token ID array.

    Returns:
        Metrics dictionary for JSON serialisation.
    """
    return TokenizeMetrics(
        vocab_size=tokenizer.vocab_size,
        n_train_sequences=int(train_ids.shape[0]),
        n_val_sequences=int(val_ids.shape[0]),
        seq_len=int(train_ids.shape[1]),
    )


def tokenize(
    corpus_path: Path,
    tokenizer_dir: Path,
    metrics_path: Path,
    params_path: Path,
) -> None:
    """Orchestrate the tokenize stage.

    Args:
        corpus_path: Path to the raw text corpus.
        tokenizer_dir: Output directory for tokenizer and tokenized arrays.
        metrics_path: Path to write the tokenization metrics JSON.
        params_path: Path to the params.yaml configuration file.
    """
    # Load params
    params: TokenizeParams = load_params(params_path).tokenize

    # Load data
    corpus = load_corpus(corpus_path)

    # Train tokenizer
    tokenizer = _train_tokenizer(corpus, params.vocab_size)

    # Tokenize and split
    token_ids = tokenize_corpus(corpus, tokenizer, params.max_seq_len)
    train_ids, val_ids = split_data(token_ids, params.val_split, params.random_seed)

    # Save
    _save_artifacts(tokenizer_dir, tokenizer, train_ids, val_ids)

    # Metrics
    save_json_metrics(_compile_metrics(tokenizer, train_ids, val_ids), metrics_path)
