"""Tokenize stage: train BPE tokenizer on corpus and produce tokenized arrays.

Reads raw text, trains a BPE tokenizer, encodes the corpus into integer
sequences, splits into train/val sets, and saves all artifacts.
"""

import json
import logging
from pathlib import Path
from typing import TypedDict

import numpy as np
import numpy.typing as npt

from rawformer.tokenizers.bpe import BPETokenizer
from rawformer_train._paths import (
    PRETRAIN_DATA_PATH,
    TOKENIZER_DIR,
)
from rawformer_train._utils import load_stage_params, pad_and_truncate

logger = logging.getLogger(__name__)


class TokenizeParams(TypedDict):
    """Typed parameters for the tokenize stage from params.yaml."""

    vocab_size: int
    max_seq_len: int
    random_seed: int
    val_split: float


def _load_params() -> TokenizeParams:
    """Load the tokenize stage parameters from params.yaml."""
    return load_stage_params("tokenize", TokenizeParams)


def load_corpus(path: str | None = None) -> list[str]:
    """Load pretrain corpus as a list of text lines.

    Args:
        path: Defaults to PRETRAIN_DATA_PATH.
    """
    corpus_path = Path(path) if path is not None else PRETRAIN_DATA_PATH
    logger.info("Loading corpus from %s", corpus_path)
    with open(corpus_path) as f:
        lines = [line.strip() for line in f if line.strip()]
    logger.info("Loaded %d lines", len(lines))
    return lines


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
    tokenizer: BPETokenizer,
    train_ids: npt.NDArray[np.intp],
    val_ids: npt.NDArray[np.intp],
) -> None:
    """Save tokenizer and tokenized arrays to artifacts/tokenizer/."""
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save(TOKENIZER_DIR / "tokenizer.json")
    np.save(TOKENIZER_DIR / "train_ids.npy", train_ids)
    np.save(TOKENIZER_DIR / "val_ids.npy", val_ids)
    logger.info("Saved tokenizer artifacts to %s", TOKENIZER_DIR)


def _save_metrics(
    tokenizer: BPETokenizer,
    train_ids: npt.NDArray[np.intp],
    val_ids: npt.NDArray[np.intp],
) -> None:
    """Write tokenization metrics to artifacts/tokenizer/tokenize-metrics.json."""
    metrics = {
        "vocab_size": tokenizer.vocab_size,
        "n_train_sequences": train_ids.shape[0],
        "n_val_sequences": val_ids.shape[0],
        "seq_len": int(train_ids.shape[1]),
    }
    metrics_path = TOKENIZER_DIR / "tokenize-metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Tokenize metrics written to %s", metrics_path)


def main() -> None:
    """Orchestrate the tokenize stage."""
    params = _load_params()

    corpus = load_corpus()

    tokenizer = BPETokenizer()
    logger.info("Training BPE tokenizer with vocab_size=%d", params["vocab_size"])
    tokenizer.train(corpus, vocab_size=params["vocab_size"])

    token_ids = tokenize_corpus(corpus, tokenizer, params["max_seq_len"])
    train_ids, val_ids = split_data(token_ids, params["val_split"], params["random_seed"])

    _save_artifacts(tokenizer, train_ids, val_ids)
    _save_metrics(tokenizer, train_ids, val_ids)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
