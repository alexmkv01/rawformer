"""Shared utilities for the rawformer_train pipeline stages."""

import json
import pickle
from logging import getLogger
from pathlib import Path

from rawformer.tokenizers.bpe import BPETokenizer
from rawformer.training.clm import DecoderOnlyModel
from rawformer_train._metrics import PipelineMetrics
from rawformer_train.exceptions import ModelLoadError

logger = getLogger(__name__)


def pad_and_truncate(ids: list[int], max_seq_len: int, pad_id: int, eos_id: int) -> list[int]:
    """Truncate or pad a token ID sequence to exactly *max_seq_len*.

    When truncating, forces the last position to *eos_id* so the
    end-of-sequence signal is never silently dropped.
    """
    if len(ids) > max_seq_len:
        result = ids[:max_seq_len]
        result[-1] = eos_id
    else:
        result = ids + [pad_id] * (max_seq_len - len(ids))
    return result


def load_tokenizer(tokenizer_dir: Path) -> BPETokenizer:
    """Load a trained BPE tokenizer from a directory.

    Args:
        tokenizer_dir: Directory containing ``tokenizer.json``.

    Returns:
        The loaded tokenizer.
    """
    path = tokenizer_dir / "tokenizer.json"
    tokenizer = BPETokenizer.load(path)
    logger.info("Loaded tokenizer from %s", path)
    return tokenizer


def load_model(model_path: Path) -> DecoderOnlyModel:
    """Deserialize a pickled decoder-only model.

    Args:
        model_path: Path to the ``.pkl`` file.

    Returns:
        The loaded model.

    Raises:
        ModelLoadError: If the file does not contain a ``DecoderOnlyModel``.
    """
    with open(model_path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, DecoderOnlyModel):
        raise ModelLoadError(
            f"{model_path} contains {type(obj).__name__}, expected DecoderOnlyModel"
        )
    logger.info("Loaded model from %s", model_path)
    return obj


def save_model(model: DecoderOnlyModel, output_dir: Path, filename: str) -> None:
    """Serialize a decoder-only model to a pickle file.

    Args:
        model: The model to save.
        output_dir: Directory to write the file into (created if missing).
        filename: Name of the output file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Saved model to %s", path)


def save_json_metrics(metrics: PipelineMetrics, path: Path) -> None:
    """Write a metrics dictionary to a JSON file.

    Args:
        metrics: Metrics to serialize.
        path: Output file path (parent directory created if missing).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics written to %s", path)
