"""Align stage: DPO preference alignment on the SFT model.

Loads the SFT model and tokenizer, formats preference triples
(prompt + chosen, prompt + rejected), and trains with the DPO
objective (Rafailov et al., 2023).
"""

import json
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import TypedDict

import numpy as np
import numpy.typing as npt

from rawformer.tokenizers.bpe import BPETokenizer
from rawformer.training.clm import DecoderOnlyModel
from rawformer.training.dpo import DPOTrainer
from rawformer_train._metrics import AlignMetrics
from rawformer_train._params import AlignParams, load_params
from rawformer_train._utils import (
    load_model,
    load_tokenizer,
    pad_and_truncate,
    save_json_metrics,
    save_model,
)

logger = getLogger(__name__)


class PreferencePair(TypedDict):
    """A single preference triple from the DPO dataset."""

    prompt: str
    chosen: str
    rejected: str


@dataclass
class AlignTrainingResult:
    """Accumulated metrics from the DPO training loop."""

    epoch_losses: list[float]
    epoch_margins: list[float]
    epoch_accuracies: list[float]


def load_preference_data(path: Path) -> list[PreferencePair]:
    """Load DPO preference triples from a JSONL file.

    Args:
        path: Path to the JSONL data file.

    Returns:
        List of preference triples.
    """
    logger.info("Loading preference data from %s", path)
    pairs: list[PreferencePair] = []
    with open(path) as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                raw: dict[str, str] = json.loads(line)
                for key in ("prompt", "chosen", "rejected"):
                    if key not in raw:
                        raise ValueError(f"DPO line {line_num}: missing {key!r} key")
                pair: PreferencePair = {
                    "prompt": raw["prompt"],
                    "chosen": raw["chosen"],
                    "rejected": raw["rejected"],
                }
                pairs.append(pair)
    logger.info("Loaded %d preference pairs", len(pairs))
    return pairs


def format_preference_pairs(
    pairs: list[PreferencePair],
    tokenizer: BPETokenizer,
    max_seq_len: int,
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Tokenize preference pairs into chosen and rejected ID arrays.

    Each sequence is formatted as ``<bos> prompt response <eos> <pad>...``.
    No explicit separator is inserted between prompt and response —
    this matches the SFT formatting so the log-probabilities are
    comparable between training stages.

    Returns:
        Tuple of (chosen_ids, rejected_ids), each of shape
        ``(n_pairs, max_seq_len)``.
    """
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    chosen_all: list[list[int]] = []
    rejected_all: list[list[int]] = []

    for pair in pairs:
        chosen_text = f"{pair['prompt']} {pair['chosen']}"
        rejected_text = f"{pair['prompt']} {pair['rejected']}"

        chosen_ids = tokenizer.encode(chosen_text, add_special_tokens=True)
        rejected_ids = tokenizer.encode(rejected_text, add_special_tokens=True)

        chosen_all.append(pad_and_truncate(chosen_ids, max_seq_len, pad_id, eos_id))
        rejected_all.append(pad_and_truncate(rejected_ids, max_seq_len, pad_id, eos_id))

    return (
        np.array(chosen_all, dtype=np.intp),
        np.array(rejected_all, dtype=np.intp),
    )


def _run_alignment(
    trainer: DPOTrainer,
    chosen_ids: npt.NDArray[np.intp],
    rejected_ids: npt.NDArray[np.intp],
    batch_size: int,
    epochs: int,
    rng: np.random.Generator,
) -> AlignTrainingResult:
    """Run the DPO training loop with periodic logging.

    Args:
        trainer: The DPO trainer.
        chosen_ids: Tokenized chosen sequences.
        rejected_ids: Tokenized rejected sequences.
        batch_size: Batch size for training.
        epochs: Number of training epochs.
        rng: Random number generator for batching.

    Returns:
        Accumulated per-epoch losses, reward margins, and accuracies.
    """
    epoch_losses: list[float] = []
    epoch_margins: list[float] = []
    epoch_accuracies: list[float] = []

    for epoch in range(epochs):
        epoch_metrics = trainer.train_epoch(chosen_ids, rejected_ids, batch_size, rng)
        epoch_losses.append(epoch_metrics.loss)
        epoch_margins.append(epoch_metrics.reward_margin)
        epoch_accuracies.append(epoch_metrics.accuracy)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                "Align Epoch %d/%d: loss=%.4f margin=%.4f acc=%.4f",
                epoch + 1,
                epochs,
                epoch_metrics.loss,
                epoch_metrics.reward_margin,
                epoch_metrics.accuracy,
            )

    return AlignTrainingResult(epoch_losses, epoch_margins, epoch_accuracies)


def _generate_sample(
    model: DecoderOnlyModel,
    tokenizer: BPETokenizer,
    pairs: list[PreferencePair],
) -> str:
    """Generate a sample response using the first prompt from the dataset."""
    prompt_text = pairs[0]["prompt"]
    prompt_ids = np.array(tokenizer.encode(prompt_text, add_special_tokens=True), dtype=np.intp)
    generated_ids = model.generate(
        prompt_ids,
        max_tokens=50,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(generated_ids.tolist())


def _compile_metrics(
    result: AlignTrainingResult,
    params: AlignParams,
    sample_generation: str,
) -> AlignMetrics:
    """Compile DPO alignment summary metrics.

    Args:
        result: Accumulated training metrics.
        params: Align stage parameters.
        sample_generation: Sample text from the aligned model.

    Returns:
        Metrics dictionary for JSON serialisation.
    """
    return AlignMetrics(
        final_loss=round(result.epoch_losses[-1], 6) if result.epoch_losses else 0.0,
        final_reward_margin=round(result.epoch_margins[-1], 6) if result.epoch_margins else 0.0,
        final_accuracy=round(result.epoch_accuracies[-1], 6) if result.epoch_accuracies else 0.0,
        n_epochs=params.epochs,
        beta=params.beta,
        epoch_losses=[round(v, 6) for v in result.epoch_losses],
        epoch_reward_margins=[round(v, 6) for v in result.epoch_margins],
        epoch_accuracies=[round(v, 6) for v in result.epoch_accuracies],
        sample_generation=sample_generation,
    )


def align(
    tokenizer_dir: Path,
    sft_dir: Path,
    dpo_data_path: Path,
    align_dir: Path,
    metrics_path: Path,
    params_path: Path,
) -> None:
    """Orchestrate the DPO alignment stage.

    Args:
        tokenizer_dir: Directory containing the trained tokenizer.
        sft_dir: Directory containing the SFT model.
        dpo_data_path: Path to the DPO preference JSONL file.
        align_dir: Output directory for the aligned model.
        metrics_path: Path to write align metrics JSON.
        params_path: Path to the params.yaml configuration file.
    """
    # Load params
    params = load_params(params_path).align
    rng = np.random.default_rng(params.random_seed)

    # Load
    tokenizer = load_tokenizer(tokenizer_dir)
    model = load_model(sft_dir / "model.pkl")
    pairs = load_preference_data(dpo_data_path)
    chosen_ids, rejected_ids = format_preference_pairs(pairs, tokenizer, params.max_seq_len)

    # Train
    trainer = DPOTrainer.from_sft_model(
        sft_model=model,
        learning_rate=params.learning_rate,
        beta=params.beta,
        pad_token_id=tokenizer.pad_token_id,
    )
    result = _run_alignment(
        trainer, chosen_ids, rejected_ids, params.batch_size, params.epochs, rng
    )
    sample_text = _generate_sample(trainer.model, tokenizer, pairs)

    # Save
    save_model(trainer.model, align_dir, "model.pkl")

    # Metrics
    save_json_metrics(_compile_metrics(result, params, sample_text), metrics_path)
