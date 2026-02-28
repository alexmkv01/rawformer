"""SFT (supervised fine-tuning) stage: fine-tune pretrained model on instructions.

Loads the pretrained decoder-only model and tokenizer, formats
instruction-response pairs, and fine-tunes with the CLM objective
(loss computed on the full sequence for simplicity at this scale).
"""

import json
from logging import getLogger
from pathlib import Path
from typing import TypedDict

import numpy as np
import numpy.typing as npt

from rawformer.tokenizers.bpe import BPETokenizer
from rawformer.training.clm import DecoderOnlyModel
from rawformer.training.lm_trainer import LMTrainer
from rawformer_train._metrics import SFTMetrics
from rawformer_train._params import SFTParams, load_params
from rawformer_train._utils import (
    load_model,
    load_tokenizer,
    pad_and_truncate,
    save_json_metrics,
    save_model,
)

logger = getLogger(__name__)


class SFTExample(TypedDict):
    """A single instruction-response pair for supervised fine-tuning."""

    instruction: str
    response: str


def load_sft_data(path: Path) -> list[SFTExample]:
    """Load SFT instruction-response pairs from a JSONL file.

    Args:
        path: Path to the JSONL data file.

    Returns:
        List of instruction-response examples.
    """
    logger.info("Loading SFT data from %s", path)
    examples: list[SFTExample] = []
    with open(path) as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                raw: dict[str, str] = json.loads(line)
                if "instruction" not in raw or "response" not in raw:
                    raise ValueError(
                        f"SFT line {line_num}: missing 'instruction' or 'response' key"
                    )
                example: SFTExample = {
                    "instruction": raw["instruction"],
                    "response": raw["response"],
                }
                examples.append(example)
    logger.info("Loaded %d SFT examples", len(examples))
    return examples


def format_sft_examples(
    examples: list[SFTExample],
    tokenizer: BPETokenizer,
    max_seq_len: int,
) -> npt.NDArray[np.intp]:
    """Format and tokenize instruction-response pairs.

    Produces ``<bos> instruction response <eos> <pad>...`` sequences,
    returned as an array of shape ``(n_examples, max_seq_len)``.

    No explicit separator is inserted between instruction and response;
    at this toy scale the model learns from the full sequence.
    """
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    all_ids: list[list[int]] = []

    for example in examples:
        text = f"{example['instruction']} {example['response']}"
        ids = tokenizer.encode(text, add_special_tokens=True)
        all_ids.append(pad_and_truncate(ids, max_seq_len, pad_id, eos_id))

    return np.array(all_ids, dtype=np.intp)


def _run_training(
    trainer: LMTrainer,
    sft_ids: npt.NDArray[np.intp],
    epochs: int,
    rng: np.random.Generator,
) -> list[float]:
    """Run the SFT training loop with periodic logging.

    Args:
        trainer: The language model trainer.
        sft_ids: Tokenized SFT examples.
        epochs: Number of training epochs.
        rng: Random number generator for batching.

    Returns:
        Per-epoch training losses.
    """
    train_losses: list[float] = []
    for epoch in range(epochs):
        loss = trainer.train_epoch(sft_ids, rng)
        train_losses.append(loss)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info("SFT Epoch %d/%d: loss=%.4f", epoch + 1, epochs, loss)
    return train_losses


def _generate_sample(model: DecoderOnlyModel, tokenizer: BPETokenizer) -> str:
    """Generate a sample response to demonstrate what the SFT model has learned."""
    prompt_text = "Write a short story about a dog."
    prompt_ids = np.array(tokenizer.encode(prompt_text, add_special_tokens=True), dtype=np.intp)
    generated_ids = model.generate(
        prompt_ids,
        max_tokens=50,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(generated_ids.tolist())


def _compile_metrics(
    train_losses: list[float],
    params: SFTParams,
    sample_generation: str,
) -> SFTMetrics:
    """Compile SFT summary metrics.

    Args:
        train_losses: Per-epoch training losses.
        params: SFT stage parameters.
        sample_generation: Sample text from the fine-tuned model.

    Returns:
        Metrics dictionary for JSON serialisation.
    """
    return SFTMetrics(
        final_train_loss=round(train_losses[-1], 6) if train_losses else 0.0,
        n_epochs=params.epochs,
        epoch_losses=[round(loss, 6) for loss in train_losses],
        sample_generation=sample_generation,
    )


def sft(
    tokenizer_dir: Path,
    pretrain_dir: Path,
    sft_data_path: Path,
    sft_dir: Path,
    metrics_path: Path,
    params_path: Path,
) -> None:
    """Orchestrate the SFT stage.

    Args:
        tokenizer_dir: Directory containing the trained tokenizer.
        pretrain_dir: Directory containing the pretrained model.
        sft_data_path: Path to the SFT instruction-response JSONL file.
        sft_dir: Output directory for the fine-tuned model.
        metrics_path: Path to write SFT metrics JSON.
        params_path: Path to the params.yaml configuration file.
    """
    # Load params
    params = load_params(params_path).sft
    rng = np.random.default_rng(params.random_seed)

    # Load
    tokenizer = load_tokenizer(tokenizer_dir)
    model = load_model(pretrain_dir / "model.pkl")
    examples = load_sft_data(sft_data_path)
    sft_ids = format_sft_examples(examples, tokenizer, params.max_seq_len)

    # Train
    trainer = LMTrainer(
        model=model,
        learning_rate=params.learning_rate,
        batch_size=params.batch_size,
        pad_token_id=tokenizer.pad_token_id,
    )
    train_losses = _run_training(trainer, sft_ids, params.epochs, rng)
    sample_text = _generate_sample(model, tokenizer)

    # Save
    save_model(model, sft_dir, "model.pkl")

    # Metrics
    save_json_metrics(_compile_metrics(train_losses, params, sample_text), metrics_path)
