"""SFT (supervised fine-tuning) stage: fine-tune pretrained model on instructions.

Loads the pretrained decoder-only model and tokenizer, formats
instruction-response pairs, and fine-tunes with the CLM objective
(loss computed on the full sequence for simplicity at this scale).
"""

import json
import logging
import pickle
from pathlib import Path
from typing import TypedDict

import numpy as np
import numpy.typing as npt

from rawformer.tokenizers.bpe import BPETokenizer
from rawformer.training.clm import DecoderOnlyModel
from rawformer.training.lm_trainer import LMTrainer
from rawformer_train._paths import (
    PRETRAIN_DIR,
    SFT_DATA_PATH,
    SFT_DIR,
    TOKENIZER_DIR,
)
from rawformer_train._utils import load_stage_params, pad_and_truncate

logger = logging.getLogger(__name__)


class SFTExample(TypedDict):
    """A single instruction-response pair for supervised fine-tuning."""

    instruction: str
    response: str


class SFTParams(TypedDict):
    """Typed parameters for the SFT stage from params.yaml."""

    batch_size: int
    epochs: int
    learning_rate: float
    max_seq_len: int
    random_seed: int


def _load_params() -> SFTParams:
    """Load the SFT stage parameters from params.yaml."""
    return load_stage_params("sft", SFTParams)


def _load_tokenizer() -> BPETokenizer:
    """Load the trained BPE tokenizer from artifacts."""
    return BPETokenizer.load(TOKENIZER_DIR / "tokenizer.json")


def _load_pretrained_model() -> DecoderOnlyModel:
    """Load the pretrained model from artifacts/pretrain/."""
    model_path = PRETRAIN_DIR / "model.pkl"
    with open(model_path, "rb") as f:
        model: DecoderOnlyModel = pickle.load(f)
    return model


def load_sft_data(path: str | None = None) -> list[SFTExample]:
    """Load SFT instruction-response pairs from a JSONL file.

    Args:
        path: Defaults to SFT_DATA_PATH.
    """
    data_path = Path(path) if path is not None else SFT_DATA_PATH
    logger.info("Loading SFT data from %s", data_path)
    examples: list[SFTExample] = []
    with open(data_path) as f:
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


def _save_model(model: DecoderOnlyModel) -> None:
    """Serialize the fine-tuned model to artifacts/sft/."""
    SFT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = SFT_DIR / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info("SFT model saved to %s", model_path)


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


def _save_metrics(
    train_losses: list[float],
    params: SFTParams,
    sample_generation: str,
) -> None:
    """Write SFT metrics to artifacts/sft/sft-metrics.json."""
    metrics = {
        "final_train_loss": round(train_losses[-1], 6) if train_losses else 0.0,
        "n_epochs": params["epochs"],
        "epoch_losses": [round(loss, 6) for loss in train_losses],
        "sample_generation": sample_generation,
    }
    metrics_path = SFT_DIR / "sft-metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("SFT metrics written to %s", metrics_path)


def main() -> None:
    """Orchestrate the SFT stage."""
    params = _load_params()
    rng = np.random.default_rng(params["random_seed"])

    tokenizer = _load_tokenizer()
    model = _load_pretrained_model()

    examples = load_sft_data()
    sft_ids = format_sft_examples(examples, tokenizer, params["max_seq_len"])

    trainer = LMTrainer(
        model=model,
        learning_rate=params["learning_rate"],
        batch_size=params["batch_size"],
        pad_token_id=tokenizer.pad_token_id,
    )

    logger.info(
        "SFT: %d epochs, batch_size=%d, lr=%s, %d examples",
        params["epochs"],
        params["batch_size"],
        params["learning_rate"],
        sft_ids.shape[0],
    )

    train_losses: list[float] = []
    for epoch in range(params["epochs"]):
        loss = trainer.train_epoch(sft_ids, rng)
        train_losses.append(loss)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info("SFT Epoch %d/%d: loss=%.4f", epoch + 1, params["epochs"], loss)

    sample_text = _generate_sample(model, tokenizer)
    logger.info("Sample generation: %s", sample_text)

    _save_model(model)
    _save_metrics(train_losses, params, sample_text)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
