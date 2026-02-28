"""Align stage: DPO preference alignment on the SFT model.

Loads the SFT model and tokenizer, formats preference triples
(prompt + chosen, prompt + rejected), and trains with the DPO
objective (Rafailov et al., 2023).
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
from rawformer.training.dpo import DPOTrainer
from rawformer_train._paths import (
    ALIGN_DIR,
    DPO_DATA_PATH,
    SFT_DIR,
    TOKENIZER_DIR,
)
from rawformer_train._utils import load_stage_params, pad_and_truncate

logger = logging.getLogger(__name__)


class PreferencePair(TypedDict):
    """A single preference triple from the DPO dataset."""

    prompt: str
    chosen: str
    rejected: str


class AlignParams(TypedDict):
    """Typed parameters for the align stage from params.yaml."""

    batch_size: int
    beta: float
    epochs: int
    learning_rate: float
    max_seq_len: int
    random_seed: int


def _load_params() -> AlignParams:
    """Load the align stage parameters from params.yaml."""
    return load_stage_params("align", AlignParams)


def _load_tokenizer() -> BPETokenizer:
    """Load the trained BPE tokenizer from artifacts."""
    return BPETokenizer.load(TOKENIZER_DIR / "tokenizer.json")


def _load_sft_model() -> DecoderOnlyModel:
    """Load the SFT model from artifacts/sft/."""
    model_path = SFT_DIR / "model.pkl"
    with open(model_path, "rb") as f:
        model: DecoderOnlyModel = pickle.load(f)
    return model


def load_preference_data(path: str | None = None) -> list[PreferencePair]:
    """Load DPO preference triples from a JSONL file.

    Args:
        path: Defaults to DPO_DATA_PATH.
    """
    data_path = Path(path) if path is not None else DPO_DATA_PATH
    logger.info("Loading preference data from %s", data_path)
    pairs: list[PreferencePair] = []
    with open(data_path) as f:
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


def _save_model(model: DecoderOnlyModel) -> None:
    """Serialize the aligned model to artifacts/align/."""
    ALIGN_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ALIGN_DIR / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Aligned model saved to %s", model_path)


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


def _save_metrics(
    epoch_losses: list[float],
    epoch_margins: list[float],
    epoch_accuracies: list[float],
    params: AlignParams,
    sample_generation: str,
) -> None:
    """Write align metrics to artifacts/align/align-metrics.json."""
    metrics = {
        "final_loss": round(epoch_losses[-1], 6) if epoch_losses else 0.0,
        "final_reward_margin": round(epoch_margins[-1], 6) if epoch_margins else 0.0,
        "final_accuracy": round(epoch_accuracies[-1], 6) if epoch_accuracies else 0.0,
        "n_epochs": params["epochs"],
        "beta": params["beta"],
        "epoch_losses": [round(v, 6) for v in epoch_losses],
        "epoch_reward_margins": [round(v, 6) for v in epoch_margins],
        "epoch_accuracies": [round(v, 6) for v in epoch_accuracies],
        "sample_generation": sample_generation,
    }
    metrics_path = ALIGN_DIR / "align-metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Align metrics written to %s", metrics_path)


def main() -> None:
    """Orchestrate the DPO alignment stage."""
    params = _load_params()
    rng = np.random.default_rng(params["random_seed"])

    tokenizer = _load_tokenizer()
    model = _load_sft_model()

    pairs = load_preference_data()
    chosen_ids, rejected_ids = format_preference_pairs(pairs, tokenizer, params["max_seq_len"])

    trainer = DPOTrainer.from_sft_model(
        sft_model=model,
        learning_rate=params["learning_rate"],
        beta=params["beta"],
        pad_token_id=tokenizer.pad_token_id,
    )

    logger.info(
        "Align: %d epochs, batch_size=%d, beta=%s, lr=%s, %d pairs",
        params["epochs"],
        params["batch_size"],
        params["beta"],
        params["learning_rate"],
        chosen_ids.shape[0],
    )

    epoch_losses: list[float] = []
    epoch_margins: list[float] = []
    epoch_accuracies: list[float] = []

    for epoch in range(params["epochs"]):
        metrics = trainer.train_epoch(chosen_ids, rejected_ids, params["batch_size"], rng)
        epoch_losses.append(metrics.loss)
        epoch_margins.append(metrics.reward_margin)
        epoch_accuracies.append(metrics.accuracy)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                "Align Epoch %d/%d: loss=%.4f margin=%.4f acc=%.4f",
                epoch + 1,
                params["epochs"],
                metrics.loss,
                metrics.reward_margin,
                metrics.accuracy,
            )

    sample_text = _generate_sample(trainer.model, tokenizer, pairs)
    logger.info("Sample generation: %s", sample_text)

    _save_model(trainer.model)
    _save_metrics(epoch_losses, epoch_margins, epoch_accuracies, params, sample_text)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
