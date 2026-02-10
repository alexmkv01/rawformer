"""Pretrain stage: train a decoder-only model with causal language modelling.

Loads tokenized data and a trained tokenizer, builds a tiny GPT-style
model from params.yaml, and trains with next-token prediction.
"""

import json
import logging
import pickle
from typing import TypedDict

import numpy as np
import numpy.typing as npt
import yaml

from rawformer.tokenizers.bpe import BPETokenizer
from rawformer.training.clm import DecoderOnlyModel
from rawformer.training.lm_trainer import LMTrainer
from rawformer_train._paths import PARAMS_PATH, PRETRAIN_DIR, TOKENIZER_DIR

logger = logging.getLogger(__name__)


class PretrainParams(TypedDict):
    """Typed parameters for the pretrain stage from params.yaml."""

    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    max_len: int
    dropout_rate: float
    batch_size: int
    epochs: int
    learning_rate: float
    random_seed: int


def _load_params() -> PretrainParams:
    """Load the pretrain stage parameters from params.yaml."""
    with open(PARAMS_PATH) as f:
        all_params: dict[str, PretrainParams] = yaml.safe_load(f)
    stage = "pretrain"
    if stage not in all_params:
        raise ValueError(f"Missing {stage!r} section in {PARAMS_PATH}")
    return all_params[stage]


def _load_tokenizer() -> BPETokenizer:
    """Load the trained BPE tokenizer from artifacts."""
    return BPETokenizer.load(TOKENIZER_DIR / "tokenizer.json")


def _load_data() -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Load tokenized train and validation arrays."""
    train_ids: npt.NDArray[np.intp] = np.load(TOKENIZER_DIR / "train_ids.npy")
    val_ids: npt.NDArray[np.intp] = np.load(TOKENIZER_DIR / "val_ids.npy")
    return train_ids, val_ids


def build_model(
    vocab_size: int,
    params: PretrainParams,
    rng: np.random.Generator,
) -> DecoderOnlyModel:
    """Build a decoder-only model from params."""
    return DecoderOnlyModel(
        vocab_size=vocab_size,
        d_model=params["d_model"],
        n_heads=params["n_heads"],
        n_layers=params["n_layers"],
        d_ff=params["d_ff"],
        max_len=params["max_len"],
        rng=rng,
        dropout_rate=params["dropout_rate"],
    )


def _save_model(model: DecoderOnlyModel) -> None:
    """Serialize the pretrained model to artifacts/pretrain/."""
    PRETRAIN_DIR.mkdir(parents=True, exist_ok=True)
    model_path = PRETRAIN_DIR / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Pretrained model saved to %s", model_path)


def _save_metrics(
    train_losses: list[float],
    val_loss: float,
    params: PretrainParams,
) -> None:
    """Write pretrain metrics to artifacts/pretrain/pretrain-metrics.json."""
    metrics = {
        "final_train_loss": round(train_losses[-1], 6) if train_losses else 0.0,
        "final_val_loss": round(val_loss, 6),
        "n_epochs": params["epochs"],
        "epoch_losses": [round(loss, 6) for loss in train_losses],
    }
    metrics_path = PRETRAIN_DIR / "pretrain-metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Pretrain metrics written to %s", metrics_path)


def main() -> None:
    """Orchestrate the pretrain stage."""
    params = _load_params()
    rng = np.random.default_rng(params["random_seed"])

    tokenizer = _load_tokenizer()
    train_ids, val_ids = _load_data()

    model = build_model(tokenizer.vocab_size, params, rng)

    trainer = LMTrainer(
        model=model,
        learning_rate=params["learning_rate"],
        batch_size=params["batch_size"],
        pad_token_id=tokenizer.pad_token_id,
    )

    logger.info(
        "Pretraining: %d epochs, batch_size=%d, lr=%s, vocab=%d",
        params["epochs"],
        params["batch_size"],
        params["learning_rate"],
        tokenizer.vocab_size,
    )

    train_losses: list[float] = []
    for epoch in range(params["epochs"]):
        loss = trainer.train_epoch(train_ids, rng)
        train_losses.append(loss)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            val_loss = trainer.eval_loss(val_ids)
            logger.info(
                "Epoch %d/%d: train_loss=%.4f, val_loss=%.4f",
                epoch + 1,
                params["epochs"],
                loss,
                val_loss,
            )

    val_loss = trainer.eval_loss(val_ids)
    logger.info("Final validation loss: %.4f", val_loss)

    _save_model(model)
    _save_metrics(train_losses, val_loss, params)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
