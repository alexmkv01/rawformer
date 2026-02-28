"""Pretrain stage: train a decoder-only model with causal language modelling.

Loads tokenized data and a trained tokenizer, builds a tiny GPT-style
model from params.yaml, and trains with next-token prediction.
"""

from dataclasses import dataclass
from logging import getLogger
from pathlib import Path

import numpy as np
import numpy.typing as npt

from rawformer.tokenizers.bpe import BPETokenizer
from rawformer.training.clm import DecoderOnlyModel
from rawformer.training.lm_trainer import LMTrainer
from rawformer_train._metrics import PretrainMetrics
from rawformer_train._params import PretrainParams, load_params
from rawformer_train._utils import load_tokenizer, save_json_metrics, save_model

logger = getLogger(__name__)


@dataclass
class PretrainResult:
    """Accumulated metrics from the pretraining loop."""

    train_losses: list[float]
    val_loss: float


def _load_data(
    tokenizer_dir: Path,
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Load tokenized train and validation arrays.

    Args:
        tokenizer_dir: Directory containing ``train_ids.npy`` and ``val_ids.npy``.

    Returns:
        Tuple of (train_ids, val_ids).
    """
    train_ids: npt.NDArray[np.intp] = np.load(tokenizer_dir / "train_ids.npy")
    val_ids: npt.NDArray[np.intp] = np.load(tokenizer_dir / "val_ids.npy")
    return train_ids, val_ids


def build_model(
    vocab_size: int,
    params: PretrainParams,
    rng: np.random.Generator,
) -> DecoderOnlyModel:
    """Build a decoder-only model from params.

    Args:
        vocab_size: Size of the token vocabulary.
        params: Pretrain stage parameters.
        rng: Random number generator for weight initialisation.

    Returns:
        An initialised decoder-only model.
    """
    return DecoderOnlyModel(
        vocab_size=vocab_size,
        d_model=params.d_model,
        n_heads=params.n_heads,
        n_layers=params.n_layers,
        d_ff=params.d_ff,
        max_len=params.max_len,
        rng=rng,
        dropout_rate=params.dropout_rate,
    )


def _run_training(
    trainer: LMTrainer,
    train_ids: npt.NDArray[np.intp],
    val_ids: npt.NDArray[np.intp],
    epochs: int,
    rng: np.random.Generator,
) -> PretrainResult:
    """Run the pretraining loop with periodic validation logging.

    Args:
        trainer: The language model trainer.
        train_ids: Training token ID array.
        val_ids: Validation token ID array.
        epochs: Number of training epochs.
        rng: Random number generator for batching.

    Returns:
        Accumulated per-epoch train losses and final validation loss.
    """
    train_losses: list[float] = []
    for epoch in range(epochs):
        loss = trainer.train_epoch(train_ids, rng)
        train_losses.append(loss)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            val_loss = trainer.eval_loss(val_ids)
            logger.info(
                "Epoch %d/%d: train_loss=%.4f, val_loss=%.4f",
                epoch + 1,
                epochs,
                loss,
                val_loss,
            )

    val_loss = trainer.eval_loss(val_ids)
    logger.info("Final validation loss: %.4f", val_loss)
    return PretrainResult(train_losses=train_losses, val_loss=val_loss)


def _generate_sample(model: DecoderOnlyModel, tokenizer: BPETokenizer) -> str:
    """Generate a sample text to demonstrate what the model has learned."""
    prompt_text = "Once upon a time"
    prompt_ids = np.array(tokenizer.encode(prompt_text, add_special_tokens=True), dtype=np.intp)
    generated_ids = model.generate(
        prompt_ids,
        max_tokens=30,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(generated_ids.tolist())


def _compile_metrics(
    result: PretrainResult,
    params: PretrainParams,
    sample_generation: str,
) -> PretrainMetrics:
    """Compile pretrain summary metrics.

    Args:
        result: Accumulated training metrics.
        params: Pretrain stage parameters.
        sample_generation: Sample text from the trained model.

    Returns:
        Metrics dictionary for JSON serialisation.
    """
    return PretrainMetrics(
        final_train_loss=round(result.train_losses[-1], 6) if result.train_losses else 0.0,
        final_val_loss=round(result.val_loss, 6),
        n_epochs=params.epochs,
        epoch_losses=[round(loss, 6) for loss in result.train_losses],
        sample_generation=sample_generation,
    )


def pretrain(
    tokenizer_dir: Path,
    pretrain_dir: Path,
    metrics_path: Path,
    params_path: Path,
) -> None:
    """Orchestrate the pretrain stage.

    Args:
        tokenizer_dir: Directory with tokenizer and tokenized arrays.
        pretrain_dir: Output directory for the pretrained model.
        metrics_path: Path to write pretrain metrics JSON.
        params_path: Path to the params.yaml configuration file.
    """
    # Load params
    params = load_params(params_path).pretrain
    rng = np.random.default_rng(params.random_seed)

    # Load
    tokenizer = load_tokenizer(tokenizer_dir)
    train_ids, val_ids = _load_data(tokenizer_dir)

    # Build model
    model = build_model(tokenizer.vocab_size, params, rng)
    trainer = LMTrainer(
        model=model,
        learning_rate=params.learning_rate,
        batch_size=params.batch_size,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Train
    result = _run_training(trainer, train_ids, val_ids, params.epochs, rng)
    sample_text = _generate_sample(model, tokenizer)

    # Save
    save_model(model, pretrain_dir, "model.pkl")

    # Metrics
    save_json_metrics(
        _compile_metrics(result, params, sample_text),
        metrics_path,
    )
