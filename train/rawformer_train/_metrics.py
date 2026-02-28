"""Typed metrics schemas for each pipeline stage.

Each TypedDict exactly mirrors the JSON written to the metrics file for
that stage.  ``PipelineMetrics`` is the union accepted by
``save_json_metrics`` so the type-checker enforces that only known metrics
shapes are serialised.
"""

from typing import TypedDict


class TokenizeMetrics(TypedDict):
    """Metrics produced by the tokenize stage."""

    vocab_size: int
    n_train_sequences: int
    n_val_sequences: int
    seq_len: int


class PretrainMetrics(TypedDict):
    """Metrics produced by the pretrain stage."""

    final_train_loss: float
    final_val_loss: float
    n_epochs: int
    epoch_losses: list[float]
    sample_generation: str


class SFTMetrics(TypedDict):
    """Metrics produced by the SFT stage."""

    final_train_loss: float
    n_epochs: int
    epoch_losses: list[float]
    sample_generation: str


class AlignMetrics(TypedDict):
    """Metrics produced by the align stage."""

    final_loss: float
    final_reward_margin: float
    final_accuracy: float
    n_epochs: int
    beta: float
    epoch_losses: list[float]
    epoch_reward_margins: list[float]
    epoch_accuracies: list[float]
    sample_generation: str


type PipelineMetrics = TokenizeMetrics | PretrainMetrics | SFTMetrics | AlignMetrics
