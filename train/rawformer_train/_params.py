"""Pipeline parameters loaded from params.yaml with Pydantic validation."""

from logging import getLogger
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, ValidationError

from rawformer_train.exceptions import ParamValidationError

logger = getLogger(__name__)


class TokenizeParams(BaseModel):
    """Parameters for the tokenize stage."""

    vocab_size: int = Field(ge=1)
    max_seq_len: int = Field(ge=1)
    random_seed: int
    val_split: float = Field(gt=0.0, lt=1.0)


class PretrainParams(BaseModel):
    """Parameters for the pretrain stage."""

    d_model: int = Field(ge=1)
    n_heads: int = Field(ge=1)
    n_layers: int = Field(ge=1)
    d_ff: int = Field(ge=1)
    max_len: int = Field(ge=1)
    dropout_rate: float = Field(ge=0.0, le=1.0)
    batch_size: int = Field(ge=1)
    epochs: int = Field(ge=1)
    learning_rate: float = Field(gt=0.0)
    random_seed: int


class SFTParams(BaseModel):
    """Parameters for the SFT stage."""

    batch_size: int = Field(ge=1)
    epochs: int = Field(ge=1)
    learning_rate: float = Field(gt=0.0)
    max_seq_len: int = Field(ge=1)
    random_seed: int


class AlignParams(BaseModel):
    """Parameters for the align stage."""

    batch_size: int = Field(ge=1)
    beta: float = Field(gt=0.0)
    epochs: int = Field(ge=1)
    learning_rate: float = Field(gt=0.0)
    max_seq_len: int = Field(ge=1)
    random_seed: int


class PipelineParams(BaseModel):
    """Top-level model that mirrors the structure of params.yaml."""

    tokenize: TokenizeParams
    pretrain: PretrainParams
    sft: SFTParams
    align: AlignParams


def load_params(path: Path) -> PipelineParams:
    """Load and validate pipeline parameters from a YAML file.

    Args:
        path: Path to the params.yaml configuration file.

    Returns:
        Validated pipeline parameters.

    Raises:
        ParamValidationError: If the YAML content fails Pydantic validation.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)

    try:
        params = PipelineParams.model_validate(raw)
    except ValidationError as exc:
        raise ParamValidationError(f"Invalid params in {path}:\n{exc}") from exc

    logger.info("Loaded and validated params from %s", path)
    return params
