"""Custom exceptions for the rawformer_train pipeline."""


class RawformerTrainError(Exception):
    """Base exception for all rawformer_train errors."""


class ParamValidationError(RawformerTrainError):
    """Raised when params.yaml cannot be loaded or fails validation."""


class ModelLoadError(RawformerTrainError):
    """Raised when a pickled model cannot be loaded or has the wrong type."""
