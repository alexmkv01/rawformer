"""Shared utilities for the rawformer_train pipeline stages."""

import yaml

from rawformer_train._paths import PARAMS_PATH


def load_stage_params[T](stage: str, param_type: type[T]) -> T:
    """Load a named stage's parameters from params.yaml.

    Args:
        stage: Section name in params.yaml (e.g. ``"sft"``, ``"align"``).
        param_type: The TypedDict class for static-analysis purposes
            (unused at runtime, but makes call-sites self-documenting).

    Returns:
        The parameter dict for the requested stage.
    """
    _ = param_type  # used only for readability at call sites
    with open(PARAMS_PATH) as f:
        all_params: dict[str, T] = yaml.safe_load(f)
    if stage not in all_params:
        raise ValueError(f"Missing {stage!r} section in {PARAMS_PATH}")
    return all_params[stage]


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
