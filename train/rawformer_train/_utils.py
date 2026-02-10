"""Shared utilities for the rawformer_train pipeline stages."""


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
