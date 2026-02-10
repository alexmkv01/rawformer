"""Byte Pair Encoding tokenizer (Sennrich et al., 2016).

Implements the BPE algorithm as used in GPT (Radford et al., 2018):
1. Start with a character-level vocabulary plus special tokens.
2. Iteratively merge the most frequent adjacent pair of tokens.
3. Encode text by greedily applying learned merges.
"""

import json
from pathlib import Path

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

_SPECIAL_TOKENS: list[str] = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
_SPECIAL_TOKENS_SET: frozenset[str] = frozenset(_SPECIAL_TOKENS)


def _get_word_freqs(corpus: list[str]) -> dict[tuple[str, ...], int]:
    """Count character-tuple frequencies across the corpus.

    Each word becomes a tuple of characters plus an end-of-word marker,
    e.g. ``"hello"`` -> ``("h", "e", "l", "l", "o", "</w>")``.
    """
    word_freqs: dict[tuple[str, ...], int] = {}
    for text in corpus:
        words = text.strip().split()
        for word in words:
            key = (*word, "</w>")
            word_freqs[key] = word_freqs.get(key, 0) + 1
    return word_freqs


def _get_pair_freqs(
    word_freqs: dict[tuple[str, ...], int],
) -> dict[tuple[str, str], int]:
    """Count frequencies of all adjacent symbol pairs across the vocabulary."""
    pair_freqs: dict[tuple[str, str], int] = {}
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
    return pair_freqs


def _merge_pair(
    word_freqs: dict[tuple[str, ...], int],
    pair: tuple[str, str],
) -> dict[tuple[str, ...], int]:
    """Merge all occurrences of *pair* in every word, return new word_freqs."""
    new_word_freqs: dict[tuple[str, ...], int] = {}
    left, right = pair
    merged = left + right
    for word, freq in word_freqs.items():
        new_word: list[str] = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == left and word[i + 1] == right:
                new_word.append(merged)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word_freqs[tuple(new_word)] = freq
    return new_word_freqs


class BPETokenizer:
    """Byte Pair Encoding tokenizer."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.inverse_vocab: dict[int, str] = {}
        self.merges: list[tuple[str, str]] = []

    def train(self, corpus: list[str], vocab_size: int) -> None:
        """Learn BPE merges from a text corpus."""
        word_freqs = _get_word_freqs(corpus)

        chars: set[str] = set()
        for word in word_freqs:
            for ch in word:
                chars.add(ch)

        self.vocab = {tok: i for i, tok in enumerate(_SPECIAL_TOKENS)}
        idx = len(_SPECIAL_TOKENS)
        for ch in sorted(chars):
            if ch not in self.vocab:
                self.vocab[ch] = idx
                idx += 1

        self.merges = []

        while len(self.vocab) < vocab_size:
            pair_freqs = _get_pair_freqs(word_freqs)
            if not pair_freqs:
                break
            best_pair = max(pair_freqs, key=lambda p: pair_freqs[p])
            if pair_freqs[best_pair] < 1:
                break

            word_freqs = _merge_pair(word_freqs, best_pair)
            merged_token = best_pair[0] + best_pair[1]

            if merged_token not in self.vocab:
                self.vocab[merged_token] = len(self.vocab)

            self.merges.append(best_pair)

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def _check_trained(self) -> None:
        if not self.vocab:
            raise RuntimeError("BPETokenizer has not been trained or loaded")

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode text into token IDs, optionally wrapping with <bos>/<eos>."""
        self._check_trained()
        ids: list[int] = []
        if add_special_tokens:
            ids.append(self.vocab[BOS_TOKEN])

        words = text.strip().split()
        for word in words:
            symbols: list[str] = [*word, "</w>"]
            for left, right in self.merges:
                merged = left + right
                i = 0
                new_symbols: list[str] = []
                while i < len(symbols):
                    if i < len(symbols) - 1 and symbols[i] == left and symbols[i + 1] == right:
                        new_symbols.append(merged)
                        i += 2
                    else:
                        new_symbols.append(symbols[i])
                        i += 1
                symbols = new_symbols

            unk_id = self.vocab[UNK_TOKEN]
            for sym in symbols:
                ids.append(self.vocab.get(sym, unk_id))

        if add_special_tokens:
            ids.append(self.vocab[EOS_TOKEN])

        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back into text, stripping special tokens."""
        self._check_trained()
        tokens: list[str] = []
        for token_id in ids:
            token = self.inverse_vocab.get(token_id, UNK_TOKEN)
            if token in _SPECIAL_TOKENS_SET:
                continue
            tokens.append(token)

        raw = "".join(tokens)
        text = raw.replace("</w>", " ")
        return text.strip()

    def save(self, path: Path) -> None:
        """Save vocabulary and merges to a JSON file."""
        data = {
            "vocab": self.vocab,
            "merges": [list(m) for m in self.merges],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        """Load a trained tokenizer from a JSON file."""
        with open(path) as f:
            data: dict[str, list[list[str]] | dict[str, int]] = json.load(f)
        tokenizer = cls()
        vocab_raw = data.get("vocab")
        if not isinstance(vocab_raw, dict):
            raise ValueError(f"Expected 'vocab' dict in {path}, got {type(vocab_raw)}")
        tokenizer.vocab = dict(vocab_raw)
        tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
        merges_raw = data.get("merges")
        if not isinstance(merges_raw, list):
            raise ValueError(f"Expected 'merges' list in {path}, got {type(merges_raw)}")
        for i, m in enumerate(merges_raw):
            if not isinstance(m, list) or len(m) != 2:
                raise ValueError(f"Merge entry {i} in {path} must be a 2-element list, got {m!r}")
        tokenizer.merges = [(str(m[0]), str(m[1])) for m in merges_raw]
        return tokenizer

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        return self.vocab[PAD_TOKEN]

    @property
    def unk_token_id(self) -> int:
        return self.vocab[UNK_TOKEN]

    @property
    def bos_token_id(self) -> int:
        return self.vocab[BOS_TOKEN]

    @property
    def eos_token_id(self) -> int:
        return self.vocab[EOS_TOKEN]
