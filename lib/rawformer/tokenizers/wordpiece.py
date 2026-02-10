"""WordPiece tokenizer (Schuster & Nakajima, 2012; Wu et al., 2016).

Implements the WordPiece algorithm as used in BERT (Devlin et al., 2018):
1. Start with a character-level vocabulary plus special tokens.
2. Iteratively merge the pair that maximises the likelihood of the
   training corpus (approximated by: freq(ab) / (freq(a) * freq(b))).
3. Encode text by greedily matching the longest subword from left to right,
   prefixing continuation subwords with '##'.
"""

import json
from pathlib import Path

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
MASK_TOKEN = "[MASK]"

_SPECIAL_TOKENS: list[str] = [PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN, MASK_TOKEN]
_SPECIAL_TOKENS_SET: frozenset[str] = frozenset(_SPECIAL_TOKENS)

_CONTINUATION_PREFIX = "##"


def _get_word_freqs(corpus: list[str]) -> dict[str, int]:
    """Count whitespace-separated word frequencies across the corpus."""
    word_freqs: dict[str, int] = {}
    for text in corpus:
        for word in text.strip().split():
            word_freqs[word] = word_freqs.get(word, 0) + 1
    return word_freqs


def _split_word(word: str) -> list[str]:
    """Split a word into initial character + '##'-prefixed continuations.

    Example:
        >>> _split_word("hello")
        ["h", "##e", "##l", "##l", "##o"]
    """
    if not word:
        return []
    return [word[0]] + [_CONTINUATION_PREFIX + ch for ch in word[1:]]


def _compute_pair_scores(
    splits: dict[str, list[str]],
    word_freqs: dict[str, int],
) -> dict[tuple[str, str], float]:
    """Compute the WordPiece pair score: freq(ab) / (freq(a) * freq(b)).

    Recomputes token frequencies from scratch each call — acceptable for
    the small vocabularies used here, but O(V^2 * iterations) overall.
    """
    token_freqs: dict[str, int] = {}
    pair_freqs: dict[tuple[str, str], int] = {}

    for word, freq in word_freqs.items():
        split = splits[word]
        for i, token in enumerate(split):
            token_freqs[token] = token_freqs.get(token, 0) + freq
            if i < len(split) - 1:
                pair = (split[i], split[i + 1])
                pair_freqs[pair] = pair_freqs.get(pair, 0) + freq

    scores: dict[tuple[str, str], float] = {}
    for pair, freq in pair_freqs.items():
        denom = token_freqs.get(pair[0], 1) * token_freqs.get(pair[1], 1)
        scores[pair] = freq / denom

    return scores


def _merge_pair_in_splits(
    splits: dict[str, list[str]],
    pair: tuple[str, str],
) -> dict[str, list[str]]:
    """Merge all occurrences of *pair* in every word's split list."""
    left, right = pair
    merged = left + right.removeprefix(_CONTINUATION_PREFIX)

    new_splits: dict[str, list[str]] = {}
    for word, split in splits.items():
        new_split: list[str] = []
        i = 0
        while i < len(split):
            if i < len(split) - 1 and split[i] == left and split[i + 1] == right:
                new_split.append(merged)
                i += 2
            else:
                new_split.append(split[i])
                i += 1
        new_splits[word] = new_split
    return new_splits


class WordPieceTokenizer:
    """WordPiece tokenizer as used in BERT."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.inverse_vocab: dict[int, str] = {}

    def train(self, corpus: list[str], vocab_size: int) -> None:
        """Learn WordPiece vocabulary from a text corpus."""
        word_freqs = _get_word_freqs(corpus)

        chars: set[str] = set()
        for word in word_freqs:
            chars.add(word[0])
            for ch in word[1:]:
                chars.add(_CONTINUATION_PREFIX + ch)

        self.vocab = {tok: i for i, tok in enumerate(_SPECIAL_TOKENS)}
        idx = len(_SPECIAL_TOKENS)
        for ch in sorted(chars):
            if ch not in self.vocab:
                self.vocab[ch] = idx
                idx += 1

        splits = {word: _split_word(word) for word in word_freqs}

        while len(self.vocab) < vocab_size:
            scores = _compute_pair_scores(splits, word_freqs)
            if not scores:
                break
            best_pair = max(scores, key=lambda p: scores[p])
            if scores[best_pair] <= 0:
                break

            splits = _merge_pair_in_splits(splits, best_pair)

            left, right = best_pair
            merged_token = left + right.removeprefix(_CONTINUATION_PREFIX)

            if merged_token not in self.vocab:
                self.vocab[merged_token] = len(self.vocab)

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def _check_trained(self) -> None:
        if not self.vocab:
            raise RuntimeError("WordPieceTokenizer has not been trained or loaded")

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode text into token IDs using greedy longest-match segmentation."""
        self._check_trained()
        ids: list[int] = []
        if add_special_tokens:
            ids.append(self.vocab[CLS_TOKEN])

        words = text.strip().split()
        unk_id = self.vocab[UNK_TOKEN]

        for word in words:
            tokens = self._tokenize_word(word)
            for token in tokens:
                ids.append(self.vocab.get(token, unk_id))

        if add_special_tokens:
            ids.append(self.vocab[SEP_TOKEN])

        return ids

    def _tokenize_word(self, word: str) -> list[str]:
        """Segment a single word into WordPiece tokens via greedy longest-match.

        Unlike BERT's original which emits a single ``<unk>`` for a fully
        OOV word, this emits one ``<unk>`` per unrecognised character.
        """
        tokens: list[str] = []
        start = 0
        while start < len(word):
            end = len(word)
            found = False
            while start < end:
                substr = word[start:end]
                if start > 0:
                    substr = _CONTINUATION_PREFIX + substr
                if substr in self.vocab:
                    tokens.append(substr)
                    found = True
                    break
                end -= 1
            if not found:
                tokens.append(UNK_TOKEN)
                start += 1
            else:
                start = end
        return tokens

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back into text, stripping special tokens."""
        self._check_trained()
        tokens: list[str] = []
        for token_id in ids:
            token = self.inverse_vocab.get(token_id, UNK_TOKEN)
            if token in _SPECIAL_TOKENS_SET:
                continue
            tokens.append(token)

        words: list[str] = []
        current_word: list[str] = []
        for token in tokens:
            if token.startswith(_CONTINUATION_PREFIX):
                current_word.append(token[len(_CONTINUATION_PREFIX) :])
            else:
                if current_word:
                    words.append("".join(current_word))
                current_word = [token]
        if current_word:
            words.append("".join(current_word))

        return " ".join(words)

    def save(self, path: Path) -> None:
        """Save vocabulary to a JSON file."""
        data = {"vocab": self.vocab}
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "WordPieceTokenizer":
        """Load a trained tokenizer from a JSON file."""
        with open(path) as f:
            data: dict[str, dict[str, int]] = json.load(f)
        tokenizer = cls()
        vocab_raw = data.get("vocab")
        if not isinstance(vocab_raw, dict):
            raise ValueError(f"Expected 'vocab' dict in {path}, got {type(vocab_raw)}")
        tokenizer.vocab = dict(vocab_raw)
        tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
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
    def cls_token_id(self) -> int:
        return self.vocab[CLS_TOKEN]

    @property
    def sep_token_id(self) -> int:
        return self.vocab[SEP_TOKEN]

    @property
    def mask_token_id(self) -> int:
        return self.vocab[MASK_TOKEN]
