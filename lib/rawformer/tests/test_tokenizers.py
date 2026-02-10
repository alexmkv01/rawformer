"""Tests for BPE tokenizer."""

import tempfile
from pathlib import Path

from rawformer.tokenizers.bpe import BPETokenizer

# ----- Shared test corpus -----

_CORPUS = [
    "the cat sat on the mat",
    "the cat ate the rat",
    "the rat sat on the mat",
    "the dog sat on the log",
    "the cat and the dog",
]


# =====================================================================
# BPE Tokenizer
# =====================================================================


class TestBPETrain:
    def test_vocab_size_respected(self) -> None:
        tok = BPETokenizer()
        tok.train(_CORPUS, vocab_size=50)
        assert tok.vocab_size <= 50

    def test_special_tokens_in_vocab(self) -> None:
        tok = BPETokenizer()
        tok.train(_CORPUS, vocab_size=50)
        assert "<pad>" in tok.vocab
        assert "<unk>" in tok.vocab
        assert "<bos>" in tok.vocab
        assert "<eos>" in tok.vocab

    def test_special_token_ids_are_first(self) -> None:
        tok = BPETokenizer()
        tok.train(_CORPUS, vocab_size=50)
        assert tok.vocab["<pad>"] == 0
        assert tok.vocab["<unk>"] == 1
        assert tok.vocab["<bos>"] == 2
        assert tok.vocab["<eos>"] == 3

    def test_merges_are_learned(self) -> None:
        tok = BPETokenizer()
        tok.train(_CORPUS, vocab_size=50)
        assert len(tok.merges) > 0

    def test_all_chars_in_vocab(self) -> None:
        tok = BPETokenizer()
        tok.train(_CORPUS, vocab_size=50)
        all_chars: set[str] = set()
        for text in _CORPUS:
            for ch in text:
                if ch != " ":
                    all_chars.add(ch)
        all_chars.add("</w>")
        for ch in all_chars:
            assert ch in tok.vocab, f"Character {ch!r} missing from vocab"


class TestBPEEncode:
    def test_encode_returns_int_list(self) -> None:
        tok = BPETokenizer()
        tok.train(_CORPUS, vocab_size=50)
        ids = tok.encode("the cat")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_encode_starts_with_bos(self) -> None:
        tok = BPETokenizer()
        tok.train(_CORPUS, vocab_size=50)
        ids = tok.encode("the cat")
        assert ids[0] == tok.bos_token_id

    def test_encode_ends_with_eos(self) -> None:
        tok = BPETokenizer()
        tok.train(_CORPUS, vocab_size=50)
        ids = tok.encode("the cat")
        assert ids[-1] == tok.eos_token_id

    def test_encode_no_special_tokens(self) -> None:
        tok = BPETokenizer()
        tok.train(_CORPUS, vocab_size=50)
        ids = tok.encode("the cat", add_special_tokens=False)
        assert ids[0] != tok.bos_token_id
        assert ids[-1] != tok.eos_token_id

    def test_encode_unknown_chars_get_unk(self) -> None:
        tok = BPETokenizer()
        tok.train(["hello world"], vocab_size=20)
        # Digit '9' was never in the training corpus.
        ids = tok.encode("9", add_special_tokens=False)
        assert tok.unk_token_id in ids


class TestBPEDecode:
    def test_roundtrip_known_text(self) -> None:
        tok = BPETokenizer()
        tok.train(_CORPUS, vocab_size=80)
        for text in _CORPUS:
            ids = tok.encode(text)
            decoded = tok.decode(ids)
            assert decoded == text

    def test_roundtrip_without_special_tokens(self) -> None:
        tok = BPETokenizer()
        tok.train(_CORPUS, vocab_size=80)
        text = "the cat sat"
        ids = tok.encode(text, add_special_tokens=False)
        decoded = tok.decode(ids)
        assert decoded == text


class TestBPEPersistence:
    def test_save_and_load_roundtrip(self) -> None:
        tok = BPETokenizer()
        tok.train(_CORPUS, vocab_size=50)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        tok.save(path)
        loaded = BPETokenizer.load(path)

        assert loaded.vocab == tok.vocab
        assert loaded.merges == tok.merges

        text = "the cat sat on the mat"
        assert tok.encode(text) == loaded.encode(text)

        path.unlink()
