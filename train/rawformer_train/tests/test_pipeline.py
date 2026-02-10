"""Smoke tests for the transformer training pipeline stages.

These tests validate the pipeline logic using synthetic data,
without requiring DVC or the actual dataset files.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from rawformer.tokenizers.bpe import BPETokenizer
from rawformer.training.clm import DecoderOnlyModel
from rawformer.training.lm_trainer import LMTrainer
from rawformer_train.prepare import load_corpus, split_data, tokenize_corpus
from rawformer_train.pretrain import PretrainParams, build_model
from rawformer_train.sft import SFTExample, format_sft_examples, load_sft_data

# ----- Shared fixtures -----

_CORPUS = [
    "the cat sat on the mat",
    "the dog ran in the park",
    "a bird flew over the tree",
    "the fish swam in the pond",
    "a rabbit hopped through the field",
]


@pytest.fixture
def trained_tokenizer() -> BPETokenizer:
    """Return a BPE tokenizer trained on the test corpus."""
    tok = BPETokenizer()
    tok.train(_CORPUS, vocab_size=100)
    return tok


# =====================================================================
# Tokenize Stage
# =====================================================================


class TestLoadCorpus:
    def test_loads_from_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("line one\nline two\n\nline three\n")
            path = Path(f.name)
        lines = load_corpus(str(path))
        assert len(lines) == 3
        assert lines[0] == "line one"
        path.unlink()


class TestTokenizeCorpus:
    def test_output_shape(self, trained_tokenizer: BPETokenizer) -> None:
        token_ids = tokenize_corpus(_CORPUS, trained_tokenizer, max_seq_len=32)
        assert token_ids.shape == (len(_CORPUS), 32)

    def test_padding(self, trained_tokenizer: BPETokenizer) -> None:
        token_ids = tokenize_corpus(_CORPUS, trained_tokenizer, max_seq_len=64)
        # Last column should mostly be pad tokens.
        pad_id = trained_tokenizer.pad_token_id
        assert np.all(token_ids[:, -1] == pad_id)

    def test_truncation(self, trained_tokenizer: BPETokenizer) -> None:
        token_ids = tokenize_corpus(_CORPUS, trained_tokenizer, max_seq_len=5)
        assert token_ids.shape[1] == 5

    def test_bos_eos_present(self, trained_tokenizer: BPETokenizer) -> None:
        token_ids = tokenize_corpus(_CORPUS, trained_tokenizer, max_seq_len=64)
        assert np.all(token_ids[:, 0] == trained_tokenizer.bos_token_id)
        # EOS should be present somewhere in each sequence.
        for row in token_ids:
            assert trained_tokenizer.eos_token_id in row


class TestSplitData:
    def test_split_sizes(self, trained_tokenizer: BPETokenizer) -> None:
        token_ids = tokenize_corpus(_CORPUS, trained_tokenizer, max_seq_len=32)
        train_ids, val_ids = split_data(token_ids, val_split=0.2, random_seed=42)
        assert train_ids.shape[0] + val_ids.shape[0] == len(_CORPUS)
        assert val_ids.shape[0] >= 1

    def test_split_is_deterministic(self, trained_tokenizer: BPETokenizer) -> None:
        token_ids = tokenize_corpus(_CORPUS, trained_tokenizer, max_seq_len=32)
        t1, v1 = split_data(token_ids, val_split=0.2, random_seed=42)
        t2, v2 = split_data(token_ids, val_split=0.2, random_seed=42)
        np.testing.assert_array_equal(t1, t2)
        np.testing.assert_array_equal(v1, v2)


# =====================================================================
# Pretrain Stage
# =====================================================================


class TestBuildModel:
    def test_builds_model_with_correct_vocab(self) -> None:
        params: PretrainParams = {
            "d_model": 16,
            "n_heads": 2,
            "n_layers": 1,
            "d_ff": 32,
            "max_len": 32,
            "dropout_rate": 0.0,
            "batch_size": 4,
            "epochs": 1,
            "learning_rate": 0.001,
            "random_seed": 42,
        }
        rng = np.random.default_rng(42)
        model = build_model(vocab_size=50, params=params, rng=rng)
        assert model.vocab_size == 50

    def test_pretrain_loss_decreases(self, trained_tokenizer: BPETokenizer) -> None:
        """A tiny model should overfit on a small dataset."""
        rng = np.random.default_rng(42)
        token_ids = tokenize_corpus(_CORPUS, trained_tokenizer, max_seq_len=16)

        model = DecoderOnlyModel(
            vocab_size=trained_tokenizer.vocab_size,
            d_model=16,
            n_heads=2,
            n_layers=1,
            d_ff=32,
            max_len=16,
            rng=rng,
            dropout_rate=0.0,
        )
        trainer = LMTrainer(
            model=model,
            learning_rate=0.01,
            batch_size=len(_CORPUS),
            pad_token_id=trained_tokenizer.pad_token_id,
        )

        loss_before = trainer.eval_loss(token_ids)
        for _ in range(10):
            trainer.train_epoch(token_ids, rng)
        loss_after = trainer.eval_loss(token_ids)

        assert loss_after < loss_before


# =====================================================================
# SFT Stage
# =====================================================================


class TestSFT:
    def test_load_sft_data(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"instruction": "Say hi", "response": "Hello!"}) + "\n")
            f.write(json.dumps({"instruction": "Count", "response": "One two three"}) + "\n")
            path = Path(f.name)

        examples = load_sft_data(str(path))
        assert len(examples) == 2
        assert examples[0]["instruction"] == "Say hi"
        path.unlink()

    def test_format_sft_examples(self, trained_tokenizer: BPETokenizer) -> None:
        examples: list[SFTExample] = [
            {"instruction": "Say hi", "response": "Hello!"},
            {"instruction": "Count", "response": "One two three"},
        ]
        sft_ids = format_sft_examples(examples, trained_tokenizer, max_seq_len=32)
        assert sft_ids.shape == (2, 32)
        assert np.all(sft_ids[:, 0] == trained_tokenizer.bos_token_id)

    def test_sft_loss_decreases(self, trained_tokenizer: BPETokenizer) -> None:
        """Fine-tuning on toy instruction data should decrease the loss."""
        rng = np.random.default_rng(42)
        examples: list[SFTExample] = [
            {"instruction": "Say hi", "response": "Hello there"},
            {"instruction": "Count to three", "response": "One two three"},
            {"instruction": "Name a color", "response": "The color is red"},
        ]
        sft_ids = format_sft_examples(examples, trained_tokenizer, max_seq_len=16)

        model = DecoderOnlyModel(
            vocab_size=trained_tokenizer.vocab_size,
            d_model=16,
            n_heads=2,
            n_layers=1,
            d_ff=32,
            max_len=16,
            rng=rng,
            dropout_rate=0.0,
        )
        trainer = LMTrainer(
            model=model,
            learning_rate=0.01,
            batch_size=len(examples),
            pad_token_id=trained_tokenizer.pad_token_id,
        )

        loss_before = trainer.eval_loss(sft_ids)
        for _ in range(10):
            trainer.train_epoch(sft_ids, rng)
        loss_after = trainer.eval_loss(sft_ids)

        assert loss_after < loss_before
