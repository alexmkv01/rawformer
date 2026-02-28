"""Smoke tests for the transformer training pipeline stages.

These tests validate the pipeline logic using synthetic data,
without requiring DVC or the actual dataset files.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from rawformer.tokenizers.bpe import BPETokenizer
from rawformer.training.clm import DecoderOnlyModel
from rawformer.training.dpo import DPOTrainer
from rawformer.training.lm_trainer import LMTrainer
from rawformer_train._params import PretrainParams, load_params
from rawformer_train.align import (
    PreferencePair,
    format_preference_pairs,
    load_preference_data,
)
from rawformer_train.exceptions import ParamValidationError
from rawformer_train.prepare import load_corpus, split_data, tokenize_corpus
from rawformer_train.pretrain import build_model
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
        lines = load_corpus(path)
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
        params = PretrainParams(
            d_model=16,
            n_heads=2,
            n_layers=1,
            d_ff=32,
            max_len=32,
            dropout_rate=0.0,
            batch_size=4,
            epochs=1,
            learning_rate=0.001,
            random_seed=42,
        )
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

        examples = load_sft_data(path)
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


# =====================================================================
# Align Stage
# =====================================================================


class TestAlign:
    def test_load_preference_data(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(
                json.dumps(
                    {
                        "prompt": "Say hi",
                        "chosen": "Hello there!",
                        "rejected": "Hi.",
                    }
                )
                + "\n"
            )
            f.write(
                json.dumps(
                    {
                        "prompt": "Count",
                        "chosen": "One two three four five",
                        "rejected": "Numbers.",
                    }
                )
                + "\n"
            )
            path = Path(f.name)

        pairs = load_preference_data(path)
        assert len(pairs) == 2
        assert pairs[0]["prompt"] == "Say hi"
        assert pairs[0]["chosen"] == "Hello there!"
        assert pairs[0]["rejected"] == "Hi."
        path.unlink()

    def test_format_preference_pairs(self, trained_tokenizer: BPETokenizer) -> None:
        pairs: list[PreferencePair] = [
            {"prompt": "Say hi", "chosen": "Hello there!", "rejected": "Hi."},
            {"prompt": "Count", "chosen": "One two three", "rejected": "Numbers."},
        ]
        chosen_ids, rejected_ids = format_preference_pairs(pairs, trained_tokenizer, max_seq_len=32)
        assert chosen_ids.shape == (2, 32)
        assert rejected_ids.shape == (2, 32)
        assert np.all(chosen_ids[:, 0] == trained_tokenizer.bos_token_id)
        assert np.all(rejected_ids[:, 0] == trained_tokenizer.bos_token_id)

    def test_dpo_loss_decreases(self, trained_tokenizer: BPETokenizer) -> None:
        """DPO training on toy preference data should decrease the loss."""
        rng = np.random.default_rng(42)
        pairs: list[PreferencePair] = [
            {"prompt": "Say hi", "chosen": "Hello there friend", "rejected": "Hi."},
            {"prompt": "Count", "chosen": "One two three four five", "rejected": "Numbers."},
            {"prompt": "Color", "chosen": "The color is bright red", "rejected": "Red."},
        ]
        chosen_ids, rejected_ids = format_preference_pairs(pairs, trained_tokenizer, max_seq_len=16)

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
        trainer = DPOTrainer.from_sft_model(
            sft_model=model,
            learning_rate=0.01,
            beta=0.1,
            pad_token_id=trained_tokenizer.pad_token_id,
        )

        first = trainer.train_epoch(chosen_ids, rejected_ids, batch_size=len(pairs), rng=rng)
        last = first
        for _ in range(19):
            last = trainer.train_epoch(chosen_ids, rejected_ids, batch_size=len(pairs), rng=rng)

        assert last.loss < first.loss


# =====================================================================
# Params validation
# =====================================================================


_VALID_PARAMS_YAML = """\
tokenize:
  vocab_size: 100
  max_seq_len: 32
  random_seed: 0
  val_split: 0.1
pretrain:
  d_model: 16
  n_heads: 2
  n_layers: 1
  d_ff: 32
  max_len: 32
  dropout_rate: 0.0
  batch_size: 4
  epochs: 1
  learning_rate: 0.001
  random_seed: 42
sft:
  batch_size: 4
  epochs: 1
  learning_rate: 0.001
  max_seq_len: 32
  random_seed: 42
align:
  batch_size: 4
  beta: 0.1
  epochs: 1
  learning_rate: 0.001
  max_seq_len: 32
  random_seed: 42
"""


class TestParamsValidation:
    def test_valid_params_loads(self, tmp_path: Path) -> None:
        """A well-formed params.yaml must load without error."""
        params_file = tmp_path / "params.yaml"
        params_file.write_text(_VALID_PARAMS_YAML)
        params = load_params(params_file)
        assert params.pretrain.d_model == 16

    def test_field_ge_rejects_zero(self, tmp_path: Path) -> None:
        """Field(ge=1) constraints must cause load_params to raise ParamValidationError."""
        bad = _VALID_PARAMS_YAML.replace("  d_model: 16", "  d_model: 0")
        params_file = tmp_path / "params.yaml"
        params_file.write_text(bad)
        with pytest.raises(ParamValidationError):
            load_params(params_file)

    def test_missing_section_raises(self, tmp_path: Path) -> None:
        """A params.yaml missing a required section must raise ParamValidationError."""
        params_file = tmp_path / "params.yaml"
        params_file.write_text(
            "tokenize:\n  vocab_size: 100\n  max_seq_len: 32\n  random_seed: 0\n  val_split: 0.1\n"
        )
        with pytest.raises(ParamValidationError):
            load_params(params_file)

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        """An empty params.yaml must raise ParamValidationError."""
        params_file = tmp_path / "params.yaml"
        params_file.write_text("")
        with pytest.raises(ParamValidationError):
            load_params(params_file)


# =====================================================================
# CLI dispatch smoke test
# =====================================================================


class TestCLIDispatch:
    def test_bad_command_exits_nonzero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """main() with an unknown subcommand must exit with a non-zero status."""
        monkeypatch.setattr(sys, "argv", ["rawformer-train", "not-a-command"])
        from rawformer_train.__main__ import main

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code != 0

    def test_no_args_exits_nonzero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """main() with no subcommand must exit with a non-zero status."""
        monkeypatch.setattr(sys, "argv", ["rawformer-train"])
        from rawformer_train.__main__ import main

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code != 0
