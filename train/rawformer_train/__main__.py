"""CLI entry point for the rawformer training pipeline."""

import sys
from logging.config import dictConfig
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, CliApp, CliPositionalArg, SettingsConfigDict

from .align import align
from .prepare import tokenize
from .pretrain import pretrain
from .sft import sft

_LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
        },
    },
    "loggers": {
        "rawformer_train": {
            "handlers": ["console"],
            "level": "INFO",
        },
    },
    "formatters": {
        "default": {
            "format": " {asctime} {module:10} {levelname:8}  {message}",
            "style": "{",
            "validate": True,
        },
    },
}


class Tokenize(BaseSettings):
    """Train a BPE tokenizer and produce tokenized arrays."""

    model_config = SettingsConfigDict(cli_kebab_case=True)

    corpus_path: CliPositionalArg[Path] = Field(
        description="Path to the raw text corpus.",
    )
    tokenizer_dir: CliPositionalArg[Path] = Field(
        description="Output directory for tokenizer and tokenized arrays.",
    )
    metrics_path: CliPositionalArg[Path] = Field(
        description="Path to write tokenization metrics JSON.",
    )
    params_path: Path = Field(
        Path("params.yaml"),
        description="Path to the params.yaml configuration file.",
    )

    def cli_cmd(self) -> None:
        tokenize(self.corpus_path, self.tokenizer_dir, self.metrics_path, self.params_path)


class Pretrain(BaseSettings):
    """Train a decoder-only model with causal language modelling."""

    model_config = SettingsConfigDict(cli_kebab_case=True)

    tokenizer_dir: CliPositionalArg[Path] = Field(
        description="Directory with tokenizer and tokenized arrays.",
    )
    pretrain_dir: CliPositionalArg[Path] = Field(
        description="Output directory for the pretrained model.",
    )
    metrics_path: CliPositionalArg[Path] = Field(
        description="Path to write pretrain metrics JSON.",
    )
    params_path: Path = Field(
        Path("params.yaml"),
        description="Path to the params.yaml configuration file.",
    )

    def cli_cmd(self) -> None:
        pretrain(self.tokenizer_dir, self.pretrain_dir, self.metrics_path, self.params_path)


class SFT(BaseSettings):
    """Fine-tune the pretrained model on instruction-response pairs."""

    model_config = SettingsConfigDict(cli_kebab_case=True)

    tokenizer_dir: CliPositionalArg[Path] = Field(
        description="Directory containing the trained tokenizer.",
    )
    pretrain_dir: CliPositionalArg[Path] = Field(
        description="Directory containing the pretrained model.",
    )
    sft_data_path: CliPositionalArg[Path] = Field(
        description="Path to the SFT instruction-response JSONL file.",
    )
    sft_dir: CliPositionalArg[Path] = Field(
        description="Output directory for the fine-tuned model.",
    )
    metrics_path: CliPositionalArg[Path] = Field(
        description="Path to write SFT metrics JSON.",
    )
    params_path: Path = Field(
        Path("params.yaml"),
        description="Path to the params.yaml configuration file.",
    )

    def cli_cmd(self) -> None:
        sft(
            self.tokenizer_dir,
            self.pretrain_dir,
            self.sft_data_path,
            self.sft_dir,
            self.metrics_path,
            self.params_path,
        )


class Align(BaseSettings):
    """Run DPO preference alignment on the SFT model."""

    model_config = SettingsConfigDict(cli_kebab_case=True)

    tokenizer_dir: CliPositionalArg[Path] = Field(
        description="Directory containing the trained tokenizer.",
    )
    sft_dir: CliPositionalArg[Path] = Field(
        description="Directory containing the SFT model.",
    )
    dpo_data_path: CliPositionalArg[Path] = Field(
        description="Path to the DPO preference JSONL file.",
    )
    align_dir: CliPositionalArg[Path] = Field(
        description="Output directory for the aligned model.",
    )
    metrics_path: CliPositionalArg[Path] = Field(
        description="Path to write align metrics JSON.",
    )
    params_path: Path = Field(
        Path("params.yaml"),
        description="Path to the params.yaml configuration file.",
    )

    def cli_cmd(self) -> None:
        align(
            self.tokenizer_dir,
            self.sft_dir,
            self.dpo_data_path,
            self.align_dir,
            self.metrics_path,
            self.params_path,
        )


_COMMANDS: dict[str, type[BaseSettings]] = {
    "tokenize": Tokenize,
    "pretrain": Pretrain,
    "sft": SFT,
    "align": Align,
}


def main() -> None:
    """Dispatch CLI subcommands to pipeline stage orchestrators."""
    dictConfig(_LOGGING)
    if len(sys.argv) < 2 or sys.argv[1] not in _COMMANDS:
        available = ", ".join(_COMMANDS)
        print(f"Usage: rawformer-train <command> [args]\n\nCommands: {available}", file=sys.stderr)
        sys.exit(1)

    command_name = sys.argv[1]
    sys.argv = [f"rawformer-train {command_name}", *sys.argv[2:]]
    CliApp.run(_COMMANDS[command_name])


if __name__ == "__main__":
    main()
