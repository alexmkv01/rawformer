"""Shared path constants for the rawformer_train pipeline stages."""

from pathlib import Path

# parents[2]: _paths.py -> rawformer_train/ -> train/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"

# Data directories (DVC-tracked)
PRETRAIN_DATA_PATH = PROJECT_ROOT / "data" / "pretrain" / "stories.txt"
SFT_DATA_PATH = PROJECT_ROOT / "data" / "sft" / "instructions.jsonl"
DPO_DATA_PATH = PROJECT_ROOT / "data" / "dpo" / "preferences.jsonl"

# Pipeline artifact directories
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
TOKENIZER_DIR = ARTIFACTS_DIR / "tokenizer"
PRETRAIN_DIR = ARTIFACTS_DIR / "pretrain"
SFT_DIR = ARTIFACTS_DIR / "sft"
ALIGN_DIR = ARTIFACTS_DIR / "align"
