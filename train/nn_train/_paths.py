"""Shared path constants for the nn_train pipeline stages."""

from pathlib import Path

# parents[2]: _paths.py -> nn_train/ -> train/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"
DATA_PATH = PROJECT_ROOT / "data" / "iris.dat"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
