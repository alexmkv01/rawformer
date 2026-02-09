#!/bin/sh
set -x -e

# ruff
ruff format --diff lib/ train/
ruff check lib/ train/

# mypy
mypy .

# test
pytest -vv
