#!/bin/sh
set -x -e

# ruff
ruff format
ruff check

# mypy
mypy .

# test
pytest -vv
