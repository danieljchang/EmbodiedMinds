#!/usr/bin/env bash
set -euo pipefail

# Activate venv if present
if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# Load .env into environment if present
if [ -f .env ]; then
  # export non-comment lines
  export $(grep -v '^\s*#' .env | xargs) || true
fi

# Allow passing a data path as first arg
DATA_ROOT_ARG=""
if [ -n "${1-}" ]; then
  DATA_ROOT_ARG="--data-root $1"
fi

python -m src.training.train \
  --config configs/train.yaml \
  --debug ${DATA_ROOT_ARG}