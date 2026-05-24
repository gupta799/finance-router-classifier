#!/usr/bin/env bash
set -euo pipefail

TRAIN_SIZE="${TRAIN_SIZE:-4000}"
EVAL_SIZE="${EVAL_SIZE:-1000}"
MODEL_NAME="${MODEL_NAME:-answerdotai/ModernBERT-base}"
OUT_DIR="${OUT_DIR:-models/finance-router}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"

uv sync --python 3.12

uv run finance-router build-data \
  --train-size "${TRAIN_SIZE}" \
  --eval-size "${EVAL_SIZE}" \
  --out data/processed

uv run finance-router train \
  --device auto \
  --model-name "${MODEL_NAME}" \
  --train data/processed/train.jsonl \
  --eval data/processed/eval.jsonl \
  --out-dir "${OUT_DIR}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --max-length "${MAX_LENGTH}" \
  --learning-rate "${LEARNING_RATE}"

uv run finance-router evaluate \
  --device auto \
  --model-dir "${OUT_DIR}" \
  --test data/processed/eval.jsonl \
  --batch-size "${BATCH_SIZE}"

mkdir -p outputs
tar -czf outputs/finance-router-artifacts.tar.gz "${OUT_DIR}" data/processed/summary.json
