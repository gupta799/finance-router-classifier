#!/usr/bin/env bash
set -euo pipefail

TRAIN_SIZE="${TRAIN_SIZE:-8000}"
EVAL_SIZE="${EVAL_SIZE:-2000}"
DATA_DIR="${DATA_DIR:-data-gen/data/synthetic-10k}"
GENERATOR_MODEL="${GENERATOR_MODEL:-gemma4:e2b}"
MODEL_NAME="${MODEL_NAME:-answerdotai/ModernBERT-base}"
OUT_DIR="${OUT_DIR:-models/finance-router}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
REPORT_DIR="${REPORT_DIR:-reports/finance-router-training}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
WANDB_MODE="${WANDB_MODE:-online}"

WANDB_ARGS=()
if [[ -n "${WANDB_PROJECT}" ]]; then
  WANDB_ARGS+=(--wandb-project "${WANDB_PROJECT}" --wandb-mode "${WANDB_MODE}")
fi
if [[ -n "${WANDB_ENTITY}" ]]; then
  WANDB_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi
if [[ -n "${WANDB_RUN_NAME}" ]]; then
  WANDB_ARGS+=(--wandb-run-name "${WANDB_RUN_NAME}")
fi

uv sync --python 3.12

git submodule update --init --recursive

if [[ ! -f "${DATA_DIR}/train.jsonl" || ! -f "${DATA_DIR}/eval.jsonl" ]]; then
  (
    cd data-gen
    uv sync --python 3.12
    uv run synthetic-data-gen build \
      --train-size "${TRAIN_SIZE}" \
      --eval-size "${EVAL_SIZE}" \
      --out "${DATA_DIR#data-gen/}" \
      --generator-model "${GENERATOR_MODEL}"
  )
fi

uv run finance-router train \
  --device auto \
  --model-name "${MODEL_NAME}" \
  --train "${DATA_DIR}/train.jsonl" \
  --eval "${DATA_DIR}/eval.jsonl" \
  --out-dir "${OUT_DIR}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --max-length "${MAX_LENGTH}" \
  --learning-rate "${LEARNING_RATE}" \
  "${WANDB_ARGS[@]}"

uv run finance-router evaluate \
  --device auto \
  --model-dir "${OUT_DIR}" \
  --test "${DATA_DIR}/eval.jsonl" \
  --batch-size "${BATCH_SIZE}"

uv run finance-router plot-metrics \
  --model-dir "${OUT_DIR}" \
  --out "${REPORT_DIR}"

mkdir -p outputs
tar -czf outputs/finance-router-artifacts.tar.gz \
  "${OUT_DIR}" \
  "${DATA_DIR}/summary.json" \
  "${REPORT_DIR}"
