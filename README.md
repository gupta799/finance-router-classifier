# Finance Router Classifier

This repo trains a finance-only prompt route classifier. It assumes the incoming prompt is already
finance-related, then predicts which finance task route should handle it. It is inspired by the
classification layer in vLLM Semantic Router, but it does not include a proxy gateway, worker health
checks, zero-copy streaming, or vLLM backend handoff.

## Routes

- `metric_extraction`: exact financial figures, KPIs, line items, ratios, table values.
- `filing_summarization`: summarize or brief a 10-K, 10-Q, annual report, or filing excerpt.
- `financial_qa`: factual finance or filing QA without heavy calculation.
- `financial_reasoning`: drivers, why/how questions, valuation, interpretation, trend analysis.
- `comparative_analysis`: compare companies, periods, segments, metrics, sectors, or excerpts.

There is no non-finance fallback class and no sentiment route. The classifier is not a
finance-vs-nonfinance detector.

## Setup

```bash
git submodule update --init --recursive
uv sync --python 3.12
```

## Build Data

Data generation lives in the `data-gen` submodule:

```bash
cd data-gen

uv run synthetic-data-gen build \
  --out data/synthetic-10k \
  --train-size 8000 \
  --eval-size 2000 \
  --generator-model gemma4:e2b \
  --ollama-base-url http://localhost:11434 \
  --embedding-model BAAI/bge-small-en-v1.5 \
  --wandb-project finance-router-data-gen \
  --langsmith-project finance-router-data-gen
```

Then train the classifier from the generated JSONL files:

```bash
cd ..
uv run finance-router train \
  --device mps \
  --train data-gen/data/synthetic-10k/train.jsonl \
  --eval data-gen/data/synthetic-10k/eval.jsonl \
  --batch-size 4 \
  --max-length 768 \
  --epochs 3 \
  --out-dir models/finance-router-synthetic-10k
```

See [docs/DATASET.md](docs/DATASET.md) for the repo boundary and data handoff.

## Train

```bash
uv run finance-router train \
  --device mps \
  --train data-gen/data/synthetic-10k/train.jsonl \
  --eval data-gen/data/synthetic-10k/eval.jsonl \
  --model-name answerdotai/ModernBERT-base \
  --epochs 3 \
  --batch-size 4 \
  --max-length 768
```

Training data must already exist. Generate it in `data-gen`, or pass explicit `--train` and `--eval`
JSONL paths.

For RunPod or any CUDA host, use `--device auto` or `--device cuda`. `auto` resolves in this order:
CUDA, Apple MPS, then CPU.

See [docs/RUNPOD.md](docs/RUNPOD.md) for the one-command GPU training path.

## Weights & Biases

Weights & Biases logging is opt-in. Set a project name, then train as usual:

```bash
wandb login

uv run finance-router train \
  --device mps \
  --train data-gen/data/synthetic-10k/train.jsonl \
  --eval data-gen/data/synthetic-10k/eval.jsonl \
  --batch-size 4 \
  --max-length 768 \
  --wandb-project finance-router-classifier \
  --wandb-run-name synthetic-10k-gemma4-e2b
```

For local dry runs that can sync later:

```bash
uv run finance-router train \
  --device mps \
  --train data-gen/data/synthetic-10k/train.jsonl \
  --eval data-gen/data/synthetic-10k/eval.jsonl \
  --batch-size 4 \
  --max-length 768 \
  --wandb-project finance-router-classifier \
  --wandb-mode offline
```

## Training Graphs

Generate pushable local reports from a saved training run:

```bash
uv run finance-router plot-metrics \
  --model-dir models/finance-router \
  --out reports/finance-router-training
```

This writes:

- `reports/finance-router-training/training_curves.png`
- `reports/finance-router-training/metrics.csv`
- `reports/finance-router-training/metrics.json`
- `reports/finance-router-training/README.md`

## Evaluate

```bash
uv run finance-router evaluate --model-dir models/finance-router
```

## Predict

```bash
uv run finance-router predict \
  --model-dir models/finance-router \
  "Compare Apple and Microsoft revenue growth."
```

Prediction output:

```json
{
  "selected_route": "comparative_analysis",
  "confidence": 0.73,
  "top_routes": [
    {"route": "comparative_analysis", "probability": 0.73}
  ],
  "model_dir": "models/finance-router",
  "input_chars": 43
}
```

## Smoke Commands

```bash
uv run pytest
uv run finance-router train \
  --device cpu \
  --train data-gen/data/smoke/train.jsonl \
  --eval data-gen/data/smoke/eval.jsonl \
  --epochs 1 \
  --max-steps 2
```
