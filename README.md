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
uv sync --python 3.12
```

## Build Data

```bash
uv run finance-router build-data \
  --train-size 4000 \
  --eval-size 1000 \
  --out data/processed
```

The builder writes:

- `data/processed/train.jsonl`
- `data/processed/eval.jsonl`
- `data/processed/summary.json`

Each split is balanced across the five routes: 800 train and 200 eval examples per route for the
default 4k/1k dataset.

Data sources:

- `PatronusAI/financebench`
- `sujet-ai/Sujet-Financial-RAG-EN-Dataset`

The builder stores `label_rule`, `template_id`, and `group_key` in every row's metadata. It fails if
the exact requested per-route quotas cannot be met without group leakage.

FinanceBench is licensed CC BY-NC 4.0, so artifacts trained from the default dataset should be
treated as non-commercial unless that source is replaced.

See [docs/DATASET.md](docs/DATASET.md) for the full candidate generation, labeling, and leakage
control rules.

## Train

```bash
uv run finance-router train \
  --device mps \
  --model-name answerdotai/ModernBERT-base \
  --epochs 3 \
  --batch-size 8 \
  --max-length 1024
```

If `data/processed/train.jsonl` and `data/processed/eval.jsonl` do not exist, `train` builds them
first with the configured `--train-size` and `--eval-size`.

For RunPod or any CUDA host, use `--device auto` or `--device cuda`. `auto` resolves in this order:
CUDA, Apple MPS, then CPU.

See [docs/RUNPOD.md](docs/RUNPOD.md) for the one-command GPU training path.

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
uv run finance-router build-data --train-size 400 --eval-size 100 --out data/smoke-sized
uv run finance-router train \
  --device cpu \
  --train data/smoke-sized/train.jsonl \
  --eval data/smoke-sized/eval.jsonl \
  --epochs 1 \
  --max-steps 2
```
