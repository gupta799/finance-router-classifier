# Dataset Handoff

Synthetic dataset generation lives in the `data-gen` Git submodule, which points to:

```text
https://github.com/gupta799/synthetic-data-gen
```

This classifier repo does not generate route-training data. It trains, evaluates, predicts, and
renders training reports from JSONL files that already exist.

## Generate Data

```bash
git submodule update --init --recursive
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

## Train From Generated Data

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

## Schema

The classifier expects each JSONL row to contain:

```json
{
  "id": "...",
  "text": "...",
  "route": "metric_extraction",
  "source": "synthetic:ollama:gemma4:e2b",
  "company": "Apple Inc",
  "metadata": {}
}
```

Routes:

- `metric_extraction`
- `filing_summarization`
- `financial_qa`
- `financial_reasoning`
- `comparative_analysis`
