# Dataset Construction

This repository builds a finance-only prompt routing dataset. Every row is assumed to be a finance
prompt; the label identifies the finance task route that should handle it.

## Splits

The default build writes:

- `data/processed/train.jsonl`: 4,000 rows
- `data/processed/eval.jsonl`: 1,000 rows

Both splits are exactly balanced:

- 800 train and 200 eval rows for `metric_extraction`
- 800 train and 200 eval rows for `filing_summarization`
- 800 train and 200 eval rows for `financial_qa`
- 800 train and 200 eval rows for `financial_reasoning`
- 800 train and 200 eval rows for `comparative_analysis`

There is no `other` label and no sentiment label.

## Sources

The builder uses only finance data:

- `PatronusAI/financebench`
- `sujet-ai/Sujet-Financial-RAG-EN-Dataset`

FinanceBench contributes high-quality public-company filing questions, metadata, and evidence
pages. Sujet contributes a larger pool of finance questions and filing contexts so the balanced
4k/1k split can be reached without duplicating prompts.

FinanceBench is CC BY-NC 4.0. Treat trained artifacts built from the default source mix as
non-commercial unless you replace that source.

## Labeling Rules

Candidates are generated first, then deduplicated and sampled into exact quotas. Ambiguous prompts
follow this precedence:

1. `comparative_analysis`
2. `filing_summarization`
3. `financial_reasoning`
4. `metric_extraction`
5. `financial_qa`

`metric_extraction` includes prompts asking for specific numbers, KPIs, ratios, line items, table
values, or period-specific financial metrics.

`filing_summarization` includes prompts asking to summarize, brief, condense, or extract themes
from filing excerpts.

`financial_qa` includes factual finance or filing questions that do not require multi-step
reasoning, comparisons, or direct metric extraction.

`financial_reasoning` includes why/how questions, driver analysis, ratio interpretation,
calculation, trend explanation, valuation implications, and capital intensity judgments.

`comparative_analysis` includes prompts comparing companies, periods, segments, sectors, metrics,
or two filing excerpts.

## Generated Prompt Families

The builder keeps original finance questions where possible and adds deterministic templates for
route shapes that should appear in live traffic:

- original FinanceBench and Sujet questions
- metric extraction prompts using finance question/context pairs
- filing summarization prompts from FinanceBench evidence pages and Sujet contexts
- financial reasoning prompts from reasoning questions and filing contexts
- comparative prompts from same-company and cross-company context pairs

Each row stores provenance in `metadata`, including:

- `source`
- `label_rule`
- `group_key`
- `template_id`

## Leakage Control

Splitting is grouped, not row-random.

FinanceBench group keys are based on company and document name. Sujet group keys are based on a
stable context hash. Comparison examples use the sorted context-hash pair as the group key.

The build fails loudly if it cannot hit the requested per-label quota without train/eval group
leakage.

## Rebuild

```bash
uv run finance-router build-data \
  --train-size 4000 \
  --eval-size 1000 \
  --out data/processed
```

The generated JSONL and model artifacts are intentionally ignored by git.
