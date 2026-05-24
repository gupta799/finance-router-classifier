"""Finance-only dataset builders for task-route classification."""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from finance_router.labels import LABELS
from finance_router.schema import RouterExample, normalize_text, stable_id, write_jsonl

FINANCEBENCH_SOURCE = "PatronusAI/financebench"
SUJET_SOURCE = "sujet-ai/Sujet-Financial-RAG-EN-Dataset"

DEFAULT_TRAIN_SIZE = 4000
DEFAULT_EVAL_SIZE = 1000

METRIC_TERMS = (
    "accounts payable",
    "assets",
    "book value",
    "capex",
    "capital expenditure",
    "cash flow",
    "current ratio",
    "debt",
    "ebit",
    "ebitda",
    "eps",
    "equity",
    "free cash flow",
    "gross margin",
    "income",
    "liabilities",
    "margin",
    "net income",
    "operating income",
    "operating margin",
    "pp&e",
    "ppne",
    "ratio",
    "revenue",
    "sales",
    "shareholders equity",
    "total equity",
)

REASONING_TERMS = (
    "assess",
    "capital-intensive",
    "decline",
    "driver",
    "drove",
    "explain",
    "growth",
    "how",
    "impact",
    "implication",
    "improve",
    "interpret",
    "performance",
    "trend",
    "valuation",
    "why",
)

COMPARISON_TERMS = (
    "against",
    "better",
    "compare",
    "compared",
    "difference between",
    "relative to",
    "versus",
    "vs.",
    " vs ",
    "which company",
)

SUMMARY_TERMS = (
    "brief",
    "condense",
    "key takeaways",
    "summarize",
    "summary",
)

FILING_TERMS = (
    "10-k",
    "10-q",
    "annual report",
    "filing",
    "form 10-k",
    "form 10-q",
    "risk factors",
)

SUMMARY_TEMPLATES = (
    (
        "filing_summary_core",
        "Summarize the key financial information in this filing excerpt.\n\n{context}",
    ),
    (
        "analyst_brief",
        "Give me a concise analyst brief of this 10-K section.\n\n{context}",
    ),
    (
        "risk_performance_summary",
        "Summarize the risks, operating performance, and financial highlights in this excerpt.\n\n"
        "{context}",
    ),
)

METRIC_TEMPLATES = (
    (
        "extract_metric_from_excerpt",
        "Extract the relevant financial metric from this filing excerpt.\n\n"
        "Question: {question}\n\nExcerpt: {context}",
    ),
)

REASONING_TEMPLATES = (
    (
        "financial_implications",
        "Explain the financial implications of this filing section.\n\n{context}",
    ),
    (
        "capital_intensity",
        "Is this company capital intensive based on the filing excerpt?\n\n{context}",
    ),
)

COMPARISON_TEMPLATES = (
    (
        "compare_cash_flow_profile",
        "Compare these two filing excerpts and identify the stronger cash-flow profile.\n\n"
        "Excerpt A:\n{context_a}\n\nExcerpt B:\n{context_b}",
    ),
    (
        "compare_revenue_margin_signals",
        "Compare the revenue and margin signals in these two filing excerpts.\n\n"
        "Excerpt A:\n{context_a}\n\nExcerpt B:\n{context_b}",
    ),
    (
        "compare_operating_performance",
        "Compare the operating performance described in these two filing excerpts.\n\n"
        "Excerpt A:\n{context_a}\n\nExcerpt B:\n{context_b}",
    ),
)


@dataclass(frozen=True)
class ContextRecord:
    context: str
    context_hash: str
    source: str
    company: str | None = None
    source_split: str | None = None

    @property
    def group_key(self) -> str:
        prefix = "financebench" if self.source == FINANCEBENCH_SOURCE else "sujet"
        return f"{prefix}:context:{self.context_hash}"


def load_hf_dataset(dataset_name: str, *args: Any, **kwargs: Any):
    from datasets import load_dataset

    return load_dataset(dataset_name, *args, **kwargs)


def contains_any(text: str, terms: Sequence[str]) -> bool:
    lowered = f" {text.lower()} "
    return any(term in lowered for term in terms)


def context_hash(text: str) -> str:
    return stable_id(normalize_text(text).lower())


def trim_context(text: str, max_chars: int = 2500) -> str:
    normalized = normalize_text(text)
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars].rsplit(" ", 1)[0]


def clean_company(candidate: str) -> str | None:
    cleaned = normalize_text(candidate).strip(" .,")
    company_suffix = r"(?:Inc\.?|Corporation|Corp\.?|Company|Co\.?|Ltd\.?|LLC|PLC|Group)"
    company_mentions = re.findall(
        rf"([A-Z][A-Za-z0-9&.'-]*(?: [A-Z][A-Za-z0-9&.'-]*){{0,5}},? {company_suffix})\b",
        cleaned,
    )
    if company_mentions:
        cleaned = normalize_text(company_mentions[-1]).strip(" .,")

    lowered = cleaned.lower()
    bad_prefixes = (
        "according ",
        "analyze ",
        "as of ",
        "by what ",
        "calculate ",
        "ceo ",
        "chief ",
        "class ",
        "compare ",
        "consolidated ",
        "delaware ",
        "describe ",
        "discuss ",
        "explain ",
        "how ",
        "identify ",
        "in what ",
        "what ",
        "which ",
    )
    if not cleaned or lowered in {"company", "the company"}:
        return None
    if lowered.startswith(bad_prefixes) or " the company" in lowered:
        return None
    if len(cleaned) > 70:
        return None
    return cleaned


def extract_company(question: str, context: str = "") -> str | None:
    company_suffix = r"(?:Inc\.?|Corporation|Corp\.?|Company|Co\.?|Ltd\.?|LLC|PLC|Group)"
    candidates = [
        rf"\bfor ([A-Z][A-Za-z0-9 .,&'-]{{2,80}}?{company_suffix})(?: as | with |,|\?|$)",
        rf"\bby ([A-Z][A-Za-z0-9 .,&'-]{{2,80}}?{company_suffix})(?: with |,|\?|$)",
        rf"\b([A-Z][A-Za-z0-9 .,&'-]{{2,80}}?{company_suffix})'s\b",
    ]
    for pattern in candidates:
        match = re.search(pattern, question)
        if match:
            company = clean_company(match.group(1))
            if company:
                return company

    match = re.search(r"Exact name of registrant.*?([A-Z][A-Za-z0-9 .,&'-]{3,80})", context)
    if match:
        return clean_company(match.group(1))
    return None


def classify_prompt(text: str, *, fallback: str = "financial_qa") -> str:
    normalized = normalize_text(text)
    if contains_any(normalized, COMPARISON_TERMS):
        return "comparative_analysis"
    if contains_any(normalized, SUMMARY_TERMS) and contains_any(normalized, FILING_TERMS):
        return "filing_summarization"
    if contains_any(normalized, REASONING_TERMS):
        return "financial_reasoning"
    if contains_any(normalized, METRIC_TERMS):
        return "metric_extraction"
    return fallback


def financebench_question_route(row: dict[str, Any]) -> str:
    question = normalize_text(row.get("question") or "")
    reasoning = str(row.get("question_reasoning") or "")
    question_type = str(row.get("question_type") or "")

    if contains_any(question, COMPARISON_TERMS):
        return "comparative_analysis"
    if contains_any(question, REASONING_TERMS) or any(
        marker.lower() in reasoning.lower()
        for marker in ("Numerical reasoning", "Logical reasoning")
    ):
        if not (
            question_type == "metrics-generated"
            and contains_any(question, METRIC_TERMS)
            and not contains_any(question, REASONING_TERMS)
        ):
            return "financial_reasoning"
    if question_type == "metrics-generated" or "information extraction" in reasoning.lower():
        return "metric_extraction"
    return classify_prompt(question)


def make_metadata(
    *,
    source: str,
    label_rule: str,
    group_key: str,
    template_id: str,
    source_split: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = {
        "source": source,
        "label_rule": label_rule,
        "group_key": group_key,
        "template_id": template_id,
    }
    if source_split:
        metadata["source_split"] = source_split
    if extra:
        metadata.update(extra)
    return metadata


def make_example(
    *,
    text: str,
    route: str,
    source: str,
    label_rule: str,
    group_key: str,
    template_id: str,
    company: str | None = None,
    source_split: str | None = None,
    extra: dict[str, Any] | None = None,
) -> RouterExample | None:
    text = normalize_text(text)
    if not text:
        return None
    return RouterExample(
        id=stable_id(source, route, label_rule, template_id, group_key, text),
        text=text,
        route=route,
        source=source,
        company=company,
        metadata=make_metadata(
            source=source,
            label_rule=label_rule,
            group_key=group_key,
            template_id=template_id,
            source_split=source_split,
            extra=extra,
        ),
    )


def financebench_contexts(row: dict[str, Any]) -> Iterable[ContextRecord]:
    for evidence in row.get("evidence") or []:
        evidence_text = (
            evidence.get("evidence_text_full_page") or evidence.get("evidence_text") or ""
        )
        evidence_text = trim_context(evidence_text)
        if len(evidence_text) < 200:
            continue
        yield ContextRecord(
            context=evidence_text,
            context_hash=context_hash(evidence_text),
            source=FINANCEBENCH_SOURCE,
            company=row.get("company"),
        )


def iter_financebench_candidates() -> Iterable[RouterExample]:
    rows = load_hf_dataset(FINANCEBENCH_SOURCE, split="train")
    seen_contexts: set[str] = set()

    for row in rows:
        question = normalize_text(row.get("question") or "")
        if question:
            route = financebench_question_route(row)
            group_key = f"financebench:{row.get('company')}:{row.get('doc_name')}"
            yield make_example(
                text=question,
                route=route,
                source=FINANCEBENCH_SOURCE,
                label_rule="financebench_question_metadata",
                group_key=group_key,
                template_id="original_question",
                company=row.get("company"),
                extra={
                    "financebench_id": row.get("financebench_id"),
                    "question_type": row.get("question_type"),
                    "question_reasoning": row.get("question_reasoning"),
                    "doc_name": row.get("doc_name"),
                    "doc_period": row.get("doc_period"),
                },
            )

        for record in financebench_contexts(row):
            if record.context_hash in seen_contexts:
                continue
            seen_contexts.add(record.context_hash)
            for template_id, template in SUMMARY_TEMPLATES:
                yield make_example(
                    text=template.format(context=record.context),
                    route="filing_summarization",
                    source=FINANCEBENCH_SOURCE,
                    label_rule="financebench_evidence_summary_template",
                    group_key=record.group_key,
                    template_id=template_id,
                    company=record.company,
                    extra={"doc_name": row.get("doc_name"), "doc_period": row.get("doc_period")},
                )


def iter_sujet_rows(max_rows: int | None = None) -> Iterable[tuple[str, dict[str, Any]]]:
    seen = 0
    for split in ("train", "test"):
        rows = load_hf_dataset(SUJET_SOURCE, split=split, streaming=True)
        for row in rows:
            yield split, row
            seen += 1
            if max_rows is not None and seen >= max_rows:
                return


def sujet_original_example(split: str, row: dict[str, Any]) -> RouterExample | None:
    question = normalize_text(row.get("question") or "")
    context = trim_context(row.get("context") or "")
    if not question or not context:
        return None
    route = classify_prompt(question)
    group_key = f"sujet:context:{context_hash(context)}"
    return make_example(
        text=question,
        route=route,
        source=SUJET_SOURCE,
        label_rule=f"{route}_heuristic_question",
        group_key=group_key,
        template_id="original_question",
        company=extract_company(question, context),
        source_split=split,
    )


def iter_sujet_candidates(
    *,
    per_route_candidate_goal: int,
    max_rows: int | None = None,
) -> tuple[list[RouterExample], list[ContextRecord]]:
    candidates: list[RouterExample] = []
    contexts: list[ContextRecord] = []
    counts: Counter[str] = Counter()
    seen_contexts: set[str] = set()

    for split, row in iter_sujet_rows(max_rows):
        question = normalize_text(row.get("question") or "")
        context = trim_context(row.get("context") or "")
        if not question or not context:
            continue

        group_hash = context_hash(context)
        group_key = f"sujet:context:{group_hash}"
        company = extract_company(question, context)

        original = sujet_original_example(split, row)
        if original and counts[original.route] < per_route_candidate_goal:
            candidates.append(original)
            counts[original.route] += 1

        if group_hash not in seen_contexts:
            seen_contexts.add(group_hash)
            contexts.append(
                ContextRecord(
                    context=context,
                    context_hash=group_hash,
                    source=SUJET_SOURCE,
                    company=company,
                    source_split=split,
                )
            )
            if counts["filing_summarization"] < per_route_candidate_goal:
                for template_id, template in SUMMARY_TEMPLATES:
                    example = make_example(
                        text=template.format(context=context),
                        route="filing_summarization",
                        source=SUJET_SOURCE,
                        label_rule="sujet_context_summary_template",
                        group_key=group_key,
                        template_id=template_id,
                        company=company,
                        source_split=split,
                    )
                    if example and counts["filing_summarization"] < per_route_candidate_goal:
                        candidates.append(example)
                        counts["filing_summarization"] += 1

            if counts["financial_reasoning"] < per_route_candidate_goal:
                for template_id, template in REASONING_TEMPLATES:
                    example = make_example(
                        text=template.format(context=context),
                        route="financial_reasoning",
                        source=SUJET_SOURCE,
                        label_rule="sujet_context_reasoning_template",
                        group_key=group_key,
                        template_id=template_id,
                        company=company,
                        source_split=split,
                    )
                    if example and counts["financial_reasoning"] < per_route_candidate_goal:
                        candidates.append(example)
                        counts["financial_reasoning"] += 1

        if classify_prompt(question) == "metric_extraction" and counts[
            "metric_extraction"
        ] < per_route_candidate_goal:
            for template_id, template in METRIC_TEMPLATES:
                example = make_example(
                    text=template.format(question=question, context=context),
                    route="metric_extraction",
                    source=SUJET_SOURCE,
                    label_rule="sujet_metric_extraction_template",
                    group_key=group_key,
                    template_id=template_id,
                    company=company,
                    source_split=split,
                )
                if example and counts["metric_extraction"] < per_route_candidate_goal:
                    candidates.append(example)
                    counts["metric_extraction"] += 1

        non_comparison_routes = [route for route in LABELS if route != "comparative_analysis"]
        enough_non_comparison = all(
            counts[route] >= per_route_candidate_goal for route in non_comparison_routes
        )
        if enough_non_comparison and len(contexts) >= per_route_candidate_goal:
            break

    return candidates, contexts


def iter_comparison_candidates(
    contexts: Sequence[ContextRecord],
    *,
    per_route_candidate_goal: int,
) -> Iterable[RouterExample]:
    usable_contexts = sorted(contexts, key=lambda item: item.context_hash)
    count = 0
    for index in range(0, max(0, len(usable_contexts) - 1)):
        left = usable_contexts[index]
        right = usable_contexts[-index - 1]
        if left.context_hash == right.context_hash:
            continue
        pair_hash = ":".join(sorted([left.context_hash, right.context_hash]))
        group_key = f"comparison:contexts:{pair_hash}"
        for template_id, template in COMPARISON_TEMPLATES:
            example = make_example(
                text=template.format(
                    context_a=trim_context(left.context, 1100),
                    context_b=trim_context(right.context, 1100),
                ),
                route="comparative_analysis",
                source=f"{left.source}+{right.source}",
                label_rule="context_pair_comparison_template",
                group_key=group_key,
                template_id=template_id,
                company=left.company or right.company,
                source_split="+".join(
                    sorted({split for split in (left.source_split, right.source_split) if split})
                )
                or None,
                extra={
                    "left_context_hash": left.context_hash,
                    "right_context_hash": right.context_hash,
                },
            )
            if example:
                yield example
                count += 1
                if count >= per_route_candidate_goal:
                    return


def dedupe_examples(rows: Iterable[RouterExample | None]) -> list[RouterExample]:
    seen: set[tuple[str, str]] = set()
    deduped: list[RouterExample] = []
    for row in rows:
        if row is None:
            continue
        key = (normalize_text(row.text).lower(), row.route)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def build_candidate_examples(
    *,
    train_size: int = DEFAULT_TRAIN_SIZE,
    eval_size: int = DEFAULT_EVAL_SIZE,
    candidate_multiplier: int = 6,
    max_sujet_rows: int | None = None,
) -> list[RouterExample]:
    if train_size % len(LABELS) != 0 or eval_size % len(LABELS) != 0:
        raise ValueError("train_size and eval_size must be divisible by the number of labels.")

    per_route_needed = (train_size + eval_size) // len(LABELS)
    per_route_candidate_goal = max(per_route_needed * candidate_multiplier, per_route_needed + 250)

    financebench = list(iter_financebench_candidates())
    sujet, contexts = iter_sujet_candidates(
        per_route_candidate_goal=per_route_candidate_goal,
        max_rows=max_sujet_rows,
    )
    comparisons = list(
        iter_comparison_candidates(
            contexts,
            per_route_candidate_goal=per_route_candidate_goal,
        )
    )
    return dedupe_examples([*financebench, *sujet, *comparisons])


def split_exact_by_route(
    rows: Sequence[RouterExample],
    *,
    train_size: int = DEFAULT_TRAIN_SIZE,
    eval_size: int = DEFAULT_EVAL_SIZE,
    seed: int = 7,
) -> tuple[list[RouterExample], list[RouterExample]]:
    if train_size % len(LABELS) != 0 or eval_size % len(LABELS) != 0:
        raise ValueError("train_size and eval_size must be divisible by the number of labels.")

    train_quota = train_size // len(LABELS)
    eval_quota = eval_size // len(LABELS)
    rng = random.Random(seed)

    by_route: dict[str, list[RouterExample]] = defaultdict(list)
    for row in rows:
        by_route[row.route].append(row)
    for route_rows in by_route.values():
        route_rows.sort(key=lambda row: row.id or "")
        rng.shuffle(route_rows)

    eval_groups: set[str] = set()
    train_groups: set[str] = set()
    eval_rows: list[RouterExample] = []
    train_rows: list[RouterExample] = []

    for route in LABELS:
        route_eval: list[RouterExample] = []
        for row in by_route.get(route, []):
            if row.group_key in train_groups:
                continue
            eval_groups.add(row.group_key)
            route_eval.append(row)
            if len(route_eval) == eval_quota:
                break
        if len(route_eval) < eval_quota:
            raise ValueError(
                f"Not enough eval examples for {route}: need {eval_quota}, got {len(route_eval)}"
            )
        eval_rows.extend(route_eval)

    for route in LABELS:
        route_train: list[RouterExample] = []
        for row in by_route.get(route, []):
            if row.group_key in eval_groups:
                continue
            train_groups.add(row.group_key)
            route_train.append(row)
            if len(route_train) == train_quota:
                break
        if len(route_train) < train_quota:
            raise ValueError(
                f"Not enough train examples for {route}: need {train_quota}, got {len(route_train)}"
            )
        train_rows.extend(route_train)

    rng.shuffle(train_rows)
    rng.shuffle(eval_rows)
    return train_rows, eval_rows


def validate_exact_splits(
    *,
    train: Sequence[RouterExample],
    eval_rows: Sequence[RouterExample],
    train_size: int,
    eval_size: int,
) -> None:
    train_counts = Counter(row.route for row in train)
    eval_counts = Counter(row.route for row in eval_rows)
    expected_train = train_size // len(LABELS)
    expected_eval = eval_size // len(LABELS)

    if len(train) != train_size:
        raise ValueError(f"Expected {train_size} train rows, got {len(train)}")
    if len(eval_rows) != eval_size:
        raise ValueError(f"Expected {eval_size} eval rows, got {len(eval_rows)}")

    for route in LABELS:
        if train_counts[route] != expected_train:
            raise ValueError(
                f"Train route {route} expected {expected_train}, got {train_counts[route]}"
            )
        if eval_counts[route] != expected_eval:
            raise ValueError(
                f"Eval route {route} expected {expected_eval}, got {eval_counts[route]}"
            )

    train_groups = {row.group_key for row in train}
    eval_groups = {row.group_key for row in eval_rows}
    leaked = train_groups & eval_groups
    if leaked:
        sample = ", ".join(sorted(leaked)[:5])
        raise ValueError(f"Group leakage detected between train and eval: {sample}")


def dataset_summary(
    rows: Sequence[RouterExample],
    train: Sequence[RouterExample],
    eval_rows: Sequence[RouterExample],
) -> dict[str, Any]:
    return {
        "total_candidate_rows": len(rows),
        "train_rows": len(train),
        "eval_rows": len(eval_rows),
        "train_routes": dict(sorted(Counter(row.route for row in train).items())),
        "eval_routes": dict(sorted(Counter(row.route for row in eval_rows).items())),
        "candidate_routes": dict(sorted(Counter(row.route for row in rows).items())),
        "sources": dict(sorted(Counter(row.source for row in rows).items())),
        "companies": sorted({row.company for row in rows if row.company})[:200],
        "schema": ["id", "text", "route", "source", "company", "metadata"],
        "labels": list(LABELS),
    }


def build_and_write_dataset(
    *,
    out_dir: Path,
    train_size: int = DEFAULT_TRAIN_SIZE,
    eval_size: int = DEFAULT_EVAL_SIZE,
    seed: int = 7,
    candidate_multiplier: int = 6,
    max_sujet_rows: int | None = None,
) -> dict[str, Any]:
    rows = build_candidate_examples(
        train_size=train_size,
        eval_size=eval_size,
        candidate_multiplier=candidate_multiplier,
        max_sujet_rows=max_sujet_rows,
    )
    train, eval_rows = split_exact_by_route(
        rows,
        train_size=train_size,
        eval_size=eval_size,
        seed=seed,
    )
    validate_exact_splits(
        train=train,
        eval_rows=eval_rows,
        train_size=train_size,
        eval_size=eval_size,
    )

    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "eval.jsonl", eval_rows)
    summary = dataset_summary(rows, train, eval_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
