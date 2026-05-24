from __future__ import annotations

from collections import Counter

import pytest

from finance_router.data import (
    classify_prompt,
    dedupe_examples,
    financebench_question_route,
    split_exact_by_route,
    validate_exact_splits,
)
from finance_router.labels import LABELS
from finance_router.schema import RouterExample


def make_example(route: str, group: str, index: int) -> RouterExample:
    return RouterExample(
        id=f"{group}-{index}",
        text=f"Example finance prompt {route} {group} {index}",
        route=route,
        source="unit-test-finance",
        company="ExampleCo",
        metadata={
            "source": "unit-test-finance",
            "label_rule": "unit_test",
            "group_key": group,
            "template_id": "unit_template",
        },
    )


def test_label_precedence_for_ambiguous_prompts() -> None:
    assert (
        classify_prompt("Compare Apple revenue growth against Microsoft.")
        == "comparative_analysis"
    )
    assert classify_prompt("Summarize this 10-K filing excerpt.") == "filing_summarization"
    assert (
        classify_prompt("Why did revenue decline while margins improved?")
        == "financial_reasoning"
    )
    assert classify_prompt("What is FY2023 revenue?") == "metric_extraction"
    assert classify_prompt("What fiscal year does this annual report cover?") == "financial_qa"


def test_financebench_question_route_uses_metadata_for_metrics() -> None:
    row = {
        "question_type": "metrics-generated",
        "question_reasoning": "Information extraction",
        "question": "What is FY2022 revenue?",
    }
    assert financebench_question_route(row) == "metric_extraction"


def test_financebench_question_route_uses_reasoning_metadata() -> None:
    row = {
        "question_type": "domain-relevant",
        "question_reasoning": "Logical reasoning (based on numerical reasoning)",
        "question": "Is 3M a capital-intensive business based on FY2022 data?",
    }
    assert financebench_question_route(row) == "financial_reasoning"


def test_dedupe_examples_keeps_same_text_for_different_finance_routes() -> None:
    rows = [
        make_example("financial_qa", "g1", 1),
        make_example("financial_qa", "g1", 1),
        RouterExample(
            id="reasoning-route",
            text="Example finance prompt financial_qa g1 1",
            route="financial_reasoning",
            source="unit-test-finance",
            metadata={
                "source": "unit-test-finance",
                "label_rule": "unit_test",
                "group_key": "g2",
                "template_id": "unit_template",
            },
        ),
    ]
    deduped = dedupe_examples(rows)
    assert len(deduped) == 2
    assert Counter(row.route for row in deduped) == {
        "financial_qa": 1,
        "financial_reasoning": 1,
    }


def test_split_exact_by_route_has_exact_counts_and_no_group_leakage() -> None:
    rows: list[RouterExample] = []
    for route in LABELS:
        for group_index in range(8):
            group = f"finance:{route}:{group_index}"
            rows.append(make_example(route, group, 1))

    train, eval_rows = split_exact_by_route(rows, train_size=10, eval_size=5, seed=11)
    validate_exact_splits(train=train, eval_rows=eval_rows, train_size=10, eval_size=5)

    assert Counter(row.route for row in train) == {route: 2 for route in LABELS}
    assert Counter(row.route for row in eval_rows) == {route: 1 for route in LABELS}
    assert {row.group_key for row in train}.isdisjoint({row.group_key for row in eval_rows})


def test_split_exact_by_route_fails_when_quota_cannot_be_met() -> None:
    rows = [make_example(route, f"finance:{route}:0", 1) for route in LABELS]
    with pytest.raises(ValueError, match="Not enough train examples"):
        split_exact_by_route(rows, train_size=10, eval_size=5, seed=7)
