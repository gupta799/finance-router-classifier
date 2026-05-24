"""Fixed v1 route labels."""

from __future__ import annotations

LABELS: tuple[str, ...] = (
    "metric_extraction",
    "filing_summarization",
    "financial_qa",
    "financial_reasoning",
    "comparative_analysis",
)

LABEL_TO_ID: dict[str, int] = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL: dict[int, str] = {index: label for label, index in LABEL_TO_ID.items()}


def validate_route(route: str) -> str:
    if route not in LABEL_TO_ID:
        raise ValueError(f"Unknown route {route!r}; expected one of {', '.join(LABELS)}")
    return route
