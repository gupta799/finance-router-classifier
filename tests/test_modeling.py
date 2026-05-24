from __future__ import annotations

from pathlib import Path

import torch

from finance_router.labels import LABEL_TO_ID, LABELS
from finance_router.modeling import format_prediction, resolve_device


def test_label_map_order_is_stable() -> None:
    assert LABELS == (
        "metric_extraction",
        "filing_summarization",
        "financial_qa",
        "financial_reasoning",
        "comparative_analysis",
    )
    assert LABEL_TO_ID["metric_extraction"] == 0
    assert LABEL_TO_ID["comparative_analysis"] == 4


def test_format_prediction_contract() -> None:
    prediction = format_prediction(
        probabilities=[0.1, 0.55, 0.2, 0.05, 0.1],
        model_dir=Path("models/finance-router"),
        input_text="Summarize Apple revenue trends from its 10-K.",
    )

    assert prediction["selected_route"] == "filing_summarization"
    assert prediction["confidence"] == 0.55
    assert prediction["model_dir"] == "models/finance-router"
    assert prediction["input_chars"] == 45
    assert prediction["top_routes"][0] == {
        "route": "filing_summarization",
        "probability": 0.55,
    }
    assert {item["route"] for item in prediction["top_routes"]} == set(LABELS)


def test_resolve_device_cpu_and_auto() -> None:
    assert resolve_device("cpu").type == "cpu"
    auto = resolve_device("auto")
    assert auto.type in {"cpu", "mps", "cuda"}
    if torch.cuda.is_available():
        assert resolve_device("cuda").type == "cuda"
    if torch.backends.mps.is_available():
        assert resolve_device("mps").type == "mps"
