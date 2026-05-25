from __future__ import annotations

from finance_router.reporting import training_history_rows


def test_training_history_rows_extracts_epoch_metrics() -> None:
    rows = training_history_rows(
        {
            "history": [
                {
                    "epoch": 1,
                    "global_step": 1000,
                    "train_loss": 0.61,
                    "eval": {
                        "loss": 0.23,
                        "accuracy": 0.915,
                        "macro_f1": 0.916,
                    },
                }
            ]
        }
    )

    assert rows == [
        {
            "epoch": 1,
            "global_step": 1000,
            "train_loss": 0.61,
            "eval_loss": 0.23,
            "eval_accuracy": 0.915,
            "eval_macro_f1": 0.916,
        }
    ]
