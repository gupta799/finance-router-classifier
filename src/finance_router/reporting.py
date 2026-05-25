"""Local training report generation."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any


def load_metrics(model_dir: Path) -> dict[str, Any]:
    metrics_path = model_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Expected metrics at {metrics_path}")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def training_history_rows(metrics: dict[str, Any]) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for entry in metrics.get("history", []):
        eval_metrics = entry.get("eval", {})
        rows.append(
            {
                "epoch": int(entry["epoch"]),
                "global_step": int(entry["global_step"]),
                "train_loss": float(entry["train_loss"]),
                "eval_loss": float(eval_metrics["loss"]),
                "eval_accuracy": float(eval_metrics["accuracy"]),
                "eval_macro_f1": float(eval_metrics["macro_f1"]),
            }
        )
    if not rows:
        raise ValueError("metrics.json does not contain training history.")
    return rows


def write_metrics_csv(rows: list[dict[str, float | int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_training_curves(rows: list[dict[str, float | int]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]
    eval_loss = [float(row["eval_loss"]) for row in rows]
    eval_accuracy = [float(row["eval_accuracy"]) for row in rows]
    eval_macro_f1 = [float(row["eval_macro_f1"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=160)
    fig.suptitle("Finance Router Classifier Training", fontsize=14, fontweight="bold")

    axes[0].plot(epochs, train_loss, marker="o", label="train loss", color="#2563eb")
    axes[0].plot(epochs, eval_loss, marker="o", label="eval loss", color="#dc2626")
    axes[0].set_title("Cross-Entropy Loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_xticks(epochs)
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, eval_accuracy, marker="o", label="eval accuracy", color="#059669")
    axes[1].plot(epochs, eval_macro_f1, marker="o", label="eval macro F1", color="#7c3aed")
    axes[1].set_title("Eval Routing Quality")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("score")
    axes[1].set_xticks(epochs)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_markdown_report(
    *,
    rows: list[dict[str, float | int]],
    metrics: dict[str, Any],
    out_dir: Path,
) -> None:
    final = rows[-1]
    lines = [
        "# Finance Router Training Report",
        "",
        "![Training curves](training_curves.png)",
        "",
        "## Final Eval",
        "",
        f"- Device: `{metrics.get('device', 'unknown')}`",
        f"- Train rows: `{metrics.get('train_rows', 'unknown')}`",
        f"- Eval rows: `{metrics.get('eval_rows', 'unknown')}`",
        f"- Global steps: `{metrics.get('global_step', 'unknown')}`",
        f"- Eval accuracy: `{float(final['eval_accuracy']):.3f}`",
        f"- Eval macro F1: `{float(final['eval_macro_f1']):.3f}`",
        f"- Eval loss: `{float(final['eval_loss']):.3f}`",
        "",
        "## Epoch History",
        "",
        "| epoch | step | train loss | eval loss | eval accuracy | eval macro F1 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {epoch} | {global_step} | {train_loss:.4f} | {eval_loss:.4f} | "
            "{eval_accuracy:.4f} | {eval_macro_f1:.4f} |".format(**row)
        )
    lines.append("")
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def build_training_report(model_dir: Path, out_dir: Path) -> dict[str, Any]:
    metrics = load_metrics(model_dir)
    rows = training_history_rows(metrics)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_out = out_dir / "metrics.json"
    config_path = model_dir / "training_config.json"
    shutil.copy2(model_dir / "metrics.json", metrics_out)
    if config_path.exists():
        shutil.copy2(config_path, out_dir / "training_config.json")

    csv_path = out_dir / "metrics.csv"
    plot_path = out_dir / "training_curves.png"
    write_metrics_csv(rows, csv_path)
    plot_training_curves(rows, plot_path)
    write_markdown_report(rows=rows, metrics=metrics, out_dir=out_dir)

    return {
        "model_dir": str(model_dir),
        "out_dir": str(out_dir),
        "metrics": str(metrics_out),
        "csv": str(csv_path),
        "plot": str(plot_path),
        "report": str(out_dir / "README.md"),
        "final": rows[-1],
    }
