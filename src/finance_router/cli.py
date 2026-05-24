"""Command line interface for the finance router classifier."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from finance_router.data import build_and_write_dataset
from finance_router.modeling import (
    TrainConfig,
    evaluate_model_dir,
    predict_text,
    train_classifier,
)


def print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2))


def cmd_build_data(args: argparse.Namespace) -> None:
    summary = build_and_write_dataset(
        out_dir=Path(args.out),
        train_size=args.train_size,
        eval_size=args.eval_size,
        seed=args.seed,
        candidate_multiplier=args.candidate_multiplier,
        max_sujet_rows=args.max_sujet_rows,
    )
    print_json(summary)


def ensure_training_data(args: argparse.Namespace) -> tuple[Path, Path]:
    train_path = Path(args.train) if args.train else Path(args.data_dir) / "train.jsonl"
    eval_path = Path(args.eval) if args.eval else Path(args.data_dir) / "eval.jsonl"
    custom_paths = args.train or args.eval
    missing_custom_path = not train_path.exists() or not eval_path.exists()
    if custom_paths and missing_custom_path:
        raise SystemExit(
            "Custom --train/--eval paths must both exist. "
            "Omit them to auto-build data under --data-dir."
        )
    missing = not train_path.exists() or not eval_path.exists()
    if missing or args.rebuild_data:
        build_and_write_dataset(
            out_dir=Path(args.data_dir),
            train_size=args.train_size,
            eval_size=args.eval_size,
            seed=args.seed,
            candidate_multiplier=args.candidate_multiplier,
            max_sujet_rows=args.max_sujet_rows,
        )
    return train_path, eval_path


def cmd_train(args: argparse.Namespace) -> None:
    train_path, eval_path = ensure_training_data(args)
    config = TrainConfig(
        model_name=args.model_name,
        train_path=str(train_path),
        eval_path=str(eval_path),
        out_dir=args.out_dir,
        device=args.device,
        max_length=args.max_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
        max_steps=args.max_steps,
        limit_train=args.limit_train,
        limit_eval=args.limit_eval,
    )
    print_json(train_classifier(config))


def cmd_evaluate(args: argparse.Namespace) -> None:
    metrics = evaluate_model_dir(
        model_dir=Path(args.model_dir),
        data_path=Path(args.test),
        device_name=args.device,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    print_json(metrics)


def cmd_predict(args: argparse.Namespace) -> None:
    text = " ".join(args.prompt).strip()
    if not text:
        text = sys.stdin.read().strip()
    print_json(
        predict_text(
            model_dir=Path(args.model_dir),
            text=text,
            device_name=args.device,
            max_length=args.max_length,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="finance-router",
        description="Build data, train, evaluate, and run a finance route classifier.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_data = subparsers.add_parser("build-data", help="Build JSONL train/eval data.")
    build_data.add_argument("--out", default="data/processed")
    build_data.add_argument("--train-size", type=int, default=4000)
    build_data.add_argument("--eval-size", type=int, default=1000)
    build_data.add_argument("--seed", type=int, default=7)
    build_data.add_argument("--candidate-multiplier", type=int, default=6)
    build_data.add_argument("--max-sujet-rows", type=int)
    build_data.set_defaults(func=cmd_build_data)

    train = subparsers.add_parser("train", help="Train the ModernBERT route classifier.")
    train.add_argument("--model-name", default="answerdotai/ModernBERT-base")
    train.add_argument("--data-dir", default="data/processed")
    train.add_argument("--train")
    train.add_argument("--eval")
    train.add_argument("--out-dir", default="models/finance-router")
    train.add_argument("--device", choices=["cuda", "mps", "cpu", "auto"], default="mps")
    train.add_argument("--max-length", type=int, default=1024)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--learning-rate", type=float, default=2e-5)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--warmup-ratio", type=float, default=0.06)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--max-steps", type=int)
    train.add_argument("--limit-train", type=int)
    train.add_argument("--limit-eval", type=int)
    train.add_argument("--train-size", type=int, default=4000)
    train.add_argument("--eval-size", type=int, default=1000)
    train.add_argument("--candidate-multiplier", type=int, default=6)
    train.add_argument("--max-sujet-rows", type=int)
    train.add_argument("--rebuild-data", action="store_true")
    train.set_defaults(func=cmd_train)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a saved classifier.")
    evaluate.add_argument("--model-dir", default="models/finance-router")
    evaluate.add_argument("--test", default="data/processed/eval.jsonl")
    evaluate.add_argument("--device", choices=["cuda", "mps", "cpu", "auto"], default="mps")
    evaluate.add_argument("--max-length", type=int)
    evaluate.add_argument("--batch-size", type=int, default=8)
    evaluate.set_defaults(func=cmd_evaluate)

    predict = subparsers.add_parser("predict", help="Predict a route for a prompt.")
    predict.add_argument("--model-dir", default="models/finance-router")
    predict.add_argument("--device", choices=["cuda", "mps", "cpu", "auto"], default="mps")
    predict.add_argument("--max-length", type=int)
    predict.add_argument("prompt", nargs="*")
    predict.set_defaults(func=cmd_predict)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
