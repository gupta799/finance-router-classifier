"""Training, evaluation, and prediction for the finance route classifier."""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from finance_router.labels import ID_TO_LABEL, LABEL_TO_ID, LABELS
from finance_router.schema import RouterExample, read_jsonl


@dataclass(frozen=True)
class TrainConfig:
    model_name: str = "answerdotai/ModernBERT-base"
    train_path: str = "data/processed/train.jsonl"
    eval_path: str = "data/processed/eval.jsonl"
    out_dir: str = "models/finance-router"
    device: str = "mps"
    max_length: int = 1024
    batch_size: int = 8
    epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    seed: int = 7
    max_steps: int | None = None
    limit_train: int | None = None
    limit_eval: int | None = None


class PromptRouteDataset(Dataset):
    def __init__(self, rows: list[RouterExample], tokenizer: Any, max_length: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        encoded = self.tokenizer(
            row.text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(LABEL_TO_ID[row.route], dtype=torch.long)
        return item


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available() and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)


def resolve_device(name: str = "mps") -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested, but PyTorch cannot see a CUDA device.")
        return torch.device("cuda")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS was requested, but PyTorch cannot see an MPS device.")
        return torch.device("mps")
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError("Device must be one of: cuda, mps, cpu, auto")


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def load_tokenizer(model_name_or_dir: str | Path):
    tokenizer = AutoTokenizer.from_pretrained(str(model_name_or_dir))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    return tokenizer


def load_sequence_classifier(model_name: str):
    kwargs: dict[str, Any] = {
        "num_labels": len(LABELS),
        "id2label": ID_TO_LABEL,
        "label2id": LABEL_TO_ID,
    }
    try:
        return AutoModelForSequenceClassification.from_pretrained(
            model_name,
            attn_implementation="eager",
            **kwargs,
        )
    except TypeError:
        return AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)


def make_loader(
    rows: list[RouterExample],
    tokenizer: Any,
    *,
    max_length: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = PromptRouteDataset(rows, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_loaded_model(
    model: torch.nn.Module,
    rows: list[RouterExample],
    tokenizer: Any,
    *,
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> dict[str, Any]:
    loader = make_loader(
        rows,
        tokenizer,
        max_length=max_length,
        batch_size=batch_size,
        shuffle=False,
    )
    model.eval()
    losses: list[float] = []
    gold: list[int] = []
    predicted: list[int] = []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            labels = batch["labels"]
            outputs = model(**batch)
            losses.append(float(outputs.loss.detach().cpu()))
            logits = outputs.logits.detach().cpu()
            predicted.extend(logits.argmax(dim=-1).tolist())
            gold.extend(labels.detach().cpu().tolist())

    report = classification_report(
        gold,
        predicted,
        labels=list(ID_TO_LABEL.keys()),
        target_names=list(LABELS),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(gold, predicted, labels=list(ID_TO_LABEL.keys())).tolist()
    return {
        "rows": len(rows),
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": accuracy_score(gold, predicted) if gold else 0.0,
        "macro_f1": f1_score(gold, predicted, labels=list(ID_TO_LABEL.keys()), average="macro")
        if gold
        else 0.0,
        "classification_report": report,
        "confusion_matrix": matrix,
    }


def save_label_map(out_dir: Path) -> None:
    payload = {
        "labels": list(LABELS),
        "label_to_id": LABEL_TO_ID,
        "id_to_label": {str(key): value for key, value in ID_TO_LABEL.items()},
    }
    (out_dir / "label_map.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_max_length(model_dir: Path, fallback: int) -> int:
    config_path = model_dir / "training_config.json"
    if not config_path.exists():
        return fallback
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return int(payload.get("max_length") or fallback)


def train_classifier(config: TrainConfig) -> dict[str, Any]:
    set_seed(config.seed)
    train_rows = read_jsonl(Path(config.train_path))
    eval_rows = read_jsonl(Path(config.eval_path))
    if config.limit_train is not None:
        train_rows = train_rows[: config.limit_train]
    if config.limit_eval is not None:
        eval_rows = eval_rows[: config.limit_eval]
    if not train_rows:
        raise ValueError("Training set is empty.")
    if not eval_rows:
        raise ValueError("Eval set is empty.")

    device = resolve_device(config.device)
    tokenizer = load_tokenizer(config.model_name)
    model = load_sequence_classifier(config.model_name)
    model.to(device)

    train_loader = make_loader(
        train_rows,
        tokenizer,
        max_length=config.max_length,
        batch_size=config.batch_size,
        shuffle=True,
    )
    total_steps = config.epochs * len(train_loader)
    if config.max_steps is not None:
        total_steps = min(total_steps, config.max_steps)
    warmup_steps = int(total_steps * config.warmup_ratio)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(total_steps, 1),
    )

    history: list[dict[str, Any]] = []
    global_step = 0
    started = perf_counter()
    stop = False

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        running_examples = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{config.epochs}", leave=False)
        for batch in progress:
            batch = move_batch(batch, device)
            labels = batch["labels"]

            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            global_step += 1
            running_loss += float(loss.detach().cpu()) * labels.numel()
            running_examples += labels.numel()
            progress.set_postfix(loss=running_loss / max(running_examples, 1))

            if config.max_steps is not None and global_step >= config.max_steps:
                stop = True
                break

        eval_metrics = evaluate_loaded_model(
            model,
            eval_rows,
            tokenizer,
            device=device,
            max_length=config.max_length,
            batch_size=config.batch_size,
        )
        epoch_metrics = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "train_loss": running_loss / max(running_examples, 1),
            "eval": {
                "loss": eval_metrics["loss"],
                "accuracy": eval_metrics["accuracy"],
                "macro_f1": eval_metrics["macro_f1"],
            },
        }
        history.append(epoch_metrics)
        if stop:
            break

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    save_label_map(out_dir)

    metrics = {
        "device": str(device),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_seconds": perf_counter() - started,
        "global_step": global_step,
        "history": history,
    }
    (out_dir / "training_config.json").write_text(
        json.dumps(asdict(config), indent=2),
        encoding="utf-8",
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def load_trained_model(model_dir: Path, device: torch.device):
    tokenizer = load_tokenizer(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return model, tokenizer


def evaluate_model_dir(
    *,
    model_dir: Path,
    data_path: Path,
    device_name: str = "mps",
    max_length: int | None = None,
    batch_size: int = 8,
) -> dict[str, Any]:
    rows = read_jsonl(data_path)
    device = resolve_device(device_name)
    model, tokenizer = load_trained_model(model_dir, device)
    resolved_max_length = max_length or load_max_length(model_dir, 1024)
    metrics = evaluate_loaded_model(
        model,
        rows,
        tokenizer,
        device=device,
        max_length=resolved_max_length,
        batch_size=batch_size,
    )
    metrics["model_dir"] = str(model_dir)
    metrics["data_path"] = str(data_path)
    metrics["device"] = str(device)
    return metrics


def format_prediction(
    *,
    probabilities: list[float],
    model_dir: Path,
    input_text: str,
) -> dict[str, Any]:
    ranked = sorted(
        zip(LABELS, probabilities, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    return {
        "selected_route": ranked[0][0],
        "confidence": ranked[0][1],
        "top_routes": [
            {"route": route, "probability": probability}
            for route, probability in ranked
        ],
        "model_dir": str(model_dir),
        "input_chars": len(input_text),
    }


def predict_text(
    *,
    model_dir: Path,
    text: str,
    device_name: str = "mps",
    max_length: int | None = None,
) -> dict[str, Any]:
    if not text.strip():
        raise ValueError("Prediction text cannot be empty.")
    device = resolve_device(device_name)
    model, tokenizer = load_trained_model(model_dir, device)
    resolved_max_length = max_length or load_max_length(model_dir, 1024)
    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=resolved_max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded).logits
        probabilities = logits.softmax(dim=-1).squeeze(0).detach().cpu().tolist()
    return format_prediction(probabilities=probabilities, model_dir=model_dir, input_text=text)
