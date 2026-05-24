"""JSONL schema helpers for classifier examples."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from finance_router.labels import validate_route

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def stable_id(*parts: object) -> str:
    payload = "\n".join("" if part is None else str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


@dataclass(frozen=True)
class RouterExample:
    text: str
    route: str
    source: str
    id: str | None = None
    company: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        text = normalize_text(self.text)
        if not text:
            raise ValueError("RouterExample text cannot be empty")
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "route", validate_route(self.route))
        if self.id is None:
            object.__setattr__(
                self,
                "id",
                stable_id(self.source, self.route, self.company or "", text),
            )

    @property
    def group_key(self) -> str:
        value = self.metadata.get("group_key")
        if value:
            return str(value)
        return str(self.id)

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "route": self.route,
            "source": self.source,
            "company": self.company,
            "metadata": self.metadata,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> RouterExample:
        return cls(
            id=payload.get("id"),
            text=payload["text"],
            route=payload["route"],
            source=payload["source"],
            company=payload.get("company"),
            metadata=dict(payload.get("metadata") or {}),
        )


def read_jsonl(path: Path) -> list[RouterExample]:
    with path.open("r", encoding="utf-8") as handle:
        return [RouterExample.from_json(json.loads(line)) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[RouterExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_json(), ensure_ascii=False) + "\n")
