"""Build a TRL-compatible conversational dataset from Q&A pairs.

TRL's ``SFTTrainer`` consumes the "messages" (conversational) format and applies the
model's chat template automatically, so we emit rows of the form::

    {"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}

The dataset is persisted as JSONL (for inspection/reuse) and can be loaded into a
``datasets.Dataset`` for training.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import datasets_dir
from .qa_gen import QAPair


def to_messages(pairs: list[QAPair], system_prompt: str | None = None) -> list[dict]:
    """Convert Q&A pairs into conversational rows."""
    rows: list[dict] = []
    for p in pairs:
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": p.question})
        messages.append({"role": "assistant", "content": p.answer})
        rows.append({"messages": messages})
    return rows


def save_jsonl(rows: list[dict], name: str) -> Path:
    """Persist rows as JSONL under the datasets directory. Returns the file path."""
    path = datasets_dir() / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def load_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def to_hf_dataset(rows: list[dict]):
    """Build a ``datasets.Dataset`` from conversational rows (lazy import)."""
    from datasets import Dataset  # type: ignore

    return Dataset.from_list(rows)
