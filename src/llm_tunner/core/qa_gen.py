"""Synthetic Q&A generation from document chunks for fine-tuning.

Each chunk is turned into one or more grounded question/answer pairs by prompting an
LLM (the same in-process :class:`ChatModel`, or a user-supplied generator callable).
The output feeds :mod:`core.dataset` to build a TRL conversational dataset.

A deterministic ``heuristic`` generator is provided as a no-LLM fallback so the
pipeline (and its tests) run without loading weights.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass

from .chunking import Chunk

# A generator maps a chunk's text -> list of (question, answer) tuples.
QAGenerator = Callable[[str], list[tuple[str, str]]]


@dataclass
class QAPair:
    question: str
    answer: str
    source: str
    page_number: int


_PROMPT = (
    "You are creating training data. Read the passage and write {n} distinct, "
    "self-contained question/answer pairs that can be answered using ONLY the passage. "
    "Return a JSON list of objects with keys 'question' and 'answer'. No prose.\n\n"
    "Passage:\n{passage}"
)


def llm_generator(chat_model, n: int = 2) -> QAGenerator:
    """Build a generator backed by a loaded :class:`~core.inference.ChatModel`."""

    def _gen(passage: str) -> list[tuple[str, str]]:
        messages = [{"role": "user", "content": _PROMPT.format(n=n, passage=passage)}]
        raw = chat_model.generate(messages)
        return _parse_pairs(raw)

    return _gen


def heuristic_generator(passage: str) -> list[tuple[str, str]]:
    """No-LLM fallback: turn the first sentence into a trivial Q&A.

    Useful for smoke tests and for users without a working generator model.
    """
    sentence = re.split(r"(?<=[.!?])\s+", passage.strip())[0] if passage.strip() else ""
    if not sentence:
        return []
    q = "What does this passage describe?"
    return [(q, sentence)]


def _parse_pairs(raw: str) -> list[tuple[str, str]]:
    """Best-effort extraction of [{question, answer}] from an LLM response."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\[", raw):
        try:
            data, _end = decoder.raw_decode(raw[match.start() :])
        except json.JSONDecodeError:
            continue
        if not isinstance(data, list):
            continue

        pairs: list[tuple[str, str]] = []
        for item in data:
            if isinstance(item, dict) and item.get("question") and item.get("answer"):
                pairs.append((str(item["question"]).strip(), str(item["answer"]).strip()))
        if pairs:
            return pairs
    return []


def generate_qa(
    chunks: list[Chunk],
    generator: QAGenerator | None = None,
    progress=None,
    should_stop: Callable[[], bool] | None = None,
) -> list[QAPair]:
    """Generate Q&A pairs for every chunk using ``generator`` (default: heuristic)."""
    generator = generator or heuristic_generator
    pairs: list[QAPair] = []
    total = len(chunks)
    for i, chunk in enumerate(chunks, start=1):
        if should_stop and should_stop():
            break
        for q, a in generator(chunk.text):
            pairs.append(
                QAPair(question=q, answer=a, source=chunk.source, page_number=chunk.page_number)
            )
        if should_stop and should_stop():
            break
        if progress:
            progress(i, total, f"Generated Q&A for {i}/{total} chunks")
    return pairs
