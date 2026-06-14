"""Q&A generation (heuristic) and dataset formatting — no ML deps required."""

from __future__ import annotations

from llm_tunner.core.chunking import Chunk
from llm_tunner.core.dataset import load_jsonl, save_jsonl, to_messages
from llm_tunner.core.qa_gen import _parse_pairs, generate_qa, heuristic_generator


def test_heuristic_generator():
    pairs = heuristic_generator("Paris is the capital of France. It is lovely.")
    assert pairs and pairs[0][1].startswith("Paris is the capital")


def test_generate_and_format(tmp_path):
    chunks = [
        Chunk(text="Paris is the capital of France.", source="a.pdf", page_number=1, index=0),
        Chunk(text="The Eiffel Tower is in Paris.", source="a.pdf", page_number=2, index=1),
    ]
    pairs = generate_qa(chunks, generator=heuristic_generator)
    assert len(pairs) == 2

    rows = to_messages(pairs, system_prompt="You are helpful.")
    assert rows[0]["messages"][0]["role"] == "system"
    assert rows[0]["messages"][1]["role"] == "user"
    assert rows[0]["messages"][2]["role"] == "assistant"

    path = save_jsonl(rows, "test_ds")
    assert load_jsonl(path) == rows


class _NullSignals:
    """A WorkerSignals shim that drops all emits (for calling task functions in tests)."""

    class _Channel:
        def emit(self, *args, **kwargs) -> None:
            pass

    def __init__(self) -> None:
        self.progress = self._Channel()
        self.log = self._Channel()
        self.metric = self._Channel()


def test_generate_dataset_task_writes_unique_files(sample_pdf):
    # The heuristic generator (use_llm_model=None) keeps this torch-free.
    from llm_tunner.workers.tasks import generate_dataset_task

    first = generate_dataset_task([str(sample_pdf)], None, signals=_NullSignals())
    second = generate_dataset_task([str(sample_pdf)], None, signals=_NullSignals())

    assert first["pairs"] >= 1
    assert "finetune_dataset_" in first["path"]
    # Each generation is preserved on disk (no overwrite), even back-to-back.
    assert first["path"] != second["path"]


def test_parse_pairs_ignores_unrelated_brackets():
    raw = (
        "Use only the passage [important].\n"
        '[{"question": "Capital?", "answer": "Paris"}]\n'
        "Done."
    )
    assert _parse_pairs(raw) == [("Capital?", "Paris")]


def test_generate_qa_stops_between_chunks():
    chunks = [
        Chunk(text=f"Chunk {index}.", source="a.pdf", page_number=1, index=index)
        for index in range(4)
    ]
    calls = 0

    def should_stop():
        return calls >= 2

    def generator(text):
        nonlocal calls
        calls += 1
        return [("Question?", text)]

    pairs = generate_qa(chunks, generator=generator, should_stop=should_stop)

    assert calls == 2
    assert len(pairs) == 2
