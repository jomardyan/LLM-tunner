from __future__ import annotations

import sys
import types

from llm_tunner.core.chunking import Chunk
from llm_tunner.core.rag import (
    RagIndex,
    Retrieved,
    _chunk_id,
    _collection_name,
    _recent_turns,
    _storage_name,
    missing_rag_dependencies,
    onnx_providers,
    rag_install_command,
    require_rag_dependencies,
)


class FakeCollection:
    def __init__(self, existing_ids: dict[str, list[str]] | None = None) -> None:
        self.existing_ids = existing_ids or {}
        self.upserts: list[dict] = []
        self.deleted: list[str] = []

    def get(self, *, where, include):
        return {"ids": self.existing_ids.get(where["source"], [])}

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)

    def delete(self, *, ids):
        self.deleted.extend(ids)


class FakeEmbedder:
    def encode(self, texts):
        return [[float(len(text))] for text in texts]


def _chunk(source: str, index: int, text: str = "text") -> Chunk:
    return Chunk(text=text, source=source, page_number=1, index=index)


def test_collection_name_handles_short_names():
    assert len(_collection_name("a")) >= 3
    assert _collection_name("---")[0].isalnum()
    assert _collection_name("---")[-1].isalnum()


def test_storage_name_is_windows_safe_and_bounded():
    assert _storage_name("CON") == "kb-con"
    first = _storage_name("a" * 100)
    second = _storage_name("a" * 99 + "b")
    assert len(first) <= 64
    assert first != second


def test_rag_dependency_error_lists_missing_packages(monkeypatch):
    monkeypatch.setattr(
        "llm_tunner.core.rag.importlib.util.find_spec",
        lambda module: None if module in {"chromadb", "onnxruntime"} else object(),
    )
    assert missing_rag_dependencies() == ["chromadb", "onnxruntime"]

    import pytest

    with pytest.raises(RuntimeError, match=r'pip install -e "\.\[rag\]"'):
        require_rag_dependencies()
    assert "python" in rag_install_command().lower()


def test_chunk_ids_do_not_collide_for_same_basename():
    first = _chunk("/one/report.pdf", 0)
    second = _chunk("/two/report.pdf", 0)
    assert _chunk_id(first) != _chunk_id(second)


def test_reindex_upserts_and_removes_stale_chunks(monkeypatch):
    source = "/docs/report.pdf"
    chunks = [_chunk(source, 0, "new zero"), _chunk(source, 1, "new one")]
    stale = _chunk(source, 2, "old")
    collection = FakeCollection(
        existing_ids={source: [_chunk_id(chunks[0]), _chunk_id(stale)]}
    )
    index = RagIndex.__new__(RagIndex)
    index._embedder = FakeEmbedder()
    monkeypatch.setattr(index, "_coll", lambda: collection)

    assert index.add_chunks(chunks) == 2
    assert collection.upserts[0]["ids"] == [_chunk_id(c) for c in chunks]
    assert collection.deleted == [_chunk_id(stale)]


def test_recent_turns_caps_and_starts_on_user_message():
    history = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
        {"role": "assistant", "content": "a3"},
    ]
    recent = _recent_turns(history, max_turns=2)
    assert len(recent) == 4
    assert recent[0] == {"role": "user", "content": "u2"}
    assert _recent_turns(None, 3) == []
    assert _recent_turns(history, 0) == []


def test_build_prompt_includes_history_and_system_prefix():
    index = RagIndex.__new__(RagIndex)  # build_prompt does not touch self.config
    ctx = Retrieved(text="Paris is the capital.", source="/d/f.pdf", page_number=1, score=0.9)
    history = [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
    ]
    messages = index.build_prompt("now?", [ctx], history=history, system_prefix="Be terse")

    assert messages[0]["role"] == "system"
    assert messages[0]["content"].startswith("Be terse")
    assert "Context:" in messages[0]["content"]
    assert {"role": "user", "content": "earlier question"} in messages
    assert messages[-1] == {"role": "user", "content": "now?"}


def test_build_prompt_backward_compatible_without_history():
    index = RagIndex.__new__(RagIndex)
    ctx = Retrieved(text="text", source="/d/f.pdf", page_number=1, score=0.5)
    messages = index.build_prompt("q", [ctx])
    assert [m["role"] for m in messages] == ["system", "user"]


def _fake_ort(monkeypatch, providers):
    """Inject a fake onnxruntime exposing get_available_providers (env-independent)."""
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        types.SimpleNamespace(get_available_providers=lambda: list(providers)),
    )


def test_onnx_providers_auto_prefers_accelerator_then_cpu(monkeypatch):
    _fake_ort(monkeypatch, ["DmlExecutionProvider", "CPUExecutionProvider"])
    assert onnx_providers("auto") == ["DmlExecutionProvider", "CPUExecutionProvider"]


def test_onnx_providers_auto_cpu_only(monkeypatch):
    _fake_ort(monkeypatch, ["CPUExecutionProvider"])
    assert onnx_providers("auto") == ["CPUExecutionProvider"]


def test_onnx_providers_auto_openvino_carries_device_type(monkeypatch):
    _fake_ort(monkeypatch, ["OpenVINOExecutionProvider", "CPUExecutionProvider"])
    providers = onnx_providers("auto")
    assert providers[0] == ("OpenVINOExecutionProvider", {"device_type": "AUTO"})
    assert providers[-1] == "CPUExecutionProvider"


def test_onnx_providers_openvino_multi_is_simultaneous_cpu_igpu(monkeypatch):
    _fake_ort(monkeypatch, ["OpenVINOExecutionProvider", "CPUExecutionProvider"])
    providers = onnx_providers("openvino-multi")
    assert providers[0] == ("OpenVINOExecutionProvider", {"device_type": "MULTI:GPU,CPU"})
    assert providers[-1] == "CPUExecutionProvider"


def test_onnx_providers_cpu_is_forced(monkeypatch):
    _fake_ort(monkeypatch, ["DmlExecutionProvider", "CPUExecutionProvider"])
    assert onnx_providers("cpu") == ["CPUExecutionProvider"]


def test_onnx_providers_unavailable_choice_falls_back_to_cpu(monkeypatch):
    # Asked for DirectML but only CPU is installed -> graceful CPU fallback, no error.
    _fake_ort(monkeypatch, ["CPUExecutionProvider"])
    assert onnx_providers("directml") == ["CPUExecutionProvider"]


def test_onnx_providers_no_onnxruntime_returns_cpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)  # import raises -> CPU floor
    assert onnx_providers("auto") == ["CPUExecutionProvider"]
