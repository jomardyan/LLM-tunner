from __future__ import annotations

from llm_tunner.settings import Settings, _MemoryBackend


def _mem_settings(monkeypatch) -> Settings:
    """A Settings backed by the in-memory dict (no Qt/registry side effects)."""
    monkeypatch.setattr("llm_tunner.settings._make_backend", lambda: _MemoryBackend())
    return Settings()


def test_generation_settings_round_trip(monkeypatch):
    s = _mem_settings(monkeypatch)
    # Defaults.
    assert s.temperature == 0.7
    assert s.top_p == 0.9
    assert s.max_new_tokens == 512
    assert s.system_prompt == ""
    # Round-trip.
    s.temperature = 0.25
    s.top_p = 0.8
    s.max_new_tokens = 256
    s.system_prompt = "be terse"
    assert s.temperature == 0.25
    assert s.top_p == 0.8
    assert s.max_new_tokens == 256
    assert s.system_prompt == "be terse"


def test_retrieval_and_chunk_settings_round_trip(monkeypatch):
    s = _mem_settings(monkeypatch)
    assert s.score_threshold == 0.0
    assert s.chunk_size == 512
    assert s.chunk_overlap == 64
    s.score_threshold = 0.3
    s.chunk_size = 1024
    s.chunk_overlap = 128
    assert s.score_threshold == 0.3
    assert s.chunk_size == 1024
    assert s.chunk_overlap == 128


def test_onnx_provider_setting_round_trip(monkeypatch):
    s = _mem_settings(monkeypatch)
    assert s.onnx_provider == "auto"
    s.onnx_provider = "openvino-multi"
    assert s.onnx_provider == "openvino-multi"


def test_storage_settings_round_trip(monkeypatch):
    s = _mem_settings(monkeypatch)
    assert s.data_dir == ""
    assert s.last_dataset == ""
    assert s.last_adapter == ""
    s.data_dir = "/data/llm"
    s.last_dataset = "/data/llm/datasets/finetune_dataset_x.jsonl"
    s.last_adapter = "/data/llm/adapters/pdf_adapter_x"
    assert s.data_dir == "/data/llm"
    assert s.last_dataset.endswith("finetune_dataset_x.jsonl")
    assert s.last_adapter.endswith("pdf_adapter_x")


def test_train_hyperparameter_settings_round_trip(monkeypatch):
    s = _mem_settings(monkeypatch)
    assert s.learning_rate == 2e-4
    assert s.lora_r == 16
    s.learning_rate = 1e-3
    s.num_train_epochs = 2.0
    s.lora_r = 8
    s.lora_alpha = 16
    s.lora_dropout = 0.1
    assert s.learning_rate == 1e-3
    assert s.num_train_epochs == 2.0
    assert s.lora_r == 8
    assert s.lora_alpha == 16
    assert s.lora_dropout == 0.1


def test_typed_getters_tolerate_string_round_trips(monkeypatch):
    # QSettings commonly returns values as strings; getters must coerce.
    s = _mem_settings(monkeypatch)
    s.set("temperature", "0.4")
    s.set("max_new_tokens", "128")
    assert s.temperature == 0.4
    assert s.max_new_tokens == 128


def test_typed_getters_fall_back_on_invalid_values(monkeypatch):
    s = _mem_settings(monkeypatch)
    s.set("temperature", "not-a-number")
    s.set("lora_r", "garbage")
    assert s.temperature == 0.7  # default fallback, not a crash
    assert s.lora_r == 16
