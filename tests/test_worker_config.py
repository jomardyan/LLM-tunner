"""Worker-layer config builders. Torch-free: tasks.py keeps heavy imports lazy."""

from __future__ import annotations

from llm_tunner.workers.tasks import _generation_config, _train_config


def test_generation_config_disables_sampling_at_zero_temperature():
    greedy = _generation_config(0.0, 0.9, 128)
    assert greedy.do_sample is False
    assert greedy.max_new_tokens == 128

    sampled = _generation_config(0.7, 0.95, 256)
    assert sampled.do_sample is True
    assert sampled.temperature == 0.7
    assert sampled.top_p == 0.95


def test_train_config_applies_overrides_and_keeps_defaults():
    cfg = _train_config(
        {"learning_rate": 1e-3, "num_train_epochs": 2, "lora_r": 8, "lora_dropout": 0.1}
    )
    assert cfg.learning_rate == 1e-3
    assert cfg.num_train_epochs == 2.0
    assert cfg.lora.r == 8
    assert cfg.lora.lora_dropout == 0.1
    # Untouched fields keep their TrainConfig/LoraConfig defaults.
    assert cfg.per_device_train_batch_size == 1
    assert cfg.gradient_accumulation_steps == 8
    assert cfg.max_seq_length == 1024
    assert cfg.lora.lora_alpha == 32


def test_train_config_none_is_all_defaults():
    cfg = _train_config(None)
    assert cfg.learning_rate == 2e-4
    assert cfg.num_train_epochs == 1.0
    assert cfg.lora.r == 16
