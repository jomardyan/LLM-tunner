"""QLoRA/LoRA training smoke test on a tiny model. Skipped unless trl+peft+torch present.

Runs a couple of steps on sshleifer/tiny-gpt2 to assert the metric callback fires and a
LoRA adapter is written. Falls back to plain LoRA when no CUDA/bitsandbytes (expected in CI).
"""

from __future__ import annotations

import importlib.util

import pytest

_HAVE_TRAIN = all(importlib.util.find_spec(m) for m in ("torch", "transformers", "trl", "peft"))

pytestmark = pytest.mark.skipif(not _HAVE_TRAIN, reason="training extras not installed")


def _tiny_rows():
    return [
        {"messages": [
            {"role": "user", "content": "Say hi."},
            {"role": "assistant", "content": "Hi there!"},
        ]}
        for _ in range(4)
    ]


def test_finetune_writes_adapter():
    from llm_tunner.config import LoraConfig, TrainConfig
    from llm_tunner.core.training import TrainHandle, adapter_exists, run_finetune

    handle = TrainHandle()
    config = TrainConfig(
        num_train_epochs=1.0,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        max_seq_length=64,
        logging_steps=1,
        use_4bit=True,  # auto-disabled off-CUDA
        lora=LoraConfig(r=4, lora_alpha=8, target_modules=("c_attn",)),  # GPT-2 module
    )
    completed = []
    result = run_finetune(
        "sshleifer/tiny-gpt2",
        _tiny_rows(),
        config=config,
        output_name="smoke",
        handle=handle,
        completion_callback=completed.append,
    )
    assert adapter_exists("smoke")
    assert result.steps >= 1
    assert completed == [result]
    # The metric queue should have received at least one logged step.
    assert not handle.metrics.empty()
