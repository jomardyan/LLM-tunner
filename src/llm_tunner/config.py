"""Application-wide defaults and the user-data directory layout.

These are *defaults*; values the user changes are persisted via :mod:`llm_tunner.settings`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

APP_NAME = "LLM-tunner"
ORG_NAME = "LLM-tunner"


def data_dir() -> Path:
    """Per-user data directory (knowledge bases, adapters, logs).

    Honours ``LLM_TUNNER_HOME`` for tests and portable installs.
    """
    override = os.environ.get("LLM_TUNNER_HOME")
    base = Path(override) if override else Path.home() / ".llm-tunner"
    base.mkdir(parents=True, exist_ok=True)
    return base


def kb_dir() -> Path:
    """Directory holding Chroma vector stores (one subdir per knowledge base)."""
    d = data_dir() / "knowledge_bases"
    d.mkdir(parents=True, exist_ok=True)
    return d


def adapters_dir() -> Path:
    """Directory holding saved LoRA adapters."""
    d = data_dir() / "adapters"
    d.mkdir(parents=True, exist_ok=True)
    return d


def datasets_dir() -> Path:
    """Directory holding generated fine-tuning datasets."""
    d = data_dir() / "datasets"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --- Model defaults ---
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SMOKE_TEST_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HIGH_QUALITY_EMBEDDING_MODEL = "BAAI/bge-m3"

# Curated picker list: (model_id, license, approx_size, notes)
MODEL_REGISTRY: list[tuple[str, str, str, str]] = [
    ("Qwen/Qwen2.5-7B-Instruct", "Apache-2.0", "7B", "Strong default; mature FT ecosystem"),
    ("Qwen/Qwen2.5-3B-Instruct", "Qwen Research", "3B", "Non-Apache license — verify terms"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "Apache-2.0", "1.5B", "Good low-VRAM choice"),
    ("Qwen/Qwen2.5-0.5B-Instruct", "Apache-2.0", "0.5B", "Smoke test / CPU"),
    ("microsoft/Phi-4-mini-instruct", "MIT", "3.8B", "Cleanest license; 128k context"),
    ("mistralai/Ministral-8B-Instruct-2410", "Apache-2.0", "8B", "Permissive"),
    ("HuggingFaceTB/SmolLM2-360M-Instruct", "Apache-2.0", "360M", "Tiny / fast tests"),
    ("sshleifer/tiny-gpt2", "MIT", "<1M", "CI 'does-it-run' only — no useful output"),
]


@dataclass
class ChunkConfig:
    """Text chunking parameters for RAG indexing and QA generation."""

    chunk_size: int = 512  # tokens (recursive splitter default per research)
    chunk_overlap: int = 64


@dataclass
class RagConfig:
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    top_k: int = 4
    chunk: ChunkConfig = field(default_factory=ChunkConfig)


@dataclass
class LoraConfig:
    """LoRA / QLoRA hyper-parameters (mirrors PEFT LoraConfig knobs we expose)."""

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # Sensible defaults for Llama/Qwen-style attention+MLP projections.
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


@dataclass
class TrainConfig:
    learning_rate: float = 2e-4  # ~10x full-FT rate, per TRL guidance for LoRA
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 1024
    logging_steps: int = 1  # stream loss every optimizer step
    use_4bit: bool = True  # auto-disabled when CUDA/bitsandbytes absent
    lora: LoraConfig = field(default_factory=LoraConfig)
