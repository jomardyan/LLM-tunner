"""QLoRA / LoRA fine-tuning via TRL's SFTTrainer.

Key pieces:

* :class:`MetricCallback` — a ``TrainerCallback`` whose ``on_log`` hook pushes per-step
  metrics onto a thread-safe queue and checks a stop flag (wired to the GUI's Stop
  button via :meth:`request_stop`). This is how live loss curves reach the UI without
  the worker touching widgets.
* :func:`run_finetune` — assembles model + LoraConfig + (optional) 4-bit
  BitsAndBytesConfig + SFTTrainer and trains, saving a small LoRA adapter.

4-bit QLoRA is enabled only when CUDA + bitsandbytes are present; otherwise we fall
back to plain LoRA (fp32/bf16) and warn, since bitsandbytes is CUDA-only.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..config import TrainConfig, adapters_dir
from ..diagnostics import configure_ml_output
from .device import detect_device

_FALLBACK_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ message['role'] | capitalize }}: {{ message['content'] }}\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}Assistant: {% endif %}"
)


@dataclass
class TrainMetric:
    step: int
    loss: float | None
    learning_rate: float | None
    epoch: float | None
    grad_norm: float | None = None


@dataclass
class TrainResult:
    adapter_path: str
    final_loss: float | None
    steps: int
    used_4bit: bool


def _make_callback_class():
    """Define MetricCallback lazily so importing this module doesn't require transformers."""
    from transformers import TrainerCallback  # type: ignore

    class MetricCallback(TrainerCallback):
        """Streams metrics out of the Trainer and supports cooperative cancellation."""

        def __init__(self, sink: queue.Queue[TrainMetric], stop_event: threading.Event) -> None:
            self.sink = sink
            self.stop_event = stop_event

        def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: D102
            logs = logs or {}
            if "loss" in logs or "learning_rate" in logs:
                self.sink.put(
                    TrainMetric(
                        step=int(state.global_step),
                        loss=logs.get("loss"),
                        learning_rate=logs.get("learning_rate"),
                        epoch=logs.get("epoch"),
                        grad_norm=logs.get("grad_norm"),
                    )
                )
            return control

        def on_step_end(self, args, state, control, **kwargs):  # noqa: D102
            if self.stop_event.is_set():
                control.should_training_stop = True
            return control

    return MetricCallback


@dataclass
class TrainHandle:
    """Lets the GUI worker drain metrics and request cancellation while training runs."""

    metrics: queue.Queue[TrainMetric] = field(default_factory=queue.Queue)
    stop_event: threading.Event = field(default_factory=threading.Event)

    def request_stop(self) -> None:
        self.stop_event.set()


def _bnb_config(use_4bit: bool):
    import torch  # type: ignore
    from transformers import BitsAndBytesConfig  # type: ignore

    if not use_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def run_finetune(
    model_id: str,
    dataset_rows: list[dict],
    config: TrainConfig | None = None,
    output_name: str = "adapter",
    handle: TrainHandle | None = None,
    completion_callback: Callable[[TrainResult], None] | None = None,
) -> TrainResult:
    """Fine-tune ``model_id`` on conversational ``dataset_rows`` and save a LoRA adapter.

    Args:
        dataset_rows: rows in TRL "messages" format (see :mod:`core.dataset`).
        handle: optional :class:`TrainHandle` to stream metrics / cancel.
    """
    if not dataset_rows:
        raise ValueError("Cannot fine-tune on an empty dataset.")
    import torch  # type: ignore
    from peft import LoraConfig as PeftLoraConfig  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    from trl import SFTConfig, SFTTrainer  # type: ignore

    from .dataset import to_hf_dataset
    from .inference import ensure_pad_token

    configure_ml_output()
    config = config or TrainConfig()
    handle = handle or TrainHandle()
    info = detect_device()

    # 4-bit QLoRA only works on CUDA + bitsandbytes; otherwise degrade to plain LoRA.
    use_4bit = config.use_4bit and info.supports_qlora
    quant_config = _bnb_config(use_4bit)

    dtype = torch.bfloat16 if info.kind == "cuda" else torch.float32
    device_map = "auto" if info.kind == "cuda" else None

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    ensure_pad_token(tokenizer)
    if not tokenizer.chat_template:
        tokenizer.chat_template = _FALLBACK_CHAT_TEMPLATE

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        dtype=dtype,
        device_map=device_map,
    )
    if device_map is None:
        model = model.to("mps" if info.kind == "mps" else "cpu")

    peft_config = PeftLoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        target_modules=list(config.lora.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )

    out_dir = adapters_dir() / output_name
    sft_config = SFTConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        logging_steps=config.logging_steps,
        max_length=config.max_seq_length,
        use_cpu=info.kind == "cpu",
        bf16=info.kind == "cuda" and torch.cuda.is_bf16_supported(),
        fp16=False,
        loss_type="chunked_nll",
        report_to=[],
        save_strategy="no",
    )

    MetricCallback = _make_callback_class()
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=to_hf_dataset(dataset_rows),
        peft_config=peft_config,
        processing_class=tokenizer,
        callbacks=[MetricCallback(handle.metrics, handle.stop_event)],
    )

    train_output = trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    final_loss = getattr(train_output, "training_loss", None)
    steps = int(getattr(train_output, "global_step", 0)) or trainer.state.global_step
    result = TrainResult(
        adapter_path=str(out_dir),
        final_loss=final_loss,
        steps=steps,
        used_4bit=use_4bit,
    )
    if completion_callback is not None:
        completion_callback(result)
    return result


def adapter_exists(name: str) -> bool:
    p = Path(adapters_dir()) / name
    return (p / "adapter_config.json").exists() or (p / "adapter_model.safetensors").exists()
