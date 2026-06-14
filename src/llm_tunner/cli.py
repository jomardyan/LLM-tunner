"""Headless command-line interface for LLM-tunner (Ubuntu/Linux, macOS, Windows).

Runs the same `core`/`workers` pipelines as the GUI without Qt: build a knowledge base,
chat (RAG or plain), generate a fine-tuning dataset, fine-tune, and inspect storage.
Progress and logs go to stderr so stdout carries only the result (answers, paths) and
stays pipe-friendly.

Entry point: ``llm-tunner-cli`` (see pyproject ``[project.scripts]``).
"""

from __future__ import annotations

import argparse
import os
import sys


class _Emit:
    """A WorkerSignals-style channel: ``.emit(*args)`` calls the bound function."""

    def __init__(self, fn) -> None:
        self.emit = fn


class ConsoleSignals:
    """Stand-in for ``WorkerSignals`` that prints progress/log/metric to stderr.

    The task functions in ``workers.tasks`` only need ``signals.progress/log/metric.emit``;
    keeping output on stderr leaves stdout clean for the actual result.
    """

    def __init__(self, quiet: bool = False) -> None:
        self.quiet = quiet
        self.progress = _Emit(self._progress)
        self.log = _Emit(self._log)
        self.metric = _Emit(self._metric)

    def _err(self, text: str) -> None:
        print(text, file=sys.stderr, flush=True)

    def _log(self, message: str) -> None:
        if not self.quiet and message:
            self._err(message)

    def _progress(self, done: int, total: int, message: str = "") -> None:
        if self.quiet:
            return
        if total:
            self._err(f"  [{done}/{total}] {message}".rstrip())
        elif message:
            self._err(f"  {message}")

    def _metric(self, metric) -> None:
        if self.quiet:
            return
        loss = getattr(metric, "loss", None)
        step = getattr(metric, "step", "?")
        if loss is not None:
            self._err(f"  step {step}: loss={loss:.4f}")


# -- commands --------------------------------------------------------------------------


def cmd_devices(args) -> int:
    from .core.device import detect_device, recommended_models

    info = detect_device()
    print(info.capability_banner())
    print("Recommended models:")
    for model in recommended_models(info):
        print(f"  {model}")
    return 0


def cmd_list(args) -> int:
    from .config import adapters_dir, datasets_dir, kb_dir
    from .core.training import adapter_exists

    def show(title: str, items: list[str]) -> None:
        print(f"{title}:")
        for item in items or ["(none)"]:
            print(f"  {item}")

    show("Knowledge bases", sorted(p.name for p in kb_dir().iterdir() if p.is_dir()))
    show("Datasets", sorted(p.name for p in datasets_dir().glob("*.jsonl")))
    show(
        "Adapters",
        sorted(p.name for p in adapters_dir().iterdir() if p.is_dir() and adapter_exists(p.name)),
    )
    print(f"\nData directory: {kb_dir().parent}")
    return 0


def cmd_build_kb(args) -> int:
    from .workers.tasks import build_kb_task

    result = build_kb_task(
        args.name,
        args.pdf,
        args.embedding_model,
        signals=ConsoleSignals(args.quiet),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        onnx_provider=args.provider,
    )
    print(
        f"Knowledge base '{result['kb']}': {result['chunks']} chunks "
        f"({result['count']} total) from {result['documents']} document(s) "
        f"in {result['elapsed_seconds']:.1f}s"
    )
    return 0


def cmd_gen_dataset(args) -> int:
    from .workers.tasks import generate_dataset_task

    model = args.model if args.use_llm else None
    result = generate_dataset_task(args.pdf, model, signals=ConsoleSignals(args.quiet))
    if result.get("cancelled"):
        print("Cancelled.", file=sys.stderr)
        return 1
    print(
        f"Dataset: {result['pairs']} pairs from {result['chunks']} chunks "
        f"-> {result['path']}"
    )
    return 0


def cmd_chat(args) -> int:
    signals = ConsoleSignals(args.quiet)
    if args.no_rag or not args.kb:
        from .workers.tasks import plain_chat_task

        result = plain_chat_task(
            args.model,
            args.adapter,
            args.query,
            [],
            signals=signals,
            system_prompt=args.system_prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        from .workers.tasks import rag_query_task

        result = rag_query_task(
            args.model,
            args.adapter,
            args.kb,
            args.query,
            args.embedding_model,
            args.top_k,
            signals=signals,
            system_prompt=args.system_prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            score_threshold=args.score_threshold,
            onnx_provider=args.provider,
        )
    print(result["answer"])
    if result.get("sources"):
        print(f"\nSources: {result['sources']}", file=sys.stderr)
    return 0


def cmd_finetune(args) -> int:
    from .config import timestamped_name
    from .core.dataset import load_jsonl
    from .core.training import TrainHandle
    from .workers.tasks import finetune_task

    rows = load_jsonl(args.dataset)
    if not rows:
        print(f"error: dataset {args.dataset} is empty", file=sys.stderr)
        return 1
    overrides = {
        key: value
        for key, value in (
            ("learning_rate", args.learning_rate),
            ("num_train_epochs", args.epochs),
            ("per_device_train_batch_size", args.batch_size),
            ("gradient_accumulation_steps", args.grad_accum),
            ("max_seq_length", args.max_seq_length),
            ("lora_r", args.lora_r),
            ("lora_alpha", args.lora_alpha),
            ("lora_dropout", args.lora_dropout),
        )
        if value is not None
    }
    output_name = args.output or timestamped_name("adapter")
    result = finetune_task(
        args.model,
        rows,
        output_name,
        TrainHandle(),
        signals=ConsoleSignals(args.quiet),
        train_overrides=overrides,
    )
    loss = result.get("final_loss")
    print(f"Adapter saved: {result['adapter_path']}")
    print(
        f"steps={result['steps']} "
        f"final_loss={'N/A' if loss is None else round(loss, 4)} "
        f"used_4bit={result['used_4bit']}"
    )
    return 0


# -- parser ----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    from .config import DEFAULT_BASE_MODEL, DEFAULT_EMBEDDING_MODEL

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--data-dir", help="Override the data directory (sets LLM_TUNNER_HOME for this run)."
    )
    common.add_argument("--quiet", action="store_true", help="Suppress progress/log output.")

    parser = argparse.ArgumentParser(
        prog="llm-tunner-cli",
        description="Headless CLI for LLM-tunner (RAG + QLoRA fine-tuning over PDFs).",
        parents=[common],
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("devices", parents=[common], help="Show detected hardware and capabilities.").set_defaults(
        func=cmd_devices
    )
    sub.add_parser(
        "list", parents=[common], help="List knowledge bases, datasets and adapters on disk."
    ).set_defaults(func=cmd_list)

    p = sub.add_parser("build-kb", parents=[common], help="Build/update a knowledge base from PDFs.")
    p.add_argument("--name", required=True, help="Knowledge-base name.")
    p.add_argument("--pdf", nargs="+", required=True, help="One or more PDF paths.")
    p.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    p.add_argument("--chunk-size", type=int, default=512)
    p.add_argument("--chunk-overlap", type=int, default=64)
    p.add_argument(
        "--provider",
        default="auto",
        help="Embedding ONNX provider: auto|cpu|directml|openvino|openvino-multi|cuda.",
    )
    p.set_defaults(func=cmd_build_kb)

    p = sub.add_parser(
        "gen-dataset", parents=[common], help="Synthesize a fine-tuning Q&A dataset from PDFs."
    )
    p.add_argument("--pdf", nargs="+", required=True)
    p.add_argument("--use-llm", action="store_true", help="Use the base model for higher quality.")
    p.add_argument("--model", default=DEFAULT_BASE_MODEL)
    p.set_defaults(func=cmd_gen_dataset)

    p = sub.add_parser("chat", parents=[common], help="Ask a question (RAG when --kb is given).")
    p.add_argument("--query", required=True)
    p.add_argument("--kb", help="Knowledge base to ground the answer in.")
    p.add_argument("--no-rag", action="store_true", help="Force plain chat (ignore --kb).")
    p.add_argument("--model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--adapter", default=None, help="Path to a LoRA adapter directory.")
    p.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--score-threshold", type=float, default=0.0)
    p.add_argument("--provider", default="auto")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--system-prompt", default="")
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("finetune", parents=[common], help="Fine-tune (QLoRA/LoRA) on a dataset.")
    p.add_argument("--dataset", required=True, help="Path to a dataset .jsonl (from gen-dataset).")
    p.add_argument("--model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--output", default=None, help="Adapter name (default: timestamped).")
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--epochs", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--max-seq-length", type=int, default=None)
    p.add_argument("--lora-r", type=int, default=None)
    p.add_argument("--lora-alpha", type=int, default=None)
    p.add_argument("--lora-dropout", type=float, default=None)
    p.set_defaults(func=cmd_finetune)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "data_dir", None):
        os.environ["LLM_TUNNER_HOME"] = args.data_dir
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 1
    try:
        return func(args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - print a clean message, not a traceback
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
