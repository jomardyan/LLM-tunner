# LLM-tunner

A **PySide6 desktop application** for customizing open-source LLMs with your own PDF
documents. Add PDFs, build a searchable knowledge base, chat with citations, and
(optionally) fine-tune a model on Q&A pairs synthesized from your documents.

---

## RAG vs. fine-tuning — read this first

There are two ways to "feed a PDF into a model", and they do different jobs. The
2025–2026 consensus (Microsoft, OpenAI, IBM, and the research literature) is:

| | **RAG** (default) | **Fine-tuning** |
|---|---|---|
| Best for | Injecting **facts / document knowledge** | Teaching **style, format, behaviour** |
| Freshness | Update by re-indexing; instant | Frozen at training time |
| Hallucination | Lower — answers grounded + cited | Higher for narrow facts |
| Cost / hardware | Light; runs on CPU | Heavy; QLoRA needs an NVIDIA GPU |
| In this app | **Knowledge (RAG)** tab | **Fine-tune** tab |

> If you want the model to *know* what's in your PDFs, use **RAG**. Fine-tuning is for
> making it *behave* a certain way. They stack — research shows the accuracy gains are
> additive — so you can do both.

This app is **RAG-first**: RAG is the recommended path, with fine-tuning available for
style/behaviour adaptation.

## Features

- **Documents** — add PDFs; text is extracted with [Docling](https://github.com/docling-project/docling) (layout/table-aware) or a pypdf fallback.
- **Knowledge (RAG)** — chunk → embed (sentence-transformers) → store in a local
  [ChromaDB](https://www.trychroma.com/) vector DB.
- **Chat** — ask questions; answers cite the source document and page.
- **Fine-tune** — synthesize Q&A pairs, then run **QLoRA/LoRA** (TRL + PEFT) with a
  **live loss curve**; a small LoRA adapter is saved and auto-loaded for chat.
- **Hardware auto-detect** — CUDA → MPS → CPU. QLoRA 4-bit is enabled only on CUDA
  (bitsandbytes is CUDA-only); elsewhere the app falls back gracefully and tells you.

## Architecture

```
PySide6 GUI (main thread)  ──submit()──▶  QThreadPool  ──▶  Worker (QRunnable)
        ▲                                                        │
        └────── WorkerSignals (progress / metric / log / result) ┘
Workers call core/: pdf · chunking · rag · qa_gen · dataset · training · inference
Device gating lives in core/device.py.
```

Long jobs never run on the GUI thread; workers stream updates back via Qt signals, so
the UI stays responsive and training loss is plotted live. Training metrics are pushed
from a `TrainerCallback.on_log` hook onto a queue that the worker drains to the plot.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"        # GUI + RAG + fine-tuning
pip install -e ".[all,gpu]"    # add bitsandbytes for 4-bit QLoRA (Linux + CUDA)
# Minimal GUI only:
pip install -e .
```

Requires Python 3.10+.

## Run

```bash
llm-tunner
```

1. **Documents** → add PDF(s).
2. **Knowledge (RAG)** → name a knowledge base → *Build*.
3. **Chat** → ask away (answers are cited). Works on CPU.
4. *(Optional)* **Fine-tune** → *Generate dataset* → *Start QLoRA fine-tune*.

## Development

```bash
pip install -e ".[dev]"
pytest                 # core tests run with no ML deps; RAG/training tests auto-skip
ruff check .
```

The RAG and training smoke tests activate automatically when their extras are installed.

## License & model notes

Code is Apache-2.0. Models you download carry their own licenses — the in-app picker
shows each one. Defaults favor permissive options (Qwen2.5 Apache-2.0, Phi-4 MIT).
Note Qwen2.5 **3B/72B** use a non-Apache license; verify per model. We avoid PyMuPDF as
a dependency because its AGPL-3.0 terms are restrictive for redistribution.
