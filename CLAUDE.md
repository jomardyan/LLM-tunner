# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

LLM-tunner is a **PySide6 desktop app** for customizing open LLMs with your own PDFs:
add PDFs → build a ChromaDB knowledge base → chat with cited answers (RAG) → optionally
QLoRA-fine-tune on synthesized Q&A. It is **RAG-first**; fine-tuning is the secondary path.
Source lives under `src/llm_tunner/` (src layout). Entry points: GUI `llm_tunner.app:main`
(`llm-tunner` / windowed `llm-tunner-gui`) and a **headless CLI** `llm_tunner.cli:main`
(`llm-tunner-cli`). The CLI is Qt-free: it reuses the `workers.tasks` functions via a
`ConsoleSignals` shim (progress/log/metric → stderr; results → stdout), so the same core
pipelines run without a display — keep it that way (no PySide6 imports in `cli.py`).

## Commands

**Python 3.14 is mandatory.** `app.py` asserts it at runtime and the Makefile refuses a
venv that isn't 3.14. The Makefile is the dev entry point — it creates/uses `.venv`
automatically (no activation needed) and every target gates on the version.

```bash
make setup          # create .venv (3.14) + install the [dev] toolchain
make install        # full runtime stack ([all,gpu]); on Windows also installs the CUDA wheel
make check          # the local gate: lint + tests + byte-compile
make test           # full pytest suite
make test-core      # pytest, skipping the optional RAG/training smoke tests
make lint / format  # ruff check / ruff format
make type-check     # mypy (best-effort, NON-gating — ~110 known findings)
make coverage       # pytest with HTML coverage
make ci             # mirror GitHub Actions locally
make run / run-gui  # launch (console entry / windowed entry)
```

Run a **single test** by passing pytest args through:

```bash
make test PYTEST_ARGS="tests/test_rag_indexing.py::test_build_prompt_includes_history_and_system_prefix"
# or directly (bypasses the Makefile's venv enforcement):
.venv/Scripts/python.exe -m pytest tests/test_inference.py -k contextualize   # Windows
.venv/bin/python -m pytest tests/test_inference.py -k contextualize           # Linux/macOS
```

Override the interpreter / venv when needed: `make check PYTHON="py -3.14"`,
`make test VENV=/opt/venvs/llm-tunner`. CI installs make on Windows via choco and runs
`make ci` under bash on both OSes.

## Architecture (the parts that span multiple files)

**Threading contract — the single most important rule.** All long work goes through
`MainWindow.submit()` (`main_window.py`), which wraps a task in a `Worker` (QRunnable,
`workers/base.py`), runs it on a shared `QThreadPool`, and wires its `WorkerSignals`
(`progress`/`metric`/`log`/`result`/`error`/`finished`) to caller callbacks. **Workers
never touch widgets** — they only emit signals (delivered to the GUI thread via queued
connections). Tabs (`ui/*.py`) never create threads. Task functions live in
`workers/tasks.py` and are thin wrappers over `core/`.

**Lazy ML imports.** torch/transformers/chromadb/etc. are imported *inside functions/
methods*, never at module top, so the GUI launches with no ML stack installed. Core
modules raise actionable errors when a feature's deps are missing (e.g.
`core/rag.require_rag_dependencies`). When editing `core/`, keep heavy imports lazy.

**Windows native-isolation (the reason `tasks.py` is large).** On `win32`, knowledge-base
builds and fine-tuning run in a **spawned child process** with file-based IPC (JSONL for
progress/metrics, JSON for result/error, a touch-file stop event), because bitsandbytes/
PyTorch/Arrow native teardown can crash the Qt host after a successful run. Hence the
`_build_kb` / `_build_kb_isolated` / `_build_kb_child` (and `_finetune_*`) trios — the
`*_isolated` path spawns, the `*_child` runs in the subprocess, the plain one runs
in-thread on non-Windows. **Consequence:** anything threaded into these tasks must cross a
process boundary, so pass **primitives/dicts, not dataclasses** (see `_train_config`,
chunk-size args), and add new args to the `Process(args=...)` tuple *and* the child
signature together.

**Device gating is centralized.** `core/device.py` is the single source of truth for
hardware capability (CUDA > MPS > CPU; `supports_qlora` only on CUDA + bitsandbytes).
`detect_device()` is cached; `detect_device_isolated()` probes in a subprocess so GUI
startup never imports torch. dtype, `device_map`, 4-bit gating, and model recommendations
all derive from `DeviceInfo` — don't re-detect hardware elsewhere.

**Config → Settings → tasks flow.** `config.py` holds the default dataclasses
(`ChunkConfig`, `RagConfig`, `TrainConfig`, `LoraConfig`, `GenerationConfig`) and
`MODEL_REGISTRY`. `settings.py` persists user choices via `QSettings` (with an in-memory
fallback so core/tests run headless). The pattern is: UI reads `window.settings` at action
time → passes **primitives** through `submit()` kwargs → the task rebuilds the dataclass.
Adding a tunable means: default in `config.py`, persisted property in `settings.py`, a
widget in `ui/settings_tab.py` (or the relevant tab), and threading it through the task.

**RAG pipeline.** `pdf` (pypdf for native text, Docling for scanned/OCR) → `chunking`
(dependency-free recursive splitter; sizes are **characters** by default unless a
tokenizer `length_fn` is passed) → `Embedder` (ONNX Runtime) → ChromaDB, one collection
per KB under `config.kb_dir()`. The embedder picks its ONNX **execution provider** via
`rag.onnx_providers()` — `RagConfig.onnx_provider` ("auto" by default) selects an
integrated/discrete GPU when an accelerated runtime is installed (`onnxruntime-directml`
or `onnxruntime-openvino`, the latter's `MULTI:GPU,CPU` running CPU+iGPU together) and
always falls back to CPU. This is the only ML path that uses an iGPU — torch chat/QLoRA do
not. The provider is threaded UI → tasks → `RagConfig` like the other knobs. `RagIndex.build_prompt` assembles a grounded, cited prompt
and now accepts conversation `history` for multi-turn RAG. Embedding-model mismatch is
detected per-KB. Storage names are made Windows-safe in `rag._storage_name`/`_collection_name`.

**Diagnostics.** `diagnostics.py` configures a rotating log + a faulthandler native-crash
log; `normalize_error()` maps raw exceptions to a user-facing `UserError`
(title/summary/action/incident_id) and redacts HF tokens; errors surface via
`ui/error_dialog.show_error_dialog`. Prefer routing failures through this rather than
bare dialogs.

## Conventions & gotchas

- Every module starts with `from __future__ import annotations`.
- Ruff is the linter/formatter (`line-length=100`, `E501` ignored); `make type-check`
  (mypy) is intentionally **non-gating** — the codebase has pre-existing findings
  (UI tabs shadow `QWidget.window()`), so don't block work on it.
- Subprocess calls that can pop a console on Windows must pass
  `creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)`.
- **Two different setuptools pins, on purpose:** the runtime venv pins `setuptools<82`
  (torch 2.12 requires it); `pyproject [build-system]` needs `setuptools>=82.0.1` but that
  applies *only* to the isolated PEP 517 build env used by `make build`. Don't "sync" them.
- **Do not add PySide6 Addons** (Qt PDF/WebEngine, etc.) — the project ships
  Essentials-only deliberately (licensing/size). Anything needing Addons requires explicit
  sign-off.
- **Durable artifacts / no overwrite:** generated datasets and trained adapters are named
  with `config.timestamped_name(prefix)` (timestamp + short random suffix) so re-runs never
  clobber prior outputs. The data directory is `config.data_dir()` (honors `LLM_TUNNER_HOME`);
  a persisted `Settings.data_dir` is applied at startup in `app.py` by setting that env var
  *before* any path resolves (and `cli.py` honors `--data-dir`). KBs/datasets/adapters/logs
  all live under it.
- Optional dependency extras: `[rag]`, `[train]`, `[gpu]` (bitsandbytes, CUDA-only marker),
  `[all]`, `[dev]`, and `[directml]`/`[openvino]` for iGPU embeddings. The latter two ship
  accelerated `onnxruntime` builds that are **mutually exclusive** with the plain
  `onnxruntime` in `[rag]` — install one *instead of* it (uninstall `onnxruntime` first);
  do not combine `[rag,directml]` in a single install.

## Testing notes

- Core tests run with **no ML deps**. RAG/training **smoke tests auto-skip** when their
  extras are absent (`importlib.util.find_spec` gating), so they run on dev machines with
  the full venv but skip in lean CI.
- GUI tests run **offscreen** — they set `QT_QPA_PLATFORM=offscreen` themselves and are
  guarded by a `_HAVE_GUI` skip.
- Message-building / config logic is deliberately **torch-free** (importing
  `core.inference` / `core.rag` / `workers.tasks` pulls no heavy deps), so most new unit
  tests can run in the core suite. Use monkeypatched `submit`/`generate` and the
  `RagIndex.__new__(RagIndex)` trick (see `tests/test_rag_indexing.py`) to avoid loading
  models.
- `tests/conftest.py` builds a real in-memory PDF (`make_pdf_bytes`) and points
  `LLM_TUNNER_HOME` at a temp dir so tests never touch real user data.
