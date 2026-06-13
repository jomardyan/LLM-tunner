# Cross-platform project tasks for GNU Make on Windows, Linux, and macOS.
#
# Override the bootstrap interpreter or virtual environment when needed:
#   make setup PYTHON=py
#   make install VENV=C:/venvs/llm-tunner
#   make test VENV=/opt/venvs/llm-tunner

PYTHON ?= python
VENV ?= .venv
PIP_ARGS ?=
PYTEST_ARGS ?=
RUFF_ARGS ?=
PYTORCH_CUDA_INDEX ?= https://download.pytorch.org/whl/cu130

ifeq ($(OS),Windows_NT)
VENV_PYTHON := $(VENV)/Scripts/python.exe
else
VENV_PYTHON := $(VENV)/bin/python
endif

PROJECT_PYTHON := "$(VENV_PYTHON)"
PIP := $(PROJECT_PYTHON) -m pip

.DEFAULT_GOAL := help

.PHONY: help venv setup install install-minimal install-rag install-train install-all install-gpu \
	install-dev upgrade \
	run run-gui test test-core lint format-check check compile build clean

help:
	@$(PYTHON) -c "print('LLM-tunner tasks:'); print('  make install       MASTER INSTALL: create .venv and install every runtime feature'); print('  make install-minimal Install only the lightweight GUI'); print('  make install-rag   Install GUI and RAG dependencies'); print('  make install-train Install GUI and training dependencies'); print('  make install-all   Alias for make install'); print('  make install-gpu   Install NVIDIA PyTorch plus every runtime feature'); print('  make venv          Create .venv and upgrade pip'); print('  make setup         Create .venv and install development tools'); print('  make install-dev   Install development tools into .venv'); print('  make run           Launch through the console entry point'); print('  make run-gui       Launch through the GUI module'); print('  make test          Run all available tests'); print('  make test-core     Skip optional RAG/training smoke tests'); print('  make lint          Run Ruff'); print('  make check         Run lint, tests, and compilation'); print('  make build         Build wheel and source distribution'); print('  make clean         Remove generated project artifacts'); print(); print('Variables: PYTHON, VENV, PIP_ARGS, PYTEST_ARGS, RUFF_ARGS, PYTORCH_CUDA_INDEX')"

$(VENV_PYTHON):
	$(PYTHON) -m venv "$(VENV)"

venv: $(VENV_PYTHON)
	"$(VENV_PYTHON)" -m pip install --upgrade pip $(PIP_ARGS)

setup: venv
	$(PIP) install -e ".[dev]" $(PIP_ARGS)

upgrade: $(VENV_PYTHON)
	$(PIP) install --upgrade pip $(PIP_ARGS)

install: venv
	$(PIP) install -e ".[all,gpu]" $(PIP_ARGS)
ifeq ($(OS),Windows_NT)
	$(PIP) install --upgrade torch torchvision --index-url $(PYTORCH_CUDA_INDEX) $(PIP_ARGS)
endif

install-minimal: venv
	$(PIP) install -e . $(PIP_ARGS)

install-rag: venv
	$(PIP) install -e ".[rag]" $(PIP_ARGS)

install-train: venv
	$(PIP) install -e ".[train]" $(PIP_ARGS)

install-all: install

install-gpu: install
ifneq ($(OS),Windows_NT)
	$(PIP) install --upgrade torch torchvision --index-url $(PYTORCH_CUDA_INDEX) $(PIP_ARGS)
endif

install-dev: $(VENV_PYTHON)
	$(PIP) install -e ".[dev]" $(PIP_ARGS)

run: $(VENV_PYTHON)
	$(PROJECT_PYTHON) -m llm_tunner.app

run-gui: $(VENV_PYTHON)
	$(PROJECT_PYTHON) -m llm_tunner.app

test: $(VENV_PYTHON)
	$(PROJECT_PYTHON) -m pytest $(PYTEST_ARGS)

test-core: $(VENV_PYTHON)
	$(PROJECT_PYTHON) -m pytest --ignore=tests/test_rag_smoke.py --ignore=tests/test_training_smoke.py $(PYTEST_ARGS)

lint: $(VENV_PYTHON)
	$(PROJECT_PYTHON) -m ruff check . $(RUFF_ARGS)

format-check: $(VENV_PYTHON)
	$(PROJECT_PYTHON) -m ruff format --check .

compile: $(VENV_PYTHON)
	$(PROJECT_PYTHON) -m compileall -q src tests

check: lint test compile

build: $(VENV_PYTHON)
	$(PIP) install --upgrade build $(PIP_ARGS)
	$(PROJECT_PYTHON) -m build

clean:
	$(PYTHON) -c "from pathlib import Path; import shutil; roots=[Path('build'),Path('dist'),Path('.pytest_cache'),Path('.ruff_cache'),Path('.mypy_cache'),Path('htmlcov')]; roots += list(Path('.').glob('*.egg-info')); roots += list(Path('src').glob('*.egg-info')); roots += list(Path('.').rglob('__pycache__')); [shutil.rmtree(path, ignore_errors=True) for path in roots]; [path.unlink(missing_ok=True) for path in (Path('.coverage'),)]"
