# Cross-platform project tasks for GNU Make on Windows, Linux, and macOS.
#
# Override the interpreter when needed:
#   make test PYTHON=py
#   make install PYTHON=.venv/Scripts/python
#   make install PYTHON=.venv/bin/python

PYTHON ?= python
VENV ?= .venv
PIP := $(PYTHON) -m pip
PIP_ARGS ?=
PYTEST_ARGS ?=
RUFF_ARGS ?=

ifeq ($(OS),Windows_NT)
VENV_PYTHON := $(VENV)/Scripts/python.exe
else
VENV_PYTHON := $(VENV)/bin/python
endif

.DEFAULT_GOAL := help

.PHONY: help venv setup install install-all install-gpu install-dev upgrade \
	run run-gui test test-core lint format-check check compile build clean

help:
	@$(PYTHON) -c "print('LLM-tunner tasks:'); print('  make venv          Create .venv and upgrade pip'); print('  make setup         Create .venv and install development tools'); print('  make install       Install the minimal GUI'); print('  make install-all   Install GUI, RAG, and training'); print('  make install-gpu   Install all features plus QLoRA backend'); print('  make install-dev   Install development tools'); print('  make run           Launch through the console entry point'); print('  make run-gui       Launch through the GUI module'); print('  make test          Run all available tests'); print('  make test-core     Skip optional RAG/training smoke tests'); print('  make lint          Run Ruff'); print('  make check         Run lint, tests, and compilation'); print('  make build         Build wheel and source distribution'); print('  make clean         Remove generated project artifacts'); print(); print('Variables: PYTHON, VENV, PIP_ARGS, PYTEST_ARGS, RUFF_ARGS')"

venv:
	$(PYTHON) -m venv "$(VENV)"
	"$(VENV_PYTHON)" -m pip install --upgrade pip $(PIP_ARGS)

setup: venv
	"$(VENV_PYTHON)" -m pip install -e ".[dev]" $(PIP_ARGS)

upgrade:
	$(PIP) install --upgrade pip $(PIP_ARGS)

install:
	$(PIP) install -e . $(PIP_ARGS)

install-all:
	$(PIP) install -e ".[all]" $(PIP_ARGS)

install-gpu:
	$(PIP) install -e ".[all,gpu]" $(PIP_ARGS)

install-dev:
	$(PIP) install -e ".[dev]" $(PIP_ARGS)

run:
	$(PYTHON) -m llm_tunner.app

run-gui:
	$(PYTHON) -m llm_tunner.app

test:
	$(PYTHON) -m pytest $(PYTEST_ARGS)

test-core:
	$(PYTHON) -m pytest --ignore=tests/test_rag_smoke.py --ignore=tests/test_training_smoke.py $(PYTEST_ARGS)

lint:
	$(PYTHON) -m ruff check . $(RUFF_ARGS)

format-check:
	$(PYTHON) -m ruff format --check .

compile:
	$(PYTHON) -m compileall -q src tests

check: lint test compile

build:
	$(PIP) install --upgrade build $(PIP_ARGS)
	$(PYTHON) -m build

clean:
	$(PYTHON) -c "from pathlib import Path; import shutil; roots=[Path('build'),Path('dist'),Path('.pytest_cache'),Path('.ruff_cache'),Path('.mypy_cache'),Path('htmlcov')]; roots += list(Path('.').glob('*.egg-info')); roots += list(Path('src').glob('*.egg-info')); roots += list(Path('.').rglob('__pycache__')); [shutil.rmtree(path, ignore_errors=True) for path in roots]; [path.unlink(missing_ok=True) for path in (Path('.coverage'),)]"
