# Cross-platform project tasks for GNU Make on Windows, Linux, and macOS.
#
# Override the bootstrap interpreter or virtual environment when needed:
#   make setup PYTHON=py
#   make install VENV=C:/venvs/llm-tunner
#   make test VENV=/opt/venvs/llm-tunner

ifeq ($(OS),Windows_NT)
WINDOWS_PYTHON_314 := $(LOCALAPPDATA)/Programs/Python/Python314/python.exe
ifneq ($(wildcard $(WINDOWS_PYTHON_314)),)
PYTHON ?= "$(WINDOWS_PYTHON_314)"
else
PYTHON ?= py -3.14
endif
else
PYTHON ?= python3.14
endif
VENV ?= .venv
PIP_ARGS ?=
PYTEST_ARGS ?=
RUFF_ARGS ?=
UPGRADE_ARGS ?= --upgrade --upgrade-strategy eager
PYTORCH_CUDA_INDEX ?= https://download.pytorch.org/whl/cu130

ifeq ($(OS),Windows_NT)
VENV_PYTHON := $(VENV)/Scripts/python.exe
else
VENV_PYTHON := $(VENV)/bin/python
endif

PROJECT_PYTHON := "$(VENV_PYTHON)"
PIP := $(PROJECT_PYTHON) -m pip

.DEFAULT_GOAL := help

.PHONY: help verify-python verify-venv venv setup install install-minimal install-rag install-train install-all install-gpu \
	install-dev upgrade \
	run run-gui test test-core lint format-check check compile build clean

help:
	@$(PYTHON) -c "print('LLM-tunner tasks (Python 3.14):'); print('  make install       Create .venv and install the latest compatible runtime stack'); print('  make install-minimal Install only the latest lightweight GUI stack'); print('  make install-rag   Install latest GUI and RAG dependencies'); print('  make install-train Install latest GUI and training dependencies'); print('  make install-all   Alias for make install'); print('  make install-gpu   Install latest NVIDIA PyTorch plus every runtime feature'); print('  make venv          Create .venv and upgrade pip/setuptools/wheel'); print('  make setup         Create .venv and install latest development tools'); print('  make install-dev   Install latest development tools into .venv'); print('  make run           Launch through the console entry point'); print('  make run-gui       Launch through the GUI module'); print('  make test          Run all available tests'); print('  make test-core     Skip optional RAG/training smoke tests'); print('  make lint          Run Ruff'); print('  make check         Run lint, tests, and compilation'); print('  make build         Build wheel and source distribution'); print('  make clean         Remove generated project artifacts'); print(); print('Variables: PYTHON, VENV, PIP_ARGS, UPGRADE_ARGS, PYTEST_ARGS, RUFF_ARGS, PYTORCH_CUDA_INDEX')"

verify-python:
	$(PYTHON) -c "import sys; assert sys.version_info[:2] == (3, 14), f'Python 3.14 required, found {sys.version.split()[0]}'"

verify-venv: $(VENV_PYTHON)
	$(PROJECT_PYTHON) -c "import sys; expected=(3, 14); actual=sys.version_info[:2]; message=f'$(VENV) uses Python {sys.version.split()[0]}, but Python 3.14 is required. Remove or rename $(VENV), then run make install.'; assert actual == expected, message"

$(VENV_PYTHON): | verify-python
	$(PYTHON) -m venv "$(VENV)"

venv: verify-venv
	"$(VENV_PYTHON)" -m pip install --upgrade pip "setuptools>=81,<82" wheel $(PIP_ARGS)

setup: venv
	$(PIP) install $(UPGRADE_ARGS) -e ".[dev]" $(PIP_ARGS)

upgrade: verify-venv
	$(PIP) install --upgrade pip "setuptools>=81,<82" wheel $(PIP_ARGS)

install: venv
	$(PIP) install $(UPGRADE_ARGS) -e ".[all,gpu]" $(PIP_ARGS)
ifeq ($(OS),Windows_NT)
	$(PIP) install --upgrade torch torchvision --index-url $(PYTORCH_CUDA_INDEX) $(PIP_ARGS)
endif

install-minimal: venv
	$(PIP) install $(UPGRADE_ARGS) -e . $(PIP_ARGS)

install-rag: venv
	$(PIP) install $(UPGRADE_ARGS) -e ".[rag]" $(PIP_ARGS)

install-train: venv
	$(PIP) install $(UPGRADE_ARGS) -e ".[train]" $(PIP_ARGS)

install-all: install

install-gpu: install
ifneq ($(OS),Windows_NT)
	$(PIP) install --upgrade torch torchvision --index-url $(PYTORCH_CUDA_INDEX) $(PIP_ARGS)
endif

install-dev: verify-venv
	$(PIP) install $(UPGRADE_ARGS) -e ".[dev]" $(PIP_ARGS)

run: verify-venv
	$(PROJECT_PYTHON) -m llm_tunner.app

run-gui: verify-venv
	$(PROJECT_PYTHON) -m llm_tunner.app

test: verify-venv
	$(PROJECT_PYTHON) -m pytest $(PYTEST_ARGS)

test-core: verify-venv
	$(PROJECT_PYTHON) -m pytest --ignore=tests/test_rag_smoke.py --ignore=tests/test_training_smoke.py $(PYTEST_ARGS)

lint: verify-venv
	$(PROJECT_PYTHON) -m ruff check . $(RUFF_ARGS)

format-check: verify-venv
	$(PROJECT_PYTHON) -m ruff format --check .

compile: verify-venv
	$(PROJECT_PYTHON) -m compileall -q src tests

check: lint test compile

build: verify-venv
	$(PIP) install --upgrade build $(PIP_ARGS)
	$(PROJECT_PYTHON) -m build

clean:
	$(PYTHON) -c "from pathlib import Path; import shutil; roots=[Path('build'),Path('dist'),Path('.pytest_cache'),Path('.ruff_cache'),Path('.mypy_cache'),Path('htmlcov')]; roots += list(Path('.').glob('*.egg-info')); roots += list(Path('src').glob('*.egg-info')); roots += list(Path('.').rglob('__pycache__')); [shutil.rmtree(path, ignore_errors=True) for path in roots]; [path.unlink(missing_ok=True) for path in (Path('.coverage'),)]"
