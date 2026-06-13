# Cross-platform project tasks for GNU Make on Windows, Linux, and macOS.
#
# Override the bootstrap interpreter or virtual environment when needed:
#   make setup PYTHON=py
#   make install VENV=C:/venvs/llm-tunner
#   make test VENV=/opt/venvs/llm-tunner
#
# Run `make` (or `make help`) for a list of targets.

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
MYPY_ARGS ?=
PIP_AUDIT_ARGS ?=
COV_ARGS ?= --cov=llm_tunner --cov-report=term-missing --cov-report=html
UPGRADE_ARGS ?= --upgrade --upgrade-strategy eager
# Build tooling for the *runtime* venv. Note: torch pins setuptools<82, so this
# must stay below 82 to avoid a runtime dependency conflict. This is deliberately
# NOT the same as pyproject.toml [build-system].requires (setuptools>=82.0.1),
# which only applies to the isolated PEP 517 build environment used by `make build`.
SETUPTOOLS_SPEC ?= setuptools>=81,<82
WHEEL_SPEC ?= wheel>=0.47,<0.48
PYTORCH_CUDA_INDEX ?= https://download.pytorch.org/whl/cu130

ifeq ($(OS),Windows_NT)
VENV_PYTHON := $(VENV)/Scripts/python.exe
GUI_SCRIPT := $(VENV)/Scripts/llm-tunner-gui.exe
else
VENV_PYTHON := $(VENV)/bin/python
GUI_SCRIPT := $(VENV)/bin/llm-tunner-gui
endif

PROJECT_PYTHON := "$(VENV_PYTHON)"
PIP := $(PROJECT_PYTHON) -m pip

.DEFAULT_GOAL := help

# These targets create/install into a shared .venv, so they must never run
# concurrently. Force serial execution even under `make -j`.
.NOTPARALLEL:

.PHONY: help verify-python verify-venv venv setup install install-minimal install-rag install-train install-all install-gpu \
	install-dev upgrade \
	run run-gui test test-core lint format format-check type-check coverage audit \
	check ci compile build build-check clean

help:  ## Show this list of targets
	@$(PYTHON) -c "import re; rows=[(m.group(1), m.group(2).strip()) for line in open('Makefile', encoding='utf-8') for m in [re.match(r'([A-Za-z0-9_-]+):.*?## (.*)', line)] if m]; width=max(len(name) for name, _ in rows); print('LLM-tunner tasks (Python 3.14):'); [print('  ' + name.ljust(width) + '   ' + desc) for name, desc in rows]; print(''); print('Variables: PYTHON VENV PIP_ARGS UPGRADE_ARGS PYTEST_ARGS RUFF_ARGS MYPY_ARGS COV_ARGS PIP_AUDIT_ARGS PYTORCH_CUDA_INDEX')"

verify-python:
	$(PYTHON) -c "import sys; assert sys.version_info[:2] == (3, 14), f'Python 3.14 required, found {sys.version.split()[0]}'"

verify-venv: $(VENV_PYTHON)
	$(PROJECT_PYTHON) -c "import sys; venv=sys.argv[1]; expected=(3, 14); actual=sys.version_info[:2]; assert actual == expected, f'{venv} uses Python {sys.version.split()[0]}, but Python 3.14 is required. Remove or rename {venv}, then run make install.'" "$(VENV)"

$(VENV_PYTHON): | verify-python
	$(PYTHON) -m venv "$(VENV)"

venv: verify-venv  ## Create .venv and upgrade pip/setuptools/wheel
	"$(VENV_PYTHON)" -m pip install --upgrade pip "$(SETUPTOOLS_SPEC)" "$(WHEEL_SPEC)" $(PIP_ARGS)

setup: venv  ## Create .venv and install the development toolchain
	$(PIP) install $(UPGRADE_ARGS) -e ".[dev]" $(PIP_ARGS)

upgrade: verify-venv  ## Upgrade pip/setuptools/wheel in .venv
	$(PIP) install --upgrade pip "$(SETUPTOOLS_SPEC)" "$(WHEEL_SPEC)" $(PIP_ARGS)

install: venv  ## Install the full runtime stack (adds the CUDA wheel on Windows)
	$(PIP) install $(UPGRADE_ARGS) -e ".[all,gpu]" $(PIP_ARGS)
ifeq ($(OS),Windows_NT)
	$(PIP) install --upgrade torch torchvision --index-url $(PYTORCH_CUDA_INDEX) $(PIP_ARGS)
endif

install-minimal: venv  ## Install only the lightweight GUI stack
	$(PIP) install $(UPGRADE_ARGS) -e . $(PIP_ARGS)

install-rag: venv  ## Install the GUI and RAG dependencies
	$(PIP) install $(UPGRADE_ARGS) -e ".[rag]" $(PIP_ARGS)

install-train: venv  ## Install the GUI and training dependencies
	$(PIP) install $(UPGRADE_ARGS) -e ".[train]" $(PIP_ARGS)

install-all: install  ## Alias for make install

install-gpu: install  ## Install the NVIDIA CUDA wheel plus every runtime feature
ifneq ($(OS),Windows_NT)
	$(PIP) install --upgrade torch torchvision --index-url $(PYTORCH_CUDA_INDEX) $(PIP_ARGS)
endif

install-dev: verify-venv  ## Install the development toolchain into .venv
	$(PIP) install $(UPGRADE_ARGS) -e ".[dev]" $(PIP_ARGS)

run: verify-venv  ## Launch through the console entry point
	$(PROJECT_PYTHON) -m llm_tunner.app

run-gui: verify-venv  ## Launch through the windowed GUI entry point
	"$(GUI_SCRIPT)"

test: verify-venv  ## Run all available tests
	$(PROJECT_PYTHON) -m pytest $(PYTEST_ARGS)

test-core: verify-venv  ## Run tests, skipping optional RAG/training smoke tests
	$(PROJECT_PYTHON) -m pytest --ignore=tests/test_rag_smoke.py --ignore=tests/test_training_smoke.py $(PYTEST_ARGS)

lint: verify-venv  ## Run Ruff lint checks
	$(PROJECT_PYTHON) -m ruff check . $(RUFF_ARGS)

format: verify-venv  ## Apply Ruff autoformatting in place
	$(PROJECT_PYTHON) -m ruff format . $(RUFF_ARGS)

format-check: verify-venv  ## Check Ruff formatting without writing changes
	$(PROJECT_PYTHON) -m ruff format --check . $(RUFF_ARGS)

type-check: verify-venv  ## Run mypy static type checks (best-effort, non-gating)
	$(PROJECT_PYTHON) -m mypy src $(MYPY_ARGS)

coverage: verify-venv  ## Run tests with a coverage report (terminal + HTML)
	$(PROJECT_PYTHON) -m pytest $(COV_ARGS) $(PYTEST_ARGS)

audit: verify-venv  ## Scan installed packages for known vulnerabilities (pip-audit)
	$(PROJECT_PYTHON) -m pip_audit $(PIP_AUDIT_ARGS)

compile: verify-venv  ## Byte-compile sources to surface syntax errors
	$(PROJECT_PYTHON) -m compileall -q src tests

check: lint test compile  ## Run lint, tests, and compilation

ci: venv install-dev lint test compile  ## Mirror the GitHub Actions checks locally
	@$(PROJECT_PYTHON) -c "print('make ci: lint, tests, and compile passed')"

build: verify-venv  ## Build the wheel and source distribution
	$(PIP) install --upgrade build $(PIP_ARGS)
	$(PROJECT_PYTHON) -m build

build-check: build  ## Build, then validate the distributions with twine
	$(PIP) install --upgrade twine $(PIP_ARGS)
	$(PROJECT_PYTHON) -c "import glob, subprocess, sys; sys.exit(subprocess.call([sys.executable, '-m', 'twine', 'check', *glob.glob('dist/*')]))"

clean:  ## Remove generated build, cache, and coverage artifacts
	$(PYTHON) -c "from pathlib import Path; import shutil; roots=[Path('build'),Path('dist'),Path('.pytest_cache'),Path('.ruff_cache'),Path('.mypy_cache'),Path('htmlcov')]; roots += list(Path('.').glob('*.egg-info')); roots += list(Path('src').glob('*.egg-info')); roots += list(Path('.').rglob('__pycache__')); [shutil.rmtree(path, ignore_errors=True) for path in roots]; [path.unlink(missing_ok=True) for path in (Path('.coverage'),)]"
