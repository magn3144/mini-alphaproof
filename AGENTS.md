# AGENTS — Guidance for AI coding agents

Purpose: help AI coding agents quickly understand, navigate, and operate safely in this repository.

Quick facts
- **Python:** >= 3.12 (see [pyproject.toml](pyproject.toml)).
- **Install deps:** `uv sync` (reads `pyproject.toml`).
- **Typical commands:**
  - Train: `uv run scripts/train.py --config configs/train_config.yaml`
  - Evaluate: `uv run scripts/evaluate.py --checkpoint <PATH> --data <PATH>`
  - Run quick check: `uv run main.py`

Where to look first
- Project overview: [README.md](README.md)
- Packaging & deps: [pyproject.toml](pyproject.toml)
- Training / eval CLIs: [scripts/train.py](scripts/train.py), [scripts/evaluate.py](scripts/evaluate.py), [scripts/interact.py](scripts/interact.py)
- Core implementation: [src/](src/)
- Configs: [configs/train_config.yaml](configs/train_config.yaml)
- Data layout: [data/processed/](data/processed/) and [data/raw/](data/raw/)
- Notebooks: [notebooks/](notebooks/)

Conventions & guidance
- Formatting & static checks: use `black`, `flake8`, and `mypy` (listed in `pyproject.toml`).
- Tests: `pytest` is the test runner (see `pyproject.toml`).
- Experiments are resource-intensive: do not start long training runs without explicit user confirmation.
- Hardware-specific packages (e.g., `bitsandbytes`, CUDA-enabled `torch`) may require GPU drivers and non-Python install steps — surface this to the user.
- Logging/telemetry: `wandb` is listed as a dependency — do not automatically upload data or credentials.

Behavior rules for agents
- When proposing changes affecting ML training or data, list expected compute, GPU requirements, and estimated runtime.
- Run unit tests and linters locally before suggesting broad refactors.
- Avoid irreversible changes to data or long-running experiments; ask the user before executing them.

Checklist for common tasks
- Setup: `uv sync`
- Quick lint/type-check: `uv run black . && uv run flake8 && uv run mypy src`
- Run tests: `uv run pytest -q`
- Run training prototype: `uv run scripts/train.py --config configs/train_config.yaml`
