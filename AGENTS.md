# Repository Guidelines

## Project Structure & Module Organization

- **Rules.md**: this file, overview of repo structure, coding style, project goals. YOU MUST READ IT.
- Source: `signstream/` with submodules: `data/`, `io/`, `models/` (RVQ, metrics), `training/`, `inference/`.
- Configs: `signstream/configs/default.yaml` (experiment, data paths, model, training, logging, export).
- Tests: `signstream/tests/` and root-level `test_*.py` helpers.
- Scripts: `train.sh`, `train_multi.sh`, and small debug utilities `debug_*.py`.

## Setup, Build, and Development Commands

- Notice: you a currently on the Windows and there is no datas provided. In reality, this repo will be run on Linux with proper data. So just pretend you have the data.
- Environment: the environment is already installed so do not need to install again.

## Coding Style & Naming Conventions

- Python (3.9+), PEP 8, 4-space indentation, type hints where practical.
- Names: modules/functions `snake_case`; classes `PascalCase`; constants `UPPER_SNAKE_CASE`.
- Keep modules cohesive: data I/O in `signstream/data|io`, model code in `signstream/models`, training loops and CLI in `signstream/training`.

## Testing Guidelines

- Framework: `pytest`.
- Run all: `pytest -q`.
- Focus a subset: `pytest -k quantizer -q` or a file: `pytest signstream/tests/test_dataset.py -q`.
- Test files: `test_*.py`; use small, deterministic tensors; avoid GPU-only assumptions unless guarded.

## Commit & Pull Request Guidelines

- **Record Keeping**: Keep a brief changelog in `CHANGELOG.md` for significant changes. Write your reason and motivation for every changes you made.
- Commits: imperative mood, concise subject (<72 chars), include rationale in body.
- Prefer Conventional Commits when helpful: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.
- PRs: clear description, linked issues, reproduction or usage steps, sample commands, and before/after metrics (loss, utilization) when training is affected.
- Ensure: tests pass locally, no stray prints, configs/docs updated when changing behavior.

## Security & Configuration Tips

- Do not commit datasets, checkpoints, or large logs; outputs go under per-run dirs created beneath `save_path` (default `./checkpoints`).
- Update `data.root` and paths in `default.yaml` to your environment before training.
- Use `--device` or `training.device` to control CPU/GPU; keep runs reproducible via `training.seed`.
