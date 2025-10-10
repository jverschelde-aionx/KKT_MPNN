# AGENTS.md

Guidelines for AI coding agents (and humans) contributing to this repository using the opencode workflow. This is similar in spirit to CLAUDE.md/CURSOR rules, adapted to this codebase.

## Goals
- Keep changes minimal, focused, and consistent with the existing code.
- Prefer clarity, type safety, and tests over cleverness.
- Document behavior and decisions where it aids future maintainers.

## Project Quick Facts
- Python: 3.9 (see `environment.yml`).
- Lint/Format: `ruff` (Black-compatible formatting; see `ruff.toml`).
- Line length: 88; indent: 4 spaces; double quotes by default.
- Testing: `pytest` available.
- Code lives under `src/`.
- GPU/ML stack: PyTorch 2.0, PyG, CUDA 11.8 (via Conda + pip wheels).

## Agent Workflow (opencode)
1. Plan briefly for multi-step work; group related actions logically.
2. Read before you write: inspect relevant files before editing.
3. Keep edits surgical; avoid unrelated refactors.
4. Run local checks (lint, format, tests) on changed code.
5. Summarize changes clearly; prefer “why” over “what”.

## Python Typing (Important)
- Target version is Python 3.9. Do not use `|` unions (PEP 604); prefer `Optional[T]` / `Union[A, B]`.
- Prefer built-in generics (PEP 585): `list[int]`, `dict[str, float]`, etc. (Python 3.9+).
- Consider `from __future__ import annotations` at module top if forward references are needed.
- Type everything that’s part of the public surface:
  - Function and method parameters and returns.
  - Class attributes (via `__init__` annotations or `dataclasses` if appropriate).
  - Complex containers (e.g., `list[torch.Tensor]` rather than `list`).
- Avoid `Any`; choose precise types or well-scoped `Protocol`/`TypedDict` where helpful.
- For NumPy/Torch tensors: include shape/meaning in docstrings; use `torch.Tensor` as the type.
- Use `Literal` for small closed sets of string/enum options when useful.

## Documentation
- Use clear, brief docstrings (Google style). Example:
  
  ```python
  def compute_scores(
      x: torch.Tensor,
      mask: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
      """Compute per-node scores.

      Args:
          x: Node features of shape [N, D].
          mask: Optional boolean mask of shape [N]. If provided, scores for
              masked nodes are ignored.

      Returns:
          Scores tensor of shape [N].
      """
  ```

- Keep types in signatures, not repeated in docstrings; docstrings describe semantics, shapes, units, and constraints.
- Module and class docstrings should state purpose and high-level behavior.
- Update README or inline docs when changing user-facing behavior.

## Code Style
- Follow `ruff` defaults in `ruff.toml`.
- Imports: standard library, third-party, then local; absolute over relative where reasonable.
- No unused variables; underscore prefix `_tmp` acceptable for intentional ignores.
- Prefer pure functions where possible; isolate side effects.
- Logging over prints for non-trivial scripts; keep messages actionable.
- Raise specific exceptions with helpful messages; avoid silent failures.
- Keep functions short and single-purpose; extract helpers rather than adding flags that change behavior.

## Testing
- Use `pytest` for new logic; put focused tests near the code they validate (e.g., `tests/` or alongside modules if that pattern emerges).
- Test the smallest unit that provides confidence; prefer fast, deterministic tests without GPU when possible.
- Provide minimal fixtures for tensor shapes and small graphs.
- When touching algorithms, add at least one correctness test and one edge-case test.

## Repository Conventions
- Source: `src/`
  - Graph transformer modules: `src/graphtrans/modules/`
  - Models and training glue: `src/models/`, `src/trainer.py`, etc.
  - Instance generation: `src/generate_instances.py`, `src/instances/`
- Configuration: `config.yml` for experiment parameters where applicable.
- Do not commit data, checkpoints, or large artifacts; prefer external storage with documented retrieval steps.

## Commands (Cheat Sheet)
- Lint and fix: `ruff check . --fix`
- Format: `ruff format .`
- Run tests: `pytest -q`
- Docker (recommended): `docker compose up --build`
- Conda env: `conda env create -f environment.yml && conda activate graph-aug`

## PRs and Commits
- Commit messages: concise, action + rationale (focus on “why”).
  - Good: "Add masked encoder to support partial graphs in training"
  - Avoid: "update", "fix stuff"
- Keep diffs small and reviewable; split unrelated changes.
- If a tool modifies files (formatters, pre-commit), include those changes in the same commit.

## Performance & ML Notes
- Clearly document tensor/device expectations; avoid implicit `.cuda()`—accept a `device` or infer from inputs.
- Avoid hidden global state; prefer explicit configuration.
- Seed randomness where determinism matters in tests.
- Be cautious with in-place ops that affect autograd or reused tensors.

## When Editing or Adding Models
- Keep encoders/blocks reusable and composable; new layers go under appropriate `modules/`.
- Separate data preparation from model logic.
- Surface hyperparameters via constructor args and/or config; validate ranges.
- Provide a minimal usage example in the docstring.

## When Generating Instances
- Keep generation logic deterministic when seeded; document seed handling.
- Validate output shapes and constraints; raise informative errors for invalid inputs.

## Safety & Secrets
- Never commit API keys, datasets, or proprietary assets.
- Respect `.gitignore` and add patterns for new artifacts as needed.

## Deviation and Ambiguity
- If an existing pattern is unclear, copy the dominant nearby style.
- If a rule conflicts with clarity or correctness, prefer clarity and explain briefly in the PR.

---
This document is a living guide. When the repository’s patterns evolve (e.g., adding `mypy`, changing docstring style, or test layout), update this file to match reality.
