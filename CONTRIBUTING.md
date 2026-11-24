# Contributing

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --frozen               # runtime + dev tools (ruff, mypy, pytest-cov)
uvx pre-commit install         # hygiene, ruff and spelling hooks on commit
uvx pre-commit install --hook-type pre-push   # mypy on push
```

## Everyday commands

| Command | What it does |
| --- | --- |
| `make test` | Run the CPU-only test suite (no network, no GPU). |
| `make coverage` | Tests with a coverage report. |
| `make lint` | `ruff check` and `ruff format --check` (what CI runs). |
| `make format` | Apply safe lint fixes and reformat. |
| `make typecheck` | `mypy` over `src/` and `scripts/`. |
| `make check` | Lint, type-check and test, in CI order. |

CI (`.github/workflows/ci.yml`) runs the same checks on every pull request, plus the
pre-commit hygiene hooks. Changes under `paper/` also trigger `paper.yml`, which compiles both
papers and fails if the milestone exceeds 3 pages.

## Branches and commits

- Branch from `main`; open a pull request; merge after CI passes and a teammate reviews.
  CodeRabbit posts an automated review on each PR.
- Use [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`,
  `refactor:`, `test:`, `docs:`, `ci:`, `build:`, `chore:`, optionally with a scope such as
  `fix(harness): ...`. Keep commits small and atomic.

## Code standards

- Every module, public class and public function has a docstring; every function signature
  is annotated (enforced by ruff and mypy).
- Fix root causes: no `# noqa` or `# type: ignore` to silence a real finding.
- New logic and bug fixes come with tests. Tests must run on CPU without network access or a
  live model endpoint; use the fakes in `tests/fakes.py`.

## Committed artifacts

`data/` and `results/` hold committed, hash-checked artifacts. Do not edit them by hand. If a
code change alters them, regenerate them with the matching `make` target and commit the
regenerated files with the change. They are excluded from formatting hooks so they stay
byte-identical.

## Updating tools

Tool versions come from `uv.lock` (ruff, mypy) and `.pre-commit-config.yaml` (other hooks).
Dependabot proposes monthly updates; to update hooks by hand, run `uvx pre-commit autoupdate`.
