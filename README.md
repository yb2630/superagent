# What Does a Wish Leave Out?

[![CI](https://github.com/yb2630/superagent/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/yb2630/superagent/actions/workflows/ci.yml)
[![Paper](https://github.com/yb2630/superagent/actions/workflows/paper.yml/badge.svg?branch=main)](https://github.com/yb2630/superagent/actions/workflows/paper.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](.python-version)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Requirement recovery in wish-driven code agents. Columbia COMS 4705.

A wish-driven coding agent receives an intent and builds an artifact. When the wish leaves a
requirement implicit, the program can run, pass the tests the user remembered to mention, and
still violate what the user meant. This repository mechanically degrades public code
specifications into *wishes* by deleting fields, trains a small **silence detector** that
predicts which requirement categories a wish leaves implicit, and uses that ranking to choose
clarification questions. The measured pilot uses Qwen3-8B on one A100, three samples per
condition, and a 100-task development split.

The accepted final milestone is
[`paper/milestone/main.pdf`](paper/milestone/main.pdf) (source:
[`paper/milestone/main.tex`](paper/milestone/main.tex)): a two-page body with references on
page 3. The extended measured report is [`paper/main.pdf`](paper/main.pdf). The submitted
milestone is preserved byte-identically under
[`paper/milestone/submitted/`](paper/milestone/submitted/).

Headline result on 100 development tasks, three samples per condition: the
full-specification ceiling B1 reaches 62.3% against 52.0% for the plain wish loop H0
(+10.3 pp, 95% CI [+3.0, +17.7]). Detector-targeted clarification H3 reaches 53.7%
(recovery 0.16) and beats generic questions H2 (44.3%) by +9.3 pp [+1.7, +17.0], while
withholding H3's answers gives 42.7% (−11.0 pp against H3) and sibling-function context H4
gives 51.7%. H3 is not distinguishable from H0 (+1.7 pp [−5.7, +9.0]). The detector reaches
macro-F1 0.792, constraints AUROC 0.952 and error-behaviour AUROC 0.744.

## Quickstart

Steps 1–3 need Python 3.12 and [`uv`](https://docs.astral.sh/uv/). Step 4 needs a Colab
A100, and step 5 needs a TeX distribution. The measured pilot is the one step that cannot be
reproduced without a GPU.

### 1. Clone, install, test

```bash
git clone https://github.com/yb2630/superagent.git
cd superagent
uv sync --frozen
make test
```

`make test` is CPU-only, needs no network after the environment is installed, and covers the
data pipeline, harness, detector, report and committed-artifact checks. The artifact checks
read the committed files under `results/`; they do not download datasets or call a model.

### 2. Rebuild the levels

```bash
make levels
```

The pinned raw files and the materialized tasks are **committed under `data/`**, so
`make levels` verifies them locally and needs no dataset network; it downloads only if a raw
file is missing (see `data/README.md`). It parses the specifications, validates every
canonical solution against visible and hidden tests, and writes the committed level artifact.
The default `inputs_only` visible-test mode is the corrected protocol used by the committed
levels. The run keeps 1,100 tasks of 1,682 source rows (1,651 parsed, 1,549 validated) in
666 seconds on eight cores; the 133 exclusions are 31 parse, 86 visible-test and 16
hidden-test failures. The data lives in `data/raw/` (three pinned files),
`data/derived/levels/` (4,400 files for 1,100 tasks) and `data/manifest.json`;
`data/cache/validation/` and `data/raw/_hf-cache/` stay gitignored.

Confirm the committed task artifact:

```bash
sha256sum results/levels/tasks.jsonl    # e4cf7564b38e17c040b8a37837c4757a8b5d4045791a1621331c91a15cd36436
```

`manifest.json` records the counts and `parsing.visible_tests = inputs_only`. Its
`created_at`, git and timing fields are provenance from the committed build, so a clean
rebuild reproduces the task hash and counts but not the manifest or stats bytes.

On macOS, use `shasum -a 256` instead of `sha256sum`. `make levels VISIBLE_TESTS=full`
rebuilds the older full-mode visible tests; the committed levels are inputs-only.

### 3. Detector

The final detector artifact is [`results/detector-smoothed/`](results/detector-smoothed/):
a frozen Qwen3-Embedding-0.6B encoder with a five-label linear head, refit on the 500
smoothed train wishes and evaluated five-fold out of fold. Two labels are constant on the 500
train tasks, so the usable ranking metrics are constraints AUROC 0.952, error-behaviour AUROC
0.744 and examples AUROC 0.632; macro-F1 is 0.792 (0.654 over the three varying labels).
The artifact contains `model.json`, `metrics.json`, `manifest.json`, `cv_predictions.jsonl`
and `report.md`.

Refit it on CPU from the committed smoothed wishes (the encoder is downloaded once and never
trained):

```bash
make detector-smoothed
```

The earlier raw-label detector artifact remains under [`results/detector/`](results/detector/)
for provenance; the final paper reports the smoothed refit.

### 4. The R2b A100 run

The measured run used one Colab **A100-SXM4-40GB** session: 2,700/2,700 units on the
100-task development split, three samples per condition, 2.902 GPU-hours, 19.36M tokens and
$0 paid API. The executed notebook is
[`results/runs/r2b-pilot/colab_a100_vllm.ipynb`](results/runs/r2b-pilot/colab_a100_vllm.ipynb);
the operator runbook is [`COLAB.md`](COLAB.md), and the short Colab runbook is
[`notebooks/colab_a100_vllm.txt`](notebooks/colab_a100_vllm.txt). The clean notebook is
[`notebooks/colab_a100_vllm.ipynb`](notebooks/colab_a100_vllm.ipynb) and the Colab copy
lives at <https://colab.research.google.com/drive/1cU_m5_HG8g9LzKzIhVtudr174CZdt3FJ?usp=sharing>.

The committed run is [`results/runs/r2b-pilot/`](results/runs/r2b-pilot/): the 2,700 unit
traces, `results.jsonl`, `summary.json`, `report.md`, the paired bootstrap, the smoothed
wishes, the detector refit, the 45-unit plumbing dry run, the executed notebook and the
provenance files. A CPU-only fake-endpoint verification of the same pipeline is committed as
[`results/runs/r2b-local-dry-run/README.md`](results/runs/r2b-local-dry-run/README.md) and can
be rerun with:

```bash
uv run python scripts/local_fake_dry_run.py
```

The actual GPU generations are the one non-reproducible-without-GPU step. Everything else —
the levels, the detector refit, the report, the bootstrap and the local dry run — runs on CPU.

### 5. The paper

[`paper/milestone/main.pdf`](paper/milestone/main.pdf) is the accepted final milestone; its
source is the corrected final text in [`paper/milestone/main.tex`](paper/milestone/main.tex).
It compiles against the repository's [`paper/neurips_2019.sty`](paper/neurips_2019.sty) and
[`paper/references.bib`](paper/references.bib) to exactly 3 pages: a 2-page body plus
references on page 3. From `paper/milestone/` (the leading `.` matters: with `..:` alone,
TeX resolves `main.tex` to the extended report in `paper/`):

```bash
TEXINPUTS=.:..: BIBINPUTS=.:..: pdflatex -interaction=nonstopmode main.tex
TEXINPUTS=.:..: BIBINPUTS=.:..: bibtex main
TEXINPUTS=.:..: BIBINPUTS=.:..: pdflatex -interaction=nonstopmode main.tex
TEXINPUTS=.:..: BIBINPUTS=.:..: pdflatex -interaction=nonstopmode main.tex
```

The extended report [`paper/main.pdf`](paper/main.pdf) is compiled from
[`paper/main.tex`](paper/main.tex) in `paper/` with the same pdfLaTeX/BibTeX sequence.

## Development

`make check` runs everything CI runs: ruff lint and format check, mypy and the test suite.
`make format` applies fixes. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for setup, pre-commit
hooks and commit conventions, and [`ARCHITECTURE.md`](ARCHITECTURE.md) for the module layout
and data flow.

## What is committed vs regenerated

| Path | Status | Notes |
| --- | --- | --- |
| `results/levels/` | committed | Rebuilt by `make levels` (inputs-only mode); pinned sources and hashes in `manifest.json`. |
| `results/detector/` | committed | Earlier raw-label detector artifact; not used by the final result. |
| `results/detector-smoothed/` | committed | Final detector refit on the committed smoothed train wishes; reproducible with `make detector-smoothed`. |
| `results/runs/r2b-pilot/` | committed | Measured GPU evidence; the 2,700 unit traces and the executed notebook are the record of the one GPU step. |
| `results/runs/r2b-local-dry-run/` | committed | CPU-only fake-endpoint verification summary; its `.local-dry-run` output is gitignored. |
| `data/` | regenerated | Gitignored downloads, validation cache and derived tests; `make levels` recreates it. |
| `.venv/` | regenerated | Created by `uv sync --frozen`. |

The only step that requires a GPU is the R2b pilot itself (vLLM serving Qwen3-8B-AWQ). The
committed traces make its analysis reproducible, and the local fake-endpoint dry run checks
the same harness path without a GPU. No paid API is used anywhere.

## Conditions

The nine conditions share the tasks, tools, visible tests and twelve-turn cap: B1 (full L0
specification, ceiling), H0 (plain wish loop), H1 (acceptance tests first), H2 (generic
clarification), H3 (detector-targeted clarification), H4 (five sibling functions with
doctests), B2 (five direct samples instead of questions), B3 (one call, wish and stub only),
and H3⁻ (H3's questions with the answers withheld). H5, offline harness evolution, remains
future work.

## Repository layout

```
src/wishrec/            level builder, parser, datasets, artifacts
src/wishrec/harness/    ReAct loop, tools, conditions, stakeholder, vLLM client, runner
src/wishrec/detector/   frozen encoder, linear head, calibration, policy, evaluation
notebooks/              clean R2b notebook and short Colab runbook
results/                levels, detector, measured R2b run, local dry run
paper/milestone/        accepted final milestone; submitted/ preserved byte-identically
paper/main.pdf          extended measured report
PLAN.md                 final milestone plan and verification
PROJECT.md              data, harness, detector and pilot design
ARCHITECTURE.md         module boundaries and data flow
CONTRIBUTING.md         developer setup, checks and conventions
.github/                CI workflows, Dependabot, issue and PR templates
COLAB.md                full operator runbook for the A100 run
paper/operationalization-notes.md  disclosed protocol limitations
```
