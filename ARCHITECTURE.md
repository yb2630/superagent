# Architecture

This document describes how the code is organised and how data moves through it. For the
experimental design (fields, levels, conditions, results) see [`PROJECT.md`](PROJECT.md).

## Pipeline overview

```
 pinned benchmarks          R0: data foundation            R1/R2: harness + pilot
┌──────────────────┐   ┌───────────────────────────┐   ┌──────────────────────────────┐
│ HumanEval+       │   │ datasets → parse →        │   │ conditions → loop (ReAct)    │
│ MBPP+            │──▶│ validate → split → levels │──▶│   ↕ tools  ↕ stakeholder     │
│ BigCodeBench     │   │ → artifacts               │   │ → hidden-test evaluation     │
└──────────────────┘   └─────────────┬─────────────┘   │ → report → bootstrap         │
     data/raw/                       │                 └──────────────▲───────────────┘
                                     ▼                                │ question policy
                         results/levels/ (committed)   ┌──────────────┴───────────────┐
                                     │                 │ detector: frozen encoder +   │
                                     └────────────────▶│ linear head → calibration →  │
                                                       │ policy                       │
                                                       └──────────────────────────────┘
                                                        results/detector-smoothed/
```

1. **R0 data foundation** (`make levels`, `wishrec.build`) downloads three pinned,
   checksum-verified benchmarks, parses each specification into five fields, validates every
   canonical solution against derived visible and hidden tests, draws a deterministic
   source-stratified split, and writes the L0–L3 levels plus a manifest of hashes.
2. **Detector** (`make detector-smoothed`, `wishrec.detector.train`) embeds the train wishes
   with a frozen Qwen3-Embedding-0.6B encoder on CPU, fits a five-label linear head, calibrates
   it with Platt scaling under cross-validation, and writes a model plus metrics.
3. **Harness and pilot** (`wishrec.harness.runner`) runs each development task under nine
   conditions against an OpenAI-compatible vLLM endpoint, evaluates the final solution on the
   hidden tests in a fresh workspace, and aggregates pass@1, recovery and paired bootstrap
   intervals.

## Packages and module boundaries

### `src/wishrec/` — R0 data foundation

| Module | Responsibility |
| --- | --- |
| `config.py` | Pinned sources, paths and milestone constants. |
| `datasets.py` | Pinned, checksum-verified downloads and row loading. |
| `parse.py` | Mechanical parsing of the three benchmarks into the five fields. |
| `validate.py` | Canonical-solution validation against visible and hidden tests. |
| `split.py` | Deterministic task selection and the 500/200/300 + 100 split. |
| `levels.py` | Level construction (L0–L3 and single-field drops). |
| `artifacts.py` | Record serialisation, artifact writing and invariant checks. |
| `materialize.py` | Regenerates and hash-checks `data/derived/levels` (committed) from the level records. |
| `smoothing.py` | Smoothed-L3 artifact produced by the R2 GPU run. |
| `build.py` | CLI entry point that runs the R0 pipeline end to end. |

### `src/wishrec/detector/` — silence detector

`data` loads the train split; `encoder` wraps the frozen embedding model (lazy, CPU-only);
`head` is the linear BCE + L2 head; `cv` and `calibration` run five-fold cross-validation
with inner-fold model selection and Platt calibration; `metrics` and `baselines` evaluate it;
`policy` turns calibrated probabilities into H3 question choices; `trained` loads a fitted
model for inference; `train` is the CLI entry point.

### `src/wishrec/harness/` — ReAct harness

`conditions` defines the nine conditions; `loop` runs the ReAct loop and the direct-sample
conditions; `tools` implements the four agent tools; `execution` runs generated code in a
subprocess with a scrubbed environment and timeouts; `stakeholder` simulates the user from the
L0 specification; `llm` is the OpenAI-compatible client with native and text tool transports;
`prompts` keeps every prompt template in one place; `runner` is the resumable CLI;
`report` and `bootstrap` aggregate results; `endpoint_check` probes an endpoint before a run.

## Design principles

- **Determinism and provenance.** Every artifact records its inputs' hashes; splits and
  sampling are seeded. The test suite checks that committed artifacts agree with their
  manifests and recomputes the report and bootstrap numbers from the committed pilot units.
  It does not retrain the detector or rebuild the levels, so changes to that code must be
  checked by rerunning `make detector-smoothed` or `make levels`.
- **No leakage.** The 300 held-out tasks are never used for training, calibration, threshold
  selection or the pilot; hidden tests never enter the agent workspace.
- **CPU-first.** Everything except the R2b pilot (vLLM on an A100) runs on CPU, and the
  harness path is exercised end to end against a fake endpoint in tests.
- **Lazy heavy imports.** `torch` and `sentence-transformers` are imported only inside the
  detector functions that need them, so the rest of the package imports quickly.

## Technology stack

Python 3.12 managed by [uv](https://docs.astral.sh/uv/) with a committed `uv.lock`;
NumPy/SciPy/scikit-learn for the detector and statistics; PyTorch and sentence-transformers
for the frozen encoder; the `openai` client for vLLM; pytest for tests; ruff and mypy for
static checks; LaTeX (NeurIPS 2019 style) for the papers.
