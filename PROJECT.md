# What Does a Wish Leave Out?

Requirement recovery in wish-driven code agents. Columbia COMS 4705.

A wish-driven coding agent receives an intent and builds an artifact. When the wish leaves a
requirement implicit, the program can run, pass the tests the user remembered to mention, and
still violate what the user meant. This project mechanically degrades benchmark
specifications into **wishes** by deleting fields, trains a small **silence detector** that
predicts which requirement categories a wish leaves implicit, and uses that ranking to target
clarification questions.

The accepted final milestone is [`paper/milestone/main.pdf`](paper/milestone/main.pdf)
(source: [`paper/milestone/main.tex`](paper/milestone/main.tex)); the extended measured report
is [`paper/main.pdf`](paper/main.pdf). The submitted milestone is preserved byte-identically
under [`paper/milestone/submitted/`](paper/milestone/submitted/).

## Milestone design

- **Generator:** Qwen3-8B-AWQ served by vLLM 0.34.0 on one A100-SXM4-40GB, temperature 0.7,
  top-p 0.95, at most twelve ReAct turns, three samples per task.
- **Detector:** frozen Qwen3-Embedding-0.6B encoder with a five-label linear head, trained
  with binary cross-entropy and an L2 penalty, calibrated out of fold on the 500 train tasks.
- **Data:** HumanEval+, MBPP+ and BigCodeBench; 1,100 kept tasks in a 500 detector-train /
  200 evolution / 300 held-out split, plus a disjoint 100-task development pilot.
- **Pilot:** the 100 development tasks with three samples per condition, 2.902 GPU-hours on
  one A100-SXM4-40GB and $0 paid API.
- **Conditions:** B1 (full L0 specification, ceiling), H0 (plain loop), H1 (acceptance tests
  first), H2 (generic questions), H3 (detector-targeted questions), H4 (sibling functions
  with doctests), B2 (five direct samples instead of questions), B3 (one call, wish and stub
  only), H3⁻ (H3's questions with the answers withheld). H5, offline harness evolution,
  remains future work.
- **Metrics:** pass@1 and *recovery*, the fraction of the L0–L3 gap closed, plus per-category
  recovery by task subset and failure buckets by exception type.

## The five fields and the levels

Every source task is parsed mechanically into five fields:

| field | HumanEval+ / MBPP+ | BigCodeBench `doc_struct` |
| --- | --- | --- |
| `intent` | whole prose block before the examples | `description` |
| `signature_types` | typed def / canonical def | `params` + `returns` |
| `examples` | doctest block / `test_list` | `examples` |
| `constraints` | notes / constraints prose | `notes` + `reqs` |
| `error_behaviour` | Raises / exception prose | `raises` |

Empty fields stay empty; a field is never invented. The levels are built by mechanical field
deletion:

- **L0** keeps every present field (full specification).
- **L1** drops examples and type detail; the signature stays, untyped.
- **L2** additionally drops constraints and error behaviour, keeping intent and the untyped
  signature.
- **L3** keeps only the intent prose.
- **L2_c = L0 − c** drops exactly one field category; the nine-condition pilot does not run
  these single-field conditions.

One stub, one hidden suite and three visible tests are shared by all levels of a task. The
hidden suite is byte-identical across L0–L3, so a level difference is information loss rather
than test variation. The hidden suite is never shown to the generator or the stakeholder.

## Data foundation

### Pinned data

| source | revision | file | rows |
| --- | --- | --- | --- |
| HumanEval+ | `d32357cf319e50e9c8d8dab5ea876c72b0fd321b` | `test.jsonl` | 164 |
| MBPP+ | `b2d74c91837c3f2a20c1299ae98133cbe7cfa077` | `data/test-00000-of-00001-d5781c9c51e02795.parquet` | 378 |
| BigCodeBench | `b74c0d0bf70d2c0bc459be537895cca163007f1a` | `data/train-00000-of-00001.parquet` | 1,140 |

Downloads are checksum-verified against the SHA-256 recorded in `src/wishrec/config.py` and
the artifact manifest, and are idempotent in gitignored `data/raw/`.

### Tests and validation

- **HumanEval+:** the visible test prints the observed output for the first three benchmark
  inputs, with expected values removed; the hidden test is the full `check()` over all cases.
  Tasks whose test embeds a `ref_func` reference implementation are excluded because the
  visible test would leak that reference.
- **MBPP+:** the visible test prints the observed output for the first three `test_list`
  assertions; the hidden test is the full test module.
- **BigCodeBench:** the hidden test is the full `unittest` class. BigCodeBench's official
  harness concatenates the solution and the test module, so tests may use imports and
  constants from the solution module; the derived hidden test therefore prepends
  `from solution import <entry_point>` and a `from solution import *` compatibility line. The
  visible test keeps the imports, the `unittest` class, `setUp`/`tearDown`, helper methods and
  class attributes, plus exactly three `test_*` methods chosen in source order. Because those
  methods assert hard-coded expectations, BigCodeBench remains assertion-based under the
  corrected protocol; this is a stated residual.

Every kept task's canonical solution is executed against both its visible and hidden tests in
an isolated subprocess with hard timeouts. Run results are cached by task content, validator
version and installed package versions. Excluded tasks and the exact exception, missing
library or timeout are recorded in the manifest and `stats.json`.

### Split and counts

The fixed seed `20261121` drives selection and splitting. All valid HumanEval+ and MBPP+ tasks
are retained, and a seeded uniform BigCodeBench sample fills the 1,000-task main pool; the
pilot is a further disjoint 100 tasks (1,100 kept in total). Each split is source-stratified,
and the 300 held-out tasks are never assigned to detector training, calibration, threshold
selection or the pilot.

The committed build parsed 1,651 of 1,682 source rows and validated 1,549; 1,100 are kept:
132 HumanEval+, 377 MBPP+ and 591 BigCodeBench. The 133 exclusions are 31 parse, 86
visible-test and 16 hidden-test failures. The pilot's source mix is 12 HumanEval+, 34 MBPP+
and 54 BigCodeBench. The build takes 666 seconds on eight cores.

### Artifacts

`results/levels/` commits:

- `tasks.jsonl` — one record per kept task: split, fields and presence, L0–L3,
  `deleted_fields`, `L2_c`, stub, visible assertions and hashes, hidden/canonical/stub/test
  hashes, third-party requirements and per-task metadata.
- `manifest.json` — provenance, pins, counts, split, exclusions, environment and artifact
  checksums.
- `stats.json` — per-source field prevalence, split counts, library requirements, level
  lengths and validation summary.

Hidden tests and canonical solutions are written to gitignored `data/derived/levels/` and are
regenerated by `make levels`; `tasks.jsonl` records their SHA-256. `make levels` defaults to
the committed inputs-only visible-test mode; `VISIBLE_TESTS=full` rebuilds the older full
mode.

## Harness

Each task is materialized in a fresh temporary workspace containing `solution.py` (the stub)
and `visible_test.py` (the derived visible suite). The agent can call exactly four tools:

- `write_file` writes `solution.py`; `acceptance_tests.py` is writable in H1 only, and no test
  file is writable;
- `run_visible_tests` executes the derived visible file in a subprocess with a timeout and
  returns pass/fail;
- `read_error` returns the traceback of the most recent failed visible run;
- `ask_stakeholder` asks one question and consumes the condition's dialogue budget.

The loop allows at most 12 assistant turns and has no repository browsing or shell tool. The
hidden suite is never in the workspace; after the loop, a separate evaluator writes
`solution.py` and the hidden test into a fresh directory and runs the full suite. The
stakeholder is a temperature-0 model whose prompt contains only the L0 specification; it
answers "That is not specified." when the specification does not contain the answer.

The generator is `Qwen/Qwen3-8B-AWQ` served by vLLM behind an OpenAI-compatible API. The
client supports native OpenAI tool calls and a JSON text protocol (`{"tool": ...,
"arguments": ...}`) for builds that do not return tool calls; `auto` mode probes native calls
once, retries the turn under the text protocol when the endpoint ignores tools, and remembers
the working transport. Endpoint, key, model, timeout, transport, thinking mode and an optional
B3 frontier endpoint are configured by `VLLM_*` environment variables;
[`configs/vllm.env.example`](configs/vllm.env.example) is the documented example. No
credential is written to an artifact.

### Conditions

All conditions share the same tasks, tools and visible tests. B1 sees the complete L0
specification; H0 is the plain wish loop; H1 may write acceptance tests first; H2 asks generic
stakeholder questions through the tool within a five-question budget; H3 asks the
detector-selected 2–3 questions before the loop and then runs one self-check-and-repair call;
H4 adds five sibling functions with doctests; B2 draws five direct samples instead of asking;
B3 calls the endpoint with only the wish and stub (the same Qwen3-8B at temperature 0.7 when
no frontier endpoint is configured); H3⁻ asks H3's questions but withholds the answers,
separating asking from knowing. H5 is a plan item and is not implemented.

## Detector

The detector is the frozen `Qwen/Qwen3-Embedding-0.6B` encoder (1024-d, CPU) plus a five-label
linear head trained with BCE and an L2 penalty. Labels are the five fields' presence in the
full L0 specification, so a positive means the wish omits information the stakeholder can
supply. It trains on the 500 train tasks only; the 200 evolution, 300 held-out and 100 pilot
tasks are excluded from fitting, calibration, threshold selection and model selection.
Five-fold out-of-fold evaluation uses inner-fold L2 selection and Platt calibration, and
compares against a prior-only predictor and TF-IDF + one-vs-rest logistic regression. Metrics
are AUROC, average precision, F1 at 0.5, macro-F1, Brier, ECE and log loss per category.

The final refit uses the 500 smoothed train wishes (12 of 600 smoothing calls fell back to the
raw intent), 600 full-batch AdamW steps at learning rate 0.01 and L2 0.3. Intent and
signature/types are positive on all 500 train tasks, so their AUROC is undefined; the
detector is effectively a three-label head with 293 constraints, 491 examples and 85
error-behaviour positives.

The H3 policy ranks calibrated presence times utility (intent 0, examples 0.2, signature/types
0.5, constraints 1.0, error behaviour 1.0) and calibrates a threshold to 2–3 questions per
task; it never asks about intent. On the deployed-style out-of-fold scores it asks 2.48
questions per task (260 tasks get 2, 240 get 3), with informative-pair precision 0.835,
recall 0.554 and utility recall 0.856. Selection rates are 0.70 constraints, 0.37 error
behaviour, 0.42 examples, 1.00 signature/types and 0 intent.

### Detector results

Five-fold out-of-fold detection on the 500 smoothed train tasks
([`results/detector-smoothed/`](results/detector-smoothed/)):

| method | macro-F1 | macro-F1 (varying) | AUROC examples | AUROC constraints | AUROC error | Brier | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| prior-only | 0.746 | 0.577 | 0.251 | 0.442 | 0.472 | 0.0806 | 0.0096 |
| TF-IDF + logistic | 0.789 | 0.648 | 0.871 | 0.947 | 0.695 | 0.0470 | 0.0131 |
| frozen Qwen3 + linear head | 0.792 | 0.654 | 0.632 | 0.952 | 0.744 | 0.0439 | 0.0142 |

The error-behaviour label is the weak one: 85 positives, average precision 0.315, and F1 at
the deployed 0.5 threshold 0.046 because calibrated probabilities stay near the 17% base
rate. Examples are nearly constant (491/500 positive), so their AUROC uses only 9 negatives.

## Measured pilot

One A100-SXM4-40GB Colab session served Qwen3-8B-AWQ on vLLM 0.34.0. The build did not
expose native tool calls, so every unit used the JSON text protocol. All 2,700 units
completed on the 100 development tasks with no held-out task: 2.902 GPU-hours on two
config-stable shards (1.385 h and 1.517 h), 19,356,531 tokens (18,075,167 input, 1,281,364
output), $0 paid API.

| condition | pass@1 | pass rate | recovery | questions | turns |
| --- | --- | --- | --- | --- | --- |
| B1 | 0.623 | 0.623 | 1.000 | 0.00 | 4.34 |
| H0 | 0.520 | 0.520 | 0.000 | 0.00 | 4.40 |
| H1 | 0.287 | 0.287 | −2.259 | 0.00 | 4.51 |
| H2 | 0.443 | 0.443 | −0.743 | 2.94 | 6.36 |
| H3 | 0.537 | 0.537 | 0.162 | 2.48 | 4.43 |
| H4 | 0.517 | 0.517 | −0.032 | 0.00 | 4.74 |
| B2 | 0.513 | 0.547 | −0.065 | 0.00 | 5.00 |
| B3 | 0.310 | 0.310 | −2.033 | 0.00 | 1.00 |
| H3⁻ | 0.427 | 0.427 | −0.903 | 2.48 | 4.09 |

The paired task-level bootstrap (2,000 resamples, seed `20261121`, each task contributing the
mean of its three pass@1 samples) gives:

| contrast | difference (pp) | 95% CI (pp) |
| --- | --- | --- |
| B1 − H0 | +10.3 | [+3.0, +17.7] |
| H2 − H0 | −7.7 | [−14.3, −1.0] |
| H3 − H0 | +1.7 | [−5.7, +9.0] |
| H4 − H0 | −0.3 | [−6.0, +5.3] |
| H1 − H0 | −23.3 | [−31.3, −15.7] |
| H3 − H2 | +9.3 | [+1.7, +17.0] |
| H3 − H4 | +2.0 | [−6.3, +9.7] |
| H3⁻ − H3 | −11.0 | [−18.7, −2.7] |
| B2 − H0 | −0.7 | [−6.7, +5.0] |

Targeting is the only clarification condition that beats another, and it does not close the
specification gap or separate from H0 and H4. Generic questions lose 7.7 points against H0,
withholding H3's answers loses 11.0 points against H3, and acceptance tests first loses 23.3
points against H0 under inputs-only visible tests. Per-category grouping (tasks grouped by a
category their L0 carries, so categories overlap and are attribution by subset):

| category | tasks | B1 | H0 | H2 | H3 | H4 | H3 recovery |
| --- | --- | --- | --- | --- | --- | --- | --- |
| signature and types | 100 | 0.623 | 0.520 | 0.443 | 0.537 | 0.517 | 0.162 |
| examples | 99 | 0.623 | 0.522 | 0.448 | 0.532 | 0.522 | 0.100 |
| constraints | 57 | 0.632 | 0.450 | 0.380 | 0.485 | 0.450 | 0.194 |
| error behaviour | 10 | 0.833 | 0.400 | 0.367 | 0.433 | 0.467 | 0.077 |

Across all 2,700 units there are 1,263 passes and 1,437 failures under the any-sample rule
(1,253 under first-sample pass@1). AssertionError dominates (858, 59.7%), followed by
NoSolution (281, 19.6%), TypeError (92, 6.4%), ValueError (63, 4.4%) and AttributeError (61,
4.2%); 26 units time out and 8 end in XMLSyntaxError.

## Limitations

The pilot uses the 100-task development split, one seed base, three samples per condition,
one A100-40GB session and the JSON text protocol; its intervals cover task-level variation
only. The 300 held-out tasks remain unrun, and the second seed and H5 are future work. The
BigCodeBench visible suite is assertion-based, so its examples field is not fully deleted.
The detector is a frozen encoder with a linear probe; two labels are near-constant, and the
error-behaviour label is sparse. Recovery uses B1 and H0 on the same 100 tasks, so it is a
descriptive normalization.

### Findings from the code review

A review of the frozen code and artifacts found the following. None changes a committed
artifact; the two analyses are reproducible with `make analysis` (see
[`results/analysis/`](results/analysis/)).

**Detector signal is largely source identity.** A predictor that only knows the source
benchmark, fit on the same outer folds, matches or beats the encoder: out-of-fold AUROC
0.963 vs 0.952 for constraints, 0.921 vs 0.632 for examples and 0.722 vs 0.744 for error
behaviour. Within a single source the encoder is near chance (constraints 0.468 on HumanEval+
and 0.534 on MBPP+; error behaviour 0.516 on BigCodeBench), and centring its scores by source
leaves AUROC 0.44–0.58. H3's questions are still sensible, because asking a BigCodeBench task
about constraints is correct, but the detector evidence shows source-conditional priors, not
per-wish silence detection ([`source_baseline.md`](results/analysis/source_baseline.md)).

**Loop termination and stub-passing visible tests.** 281 pilot units ended with no solution
(H1 146, H3⁻ 55, H2 54, B1 10, H0 9, H4 6, H3 1). 129 ended because the harness stopped on a
reply it could not parse (57 malformed tool JSON, 14 truncated, 58 prose); the pilot loop
treated any turn without a parseable tool call as finishing. The other 152 were explicit
finishes, and 125 units overall finished after the inputs-only visible tests had passed on
the unchanged stub. Both mechanisms concentrate in H1, H2 and H3⁻, so contrasts involving
those conditions partly measure termination behaviour. Excluding the harness-terminated units
moves H1 from 0.287 to 0.381 pass@1 and H2 from 0.443 to 0.477 (a descriptive sensitivity
check, not a corrected result; [`nosolution_causes.md`](results/analysis/nosolution_causes.md)).
The harness now returns feedback on unparseable replies instead of ending the loop; the
committed pilot predates that change.

**Field extraction.** MBPP+ intent is the first prose sentence, not the whole prose block;
later sentences become constraints or error behaviour and leave L2/L3 (31 tasks, e.g.
`mbppplus/607` loses its return requirement). For HumanEval+, labelled Note/Raises sentences
are copied into constraints and error behaviour but stay in the intent, so all 19 constraint
and all 5 error-behaviour fields remain verbatim in L3 although their labels mark them
deleted; 8 HumanEval+ tasks with an empty examples field carry I/O examples inside the L3
wish. The intent check accepts only a short verb list (return, create, write, compute, find,
count, implement), which excludes 26 HumanEval+ tasks that begin "Check…", "Filter…",
"Add…" and similar.

**Split stability and B2 seeds.** Each source is shuffled as a whole, so a rebuild in an
environment that validates a different task set reassigns many tasks, including held-out ones
(dropping one unused task moved 97 held-out tasks). The build now refuses to change a
committed split without `--allow-resplit`. B2's three samples per task use overlapping seed
windows (`seed + sample + i`), so they share four of five draws; B2 pass@1 is unaffected, but
its any-of-five pass rate averages three overlapping estimates.

## Repository layout

```
src/wishrec/            data pipeline (download, parse, levels, split, validation, artifacts)
src/wishrec/harness/    ReAct loop, four tools, conditions, stakeholder, vLLM client, runner
src/wishrec/detector/   five-label detector, encoder, CV, calibration, metrics, policy
configs/                vLLM endpoint example (no secrets)
tests/                  unit tests, fake-endpoint integration tests and artifact checks
results/levels/         committed level artifact (manifest, stats, tasks.jsonl)
results/detector/       earlier raw-label detector artifact
results/detector-smoothed/  final detector refit on smoothed train wishes
results/runs/r2b-pilot/     measured GPU pilot evidence and executed notebook
notebooks/              clean R2b notebook and short Colab runbook
paper/milestone/        accepted final milestone; submitted/ preserved byte-identically
paper/main.tex          extended measured report
data/                   gitignored downloads, validation cache and derived hidden tests
```

## Budget and reproduction

The measured pilot used 2.902 GPU-hours on one A100-SXM4-40GB and $0 paid API. Data
preparation and the detector refit are CPU-only. A fresh clone reproduces everything except
the GPU generations:

```bash
uv sync --frozen
make test
make levels
make detector-smoothed
uv run python scripts/local_fake_dry_run.py
```

The measured pilot details are in [`COLAB.md`](COLAB.md) and
[`results/runs/r2b-pilot/README.md`](results/runs/r2b-pilot/README.md).
