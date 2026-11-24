# Offline analyses

Regenerate with `make analysis`. Both analyses run on CPU, read only committed
artifacts, and write deterministic output (no timestamps). Rerunning them reproduces these
files byte-for-byte, and `tests/test_analysis_*.py` checks that.

| File | Produced by | What it shows |
| --- | --- | --- |
| `source_baseline.json` / `.md` | `python -m wishrec.analysis.source_baseline` | How much of the detector's out-of-fold ranking a source-only predictor reproduces. The predictor is the Laplace-smoothed per-source label rate on the same outer folds, which are recomputed and verified against the committed prior column. It also reports encoder AUROC within each source and after centring scores by source. Input: `results/detector-smoothed/cv_predictions.jsonl`. |
| `nosolution_causes.json` / `.md` | `python -m wishrec.analysis.nosolution_causes` | Why loop units ended without writing `solution.py` (explicit finish, truncated output, unparseable tool JSON, prose reply, …). It also counts units whose visible tests passed on the stub, and gives a pass@1 sensitivity check with harness-terminated units excluded. That check is descriptive only, not a corrected result. Input: `results/runs/r2b-pilot/units/`. |

Within-source AUROCs computed from only a handful of positives (for example 1 of 171) are
too noisy to interpret.
