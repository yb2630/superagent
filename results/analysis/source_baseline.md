# Source-only baseline for the silence detector

500 train tasks, 5 outer folds (seed 20261121), out-of-fold scores. Source-only = Laplace-smoothed per-source label rate on each training fold.

| label | positives | source-only AUROC | encoder AUROC (cal.) | encoder AUROC (raw) | raw, centred by source | source-only AP | encoder AP (cal.) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| examples | 491 | 0.921 | 0.632 | 0.696 | 0.442 | 0.998 | 0.990 |
| constraints | 293 | 0.963 | 0.952 | 0.951 | 0.578 | 0.980 | 0.974 |
| error_behaviour | 85 | 0.722 | 0.744 | 0.732 | 0.501 | 0.267 | 0.315 |

## Encoder AUROC within each source (calibrated / raw; positives of n)

| label | bigcodebench | humanevalplus | mbppplus |
| --- | --- | --- | --- |
| examples | — / — (269/269) | 0.386 / 0.481 (51/60) | — / — (171/171) |
| constraints | — / — (269/269) | 0.468 / 0.480 (10/60) | 0.534 / 0.524 (14/171) |
| error_behaviour | 0.516 / 0.512 (80/269) | 0.728 / 0.576 (4/60) | 0.906 / 0.824 (1/171) |

## Label rate by source

| source | n | examples | constraints | error_behaviour |
| --- | --- | --- | --- | --- |
| bigcodebench | 269 | 1.00 | 1.00 | 0.30 |
| humanevalplus | 60 | 0.85 | 0.17 | 0.07 |
| mbppplus | 171 | 1.00 | 0.08 | 0.01 |

AUROC is undefined (—) when a source has only one class for a label.
