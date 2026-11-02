# Full results — global model (both variants)

Reproduce: `python -m training.report` (raw numbers in `full_report.json`, `metrics.json`, `runs.jsonl`).
Chronos-2: **not run** (no weights/network where this was built) — there are no Chronos scores.

Trained on 8 series (A,C,D,E,F,G,J,L) with targets < 2016-01-01; conformal offsets fitted on 2016-01..2016-08; test = 2016-09..2017-02.
Origins every 12 h, horizons 1..48. MAE/RMSE/pinball in target units; coverage shown as raw → conformal-calibrated.
Train rows are in-sample (shown to expose over-fitting only).

## Training run
| Variant | Train rows | Features | Fit time (1 CPU) | Artifact sha256 |
|---|---|---|---|---|
| global_exog | 225,294 | 88 | 86.7 s | 107a46892146… |
| global_noexog | 225,294 | 39 | 42.1 s | 69b0f29e6cc2… |

150 trees × 5 quantile models, lr 0.08, 31 leaves, min leaf 200, L2 = 1, seed 42.

## Train / validation / test
| Variant | Window | n | MAE | RMSE | sMAPE % | Pinball | 50% cov | 90% cov | Persistence MAE |
|---|---|---|---|---|---|---|---|---|---|
| exog | train (in-sample) | 772,244 | 38.74 | 60.02 | 55.3 | 11.76 | 0.497→0.589 | 0.893→0.939 | 56.48 |
| exog | validation | 182,064 | 36.72 | 55.11 | 63.7 | 11.41 | 0.431→0.500 | 0.843→0.900 | 43.32 |
| exog | **test** | 134,908 | **55.79** | **84.81** | 68.8 | 17.16 | 0.427→**0.499** | 0.827→**0.892** | 69.87 |
| no-exog | train (in-sample) | 772,244 | 41.33 | 63.71 | 58.4 | 12.49 | 0.497→0.548 | 0.893→0.928 | 56.48 |
| no-exog | validation | 182,064 | 37.43 | 55.32 | 65.1 | 11.56 | 0.462→0.500 | 0.861→0.900 | 43.32 |
| no-exog | **test** | 134,908 | 58.27 | 88.57 | 72.1 | 17.96 | 0.432→0.472 | 0.835→0.879 | 69.87 |

Validation coverage is exactly nominal *by construction* (offsets are fitted on it). Test coverage is the honest check.
The higher test than validation MAE reflects the 2016-09..2017-02 period being harder (persistence MAE also rises 43.3 → 69.9), not over-fitting
(train→validation MAE does not rise).

## Test, by horizon (MAE | RMSE | 50% cov | 90% cov | 90% width)
| Horizon | exog | no-exog | Persistence MAE | Seasonal-naive MAE |
|---|---|---|---|---|
| 1–6 h | 25.9 \| 49.8 \| 0.524 \| 0.897 \| 104 | 25.9 \| 49.4 \| 0.518 \| 0.884 \| 107 | 26.6 | 73.1 |
| 7–12 h | 42.0 \| 67.3 \| 0.527 \| 0.890 \| 163 | 44.6 \| 69.3 \| 0.499 \| 0.891 \| 168 | 49.8 | 71.9 |
| 13–24 h | 54.9 \| 82.6 \| 0.512 \| 0.900 \| 193 | 58.2 \| 85.9 \| 0.497 \| 0.886 \| 195 | 64.7 | 72.5 |
| 25–48 h | 67.2 \| 96.0 \| 0.478 \| 0.887 \| 222 | 69.8 \| 101.0 \| 0.441 \| 0.872 \| 221 | 88.3 | 93.4 |

## Test, by series — MAE (90% coverage), seen in training
| | A | C | D | E | F | G | J | L |
|---|---|---|---|---|---|---|---|---|
| exog | 57.7 (0.891) | 46.1 (0.893) | 61.5 (0.888) | 56.8 (0.902) | 60.9 (0.880) | 46.5 (0.890) | 56.0 (0.896) | 60.6 (0.897) |
| no-exog | 59.9 (0.875) | 48.3 (0.885) | 64.6 (0.871) | 59.4 (0.889) | 62.5 (0.873) | 48.2 (0.877) | 59.0 (0.885) | 64.0 (0.879) |

## Truly unseen series (never in training **or** calibration): series_H, series_I, test window
| Variant | Series | MAE | RMSE | sMAPE % | 50% cov | 90% cov | Persistence MAE |
|---|---|---|---|---|---|---|---|
| exog | H | 59.58 | 88.96 | 69.4 | 0.489 | 0.884 | 75.06 |
| exog | I | 55.85 | 84.00 | 69.5 | 0.502 | 0.889 | 74.58 |
| exog | both | 57.72 | 86.53 | – | 0.495 | 0.887 | 74.82 |
| no-exog | H | 62.49 | 93.84 | 73.5 | 0.458 | 0.866 | 75.06 |
| no-exog | I | 57.85 | 87.28 | 73.3 | 0.474 | 0.869 | 74.58 |
| no-exog | both | 60.18 | 90.64 | – | 0.466 | 0.867 | 74.82 |

## Leave-one-series-out (lighter config: 80 trees, 24 h origin stride; exog variant)
| Held-out | MAE | 90% cov | 50% cov |
|---|---|---|---|
| A | 59.5 | 0.867 | 0.468 |
| C | 48.4 | 0.868 | 0.457 |
| D | 64.0 | 0.861 | 0.475 |
| E | 58.8 | 0.872 | 0.496 |
| F | 63.1 | 0.861 | 0.476 |
| G | 48.0 | 0.872 | 0.503 |
| J | 57.7 | 0.877 | 0.477 |
| L | 63.0 | 0.873 | 0.469 |
| **mean** | **57.81** | **0.869** | **0.478** |

## Code tests (offline, minimal pytest shim)
test_validation 25/25 · test_forecasting 17/17 · test_unseen_series 4/4 · test_api.py not run (no FastAPI in the build sandbox).
