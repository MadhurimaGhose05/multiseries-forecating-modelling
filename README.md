# Multi-series hourly forecasting: global ML model + Chronos-2, with calibrated intervals

Two models on one evaluation protocol, a REST endpoint (`POST /forecast`) that works on series it has never seen, and a
small Streamlit UI. Production plan: [MLOPS.md](MLOPS.md).

## Verification status (read this first)

| Part | Status |
|---|---|
| Data prep, feature code, global model, calibration, evaluation, LOSO | **Run end-to-end** on the 8 distinct series provided; numbers below are real |
| Validation, forecasting, determinism, causality, unseen-series tests (46 tests) | **Pass** (run with a minimal pytest shim in an offline sandbox) |
| FastAPI app, `tests/test_api.py`, Streamlit demo, Dockerfile/compose | **Written, not executed** (FastAPI/Streamlit/Docker unavailable where this was built). Expect to fix small things on first `docker compose up` |
| Chronos-2 wrapper, calibration, evaluation | **Written, not executed** (no network/torch/weights). Parsing contract is unit-tested against a stub only. Run `python -m training.train_chronos --smoke` first. **No Chronos numbers are reported below because none exist yet.** |
| Dependency pins | Data stack pinned exactly (artifacts are scikit-learn pickles → same versions required). Serving/UI/Chronos pins are ranges; generate a hash lock (see `requirements.txt`) and pin the Chronos weights revision |

## Quick start

```bash
# 0. data (not in git, confidential): put the series CSVs in data/raw/, then
python -m training.prepare_data --src data/raw      # dedupes identical files, writes manifest

# 1. everything: train global (exog / no-exog), calibrate, evaluate
python train.py --src data/raw                       # add --chronos (needs weights) and/or --loso

# 2. serve + demo
docker compose up --build                            # API :8000, UI :8501
#   or locally: uvicorn app.main:app  &  streamlit run demo/streamlit_app.py

# 3. tests
pytest -q
```

Trained artifacts for the global model are included in `artifacts/`, so the API/UI work without retraining
(Chronos additionally needs `python -m training.train_chronos` to create its calibration file; without it
`model=chronos` returns a structured `503 model_unavailable`).

## Data observations that shaped the design

* 8 distinct hourly series (A, C, D, E, F, G, J, L), 2013-03-01 → 2017-02-28; `series_A_1.csv` / `series_D_1.csv` are byte-identical copies and are skipped.
* Target is very persistent (lag-1 autocorr ≈ 0.97) but weakly seasonal (lag-24 ≈ 0.4, lag-168 ≈ 0) → weekly features add little; error is dominated by horizon.
* `aux_1` is a near-copy of `target` (corr ≈ 0.88, equal in ~20–27% of rows) and is observed *with* the target → **lags only**, never same-time-in-the-future.
* Target missingness is 1.8–2.7% but peaks in a different year per series, with multi-day gaps → missingness is a feature, not just something to impute.
* `cov_cat` has 16 compass buckets in every series.

## Split (no leakage)

Applied identically to every series, on the **target timestamp** of each (origin, horizon) example:

| Window | Targets | Used for |
|---|---|---|
| Train | < 2016-01-01 | fitting the global model (an assertion checks no label ≥ boundary) |
| Validation | 2016-01-01 → 2016-09-01 | conformal calibration of **both** models |
| Test | 2016-09-01 → 2017-03-01 | metrics only |

Test/validation origins are every 12 h with the whole 48 h path inside the window; both models are scored on the same origins.
Features at an origin use data ≤ origin only; `tests/test_forecasting.py` proves this by altering/removing the future and asserting identical features.

## Models

**Global ML model** (`models/global_model.py`): scikit-learn `HistGradientBoostingRegressor`, one model per quantile
(0.05/0.25/0.5/0.75/0.95), *direct multi-horizon* (horizon `h` is an input; trained on h ∈ {1,2,3,4,6,8,12,…,48}, served for any 1..48).
Chosen over LightGBM/TFT because it handles quantile loss, native categoricals and NaNs, is seeded/deterministic, and keeps the
dependency footprint small; with LightGBM available the swap is local to this file.

* **Normalisation:** targets, lags and rolling stats are divided by the trailing-168 h target mean at the origin; `log_scale` is a feature. This is what lets one model score a series with a level it never saw. There is **no `series_id` feature** (a test asserts the forecast is invariant to the id).
* **Target features:** 13 lags (up to 168 h), rolling mean/std/min/max over 6/24/168 h.
* **Data-quality features:** hours since last observation, missing fraction in last 24/168 h, aux missing fraction.
* **Calendar of the target time:** hour (sin/cos), day of week, month (sin/cos).
* **Exogenous (toggle):** `aux_*` as ratios to their own weekly mean at lags 0/1/6/24 + 24 h mean; `cov_1..5` raw at origin, 3 h/24 h change, 24 h mean; `cov_cat` as a **native categorical** over a fixed 16+`other` vocabulary (identical codes in training and serving) plus sin/cos of the compass angle so trees can use direction ordering.

**Foundation model: Chronos-2, zero-shot** (`models/chronos_model.py`): last 1024 h of history as context, `aux_*`, `cov_*` and `cov_cat` as *past* covariates (Chronos-2 accepts numeric and categorical), native quantiles. Nothing is trained; only the conformal offsets are fitted. Forward-fill of covariates is bounded identically to the global model.

## Exogenous features at serving time

The caller uploads **history only**. Assumptions, all enforced by one code path used in training and serving:

| Signal | Assumed known at forecast time? | Treatment |
|---|---|---|
| `target` | up to the last uploaded timestamp | lags/rolling stats; gaps stay `NaN` (+ missingness features), never interpolated |
| `aux_1..5` | observed with the target → **unknown for the future** | lagged/rolling only (lag 0 = value at the origin) |
| `cov_1..5`, `cov_cat` | only up to the last timestamp (conservative: **no future covariate forecasts accepted**) | value at origin + backward-looking tendencies; gaps ≤ 6 h forward-filled, longer gaps left `NaN` |
| Calendar | always | derived from the target time |
| Unknown `cov_cat` label | – | mapped to `other` + a `warnings` entry (never an error) |

**Why past-only?** The brief does not say future covariates are available, and if they are weather-like, the *future* values
would themselves be forecasts with their own error. A natural next step is an optional `future_covariates` block; it is
deliberately not in the API so that the contract cannot be misused to leak.
The UI toggle ("Use extra signals") switches between the with-exog and no-exog model variants (separately trained and
separately calibrated), so the effect on both forecast and interval width is visible.

## Uncertainty

Both models emit raw quantiles (0.05, 0.25, 0.5, 0.75, 0.95). The **50% interval** is [q25, q75] and the **90% interval** is [q05, q95],
each corrected with **conformalized quantile regression** (CQR) fitted per horizon bucket (1–6, 7–12, 13–24, 25–48 h) on the validation window, using scores
normalised by the origin scale so offsets transfer to unseen series.

* *Why quantile models for the global model:* intervals adapt to the situation (width varies with level, horizon and recent volatility) rather than a constant ± residual band.
* *Why CQR:* the raw quantiles under-cover (below). CQR gives a distribution-free finite-sample guarantee **under exchangeability**; time series violate that, so coverage is *verified* on the untouched test window rather than assumed.
* *Same treatment for Chronos* so the two models' intervals are comparable.

## Results (global model; test window 2016-09 → 2017-02, all 8 series, h = 1..48)

| Model | Exog | MAE | RMSE | sMAPE % | 50% cov (raw → calibrated) | 90% cov (raw → calibrated) | 90% width |
|---|---|---|---|---|---|---|---|
| Global GBM | yes | **55.79** | **84.81** | 68.8 | 0.427 → **0.499** | 0.827 → **0.892** | 192.4 |
| Global GBM | no | 58.27 | 88.57 | 72.1 | 0.432 → 0.472 | 0.835 → 0.879 | 193.6 |
| Chronos-2 (zero-shot) | yes / no | *not run yet* | | | | | |
| Baseline: persistence | – | 69.87 | | | | | |
| Baseline: seasonal-naive 24 h | – | 82.96 | | | | | |

By horizon (global, exog): MAE 25.9 (1–6 h; persistence 26.6) → 42.0 (7–12) → 54.9 (13–24) → 67.2 (25–48; persistence 88.3). Exogenous features
help from ~7 h onward (e.g. 54.9 vs 58.2 at 13–24 h) and are neutral at 1–6 h, where recent history dominates. 90% coverage per horizon bucket is 0.887–0.900 and
per series 0.880–0.902. sMAPE is high because the target is spiky and low-valued at times; MAE/RMSE are the more reliable reading.

**Raw quantiles under-cover** (50% → 0.43, 90% → 0.83); calibration brings the exog model to 0.499 / 0.892 on the test window.
The no-exog model keeps a ~2-pt shortfall (0.879) after the validation→test shift: conformal offsets are not magic under drift (see MLOPS.md, recalibration).

### Unseen series (leave-one-series-out)

Each fold trains **and calibrates** on 7 series and scores the 8th on the test window. To keep this affordable on one CPU the folds use a lighter configuration (80 trees, origins every 24 h), so part of the gap below is the weaker model, not only unseen-ness.

| | MAE (mean of 8 folds) | 90% coverage | 50% coverage |
|---|---|---|---|
| LOSO, series unseen | 57.81 | 0.869 | 0.478 |
| Same series, seen in training (full model) | 55.76 | ≈ 0.892 | 0.499 |
| Persistence | 69.8 | | |

Per-series LOSO MAE ranges 48.0–64.0, always better than persistence on average. **Intervals on a genuinely new series run about 3 pts narrow at 90%** (every fold between 0.861 and 0.877). Honest takeaway:
point accuracy transfers well, calibration transfers imperfectly; recalibrating on the new series' own recent history (MLOPS.md §4) is the fix.

## API

`POST /forecast` (multipart/form-data)

| Field | Type | Default | Notes |
|---|---|---|---|
| `file` | CSV | required | full schema; hourly; ≥ 14 days; ≤ 25 MB |
| `horizon` | int 1–48 | 24 | hours ahead |
| `interval` | 50 \| 90 | 90 | |
| `model` | `global` \| `chronos` | `global` | |
| `use_exog` | bool | true | |
| `series_id` | string | – | only if the file holds several series |

```bash
curl -F file=@series_L.csv -F horizon=24 -F interval=90 -F model=global -F use_exog=true localhost:8000/forecast
```

Returns `{series_id, model, use_exog, horizon, interval, origin, forecast:[{timestamp, point, lower, upper}], mean_interval_width, warnings, model_version}`.
`POST /backtest` (same inputs + `n_origins`) replays daily origins over the last days of the *uploaded* file (history ≤ origin only) and returns MAE, RMSE, sMAPE, coverage, width and the persistence MAE — this feeds the UI metrics card.
`GET /health`, `GET /models` (artifact hashes, calibration window, Chronos revision).

**Errors** always use `{"error": {"code", "message", "details"}}`: `400 empty_file | malformed_file`, `413 file_too_large`,
`422 schema_mismatch | invalid_timestamp | duplicate_timestamps | irregular_frequency | non_numeric_values | insufficient_history | insufficient_recent_data | stale_target | multiple_series | unknown_series_id | invalid_horizon | invalid_interval | unknown_model | invalid_request`, `503 model_unavailable`, `500 internal_error`.

**No train-serve skew:** the endpoint recomputes every feature from the uploaded history via `models.preprocessing`, the same functions used to build training data (a test compares offline features with features recomputed from the CSV).
**Determinism:** seeded training (`random_state=42`), exact-pinned stack, no sampling at inference, SHA-256 verified artifact loading; a test asserts identical output on re-run.

## Demo UI

Upload CSV → choose model → toggle extra signals → pick horizon / 50 or 90% band. The chart shows observed history, forecast and shaded band, with the
toggled alternative as a dashed overlay; cards show band width and, from a 14-day replay on *your* file, MAE vs "same as now", and coverage vs nominal. A "How to read this page" panel explains it for non-technical reviewers.

## Repository map

`app/` API, validation, serving logic, conformal code · `models/` features, global model, Chronos wrapper · `training/` prepare, train, calibrate, evaluate ·
`demo/` Streamlit · `tests/` · `artifacts/` model versions, calibration, metrics, run log · `train.py` orchestrator.

## What I would do next

1. Run Chronos-2 on a machine with the weights, pin its revision, and report it in the table above (the largest open item).
2. Optional `future_covariates` input (and a model trained with them) if weather forecasts are available in production; compare against past-only.
3. Adaptive conformal calibration (rolling window / ACI) to close the ~3-pt unseen-series and drift shortfall; per-series online recalibration.
4. Train on more series than 8 and on more origins/trees (the final model used a 12 h origin stride and 150 trees for CPU time); tune hyper-parameters on the validation window; try LightGBM and a TFT.
5. Probabilistic point metric (CRPS/pinball) in the results table; Winkler score for intervals.
6. Hash-locked dependencies, CI with the gates in MLOPS.md, drift/coverage monitoring jobs.
