# MLOPS.md — taking this to production

This is a plan for a cloud-native deployment (Kubernetes + an object store + MLflow-style registry; the same ideas map
onto SageMaker/Vertex). Section 7 says what exists in this repo today versus what is only planned.

## 1. What a "model version" is (registry)

The unit of release is **not** just weights. A version is an immutable bundle:

| Part | Global model | Foundation model (Chronos-2) |
|---|---|---|
| Weights / config | `model.joblib` (5 quantile boosters) | model id + **exact HF commit hash** + context length; weights mirrored to our own bucket so serving never depends on the public hub |
| Calibration | `calibration.json` (CQR offsets per horizon bucket and level) | its own `calibration.json` |
| Feature code | git SHA of `models/preprocessing.py` | same (covariate handling + scale) |
| Lineage | training-data manifest hash, split dates, params, seed, library versions | calibration-window manifest, library versions |
| Integrity | SHA-256 verified on load (already implemented) | revision pinned + checksum of mirrored weights |

Rules: offsets are valid only for the model *and* feature code that produced them, so calibration is versioned and
promoted **with** its model, never alone. Registry aliases (`champion`, `challenger`, `previous`) point at versions;
deployments reference an alias resolved to an immutable version at deploy time. Every response reports the version
and calibration window (`model_version` field) so any forecast can be traced back.

Two *variants* per model (with/without exogenous features) are separate registry entries because the UI toggle needs both
and each has its own calibration.

## 2. Online vs batch serving

* **Online** (this service): ad-hoc/unseen series uploaded by a user or a caller, ~1 s for the global model on one CPU.
  The global model and the foundation model run as **separate deployments** behind one gateway route
  (`model=global|chronos`) so a slow or failing foundation model cannot degrade the cheap model. Chronos gets its own
  autoscaling (CPU or GPU), request batching, and a hard timeout. Because forecasts are deterministic, responses are
  cacheable by `hash(file) + params`.
* **Batch**: the production steady state for known series — an hourly/daily job reads fresh history from the
  warehouse, computes features with the *same* `build_origin_features`, scores both models (Chronos in large GPU batches,
  which is far cheaper per series), and writes forecasts + intervals to a table with `model_version`. Downstream
  consumers read the table; the REST endpoint is for what-if/unseen use.
* Both paths share one library, so there is no online/batch skew.

## 3. Monitoring

| Layer | Signals | Alert |
|---|---|---|
| Input data quality | rejection rate by error code, warnings (unknown `cov_cat`, inserted gaps, high missingness), staleness | spike vs 7-day baseline |
| Feature / data drift | PSI or KS per feature group (scale, `cov_*`, missingness rates, `cov_cat` mix incl. share of `other`) vs the training reference | PSI > 0.2 on any group for 3 days |
| Prediction drift | distribution of point forecasts and interval widths per horizon bucket | sudden width inflation/collapse |
| **Performance (needs actuals)** | rolling MAE and **skill vs persistence** per horizon bucket; rolling **coverage at 50% and 90%** per bucket | coverage outside nominal ±0.05 over 7 days; skill < 1 |
| Service | p50/p95 latency, error rate, Chronos timeout/fallback rate, memory | SLO burn |

Actuals arrive with a lag equal to the horizon, so performance metrics are computed by a daily join of stored forecasts
with realised values. Coverage is the headline health metric: it is the first thing to move when the data
distribution shifts (in this repo's own test, no-exog coverage already sits ~2 pts under nominal after the
val→test shift).

## 4. Retraining and recalibration triggers

Cheapest fix first:

1. **Recalibrate** (hours, no retraining): refit CQR offsets on the most recent N months of forecasts-vs-actuals.
   Trigger: coverage out of band for 7 days. Works for both models, and is the *only* adaptation the foundation model
   needs.
2. **Retrain the global model**: scheduled (quarterly) **or** event-driven — sustained skill loss, feature-drift alert,
   ≥ k new series onboarded, upstream schema/units change. Always re-run the leave-one-series-out evaluation, because
   generalisation to new series is the requirement.
3. **Foundation model**: no retraining; change = bump revision, recalibrate, re-run the same evaluation. Consider
   fine-tuning only if it persistently trails the global model on the monitored metric.

## 5. Release and rollback

* **CI gate** (every PR): unit tests; **causality test** (features unchanged when the future is altered);
  **train-serve parity test**; golden-file test (fixed CSV → expected JSON within tolerance); lint; image build.
* **Offline promotion gate**: challenger must beat champion on the time-ordered test window **and** the LOSO
  evaluation, with calibrated coverage within ±0.03 of nominal at both levels.
* **Rollout**: shadow (mirror traffic, log both, no user impact) → canary 5% → 50% → 100%, automated abort on
  error rate, latency, or live coverage/skill guardrails.
* **Rollback** is an alias flip to `previous` (immutable artifacts + immutable image tags; model and calibration move
  together) — seconds, no rebuild. Runbook includes a feature-store snapshot restore if a feature change was involved.
* **Foundation-model failure policy**: circuit breaker → 503 with a structured error. Falling back to the global
  model is allowed only if the response says so explicitly (`model` field reflects what actually ran); silently swapping
  models would corrupt monitoring.

## 6. Security and governance

Joblib/pickle artifacts execute code on load: load only from the registry, only after hash verification (done here),
or move to a non-executable format (LightGBM text model / ONNX). The data is confidential: no raw data in images or git;
the service does not persist uploads; logs record hashes and row counts, not content. Access to registry and buckets is
role-based.

## 7. What is implemented here vs planned

| Implemented in this repo | Planned only |
|---|---|
| Versioned artifact dirs with SHA-256 check at load; `model_version` in responses | Registry service + aliases, signed artifacts |
| Structured run log (`artifacts/runs.jsonl`), data manifest | MLflow tracking server |
| Deterministic seeded training; pinned data stack | Hash-locked full dependency lock; pinned HF revision |
| Validation + error envelope, `/health`, `/models` | Metrics endpoint, tracing, drift jobs, actuals join |
| Offline LOSO + test-window evaluation | CI gate, shadow/canary, automated rollback |
