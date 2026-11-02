# Global model vs Chronos-2 — all 10 series, test window (2016-09-01 → 2017-03-01)

All rows are **measured** (Chronos-2 run on Google Colab, zero-shot, weights `amazon/chronos-2`, revision `main` — **not pinned to a commit hash**).

**Protocol (identical for both models):** forecast origins every **48 h** (90 origins per series on test, 122 on validation), horizons 1–48 h,
intervals = raw quantiles → conformal (CQR) offsets fitted per horizon bucket on the validation window. MAE/RMSE/pinball in target units.
Coverage is shown raw → calibrated. Persistence = last observed value carried forward.
These numbers use stride 48, so they differ slightly from the stride-12 global-model tables in `FULL_RESULTS.md` (e.g. global+exog MAE 57.02 here vs 56.17 over
10 series at stride 12): different origins, same models. Compare rows **within** this file.

**"Seen"** (global model only): trained on A, C, D, E, F, G, J, L; calibrated on those 8. **H and I were never used by the global model for fitting or calibration.**
Chronos-2 is zero-shot, so no series was "trained on"; its conformal offsets were fitted on the validation window of **all 10** series.

## Summary (pooled over all 10 series)

| Model | Exogenous signals | MAE | RMSE | sMAPE % | Pinball | 50% cov (raw→cal) | 90% cov (raw→cal) |
|---|---|---|---|---|---|---|---|
| **Chronos-2** | **no** | **54.98** | **85.53** | 67.3 | **17.03** | 0.424→**0.506** | 0.819→0.884 |
| Global GBM | yes | 57.02 | 86.15 | 69.9 | 17.42 | 0.422→0.494 | 0.829→**0.901** |
| Global GBM | no | 59.09 | 88.36 | 73.0 | 18.10 | 0.426→0.465 | 0.837→0.884 |
| Chronos-2 | yes | 62.55 | 93.44 | 73.9 | 19.28 | 0.398→0.445 | 0.818→0.858 |
| Baseline: persistence | – | 74.91 | | | | | |

Per-series MAE (lowest in bold):

| Series | Seen by global | Global + exog | Global no exog | Chronos-2 + exog | Chronos-2 no exog |
|---|---|---|---|---|---|
| A | yes | 58.62 | 60.48 | 65.46 | **56.58** |
| C | yes | 48.96 | 50.11 | 51.79 | **46.70** |
| D | yes | 61.24 | 64.07 | 68.33 | **60.98** |
| E | yes | 57.84 | 58.64 | 62.96 | **56.17** |
| F | yes | 61.17 | 62.85 | 66.96 | **59.25** |
| G | yes | **46.43** | 48.57 | 51.17 | 46.69 |
| H | **no** | 59.99 | 62.49 | 66.14 | **56.36** |
| I | **no** | 58.55 | 60.21 | 61.73 | **54.14** |
| J | yes | 57.09 | 59.97 | 62.93 | **54.81** |
| L | yes | 60.13 | 63.35 | 67.75 | **57.92** |

## What the numbers say (and what they don't)

* **Every variant beats persistence on every series.** The ranking by pooled MAE is Chronos-2 (no exog) < global+exog < global (no exog) < Chronos-2 (+exog).
* **Chronos-2 on the target alone is the most accurate model here:** lowest MAE on 9 of 10 series; global+exog wins on series G (46.43 vs 46.69) and is within 0.3 on D.
  On the two series the global model never saw (H, I) Chronos-2 leads by 3.6 and 4.4 MAE, a larger margin than on seen series. Chronos-2 also has the lowest pinball loss (17.03).
* **Exogenous signals help the global model (MAE 59.09 → 57.02) but hurt Chronos-2 (54.98 → 62.55)**, and Chronos-2 + exog is the worst variant on all 10 series.
  I have **not diagnosed why**. Candidate causes, all untested: how past covariates (including `aux_1`, a near-copy of the target) interact with Chronos-2's context handling;
  my bounded forward-fill and `other`-for-missing `cov_cat` handling; or 11 noisy channels diluting the signal. An ablation by group (aux only / cov only / cov_cat only) would separate these.
  Until then, treat "exog hurts Chronos-2" as an observation about *this wrapper's configuration*, not about the model.
* **Intervals.** The global model + exog is closest to nominal at 90% (0.901) and 50% (0.494). Chronos-2 no-exog is good at 50% (0.506) but ~1.6 pts short at 90% (0.884); Chronos-2 + exog under-covers
  at both levels (0.445 / 0.858) even after calibration, i.e. offsets fitted on validation did not transfer to the test period for that variant. Raw (uncalibrated) quantiles under-cover for all models (≈ 0.40–0.43 at 50%, ≈ 0.80–0.84 at 90%).
* **Validation coverage after calibration** is nominal by construction (Chronos val raw → calibrated: with exog 0.867 → 0.900, without 0.850 → 0.900 at 90%).
* **Caveats.** One test window (6 months), 90 origins per series, no confidence intervals or significance tests on the MAE differences (differences below ~1 MAE, e.g. series D and G, should be read as ties).
  The global model is small (150 trees, 8 training series, origins every 12 h) and untuned, so "Chronos-2 beats the global model" means *this* global model, not gradient boosting in general.
  The Chronos revision is `main`, which can change; pin a commit hash before relying on these numbers.

## Full per-series table

| Model | Exog | Window | Series | Seen | MAE | RMSE | sMAPE % | Pinball | 50% cov raw→cal | 90% cov raw→cal | Persistence MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| global | yes | test | series_A | yes | 58.62 | 87.37 | 69.8 | 17.92 | 0.421→0.486 | 0.819→0.897 | 78.98 |
| global | yes | test | series_C | yes | 48.96 | 70.48 | 76.8 | 14.70 | 0.374→0.459 | 0.825→0.884 | 61.15 |
| global | yes | test | series_D | yes | 61.24 | 91.52 | 67.6 | 18.76 | 0.427→0.507 | 0.832→0.905 | 82.38 |
| global | yes | test | series_E | yes | 57.84 | 86.15 | 67.0 | 17.38 | 0.437→0.506 | 0.850→0.924 | 75.51 |
| global | yes | test | series_F | yes | 61.17 | 98.50 | 69.0 | 19.50 | 0.452→0.523 | 0.819→0.881 | 79.40 |
| global | yes | test | series_G | yes | 46.43 | 69.31 | 72.0 | 14.26 | 0.419→0.500 | 0.839→0.895 | 60.36 |
| global | yes | test | series_H | no | 59.99 | 88.56 | 70.0 | 18.33 | 0.413→0.483 | 0.810→0.898 | 80.46 |
| global | yes | test | series_I | no | 58.55 | 87.88 | 72.2 | 17.59 | 0.431→0.500 | 0.817→0.909 | 82.35 |
| global | yes | test | series_J | yes | 57.09 | 85.17 | 68.8 | 17.48 | 0.412→0.481 | 0.833→0.903 | 71.29 |
| global | yes | test | series_L | yes | 60.13 | 91.93 | 66.5 | 18.21 | 0.432→0.496 | 0.842→0.915 | 77.18 |
| global | yes | test | **ALL** |  | **57.02** | 86.15 | 69.9 | 17.42 | 0.422→0.494 | 0.829→0.901 | 74.91 |
| global | no | test | series_A | yes | 60.48 | 89.43 | 73.2 | 18.53 | 0.412→0.451 | 0.830→0.879 | 78.98 |
| global | no | test | series_C | yes | 50.11 | 70.91 | 78.5 | 15.19 | 0.391→0.427 | 0.831→0.876 | 61.15 |
| global | no | test | series_D | yes | 64.07 | 94.07 | 71.8 | 19.67 | 0.424→0.465 | 0.840→0.880 | 82.38 |
| global | no | test | series_E | yes | 58.64 | 86.09 | 69.6 | 17.76 | 0.441→0.483 | 0.863→0.913 | 75.51 |
| global | no | test | series_F | yes | 62.85 | 99.21 | 71.1 | 19.82 | 0.447→0.482 | 0.829→0.876 | 79.40 |
| global | no | test | series_G | yes | 48.57 | 69.88 | 74.0 | 14.93 | 0.435→0.473 | 0.829→0.875 | 60.36 |
| global | no | test | series_H | no | 62.49 | 91.71 | 74.1 | 19.12 | 0.410→0.448 | 0.821→0.874 | 80.46 |
| global | no | test | series_I | no | 60.21 | 89.33 | 75.1 | 18.35 | 0.457→0.497 | 0.824→0.883 | 82.35 |
| global | no | test | series_J | yes | 59.97 | 89.81 | 72.5 | 18.27 | 0.419→0.454 | 0.846→0.889 | 71.29 |
| global | no | test | series_L | yes | 63.35 | 97.68 | 70.6 | 19.26 | 0.427→0.470 | 0.854→0.900 | 77.18 |
| global | no | test | **ALL** |  | **59.09** | 88.36 | 73.0 | 18.10 | 0.426→0.465 | 0.837→0.884 | 74.91 |
| chronos-2 | yes | test | series_A |  | 65.46 | 95.15 | 75.6 | 19.96 | 0.390→0.437 | 0.816→0.859 | 78.98 |
| chronos-2 | yes | test | series_C |  | 51.79 | 75.33 | 80.0 | 16.02 | 0.359→0.408 | 0.799→0.843 | 61.15 |
| chronos-2 | yes | test | series_D |  | 68.33 | 99.73 | 72.8 | 21.02 | 0.406→0.454 | 0.817→0.862 | 82.38 |
| chronos-2 | yes | test | series_E |  | 62.96 | 92.92 | 70.4 | 19.35 | 0.409→0.455 | 0.820→0.864 | 75.51 |
| chronos-2 | yes | test | series_F |  | 66.96 | 103.09 | 71.7 | 21.12 | 0.407→0.454 | 0.796→0.831 | 79.40 |
| chronos-2 | yes | test | series_G |  | 51.17 | 79.58 | 74.7 | 15.73 | 0.396→0.447 | 0.828→0.863 | 60.36 |
| chronos-2 | yes | test | series_H |  | 66.14 | 96.63 | 75.2 | 20.16 | 0.397→0.446 | 0.825→0.863 | 80.46 |
| chronos-2 | yes | test | series_I |  | 61.73 | 93.04 | 75.4 | 19.05 | 0.414→0.461 | 0.829→0.863 | 82.35 |
| chronos-2 | yes | test | series_J |  | 62.93 | 92.33 | 72.7 | 19.37 | 0.389→0.439 | 0.829→0.869 | 71.29 |
| chronos-2 | yes | test | series_L |  | 67.75 | 102.27 | 70.9 | 20.98 | 0.409→0.452 | 0.823→0.863 | 77.18 |
| chronos-2 | yes | test | **ALL** |  | **62.55** | 93.44 | 73.9 | 19.28 | 0.398→0.445 | 0.818→0.858 | 74.91 |
| chronos-2 | no | test | series_A |  | 56.58 | 85.76 | 68.4 | 17.46 | 0.414→0.502 | 0.820→0.883 | 78.98 |
| chronos-2 | no | test | series_C |  | 46.70 | 71.11 | 72.2 | 14.48 | 0.403→0.479 | 0.814→0.873 | 61.15 |
| chronos-2 | no | test | series_D |  | 60.98 | 92.22 | 67.4 | 18.76 | 0.419→0.496 | 0.817→0.883 | 82.38 |
| chronos-2 | no | test | series_E |  | 56.17 | 85.68 | 64.5 | 17.33 | 0.423→0.496 | 0.818→0.883 | 75.51 |
| chronos-2 | no | test | series_F |  | 59.25 | 96.38 | 64.4 | 18.96 | 0.449→0.532 | 0.808→0.869 | 79.40 |
| chronos-2 | no | test | series_G |  | 46.69 | 75.18 | 69.6 | 14.34 | 0.419→0.512 | 0.825→0.884 | 60.36 |
| chronos-2 | no | test | series_H |  | 56.36 | 86.35 | 67.6 | 17.41 | 0.437→0.522 | 0.822→0.895 | 80.46 |
| chronos-2 | no | test | series_I |  | 54.14 | 84.55 | 69.1 | 16.77 | 0.428→0.512 | 0.833→0.894 | 82.35 |
| chronos-2 | no | test | series_J |  | 54.81 | 83.95 | 66.4 | 16.89 | 0.420→0.504 | 0.819→0.888 | 71.29 |
| chronos-2 | no | test | series_L |  | 57.92 | 90.84 | 63.8 | 17.89 | 0.427→0.510 | 0.811→0.885 | 77.18 |
| chronos-2 | no | test | **ALL** |  | **54.98** | 85.53 | 67.3 | 17.03 | 0.424→0.506 | 0.819→0.884 | 74.91 |
