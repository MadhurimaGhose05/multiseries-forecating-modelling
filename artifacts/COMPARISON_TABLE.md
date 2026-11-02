# Global model vs Chronos-2 — all 10 series, test window (2016-09-01 → 2017-03-01)

**Status: global-model rows are real, measured results. Chronos-2 rows are NOT RUN** — the environment this repo was built in has no
torch, no `chronos-forecasting` package, no model weights and no network, so no Chronos-2 score exists yet. None are estimated.

## Produce the Chronos-2 rows (on a machine with the weights, ~CPU hours; GPU much faster)
```bash
pip install -r requirements.txt
python -m training.prepare_data --src <folder with the 10 series CSVs>
python -m training.train_chronos          # zero-shot: "training" = fitting conformal offsets on 2016-01..08 (writes artifacts/chronos_*/)
python -m training.compare --chronos --windows val test    # writes artifacts/comparison.md / .json  (this exact table, both models)
```
(`python -m training.train_chronos --smoke` first, to verify the wiring on 1 series / 4 origins.)

Chronos-2 has no training set (zero-shot). Its validation window is used only for the conformal offsets, so every series counts as unseen for it.

## Series and what "Seen" means for the global model
Trained and calibrated on A, C, D, E, F, G, J, L (targets < 2016-01-01; calibration 2016-01..08). **H and I were never used for fitting or calibration.**
Origins every 12 h, horizons 1–48 h; MAE/RMSE/pinball in target units; coverage raw → conformal-calibrated; the **ALL** row pools all 10 series (incl. H, I;
the 8-series-only figure is MAE 55.79 / 58.27, see FULL_RESULTS.md).

| Model | Exog | Window | Series | Seen | MAE | RMSE | sMAPE % | Pinball | 50% cov raw→cal | 90% cov raw→cal | Persistence MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| global | yes | test | series_A | yes | 57.68 | 85.91 | 69.2 | 17.59 | 0.417→0.488 | 0.821→0.891 | 72.28 |
| global | yes | test | series_C | yes | 46.11 | 67.87 | 72.5 | 14.07 | 0.411→0.487 | 0.831→0.893 | 58.20 |
| global | yes | test | series_D | yes | 61.51 | 92.45 | 67.4 | 18.91 | 0.427→0.504 | 0.825→0.888 | 77.70 |
| global | yes | test | series_E | yes | 56.79 | 84.92 | 66.7 | 17.39 | 0.434→0.505 | 0.834→0.902 | 71.38 |
| global | yes | test | series_F | yes | 60.86 | 95.75 | 69.4 | 19.01 | 0.430→0.501 | 0.817→0.880 | 75.72 |
| global | yes | test | series_G | yes | 46.53 | 70.10 | 71.0 | 14.40 | 0.435→0.503 | 0.828→0.890 | 58.50 |
| global | yes | test | series_H | no | 59.58 | 88.96 | 69.4 | 18.39 | 0.420→0.489 | 0.807→0.884 | 75.06 |
| global | yes | test | series_I | no | 55.85 | 84.00 | 69.5 | 17.26 | 0.432→0.502 | 0.807→0.889 | 74.58 |
| global | yes | test | series_J | yes | 56.02 | 84.24 | 67.4 | 17.17 | 0.434→0.501 | 0.830→0.896 | 69.25 |
| global | yes | test | series_L | yes | 60.59 | 92.56 | 66.9 | 18.65 | 0.431→0.500 | 0.826→0.897 | 75.65 |
| global | yes | test | **ALL** |  | **56.17** | 85.16 | 68.9 | 17.29 | 0.427→0.498 | 0.823→0.891 | 70.85 |
| global | no | test | series_A | yes | 59.90 | 89.46 | 72.6 | 18.45 | 0.416→0.459 | 0.830→0.875 | 72.28 |
| global | no | test | series_C | yes | 48.27 | 69.10 | 75.2 | 14.68 | 0.420→0.459 | 0.840→0.885 | 58.20 |
| global | no | test | series_D | yes | 64.65 | 97.93 | 71.7 | 20.03 | 0.432→0.473 | 0.833→0.871 | 77.70 |
| global | no | test | series_E | yes | 59.44 | 89.17 | 70.3 | 18.28 | 0.441→0.483 | 0.843→0.889 | 71.38 |
| global | no | test | series_F | yes | 62.49 | 97.02 | 71.4 | 19.45 | 0.434→0.473 | 0.829→0.873 | 75.72 |
| global | no | test | series_G | yes | 48.16 | 71.01 | 72.8 | 14.90 | 0.439→0.478 | 0.829→0.877 | 58.50 |
| global | no | test | series_H | no | 62.49 | 93.84 | 73.5 | 19.39 | 0.418→0.458 | 0.817→0.866 | 75.06 |
| global | no | test | series_I | no | 57.85 | 87.28 | 73.3 | 18.12 | 0.434→0.474 | 0.811→0.869 | 74.58 |
| global | no | test | series_J | yes | 58.97 | 89.30 | 71.5 | 18.05 | 0.436→0.474 | 0.843→0.885 | 69.25 |
| global | no | test | series_L | yes | 64.03 | 99.67 | 71.4 | 19.78 | 0.434→0.475 | 0.835→0.879 | 75.65 |
| global | no | test | **ALL** |  | **58.65** | 88.98 | 72.3 | 18.12 | 0.430→0.471 | 0.831→0.877 | 70.85 |

## Chronos-2 (pending)
| Model | Exog | Window | Series | Seen | MAE | RMSE | sMAPE % | Pinball | 50% cov raw→cal | 90% cov raw→cal | Persistence MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chronos-2 | yes | test | all 10 series | n/a (zero-shot) | not run | not run | not run | not run | not run | not run | – |
| chronos-2 | no | test | all 10 series | n/a (zero-shot) | not run | not run | not run | not run | not run | not run | – |
