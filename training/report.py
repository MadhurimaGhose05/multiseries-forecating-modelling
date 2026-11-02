"""Full score report: train / validation / test for every trained variant.

    python -m training.report

Train rows are in-sample (the model saw these labels) and are shown only to expose over-fitting;
validation was used for conformal calibration; test is the untouched hold-out.
Pinball loss is the mean over the 5 quantiles (0.05/0.25/0.5/0.75/0.95) in original units.
"""
import json

import numpy as np

from app.uncertainty import QUANTILES
from models.global_model import GlobalQuantileModel
from training.calibrate import ARTIFACTS, global_table
from training.evaluate import evaluate_table
from training.prepare_data import load_processed
from app.uncertainty import load_calibration


def pinball(tbl) -> float:
    y, s = tbl["y"].to_numpy(), tbl["scale"].to_numpy()
    ok = np.isfinite(y)
    loss = []
    for i, q in enumerate(QUANTILES):
        d = y[ok] - tbl[f"q{i}"].to_numpy()[ok] * s[ok]
        loss.append(np.mean(np.maximum(q * d, (q - 1) * d)))
    return float(np.mean(loss))


if __name__ == "__main__":
    all_frames = load_processed()
    out = {}
    for name, exog in [("global_exog", True), ("global_noexog", False)]:
        model = GlobalQuantileModel.load(ARTIFACTS / name)
        trained_on = json.loads((ARTIFACTS / name / "meta.json").read_text())["series"]
        frames = {k: v for k, v in all_frames.items() if k in trained_on}      # series the model was fitted on
        unseen = {k: v for k, v in all_frames.items() if k not in trained_on}  # never seen: fitting OR calibration
        cal = load_calibration(ARTIFACTS / name / "calibration.json")
        out[name] = {}
        for w in ("train", "val", "test"):
            t = global_table(model, frames, w, 12)
            r = evaluate_table(t, cal)
            r["overall"]["pinball"] = pinball(t)
            out[name][w] = r
            o = r["overall"]
            print(f"{name:14s} {w:5s} n={o['n']:>8,} MAE {o['mae']:.2f} RMSE {o['rmse']:.2f} sMAPE {o['smape']:.1f} "
                  f"pinball {o['pinball']:.2f} | cov50 {o['raw_cov50']:.3f}->{o['cov50']:.3f} cov90 {o['raw_cov90']:.3f}->{o['cov90']:.3f} "
                  f"| persist MAE {o['naive_mae']:.2f}", flush=True)
        if unseen:
            t = global_table(model, unseen, "test", 12)
            r = evaluate_table(t, cal); r["overall"]["pinball"] = pinball(t); out[name]["unseen_test"] = r
            for sid, o in r["by_series"].items():
                print(f"{name:14s} UNSEEN {sid} test: MAE {o['mae']:.2f} RMSE {o['rmse']:.2f} sMAPE {o['smape']:.1f} "
                      f"cov50 {o['cov50']:.3f} cov90 {o['cov90']:.3f} | persist MAE {o['naive_mae']:.2f}", flush=True)
            o = r["overall"]
            print(f"{name:14s} UNSEEN all   test: MAE {o['mae']:.2f} RMSE {o['rmse']:.2f} pinball {o['pinball']:.2f} "
                  f"cov50 {o['cov50']:.3f} cov90 {o['cov90']:.3f} | persist MAE {o['naive_mae']:.2f}", flush=True)
    (ARTIFACTS / "full_report.json").write_text(json.dumps(out, indent=2, default=float))
