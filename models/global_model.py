"""Global (cross-series) gradient-boosted quantile model.

One direct multi-horizon model per quantile: `h` (horizon) is an input feature, so
a single set of trees serves every horizon 1..48. Targets are normalised by the
trailing 168h mean at the forecast origin, which is what lets the model score a
series it never saw in training. No series_id feature exists by construction.

Uses scikit-learn's HistGradientBoostingRegressor (quantile loss, native
categoricals, NaN-aware splits, seeded & deterministic).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

from app.uncertainty import QUANTILES
from models.preprocessing import SEED, model_columns

DEFAULT_PARAMS = dict(
    max_iter=150, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=200,
    l2_regularization=1.0, max_bins=255, early_stopping=False,
)


class GlobalQuantileModel:
    def __init__(self, exog: bool, params: dict | None = None):
        self.exog = exog
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.models: dict[float, HistGradientBoostingRegressor] = {}
        self.feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "GlobalQuantileModel":
        self.feature_names = model_columns(X)
        Xm = X[self.feature_names]
        for q in QUANTILES:
            m = HistGradientBoostingRegressor(
                loss="quantile", quantile=q, categorical_features="from_dtype",
                random_state=SEED, **self.params)
            self.models[q] = m.fit(Xm, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Quantiles in *normalised* units, shape (n, 5), non-crossing."""
        Xm = X[self.feature_names]
        P = np.column_stack([self.models[q].predict(Xm) for q in QUANTILES])
        return np.sort(P, axis=1)  # enforce monotone quantiles

    # ---- persistence (a "model version" = directory + meta.json with content hash) ----
    def save(self, directory: str | Path, extra_meta: dict | None = None) -> dict:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump({"models": self.models, "features": self.feature_names}, d / "model.joblib", compress=3)
        digest = hashlib.sha256((d / "model.joblib").read_bytes()).hexdigest()
        meta = {"kind": "global_hgb_quantile", "exog": self.exog, "params": self.params,
                "quantiles": list(QUANTILES), "n_features": len(self.feature_names),
                "sha256": digest, "sklearn": sklearn.__version__, "seed": SEED, **(extra_meta or {})}
        (d / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
        return meta

    @classmethod
    def load(cls, directory: str | Path) -> "GlobalQuantileModel":
        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text())
        digest = hashlib.sha256((d / "model.joblib").read_bytes()).hexdigest()
        if digest != meta["sha256"]:
            raise RuntimeError(f"Artifact hash mismatch in {d}: refusing to load.")
        blob = joblib.load(d / "model.joblib")
        obj = cls(meta["exog"], meta["params"])
        obj.models, obj.feature_names = blob["models"], blob["features"]
        return obj
