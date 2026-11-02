"""Chronos-2 (zero-shot foundation model) wrapper.

Design notes
* Zero-shot: no weights are trained. "Training" (training/train_chronos.py) only
  fits the conformal offsets on the validation window.
* Covariates: Chronos-2 natively accepts *past* covariates (numeric and
  categorical). We pass aux_1..5, cov_1..5 and cov_cat as past covariates only
  (no future_df) -- consistent with the global model's "known up to origin" rule.
* NaNs: Chronos-2 is NaN-aware for the target; covariates get the same bounded
  forward-fill (COV_FFILL_LIMIT) as the global model, cov_cat NaN -> 'other'.
* Determinism: Chronos-2 produces quantiles in a single forward pass (no
  sampling). We still seed torch and pin model id + revision.
* The pipeline object is injectable so unit tests can use a stub; the real
  weights are loaded lazily and failure surfaces as `ModelUnavailable`.

NOT VERIFIED IN THE AUTHORING SANDBOX (no network / no torch): the call
`Chronos2Pipeline.predict_df(...)` follows the chronos-forecasting 2.x README.
`parse_prediction_frame` is deliberately tolerant about quantile column names.
Run `pytest tests/test_forecasting.py -k chronos` and
`python -m training.train_chronos --smoke` on a machine with the weights.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from app.uncertainty import QUANTILES
from models.preprocessing import AUX, COV, COV_FFILL_LIMIT, OTHER, SEED, normalise_cat

MODEL_ID = os.environ.get("CHRONOS_MODEL_ID", "amazon/chronos-2")
# TODO(pin): set to the exact Hugging Face commit hash you validated, e.g. via
# `huggingface_hub.model_info(MODEL_ID).sha`; "main" is NOT reproducible.
MODEL_REVISION = os.environ.get("CHRONOS_REVISION", "main")
CONTEXT_LENGTH = int(os.environ.get("CHRONOS_CONTEXT", "1024"))   # hours of history fed to the model


class ModelUnavailable(RuntimeError):
    """Chronos weights / libraries are not available in this environment."""


def parse_prediction_frame(pred: pd.DataFrame, ids: list[str], horizon: int) -> list[np.ndarray]:
    """Turn predict_df output into one (horizon, 5) array per id, columns = QUANTILES."""
    qcols: dict[float, str] = {}
    for c in pred.columns:
        try:
            v = float(c)
        except (TypeError, ValueError):
            continue
        qcols[round(v, 4)] = c
    missing = [q for q in QUANTILES if round(q, 4) not in qcols]
    if missing:
        raise ModelUnavailable(f"Chronos output lacks quantile levels {missing}; columns={list(pred.columns)}")
    out = []
    for i in ids:
        g = pred[pred["id"] == i].sort_values("timestamp")
        if len(g) != horizon:
            raise ModelUnavailable(f"Chronos returned {len(g)} rows for {i}, expected {horizon}")
        out.append(np.column_stack([g[qcols[round(q, 4)]].to_numpy(dtype=float) for q in QUANTILES]))
    return out


class ChronosForecaster:
    def __init__(self, model_id: str = MODEL_ID, revision: str = MODEL_REVISION,
                 device: str = "cpu", context_length: int = CONTEXT_LENGTH, pipeline: Any = None):
        self.model_id, self.revision, self.device = model_id, revision, device
        self.context_length = context_length
        self._pipeline = pipeline

    # ---- lazy load -------------------------------------------------------
    def _pipe(self):
        if self._pipeline is None:
            try:
                import torch
                from chronos import Chronos2Pipeline
                torch.manual_seed(SEED)
                self._pipeline = Chronos2Pipeline.from_pretrained(
                    self.model_id, revision=self.revision, device_map=self.device)
            except Exception as e:  # ImportError, OSError (offline / no weights), ...
                raise ModelUnavailable(f"Could not load {self.model_id}@{self.revision}: {e}") from e
        return self._pipeline

    def available(self) -> bool:
        try:
            self._pipe()
            return True
        except ModelUnavailable:
            return False

    # ---- inference -------------------------------------------------------
    def _context_frame(self, hist: pd.DataFrame, sid: str, exog: bool) -> pd.DataFrame:
        h = hist.tail(self.context_length)
        d = pd.DataFrame({"id": sid, "timestamp": h.index, "target": h["target"].astype(float).to_numpy()})
        if exog:
            for c in AUX:
                d[c] = h[c].astype(float).to_numpy()
            for c in COV:
                d[c] = h[c].astype(float).ffill(limit=COV_FFILL_LIMIT).to_numpy()
            cat, _ = normalise_cat(h["cov_cat"].astype("object").ffill(limit=COV_FFILL_LIMIT))
            d["cov_cat"] = cat.astype("object").fillna(OTHER).to_numpy()
        return d

    def predict(self, hists: list[pd.DataFrame], horizon: int, exog: bool) -> list[np.ndarray]:
        """hists: regular hourly frames (index = timestamp). Returns per-history
        arrays of shape (horizon, 5) in *original* target units, non-crossing."""
        ids = [f"s{i}" for i in range(len(hists))]
        ctx = pd.concat([self._context_frame(h, i, exog) for h, i in zip(hists, ids)], ignore_index=True)
        pipe = self._pipe()
        pred = pipe.predict_df(ctx, prediction_length=horizon, quantile_levels=list(QUANTILES),
                               id_column="id", timestamp_column="timestamp", target="target")
        return [np.sort(a, axis=1) for a in parse_prediction_frame(pred, ids, horizon)]

    def info(self) -> dict:
        return {"model_id": self.model_id, "revision": self.revision, "context_length": self.context_length,
                "device": self.device}
