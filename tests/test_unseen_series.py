"""The service must forecast a series it never saw, using only the uploaded history."""
import numpy as np
import pytest

from app import forecasting
from app.validation import parse_csv
from tests.conftest import make_series, to_csv_bytes


def _run(store, df, **kw):
    hist, info = parse_csv(to_csv_bytes(df))
    return forecasting.forecast(store, hist, kw.get("horizon", 24), 90, kw.get("model", "global"),
                                kw.get("exog", True)), info


def test_unseen_id_scale_and_category(store):
    df = make_series("series_never_seen", seed=123, level=400.0, start="2016-06-01", end="2016-09-01",
                     unknown_cat=True)
    out, info = _run(store, df)
    pts = np.array([p["point"] for p in out["forecast"]])
    last = df["target"].dropna().tail(24).mean()
    assert np.isfinite(pts).all() and pts.min() >= 0
    assert 0.1 * last < pts.mean() < 10 * last          # right order of magnitude despite a never-seen level
    assert any("other" in w for w in info["warnings"])


def test_forecast_does_not_depend_on_series_id(store):
    a = make_series("alpha", seed=7, start="2016-06-01", end="2016-09-01")
    b = a.assign(series_id="completely_different_name")
    assert _run(store, a)[0]["forecast"] == _run(store, b)[0]["forecast"]


@pytest.mark.parametrize("model", ["global", "chronos"])
def test_all_models_handle_unseen(store, model):
    df = make_series("held_out", seed=5, level=25.0, start="2016-06-01", end="2016-09-01")
    out, _ = _run(store, df, model=model)
    assert len(out["forecast"]) == 24
