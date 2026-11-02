import numpy as np
import pandas as pd
import pytest

from app import forecasting
from app.uncertainty import apply_cqr, coverage, fit_cqr
from app.validation import parse_csv
from models.chronos_model import ChronosForecaster, parse_prediction_frame
from models.preprocessing import (build_origin_features, expand_horizons, model_columns, normalise_cat,
                                  to_hourly)
from tests.conftest import StubPipeline, make_series, to_csv_bytes


def _hourly(df):
    return to_hourly(df)


def _same(a: pd.Series, b: pd.Series) -> bool:
    """Compare two one-element Series (numeric, categorical string, or NaN)."""
    x, y = a.iloc[0], b.iloc[0]
    if pd.isna(x) and pd.isna(y):
        return True
    if isinstance(x, str) or isinstance(y, str):
        return x == y
    return bool(np.isclose(float(x), float(y), rtol=1e-5, atol=1e-6))


@pytest.mark.parametrize("exog", [True, False])
def test_features_are_causal(series_df, exog):
    """Features at origin t must be identical whether or not the future exists."""
    full = _hourly(series_df)
    F_full = build_origin_features(full, exog)
    for t in [full.index[400], full.index[1200], full.index[-300]]:
        F_cut = build_origin_features(full.loc[:t], exog)
        a, b = F_full.loc[t], F_cut.loc[t]
        for c in F_full.columns:
            assert _same(pd.Series([a[c]]), pd.Series([b[c]])), f"{c} differs at {t}: future leaked into features"


def test_features_ignore_garbage_future(series_df):
    full = _hourly(series_df)
    t = full.index[1500]
    poisoned = full.copy()
    poisoned.loc[poisoned.index > t, ["target", "aux_1", "cov_1"]] = 1e9
    a = build_origin_features(full, True).loc[t]
    b = build_origin_features(poisoned, True).loc[t]
    assert all(_same(pd.Series([a[c]]), pd.Series([b[c]])) for c in a.index)


def test_train_serve_parity_through_csv(series_df):
    """Offline features == features recomputed from the uploaded CSV (no train-serve skew)."""
    offline = build_origin_features(_hourly(series_df), True)
    hist, _ = parse_csv(to_csv_bytes(series_df))
    served = build_origin_features(hist, True)
    t = hist.index[-1]
    assert all(_same(pd.Series([offline.loc[t, c]]), pd.Series([served.loc[t, c]])) for c in offline.columns)
    Xo = expand_horizons(offline.loc[[t]], list(range(1, 25)))
    Xs = expand_horizons(served.loc[[t]], list(range(1, 25)))
    assert model_columns(Xo) == model_columns(Xs)


def test_unknown_category_maps_to_other():
    s, n = normalise_cat(pd.Series(["N", "ne", " SW ", "XYZ", None]))
    assert n == 1
    assert s.astype(object).tolist()[:4] == ["N", "NE", "SW", "other"] and pd.isna(s.iloc[4])


def test_forecast_is_deterministic(store, series_df):
    hist, _ = parse_csv(to_csv_bytes(series_df))
    a = forecasting.forecast(store, hist, 24, 90, "global", True)
    b = forecasting.forecast(store, hist, 24, 90, "global", True)
    assert a == b


@pytest.mark.parametrize("model", ["global", "chronos"])
@pytest.mark.parametrize("exog", [True, False])
def test_forecast_shape_and_ordering(store, series_df, model, exog):
    hist, _ = parse_csv(to_csv_bytes(series_df))
    r90 = forecasting.forecast(store, hist, 36, 90, model, exog)
    r50 = forecasting.forecast(store, hist, 36, 50, model, exog)
    assert len(r90["forecast"]) == 36
    assert r90["forecast"][0]["timestamp"] == str(hist.index[-1] + pd.Timedelta(hours=1))
    for p90, p50 in zip(r90["forecast"], r50["forecast"]):
        assert 0 <= p90["lower"] <= p90["point"] <= p90["upper"]
        assert p90["lower"] <= p50["lower"] + 1e-6 and p50["upper"] <= p90["upper"] + 1e-6   # 50% nested in 90%
        assert np.isfinite([p90["point"], p90["lower"], p90["upper"]]).all()


def test_forecast_survives_missing_tail(store, series_df):
    d = series_df.copy()
    d.loc[d.index[-6:], ["target", "aux_1", "aux_4"]] = np.nan
    hist, _ = parse_csv(to_csv_bytes(d))
    r = forecasting.forecast(store, hist, 12, 90, "global", True)
    assert np.isfinite([p["point"] for p in r["forecast"]]).all()


def test_exog_toggle_changes_the_forecast(store, series_df):
    hist, _ = parse_csv(to_csv_bytes(series_df))
    a = forecasting.forecast(store, hist, 24, 90, "global", True)
    b = forecasting.forecast(store, hist, 24, 90, "global", False)
    assert [p["point"] for p in a["forecast"]] != [p["point"] for p in b["forecast"]]


def test_backtest_metrics_present(store, series_df):
    hist, _ = parse_csv(to_csv_bytes(series_df))
    r = forecasting.backtest(store, hist, 24, 90, "global", True, n_origins=5)
    assert r["n_origins"] == 5 and 0 <= r["coverage"] <= 1 and r["mae"] > 0 and r["n_points"] > 0


def test_backtest_does_not_see_the_future(store, series_df):
    """Changing values after the last backtest origin's horizon must not change its forecasts."""
    hist, _ = parse_csv(to_csv_bytes(series_df))
    o = hist.index[-24 - 100]
    t1 = forecasting.predict_table(store, "global", True, hist, [o], 24)
    h2 = hist.copy(); h2.loc[h2.index > o, ["target", "aux_1", "cov_1"]] = 5e5
    t2 = forecasting.predict_table(store, "global", True, h2, [o], 24)
    assert np.allclose(t1[[f"q{i}" for i in range(5)]].to_numpy(), t2[[f"q{i}" for i in range(5)]].to_numpy())


# ---------------------------------------------------------------- conformal calibration
def test_cqr_restores_coverage():
    rng = np.random.default_rng(0)
    n = 20000
    h = rng.integers(1, 49, n)
    sigma = 1.0 + h / 24
    y = rng.normal(0, sigma)
    lo, hi = -0.5 * sigma, 0.5 * sigma            # badly under-covering raw band
    half = n // 2
    off = fit_cqr(y[:half], lo[:half], hi[:half], h[:half], 90)
    l2, h2 = apply_cqr(lo[half:], hi[half:], h[half:], off)
    assert coverage(y[half:], lo[half:], hi[half:]) < 0.5
    assert abs(coverage(y[half:], l2, h2) - 0.90) < 0.02


# ---------------------------------------------------------------- foundation-model wrapper contract
def test_chronos_parsing_contract():
    fc = ChronosForecaster(pipeline=StubPipeline())
    hists = [_hourly(make_series(seed=s, start="2016-06-01", end="2016-08-01")) for s in (1, 2)]
    out = fc.predict(hists, 12, exog=True)
    assert len(out) == 2 and out[0].shape == (12, 5)
    assert (np.diff(out[0], axis=1) >= 0).all()


def test_chronos_parser_rejects_missing_quantiles():
    from models.chronos_model import ModelUnavailable
    bad = pd.DataFrame({"id": ["s0"] * 3, "timestamp": pd.date_range("2020", periods=3, freq="h"), "0.5": 1.0})
    with pytest.raises(ModelUnavailable):
        parse_prediction_frame(bad, ["s0"], 3)
