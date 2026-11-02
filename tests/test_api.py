"""API contract tests (need fastapi + httpx; run in CI / the docker image)."""
import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

import app.main as main  # noqa: E402
from models.chronos_model import ChronosForecaster  # noqa: E402
from tests.conftest import BrokenPipeline, make_series, to_csv_bytes  # noqa: E402


@pytest.fixture()
def client(store, monkeypatch):
    monkeypatch.setattr(main, "store", store)
    return TestClient(main.app, raise_server_exceptions=False)


def post(client, raw, **form):
    return client.post("/forecast", files={"file": ("x.csv", raw, "text/csv")}, data=form)


def assert_error(r, status, code):
    assert r.status_code == status, r.text
    body = r.json()["error"]
    assert body["code"] == code and isinstance(body["message"], str) and "details" in body


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_forecast_ok(client, csv_bytes):
    r = post(client, csv_bytes, horizon=24, interval=90, model="global", use_exog="true")
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["series_id"] == "series_new" and len(j["forecast"]) == 24
    assert set(j["forecast"][0]) == {"timestamp", "point", "lower", "upper"}


def test_same_input_same_output(client, csv_bytes):
    a = post(client, csv_bytes, horizon=24, interval=90, model="global").json()
    b = post(client, csv_bytes, horizon=24, interval=90, model="global").json()
    assert a == b


@pytest.mark.parametrize("model", ["global", "chronos"])
def test_both_models_and_intervals(client, csv_bytes, model):
    for interval in (50, 90):
        assert post(client, csv_bytes, horizon=12, interval=interval, model=model).status_code == 200


def test_unseen_series_id_is_fine(client):
    raw = to_csv_bytes(make_series("totally_new_series", seed=99, level=15, start="2016-06-01", end="2016-09-01"))
    assert post(client, raw, horizon=12, interval=90, model="global").status_code == 200


@pytest.mark.parametrize("form,code", [
    ({"horizon": 0}, "invalid_horizon"), ({"horizon": 49}, "invalid_horizon"),
    ({"interval": 80}, "invalid_interval"), ({"model": "arima"}, "unknown_model"),
])
def test_bad_params(client, csv_bytes, form, code):
    assert_error(post(client, csv_bytes, **{"horizon": 24, "interval": 90, "model": "global", **form}), 422, code)


def test_non_integer_horizon_uses_same_envelope(client, csv_bytes):
    assert_error(post(client, csv_bytes, horizon="abc"), 422, "invalid_request")


def test_missing_file(client):
    assert_error(client.post("/forecast", data={"horizon": 24}), 422, "invalid_request")


def test_malformed_files(client):
    assert_error(post(client, b"", horizon=24), 400, "empty_file")
    assert_error(post(client, b"\x89PNG\r\n\x00\x00binary", horizon=24), 400, "malformed_file")
    assert_error(post(client, b"a,b\n1,2\n", horizon=24), 422, "schema_mismatch")


def test_short_history(client, series_df):
    assert_error(post(client, to_csv_bytes(series_df.head(50)), horizon=24), 422, "insufficient_history")


def test_foundation_model_unavailable_is_503(client, store, csv_bytes, monkeypatch):
    monkeypatch.setattr(store, "chronos", ChronosForecaster(pipeline=BrokenPipeline()))
    assert_error(post(client, csv_bytes, horizon=12, model="chronos"), 503, "model_unavailable")


def test_backtest(client, csv_bytes):
    r = client.post("/backtest", files={"file": ("x.csv", csv_bytes, "text/csv")},
                    data={"horizon": 24, "interval": 90, "model": "global", "n_origins": 5})
    assert r.status_code == 200, r.text
    assert {"mae", "coverage", "rmse", "smape", "naive_mae"} <= set(r.json())


def test_models_endpoint(client):
    j = client.get("/models").json()
    assert "global_exog" in j and "chronos" in j
