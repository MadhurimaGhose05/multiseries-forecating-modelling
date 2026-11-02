import numpy as np
import pandas as pd
import pytest

from app.validation import ApiError, parse_csv, validate_params
from tests.conftest import make_series, to_csv_bytes


def code(fn, *a, **k):
    with pytest.raises(ApiError) as e:
        fn(*a, **k)
    return e.value.code


def test_valid_file_parses(csv_bytes):
    hist, info = parse_csv(csv_bytes)
    assert info["series_id"] == "series_new"
    assert hist.index.is_monotonic_increasing and len(hist) == len(csv_bytes.splitlines()) - 1


@pytest.mark.parametrize("h", [0, -1, 49, 1000, "abc", None, 2.5 * 1e9])
def test_bad_horizon(h):
    assert code(validate_params, h, 90, "global") == "invalid_horizon"


@pytest.mark.parametrize("i", [0, 80, 95, 100, "x"])
def test_bad_interval(i):
    assert code(validate_params, 24, i, "global") == "invalid_interval"


def test_unknown_model():
    assert code(validate_params, 24, 90, "prophet") == "unknown_model"


def test_empty_and_binary_and_garbage():
    assert code(parse_csv, b"") == "empty_file"
    assert code(parse_csv, b"\x00\x01\x02" * 100) == "malformed_file"
    assert code(parse_csv, b"just,some\nrandom,text\n") == "schema_mismatch"


def test_missing_columns(series_df):
    raw = to_csv_bytes(series_df.drop(columns=["cov_3", "aux_2"]))
    with pytest.raises(ApiError) as e:
        parse_csv(raw)
    assert e.value.code == "schema_mismatch" and set(e.value.details["missing"]) == {"cov_3", "aux_2"}


def test_bad_timestamps(series_df):
    d = series_df.copy(); d["timestamp"] = d["timestamp"].astype(str); d.loc[5, "timestamp"] = "not a date"
    assert code(parse_csv, to_csv_bytes(d)) == "invalid_timestamp"


def test_duplicate_timestamps(series_df):
    assert code(parse_csv, to_csv_bytes(pd.concat([series_df, series_df.iloc[:3]]))) == "duplicate_timestamps"


def test_non_hourly(series_df):
    d = series_df.copy(); d["timestamp"] = pd.date_range("2016-06-01", periods=len(d), freq="30min")
    assert code(parse_csv, to_csv_bytes(d)) == "irregular_frequency"


def test_non_numeric(series_df):
    d = series_df.copy(); d["target"] = d["target"].astype(object); d.loc[10, "target"] = "oops"
    assert code(parse_csv, to_csv_bytes(d)) == "non_numeric_values"


def test_too_short(series_df):
    assert code(parse_csv, to_csv_bytes(series_df.head(100))) == "insufficient_history"


def test_stale_target(series_df):
    d = series_df.copy(); d.loc[d.index[-100:], "target"] = np.nan
    assert code(parse_csv, to_csv_bytes(d)) in ("stale_target", "insufficient_recent_data")


def test_multiple_series_need_id(series_df):
    two = pd.concat([series_df, make_series("series_other", seed=3, start="2016-06-01", end="2016-09-01")])
    assert code(parse_csv, to_csv_bytes(two)) == "multiple_series"
    hist, info = parse_csv(to_csv_bytes(two), "series_other")
    assert info["series_id"] == "series_other"
    assert code(parse_csv, to_csv_bytes(two), "nope") == "unknown_series_id"


def test_gaps_inserted_not_interpolated(series_df):
    d = series_df.drop(series_df.index[1000:1005])
    hist, info = parse_csv(to_csv_bytes(d))
    assert hist["target"].iloc[1000:1005].isna().all()
    assert any("inserted" in w for w in info["warnings"])


def test_unknown_category_warns_not_fails():
    d = make_series("series_u", seed=5, start="2016-06-01", end="2016-09-01", unknown_cat=True)
    hist, info = parse_csv(to_csv_bytes(d))
    assert any("other" in w for w in info["warnings"])
