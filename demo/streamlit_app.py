"""Minimal reviewer UI.  streamlit run demo/streamlit_app.py

Talks to the REST service only (API_URL), so what you see is exactly what the
endpoint returns. Upload a CSV in the training schema, pick a model, toggle the
exogenous signals, and compare.
"""
import io
import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
MODEL_LABELS = {"global": "Global ML model (gradient boosting)", "chronos": "Foundation model (Chronos-2)"}

st.set_page_config(page_title="Forecast demo", layout="wide")
st.title("Hourly forecast with uncertainty")
st.caption("Upload one series, choose a model, and see the forecast, its uncertainty band, and how well the "
           "method would have done on the most recent days of *your* file.")


@st.cache_data(show_spinner=False)
def call_api(endpoint: str, raw: bytes, params: tuple):
    r = requests.post(f"{API_URL}/{endpoint}", files={"file": ("upload.csv", raw, "text/csv")},
                      data=dict(params), timeout=600)
    if r.ok:
        return r.json(), None
    try:
        err = r.json()["error"]
    except Exception:
        err = {"code": f"http_{r.status_code}", "message": r.text[:300], "details": {}}
    return None, err


def show_error(err: dict):
    st.error(f"**{err['code']}** — {err['message']}")
    if err.get("details"):
        with st.expander("details"):
            st.json(err["details"])


with st.sidebar:
    st.header("1 · Data")
    up = st.file_uploader("CSV file (same columns as the training data)", type=["csv"])
    st.header("2 · Model")
    model = st.radio("Which model?", list(MODEL_LABELS), format_func=MODEL_LABELS.get)
    use_exog = st.toggle("Use extra signals (aux_*, cov_*, cov_cat)", value=True,
                         help="Past readings of the other channels and the weather-like covariates, "
                              "known only up to the last timestamp of your file.")
    st.header("3 · Forecast")
    horizon = st.slider("Hours ahead", 1, 48, 24)
    interval = st.radio("Uncertainty band", [50, 90], index=1, format_func=lambda v: f"{v}% interval", horizontal=True)
    history_days = st.slider("Days of history to plot", 2, 60, 7)
    go_btn = st.button("Run forecast", type="primary", disabled=up is None)

with st.expander("How to read this page", expanded=up is None):
    st.markdown(
        "- **Black line** – what was actually observed.\n"
        "- **Coloured line** – the model's best single guess for each future hour.\n"
        "- **Shaded band** – the range the model expects the real value to land in. A **90% band** should contain the "
        "true value about 9 times out of 10; a 50% band about 5 times out of 10. Wider band = less certain.\n"
        "- **Dashed line/band** – the same model *without* (or with) the extra signals, so you can see what they change.\n"
        "- **Check on your file** – we replay the method on the last two weeks of your data (using only information "
        "available at each moment) and compare with what really happened.")
    st.markdown("Required columns: `timestamp, series_id, target, aux_1..aux_5, cov_1..cov_5, cov_cat`, hourly, "
                "at least two weeks of history.")

if up is None:
    st.info("Upload a CSV in the sidebar to begin.")
    st.stop()

raw = up.getvalue()
try:
    hist = pd.read_csv(io.BytesIO(raw), parse_dates=["timestamp"])
except Exception as e:  # the API will give the formal error; this is only for plotting
    hist = None
    st.warning(f"Could not preview file locally: {e}")

if not go_btn:
    st.info("Choose settings in the sidebar, then press **Run forecast**.")
    st.stop()


def params(exog: bool, extra: dict | None = None) -> tuple:
    d = {"horizon": horizon, "interval": interval, "model": model, "use_exog": str(exog).lower(), **(extra or {})}
    return tuple(sorted(d.items()))


with st.spinner("Forecasting…"):
    main, err = call_api("forecast", raw, params(use_exog))
    if err:
        show_error(err)
        st.stop()
    other, err2 = call_api("forecast", raw, params(not use_exog))
    bt, err3 = call_api("backtest", raw, params(use_exog, {"n_origins": 14}))

for w in main.get("warnings", []):
    st.warning(w)

f = pd.DataFrame(main["forecast"])
f["timestamp"] = pd.to_datetime(f["timestamp"])
fig = go.Figure()
if hist is not None:
    h = hist.sort_values("timestamp").tail(history_days * 24)
    fig.add_scatter(x=h["timestamp"], y=h["target"], name="Observed", line=dict(color="#222"), connectgaps=False)


def band(df, color_rgba, name, dash=None, line_color=None):
    fig.add_scatter(x=df["timestamp"], y=df["upper"], line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=df["timestamp"], y=df["lower"], fill="tonexty", fillcolor=color_rgba, line=dict(width=0),
                    name=f"{name} {interval}% band", hoverinfo="skip")
    fig.add_scatter(x=df["timestamp"], y=df["point"], name=f"{name} forecast",
                    line=dict(color=line_color, dash=dash, width=2 if dash is None else 1.5))


label = "with extra signals" if use_exog else "target history only"
band(f, "rgba(31,119,180,0.25)", label, None, "#1f77b4")
if other:
    o = pd.DataFrame(other["forecast"])
    o["timestamp"] = pd.to_datetime(o["timestamp"])
    band(o, "rgba(255,127,14,0.12)", "target history only" if use_exog else "with extra signals", "dash", "#ff7f0e")
fig.update_layout(height=480, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                  legend=dict(orientation="h", y=-0.15), yaxis_title="target")
st.plotly_chart(fig, use_container_width=True)

c1, c2, c3 = st.columns(3)
c1.metric("Series", main["series_id"])
c2.metric(f"Average {interval}% band width", f"{main['mean_interval_width']:.1f}",
          delta=(f"{main['mean_interval_width'] - other['mean_interval_width']:+.1f} vs. toggled" if other else None),
          delta_color="off")
c3.metric("Model", MODEL_LABELS[model].split(" (")[0])

st.subheader("Check on your file (last 14 days replayed)")
if err3:
    show_error(err3)
elif bt:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE (average miss)", f"{bt['mae']:.2f}", delta=f"{bt['mae'] - bt['naive_mae']:+.2f} vs. 'same as now'",
              delta_color="inverse")
    m2.metric(f"Coverage of {interval}% band", f"{bt['coverage']:.1%}", delta=f"target {bt['nominal_coverage']:.0%}",
              delta_color="off")
    m3.metric("RMSE", f"{bt['rmse']:.2f}")
    m4.metric("sMAPE", f"{bt['smape']:.1f}%")
    st.caption(f"{bt['n_origins']} daily forecast start-points, {bt['n_points']} hourly comparisons. "
               "Coverage on so few days is noisy: a few percentage points either side of the target is normal.")

with st.expander("Raw API response"):
    st.json({k: v for k, v in main.items() if k != "forecast"})
    st.dataframe(f)
