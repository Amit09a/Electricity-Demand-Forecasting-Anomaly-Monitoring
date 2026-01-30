import os
import sys

# ---------------------------------------------------
# PATH SETUP
# ---------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_PATH = os.path.join(BASE_DIR, "artifacts")

sys.path.append(BASE_DIR)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.inference import predict


# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Electricity Demand Forecast",
    layout="wide"
)

# ---------------------------------------------------
# LOAD PREDICTION DATA (Dashboard Analytics)
# ---------------------------------------------------

@st.cache_data
def load_predictions():
    df = pd.read_csv(
        os.path.join(ARTIFACTS_PATH, "interval_predictions.csv"),
        parse_dates=["datetime"]
    )

    df["abs_error"] = (df["y_true"] - df["p50"]).abs()
    df["hour"] = df["datetime"].dt.hour

    return df


# ---------------------------------------------------
# LOAD FEATURE STORE (Live Inference)
# ---------------------------------------------------

@st.cache_data
def load_feature_store():
    return pd.read_parquet(
        os.path.join(ARTIFACTS_PATH, "feature_store.parquet")
    )


df = load_predictions()
feature_df = load_feature_store()

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.title("Electricity Demand Forecasting & Anomaly Monitoring")

st.markdown("""
Machine learning-based short-term electricity demand forecasting  
with calibrated uncertainty intervals and automated anomaly detection.
""")

# ---------------------------------------------------
# SIDEBAR FILTER
# ---------------------------------------------------

st.sidebar.header("Filter Timeline")

start = st.sidebar.date_input(
    "Start Date",
    df["datetime"].min().date()
)

end = st.sidebar.date_input(
    "End Date",
    df["datetime"].max().date()
)

filtered = df[
    (df["datetime"].dt.date >= start) &
    (df["datetime"].dt.date <= end)
]

if filtered.empty:
    st.warning("No data available for selected date range.")
    st.stop()

# ---------------------------------------------------
# METRICS
# ---------------------------------------------------

st.subheader("Model Performance")

col1, col2, col3, col4 = st.columns(4)

mae = (filtered["y_true"] - filtered["p50"]).abs().mean()

smape = (
    abs(filtered["y_true"] - filtered["p50"]) /
    ((abs(filtered["y_true"]) + abs(filtered["p50"])) / 2)
).mean() * 100

coverage_95 = (
    (filtered["y_true"] >= filtered["p025_cal"]) &
    (filtered["y_true"] <= filtered["p975_cal"])
).mean() * 100

anomaly_rate = filtered["anomaly"].mean() * 100

col1.metric("MAE", f"{mae:.1f} kW")
col2.metric("SMAPE", f"{smape:.2f}%")
col3.metric("95% Coverage", f"{coverage_95:.1f}%")
col4.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")

# ---------------------------------------------------
# FORECAST CHART
# ---------------------------------------------------

st.subheader("Actual vs Forecast with 95% Prediction Interval")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=filtered["datetime"],
    y=filtered["y_true"],
    name="Actual",
    line=dict(color="black")
))

fig.add_trace(go.Scatter(
    x=filtered["datetime"],
    y=filtered["p50"],
    name="Forecast",
    line=dict(color="royalblue")
))

fig.add_trace(go.Scatter(
    x=filtered["datetime"],
    y=filtered["p975_cal"],
    line=dict(width=0),
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=filtered["datetime"],
    y=filtered["p025_cal"],
    fill='tonexty',
    fillcolor='rgba(65,105,225,0.2)',
    line=dict(width=0),
    name="95% Interval"
))

st.plotly_chart(fig, width='stretch')

# ---------------------------------------------------
# LATEST FORECAST
# ---------------------------------------------------

st.divider()

st.subheader("Latest Forecast In Selected Window")

latest = filtered.sort_values("datetime").iloc[-1]

st.success(f"""
### Next Hour Expected Demand: **{int(latest['p50'])} kW**

### Likely Range:  
**{int(latest['p025_cal'])} — {int(latest['p975_cal'])} kW**
""")

# ===================================================
# 🔥 LIVE DEMAND FORECAST (PRODUCTION STYLE)
# ===================================================

st.divider()
st.header("Live Demand Forecast")

st.markdown(
    "Adjust current conditions to simulate real-time electricity demand."
)

# ⭐ IMPORTANT: pull from FEATURE STORE
latest_features = feature_df.sort_values("datetime").iloc[-1].copy()

col1, col2, col3 = st.columns(3)

temp = col1.slider("Temperature (°C)", 0.0, 50.0, float(latest_features["temp"]))
rhum = col2.slider("Humidity (%)", 0, 100, int(latest_features["rhum"]))
wspd = col3.slider("Wind Speed (m/s)", 0.0, 20.0, float(latest_features["wspd"]))

col4, col5 = st.columns(2)

pres = col4.slider("Pressure (hPa)", 980, 1050, int(latest_features["pres"]))

lag_1 = col5.number_input(
    "Current Demand (kW)",
    value=float(latest_features["lag_1"])
)

if st.button("Generate Live Forecast"):

    input_row = latest_features.copy()

    # overwrite only safe fields
    input_row["temp"] = temp
    input_row["rhum"] = rhum
    input_row["wspd"] = wspd
    input_row["pres"] = pres
    input_row["lag_1"] = lag_1

    input_df = pd.DataFrame([input_row])

    result = predict(input_df)

    st.success(f"""
    ## Forecasted Demand: **{int(result['forecast'])} kW**

    ### Expected Range (95%):
    **{int(result['lower_95'])} — {int(result['upper_95'])} kW**
    """)
