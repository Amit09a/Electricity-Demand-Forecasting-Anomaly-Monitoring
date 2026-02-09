import joblib
import json
import pandas as pd
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ARTIFACTS_PATH = os.path.join(BASE_DIR, "artifacts")

model_p50 = joblib.load(os.path.join(ARTIFACTS_PATH, "model_p50.pkl"))
model_p025 = joblib.load(os.path.join(ARTIFACTS_PATH, "model_p025.pkl"))
model_p975 = joblib.load(os.path.join(ARTIFACTS_PATH, "model_p975.pkl"))

with open(os.path.join(ARTIFACTS_PATH, "calibration.json")) as f:
    q_hat = json.load(f)["q_hat"]

with open(os.path.join(ARTIFACTS_PATH, "features.json")) as f:
    FEATURE_COLS = json.load(f)


# -----------------------------
# PREDICTION FUNCTION
# -----------------------------

def predict(features_df):

    # ensure correct column order
    X = features_df[FEATURE_COLS]

    p50 = model_p50.predict(X)[0]
    lower = model_p025.predict(X)[0]
    upper = model_p975.predict(X)[0]

    return {
        "forecast": float(round(p50, 2)),
        "lower_95": float(round(lower, 2)),
        "upper_95": float(round(upper, 2))
    }

