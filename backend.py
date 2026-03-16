"""
Crop Recommender — Python Backend
==================================
Loads your trained XGBoost model and serves yield predictions to the HTML frontend.

Run:
    python backend.py

Then open crop_recommender.html in your browser.
The page will automatically use live model predictions instead of historical averages.

Requirements:
    pip install flask flask-cors pandas xgboost scikit-learn
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

warnings.filterwarnings("ignore")

# ── FILE PATHS ────────────────────────────────────────────────────────────────
MODEL_FILE      = "yield_model.pkl"
SCALER_FILE     = "scaler.pkl"
FEAT_COLS_FILE  = "feat_cols.pkl"
CROP_STATS_FILE = "crop_stats.json"

# ── LOAD ARTIFACTS ────────────────────────────────────────────────────────────
print("\nLoading model artifacts ...")

missing = [f for f in [MODEL_FILE, SCALER_FILE, FEAT_COLS_FILE, CROP_STATS_FILE]
           if not Path(f).exists()]
if missing:
    print("\nERROR: Missing files:")
    for f in missing:
        print(f"  - {f}")
    print("\nExport them from your notebook first:")
    print("  import pickle")
    print("  pickle.dump(model_xgb,      open('yield_model.pkl',  'wb'))")
    print("  pickle.dump(sc_all,         open('scaler.pkl',        'wb'))")
    print("  pickle.dump(list(feat_all), open('feat_cols.pkl',     'wb'))")
    print("  crop_stats.to_json('crop_stats.json')")
    raise SystemExit(1)

model       = pickle.load(open(MODEL_FILE,     "rb"))
scaler      = pickle.load(open(SCALER_FILE,    "rb"))
feat_cols   = pickle.load(open(FEAT_COLS_FILE, "rb"))
crop_stats  = pd.read_json(CROP_STATS_FILE)
valid_crops = crop_stats.index.tolist()

print(f"  Model loaded — {len(valid_crops)} crops available for prediction.")
print(f"  Valid crops: {valid_crops}\n")

# ── FLASK APP ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)   # allows the HTML file (opened locally) to call this API


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON body:
    {
      "crop":                    "Rice",
      "district":                "Dhalai",
      "Fertilizer_kg_per_ha":    70,
      "Area (Hectare)":          500,
      "Pest_Disease_Incidence":  "Low"    (string — converted to 0/1/2 here)
    }

    Returns:
    {
      "yield":   2.45,
      "source":  "model"         -- or "hist_avg" if crop is sparse
    }
    """
    data = request.get_json()

    crop     = data.get("crop", "")
    district = data.get("district", "Dhalai")

    # Sparse crop — model was never trained on it
    if crop not in valid_crops:
        return jsonify({"yield": None, "source": "hist_avg"})

    pest_map = {"Low": 0, "Medium": 1, "High": 2}

    row = {
        "Area (Hectare)":        data.get("Area (Hectare)", 500),
        "Fertilizer_kg_per_ha":  data.get("Fertilizer_kg_per_ha", 70),
        "Pest_Disease_Incidence": pest_map.get(data.get("Pest_Disease_Incidence", "Low"), 0),
        # Lag features = 0 (normalised) → assumes last season was at crop mean
        "Yield_Lag1":  0.0,
        "Yield_Roll3": 0.0,
        "Yield_Trend": 0.0,
        "District_Name": district,
        "Crop": crop,
    }

    X = pd.get_dummies(pd.DataFrame([row]), drop_first=True)
    X = X.reindex(columns=feat_cols, fill_value=0)
    X_scaled = scaler.transform(X)

    norm_pred = model.predict(X_scaled)[0]
    mu  = crop_stats.loc[crop, "crop_mean"]
    std = crop_stats.loc[crop, "crop_std"]
    yield_pred = float(norm_pred * std + mu)

    return jsonify({"yield": round(yield_pred, 3), "source": "model"})


@app.route("/valid_crops", methods=["GET"])
def get_valid_crops():
    """Returns the list of crops the model can predict for."""
    return jsonify({"valid_crops": valid_crops})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# ── RUN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Backend running at http://localhost:5000")
    print("Open crop_recommender.html in your browser.\n")
    app.run(port=5000, debug=False)