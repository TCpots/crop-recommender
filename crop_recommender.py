"""
Crop Recommendation + What-If Yield Estimator
==============================================

Before running this script, add these 4 lines to the END of your notebook
and run them once to export your trained model:

    import pickle
    pickle.dump(model_xgb,      open("yield_model.pkl",  "wb"))
    pickle.dump(sc_all,         open("scaler.pkl",        "wb"))
    pickle.dump(list(feat_all), open("feat_cols.pkl",     "wb"))
    crop_stats.to_json("crop_stats.json")

Then put these 5 files in the same folder:
    crop_recommender.py
    yield_model.pkl
    scaler.pkl
    feat_cols.pkl
    crop_stats.json
    merged_crop_weather_joined.xlsx      <-- for building suitability profiles

Run:
    python crop_recommender.py

Requirements:
    pip install pandas openpyxl scikit-learn xgboost
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── FILE PATHS ────────────────────────────────────────────────────────────────
WEATHER_FILE    = "merged_crop_weather_joined.xlsx"
MODEL_FILE      = "yield_model.pkl"
SCALER_FILE     = "scaler.pkl"
FEAT_COLS_FILE  = "feat_cols.pkl"
CROP_STATS_FILE = "crop_stats.json"

YIELD_COL = "Yield (Tonne or Bales/Hectare)"

# ── SUITABILITY RANKER CONFIG ─────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "Temp_Mean_C", "Annual_Rainfall_mm", "Windspeed_Max_kmh",
    "Humidity_Max_pct", "Humidity_Min_pct", "Evapotranspiration_mm",
    "Solar_Radiation_MJm2", "Rainy_Days", "Fertilizer_kg_per_ha",
]
NUMERIC_WEIGHTS = {
    "Fertilizer_kg_per_ha":  3.0,
    "Annual_Rainfall_mm":    1.5,
    "Temp_Mean_C":           1.2,
    "Humidity_Max_pct":      1.0,
    "Humidity_Min_pct":      1.0,
    "Evapotranspiration_mm": 1.0,
    "Solar_Radiation_MJm2":  1.0,
    "Rainy_Days":            1.0,
    "Windspeed_Max_kmh":     0.8,
}
PEST_PENALTY = {"Low": 1.0, "Medium": 0.85, "High": 0.6}


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD ARTIFACTS  (exported from your notebook)
# ══════════════════════════════════════════════════════════════════════════════

def load_artifacts():
    missing = [f for f in [MODEL_FILE, SCALER_FILE, FEAT_COLS_FILE, CROP_STATS_FILE]
               if not Path(f).exists()]
    if missing:
        print("\nERROR: The following files are missing from this folder:")
        for f in missing:
            print(f"  - {f}")
        print("\nRun these lines at the end of your notebook to generate them:")
        print("  import pickle")
        print("  pickle.dump(model_xgb,      open('yield_model.pkl',  'wb'))")
        print("  pickle.dump(sc_all,         open('scaler.pkl',        'wb'))")
        print("  pickle.dump(list(feat_all), open('feat_cols.pkl',     'wb'))")
        print("  crop_stats.to_json('crop_stats.json')")
        raise SystemExit(1)

    model      = pickle.load(open(MODEL_FILE,      "rb"))
    scaler     = pickle.load(open(SCALER_FILE,     "rb"))
    feat_cols  = pickle.load(open(FEAT_COLS_FILE,  "rb"))
    crop_stats = pd.read_json(CROP_STATS_FILE)
    valid_crops = crop_stats.index.tolist()

    print(f"  Yield model loaded — {len(valid_crops)} crops with enough training data.")
    return model, scaler, feat_cols, crop_stats, valid_crops


# ══════════════════════════════════════════════════════════════════════════════
#  BUILD SUITABILITY PROFILES  (from weather_joined — all 25 crops)
# ══════════════════════════════════════════════════════════════════════════════

def build_profiles():
    if not Path(WEATHER_FILE).exists():
        print(f"\nERROR: '{WEATHER_FILE}' not found in this folder.")
        raise SystemExit(1)

    df = pd.read_excel(WEATHER_FILE)
    profiles = {}
    for crop, sub in df.groupby("Crop"):
        p = {}
        for f in NUMERIC_FEATURES:
            if f in sub.columns:
                p[f] = {"mean": float(sub[f].mean()), "std": float(sub[f].std() + 1e-3)}
        for cat in ["Soil_Type", "Irrigation_Type", "Season"]:
            p[cat] = sub[cat].value_counts(normalize=True).to_dict()
        p["avg_yield"] = float(sub[YIELD_COL].mean())
        profiles[crop] = p

    print(f"  Suitability profiles built for {len(profiles)} crops.")
    return profiles


def get_districts():
    df = pd.read_excel(WEATHER_FILE)
    return sorted(df["District_Name"].unique().tolist())


# ══════════════════════════════════════════════════════════════════════════════
#  SUITABILITY RANKING
# ══════════════════════════════════════════════════════════════════════════════

def _gaussian(val, mean, std):
    return float(np.exp(-0.5 * ((val - mean) / std) ** 2))


def rank_crops(user, profiles, top_n=7):
    scores = []
    for crop, p in profiles.items():
        num, wsum = 0.0, 0.0
        for feat, wt in NUMERIC_WEIGHTS.items():
            if feat in user and feat in p:
                num  += wt * _gaussian(user[feat], p[feat]["mean"], p[feat]["std"])
                wsum += wt
        num = num / wsum if wsum > 0 else 0.0

        soil    = p.get("Soil_Type",       {}).get(user.get("Soil_Type",       ""), 0)
        irr     = p.get("Irrigation_Type", {}).get(user.get("Irrigation_Type", ""), 0)
        season  = p.get("Season",          {}).get(user.get("Season",          ""), 0)
        penalty = PEST_PENALTY.get(user.get("Pest_Disease_Incidence", "Low"), 1.0)

        combined = (num * 0.50 + soil * 0.12 + irr * 0.12 + season * 0.26) * penalty
        scores.append({"crop": crop, "score": combined, "season_fit": season})

    scores.sort(key=lambda x: x["score"], reverse=True)
    max_s = scores[0]["score"] if scores else 1.0
    for s in scores:
        s["pct"] = round(s["score"] / max_s * 100)
    return scores[:top_n]


# ══════════════════════════════════════════════════════════════════════════════
#  YIELD PREDICTION
#  - Crops with enough training data  -> XGBoost model prediction
#  - Sparse crops (dropped at <60)    -> historical average from profiles
# ══════════════════════════════════════════════════════════════════════════════

def predict_yield(crop, district, user, model, scaler, feat_cols, crop_stats, valid_crops, profiles):
    # Sparse crop — fall back to historical average from the weather file
    if crop not in valid_crops:
        return profiles[crop]["avg_yield"], "hist_avg"

    # Build a single-row dataframe matching the model's feature schema
    row = {
        "Area (Hectare)":        user.get("Area (Hectare)", 500),
        "Fertilizer_kg_per_ha":  user["Fertilizer_kg_per_ha"],
        "Pest_Disease_Incidence": {"Low": 0, "Medium": 1, "High": 2}.get(
                                    user.get("Pest_Disease_Incidence", "Low"), 0),
        # Lag features set to 0 (normalised) = assume last season was at crop mean
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
    return float(norm_pred * std + mu), "model"


# ══════════════════════════════════════════════════════════════════════════════
#  INTERACTIVE CLI
# ══════════════════════════════════════════════════════════════════════════════

def _ask(prompt, default, cast=float):
    val = input(f"  {prompt} [{default}]: ").strip()
    return cast(val) if val else cast(default)


def _ask_choice(prompt, options, default):
    opts = " / ".join(options)
    while True:
        val = input(f"  {prompt} ({opts}) [{default}]: ").strip() or default
        if val in options:
            return val
        print(f"    Please choose one of: {opts}")


def interactive_loop(model, scaler, feat_cols, crop_stats, valid_crops, profiles, districts):
    print("\n" + "=" * 66)
    print("  CROP RECOMMENDATION + WHAT-IF YIELD ESTIMATOR")
    print("=" * 66)
    print("  Press Enter to accept the default value shown in [brackets].\n")

    while True:
        print("\n-- Weather conditions (used for suitability ranking) ----------")
        user = {}
        user["Temp_Mean_C"]           = _ask("Mean temperature (C)",          24.5)
        user["Annual_Rainfall_mm"]    = _ask("Annual rainfall (mm)",          1800)
        user["Humidity_Max_pct"]      = _ask("Max humidity (%)",                95)
        user["Humidity_Min_pct"]      = _ask("Min humidity (%)",                58)
        user["Solar_Radiation_MJm2"]  = _ask("Solar radiation (MJ/m2)",       5800)
        user["Rainy_Days"]            = _ask("Rainy days / year",              170)
        user["Evapotranspiration_mm"] = _ask("Evapotranspiration (mm)",       1240)
        user["Windspeed_Max_kmh"]     = _ask("Max windspeed (km/h)",          11.2)

        print("\n-- Farm inputs (used for both ranking and yield prediction) ---")
        user["Fertilizer_kg_per_ha"]   = _ask("Fertilizer (kg/ha)",            70)
        user["Area (Hectare)"]         = _ask("Field area (ha)",              500, int)
        user["Soil_Type"]              = _ask_choice("Soil type",
                                             ["Red Laterite", "Alluvial"], "Alluvial")
        user["Irrigation_Type"]        = _ask_choice("Irrigation",
                                             ["Rainfed", "Canal", "Drip"], "Rainfed")
        user["Season"]                 = _ask_choice("Season",
                                             ["Kharif", "Rabi", "Whole Year",
                                              "Summer", "Autumn", "Winter"], "Kharif")
        user["Pest_Disease_Incidence"] = _ask_choice("Pest/disease incidence",
                                             ["Low", "Medium", "High"], "Low")
        district = _ask_choice("District", districts, districts[0])

        ranked = rank_crops(user, profiles, top_n=7)

        print("\n" + "=" * 66)
        print(f"  TOP RECOMMENDATIONS   district: {district}")
        print("=" * 66)
        print(f"  {'#':<3} {'Crop':<28} {'Suit%':>5}  {'Est. Yield':>22}  Season")
        print("  " + "-" * 62)

        for i, r in enumerate(ranked):
            crop = r["crop"]
            yhat, source = predict_yield(
                crop, district, user,
                model, scaler, feat_cols, crop_stats, valid_crops, profiles
            )

            if source == "hist_avg":
                yield_str = f"~{yhat:.2f} T/ha (hist. avg)"
            else:
                yield_str = f"{yhat:.2f} T/ha"

            if   r["season_fit"] > 0.5: season_tag = "good"
            elif r["season_fit"] > 0:   season_tag = "partial"
            else:                        season_tag = "off-season"

            marker = "  <-- BEST" if i == 0 else ""
            print(f"  #{i+1:<2} {crop:<28} {r['pct']:>4}%  {yield_str:>22}  {season_tag}{marker}")

        print("  " + "-" * 62)
        print("  Model prediction: assumes last season was at the historical crop mean.")
        print("  hist. avg: crop had <60 training rows, XGBoost not used.\n")

        if input("  Run another query? (y/n) [y]: ").strip().lower() == "n":
            break

    print("\n  Done.\n")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\nLoading model artifacts ...")
    model, scaler, feat_cols, crop_stats, valid_crops = load_artifacts()

    print("Building suitability profiles ...")
    profiles  = build_profiles()
    districts = get_districts()

    interactive_loop(model, scaler, feat_cols, crop_stats, valid_crops, profiles, districts)


if __name__ == "__main__":
    main()
