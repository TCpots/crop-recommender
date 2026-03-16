# Crop Recommender

ML-powered crop recommendation system for Tripura districts.
Ranks crops by suitability based on soil and weather conditions,
and predicts yield using a trained XGBoost model.

## Files
- `crop_recommender.html` — open this in your browser to use the UI
- `backend.py` — Flask server that serves live yield predictions
- `crop_recommender.py` — command line version of the recommender

## Setup

### 1. Export model files from your notebook
Run these lines at the end of your XGBoost notebook:
```python
import pickle
pickle.dump(model_xgb,      open("yield_model.pkl",  "wb"))
pickle.dump(sc_all,         open("scaler.pkl",        "wb"))
pickle.dump(list(feat_all), open("feat_cols.pkl",     "wb"))
crop_stats.to_json("crop_stats.json")
```

### 2. Install dependencies
```
pip install flask flask-cors pandas xgboost scikit-learn openpyxl
```

### 3. Run
```
python backend.py
```
Then open `crop_recommender.html` in your browser.

## Notes
- Model files (pkl) and data (xlsx) are excluded from this repo via .gitignore
- Crops with fewer than 60 training rows fall back to historical average yield
```
