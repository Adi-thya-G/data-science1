"""
=============================================================
  Water Usage Prediction — Flask Backend
  Endpoints:
    GET  /api/health           → server health check
    GET  /api/models           → list models + metrics
    POST /api/predict          → predict with chosen model
    GET  /api/counties         → list of counties
    GET  /api/stats            → dataset statistics
=============================================================
"""
import os, json
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)   # allow all origins — tighten in production

# ── Paths ─────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# ── Load shared artifacts once at startup ─────────────────
scaler        = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
feature_names = joblib.load(os.path.join(MODELS_DIR, 'feature_names.pkl'))
county_enc    = joblib.load(os.path.join(MODELS_DIR, 'county_encoder.pkl'))

with open(os.path.join(MODELS_DIR, 'dataset_stats.json')) as f:
    stats = json.load(f)

COUNTY_MAP     = stats['county_mapping']        # name → code
FEATURE_MEANS  = stats['feature_means']         # fallback values
NATIONAL_AVG   = stats['national_avg_litres']
MODEL_METRICS  = stats['model_metrics']

# ── Lazy-load models ──────────────────────────────────────
_models = {}

def get_model(name: str):
    if name not in _models:
        path = os.path.join(MODELS_DIR, f'{name}_model.pkl')
        if not os.path.exists(path):
            return None
        _models[name] = joblib.load(path)
    return _models[name]

AVAILABLE_MODELS = ['random_forest', 'gradient_boosting']

# ── Feature builder ───────────────────────────────────────
def build_input(user_inputs: dict, county: str | None) -> pd.DataFrame:
    """
    Merge user inputs with dataset means, apply county encoding,
    compute engineered features, return scaled DataFrame row.
    """
    # Start from dataset feature means
    base = dict(FEATURE_MEANS)
    base.update({k: float(v) for k, v in user_inputs.items()})

    # County encoding
    if county:
        for k, v in COUNTY_MAP.items():
            if county.lower() in k.lower():
                base['County_encoded'] = v
                break

    # Engineered features
    base['Weekly_Shower_Minutes']    = base.get('Showers-Per-Week', 7) * base.get('Shower-Duration-Minutes', 8)
    base['Weekly_Bathing_Score']     = base.get('Bath-Frequency-Per-Week', 1) * 80
    base['Total_Appliance_Per_Week'] = base.get('Washing-Machine-Per-Week', 3) + base.get('Dishwasher-Per-Week', 3)
    base['Showers_Per_Person']       = base.get('Showers-Per-Week', 7) / max(base.get('Number-Of-People', 2), 1)

    row = pd.DataFrame([{f: base.get(f, 0.0) for f in feature_names}])
    return scaler.transform(row)

def water_tips(pred: int, inputs: dict) -> list[str]:
    tips = []
    if inputs.get('Shower-Duration-Minutes', 8) > 10:
        tips.append("Reduce shower time to under 8 minutes — saves ~15,000 L/yr")
    if inputs.get('Bath-Frequency-Per-Week', 1) > 2:
        tips.append("Replace baths with showers — saves ~8,000 L/yr")
    if inputs.get('Washing-Machine-Per-Week', 3) > 4:
        tips.append("Run washing machine on full loads only — saves ~5,000 L/yr")
    if inputs.get('Has-Garden', 0) == 1 and inputs.get('Garden-Watering-Per-Week', 1) > 2:
        tips.append("Install a rainwater harvesting system for your garden")
    if inputs.get('Has-Swimming-Pool', 0) == 1:
        tips.append("Cover your pool when not in use to reduce evaporation")
    if pred > 300000:
        tips.append("Consider a professional home water audit")
    if not tips:
        tips.append("Excellent water management — keep it up!")
    return tips

def rating(pred: int) -> str:
    if pred < 80000:   return "Excellent"
    if pred < 150000:  return "Good"
    if pred < 250000:  return "Moderate"
    if pred < 400000:  return "High"
    return "Very High"

# ═══════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "models_available": AVAILABLE_MODELS})

@app.route('/api/models', methods=['GET'])
def list_models():
    return jsonify({
        "models": [
            {
                "id":    name,
                "label": name.replace('_', ' ').title(),
                "metrics": MODEL_METRICS.get(name, {}),
            }
            for name in AVAILABLE_MODELS
        ]
    })

@app.route('/api/counties', methods=['GET'])
def counties():
    return jsonify({"counties": sorted(COUNTY_MAP.keys())})

@app.route('/api/stats', methods=['GET'])
def dataset_stats():
    return jsonify({
        "national_avg_litres": NATIONAL_AVG,
        "national_avg_daily_pp": round(NATIONAL_AVG / 365 / 2.5, 1),
        "model_metrics": MODEL_METRICS,
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    body = request.get_json(force=True)

    # ── Validate model choice ─────────────────────────────
    model_name = body.get('model', 'random_forest')
    if model_name not in AVAILABLE_MODELS:
        return jsonify({"error": f"Unknown model '{model_name}'. Choose from {AVAILABLE_MODELS}"}), 400

    model = get_model(model_name)
    if model is None:
        return jsonify({"error": "Model file not found. Run ML/train_models.py first."}), 500

    # ── Parse inputs ──────────────────────────────────────
    inputs = body.get('inputs', {})
    county = body.get('county', None)

    # Validate required fields
    required = ['Number-Of-People']
    missing = [f for f in required if f not in inputs]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    # ── Build + predict ───────────────────────────────────
    try:
        row_scaled = build_input(inputs, county)
        pred       = int(model.predict(row_scaled)[0])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    people   = max(int(inputs.get('Number-Of-People', 2)), 1)
    daily_pp = round(pred / 365 / people, 1)
    diff     = pred - NATIONAL_AVG

    return jsonify({
        "model_used":             model_name,
        "county":                 county or "Not specified",
        "predicted_litres_year":  pred,
        "daily_per_person_litres": daily_pp,
        "vs_national_avg": {
            "diff_litres": diff,
            "percent":     round(diff / NATIONAL_AVG * 100, 1),
        },
        "rating":          rating(pred),
        "tips":            water_tips(pred, inputs),
        "model_metrics":   MODEL_METRICS.get(model_name, {}),
    })

@app.route('/api/compare', methods=['POST'])
def compare():
    """Run both models and return side-by-side results."""
    body   = request.get_json(force=True)
    inputs = body.get('inputs', {})
    county = body.get('county', None)

    required = ['Number-Of-People']
    missing = [f for f in required if f not in inputs]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        row_scaled = build_input(inputs, county)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    results = {}
    for name in AVAILABLE_MODELS:
        m    = get_model(name)
        pred = int(m.predict(row_scaled)[0])
        results[name] = {
            "predicted_litres_year": pred,
            "daily_per_person_litres": round(pred / 365 / max(int(inputs.get('Number-Of-People',2)),1), 1),
            "rating": rating(pred),
            "metrics": MODEL_METRICS.get(name, {}),
        }

    avg_pred = int(np.mean([r['predicted_litres_year'] for r in results.values()]))
    return jsonify({
        "county":     county or "Not specified",
        "results":    results,
        "avg_prediction": avg_pred,
        "tips":       water_tips(avg_pred, inputs),
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    print(f"🚀 Starting Water Prediction API on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)