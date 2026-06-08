# 💧 AquaPredict — Household Water Usage Prediction

A full-stack ML project that predicts yearly household water usage using **two trained models** (Random Forest + Gradient Boosting) with area/county-based variation, served via a Flask REST API and a polished frontend.

---

## 📁 Project Structure

```
water-project/
├── ML/
│   ├── water.csv            ← Dataset (place your file here)
│   └── train_models.py      ← Run this to train and save both models
│
├── Backend/
│   ├── app.py               ← Flask REST API (5 endpoints)
│   ├── requirements.txt     ← Python dependencies
│   └── models/              ← Auto-created by train_models.py
│       ├── random_forest_model.pkl
│       ├── gradient_boosting_model.pkl
│       ├── scaler.pkl
│       ├── feature_names.pkl
│       ├── county_encoder.pkl
│       └── dataset_stats.json
│
└── Frontend/
    └── index.html           ← Open in browser (no build step needed)
```

---

## 🚀 Quick Start

### Step 1 — Install Python dependencies
```bash
cd Backend
pip install -r requirements.txt
```

### Step 2 — Train both ML models
```bash
cd ML
python train_models.py
```
This reads `water.csv`, trains Random Forest + Gradient Boosting, and saves all `.pkl` files into `Backend/models/`.

### Step 3 — Start the backend API
```bash
cd Backend
python app.py
# API running at http://localhost:5000
```

### Step 4 — Open the frontend
Simply open `Frontend/index.html` in your browser. No build step, no npm install.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Server health + model list |
| GET | `/api/models` | Model names + metrics (R², MAE, CV) |
| GET | `/api/counties` | List of all counties |
| GET | `/api/stats` | Dataset statistics |
| POST | `/api/predict` | Predict with one chosen model |
| POST | `/api/compare` | Run both models, return side-by-side results |

### Example — POST `/api/predict`
```json
{
  "model": "gradient_boosting",
  "county": "Devon",
  "inputs": {
    "Number-Of-People": 3,
    "Showers-Per-Week": 14,
    "Shower-Duration-Minutes": 10,
    "Bath-Frequency-Per-Week": 1,
    "Washing-Machine-Per-Week": 4,
    "Dishwasher-Per-Week": 5,
    "Has-Garden": 1,
    "Garden-Watering-Per-Week": 2,
    "Car-Wash-Per-Month": 1
  }
}
```

### Example — POST `/api/compare`
```json
{
  "county": "Yorkshire",
  "inputs": { "Number-Of-People": 4, "Showers-Per-Week": 20 }
}
```

---

## 🤖 ML Models

| Model | R² Test | MAE | CV R² |
|-------|---------|-----|-------|
| Random Forest | 0.9717 | ~5,776 L/yr | 0.9669 |
| Gradient Boosting | 0.9782 | ~4,983 L/yr | 0.9726 |

**Key features (by importance):**
1. `Weekly_Shower_Minutes` (showers/week × duration) — ~71%
2. `Number-Of-People` — ~24%
3. `Washing-Machine-Per-Week`, `Dishwasher-Per-Week`, `Bath-Frequency-Per-Week`

**Engineered features** (auto-computed, no need to pass in API):
- `Weekly_Shower_Minutes = Showers-Per-Week × Shower-Duration-Minutes`
- `Weekly_Bathing_Score = Bath-Frequency-Per-Week × 80`
- `Total_Appliance_Per_Week = Washing + Dishwasher`
- `Showers_Per_Person = Showers-Per-Week / Number-Of-People`

---

## 🌍 Deploying to the Web

### Backend → Railway (free)
1. Push your repo to GitHub
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Set root directory to `Backend/`
4. Railway auto-detects Flask. Set start command: `gunicorn app:app`
5. Copy the URL (e.g. `https://your-app.railway.app`)

### Frontend → Update API URL
In `Frontend/index.html`, find this line and update it:
```javascript
: 'https://your-backend.railway.app';   // ← UPDATE THIS
```

### Frontend → GitHub Pages / Netlify / Vercel
- Drop `Frontend/index.html` into any static hosting — it's a single file.

---

## 🔄 Retraining on New Data
1. Replace `ML/water.csv` with your new dataset
2. Run `python ML/train_models.py`
3. Restart `Backend/app.py` — it auto-loads the new `.pkl` files

---

## 📌 Notes
- The frontend works offline against `localhost:5000` by default
- Area/county predictions vary because `County_encoded` is a real feature the model learned from
- Missing inputs fall back to dataset mean values automatically
