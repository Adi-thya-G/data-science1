"""
=============================================================
  Water Usage Prediction — Dual Model Training Script
  Models: Random Forest + Gradient Boosting
  Run this script to retrain and save both models.
  All .pkl files go to ../Backend/models/
=============================================================
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, joblib, os, json

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Output directory ──────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'Backend', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

TARGET = "Household-Water-Use-Litres-Yearly"

# ═══════════════════════════════════════════════════════════
# 1. LOAD
# ═══════════════════════════════════════════════════════════
print("Loading data...")
df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'water.csv'))
print(f"  Raw shape: {df.shape}")

# ═══════════════════════════════════════════════════════════
# 2. CLEAN
# ═══════════════════════════════════════════════════════════
df = df.loc[:, ~df.columns.duplicated()]
df = df.replace("NULL", np.nan)

LEAKAGE = [
    "Bathroom-Water-Use-Litres-Yearly","Kitchen-Water-Use-Litres-Yearly",
    "Outdoor-Water-Use-Litres-Yearly","Person-Water-Use-Litres-Yearly",
    "Person-Water-Use-Litres-Per-Day","Household-Water-Use-Money-£-Yearly",
    "Household-Water-Saving-Litres-Yearly","Household-Water-Saving-Money-£-Yearly",
    "Household-Energy-Use-Money-£-Yearly","Household-Energy-Saving-kWh-Yearly",
    "Household-Energy-Saving-Cost-£-Yearly",
]
IDS = ["Date","Postal outcode","Latitude","Longitude","Wash-Dishes-By-Hand"]
df.drop(columns=[c for c in LEAKAGE+IDS if c in df.columns], inplace=True)

# Binary encode
for col in ['Has-Garden','Has-Swimming-Pool']:
    if col in df.columns:
        df[col] = df[col].str.lower().map({'yes':1,'no':0})

# Label encode County → County_encoded
county_map = {}
if 'County' in df.columns:
    le = LabelEncoder()
    df['County_encoded'] = le.fit_transform(df['County'].astype(str))
    county_map = dict(zip(le.classes_, le.transform(le.classes_).tolist()))
    df.drop(columns=['County'], inplace=True)
    joblib.dump(le, os.path.join(MODELS_DIR, 'county_encoder.pkl'))

# Numeric convert + fill NaN
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

# Outlier removal (continuous cols only)
feat_cols = [c for c in df.columns if c != TARGET]
mask = pd.Series([True]*len(df), index=df.index)
for col in feat_cols:
    if df[col].nunique() > 10:
        mask &= (df[col] >= df[col].quantile(0.01)) & (df[col] <= df[col].quantile(0.99))
df = df[mask]
print(f"  Clean shape: {df.shape}")

# ═══════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════
df['Weekly_Shower_Minutes']    = df['Showers-Per-Week'] * df['Shower-Duration-Minutes']
df['Weekly_Bathing_Score']     = df['Bath-Frequency-Per-Week'] * 80
df['Total_Appliance_Per_Week'] = df['Washing-Machine-Per-Week'] + df['Dishwasher-Per-Week']
df['Showers_Per_Person']       = df['Showers-Per-Week'] / df['Number-Of-People'].clip(1)

# ═══════════════════════════════════════════════════════════
# 4. SPLIT + SCALE
# ═══════════════════════════════════════════════════════════
X = df.drop(columns=[TARGET]); y = df[TARGET]
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_train)
X_te = scaler.transform(X_test)

# ═══════════════════════════════════════════════════════════
# 5. TRAIN BOTH MODELS
# ═══════════════════════════════════════════════════════════
kf = KFold(n_splits=5, shuffle=True, random_state=42)

MODELS = {
    "random_forest": RandomForestRegressor(
        n_estimators=200, min_samples_split=5,
        min_samples_leaf=2, random_state=42, n_jobs=-1
    ),
    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05,
        max_depth=5, random_state=42
    ),
}

metrics = {}
for name, model in MODELS.items():
    print(f"\nTraining {name}...")
    model.fit(X_tr, y_train)
    yp  = model.predict(X_te)
    cv  = cross_val_score(model, X_tr, y_train, cv=kf, scoring='r2')
    mae  = mean_absolute_error(y_test, yp)
    rmse = np.sqrt(mean_squared_error(y_test, yp))
    r2   = r2_score(y_test, yp)
    metrics[name] = {
        "mae":   round(mae, 0),
        "rmse":  round(rmse, 0),
        "r2":    round(r2, 4),
        "cv_r2": round(cv.mean(), 4),
        "cv_std": round(cv.std(), 4),
    }
    print(f"  MAE={mae:,.0f} | RMSE={rmse:,.0f} | R²={r2:.4f} | CV={cv.mean():.4f}±{cv.std():.4f}")
    joblib.dump(model, os.path.join(MODELS_DIR, f'{name}_model.pkl'))
    print(f"  ✅ Saved: Backend/models/{name}_model.pkl")

# ═══════════════════════════════════════════════════════════
# 6. SAVE SHARED ARTIFACTS
# ═══════════════════════════════════════════════════════════
joblib.dump(scaler,        os.path.join(MODELS_DIR, 'scaler.pkl'))
joblib.dump(feature_names, os.path.join(MODELS_DIR, 'feature_names.pkl'))

# Dataset stats for the API (national averages, ranges, etc.)
dataset_stats = {
    "national_avg_litres": int(df[TARGET].mean()),
    "target_min": int(df[TARGET].min()),
    "target_max": int(df[TARGET].max()),
    "county_mapping": county_map,
    "feature_means": {
        c: round(float(df[c].mean()), 2)
        for c in feature_names
        if c not in ['Weekly_Shower_Minutes','Weekly_Bathing_Score',
                     'Total_Appliance_Per_Week','Showers_Per_Person']
    },
    "model_metrics": metrics,
}
with open(os.path.join(MODELS_DIR, 'dataset_stats.json'), 'w') as f:
    json.dump(dataset_stats, f, indent=2)

print("\n" + "="*50)
print("✅ Training Complete! Files saved to Backend/models/")
print("="*50)
print(f"  random_forest_model.pkl")
print(f"  gradient_boosting_model.pkl")
print(f"  scaler.pkl")
print(f"  feature_names.pkl")
print(f"  county_encoder.pkl")
print(f"  dataset_stats.json")
print("\nModel comparison:")
for name, m in metrics.items():
    print(f"  {name:20s} → R²={m['r2']} | MAE={m['mae']:,.0f} | CV={m['cv_r2']}")
