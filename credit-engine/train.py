"""
GoPay Credit Engine — Model Training
=====================================
Trains a GradientBoostingRegressor to predict credit scores (300-900).

Data strategy (auto-detected at runtime):
  1. SDV-enhanced data  — if `credit_sdv_data.csv` exists (run generate_sdv_data.py first)
     50,000+ rows with realistic inter-feature correlations learned by SDV.
  2. Numpy fallback     — if CSV not found, generates 15,000 profiles on the fly
     using the original independent-sampling approach.

The SDV path produces a better-generalising model because training data preserves
real-world correlations between features (e.g. higher balance → more transactions).

Usage:
    python generate_sdv_data.py   # (optional) generate enhanced training data first
    python train.py               # trains on SDV data if available, numpy otherwise

Output:
    model.pkl              — trained model + feature list (loaded by app.py)
    feature_importance.txt — ranked feature importances
    training_report.txt    — full training metadata (data source, metrics, etc.)
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

SDV_DATA_PATH = 'credit_sdv_data.csv'

FEATURE_COLS = [
    'wallet_balance',
    'total_transactions',
    'total_sent',
    'total_received',
    'avg_transaction_amount',
    'account_age_days',
    'days_since_last_txn',
    'txn_frequency_per_week',
]

# ─────────────────────────────────────────────────────────────────────────────
# Scoring formula — single source of truth (identical to generate_sdv_data.py)
# ─────────────────────────────────────────────────────────────────────────────
def compute_label(f):
    """
    Deterministic credit score formula. Score range: 300–900.
    This is the 'ground truth' the model learns to approximate.

    Factor weights:
      Balance (30%)  — higher wallet balance → better creditworthiness
      Activity (25%) — more transactions → higher engagement
      Net flow (25%) — received > sent → positive cash flow
      Recency (10%)  — recent activity → active account
      Age (10%)      — older account → established relationship
    """
    b_score  = min(1.0, f['wallet_balance'] / 75_000)
    a_score  = min(1.0, f['total_transactions'] / 200)

    total_flow = f['total_sent'] + f['total_received']
    nf_score   = (f['total_received'] / total_flow) if total_flow > 0 else 0.5

    r_score    = max(0.0, 1.0 - f['days_since_last_txn'] / 60) if f['total_transactions'] > 0 else 0.0
    age_score  = min(1.0, f['account_age_days'] / 1095)

    raw = (b_score * 0.30 + a_score * 0.25 + nf_score * 0.25
           + r_score * 0.10 + age_score * 0.10)

    noise = np.random.normal(0, 0.02)
    raw   = np.clip(raw + noise, 0.0, 1.0)
    return round(300 + raw * 600)


# ─────────────────────────────────────────────────────────────────────────────
# Numpy fallback data generation (used when SDV CSV is not present)
# ─────────────────────────────────────────────────────────────────────────────
def generate_profile(profile_type):
    """Generate one synthetic user profile via independent numpy sampling."""
    if profile_type == 'high':
        balance          = np.random.lognormal(mean=10.0, sigma=0.8)
        total_txns       = int(np.random.lognormal(mean=4.0, sigma=0.7))
        total_received   = np.random.lognormal(mean=10.5, sigma=0.7)
        sent_ratio       = np.random.uniform(0.2, 0.7)
        account_age_days = int(np.random.uniform(180, 1200))
        days_since_last  = int(np.random.uniform(0, 7))
    elif profile_type == 'medium':
        balance          = np.random.lognormal(mean=8.5, sigma=0.9)
        total_txns       = int(np.random.lognormal(mean=2.5, sigma=0.8))
        total_received   = np.random.lognormal(mean=9.0, sigma=0.8)
        sent_ratio       = np.random.uniform(0.4, 0.9)
        account_age_days = int(np.random.uniform(30, 365))
        days_since_last  = int(np.random.uniform(5, 30))
    else:
        balance          = np.random.lognormal(mean=6.5, sigma=1.0)
        total_txns       = int(np.random.lognormal(mean=1.0, sigma=1.0))
        total_received   = np.random.lognormal(mean=7.5, sigma=1.0)
        sent_ratio       = np.random.uniform(0.7, 1.2)
        account_age_days = int(np.random.uniform(1, 120))
        days_since_last  = int(np.random.uniform(14, 90))

    balance        = max(0.0, balance)
    total_txns     = max(0, total_txns)
    total_received = max(0.0, total_received)
    total_sent     = max(0.0, total_received * sent_ratio)
    avg_txn        = (total_sent + total_received) / max(1, total_txns * 2) if total_txns > 0 else 0.0
    weeks          = max(1, account_age_days / 7)

    return {
        'wallet_balance':         balance,
        'total_transactions':     float(total_txns),
        'total_sent':             total_sent,
        'total_received':         total_received,
        'avg_transaction_amount': avg_txn,
        'account_age_days':       float(account_age_days),
        'days_since_last_txn':    float(days_since_last if total_txns > 0 else account_age_days),
        'txn_frequency_per_week': total_txns / weeks,
    }


def generate_numpy_dataset(n=15_000):
    n_high   = int(n * 0.30)
    n_medium = int(n * 0.40)
    n_low    = n - n_high - n_medium

    records = []
    for ptype, count in [('high', n_high), ('medium', n_medium), ('low', n_low)]:
        for _ in range(count):
            f = generate_profile(ptype)
            records.append(f)
    return records


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load or generate data
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("GoPay Credit Engine — Training")
print("=" * 60)

if os.path.exists(SDV_DATA_PATH):
    try:
        import pandas as pd
        df = pd.read_csv(SDV_DATA_PATH)
        if 'credit_score' not in df.columns:
            raise ValueError("CSV missing 'credit_score' column")
        X = df[FEATURE_COLS].values
        y = df['credit_score'].values
        n_rows = len(df)
        data_source = f"SDV-enhanced  ({SDV_DATA_PATH}, {n_rows:,} rows)"
        print(f"\n[Data] Loaded SDV-enhanced dataset: {SDV_DATA_PATH}")
        print(f"       Rows: {n_rows:,}")
    except Exception as e:
        print(f"\n[Warn] Failed to load SDV data ({e}). Falling back to numpy generation.")
        os.path.exists(SDV_DATA_PATH) and print(f"       Corrupt file: {SDV_DATA_PATH}")
        records    = generate_numpy_dataset(15_000)
        X = np.array([[r[c] for c in FEATURE_COLS] for r in records])
        y = np.array([compute_label(r) for r in records])
        n_rows     = len(records)
        data_source = "numpy fallback (15,000 rows — run generate_sdv_data.py for better data)"
else:
    print(f"\n[Data] SDV data not found ({SDV_DATA_PATH}). Using numpy generation.")
    print("       Tip: run `python generate_sdv_data.py` first for enhanced training.")
    N_SAMPLES = 15_000
    print(f"       Generating {N_SAMPLES:,} synthetic profiles ...")
    records    = generate_numpy_dataset(N_SAMPLES)
    X = np.array([[r[c] for c in FEATURE_COLS] for r in records])
    y = np.array([compute_label(r) for r in records])
    n_rows     = N_SAMPLES
    data_source = "numpy fallback (15,000 rows — run generate_sdv_data.py for better data)"

print(f"\n[Data] Score range : {y.min():.0f} – {y.max():.0f}")
print(f"       Score mean  : {y.mean():.1f}")
print(f"       Score std   : {y.std():.1f}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Train / test split
# ─────────────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
print(f"\n[Split] Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Model selection — scale parameters with dataset size
# ─────────────────────────────────────────────────────────────────────────────
# Keep GradientBoostingRegressor for consistent feature importances and stable
# behavior. For larger datasets we increase estimator count and depth modestly.
if n_rows >= 30_000:
    print("\n[Model] Large dataset detected -> tuned GradientBoostingRegressor")
    n_estimators = 600
    max_depth = 6
    learning_rate = 0.03
    subsample = 0.85
else:
    print("\n[Model] Standard GradientBoostingRegressor")
    n_estimators = 300
    max_depth = 5
    learning_rate = 0.05
    subsample = 0.8

estimator = GradientBoostingRegressor(
    n_estimators=n_estimators,
    max_depth=max_depth,
    learning_rate=learning_rate,
    subsample=subsample,
    random_state=RANDOM_STATE,
)
model = Pipeline([
    ('scaler', StandardScaler()),
    ('gbr',   estimator),
])
model_name = "GradientBoostingRegressor"

# ─────────────────────────────────────────────────────────────────────────────
# 4. Train
# ─────────────────────────────────────────────────────────────────────────────
print(f"[Train] Fitting {model_name} ...")
model.fit(X_train, y_train)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Evaluate
# ─────────────────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
mae    = mean_absolute_error(y_test, y_pred)
r2     = r2_score(y_test, y_pred)

print(f"\n[Eval]  MAE on test set : {mae:.2f} score points")
print(f"        R2  on test set : {r2:.4f}")

# 5-fold cross validation on a sample for speed
cv_size    = min(10_000, len(X_train))
cv_indices = np.random.choice(len(X_train), cv_size, replace=False)
cv_scores  = cross_val_score(
    model, X_train[cv_indices], y_train[cv_indices],
    cv=5, scoring='neg_mean_absolute_error', n_jobs=-1,
)
cv_mae = -cv_scores.mean()
print(f"        CV-5 MAE        : {cv_mae:.2f} score points")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Save model
# ─────────────────────────────────────────────────────────────────────────────
joblib.dump({'model': model, 'features': FEATURE_COLS}, 'model.pkl')
print("\n[Save] model.pkl  -> saved")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Feature importances
# ─────────────────────────────────────────────────────────────────────────────
gbr_step    = model.named_steps['gbr']
importances = gbr_step.feature_importances_

with open('feature_importance.txt', 'w') as f:
    f.write(f"Feature importances ({model_name}):\n")
    f.write(f"Trained on: {data_source}\n\n")
    for name, imp in sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1]):
        bar = '#' * int(imp * 50)
        f.write(f"  {name:<30} {imp:.4f}  {bar}\n")

print("[Save] feature_importance.txt -> saved")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Training report
# ─────────────────────────────────────────────────────────────────────────────
report_lines = [
    "=" * 60,
    "GoPay Credit Engine — Training Report",
    "=" * 60,
    f"Data source    : {data_source}",
    f"Training rows  : {len(X_train):,}",
    f"Test rows      : {len(X_test):,}",
    f"Model          : {model_name}",
    f"MAE (test)     : {mae:.2f} score points",
    f"R2  (test)     : {r2:.4f}",
    f"CV-5 MAE       : {cv_mae:.2f} score points",
    "",
    "Feature importances:",
]
for name, imp in sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1]):
    report_lines.append(f"  {name:<30} {imp:.4f}")

with open('training_report.txt', 'w') as f:
    f.write("\n".join(report_lines))

print("[Save] training_report.txt -> saved")

print("\n" + "=" * 60)
print("Training complete.")
print(f"  Model    : {model_name}")
print(f"  Data     : {data_source}")
print(f"  MAE      : {mae:.2f}  (lower is better)")
print(f"  R2       : {r2:.4f}  (1.0 = perfect)")
print("\nNext: run `python app.py` to start the scoring API.")
print("=" * 60)
