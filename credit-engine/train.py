"""
GoPay Credit Engine — Model Training
=====================================
Generates a synthetic dataset of 15,000 user financial profiles and
trains a GradientBoostingRegressor to predict credit scores (300-900).

Usage:
    python train.py

Output:
    model.pkl   — trained model (loaded by app.py at runtime)
    feature_importance.txt — human-readable feature importances
"""

import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
N_SAMPLES = 15_000

np.random.seed(RANDOM_STATE)

# ---------------------------------------------------------------------------
# 1. Synthetic feature generation
# ---------------------------------------------------------------------------
# Features we derive from real platform data:
#   wallet_balance            — current wallet balance (INR)
#   total_transactions        — total number of transactions
#   total_sent                — total INR sent
#   total_received            — total INR received
#   avg_transaction_amount    — average transaction size (INR)
#   account_age_days          — days since account was created
#   days_since_last_txn       — 0 if no transactions
#   txn_frequency_per_week    — transactions per week

def generate_profile(profile_type):
    """
    Generate one synthetic user profile. Profile types:
      'high'   → creditworthy (score 650-900)
      'medium' → borderline   (score 450-700)
      'low'    → risky        (score 300-500)
    """
    if profile_type == 'high':
        balance            = np.random.lognormal(mean=10.0, sigma=0.8)  # ~22k median
        total_txns         = int(np.random.lognormal(mean=4.0, sigma=0.7))
        total_received     = np.random.lognormal(mean=10.5, sigma=0.7)
        sent_ratio         = np.random.uniform(0.2, 0.7)
        account_age_days   = int(np.random.uniform(180, 1200))
        days_since_last    = int(np.random.uniform(0, 7))
    elif profile_type == 'medium':
        balance            = np.random.lognormal(mean=8.5, sigma=0.9)   # ~5k median
        total_txns         = int(np.random.lognormal(mean=2.5, sigma=0.8))
        total_received     = np.random.lognormal(mean=9.0, sigma=0.8)
        sent_ratio         = np.random.uniform(0.4, 0.9)
        account_age_days   = int(np.random.uniform(30, 365))
        days_since_last    = int(np.random.uniform(5, 30))
    else:  # low
        balance            = np.random.lognormal(mean=6.5, sigma=1.0)   # ~665 median
        total_txns         = int(np.random.lognormal(mean=1.0, sigma=1.0))
        total_received     = np.random.lognormal(mean=7.5, sigma=1.0)
        sent_ratio         = np.random.uniform(0.7, 1.2)  # may have spent more than received
        account_age_days   = int(np.random.uniform(1, 120))
        days_since_last    = int(np.random.uniform(14, 90))

    balance = max(0.0, balance)
    total_txns = max(0, total_txns)
    total_received = max(0.0, total_received)
    total_sent = max(0.0, total_received * sent_ratio)
    avg_txn = (total_sent + total_received) / max(1, total_txns * 2) if total_txns > 0 else 0.0
    weeks = max(1, account_age_days / 7)
    freq_per_week = total_txns / weeks

    return {
        'wallet_balance': balance,
        'total_transactions': total_txns,
        'total_sent': total_sent,
        'total_received': total_received,
        'avg_transaction_amount': avg_txn,
        'account_age_days': account_age_days,
        'days_since_last_txn': days_since_last if total_txns > 0 else account_age_days,
        'txn_frequency_per_week': freq_per_week,
    }


def compute_label(f):
    """
    Deterministic score formula — this is the 'ground truth' the model learns.
    Score range: 300-900.
    """
    # Balance factor (0-1): full marks at ₹75,000
    b_score = min(1.0, f['wallet_balance'] / 75_000)

    # Activity factor (0-1): full marks at 200 transactions
    a_score = min(1.0, f['total_transactions'] / 200)

    # Net flow factor (0-1): penalise if sent >> received
    total_flow = f['total_sent'] + f['total_received']
    if total_flow > 0:
        nf_score = f['total_received'] / total_flow       # 1.0 if all received, 0.5 if balanced
    else:
        nf_score = 0.5

    # Recency factor (0-1): penalise long gaps
    r_score = max(0.0, 1.0 - f['days_since_last_txn'] / 60) if f['total_transactions'] > 0 else 0.0

    # Account age factor (0-1): full marks at 3 years
    age_score = min(1.0, f['account_age_days'] / 1095)

    # Weighted combination
    raw = (
        b_score  * 0.30 +
        a_score  * 0.25 +
        nf_score * 0.25 +
        r_score  * 0.10 +
        age_score * 0.10
    )

    # Add mild noise so the model doesn't overfit perfectly
    noise = np.random.normal(0, 0.02)
    raw = np.clip(raw + noise, 0.0, 1.0)

    return round(300 + raw * 600)   # Map [0,1] → [300,900]


# ---------------------------------------------------------------------------
# 2. Generate dataset
# ---------------------------------------------------------------------------
print(f"Generating {N_SAMPLES} synthetic profiles …")

profiles_per_type = {
    'high':   int(N_SAMPLES * 0.30),
    'medium': int(N_SAMPLES * 0.40),
    'low':    N_SAMPLES - int(N_SAMPLES * 0.30) - int(N_SAMPLES * 0.40),
}

records = []
for ptype, count in profiles_per_type.items():
    for _ in range(count):
        f = generate_profile(ptype)
        f['score'] = compute_label(f)
        records.append(f)

FEATURE_COLS = [
    'wallet_balance', 'total_transactions', 'total_sent', 'total_received',
    'avg_transaction_amount', 'account_age_days', 'days_since_last_txn',
    'txn_frequency_per_week',
]

X = np.array([[r[c] for c in FEATURE_COLS] for r in records])
y = np.array([r['score'] for r in records])

print(f"  Score distribution — min: {y.min()}, max: {y.max()}, mean: {y.mean():.1f}")

# ---------------------------------------------------------------------------
# 3. Train / test split
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# ---------------------------------------------------------------------------
# 4. Build pipeline: scaler + GradientBoosting
# ---------------------------------------------------------------------------
model = Pipeline([
    ('scaler', StandardScaler()),
    ('gbr', GradientBoostingRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=RANDOM_STATE,
    )),
])

print("Training GradientBoostingRegressor …")
model.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# 5. Evaluate
# ---------------------------------------------------------------------------
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
print(f"  MAE on test set : {mae:.1f} score points")
print(f"  R²  on test set : {r2:.4f}")

# ---------------------------------------------------------------------------
# 6. Save model + feature importances
# ---------------------------------------------------------------------------
joblib.dump({'model': model, 'features': FEATURE_COLS}, 'model.pkl')
print("Model saved -> model.pkl")

importances = model.named_steps['gbr'].feature_importances_
with open('feature_importance.txt', 'w') as f:
    f.write("Feature importances (GradientBoosting):\n\n")
    for name, imp in sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1]):
        bar = '#' * int(imp * 50)
        f.write(f"  {name:<30} {imp:.4f}  {bar}\n")

print("Feature importances saved -> feature_importance.txt")
print("\nDone. Run `python app.py` to start the scoring API.")
