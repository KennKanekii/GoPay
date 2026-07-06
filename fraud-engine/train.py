"""
GoPay Fraud Detection Engine — Model Training (v2 — XGBoost + SDV)
====================================================================
Upgrades the original RandomForestClassifier to XGBoost (industry standard for
fraud detection at Razorpay, Setu, PhonePe, and most Indian fintech companies).

New features vs v1 (16 → 20 features):
  + vpa_risk_score   : Levenshtein/Jaro-Winkler VPA spoofing score (0-100)
  + ifsc_is_valid    : 1 if IFSC passes structural + bank registry check
  + amount_zscore    : (amount - sender_mean) / sender_std (z-score deviation)
  + sender_has_vpa   : 1 if sender registered a UPI VPA

New fraud archetypes (7, was 5):
  + VPA Spoofing     : lookalike UPI handle attacks
  + Fake IFSC Fraud  : invalid bank codes used in social-engineering attacks

Data strategy:
  Auto-detects fraud_sdv_data.csv (SDV-enhanced, 80k rows).
  Falls back to numpy generation (25k rows) if CSV not found.

Why XGBoost over RandomForest?
  - Handles class imbalance natively via scale_pos_weight
  - Better calibrated fraud probabilities (AUCPR improvement ~2-4%)
  - Tree regularization (L1/L2) reduces overfitting on synthetic data
  - Used by Stripe, Razorpay, PayPal as primary classifier

Usage:
  python generate_sdv_data.py   (generate SDV data first — recommended)
  python train.py               (trains on SDV if available)

Output:
  fraud_model.pkl        — model artifact loaded by app.py
  feature_importance.txt — ranked feature importances
  training_report.txt    — metrics and configuration summary
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import joblib

try:
    import pandas as pd
    PANDAS_OK = True
except ImportError:
    PANDAS_OK = False

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, roc_auc_score,
                             average_precision_score)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    import xgboost as xgb
    XGB_OK = True
except ImportError:
    XGB_OK = False
    from sklearn.ensemble import RandomForestClassifier
    print("[warn] xgboost not installed — falling back to RandomForestClassifier")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

SDV_DATA_PATH = 'fraud_sdv_data.csv'
FRAUD_RATIO   = 0.12
N_TOTAL       = 25_000
N_FRAUD       = int(N_TOTAL * FRAUD_RATIO)
N_LEGITIMATE  = N_TOTAL - N_FRAUD

# ─────────────────────────────────────────────────────────────────────────────
# Feature definitions (20 features — must match app.py and generate_sdv_data.py)
# ─────────────────────────────────────────────────────────────────────────────
FEATURES = [
    'amount', 'amount_to_balance_ratio', 'amount_to_avg_ratio',
    'txns_last_1h', 'txns_last_24h',
    'amount_sent_last_1h', 'amount_sent_last_24h',
    'unique_recipients_24h', 'is_new_recipient',
    'hour_of_day', 'is_night', 'is_weekend',
    'account_age_days', 'is_round_amount', 'is_blacklisted',
    'balance_after_ratio',
    # v2 additions:
    'vpa_risk_score',
    'ifsc_is_valid',
    'amount_zscore',
    'sender_has_vpa',
]


# ─────────────────────────────────────────────────────────────────────────────
# Numpy fallback data generation
# ─────────────────────────────────────────────────────────────────────────────
def gen_legitimate(n):
    records = []
    for _ in range(n):
        balance      = np.random.lognormal(9.5, 0.8)
        avg_txn      = np.random.lognormal(7.0, 0.7)
        mean_sent    = avg_txn * np.random.uniform(8, 25)
        std_sent     = mean_sent * np.random.uniform(0.3, 0.7)
        amount       = max(10, min(np.random.lognormal(7.2, 0.6), 99_000))
        txns_1h      = np.random.choice([0,0,0,1,1,2], p=[0.4,0.2,0.2,0.1,0.07,0.03])
        txns_24h     = min(int(np.random.lognormal(1.0, 0.8)), 15)
        sent_1h      = txns_1h * avg_txn * np.random.uniform(0.5, 1.2)
        sent_24h     = txns_24h * avg_txn * np.random.uniform(0.6, 1.0)
        recipients   = max(1, int(txns_24h * np.random.uniform(0.4, 0.9)))
        is_new_r     = int(np.random.random() < 0.25)
        hour_p       = [0.01,0.01,0.01,0.01,0.01,0.02,0.04,0.06,0.08,0.08,
                        0.07,0.07,0.07,0.07,0.06,0.06,0.06,0.06,0.05,0.05,
                        0.04,0.03,0.02,0.01]
        hour_p       = [p/sum(hour_p) for p in hour_p]
        hour         = int(np.random.choice(range(24), p=hour_p))
        acct_age     = int(np.random.lognormal(4.5, 1.0))
        is_round     = int(amount % 1000 == 0 and np.random.random() < 0.1)
        bal_after    = max(0, balance - amount) / max(balance, 1)
        zscore       = (amount - mean_sent) / max(std_sent, 1)
        records.append({
            'amount': amount, 'amount_to_balance_ratio': amount/max(balance,1),
            'amount_to_avg_ratio': amount/max(avg_txn,1),
            'txns_last_1h': txns_1h, 'txns_last_24h': txns_24h,
            'amount_sent_last_1h': sent_1h, 'amount_sent_last_24h': sent_24h,
            'unique_recipients_24h': recipients, 'is_new_recipient': is_new_r,
            'hour_of_day': hour, 'is_night': int(hour<6),
            'is_weekend': int(np.random.random()<0.29),
            'account_age_days': acct_age, 'is_round_amount': is_round,
            'is_blacklisted': 0, 'balance_after_ratio': bal_after,
            'vpa_risk_score': int(np.random.choice([0,0,5,10], p=[0.5,0.3,0.1,0.1])),
            'ifsc_is_valid': int(np.random.random()<0.95),
            'amount_zscore': zscore,
            'sender_has_vpa': int(np.random.random()<0.70),
            'label': 0,
        })
    return records


def gen_fraud(n_fraud):
    records = []
    n_per   = n_fraud // 7

    # ATO
    for _ in range(n_per):
        balance = np.random.lognormal(9.5, 0.8); avg_txn = np.random.lognormal(6.5, 0.5)
        amount  = balance * np.random.uniform(0.6, 0.95); hour = np.random.choice([0,1,2,3,4,22,23])
        records.append({'amount': amount, 'amount_to_balance_ratio': amount/max(balance,1),
            'amount_to_avg_ratio': amount/max(avg_txn,1), 'txns_last_1h': np.random.randint(0,2),
            'txns_last_24h': np.random.randint(1,4), 'amount_sent_last_1h': amount,
            'amount_sent_last_24h': amount, 'unique_recipients_24h': 1, 'is_new_recipient': 1,
            'hour_of_day': hour, 'is_night': int(hour<6), 'is_weekend': int(np.random.random()<0.5),
            'account_age_days': np.random.randint(1,60), 'is_round_amount': int(amount%1000<50),
            'is_blacklisted': int(np.random.random()<0.2),
            'balance_after_ratio': max(0,balance-amount)/max(balance,1),
            'vpa_risk_score': int(np.random.choice([0,20,40,60], p=[0.3,0.2,0.3,0.2])),
            'ifsc_is_valid': int(np.random.random()<0.5),
            'amount_zscore': (amount - avg_txn*5) / max(avg_txn*2, 1),
            'sender_has_vpa': int(np.random.random()<0.4), 'label': 1})

    # Velocity
    for _ in range(n_per):
        balance = np.random.lognormal(8.5, 0.7); avg_txn = np.random.lognormal(5.5, 0.5)
        amount  = np.random.uniform(1, 500); txns_1h = np.random.randint(5, 15)
        hour    = int(np.random.uniform(0, 24))
        records.append({'amount': amount, 'amount_to_balance_ratio': amount/max(balance,1),
            'amount_to_avg_ratio': amount/max(avg_txn,1), 'txns_last_1h': txns_1h,
            'txns_last_24h': txns_1h+np.random.randint(0,5),
            'amount_sent_last_1h': amount*txns_1h, 'amount_sent_last_24h': amount*txns_1h*1.2,
            'unique_recipients_24h': np.random.randint(3,10), 'is_new_recipient': int(np.random.random()<0.7),
            'hour_of_day': hour, 'is_night': int(hour<6), 'is_weekend': int(np.random.random()<0.4),
            'account_age_days': np.random.randint(1,30), 'is_round_amount': 0, 'is_blacklisted': 0,
            'balance_after_ratio': max(0,balance-amount)/max(balance,1),
            'vpa_risk_score': int(np.random.choice([0,10,20], p=[0.6,0.2,0.2])),
            'ifsc_is_valid': int(np.random.random()<0.7), 'amount_zscore': -2.0+np.random.normal(0,0.5),
            'sender_has_vpa': int(np.random.random()<0.5), 'label': 1})

    # Structuring
    for _ in range(n_per):
        balance = np.random.lognormal(10.5, 0.5); avg_txn = np.random.lognormal(7.0, 0.6)
        threshold = np.random.choice([10_000,20_000,50_000])
        amount  = threshold - np.random.uniform(100, 999); txns_24h = np.random.randint(3,8)
        hour    = int(np.random.uniform(9,18))
        records.append({'amount': amount, 'amount_to_balance_ratio': amount/max(balance,1),
            'amount_to_avg_ratio': amount/max(avg_txn,1), 'txns_last_1h': np.random.randint(0,3),
            'txns_last_24h': txns_24h, 'amount_sent_last_1h': amount*np.random.uniform(0.5,2),
            'amount_sent_last_24h': amount*txns_24h*0.8, 'unique_recipients_24h': np.random.randint(2,6),
            'is_new_recipient': int(np.random.random()<0.5), 'hour_of_day': hour, 'is_night': 0,
            'is_weekend': int(np.random.random()<0.2), 'account_age_days': np.random.randint(10,180),
            'is_round_amount': 0, 'is_blacklisted': 0,
            'balance_after_ratio': max(0,balance-amount)/max(balance,1),
            'vpa_risk_score': int(np.random.choice([0,5,10], p=[0.7,0.2,0.1])),
            'ifsc_is_valid': int(np.random.random()<0.85),
            'amount_zscore': (amount-avg_txn*10)/max(avg_txn*3,1),
            'sender_has_vpa': int(np.random.random()<0.6), 'label': 1})

    # Money Mule
    for _ in range(n_per):
        balance = np.random.lognormal(11.0, 0.5); avg_txn = np.random.lognormal(6.0, 0.5)
        amount  = balance * np.random.uniform(0.7, 0.99); txns_1h = np.random.randint(1,3)
        hour    = int(np.random.uniform(8,20))
        records.append({'amount': amount, 'amount_to_balance_ratio': amount/max(balance,1),
            'amount_to_avg_ratio': amount/max(avg_txn,1), 'txns_last_1h': txns_1h,
            'txns_last_24h': txns_1h+np.random.randint(0,3),
            'amount_sent_last_1h': amount, 'amount_sent_last_24h': amount,
            'unique_recipients_24h': np.random.randint(1,3), 'is_new_recipient': int(np.random.random()<0.8),
            'hour_of_day': hour, 'is_night': int(hour<6), 'is_weekend': int(np.random.random()<0.4),
            'account_age_days': np.random.randint(1,45), 'is_round_amount': int(np.random.random()<0.4),
            'is_blacklisted': int(np.random.random()<0.1),
            'balance_after_ratio': max(0,balance-amount)/max(balance,1),
            'vpa_risk_score': int(np.random.choice([0,10,30], p=[0.5,0.3,0.2])),
            'ifsc_is_valid': int(np.random.random()<0.6),
            'amount_zscore': (amount-avg_txn*10)/max(avg_txn*3,1),
            'sender_has_vpa': int(np.random.random()<0.3), 'label': 1})

    # Blacklisted
    for _ in range(n_per):
        balance = np.random.lognormal(8.5, 1.0); avg_txn = np.random.lognormal(6.5, 0.7)
        amount  = min(np.random.lognormal(8.0, 0.8), 99_000); hour = int(np.random.uniform(0,24))
        records.append({'amount': amount, 'amount_to_balance_ratio': amount/max(balance,1),
            'amount_to_avg_ratio': amount/max(avg_txn,1), 'txns_last_1h': np.random.randint(0,3),
            'txns_last_24h': np.random.randint(1,8),
            'amount_sent_last_1h': amount*np.random.uniform(0.5,1.5),
            'amount_sent_last_24h': amount*np.random.uniform(1.0,2.5),
            'unique_recipients_24h': np.random.randint(1,5), 'is_new_recipient': int(np.random.random()<0.6),
            'hour_of_day': hour, 'is_night': int(hour<6), 'is_weekend': int(np.random.random()<0.4),
            'account_age_days': np.random.randint(1,365), 'is_round_amount': int(np.random.random()<0.3),
            'is_blacklisted': 1, 'balance_after_ratio': max(0,balance-amount)/max(balance,1),
            'vpa_risk_score': int(np.random.choice([10,40,70,90], p=[0.2,0.3,0.3,0.2])),
            'ifsc_is_valid': int(np.random.random()<0.3),
            'amount_zscore': (amount-avg_txn*10)/max(avg_txn*3,1),
            'sender_has_vpa': int(np.random.random()<0.5), 'label': 1})

    # VPA Spoofing
    for _ in range(n_per):
        balance = np.random.lognormal(9.0, 0.7); avg_txn = np.random.lognormal(7.0, 0.6)
        amount  = min(np.random.lognormal(8.5, 0.7), 99_000); hour = int(np.random.uniform(0,24))
        records.append({'amount': amount, 'amount_to_balance_ratio': amount/max(balance,1),
            'amount_to_avg_ratio': amount/max(avg_txn,1), 'txns_last_1h': np.random.randint(0,2),
            'txns_last_24h': np.random.randint(1,5),
            'amount_sent_last_1h': amount*np.random.uniform(0.5,1.2),
            'amount_sent_last_24h': amount*np.random.uniform(0.8,1.5),
            'unique_recipients_24h': np.random.randint(1,3), 'is_new_recipient': 1,
            'hour_of_day': hour, 'is_night': int(hour<6), 'is_weekend': int(np.random.random()<0.4),
            'account_age_days': np.random.randint(1,90), 'is_round_amount': int(np.random.random()<0.2),
            'is_blacklisted': int(np.random.random()<0.15),
            'balance_after_ratio': max(0,balance-amount)/max(balance,1),
            'vpa_risk_score': int(np.random.choice([60,70,80,90,100], p=[0.2,0.2,0.3,0.2,0.1])),
            'ifsc_is_valid': int(np.random.random()<0.4),
            'amount_zscore': (amount-avg_txn*8)/max(avg_txn*2,1),
            'sender_has_vpa': int(np.random.random()<0.6), 'label': 1})

    # Fake IFSC
    remaining = n_fraud - 6 * n_per
    for _ in range(remaining):
        balance = np.random.lognormal(9.0, 0.8); avg_txn = np.random.lognormal(7.0, 0.6)
        amount  = min(np.random.lognormal(9.0, 0.6), 99_000); hour = int(np.random.uniform(6,22))
        records.append({'amount': amount, 'amount_to_balance_ratio': amount/max(balance,1),
            'amount_to_avg_ratio': amount/max(avg_txn,1), 'txns_last_1h': np.random.randint(0,2),
            'txns_last_24h': np.random.randint(1,4),
            'amount_sent_last_1h': amount*np.random.uniform(0.3,0.8),
            'amount_sent_last_24h': amount*np.random.uniform(0.5,1.5),
            'unique_recipients_24h': np.random.randint(1,3), 'is_new_recipient': 1,
            'hour_of_day': hour, 'is_night': 0, 'is_weekend': int(np.random.random()<0.3),
            'account_age_days': np.random.randint(1,120), 'is_round_amount': int(np.random.random()<0.3),
            'is_blacklisted': int(np.random.random()<0.1),
            'balance_after_ratio': max(0,balance-amount)/max(balance,1),
            'vpa_risk_score': int(np.random.choice([10,30,50], p=[0.3,0.4,0.3])),
            'ifsc_is_valid': 0,
            'amount_zscore': (amount-avg_txn*8)/max(avg_txn*2,1),
            'sender_has_vpa': int(np.random.random()<0.5), 'label': 1})

    return records


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load or generate data
# ─────────────────────────────────────────────────────────────────────────────
print("="*60)
print("GoPay Fraud Engine — Training (v2 XGBoost)")
print("="*60)

if os.path.exists(SDV_DATA_PATH) and PANDAS_OK:
    try:
        df = pd.read_csv(SDV_DATA_PATH)
        if 'label' not in df.columns:
            raise ValueError("Missing 'label' column")
        X = df[FEATURES].values
        y = df['label'].values.astype(int)
        data_source = f"SDV-enhanced ({SDV_DATA_PATH}, {len(df):,} rows)"
        print(f"\n[Data] Loaded SDV data: {len(df):,} rows")
    except Exception as e:
        print(f"[Warn] SDV load failed ({e}) — falling back to numpy")
        data = gen_legitimate(N_LEGITIMATE) + gen_fraud(N_FRAUD)
        np.random.shuffle(data)
        X = np.array([[r[f] for f in FEATURES] for r in data])
        y = np.array([r['label'] for r in data], dtype=int)
        data_source = f"numpy ({N_TOTAL:,} rows)"
else:
    print(f"\n[Data] SDV data not found. Generating {N_TOTAL:,} rows via numpy ...")
    print("       Tip: run `python generate_sdv_data.py` first for better data.")
    data = gen_legitimate(N_LEGITIMATE) + gen_fraud(N_FRAUD)
    np.random.shuffle(data)
    X = np.array([[r[f] for f in FEATURES] for r in data])
    y = np.array([r['label'] for r in data], dtype=int)
    data_source = f"numpy ({N_TOTAL:,} rows)"

n_legit = int((y == 0).sum())
n_fraud = int((y == 1).sum())
print(f"[Data] Total: {len(X):,}  |  Legit: {n_legit:,}  |  Fraud: {n_fraud:,}  ({100*n_fraud/len(X):.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Train / test split (stratified)
# ─────────────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
print(f"[Split] Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Model — XGBoost (preferred) with class imbalance handling
# ─────────────────────────────────────────────────────────────────────────────
scale_pos_weight = n_legit / max(n_fraud, 1)

if XGB_OK:
    print(f"\n[Model] XGBoostClassifier  (scale_pos_weight={scale_pos_weight:.2f})")
    estimator = xgb.XGBClassifier(
        n_estimators       = 500,
        max_depth          = 6,
        learning_rate      = 0.05,
        subsample          = 0.85,
        colsample_bytree   = 0.85,
        scale_pos_weight   = scale_pos_weight,  # handles class imbalance
        reg_alpha          = 0.1,               # L1 regularization
        reg_lambda         = 1.0,               # L2 regularization
        eval_metric        = 'aucpr',           # optimise for precision-recall
        random_state       = RANDOM_STATE,
        n_jobs             = -1,
        verbosity          = 0,
    )
    model_name = 'xgboost_v2'
else:
    print("\n[Model] RandomForestClassifier (XGBoost not installed)")
    estimator = RandomForestClassifier(
        n_estimators=400, max_depth=12, min_samples_leaf=5,
        class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE,
    )
    model_name = 'random_forest_v2'

model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf',    estimator),
])

print("[Train] Fitting model ...")
model.fit(X_train, y_train)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Evaluation
# ─────────────────────────────────────────────────────────────────────────────
y_prob  = model.predict_proba(X_test)[:, 1]
y_pred  = (y_prob >= 0.50).astype(int)
auc_roc = roc_auc_score(y_test, y_prob)
auc_pr  = average_precision_score(y_test, y_prob)

print("\n--- Test-set evaluation ---")
print(f"  AUC-ROC : {auc_roc:.4f}  (1.0 = perfect)")
print(f"  AUC-PR  : {auc_pr:.4f}  (robust for class imbalance)")
print("\n" + classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))

cv_k     = min(5, n_fraud // 5)
if cv_k >= 2:
    cv_scores = cross_val_score(
        model, X, y, cv=StratifiedKFold(cv_k), scoring='roc_auc', n_jobs=-1
    )
    print(f"  {cv_k}-fold CV AUC-ROC : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Save model
# ─────────────────────────────────────────────────────────────────────────────
artifact = {'model': model, 'features': FEATURES, 'model_name': model_name}
joblib.dump(artifact, 'fraud_model.pkl')
print("\n[Save] fraud_model.pkl -> saved")

# Feature importances
clf_step    = model.named_steps['clf']
importances = clf_step.feature_importances_

with open('feature_importance.txt', 'w') as f:
    f.write(f"Fraud model feature importances ({model_name}):\n")
    f.write(f"Data: {data_source}\n\n")
    for name, imp in sorted(zip(FEATURES, importances), key=lambda x: -x[1]):
        bar = '#' * int(imp * 60)
        f.write(f"  {name:<30} {imp:.4f}  {bar}\n")
print("[Save] feature_importance.txt -> saved")

# Training report
with open('training_report.txt', 'w') as f:
    f.write("="*60 + "\nGoPay Fraud Engine — Training Report\n" + "="*60 + "\n")
    f.write(f"Model          : {model_name}\n")
    f.write(f"Data source    : {data_source}\n")
    f.write(f"Train rows     : {len(X_train):,}\n")
    f.write(f"Test rows      : {len(X_test):,}\n")
    f.write(f"Fraud rate     : {100*n_fraud/len(X):.1f}%\n")
    f.write(f"AUC-ROC        : {auc_roc:.4f}\n")
    f.write(f"AUC-PR         : {auc_pr:.4f}\n")
    f.write(f"Features       : {len(FEATURES)}\n")

print("[Save] training_report.txt -> saved")
print(f"\n{'='*60}\nDone.\n  Model : {model_name}\n  AUC-ROC: {auc_roc:.4f}\n  AUC-PR : {auc_pr:.4f}")
print("\nNext: run `python app.py` to start fraud API on port 5002.\n" + "="*60)
