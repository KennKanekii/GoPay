"""
GoPay Fraud Detection Engine — Model Training
================================================
Replicates real-world fraud patterns used by leading fintech companies
(Stripe Radar, PayPal, Razorpay, PhonePe) to train a RandomForestClassifier
that scores transactions 0-100 for fraud probability.

Fraud patterns modelled:
  1. Account Takeover (ATO)      — large amount to new recipient, unusual hour
  2. Velocity Fraud              — many rapid transactions (card testing pattern)
  3. Money Muling                — receives then immediately sends out large sums
  4. Structuring / Smurfing      — repeated just-below-limit amounts to evade detection
  5. Social Engineering          — normal-looking but new recipient + odd behavior

Usage:
    python train.py

Output:
    fraud_model.pkl   — trained model artifact
    feature_importance.txt
"""

import json
import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, roc_auc_score,
                             precision_recall_curve, average_precision_score)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Ratio of fraudulent samples in training data.
# Real-world fraud rate is 0.1-2%; we oversample to ~12% for better model learning.
FRAUD_RATIO    = 0.12
N_TOTAL        = 20_000
N_FRAUD        = int(N_TOTAL * FRAUD_RATIO)
N_LEGITIMATE   = N_TOTAL - N_FRAUD

# ---------------------------------------------------------------------------
# Feature definitions — mirroring what Spring Boot sends to this service
# ---------------------------------------------------------------------------
FEATURES = [
    'amount',                   # INR amount of transaction
    'amount_to_balance_ratio',  # amount / sender wallet balance (0-1+)
    'amount_to_avg_ratio',      # amount / sender's historical average (deviation)
    'txns_last_1h',             # transactions sent in last 1 hour
    'txns_last_24h',            # transactions sent in last 24 hours
    'amount_sent_last_1h',      # total INR sent in last 1 hour
    'amount_sent_last_24h',     # total INR sent in last 24 hours
    'unique_recipients_24h',    # distinct recipients in last 24 hours
    'is_new_recipient',         # 1 = first time sending to this person
    'hour_of_day',              # 0-23
    'is_night',                 # 1 if hour 0-5 (high-risk window)
    'is_weekend',               # 1 if Saturday or Sunday
    'account_age_days',         # how long sender's account has existed
    'is_round_amount',          # 1 if amount is multiple of 1000 (structuring signal)
    'is_blacklisted',           # 1 if recipient domain is on blacklist
    'balance_after_ratio',      # (balance - amount) / balance — how much is left
]


# ---------------------------------------------------------------------------
# 1. Generate LEGITIMATE transactions
# ---------------------------------------------------------------------------
def gen_legitimate(n):
    records = []
    for _ in range(n):
        balance         = np.random.lognormal(9.5, 0.8)           # ~13k median
        avg_txn         = np.random.lognormal(7.0, 0.7)           # ~1100 median
        amount          = max(10, np.random.lognormal(7.2, 0.6))  # typical payment
        amount          = min(amount, 99_000)

        # Legitimate: moderate velocity
        txns_1h         = np.random.choice([0, 0, 0, 1, 1, 2], p=[0.4, 0.2, 0.2, 0.1, 0.07, 0.03])
        txns_24h        = int(np.random.lognormal(1.0, 0.8))
        txns_24h        = min(txns_24h, 15)
        sent_1h         = txns_1h * avg_txn * np.random.uniform(0.5, 1.2)
        sent_24h        = txns_24h * avg_txn * np.random.uniform(0.6, 1.0)
        recipients_24h  = max(1, int(txns_24h * np.random.uniform(0.4, 0.9)))
        is_new_recip    = int(np.random.random() < 0.25)          # 25% new recipients
        hour_probs = [0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.04, 0.06,
                      0.08, 0.08, 0.07, 0.07, 0.07, 0.07, 0.06, 0.06,
                      0.06, 0.06, 0.05, 0.05, 0.04, 0.03, 0.02, 0.01]
        hour_probs = [p / sum(hour_probs) for p in hour_probs]  # normalize to sum=1
        hour            = int(np.random.choice(range(24), p=hour_probs))
        is_night        = int(hour < 6)
        is_weekend      = int(np.random.random() < 0.29)
        acct_age        = int(np.random.lognormal(4.5, 1.0))      # ~90 days median
        is_round        = int(amount % 1000 == 0 and np.random.random() < 0.1)
        balance_after   = max(0, balance - amount) / max(balance, 1)

        records.append({
            'amount':                  amount,
            'amount_to_balance_ratio': amount / max(balance, 1),
            'amount_to_avg_ratio':     amount / max(avg_txn, 1),
            'txns_last_1h':            txns_1h,
            'txns_last_24h':           txns_24h,
            'amount_sent_last_1h':     sent_1h,
            'amount_sent_last_24h':    sent_24h,
            'unique_recipients_24h':   recipients_24h,
            'is_new_recipient':        is_new_recip,
            'hour_of_day':             hour,
            'is_night':                is_night,
            'is_weekend':              is_weekend,
            'account_age_days':        acct_age,
            'is_round_amount':         is_round,
            'is_blacklisted':          0,
            'balance_after_ratio':     balance_after,
            'label':                   0,
        })
    return records


# ---------------------------------------------------------------------------
# 2. Generate FRAUDULENT transactions
#    Each pattern replicates a documented real-world fraud type.
# ---------------------------------------------------------------------------
def gen_fraud(n):
    records = []
    n_per_pattern = n // 5

    # --- Pattern 1: Account Takeover (ATO) ---
    # Attacker gains credentials, sends large amount to unfamiliar recipient
    # at an unusual hour (late night / early morning).
    for _ in range(n_per_pattern):
        balance = np.random.lognormal(9.5, 0.8)
        amount  = balance * np.random.uniform(0.6, 0.95)   # drain most of balance
        avg_txn = np.random.lognormal(6.5, 0.5)             # historical avg is low
        hour    = np.random.choice([0, 1, 2, 3, 4, 22, 23])

        records.append({
            'amount':                  amount,
            'amount_to_balance_ratio': amount / max(balance, 1),
            'amount_to_avg_ratio':     amount / max(avg_txn, 1),
            'txns_last_1h':            np.random.randint(0, 2),
            'txns_last_24h':           np.random.randint(1, 4),
            'amount_sent_last_1h':     amount,
            'amount_sent_last_24h':    amount,
            'unique_recipients_24h':   1,
            'is_new_recipient':        1,              # always new recipient
            'hour_of_day':             hour,
            'is_night':                int(hour < 6),
            'is_weekend':              int(np.random.random() < 0.5),
            'account_age_days':        np.random.randint(1, 60),
            'is_round_amount':         int(amount % 1000 < 50),
            'is_blacklisted':          int(np.random.random() < 0.2),
            'balance_after_ratio':     max(0, balance - amount) / max(balance, 1),
            'label':                   1,
        })

    # --- Pattern 2: Velocity Fraud / Card-Testing ---
    # Many rapid small transactions to test if account works,
    # escalating to larger amounts.
    for _ in range(n_per_pattern):
        balance = np.random.lognormal(8.5, 0.7)
        amount  = np.random.uniform(1, 500)        # small probe amounts
        avg_txn = np.random.lognormal(5.5, 0.5)
        txns_1h = np.random.randint(5, 15)          # high velocity
        hour    = int(np.random.uniform(0, 24))

        records.append({
            'amount':                  amount,
            'amount_to_balance_ratio': amount / max(balance, 1),
            'amount_to_avg_ratio':     amount / max(avg_txn, 1),
            'txns_last_1h':            txns_1h,
            'txns_last_24h':           txns_1h + np.random.randint(0, 5),
            'amount_sent_last_1h':     amount * txns_1h,
            'amount_sent_last_24h':    amount * txns_1h * 1.2,
            'unique_recipients_24h':   np.random.randint(3, 10),
            'is_new_recipient':        int(np.random.random() < 0.7),
            'hour_of_day':             hour,
            'is_night':                int(hour < 6),
            'is_weekend':              int(np.random.random() < 0.4),
            'account_age_days':        np.random.randint(1, 30),
            'is_round_amount':         0,
            'is_blacklisted':          0,
            'balance_after_ratio':     max(0, balance - amount) / max(balance, 1),
            'label':                   1,
        })

    # --- Pattern 3: Structuring / Smurfing ---
    # Multiple transactions just below reporting thresholds.
    # E.g., sending Rs. 9,900 repeatedly to avoid the Rs. 10,000 limit.
    for _ in range(n_per_pattern):
        balance    = np.random.lognormal(10.5, 0.5)
        threshold  = np.random.choice([10_000, 20_000, 50_000])
        amount     = threshold - np.random.uniform(100, 999)   # just under threshold
        avg_txn    = np.random.lognormal(7.0, 0.6)
        txns_24h   = np.random.randint(3, 8)
        hour       = int(np.random.uniform(9, 18))              # business hours (evasive)

        records.append({
            'amount':                  amount,
            'amount_to_balance_ratio': amount / max(balance, 1),
            'amount_to_avg_ratio':     amount / max(avg_txn, 1),
            'txns_last_1h':            np.random.randint(0, 3),
            'txns_last_24h':           txns_24h,
            'amount_sent_last_1h':     amount * np.random.uniform(0.5, 2),
            'amount_sent_last_24h':    amount * txns_24h * 0.8,
            'unique_recipients_24h':   np.random.randint(2, 6),
            'is_new_recipient':        int(np.random.random() < 0.5),
            'hour_of_day':             hour,
            'is_night':                0,
            'is_weekend':              int(np.random.random() < 0.2),
            'account_age_days':        np.random.randint(10, 180),
            'is_round_amount':         0,      # specifically NOT round (evasion)
            'is_blacklisted':          0,
            'balance_after_ratio':     max(0, balance - amount) / max(balance, 1),
            'label':                   1,
        })

    # --- Pattern 4: Money Mule ---
    # Account receives a large deposit then immediately sends it out.
    # High amount relative to historical average; often new recipients.
    for _ in range(n_per_pattern):
        balance  = np.random.lognormal(11.0, 0.5)   # high balance (just received)
        avg_txn  = np.random.lognormal(6.0, 0.5)    # historical avg is low
        amount   = balance * np.random.uniform(0.7, 0.99)
        txns_1h  = np.random.randint(1, 3)
        hour     = int(np.random.uniform(8, 20))

        records.append({
            'amount':                  amount,
            'amount_to_balance_ratio': amount / max(balance, 1),
            'amount_to_avg_ratio':     amount / max(avg_txn, 1),  # very high ratio
            'txns_last_1h':            txns_1h,
            'txns_last_24h':           txns_1h + np.random.randint(0, 3),
            'amount_sent_last_1h':     amount,
            'amount_sent_last_24h':    amount,
            'unique_recipients_24h':   np.random.randint(1, 3),
            'is_new_recipient':        int(np.random.random() < 0.8),
            'hour_of_day':             hour,
            'is_night':                int(hour < 6),
            'is_weekend':              int(np.random.random() < 0.4),
            'account_age_days':        np.random.randint(1, 45),
            'is_round_amount':         int(np.random.random() < 0.4),
            'is_blacklisted':          int(np.random.random() < 0.1),
            'balance_after_ratio':     max(0, balance - amount) / max(balance, 1),
            'label':                   1,
        })

    # --- Pattern 5: Blacklisted / Known Bad Actor ---
    # Recipient is on the blacklist — immediate high-risk signal.
    for _ in range(N_FRAUD - 4 * n_per_pattern):
        balance = np.random.lognormal(8.5, 1.0)
        avg_txn = np.random.lognormal(6.5, 0.7)
        amount  = np.random.lognormal(8.0, 0.8)
        amount  = min(amount, 99_000)
        hour    = int(np.random.uniform(0, 24))

        records.append({
            'amount':                  amount,
            'amount_to_balance_ratio': amount / max(balance, 1),
            'amount_to_avg_ratio':     amount / max(avg_txn, 1),
            'txns_last_1h':            np.random.randint(0, 3),
            'txns_last_24h':           np.random.randint(1, 8),
            'amount_sent_last_1h':     amount * np.random.uniform(0.5, 1.5),
            'amount_sent_last_24h':    amount * np.random.uniform(1.0, 2.5),
            'unique_recipients_24h':   np.random.randint(1, 5),
            'is_new_recipient':        int(np.random.random() < 0.6),
            'hour_of_day':             hour,
            'is_night':                int(hour < 6),
            'is_weekend':              int(np.random.random() < 0.4),
            'account_age_days':        np.random.randint(1, 365),
            'is_round_amount':         int(np.random.random() < 0.3),
            'is_blacklisted':          1,   # the defining signal for this pattern
            'balance_after_ratio':     max(0, balance - amount) / max(balance, 1),
            'label':                   1,
        })

    return records


# ---------------------------------------------------------------------------
# 3. Assemble dataset
# ---------------------------------------------------------------------------
print(f"Generating {N_LEGITIMATE} legitimate + {N_FRAUD} fraudulent transactions ...")
data = gen_legitimate(N_LEGITIMATE) + gen_fraud(N_FRAUD)
np.random.shuffle(data)

X = np.array([[r[f] for f in FEATURES] for r in data])
y = np.array([r['label'] for r in data])
print(f"  Dataset: {len(X)} samples  |  Fraud: {y.sum()} ({100*y.mean():.1f}%)")

# ---------------------------------------------------------------------------
# 4. Train / test split (stratified to keep fraud ratio in both sets)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------------------------
# 5. Model — RandomForestClassifier with class balancing
#    class_weight='balanced' automatically handles the fraud minority class.
# ---------------------------------------------------------------------------
print("Training RandomForestClassifier ...")
model = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=5,
        class_weight='balanced',
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )),
])
model.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# 6. Evaluation
# ---------------------------------------------------------------------------
y_prob  = model.predict_proba(X_test)[:, 1]
y_pred  = (y_prob >= 0.5).astype(int)
auc_roc = roc_auc_score(y_test, y_prob)
auc_pr  = average_precision_score(y_test, y_prob)

print("\n--- Test-set evaluation ---")
print(f"  AUC-ROC              : {auc_roc:.4f}  (1.0 = perfect)")
print(f"  AUC-PR               : {auc_pr:.4f}  (precision-recall; robust for imbalanced data)")
print("\n" + classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))

# 5-fold cross-validation AUC-ROC
cv_scores = cross_val_score(model, X, y, cv=StratifiedKFold(5), scoring='roc_auc', n_jobs=-1)
print(f"  5-fold CV AUC-ROC    : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

# ---------------------------------------------------------------------------
# 7. Save
# ---------------------------------------------------------------------------
artifact = {'model': model, 'features': FEATURES}
joblib.dump(artifact, 'fraud_model.pkl')
print("\nModel saved -> fraud_model.pkl")

importances = model.named_steps['rf'].feature_importances_
with open('feature_importance.txt', 'w') as f:
    f.write("Fraud model — feature importances (RandomForest):\n\n")
    for name, imp in sorted(zip(FEATURES, importances), key=lambda x: -x[1]):
        bar = '#' * int(imp * 60)
        f.write(f"  {name:<30} {imp:.4f}  {bar}\n")
print("Feature importances saved -> feature_importance.txt")
print("\nDone. Run `python app.py` to start the fraud scoring API on port 5002.")
