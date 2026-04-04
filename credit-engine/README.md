# GoPay Credit Engine

A **GradientBoosting ML model** (scikit-learn) that scores users 300–900 based on
their real platform activity (wallet balance, transaction patterns, net cash flow, etc.)

## Architecture

```
Frontend (React)
    ↓ GET /api/v1/credit/score
Spring Boot (port 8080)
    ↓ POST http://localhost:5001/score  (features JSON)
Python Flask ML service (port 5001)
    ↓ GradientBoostingRegressor
Score 300-900 + breakdown
```

If the Python service is **not running**, Spring Boot automatically falls back to
a **rule-based score** so the feature still works.

## Features used by the model

| Feature | Description |
|---|---|
| `wallet_balance` | Current INR balance |
| `total_transactions` | Total number of transactions |
| `total_sent` | Total INR sent to others |
| `total_received` | Total INR received from others |
| `avg_transaction_amount` | Average transaction size |
| `account_age_days` | Days since account creation |
| `days_since_last_txn` | Days since last transaction |
| `txn_frequency_per_week` | Transactions per week |

## Setup

```bash
cd credit-engine

# 1. Install dependencies (all free, no paid APIs)
pip install -r requirements.txt

# 2. Train the model (generates model.pkl from 15,000 synthetic profiles)
python train.py

# 3. Start the scoring API
python app.py
```

Service runs on `http://localhost:5001`.

## Retraining on real data

As your user base grows, replace synthetic data with real anonymised profiles:

```python
# In train.py, replace the generate_profile() loop with:
real_records = load_from_your_db()  # your actual data
X = extract_features(real_records)
y = compute_labels(real_records)
model.fit(X_train, y_train)
joblib.dump({'model': model, 'features': FEATURE_COLS}, 'model.pkl')
```

Then restart `app.py` — Spring Boot picks it up automatically.
