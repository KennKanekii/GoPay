# GoPay ML Models — Complete Technical Reference

> **Purpose:** Interview-ready deep dive into the Credit Score Engine and Fraud Risk Scorer built inside GoPay. Every algorithm decision, architectural choice, and data engineering detail is documented here.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Credit Score Engine](#2-credit-score-engine)
   - 2.1 [What It Does](#21-what-it-does)
   - 2.2 [Algorithm: Gradient Boosting Regressor](#22-algorithm-gradient-boosting-regressor)
   - 2.3 [How Gradient Boosting Works — Step by Step](#23-how-gradient-boosting-works--step-by-step)
   - 2.4 [Feature Engineering](#24-feature-engineering)
   - 2.5 [Ground Truth Formula (Synthetic Label Generation)](#25-ground-truth-formula-synthetic-label-generation)
   - 2.6 [Synthetic Data Generation Strategy](#26-synthetic-data-generation-strategy)
   - 2.7 [Model Training Pipeline](#27-model-training-pipeline)
   - 2.8 [Score Bands](#28-score-bands)
   - 2.9 [System Architecture (End-to-End)](#29-system-architecture-end-to-end)
   - 2.10 [Rule-Based Java Fallback](#210-rule-based-java-fallback)
   - 2.11 [Model Performance](#211-model-performance)
   - 2.12 [Retraining on Real Data](#212-retraining-on-real-data)
3. [Fraud Risk Scorer](#3-fraud-risk-scorer)
   - 3.1 [What It Does](#31-what-it-does)
   - 3.2 [Algorithm: Random Forest Classifier](#32-algorithm-random-forest-classifier)
   - 3.3 [How Random Forest Works — Step by Step](#33-how-random-forest-works--step-by-step)
   - 3.4 [The 5 Fraud Archetypes](#34-the-5-fraud-archetypes)
   - 3.5 [Feature Vector (16 Features)](#35-feature-vector-16-features)
   - 3.6 [Synthetic Fraud Data Generation](#36-synthetic-fraud-data-generation)
   - 3.7 [Class Imbalance — The Core Challenge](#37-class-imbalance--the-core-challenge)
   - 3.8 [The 5-Layer Decision Stack (Cascade Architecture)](#38-the-5-layer-decision-stack-cascade-architecture)
   - 3.9 [Velocity Rules — The Hard Limits](#39-velocity-rules--the-hard-limits)
   - 3.10 [Blacklist Engine](#310-blacklist-engine)
   - 3.11 [Behavioural Signal Extraction](#311-behavioural-signal-extraction)
   - 3.12 [Risk Bands and Recommendations](#312-risk-bands-and-recommendations)
   - 3.13 [Audit Trail and Compliance](#313-audit-trail-and-compliance)
   - 3.14 [System Architecture (End-to-End)](#314-system-architecture-end-to-end)
   - 3.15 [Model Performance](#315-model-performance)
4. [Comparison: Credit Score vs Fraud Score](#4-comparison-credit-score-vs-fraud-score)
5. [Key Interview Q&A](#5-key-interview-qa)
6. [Real-World Benchmarks and Industry Parallels](#6-real-world-benchmarks-and-industry-parallels)

---

## 1. System Overview

GoPay runs **two independent ML systems** that serve fundamentally different purposes:

| Dimension | Credit Score Engine | Fraud Risk Scorer |
|---|---|---|
| **Question answered** | "Is this user creditworthy long-term?" | "Is this specific transaction suspicious right now?" |
| **Time horizon** | Longitudinal (full account history) | Transactional (last 1–24 hours) |
| **ML task** | Regression (continuous score 300–900) | Classification (fraud probability 0–1 → score 0–100) |
| **Model** | GradientBoostingRegressor | RandomForestClassifier |
| **Python port** | 5001 | 5002 |
| **Decision** | Score + risk band (informational) | ALLOW / REVIEW / BLOCK (operational) |
| **When it runs** | On demand (user views credit page) | Before every transaction (blocking call) |
| **Fallback** | Java rule-based scoring | Java rule-based scoring + velocity rules |

---

## 2. Credit Score Engine

### 2.1 What It Does

The Credit Score Engine assigns every GoPay user a score between **300 and 900** — the same scale used by CIBIL (Credit Information Bureau India), Experian, and Equifax — representing their financial creditworthiness based entirely on their behaviour on the GoPay platform.

A higher score indicates:
- Consistent wallet activity
- Healthy net cash inflows
- Long, stable account history
- Recent transaction activity (not dormant)

The score is **not a random number** — it is the output of a trained ML model that has learned to weigh these factors the same way traditional credit bureaus do, but using payment app data instead of loan repayment history.

---

### 2.2 Algorithm: Gradient Boosting Regressor

**Model chosen:** `sklearn.ensemble.GradientBoostingRegressor`

**Hyperparameters used (adaptive by dataset size):**
```python
GradientBoostingRegressor(
    # baseline config (15k numpy dataset):
    # n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8
    #
    # large-data config (>=30k, used with SDV dataset):
    n_estimators   = 600,    # more trees for richer decision boundaries
    max_depth      = 6,      # captures higher-order feature interactions
    learning_rate  = 0.03,   # smaller step for better generalization
    subsample      = 0.85,   # stochastic boosting with slightly more coverage
    random_state   = 42,
)
```

**Why GBR over alternatives?**

| Alternative | Why not chosen |
|---|---|
| Linear Regression | Cannot capture non-linear relationships (e.g., balance effect is not linear at high values) |
| Decision Tree | High variance — overfits to training data |
| Random Forest | Better for classification; GBR typically achieves lower MSE on regression tasks |
| Neural Network | Black box — cannot explain individual factor contributions; overkill for tabular data |
| GradientBoostingRegressor | Best bias-variance tradeoff for tabular regression; used by FICO, Experian in production |

**The Pipeline:**
```python
Pipeline([
    ('scaler', StandardScaler()),         # normalise features to N(0,1)
    ('gbr',    GradientBoostingRegressor(...))
])
```

`StandardScaler` is critical — it ensures that `wallet_balance` (values in thousands) and `days_since_last_txn` (values in single digits) are on the same scale, preventing the model from being dominated by large-magnitude features.

---

### 2.3 How Gradient Boosting Works — Step by Step

Gradient Boosting builds trees **sequentially**, where each tree corrects the mistakes of all previous trees.

**Mathematical foundation:**

```
Step 0:  F₀(x) = mean(y)     ← start with the simplest possible prediction
         e.g., F₀ = 494.5 (mean of all training scores)

Step 1:  Compute residuals:
         r₁ = y - F₀(x)
         (how wrong is our current prediction for each sample?)

         Fit Tree₁ to predict r₁
         F₁(x) = F₀(x) + learning_rate × Tree₁(x)

Step 2:  Compute new residuals:
         r₂ = y - F₁(x)

         Fit Tree₂ to predict r₂
         F₂(x) = F₁(x) + learning_rate × Tree₂(x)

...repeat for n_estimators=300 trees...

Final:   F₃₀₀(x) = F₀ + 0.05×T₁ + 0.05×T₂ + ... + 0.05×T₃₀₀
```

**Why learning_rate = 0.05 (small)?**

A small learning rate means each tree makes a small correction. This forces the model to need more trees (300 instead of maybe 50) but produces a much smoother, more generalizable fit. This is called **shrinkage**.

**Why subsample = 0.8?**

Each tree is trained on only 80% of the data (chosen randomly). This is called **Stochastic Gradient Boosting**. The randomness:
- Reduces correlation between trees
- Reduces overfitting
- Speeds up training

**Why max_depth = 5?**

Each tree can make at most 5 splits. A depth-5 tree can model interactions of up to 5 features simultaneously (e.g., "high balance AND high activity AND positive net flow"). Deeper trees overfit; shallower trees underfit.

---

### 2.4 Feature Engineering

Eight features are computed from real platform data (users.json + transactions.json):

```
Feature                     │ Source                    │ Computation
────────────────────────────┼───────────────────────────┼──────────────────────────────────────
wallet_balance              │ users.json                │ current balance (INR)
total_transactions          │ transactions.json         │ count of all txns for user
total_sent                  │ transactions.json         │ sum(amount) where fromUserId=user
total_received              │ transactions.json         │ sum(amount) where toUserId=user
avg_transaction_amount      │ derived                   │ (total_sent + total_received) / total_transactions
account_age_days            │ users.json.createdAt      │ ChronoUnit.DAYS.between(createdAt, now)
days_since_last_txn         │ transactions.json         │ days since most recent transaction
txn_frequency_per_week      │ derived                   │ total_transactions / (account_age_days / 7)
```

**Why these 8 specifically?**

These mirror the **5 C's of Credit** used by traditional lenders:
- **Capacity** → `wallet_balance`, `total_sent`, `total_received`
- **Character** → `account_age_days`, `txn_frequency_per_week`
- **Capital** → `avg_transaction_amount`
- **Conditions** → `days_since_last_txn`
- **Collateral** → Not applicable (digital wallet)

---

### 2.5 Ground Truth Formula (Synthetic Label Generation)

Since we have no historical loan default data, we generate labels using a **deterministic formula** that encodes the same logic credit bureaus use. The model then learns to approximate this formula from examples.

```python
def compute_label(features):

    # 1. Balance factor (0–1): full marks at Rs. 75,000
    #    Why 75k? Represents ~6 months of median Indian salary in savings.
    b_score = min(1.0, wallet_balance / 75_000)

    # 2. Activity factor (0–1): full marks at 200 transactions
    #    Active users have more data, lower uncertainty = lower risk.
    a_score = min(1.0, total_transactions / 200)

    # 3. Net flow factor (0–1): what fraction of flow is incoming?
    #    received/(sent+received) = 1.0 if all money is received (healthy)
    #                             = 0.5 if perfectly balanced
    #                             = 0.0 if all money sent out (concerning)
    total_flow = total_sent + total_received
    nf_score = total_received / total_flow if total_flow > 0 else 0.5

    # 4. Recency factor (0–1): penalise dormant accounts
    #    Score decays linearly to 0 after 60 days of inactivity.
    r_score = max(0.0, 1.0 - days_since_last_txn / 60) if total_transactions > 0 else 0.0

    # 5. Account age factor (0–1): full marks at 3 years (1095 days)
    #    Older accounts have more stable, predictable behaviour.
    age_score = min(1.0, account_age_days / 1095)

    # Weighted combination (weights sum to 1.0)
    raw = (
        b_score   * 0.30 +    # balance is strongest signal
        a_score   * 0.25 +    # activity is second strongest
        nf_score  * 0.25 +    # net flow is equally important
        r_score   * 0.10 +    # recency adds nuance
        age_score * 0.10      # maturity adds stability
    )

    # Add mild Gaussian noise so model doesn't memorise perfectly
    noise = np.random.normal(0, 0.02)
    raw = np.clip(raw + noise, 0.0, 1.0)

    # Map [0, 1] → [300, 900] (CIBIL scale)
    return round(300 + raw * 600)
```

**Why add noise?**

Without noise, the model would achieve R²=1.0 (perfect) by memorising the exact formula. The noise (σ=0.02, equivalent to ±12 score points) forces the model to learn the **generalizable pattern** rather than the exact arithmetic.

---

### 2.6 Synthetic Data Generation Strategy

We now use a **two-stage synthetic strategy**:

1. **Seed generation (rule-driven, 5,000-6,000 rows):** controlled profile buckets.
2. **SDV expansion (60,000+ rows):** learn joint feature distribution and sample large-scale realistic data.

This is implemented in `credit-engine/generate_sdv_data.py`.

#### Stage A: Seed data (original logic)

Seed profiles are still generated across three risk categories:

```
High credit (30%)  → expected scores 650–900
  - balance:       lognormal(mean=10.0, σ=0.8)  → median ~Rs. 22,000
  - transactions:  lognormal(mean=4.0, σ=0.7)   → median ~54 transactions
  - account_age:   uniform(180, 1200)            → 6 months to 3.3 years
  - days_since:    uniform(0, 7)                 → very recently active

Medium credit (40%) → expected scores 450–700
  - balance:       lognormal(mean=8.5, σ=0.9)   → median ~Rs. 4,900
  - transactions:  lognormal(mean=2.5, σ=0.8)   → median ~12 transactions
  - account_age:   uniform(30, 365)              → 1 month to 1 year
  - days_since:    uniform(5, 30)                → active but not daily

Low credit (30%)   → expected scores 300–500
  - balance:       lognormal(mean=6.5, σ=1.0)   → median ~Rs. 665
  - transactions:  lognormal(mean=1.0, σ=1.0)   → median ~2–3 transactions
  - account_age:   uniform(1, 120)               → new account
  - days_since:    uniform(14, 90)               → infrequent/dormant
```

#### Stage B: SDV enhancement (new)

Using **Synthetic Data Vault (SDV)**:
- Metadata inferred via `SingleTableMetadata`.
- Synthesizer selection supports `gaussian`, `ctgan`, `tvae`, and `auto`.
- In `auto` mode, pipeline evaluates candidate synthesizers and selects the best-quality output.
- Generated dataset is saved to `credit_sdv_data.csv` and consumed directly by `train.py`.

Quality guardrails are computed in `sdv_quality_report.txt`:
- Mean and standard deviation drift per feature
- Correlation drift across key pairs
- Composite quality score (target >= 0.65)

Latest run:
- Selected synthesizer: `GaussianCopulaSynthesizer`
- Output rows: `60,000`
- Quality score: `0.876`

**Why this is better than only direct numpy sampling?**
- Numpy seed generation creates each profile type from predefined independent random draws.
- SDV learns cross-feature dependencies (joint distribution), so generated users have more realistic combinations of balance, activity, recency, and account age.

**Why lognormal distributions in the seed stage?**

Financial quantities (income, balance, transaction amounts) follow **lognormal distributions** in the real world — most people have moderate amounts, but a few have very large amounts (long right tail). Using `np.random.lognormal` replicates this accurately.

---

### 2.7 Model Training Pipeline

```python
# 1. Load SDV data if present, else fallback to numpy generation
if os.path.exists('credit_sdv_data.csv'):
    df = pd.read_csv('credit_sdv_data.csv')
    X = df[FEATURE_COLS].values
    y = df['credit_score'].values
else:
    records = generate_profiles()
    X = np.array([[r[f] for f in FEATURE_COLS] for r in records])
    y = np.array([compute_label(r) for r in records])

# 2. Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Configure model capacity by dataset size
if len(X) >= 30000:
    params = dict(n_estimators=600, max_depth=6, learning_rate=0.03, subsample=0.85)
else:
    params = dict(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8)

# 4. Build sklearn Pipeline (scaler + GBR)
model = Pipeline([
    ('scaler', StandardScaler()),
    ('gbr',    GradientBoostingRegressor(**params))
])

# 5. Train
model.fit(X_train, y_train)

# 6. Evaluate (holdout + CV)
y_pred = model.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)
cv5_mae = -cross_val_score(model, X_train_sample, y_train_sample,
                            scoring='neg_mean_absolute_error', cv=5).mean()

# 7. Persist
joblib.dump({'model': model, 'features': FEATURE_COLS}, 'model.pkl')
```

**Current production-oriented run (with SDV data):**

- Data source: `credit_sdv_data.csv` (`60,000` rows)
- Train/Test: `48,000 / 12,000`
- Holdout MAE: `2.09`
- Holdout R2: `0.9989`
- 5-fold CV MAE: `3.35`

**Interpretation of MAE:**

On average, prediction error is ~2 score points on holdout data. On the 300-900 scale (600 points wide), this is ~0.35% error, showing very tight fit to the synthetic ground-truth process.

---

### 2.8 Score Bands

```
Score Range │ Band       │ Colour   │ Interpretation
────────────┼────────────┼──────────┼─────────────────────────────────────────────────────
800 – 900   │ EXCELLENT  │ #16a34a  │ Exceptional. Pre-approved for best loan rates.
740 – 799   │ VERY_GOOD  │ #4ade80  │ Very strong. Most lenders approve without conditions.
670 – 739   │ GOOD       │ #86efac  │ Good standing. Eligible for standard loan products.
580 – 669   │ FAIR       │ #fbbf24  │ Fair. Some lenders require higher interest rates.
300 – 579   │ POOR       │ #ef4444  │ Poor. Limited eligibility; focus on improving activity.
```

These thresholds exactly match CIBIL's published score bands, making the GoPay score directly comparable to real bureau scores.

---

### 2.9 System Architecture (End-to-End)

```
User navigates to /credit
        │
        ▼
React CreditScore.tsx
        │  GET /api/v1/credit/score
        │  Authorization: Bearer <token>
        ▼
Spring Boot CreditController.java
        │  getUserByToken(authHeader) → StoredUser
        ▼
CreditService.java — Feature Engineering
        │
        ├── Read users.json         → wallet_balance, createdAt
        ├── Read transactions.json  → total_sent, total_received,
        │                             total_transactions, last_txn_date
        └── Compute 8 features
        │
        │  POST http://localhost:5001/score   (3s timeout)
        │  Body: { "wallet_balance": 11000, "total_transactions": 3, ... }
        ▼
Python Flask app.py (port 5001)
        │
        ├── StandardScaler.transform(features)
        └── GradientBoostingRegressor.predict()
        │
        │  Response: { "score": 484, "riskBand": "POOR", "breakdown": {...} }
        ▼
CreditService.java — Parse response
        │
        ├── If Python service times out → ruleFallback(feat)
        │                                  (same formula as train.py, computed in Java)
        └── Return CreditScoreResponse
        │
        ▼
React renders:
        ├── SVG arc gauge (300–900)
        ├── Per-factor progress bars (0–100)
        └── Raw feature grid (8 metrics)
```

---

### 2.10 Rule-Based Java Fallback

If the Python service is unreachable (timeout or not running), CreditService computes the score deterministically in Java using the exact same formula:

```java
double bScore   = Math.min(1.0, f.walletBalance / 75_000.0);
double aScore   = Math.min(1.0, f.totalTransactions / 200.0);
double total    = f.totalSent + f.totalReceived;
double nfScore  = total > 0 ? f.totalReceived / total : 0.5;
double rScore   = f.totalTransactions > 0
    ? Math.max(0, 1.0 - f.daysSinceLastTxn / 60.0) : 0.0;
double ageScore = Math.min(1.0, f.accountAgeDays / 1095.0);

double raw   = bScore*0.30 + aScore*0.25 + nfScore*0.25 + rScore*0.10 + ageScore*0.10;
int    score = (int) Math.round(300 + raw * 600);
score = Math.max(300, Math.min(900, score));
```

The response includes `"model": "rule_based_fallback"` so the frontend can indicate to the user that ML scoring was unavailable.

---

### 2.11 Model Performance

```
Metric                           │ Baseline (15k) │ Improved (SDV 60k) │ Interpretation
─────────────────────────────────┼────────────────┼────────────────────┼──────────────────────────────────────────────
MAE (holdout)                    │ 10.4           │ 2.09               │ Average absolute prediction error
R² (holdout)                     │ 0.9818         │ 0.9989             │ Variance explained by model
CV-5 MAE                         │ N/A            │ 3.35               │ Stability across folds
Training set size                │ 12,000         │ 48,000             │ 80% split
Test set size                    │ 3,000          │ 12,000             │ 20% split
Primary training data source     │ Numpy synthetic│ SDV-enhanced       │ Better joint-distribution realism
```

---

### 2.12 Retraining on Real Data

As the GoPay user base grows, retraining can be done in a **hybrid loop**:

1. Extract real anonymised platform data.
2. Build high-quality seed records from real behaviour.
3. Expand dataset using SDV (`generate_sdv_data.py`) for volume + coverage.
4. Retrain `train.py` (auto-detects `credit_sdv_data.csv`).
5. Save `model.pkl` and restart the credit Flask service.

```python
# Real-data seed creation (example pattern)
real_users = load_from_users_json()
real_txns  = load_from_transactions_json()

records = []
for user in real_users:
    user_txns  = [t for t in real_txns if t['fromUserId'] == user['id'] or t['toUserId'] == user['id']]
    features   = extract_features(user, user_txns)
    label      = compute_label(features)   # same formula
    records.append({**features, 'score': label})

# Optional: merge real-seed + SDV-expanded samples
save_seed_csv(records, 'credit_seed_real.csv')
run_sdv_expansion('credit_seed_real.csv', output='credit_sdv_data.csv')

# Train the same pipeline
model.fit(X_train, y_train)
joblib.dump({'model': model, 'features': FEATURE_COLS}, 'model.pkl')
# Restart app.py — Spring Boot picks it up automatically
```

---

## 3. Fraud Risk Scorer

### 3.1 What It Does

The Fraud Risk Scorer evaluates every transaction **before money moves**. It assigns a fraud probability score (0–100) and issues one of three recommendations:

- **ALLOW** (score 0–59): Transaction proceeds normally
- **REVIEW** (score 60–79): Transaction proceeds but is flagged prominently in the audit log
- **BLOCK** (score 80–100): Transaction is rejected with a user-facing reason

Unlike the credit score (which is informational), the fraud score is **operational** — a BLOCK recommendation stops the transaction completely.

Every assessment is logged to `fraud_events.json` for compliance, dispute resolution, and model retraining.

---

### 3.2 Algorithm: Random Forest Classifier

**Model chosen:** `sklearn.ensemble.RandomForestClassifier`

**Hyperparameters used:**
```python
RandomForestClassifier(
    n_estimators   = 400,          # 400 independent decision trees
    max_depth      = 12,           # each tree can make 12 splits
    min_samples_leaf = 5,          # each leaf must have ≥5 samples (reduces overfitting)
    class_weight   = 'balanced',   # automatically handles fraud class imbalance
    n_jobs         = -1,           # use all CPU cores for parallel training
    random_state   = 42,
)
```

**Why Random Forest for fraud (not GBR)?**

| Criterion | Random Forest | Gradient Boosting |
|---|---|---|
| Task type | Classification (fraud/not fraud) | Regression (continuous score) |
| Class imbalance | `class_weight='balanced'` built-in | Requires manual sample weights |
| Explainability | Feature importances per tree, easy audit | Similar but slower to compute |
| Outlier robustness | Very robust (each tree votes) | More sensitive to extreme values |
| Parallel training | Yes (trees are independent) | No (sequential by design) |
| Speed at inference | O(n_estimators × max_depth) | Same |

Random Forest is the **industry standard for fraud detection** (used by Stripe Radar, Square, Razorpay) because:
1. It outputs calibrated probabilities via `predict_proba()`
2. `class_weight='balanced'` elegantly handles the 99:1 class imbalance
3. It is fully explainable (SHAP values, feature importances)
4. Regulators (RBI, SEBI) require explainability for financial decisions

---

### 3.3 How Random Forest Works — Step by Step

```
Training phase:
───────────────
Dataset: 20,000 transactions (17,600 legitimate + 2,400 fraud)

For each of 400 trees:
  1. Bootstrap sample: randomly select N transactions WITH replacement
     (each tree sees ~63% of unique samples; ~37% are out-of-bag)

  2. For each node in the tree, consider only √16 ≈ 4 random features
     (not all 16 — this is the key innovation of Random Forest)

  3. Split on the feature + threshold that best separates fraud from legit
     (using Gini impurity criterion)

  4. Stop splitting when:
     - max_depth = 12 is reached, OR
     - node has < min_samples_leaf = 5 samples

Result: 400 diverse decision trees, each slightly different


Inference phase (for a new transaction):
─────────────────────────────────────────
Feed the 16-feature vector to all 400 trees simultaneously

Each tree outputs a vote:
  - Tree 1:   "NOT FRAUD"
  - Tree 2:   "FRAUD"
  - Tree 3:   "NOT FRAUD"
  - ...
  - Tree 400: "FRAUD"

Tally:
  fraud_votes = 160
  legit_votes = 240

fraud_probability = 160 / 400 = 0.40
fraud_score       = round(0.40 × 100) = 40   → MEDIUM risk
```

**Why bootstrap sampling (step 1)?**

Each tree is trained on a different random subset of data, so they make different errors. When you average their votes, the errors **cancel out**. This is called **bagging** (Bootstrap AGGregating) and is the core reason Random Forests outperform single trees dramatically.

**Why random feature selection at each split (step 2)?**

If all trees used all features, they would all make similar splits (dominated by the strongest features like `is_blacklisted`). By restricting each split to √16 ≈ 4 random features, trees are forced to use different features, making them **decorrelated**. Decorrelated trees produce much better ensembles.

---

### 3.4 The 5 Fraud Archetypes

The training data is built by simulating **5 documented real-world fraud patterns** observed in RBI FIU reports, Stripe Radar research, and PayPal fraud disclosures.

---

#### Archetype 1: Account Takeover (ATO)

**Real-world description:**
An attacker obtains the victim's login credentials (via phishing, credential stuffing, or SIM swap), logs in, and immediately transfers the wallet balance to their own account — typically at night when the victim is asleep.

**Synthetic profile:**
```python
balance         = lognormal(9.5, 0.8)          # victim had normal balance
amount          = balance × uniform(0.6, 0.95)  # drain 60–95% of balance
avg_txn         = lognormal(6.5, 0.5)           # historical avg is low
                                                 # (victim made small payments before)
hour            = choice([0,1,2,3,4,22,23])     # always late night
is_new_recip    = 1                              # attacker's account is new
account_age     = randint(1, 60)                 # short account history (targeted)
```

**Signals that fire:**
- `amount_to_balance_ratio`: 0.6–0.95 → draining the account
- `is_new_recipient`: 1 → sending to unknown person
- `is_night`: 1 → midnight to 4 AM
- `amount_to_avg_ratio`: 5–20× → massive deviation from their normal pattern
- `account_age_days` < 60 → new accounts are targeted more often

---

#### Archetype 2: Velocity Fraud / Card Testing

**Real-world description:**
A fraudster who has stolen access to multiple accounts runs automated scripts to make many small transactions rapidly — testing if accounts are still valid and finding the spend limit before making a larger withdrawal.

**Synthetic profile:**
```python
balance         = lognormal(8.5, 0.7)
amount          = uniform(1, 500)          # very small probe amounts
txns_last_1h    = randint(5, 15)           # high velocity — 5 to 15 per hour
unique_recips   = randint(3, 10)           # sending to many different people
account_age     = randint(1, 30)           # freshly compromised accounts
is_new_recip    = bernoulli(p=0.7)         # 70% chance recipient is new
```

**Signals that fire:**
- `txns_last_1h` ≥ 5 → immediately triggers velocity hard rule (BLOCK)
- `unique_recipients_24h` high → card testing pattern
- `amount` very small but many of them → probing behaviour
- `account_age_days` < 30 → recently compromised

---

#### Archetype 3: Structuring / Smurfing

**Real-world description:**
To avoid transaction monitoring thresholds (Rs. 10,000, Rs. 20,000, Rs. 50,000), fraudsters deliberately break up large transfers into multiple smaller amounts just below the threshold. This is called **structuring** and is a recognised money laundering technique under PMLA (Prevention of Money Laundering Act).

**Example:** Instead of transferring Rs. 50,000 once, they send Rs. 9,900 × 6 = Rs. 59,400 across 6 transactions to 6 different recipients.

**Synthetic profile:**
```python
balance         = lognormal(10.5, 0.5)     # has significant funds
threshold       = choice([10_000, 20_000, 50_000])
amount          = threshold - uniform(100, 999)   # e.g., Rs. 9,100–9,999
avg_txn         = lognormal(7.0, 0.6)             # normal historical avg
txns_24h        = randint(3, 8)                   # multiple transactions today
hour            = uniform(9, 18)                  # business hours (evasive!)
is_round_amount = 0                               # deliberately NOT round
```

**Signals that fire:**
- `amount` consistently just below threshold values
- `txns_last_24h` high (3–8)
- `is_round_amount`: 0 — but amount is suspiciously close to a round threshold
- `unique_recipients_24h`: spread across accounts (smurfing = multiple mules)

**Note on `hour`:** Structuring often happens during business hours (9 AM–6 PM) to blend in with legitimate activity — making it harder to detect with simple time-of-day rules alone. The ML model catches it via the combination of amount + frequency + recipient spread.

---

#### Archetype 4: Money Muling

**Real-world description:**
A "money mule" is someone (witting or unwitting) who receives stolen funds in their account and is instructed to immediately forward them to another account — taking a commission. The account is used as a layer of indirection to obscure the original crime.

**Synthetic profile:**
```python
balance         = lognormal(11.0, 0.5)     # high balance (just received large deposit)
avg_txn         = lognormal(6.0, 0.5)      # historical avg is LOW
                                            # (normal user with suddenly large balance)
amount          = balance × uniform(0.7, 0.99)  # forward almost everything
txns_last_1h    = randint(1, 3)             # one or a few large transactions
is_new_recip    = bernoulli(p=0.8)          # 80% chance — forwarding to unknown
```

**The key signal:**
`amount_to_avg_ratio` = (Rs. 80,000 forward) / (Rs. 1,200 historical avg) = **66.7×**

This is an extreme outlier. No legitimate user forwards 66× their normal transaction amount unless something unusual has happened.

**Signals that fire:**
- `amount_to_avg_ratio`: 10–50× historical average
- `amount_to_balance_ratio`: 0.7–0.99
- `is_new_recipient`: 1 (forwarding to the criminal's destination account)
- `balance` unusually high relative to account history

---

#### Archetype 5: Blacklisted Actor

**Real-world description:**
The recipient's email or domain is on the GoPay Trust & Safety blacklist — known disposable email providers, accounts reported for fraud, or known criminal destinations.

**Design decision:** This is handled as a **hard rule before ML** (Layer 1), not as a learned pattern. The ML model does include `is_blacklisted` as a feature (to inform the probability even in borderline cases), but any transaction where `is_blacklisted=1` is immediately flagged as CRITICAL regardless of the ML score.

**Blacklisted domains include:**
```
mailinator.com      guerrillamail.com    guerrillamail.net
tempmail.com        temp-mail.org        throwaway.email
dispostable.com     sharklasers.com      spam4.me
trashmail.com       trashmail.net        yopmail.com
fakeinbox.com       maildrop.cc          getairmail.com
spamgourmet.com     spamgourmet.net      mailnull.com
```

**Why disposable emails?**
Fraudsters use disposable email providers to create throwaway accounts that cannot be traced back to them. Any GoPay account registered with a disposable email is a strong fraud signal for any transaction involving them.

---

### 3.5 Feature Vector (16 Features)

```
Feature                   │ Type    │ Range         │ Fraud signal direction
──────────────────────────┼─────────┼───────────────┼──────────────────────────────────────────
amount                    │ float   │ 1 – 100,000   │ Higher = more risk (for unusual accounts)
amount_to_balance_ratio   │ float   │ 0 – 2+        │ Higher = more risk (draining account)
amount_to_avg_ratio       │ float   │ 0 – 100+      │ Higher = more risk (deviation from normal)
txns_last_1h              │ int     │ 0 – 50+       │ Higher = more risk (velocity)
txns_last_24h             │ int     │ 0 – 200+      │ Higher = more risk (velocity)
amount_sent_last_1h       │ float   │ 0 – 1,00,000  │ Higher = more risk (high hourly spend)
amount_sent_last_24h      │ float   │ 0 – 1,00,000  │ Higher = more risk (high daily spend)
unique_recipients_24h     │ int     │ 0 – 20+       │ Higher = more risk (fan-out pattern)
is_new_recipient          │ binary  │ 0 or 1        │ 1 = more risk (unknown destination)
hour_of_day               │ int     │ 0 – 23        │ 0–5 = higher risk (night)
is_night                  │ binary  │ 0 or 1        │ 1 = more risk (midnight to 5 AM)
is_weekend                │ binary  │ 0 or 1        │ 1 = slightly more risk
account_age_days          │ int     │ 1 – 3650+     │ Lower = more risk (new account)
is_round_amount           │ binary  │ 0 or 1        │ 1 = structuring signal (contextual)
is_blacklisted            │ binary  │ 0 or 1        │ 1 = CRITICAL (immediate block)
balance_after_ratio       │ float   │ 0 – 1         │ Lower = more risk (leaving account empty)
```

**Real example feature vector (ATO scenario):**
```json
{
  "amount":                  8500.0,
  "amount_to_balance_ratio": 0.85,
  "amount_to_avg_ratio":     12.3,
  "txns_last_1h":            1,
  "txns_last_24h":           1,
  "amount_sent_last_1h":     8500.0,
  "amount_sent_last_24h":    8500.0,
  "unique_recipients_24h":   1,
  "is_new_recipient":        1,
  "hour_of_day":             2,
  "is_night":                1,
  "is_weekend":              0,
  "account_age_days":        22,
  "is_round_amount":         0,
  "is_blacklisted":          0,
  "balance_after_ratio":     0.15
}
→ fraud_probability: 0.94 → fraud_score: 94 → CRITICAL → BLOCK
```

---

### 3.6 Synthetic Fraud Data Generation

```
Total samples:   20,000
Legitimate:      17,600  (88%)
Fraudulent:       2,400  (12%)

Breakdown by archetype (2,400 fraud total):
  ATO          : 480  (2.4% of total)
  Velocity     : 480  (2.4% of total)
  Structuring  : 480  (2.4% of total)
  Money Mule   : 480  (2.4% of total)
  Blacklisted  : 480  (2.4% of total)
```

**Why 12% fraud rate in training (vs 0.1% in real life)?**

If we trained on 0.1% fraud, the model would see only 20 fraud examples — not enough to learn meaningful patterns. We use **oversampling** (12%) in training while acknowledging that real deployment will have much lower rates. The `class_weight='balanced'` parameter adjusts the loss function to compensate.

---

### 3.7 Class Imbalance — The Core Challenge

**The problem:**
In real-world fraud detection, legitimate transactions vastly outnumber fraudulent ones (typically 99.9% vs 0.1%). A naive model that always predicts "NOT FRAUD" would achieve 99.9% accuracy — completely useless.

**The metric that matters: AUC-PR, not accuracy**

```
Accuracy = 99.9%  → model always says "not fraud"
Precision = 1.0   → of all "fraud" predictions, what fraction are actually fraud?
Recall    = 1.0   → of all actual frauds, what fraction did we catch?
AUC-PR    = 1.0   → area under precision-recall curve (best for imbalanced data)
```

**How `class_weight='balanced'` fixes it:**

```python
# Scikit-learn computes per-class weights automatically:
weight_for_fraud     = n_total / (n_classes × n_fraud_samples)
                     = 20000  / (2        × 2400)
                     = 4.17

weight_for_legitimate = 20000 / (2 × 17600)
                      = 0.568
```

Each fraud training example now contributes **4.17×** as much to the loss function as a legitimate example. The model is forced to learn fraud patterns aggressively, accepting some false positives (flagging legitimate transactions) in exchange for high recall (catching actual fraud).

**The false positive / false negative tradeoff:**

```
False Positive (FP): blocking a LEGITIMATE transaction
  → User is frustrated, loses trust, may churn
  → Handled by: REVIEW band (60–79) instead of BLOCK — allow but monitor

False Negative (FN): allowing a FRAUDULENT transaction
  → User loses money, GoPay bears liability
  → Handled by: conservative threshold (score ≥ 60 = flag, ≥ 80 = block)
```

---

### 3.8 The 5-Layer Decision Stack (Cascade Architecture)

This architecture mirrors how Stripe Radar and PayPal's fraud engine are structured: **cheapest checks first**, most expensive last.

```
┌─────────────────────────────────────────────────────────────────────┐
│                     TRANSACTION REQUEST                              │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 1: BLACKLIST CHECK                                            │
│  Cost: O(1) HashSet lookup — < 1ms                                  │
│  Check: recipient email/domain in blockedDomains set?               │
│  If YES → fraudScore=95, riskLevel=CRITICAL, recommendation=BLOCK   │
│  If NO  → continue                                                   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 2: HARD VELOCITY RULES                                        │
│  Cost: Read transactions.json + filter — < 5ms                      │
│                                                                      │
│  Rule 1: txns_last_1h ≥ 5         → BLOCK (RBI P2P limit)           │
│  Rule 2: amount_sent_last_1h      │
│          + new_amount > Rs.20,000 → BLOCK (hourly amount limit)     │
│  Rule 3: txns_last_24h ≥ 20       → BLOCK (daily count limit)       │
│  Rule 4: amount_sent_last_24h     │
│          + new_amount > Rs.1,00,000 → BLOCK (daily amount limit)    │
│                                                                      │
│  If any rule fires → immediate BLOCK                                 │
│  If none fire      → continue                                        │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 3+4: ML SCORING                                               │
│  Cost: HTTP POST to Python service — 50–500ms                       │
│                                                                      │
│  1. Build 16-feature vector (from real platform data)               │
│  2. POST http://localhost:5002/assess                                │
│  3. RandomForest.predict_proba() → fraud_probability                │
│  4. fraud_score = round(fraud_probability × 100)                    │
│  5. Extract human-readable signals from feature values              │
│                                                                      │
│  score 0–34   → LOW      → ALLOW                                    │
│  score 35–59  → MEDIUM   → ALLOW  (logged, monitored)               │
│  score 60–79  → HIGH     → REVIEW (flagged in transaction history)  │
│  score ≥ 80   → CRITICAL → BLOCK                                    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  (Python service unreachable)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 5: RULE-BASED FALLBACK                                        │
│  Cost: Pure Java computation — < 1ms                                 │
│                                                                      │
│  Same signals as ML model but computed deterministically:            │
│  - velocity score     (0–35 points)                                  │
│  - balance drain      (+20 if ratio ≥ 0.75)                         │
│  - unusual amount     (+15 if ratio ≥ 5×)                           │
│  - new large recip    (+15 if new + amount ≥ Rs.10,000)             │
│  - night + new recip  (+10 if 0–5 AM + new recipient)              │
│  - many recipients    (+10 if ≥ 8 in 24h)                           │
│  - new account large  (+10 if age < 7 days + amount ≥ Rs.5,000)    │
│                                                                      │
│  Same thresholds: ≥ 80 → BLOCK                                       │
│  Response includes "model": "rule_based_fallback" for transparency  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  AUDIT LOGGING                                                       │
│  Every assessment → fraud_events.json                               │
│  Fields: id, fromUserId, toIdentifier, amount, fraudScore,          │
│          riskLevel, recommendation, signals[], model, createdAt      │
│  Retention: last 1,000 events (configurable)                        │
└─────────────────────────────────────────────────────────────────────┘
```

**Why this ordering is critical:**

The **cascade** structure means we spend expensive compute (HTTP call, ML inference) only when necessary. If the blacklist catches 5% of fraud cases in 1ms, we avoid the 500ms ML call for those cases entirely. At scale (millions of transactions/day), this saves enormous compute cost.

---

### 3.9 Velocity Rules — The Hard Limits

These limits are derived from RBI's transaction monitoring guidelines and NPCI's UPI velocity framework:

```
Limit                          │ Value        │ Rationale
───────────────────────────────┼──────────────┼──────────────────────────────────────────
Max transactions per hour      │ 5            │ NPCI UPI recommendation for P2P
Max amount sent per hour       │ Rs. 20,000   │ Typical hourly spend for a retail user
Max transactions per 24 hours  │ 20           │ Covers daily payments + some buffer
Max amount sent per 24 hours   │ Rs. 1,00,000 │ RBI P2P transaction monitoring threshold
Max unique recipients per day  │ 8            │ Limits fan-out patterns
New recipient limit            │ Rs. 10,000   │ UPI guideline for first-time payee
Large transaction threshold    │ Rs. 50,000   │ Enhanced monitoring flag
```

These are **hard rules** (not ML) because:
1. Regulators require demonstrable, explainable controls
2. Zero latency — no network calls needed
3. They catch velocity fraud patterns with 100% recall

---

### 3.10 Blacklist Engine

```java
// Loaded from fraud-engine/blacklist.json at Spring Boot startup
private final Set<String> blockedDomains   = new HashSet<>();   // O(1) lookup
private final Set<String> blockedEmails    = new HashSet<>();
private final Set<String> blockedKeywords  = new HashSet<>();

private boolean isBlacklisted(String identifier) {
    // 1. Exact email match
    if (blockedEmails.contains(identifier)) return true;

    // 2. Domain match (extract domain from email)
    int at = identifier.indexOf('@');
    if (at >= 0 && blockedDomains.contains(identifier.substring(at+1))) return true;

    // 3. Keyword match (e.g., "fraud" in identifier)
    for (String kw : blockedKeywords)
        if (identifier.contains(kw)) return true;

    return false;
}
```

**Three-level check:**
1. Exact email address (specific reported bad actors)
2. Domain (all accounts from disposable email providers)
3. Keywords (catches variations like `fraud123@gmail.com`)

---

### 3.11 Behavioural Signal Extraction

After the ML model scores a transaction, the Python service extracts **human-readable signals** from the feature values. These are shown in the Fraud Shield dashboard and the "Transaction Blocked" error message.

```python
def extract_signals(data, fraud_prob):
    signals = []

    if data['is_blacklisted']:
        signals.append({'code': 'blacklisted_recipient', 'severity': 'CRITICAL',
                        'label': 'Recipient is on the GoPay fraud blacklist'})

    if data['txns_last_1h'] >= 4:
        signals.append({'code': 'high_velocity_1h', 'severity': 'HIGH',
                        'label': f"{data['txns_last_1h']} transactions in last 1 hour"})

    if data['amount_to_balance_ratio'] >= 0.75:
        signals.append({'code': 'high_balance_drain', 'severity': 'HIGH',
                        'label': f"Transaction drains {pct}% of wallet balance"})

    if data['amount_to_avg_ratio'] >= 5:
        signals.append({'code': 'unusual_amount', 'severity': 'HIGH',
                        'label': f"Amount is {ratio}x the sender's historical average"})

    if data['is_new_recipient'] and data['amount'] >= 10_000:
        signals.append({'code': 'large_new_recipient', 'severity': 'MEDIUM',
                        'label': 'Large amount to a first-time recipient'})

    if data['is_night'] and data['is_new_recipient']:
        signals.append({'code': 'night_new_recipient', 'severity': 'MEDIUM',
                        'label': 'Late-night transaction to new recipient'})

    # ... 6 more signals

    # Catch-all: if model flagged it but no rule fired, report ML anomaly
    if not signals and fraud_prob >= 0.35:
        signals.append({'code': 'ml_anomaly', 'severity': 'MEDIUM',
                        'label': 'Behavioural pattern anomaly detected by ML model'})

    return signals
```

This is **explainable AI in practice** — the ML model makes the decision, but the signal extractor provides the human-readable justification for regulators and users.

---

### 3.12 Risk Bands and Recommendations

```
Score  │ Band      │ Recommendation │ User experience
───────┼───────────┼────────────────┼─────────────────────────────────────────────────
0–34   │ LOW       │ ALLOW          │ Transaction proceeds silently
35–59  │ MEDIUM    │ ALLOW          │ Proceeds; yellow badge in transaction history
60–79  │ HIGH      │ REVIEW         │ Proceeds; orange badge + flagged in fraud log
≥ 80   │ CRITICAL  │ BLOCK          │ Transaction rejected; user sees 🚫 error with link
```

**Why REVIEW exists (not just ALLOW/BLOCK):**

The 60–79 range represents transactions where the model has significant confidence of fraud but not certainty. Blocking legitimate transactions in this range would cause too many false positives (frustrated users). By allowing but flagging, we:
- Don't harm legitimate users
- Create an audit trail for investigation
- Feed these cases back into model retraining

---

### 3.13 Audit Trail and Compliance

Every fraud assessment is written to `fraud_events.json`:

```json
{
  "id":             "fe_1775296292242",
  "fromUserId":     "user_90ba1a56ad5dd250",
  "fromName":       "Yash Vardhan Sharma",
  "fromIdentifier": "yash.rpsp@gmail.com",
  "toIdentifier":   "test@example.com",
  "amount":         500.0,
  "fraudScore":     0,
  "riskLevel":      "LOW",
  "recommendation": "ALLOW",
  "signals":        [],
  "model":          "ml_random_forest",
  "createdAt":      "2026-04-04T09:51:32.242154600Z"
}
```

**Why this matters for compliance:**
- **RBI KYC/AML requirements** mandate that financial institutions maintain transaction monitoring records
- **Dispute resolution**: If a user disputes a blocked transaction, the audit log shows exactly what signals triggered it
- **Model accountability**: Regulators can audit which model version made each decision (`"model"` field)
- **Retraining data**: Human-reviewed events can be used as ground truth labels for the next model version

---

### 3.14 System Architecture (End-to-End)

```
User initiates Send Money
        │
        ▼
React SendMoney.tsx
        │  POST /api/v1/transactions/send
        │  Body: { recipientIdentifier, amount, note }
        │  Authorization: Bearer <token>
        ▼
Spring Boot TransactionController.java
        │  transactionService.send(authHeader, body)
        ▼
TransactionService.java — Validation
        │  ├─ Authenticate sender
        │  ├─ Find recipient (or throw 400)
        │  ├─ Check sender has sufficient balance
        │  └─ Build sender + recipient StoredUser objects
        │
        │  ⬇ FRAUD CHECK (before any money moves)
        ▼
FraudService.java
        │  ├─ Layer 1: isBlacklisted(recipient.identifier)
        │  ├─ Layer 2: checkHardVelocity(features, recentSent, amount)
        │  ├─ buildFeatures(sender, recipient, amount, recentSent)
        │  └─ callMlService(features)
        │         │
        │         │  POST http://localhost:5002/assess   (3s timeout)
        │         ▼
        │  Python Flask app.py
        │         │  RandomForest.predict_proba() → 0.94
        │         │  extract_signals() → ["high_balance_drain", "is_night"]
        │         │  Response: { fraudScore: 94, riskLevel: "CRITICAL",
        │         │              recommendation: "BLOCK", signals: [...] }
        │         ▼
        │  (timeout) → ruleFallback(features)
        │
        │  FraudAssessment { fraudScore, riskLevel, recommendation, signals }
        ▼
TransactionService.java — Post-fraud decision
        │
        ├─ If BLOCK:
        │    throw BadRequestException("Transaction blocked: " + signals[0].label)
        │    → HTTP 400 → React shows 🚫 banner
        │
        └─ If ALLOW/REVIEW:
             Deduct from sender balance
             Credit recipient balance
             Write transaction with fraud metadata:
               { fraudScore, fraudRiskLevel, fraudSignals, fraudRecommendation }
             logEvent(sender, recipient, amount, assessment) → fraud_events.json
        ▼
React renders:
        ├─ Success: transaction details + optional risk badge
        └─ Blocked: 🚫 banner with signal description + link to /fraud
```

---

### 3.15 Model Performance

```
Metric                      │ Value  │ Notes
────────────────────────────┼────────┼──────────────────────────────────────────────
AUC-ROC                     │ 1.0000 │ Perfect separation on synthetic data
AUC-PR (avg precision)      │ 1.0000 │ Precision-recall curve (robust for imbalance)
Precision (fraud class)     │ 0.99   │ 99% of "fraud" predictions are correct
Recall (fraud class)        │ 1.00   │ 100% of actual fraud cases are caught
F1-score (fraud class)      │ 0.99   │ Harmonic mean of precision and recall
Overall accuracy            │ 1.00   │ 99.98% on 4,000-sample test set
5-fold CV AUC-ROC           │ 1.0000 │ ± 0.0000 — very stable across folds
Training samples            │ 16,000 │ 80% of 20,000
Test samples                │ 4,000  │ 20% of 20,000 (stratified split)
```

**Note on perfect scores:**
AUC-ROC=1.0 on synthetic data is expected because the fraud archetypes are designed with very distinct feature signatures. Real-world fraud data will have more overlap with legitimate transactions (fraud evolves to mimic legitimate behaviour), so real-world performance will be lower — typically AUC-ROC 0.92–0.97 for a well-tuned fraud model.

---

## 4. Comparison: Credit Score vs Fraud Score

```
Dimension              │ Credit Score Engine      │ Fraud Risk Scorer
───────────────────────┼──────────────────────────┼────────────────────────────────────────
ML task                │ Regression               │ Classification
Model                  │ GradientBoostingRegressor│ RandomForestClassifier
Output                 │ Continuous (300–900)     │ Probability → score (0–100)
Label generation       │ Deterministic formula    │ 5 archetypal fraud patterns
Training data          │ 60,000 SDV-enhanced profiles │ 20,000 transactions
Class balance          │ 3 buckets (30/40/30)     │ 88% legit / 12% fraud
Key metric             │ MAE, R²                  │ AUC-ROC, AUC-PR, Recall
Time window            │ All-time history          │ Last 1h / 24h / 30 days
Decision type          │ Informational             │ Operational (blocks money)
When computed          │ On user request           │ Before every transaction
Fallback               │ Java formula              │ Java rules + velocity checks
Python port            │ 5001                      │ 5002
Industry parallel      │ CIBIL, Experian, Equifax │ Stripe Radar, PayPal, Razorpay
```

---

## 5. Key Interview Q&A

**Q: Why did you use GBR for credit and RF for fraud?**
> Credit scoring is a **regression problem** — we need a continuous score 300–900. GBR achieves lower MSE than RF for regression. Fraud is a **classification problem** with severe class imbalance — RF with `class_weight='balanced'` handles this natively and outputs well-calibrated probabilities. The choice is driven by the problem formulation, not preference.

**Q: How do you handle the cold start problem for new users?**
> New users have `total_transactions=0`, `days_since_last_txn=account_age_days`, `avg_transaction_amount=0`. Both models were trained on profiles that include new users (account_age_days = 1–30 in the low-score bucket for credit; newly created accounts in multiple fraud archetypes). The models produce conservative outputs for new users — low credit scores and higher fraud risk for large amounts — which is the correct real-world behaviour.

**Q: What's your false positive rate and why does it matter more than false negatives in some contexts?**
> False positives (blocking legitimate transactions) are operationally more dangerous than false negatives in UX terms — a user whose legitimate Rs. 5,000 payment is blocked will likely abandon the app. That's why we use the REVIEW band (60–79): allow the transaction but flag it, rather than blocking it. We only block at score ≥ 80, where confidence is very high.

**Q: How would you retrain these models as real data accumulates?**
> For the credit model: replace the synthetic profile generation with real `users.json` and `transactions.json` data, apply the same label formula, retrain the pipeline, save `model.pkl`, restart the Flask service. For the fraud model: supplement synthetic data with real fraud cases (confirmed via disputes or manual review), retrain with the same architecture. No code changes to Spring Boot or React — they consume the API, not the model directly.

**Q: What if the Python service goes down in production?**
> Both services have Java fallbacks that run the same logic deterministically without any Python dependency. The `model` field in the response indicates whether ML or rules were used (`"ml_gradient_boosting"` vs `"rule_based_fallback"`). The fraud service also has velocity rules as Layer 2, which catch the most critical cases even without ML.

**Q: How is this different from what real companies like Stripe do?**
> Stripe Radar uses: (1) global network effects — it sees 100s of billions of transactions across all merchants, (2) device fingerprinting, (3) graph neural networks to detect fraud rings, (4) a rules engine that merchants can customise. We've replicated the core ML architecture (RF classifier + velocity rules + blacklist + explainable signals) but without the global network effect — which is the most powerful signal in real production systems.

**Q: Why is the credit score range 300–900 specifically?**
> This is the CIBIL score range (300 minimum, 900 maximum), established to align with India's credit bureau standard. Scores below 300 or above 900 are theoretically impossible in the CIBIL system. By using the same scale, GoPay's credit score is directly interpretable by users familiar with their CIBIL score.

**Q: How do you ensure the ML model is explainable for regulators?**
> Two mechanisms: (1) Feature importances from the ensemble tell us which features drove predictions globally. (2) The signal extraction layer in `app.py` translates feature values into human-readable strings (`"Amount is 12.3× the sender's historical average"`) for each individual transaction. This satisfies RBI's requirement for explainability in automated credit/fraud decisions.

---

## 6. Real-World Benchmarks and Industry Parallels

### Credit Scoring

| Company | Model type | Features | Scale |
|---|---|---|---|
| CIBIL | Logistic Regression + scorecard | 200+ loan repayment variables | 1.5B+ records |
| Experian | GBM ensemble | 700+ variables | Global |
| Upstart | Deep Learning | 1,600+ variables including education, employment | US lending |
| GoPay (ours) | GradientBoostingRegressor | 8 payment behaviour variables | Platform data |

**Key difference:** Traditional bureaus use loan repayment history (did you pay your EMI on time?). GoPay uses payment behaviour (how actively do you transact, what's your balance pattern?). As fintech matures, payment behaviour is increasingly used as a credit proxy — Paytm, PhonePe, and Razorpay all have internal credit scoring using UPI data.

### Fraud Detection

| Company | Architecture | Key signals |
|---|---|---|
| Stripe Radar | RF + network graph | 4,000+ signals, cross-merchant patterns |
| PayPal | Adaptive AI (deep learning) | Device fingerprint, behavioural biometrics |
| Razorpay Shield | Rule engine + ML | Velocity, BIN analysis, device trust |
| NPCI (UPI) | Rule-based | Velocity limits, FIDO device binding |
| GoPay (ours) | RF + 5-layer cascade | 16 signals, velocity rules, blacklist |

**The key signal we're missing vs production systems:** Device fingerprinting and IP geolocation. In a real implementation, you would also include:
- `ip_country` vs `account_country` mismatch
- `device_id` seen for the first time
- `user_agent` anomalies
- `latitude/longitude` jump (impossible travel — account accessed from Delhi and Mumbai within 5 minutes)

These signals are the **strongest fraud indicators** in production but require additional infrastructure (device SDKs, IP geolocation DBs) beyond the scope of this project.

---

*Document generated for GoPay Payment App.*
*Models: GradientBoostingRegressor (credit, port 5001) + RandomForestClassifier (fraud, port 5002)*
*Stack: Python 3.12 · scikit-learn 1.5.2 · Flask 3.0.3 · Spring Boot 2.7.17 · React 18*
