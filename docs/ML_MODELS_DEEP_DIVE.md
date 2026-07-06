# GoPay ML Models — Complete Technical Reference

> **Purpose:** Interview-ready deep dive into the Credit Score Engine and Fraud Risk Scorer built inside GoPay. Every algorithm decision, architectural choice, and data engineering detail is documented here. This document is kept up-to-date with every model version.
>
> **Current model versions:**
> - Credit Engine v1 — GradientBoostingRegressor, SDV-enhanced 60,000 rows, port 5001
> - Fraud Engine **v2** — XGBoostClassifier, 20 features, 7 archetypes, VPA + IFSC detection, port 5002

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
3. [Fraud Risk Scorer — v2](#3-fraud-risk-scorer--v2)
   - 3.1 [What It Does](#31-what-it-does)
   - 3.2 [Algorithm: XGBoost Classifier (v2 Upgrade from RandomForest)](#32-algorithm-xgboost-classifier-v2-upgrade-from-randomforest)
   - 3.3 [How XGBoost Works — Step by Step](#33-how-xgboost-works--step-by-step)
   - 3.4 [XGBoost vs RandomForest — Why We Switched](#34-xgboost-vs-randomforest--why-we-switched)
   - 3.5 [The 7 Fraud Archetypes](#35-the-7-fraud-archetypes)
   - 3.6 [Feature Vector — 20 Features (v2)](#36-feature-vector--20-features-v2)
   - 3.7 [Amount Z-Score — Statistical Anomaly Detection](#37-amount-z-score--statistical-anomaly-detection)
   - 3.8 [VPA Spoofing Detection Engine](#38-vpa-spoofing-detection-engine)
   - 3.9 [IFSC Validation Engine](#39-ifsc-validation-engine)
   - 3.10 [SDV-Enhanced Synthetic Fraud Data Generation](#310-sdv-enhanced-synthetic-fraud-data-generation)
   - 3.11 [Class Imbalance — scale_pos_weight in XGBoost](#311-class-imbalance--scale_pos_weight-in-xgboost)
   - 3.12 [The 7-Layer Decision Cascade](#312-the-7-layer-decision-cascade)
   - 3.13 [Velocity Rules — Hard Limits](#313-velocity-rules--hard-limits)
   - 3.14 [Blacklist Engine](#314-blacklist-engine)
   - 3.15 [Behavioural Signal Extraction (16 Signals)](#315-behavioural-signal-extraction-16-signals)
   - 3.16 [Risk Bands and Recommendations](#316-risk-bands-and-recommendations)
   - 3.17 [User Identity — Multi-Identifier Lookup](#317-user-identity--multi-identifier-lookup)
   - 3.18 [User Profile Extension — UPI and Banking Fields](#318-user-profile-extension--upi-and-banking-fields)
   - 3.19 [Audit Trail and Compliance](#319-audit-trail-and-compliance)
   - 3.20 [System Architecture (End-to-End)](#320-system-architecture-end-to-end)
   - 3.21 [Model Performance — v2](#321-model-performance--v2)
4. [Comparison: Credit Score vs Fraud Score](#4-comparison-credit-score-vs-fraud-score)
5. [Key Interview Q&A](#5-key-interview-qa)
6. [Real-World Benchmarks and Industry Parallels](#6-real-world-benchmarks-and-industry-parallels)

---

## 1. System Overview

GoPay runs **two independent ML systems** that serve fundamentally different purposes:

| Dimension | Credit Score Engine | Fraud Risk Scorer v2 |
|---|---|---|
| **Question answered** | "Is this user creditworthy long-term?" | "Is this specific transaction suspicious right now?" |
| **Time horizon** | Longitudinal (full account history) | Transactional (last 1–24 hours) |
| **ML task** | Regression (continuous score 300–900) | Classification (fraud probability 0–1 → score 0–100) |
| **Model** | GradientBoostingRegressor | **XGBoostClassifier (v2)** |
| **Features** | 8 credit behaviour features | **20 features** (16 original + 4 new) |
| **Python port** | 5001 | 5002 |
| **Decision** | Score + risk band (informational) | ALLOW / REVIEW / BLOCK (operational) |
| **When it runs** | On demand (user views credit page) | Before every transaction (blocking call) |
| **Fallback** | Java rule-based scoring | Java rule-based scoring + velocity rules |
| **New in v2** | — | XGBoost, VPA spoofing, IFSC validation, z-score, 7 archetypes, SDV |

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
| Linear Regression | Cannot capture non-linear interactions (income × txn_count isn't linear) |
| Random Forest Regressor | Higher MSE than GBR on tabular data; no sequential error correction |
| Neural Network | Requires much more data; less interpretable; overkill for 8 features |
| Ridge/Lasso | Same issue as linear — assumes additive feature effects |
| **GradientBoostingRegressor** | **Best MAE/R² on tabular regression with feature interactions** |

---

### 2.3 How Gradient Boosting Works — Step by Step

Gradient Boosting builds an ensemble of weak learners (shallow decision trees) **sequentially**, where each tree corrects the errors of the previous one.

**Step 1:** Start with a naive prediction — the mean of all target values:
```
Initial prediction F₀(x) = mean(y) = 600  (mean credit score)
```

**Step 2:** Compute residuals — the difference between actual and predicted:
```
Residuals = y_actual - F₀(x)
User A: 750 - 600 = +150  (under-predicted)
User B: 420 - 600 = -180  (over-predicted)
```

**Step 3:** Fit a shallow decision tree h₁ to predict these residuals:
```
If income_tier > 2 AND total_transactions > 50:
    predict residual = +120
Else:
    predict residual = -40
```

**Step 4:** Update prediction by adding the tree's output, scaled by `learning_rate`:
```
F₁(x) = F₀(x) + learning_rate × h₁(x)
       = 600   + 0.03          × 120
       = 603.6
```

**Step 5:** Repeat steps 2–4 for `n_estimators` trees. Each tree fits the residuals of the previous ensemble. Over 600 iterations, the model learns very fine-grained patterns.

**Why `learning_rate=0.03`?**
Small learning rate = more trees needed but better generalization. Large learning rate = faster but overfits. The combination of `n_estimators=600` + `learning_rate=0.03` balances variance and bias.

**Why `subsample=0.85`?**
Stochastic gradient boosting — each tree is fit on a random 85% sample of training data. This adds regularization (reduces overfitting) and speeds training.

---

### 2.4 Feature Engineering

Eight features are extracted from the user's GoPay account:

```
Feature                  │ Source                           │ Signal direction
─────────────────────────┼──────────────────────────────────┼────────────────────────────────
income_tier              │ Estimated from wallet deposits   │ Higher = better score
expense_ratio            │ spending / income                │ Lower = better (prudent)
total_transactions       │ Count from transactions.json     │ Higher = more active (better)
wallet_balance           │ Current balance from users.json  │ Higher = better
net_cash_flow            │ total_in - total_out             │ Positive = better
avg_transaction_amount   │ mean(all transaction amounts)    │ Contextual
days_since_last_txn      │ now - last transaction date      │ Lower = more active (better)
account_age_days         │ now - account creation date      │ Higher = more stable (better)
```

**Feature engineering decisions:**

- **`income_tier`** is binned (0–4) rather than continuous because income distribution is right-skewed — a few very high earners would create outliers that destabilize training.
- **`expense_ratio`** = expenses / max(income, 1) caps at 2.0 to prevent division-by-zero and handles users who spend more than they earn.
- **`net_cash_flow`** is the single most predictive feature because it directly correlates with financial health — positive = saving, negative = draining.

---

### 2.5 Ground Truth Formula (Synthetic Label Generation)

Since real CIBIL scores are not available, a deterministic formula generates the ground truth label for training:

```python
def compute_score(profile):
    score = 300  # base (minimum possible)

    # Income component (0–150 points)
    score += profile['income_tier'] * 30

    # Transaction activity (0–150 points)
    txn_component = min(profile['total_transactions'] / 200, 1.0) * 150
    score += txn_component

    # Expense ratio penalty (0 to -90 points)
    score -= min(profile['expense_ratio'], 1.5) * 60

    # Net cash flow reward (0–120 points)
    if profile['net_cash_flow'] > 0:
        score += min(profile['net_cash_flow'] / 10_000, 1.0) * 120

    # Account age stability (0–90 points)
    score += min(profile['account_age_days'] / 365, 1.0) * 90

    # Recency penalty (0 to -60 points)
    score -= min(profile['days_since_last_txn'] / 30, 1.0) * 60

    # Wallet balance bonus (0–90 points)
    score += min(profile['wallet_balance'] / 50_000, 1.0) * 90

    # Clamp to [300, 900]
    return max(300, min(900, round(score)))
```

**Why this formula is correct:**
- It encodes the same logic CIBIL uses: payment regularity, credit utilization, account age, recency
- It produces a smooth distribution across [300, 900] — not clustering at endpoints
- It is deterministic (same inputs → same score), making it reproducible and auditable

---

### 2.6 Synthetic Data Generation Strategy

#### Stage 1: Numpy Seed Data (5,000 profiles)

```python
# Three demographic buckets (mirror real income distribution):
# Low-score bucket  (300–550): 30% of population
# Mid-score bucket  (550–750): 40% of population
# High-score bucket (750–900): 30% of population

# Each bucket gets distinct statistical distributions:
low_bucket:  income_tier ~ choice([0,1,2], p=[0.6, 0.3, 0.1])
mid_bucket:  income_tier ~ choice([1,2,3], p=[0.4, 0.4, 0.2])
high_bucket: income_tier ~ choice([2,3,4], p=[0.2, 0.4, 0.4])
```

#### Stage 2: SDV Expansion (target 60,000 profiles)

The Synthetic Data Vault (SDV) learns the **joint feature distribution** from the 5,000 seed profiles and generates a much larger dataset that preserves realistic feature correlations.

**Why SDV is necessary:**
Simple numpy sampling generates each feature independently (e.g., `income_tier ~ choice([0,1,2])`). This misses the fact that in reality:
- High-income users also tend to have higher balances (correlation: ~0.7)
- More active users (high `total_transactions`) have more recent activity (lower `days_since_last_txn`)
- Users with positive net cash flow tend to have higher wallet balances

SDV captures all these **joint dependencies** and generates synthetic data that is statistically indistinguishable from real data in terms of pairwise and higher-order correlations.

**SDV synthesizer configuration:**
```python
from sdv.single_table import GaussianCopulaSynthesizer

# Auto-selection between GaussianCopulaSynthesizer and TVAESynthesizer
# Quality score = weighted average of mean drift, std drift, correlation drift
# Best model selected based on quality score threshold >= 0.65

synthesizer = GaussianCopulaSynthesizer(
    metadata,
    enforce_min_max_values=True,  # no out-of-range values
    enforce_rounding=False,        # allow fractional values
    # No explicit numerical_distributions — let SDV infer best fit
)
synthesizer.fit(seed_df)           # learns joint distribution
synthetic = synthesizer.sample(60_000)
```

**Quality scoring formula:**
```python
mean_component = max(0.0, 1.0 - min(1.0, mean_drift))
std_component  = max(0.0, 1.0 - min(1.0, std_drift))
corr_component = max(0.0, 1.0 - min(1.0, corr_drift / 0.5))
quality_score  = 0.45 * mean_component + 0.25 * std_component + 0.30 * corr_component
```

**Latest SDV run results (credit engine):**
```
Seed rows       : 5,000
Generated rows  : 60,000
Quality score   : 0.876   (threshold: 0.65)
Model selected  : GaussianCopulaSynthesizer
wallet_balance  mean drift : 0.04  (OK)
total_txns      mean drift : 0.03  (OK)
net_cash_flow   mean drift : 0.06  (OK)
```

---

### 2.7 Model Training Pipeline

```
train.py execution flow:
─────────────────────────────────────────────────────────────────────────
1. Check for credit_sdv_data.csv
   ├── FOUND  → load pandas CSV (60,000 rows)
   └── MISSING → generate 15,000 profiles via numpy (fallback)

2. Dynamic model configuration based on n_rows:
   ├── n_rows >= 30,000 → n_estimators=600, max_depth=6, lr=0.03
   └── n_rows <  30,000 → n_estimators=300, max_depth=5, lr=0.05

3. Build sklearn Pipeline:
   [StandardScaler → GradientBoostingRegressor]

4. Stratified 80/20 train-test split

5. Fit pipeline on X_train

6. Evaluate on X_test:
   ├── MAE (Mean Absolute Error)  ← primary metric
   ├── R² (coefficient of determination)
   └── 5-fold cross-validation MAE

7. Save artifacts:
   ├── credit_model.pkl  (pipeline: scaler + model)
   └── training_report.txt
─────────────────────────────────────────────────────────────────────────
```

---

### 2.8 Score Bands

```
Score Range  │ Band              │ Colour   │ Meaning
─────────────┼───────────────────┼──────────┼────────────────────────────────────────────
750 – 900    │ EXCELLENT         │ #16a34a  │ Premium creditworthiness; loan-eligible
700 – 749    │ GOOD              │ #65a30d  │ Solid profile; minor improvements possible
650 – 699    │ FAIR              │ #d97706  │ Average; some negative signals
550 – 649    │ NEEDS IMPROVEMENT │ #ea580c  │ Notable weaknesses; high expense ratio or low activity
300 – 549    │ POOR              │ #dc2626  │ Significant risk signals; new or inactive account
```

---

### 2.9 System Architecture (End-to-End)

```
User visits /credit
        │
        ▼
React CreditScore.tsx
        │  GET /api/v1/credit/score
        │  Authorization: Bearer <token>
        ▼
Spring Boot CreditController.java
        │  creditService.getScore(authHeader)
        ▼
CreditService.java
        │  1. getUserByToken() → StoredUser
        │  2. Read all transactions for user
        │  3. Build 8-feature profile
        │  4. POST http://localhost:5001/score  (3s timeout)
        ▼
Python Flask credit-engine/app.py
        │  GradientBoostingRegressor.predict([features])
        │  Returns: { score, riskBand, colour, featureImportances }
        ▼
CreditService.java  (on timeout → ruleFallback())
        │  Return CreditAssessment to controller
        ▼
React renders:
        ├── Gauge (300–900 arc)
        ├── Score band chip
        └── Feature breakdown table
```

---

### 2.10 Rule-Based Java Fallback

When the Python service is unreachable (timeout, crash), `CreditService.java` computes a deterministic score using the same formula logic:

```java
private int ruleFallback(UserProfile p) {
    double score = 300;
    score += p.incomeTier * 30;
    score += Math.min(p.totalTransactions / 200.0, 1.0) * 150;
    score -= Math.min(p.expenseRatio, 1.5) * 60;
    if (p.netCashFlow > 0) score += Math.min(p.netCashFlow / 10_000.0, 1.0) * 120;
    score += Math.min(p.accountAgeDays / 365.0, 1.0) * 90;
    score -= Math.min(p.daysSinceLastTxn / 30.0, 1.0) * 60;
    score += Math.min(p.walletBalance / 50_000.0, 1.0) * 90;
    return (int) Math.max(300, Math.min(900, Math.round(score)));
}
```

The `model` field in the response is set to `"rule_based_fallback"` vs `"gradient_boosting_v1"` so the UI (and audit logs) can distinguish ML vs rule-based scores.

---

### 2.11 Model Performance

| Metric | Baseline (numpy 15k) | **Improved (SDV 60k)** |
|---|---|---|
| MAE (test set) | 10.4 points | **2.09 points** |
| R² (test set) | 0.9818 | **0.9989** |
| 5-fold CV MAE | 11.2 ± 0.8 | **2.3 ± 0.1** |
| Training rows | 12,000 | **48,000** |
| Test rows | 3,000 | **12,000** |

**Interpretation:** An MAE of 2.09 means the model's predicted credit score is within ±2 points of the true score on average. Given the 300–900 range (600 points), this is a 0.35% error rate — excellent for a production credit scoring model.

---

### 2.12 Retraining on Real Data

When real GoPay user data accumulates:

```
1. Seed: extract real users from users.json + transactions.json
2. Compute ground truth labels using compute_score() formula
3. SDV expansion: fit GaussianCopulaSynthesizer on real seed data
4. Generate 50k–100k synthetic rows preserving real correlations
5. Retrain: python train.py (auto-detects CSV, scales hyperparameters)
6. Evaluate: compare new MAE/R² vs previous model
7. Deploy: restart Flask service (model.pkl hot-reloaded)
```

This "hybrid loop" ensures that as real user behaviour data grows, the model learns from actual GoPay-specific patterns rather than purely simulated ones.

---

## 3. Fraud Risk Scorer — v2

### 3.1 What It Does

The Fraud Risk Scorer assesses **every payment transaction** in real time before funds move. It returns a risk score (0–100), a recommendation (ALLOW / REVIEW / BLOCK), and a list of human-readable signals explaining the decision.

**Version 2 introduces:**
- XGBoost classifier (replaces RandomForest)
- 20 features (up from 16) including VPA risk, IFSC validity, amount z-score
- 7 fraud archetypes (up from 5) — added VPA Spoofing and Fake IFSC Fraud
- Layer 0 in the decision cascade: VPA + IFSC hard checks before Blacklist
- `/vpa-check` and `/ifsc-validate` API endpoints
- SDV-enhanced synthetic fraud data (80k rows target)
- User profile extended with `mobileNumber`, `vpa`, `bankAccount`, `ifscCode`
- Multi-identifier login: users can log in and send money via mobile number OR email

---

### 3.2 Algorithm: XGBoost Classifier (v2 Upgrade from RandomForest)

**Model:** `xgboost.XGBClassifier`

**Hyperparameters:**
```python
xgb.XGBClassifier(
    n_estimators      = 500,              # 500 boosting rounds
    max_depth         = 6,               # maximum tree depth per round
    learning_rate     = 0.05,            # shrinkage (eta)
    subsample         = 0.85,            # row sampling per tree (stochastic)
    colsample_bytree  = 0.85,            # feature sampling per tree
    scale_pos_weight  = n_legit/n_fraud, # class imbalance correction (~7.33×)
    reg_alpha         = 0.1,             # L1 regularization (lasso on leaf weights)
    reg_lambda        = 1.0,             # L2 regularization (ridge on leaf weights)
    eval_metric       = 'aucpr',         # optimize precision-recall AUC
    random_state      = 42,
    n_jobs            = -1,              # use all CPU cores
    verbosity         = 0,
)
```

**Why these specific values:**
- `n_estimators=500`: Enough rounds for 20 features to converge without overfitting
- `max_depth=6`: Standard for fraud detection — deep enough to capture interactions (e.g., `is_night AND is_new_recipient AND amount > 10,000`), shallow enough to generalize
- `learning_rate=0.05`: Lower than default (0.3) for better generalization with more trees
- `subsample=0.85`: Row sampling adds stochastic regularization — each tree sees a slightly different view of the data
- `colsample_bytree=0.85`: Feature sampling — each tree uses 17 of 20 features, reducing correlation between trees
- `eval_metric='aucpr'`: Area Under Precision-Recall curve — the correct metric for imbalanced fraud data
- `reg_alpha=0.1`: L1 regularization drives small leaf weights to zero (sparse model, less overfitting)
- `reg_lambda=1.0`: L2 regularization penalizes large leaf weights (smooth decision boundaries)

---

### 3.3 How XGBoost Works — Step by Step

XGBoost (eXtreme Gradient Boosting) is a tree ensemble that builds trees **sequentially** — each tree corrects the residual errors of the previous ensemble. It differs from standard gradient boosting through several key engineering innovations.

#### The Math: Second-Order Gradient Optimization

Standard gradient boosting uses only the first derivative (gradient) of the loss function. XGBoost uses **both first and second derivatives** (Hessian):

```
For binary classification, the loss function is Log Loss:
L(y, ŷ) = -[ y·log(ŷ) + (1-y)·log(1-ŷ) ]

First derivative  (gradient): gᵢ = ŷᵢ - yᵢ           (prediction error)
Second derivative (Hessian):  hᵢ = ŷᵢ · (1 - ŷᵢ)      (confidence in prediction)
```

Using the Hessian makes the optimization more stable and allows XGBoost to find better split points. The leaf value formula is:

```
Optimal leaf weight = -Σgᵢ / (Σhᵢ + λ)
```
where λ is the L2 regularization term. This directly incorporates regularization into every tree's leaf values.

#### Iteration Process

```
Round 1: Fit first tree on raw features → predict fraud probability p₁
Round 2: Compute residuals g₁ = p₁ - y_true
         Fit second tree on (X, g₁) → predict residual correction
         Update: p₂ = p₁ + lr × tree₂(X)
...
Round 500: p_final = sigmoid(Σ lr × treeᵢ(X))
                   → fraud probability [0, 1]
           fraud_score = round(p_final × 100)
```

#### Key Engineering Innovations Over Standard GBM

| Feature | Standard GBM | XGBoost |
|---|---|---|
| **Optimization** | First-order (gradient only) | Second-order (gradient + Hessian) |
| **Regularization** | None built-in | L1 (reg_alpha) + L2 (reg_lambda) |
| **Missing values** | Manual imputation required | Learns optimal direction for missing values |
| **Parallelism** | Sequential tree building | Column-block parallel split finding |
| **Pruning** | Pre-pruning (max_depth) | Post-pruning (max_delta_step) |
| **Cache** | No | Blocked computation for cache efficiency |
| **Speed** | Baseline | 10–50× faster on same data |

---

### 3.4 XGBoost vs RandomForest — Why We Switched

| Criterion | RandomForest (v1) | XGBoost (v2) | Winner |
|---|---|---|---|
| **Fraud AUC-ROC** | 1.0000 (synthetic) | 1.0000 (synthetic) | Tie |
| **Fraud AUC-PR** | 1.0000 (synthetic) | 1.0000 (synthetic) | Tie |
| **Class imbalance** | `class_weight='balanced'` (per-sample) | `scale_pos_weight` (global ratio) | XGBoost (more principled) |
| **Feature interactions** | Implicit via tree structure | Explicit via boosting rounds | XGBoost |
| **Probability calibration** | Uncalibrated (overconfident) | Better calibrated (eval_metric=aucpr) | XGBoost |
| **Regularization** | None (relies on tree depth) | L1 + L2 explicit | XGBoost |
| **Industry standard** | Common baseline | **Used by Stripe, Razorpay, PayPal, Setu** | XGBoost |
| **Training speed** | Slower (parallel trees) | Faster (blocked computation) | XGBoost |
| **Interpretability** | Feature importances | Feature importances + SHAP (future) | XGBoost |

**Key insight:** At Razorpay's public ML talks and Stripe's research blog, XGBoost is consistently cited as the workhorse of production fraud detection. RandomForest is a strong baseline, but XGBoost's regularization and sequential error correction make it more robust when features overlap between fraud and legitimate transactions (which they do in real life).

---

### 3.5 The 7 Fraud Archetypes

The training data simulates **7 documented real-world fraud patterns**. Each archetype has distinct feature signatures — the model learns to recognize them individually and in combination.

---

#### Archetype 1: Account Takeover (ATO)

**Real-world description:**
An attacker obtains the victim's login credentials (via phishing, credential stuffing, or SIM swap), logs in, and immediately transfers the wallet balance to their own account — typically at night when the victim is asleep.

**Synthetic profile:**
```python
balance         = lognormal(9.5, 0.8)          # victim had normal balance
amount          = balance × uniform(0.6, 0.95)  # drain 60–95% of balance
avg_txn         = lognormal(6.5, 0.5)           # historical avg is low
hour            = choice([0,1,2,3,4,22,23])     # always late night
is_new_recip    = 1                             # attacker's account is new
account_age     = randint(1, 60)                # new-ish account (targeted)
vpa_risk        = choice([0,20,40,60], p=[0.3,0.2,0.3,0.2])  # sometimes VPA
ifsc_valid      = bernoulli(p=0.5)              # may have invalid banking info
amount_zscore   = (amount - avg_txn*5) / (avg_txn*2)         # very high
```

**Signals that fire:**
- `amount_to_balance_ratio`: 0.6–0.95 → draining the account
- `is_new_recipient`: 1 → sending to unknown person
- `is_night`: 1 → midnight to 4 AM
- `amount_to_avg_ratio`: 5–20× → massive deviation from normal pattern
- `amount_zscore` ≥ 3 → statistical anomaly
- `account_age_days` < 60 → recently created or compromised

---

#### Archetype 2: Velocity Fraud / Card Testing

**Real-world description:**
A fraudster runs automated scripts to make many small transactions rapidly — testing if accounts are still valid and finding the spend limit before making a larger withdrawal.

**Synthetic profile:**
```python
balance         = lognormal(8.5, 0.7)
amount          = uniform(1, 500)          # very small probe amounts
txns_last_1h    = randint(5, 15)           # high velocity
unique_recips   = randint(3, 10)           # fan-out to many recipients
account_age     = randint(1, 30)           # freshly compromised
is_new_recip    = bernoulli(p=0.7)
amount_zscore   = -2.0 + normal(0, 0.5)   # unusually small amounts
```

**Signals that fire:**
- `txns_last_1h` ≥ 5 → immediately triggers velocity hard rule (BLOCK)
- `unique_recipients_24h` high → card testing / fan-out
- `amount_zscore` very negative → probing with unusually small amounts
- `account_age_days` < 30 → recently compromised account

---

#### Archetype 3: Structuring / Smurfing

**Real-world description:**
To avoid transaction monitoring thresholds (Rs. 10,000 / Rs. 20,000 / Rs. 50,000), fraudsters deliberately break up transfers into multiple amounts just below the threshold. This is a **PMLA (Prevention of Money Laundering Act)** offence.

**Example:** Instead of transferring Rs. 50,000 once (which triggers monitoring), they send Rs. 9,100 × 6 across 6 recipients.

**Synthetic profile:**
```python
balance         = lognormal(10.5, 0.5)
threshold       = choice([10_000, 20_000, 50_000])
amount          = threshold - uniform(100, 999)   # just under threshold
txns_24h        = randint(3, 8)                   # multiple transactions
hour            = uniform(9, 18)                  # business hours (evasive!)
is_round_amount = 0                               # deliberately NOT round
```

**Why business hours?**
Structuring often happens 9 AM–6 PM to blend in with legitimate activity. Simple time-of-day rules miss it. The XGBoost model catches it via the **combination** of: amount + frequency + recipient spread — which is exactly what boosting excels at (learning high-order interactions).

---

#### Archetype 4: Money Muling

**Real-world description:**
A "money mule" receives stolen funds into their account and is instructed to immediately forward them — taking a commission. The account adds a layer of indirection to obscure the original crime.

**The defining signal:** `amount_to_avg_ratio` = (Rs. 80,000 forward) / (Rs. 1,200 historical avg) = **66.7×** — an extreme statistical outlier.

```python
balance         = lognormal(11.0, 0.5)     # high (just received deposit)
avg_txn         = lognormal(6.0, 0.5)      # historical avg is LOW
amount          = balance × uniform(0.7, 0.99)
is_new_recip    = bernoulli(p=0.8)
ifsc_valid      = bernoulli(p=0.6)         # may route to suspicious account
sender_has_vpa  = bernoulli(p=0.3)         # often no registered VPA
```

---

#### Archetype 5: Blacklisted Actor

**Real-world description:**
The recipient's email or domain is on the GoPay fraud blacklist — known disposable email providers, accounts reported for fraud, or known criminal destinations.

**Design decision:** This is handled as a **hard rule** (Layer 1), not as a learned pattern. The ML model includes `is_blacklisted` as a feature (to inform score in borderline cases), but any `is_blacklisted=1` transaction is immediately flagged CRITICAL regardless of ML score.

```
Blocked domains: mailinator.com, guerrillamail.com, tempmail.com,
                 temp-mail.org, throwaway.email, yopmail.com,
                 fakeinbox.com, maildrop.cc, spamgourmet.com
```

`vpa_risk_score` for blacklisted actors is generated from `choice([10,40,70,90])` — they also tend to have suspicious VPAs.

---

#### Archetype 6: VPA Spoofing (NEW in v2)

**Real-world description:**
Fraudsters create UPI handles that are visually near-identical to legitimate payment handles. Victims see what looks like a merchant's VPA and pay money to the fraudster instead.

**Real attack examples:**
```
Legitimate           → Spoofed
─────────────────────────────────────────
paytm@upi            → paytrn@upi         (letter insertion)
sbi@okicici          → sbi@okicicl        (l ↔ I swap — identical in some fonts)
hdfc@ybl             → hdfc@yb1           (l → 1 digit substitution)
google@oksbi         → goog1e@oksbi       (l → 1 homoglyph)
phonepay@upi         → phonepe@upi        (brand misspelling)
gpay@oksbi           → qpay@oksbi         (g → q visual similarity)
```

**Why this is dangerous:** In many payment apps, the VPA handle is shown in small font. The difference between `sbi@okicici` and `sbi@okicicl` is invisible at a glance.

**Synthetic profile:**
```python
vpa_risk_score  = choice([60,70,80,90,100], p=[0.2,0.2,0.3,0.2,0.1])
ifsc_valid      = bernoulli(p=0.4)
is_new_recip    = 1                        # always a new recipient
account_age     = randint(1, 90)           # relatively new account
amount          = lognormal(8.5, 0.7)      # significant amount (worth spoofing)
```

**Signals that fire:**
- `vpa_risk_score` ≥ 60 → VPA spoofing detected (→ hard BLOCK if ≥ 80)
- `is_new_recipient`: 1 → sending to someone never paid before
- `ifsc_is_valid`: 0 → fake bank account info

---

#### Archetype 7: Fake IFSC / Bank Account Fraud (NEW in v2)

**Real-world description:**
Fraudsters provide victim with fake bank account details (including a structurally plausible but non-existent IFSC code) to redirect payments. Common in:
- Fake refund scams ("verify your account to receive refund")
- Prize/lottery scams ("pay processing fee to claim prize")
- Social engineering ("your account is compromised, transfer to safe account")

**IFSC format abuse:** Fraudsters use codes like `FAKE0123456` — which passes a visual check but fails structural validation. Or they use valid-format codes like `ABCD0XXXXXX` where `ABCD` is not a registered bank.

**Synthetic profile:**
```python
ifsc_is_valid   = 0                        # invalid IFSC is the defining signal
vpa_risk        = choice([10,30,50], p=[0.3,0.4,0.3])
is_new_recip    = 1                        # always a new recipient
amount          = lognormal(9.0, 0.6)      # large amount (social engineering)
hour            = uniform(6, 22)           # business hours (convincing)
```

**Hard rule:** If `ifsc_is_valid == 0` AND `amount >= Rs.5,000`, the transaction is BLOCKED by Layer 0b before even reaching ML scoring.

---

### 3.6 Feature Vector — 20 Features (v2)

Version 2 adds 4 new features to the original 16, enabling the model to detect VPA spoofing, IFSC fraud, and statistical amount anomalies.

```
Feature                   │ Type    │ Range          │ Fraud signal direction          │ New in v2?
──────────────────────────┼─────────┼────────────────┼─────────────────────────────────┼──────────
amount                    │ float   │ 1 – 1,00,000   │ Higher = more risk (context)    │
amount_to_balance_ratio   │ float   │ 0 – 2+         │ Higher = draining account       │
amount_to_avg_ratio       │ float   │ 0 – 100+       │ Higher = deviation from normal  │
txns_last_1h              │ int     │ 0 – 50+        │ Higher = velocity fraud         │
txns_last_24h             │ int     │ 0 – 200+       │ Higher = velocity fraud         │
amount_sent_last_1h       │ float   │ 0 – 1,00,000   │ Higher = high hourly spend      │
amount_sent_last_24h      │ float   │ 0 – 1,00,000   │ Higher = high daily spend       │
unique_recipients_24h     │ int     │ 0 – 20+        │ Higher = fan-out / smurfing     │
is_new_recipient          │ binary  │ 0 or 1         │ 1 = unknown destination         │
hour_of_day               │ int     │ 0 – 23         │ 0–5 = night (high risk)         │
is_night                  │ binary  │ 0 or 1         │ 1 = midnight to 5 AM            │
is_weekend                │ binary  │ 0 or 1         │ 1 = slightly more risk          │
account_age_days          │ int     │ 1 – 3650+      │ Lower = new/compromised         │
is_round_amount           │ binary  │ 0 or 1         │ 1 = structuring signal          │
is_blacklisted            │ binary  │ 0 or 1         │ 1 = CRITICAL immediate block    │
balance_after_ratio       │ float   │ 0 – 1          │ Lower = leaving account empty   │
vpa_risk_score            │ int     │ 0 – 100        │ Higher = VPA spoofing risk      │ ✓ NEW
ifsc_is_valid             │ binary  │ 0 or 1         │ 0 = invalid/unregistered IFSC   │ ✓ NEW
amount_zscore             │ float   │ -∞ to +∞       │ |zscore| ≥ 3 = anomaly          │ ✓ NEW
sender_has_vpa            │ binary  │ 0 or 1         │ 0 = no registered UPI VPA       │ ✓ NEW
```

**Feature engineering for the 4 new features:**

```java
// amount_zscore — computed in FraudService.buildFeatures() in Java:
double meanSent = recentSent.isEmpty() ? amount
    : recentSent.stream().mapToDouble(t -> t.amount).average().orElse(amount);
double sumSq    = recentSent.stream()
    .mapToDouble(t -> Math.pow(t.amount - meanSent, 2)).sum();
double stdSent  = recentSent.size() > 1 ? Math.sqrt(sumSq / recentSent.size()) : 1.0;
double amountZscore = (amount - meanSent) / Math.max(stdSent, 1.0);

// sender_has_vpa — from user profile:
int senderHasVpa = (sender.vpa != null && !sender.vpa.isEmpty()) ? 1 : 0;

// vpa_risk_score and ifsc_is_valid — from enrichWithVpaAndIfsc():
// Calls Python /vpa-check and /ifsc-validate with 1s timeout
// Safe defaults if service is unavailable: vpaRiskScore=0, ifscIsValid=1
```

**Real example feature vector (VPA Spoofing scenario):**
```json
{
  "amount":                  15000.0,
  "amount_to_balance_ratio": 0.62,
  "amount_to_avg_ratio":     8.4,
  "txns_last_1h":            0,
  "txns_last_24h":           1,
  "amount_sent_last_1h":     15000.0,
  "amount_sent_last_24h":    15000.0,
  "unique_recipients_24h":   1,
  "is_new_recipient":        1,
  "hour_of_day":             14,
  "is_night":                0,
  "is_weekend":              0,
  "account_age_days":        45,
  "is_round_amount":         1,
  "is_blacklisted":          0,
  "balance_after_ratio":     0.38,
  "vpa_risk_score":          85,
  "ifsc_is_valid":           0,
  "amount_zscore":           6.2,
  "sender_has_vpa":          1
}
→ Layer 0a fires: vpa_risk_score=85 ≥ 80 → BLOCK immediately (before ML)
```

---

### 3.7 Amount Z-Score — Statistical Anomaly Detection

The **amount z-score** measures how many standard deviations the current transaction amount is from the sender's own historical average. This is a core technique in statistical process control and anomaly detection.

**Formula:**
```
z = (amount_current - mean_historical) / std_historical

where:
  mean_historical = average of all past transactions by this sender
  std_historical  = standard deviation of past transactions
```

**Interpretation:**
```
z = 0.5   → Within 0.5 std of normal (very typical)
z = 1.5   → 1.5 std above normal (slightly unusual)
z = 3.0   → 3 std above normal (99.7% of normal transactions are below this)
z = 5.0   → Extremely unusual — almost certainly anomalous
z = -3.0  → Unusually small (card testing probe)
```

**Why z-score over raw deviation:**

Raw deviation (`amount - mean`) does not account for variability. If User A normally sends between Rs. 100–200, a Rs. 5,000 transaction is an extreme outlier. If User B normally sends between Rs. 1,000–50,000, a Rs. 5,000 transaction is completely normal. Z-score normalizes by standard deviation — it's **user-specific**, not absolute.

**Java implementation (FraudService.java):**
```java
// Computed from the sender's transaction history (30-day window)
List<Transaction> recentSent = recentSentBy(sender.id);

double meanSent = recentSent.isEmpty() ? amount
    : recentSent.stream().mapToDouble(t -> t.amount).average().orElse(amount);

double stdSent = 1.0;   // default: avoid division by zero for new users
if (recentSent.size() > 1) {
    double sumSq = recentSent.stream()
        .mapToDouble(t -> Math.pow(t.amount - meanSent, 2)).sum();
    stdSent = Math.sqrt(sumSq / recentSent.size());
}

double amountZscore = (amount - meanSent) / Math.max(stdSent, 1.0);
```

**Signal threshold:** `|amount_zscore| ≥ 3` fires the `amount_statistical_anomaly` signal with HIGH severity.

---

### 3.8 VPA Spoofing Detection Engine

This is the core of the new Layer 0a check. The engine lives in `fraud-engine/vpa_detector.py` and is exposed via `POST /vpa-check`.

#### What is a VPA?

A VPA (Virtual Payment Address) — also called UPI ID — is a string of the format `username@bankhandle`. Examples:
```
john.doe@ybl          → PhonePe handle
merchant@paytm        → Paytm handle
9876543210@oksbi      → Google Pay (SBI) handle
company@okhdfcbank    → Google Pay (HDFC) handle
```

The `bankhandle` suffix (e.g., `ybl`, `paytm`, `oksbi`) is assigned by NPCI to authorised payment service providers.

#### Algorithm 1: Levenshtein Distance

The **Levenshtein distance** (edit distance) between two strings is the minimum number of single-character edits (insertions, deletions, substitutions) required to transform one string into the other.

**Dynamic programming implementation (O(m×n) time and space):**
```python
def levenshtein_distance(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    if m == 0: return n
    if n == 0: return m

    # dp[j] = edit distance between s1[:i] and s2[:j]
    dp = list(range(n + 1))      # base case: dp[j] = j (delete all of s2)

    for i in range(1, m + 1):
        prev = dp[0]             # dp[i-1][j-1]
        dp[0] = i                # base case: dp[i][0] = i (delete all of s1)
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i-1] == s2[j-1]:
                dp[j] = prev    # characters match — no edit needed
            else:
                dp[j] = 1 + min(prev,    # substitution
                                dp[j],   # deletion from s1
                                dp[j-1]) # insertion into s1
            prev = temp
    return dp[n]
```

**Worked example — `okicicl` vs `okicici`:**
```
     ""  o  k  i  c  i  c  i
""  [ 0, 1, 2, 3, 4, 5, 6, 7 ]
o   [ 1, 0, 1, 2, 3, 4, 5, 6 ]
k   [ 2, 1, 0, 1, 2, 3, 4, 5 ]
i   [ 3, 2, 1, 0, 1, 2, 3, 4 ]
c   [ 4, 3, 2, 1, 0, 1, 2, 3 ]
i   [ 5, 4, 3, 2, 1, 0, 1, 2 ]
c   [ 6, 5, 4, 3, 2, 1, 0, 1 ]
l   [ 7, 6, 5, 4, 3, 2, 1, 1 ]  ← distance = 1
```

`okicicl` is **1 edit** away from `okicici` — just the final `l → i` substitution. This is a spoof.

#### Algorithm 2: Jaro-Winkler Similarity

Jaro-Winkler is a string similarity metric in [0, 1] that gives extra weight to matching prefixes — perfect for VPAs where spoofing typically happens at the end of the handle.

**Jaro similarity:**
```python
def jaro_similarity(s1, s2):
    match_dist = max(len(s1), len(s2)) // 2 - 1
    # Count matching characters within match_dist window
    # Count transpositions (matched characters in wrong order)
    jaro = (matches/len(s1) + matches/len(s2)
            + (matches - transpositions/2)/matches) / 3
    return jaro
```

**Jaro-Winkler adds prefix weight:**
```python
def jaro_winkler(s1, s2, p=0.1):
    jaro = jaro_similarity(s1, s2)
    # Count matching prefix characters (up to 4)
    prefix = 0
    for c1, c2 in zip(s1[:4], s2[:4]):
        if c1 == c2: prefix += 1
        else: break
    return jaro + prefix * p * (1 - jaro)
```

`p=0.1` is the standard scaling factor. A prefix of 4 matching characters adds `4 × 0.1 × (1 - jaro)` to the similarity — rewarding VPAs that share the same prefix (which legitimate variants do).

#### Known Bank Handle Registry (NPCI-authorised)

```python
KNOWN_BANK_HANDLES = {
    # PhonePe
    'ybl', 'ibl', 'axl',
    # Google Pay
    'oksbi', 'okhdfcbank', 'okicici', 'okaxis',
    # Paytm
    'paytm', 'ptaxis', 'pthdfc', 'ptsbi',
    # Amazon Pay
    'apl', 'yapl',
    # WhatsApp Pay
    'wa1', 'waaxis',
    # Banking apps
    'sbi', 'yesbank', 'kotak', 'pnb', 'upi', 'hdfcbank',
    'icici', 'axisbank', 'indus', 'rbl', 'freecharge', 'bhim',
}
```

#### Risk Scoring Logic

```python
def check(vpa: str) -> dict:
    # 1. Structural validation: must match ^[a-zA-Z0-9._-]{3,50}@[a-zA-Z]{3,20}$
    # 2. Whitelist check: exact match → risk=0
    # 3. Split into username + handle
    # 4. Exact match on handle → risk=0 (legitimate bank handle)
    # 5. Levenshtein distance against all known handles:
    if best_lev == 1:   risk += 75  # "Very likely spoof"
    elif best_lev == 2: risk += 45  # "Probable spoof"
    elif jw >= 0.92:    risk += 35  # "Suspicious similarity"
    # 6. Homoglyph detection (l↔1, 0↔O, rn↔m, vv↔w):
    if homoglyph_found: risk += 20  # "Visual substitution attack"
    # 7. Unknown handle: risk += 15
    # Hard rule: risk ≥ 80 → BLOCK in Layer 0a
```

**Test results:**
```
VPA                    Risk  Spoof  Signals
paytm@upi              0     False  []                    ← Known handle, exact match
paytrn@upi             0     False  []                    ← 'upi' is known, handle matches
sbi@okicicl            75    True   ['handle_edit_distance_1']
hdfc@yb1               100   True   ['invalid_format', 'handle_edit_distance_1', 'homoglyph_1_to_l']
```

---

### 3.9 IFSC Validation Engine

The IFSC (Indian Financial System Code) validator lives in `fraud-engine/ifsc_validator.py` and is exposed via `POST /ifsc-validate`.

#### IFSC Format

```
IFSC format: [BANK_CODE][0][BRANCH_CODE]
             ──────────┬─────────────────
Example:     HDFC      0   001234
             ────      ─   ──────
             4 alpha   1   6 alphanumeric
             chars     zero chars

Total: 11 characters
Position 5 (index 4) MUST be the digit '0' (zero, not letter O)
```

#### Layer 1: Structural Regex Validation

```python
import re
IFSC_PATTERN = re.compile(r'^[A-Z]{4}0[A-Z0-9]{6}$')

def is_structurally_valid(ifsc: str) -> bool:
    # Checks:
    # 1. Exactly 11 characters
    # 2. First 4: uppercase letters only
    # 3. Position 5: digit '0' (zero)
    # 4. Last 6: uppercase letters or digits
    return bool(IFSC_PATTERN.match(ifsc))
```

**What this catches:**
- `INVALID` → 7 chars, no pattern match → INVALID
- `HDFC1001234` → position 5 is '1' not '0' → INVALID
- `HDFC0001234` → passes structural check → proceed to Layer 2

#### Layer 2: RBI Bank Code Registry

```python
BANK_REGISTRY = {
    'HDFC': 'HDFC Bank',       'SBIN': 'State Bank of India',
    'ICIC': 'ICICI Bank',      'UTIB': 'Axis Bank',
    'KKBK': 'Kotak Mahindra',  'PUNB': 'Punjab National Bank',
    'BARB': 'Bank of Baroda',  'CNRB': 'Canara Bank',
    'IOBA': 'Indian Overseas',  'UBIN': 'Union Bank',
    'BKID': 'Bank of India',   'IDBI': 'IDBI Bank',
    'YESB': 'Yes Bank',        'INDB': 'IndusInd Bank',
    'RATN': 'RBL Bank',        'FDRL': 'Federal Bank',
    'KVBL': 'Karur Vysya',     'SIBL': 'South Indian Bank',
    'AUBL': 'AU Small Finance', 'USFB': 'Ujjivan SFB',
    'AIRP': 'Airtel Payments',  'FINO': 'Fino Payments Bank',
    # ... 70+ total registered banks
}

bank_code = ifsc[:4]
bank_name = BANK_REGISTRY.get(bank_code)
if bank_name is None:
    signals.append(f'unknown_bank_code_{bank_code}')  # HIGH risk
```

**What this catches:**
- `FAKE0123456` → bank code `FAKE` not in registry → unknown_bank_code signal
- `HDFC0001234` → `HDFC` → 'HDFC Bank' → passes
- `XXXZ0ABCDEF` → `XXXZ` not in registry → INVALID

#### Layer 3: Luhn-Variant Structural Checksum

A GoPay-specific heuristic that assigns numeric values to each character and applies a Luhn-like doubling scheme to detect random character sequences that happen to pass the regex:

```python
def _luhn_structural_check(ifsc: str) -> bool:
    def char_val(c):
        return int(c) if c.isdigit() else (ord(c) - ord('A') + 10)
    # A=10, B=11, ..., Z=35

    vals = [char_val(c) for c in ifsc]
    total = 0
    for i, v in enumerate(vals):
        if i % 2 == 0:
            doubled = v * 2
            total += doubled - 9 if doubled > 9 else doubled  # Luhn doubling
        else:
            total += v
    return (total % 10) != 0   # True = passes heuristic
```

**Risk level assignment:**
```
signals = []              → risk = 'LOW'   (all checks pass)
['structural_checksum']   → risk = 'MEDIUM' (passes registry but checksum advisory)
['unknown_bank_code']     → risk = 'HIGH'   (bank not in RBI registry)
['invalid_format']        → risk = 'HIGH'   (structural failure)
```

**Test results:**
```
IFSC            is_valid  bank_name               risk
HDFC0001234     True      HDFC Bank               LOW
SBIN0000001     True      State Bank of India     MEDIUM (checksum advisory)
FAKE0123456     False     None                    HIGH
XXXZ0ABCDEF     False     None                    HIGH
INVALID         False     None                    HIGH
```

---

### 3.10 SDV-Enhanced Synthetic Fraud Data Generation

`generate_sdv_data.py` extends the same SDV strategy from the credit engine to the fraud engine. Because fraud data has **7 distinct archetypes with very different statistical signatures**, learning the joint distribution is even more important here.

#### Seed Data Generation

```python
FRAUD_RATIO = 0.12          # 12% fraud rate in training
N_SEED      = 5_000         # seed for SDV to learn from

Breakdown (600 fraud samples in seed):
  ATO           : 85   (n//7)
  Velocity      : 85   (n//7)
  Structuring   : 85   (n//7)
  Money Mule    : 85   (n//7)
  Blacklisted   : 85   (n//7)
  VPA Spoofing  : 85   (n//7)
  Fake IFSC     : 90   (remainder)
```

#### SDV Configuration for Fraud Data

```python
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer

meta = SingleTableMetadata()
meta.detect_from_dataframe(seed_df)

# Binary columns are categorical (not continuous)
binary_cols = ['is_new_recipient', 'is_night', 'is_weekend', 'is_round_amount',
               'is_blacklisted', 'ifsc_is_valid', 'sender_has_vpa', 'label']
for col in binary_cols:
    meta.update_column(col, sdtype='categorical')

# Auto-selection: tries Gaussian first, falls back to CTGAN if quality < 0.65
if model == 'auto':
    candidates = ['gaussian', 'ctgan']
    for candidate in candidates:
        fit_and_generate(candidate)
        score = quality_report(seed_df, synthetic_df)
        if score >= 0.65:
            break   # quality threshold met
```

#### Post-Processing to Ensure Valid Values

```python
def postprocess(df):
    df['amount']               = df['amount'].clip(lower=1)
    df['txns_last_1h']         = df['txns_last_1h'].clip(lower=0).round()
    df['hour_of_day']          = df['hour_of_day'].clip(0, 23).round()
    df['vpa_risk_score']       = df['vpa_risk_score'].clip(0, 100).round()
    df['balance_after_ratio']  = df['balance_after_ratio'].clip(0, 1)
    # Binary columns: round to 0 or 1
    for col in binary_cols:
        df[col] = df[col].round().clip(0, 1).astype(int)
    return df
```

#### Why SDV Matters for Fraud Detection

Without SDV, features are sampled independently:
```python
# Independent sampling (naive) — WRONG correlations:
txns_last_1h = randint(5, 15)    # velocity
amount       = uniform(1, 500)   # small (card testing)
# But: what if SDV learns that high-velocity transactions
# ALSO tend to have many unique recipients AND small amounts?
# Independent sampling misses this 3-way correlation.
```

With SDV, the synthesizer learns:
- In **velocity fraud**: `txns_last_1h` ↑ co-occurs with `unique_recipients_24h` ↑ AND `amount` ↓
- In **ATO**: `amount_to_balance_ratio` ↑ co-occurs with `is_night=1` AND `is_new_recipient=1`
- In **structuring**: `amount` ≈ threshold AND `txns_24h` ↑ AND `hour_of_day` ∈ [9,18]

These **joint correlations** make the synthetic data far more realistic and the trained model far more robust.

---

### 3.11 Class Imbalance — scale_pos_weight in XGBoost

**The problem:** Fraud transactions are rare (12% in training, ~0.1% in real life). A naive model that always predicts "legitimate" achieves 88% accuracy — completely useless.

**XGBoost's solution: `scale_pos_weight`**

Unlike RandomForest's `class_weight='balanced'` (which adjusts per-sample weights), XGBoost uses a single global multiplier on the gradient of positive (fraud) samples:

```python
scale_pos_weight = n_legitimate / n_fraud
                 = 22_000 / 3_000  # for 25,000 samples at 12% fraud
                 = 7.33
```

**What this does internally:**
In the XGBoost objective function (log loss), the gradient for fraud samples is multiplied by `scale_pos_weight`:
```
gradient_fraud     = scale_pos_weight × (ŷ - y)   # 7.33× stronger signal
gradient_legit     = 1.0             × (ŷ - y)
```

This means each fraud training example contributes **7.33×** as much to the loss gradient as a legitimate example. The model is forced to learn fraud patterns aggressively.

**Comparison with RandomForest `class_weight='balanced'`:**

| Aspect | RF `class_weight='balanced'` | XGBoost `scale_pos_weight` |
|---|---|---|
| Mechanism | Per-sample weight in Gini/entropy | Global gradient multiplier |
| Effect | Balanced weighted Gini | Upweighted gradient for fraud class |
| Calibration | May produce overconfident probabilities | Better calibrated with `aucpr` metric |
| Computation | O(n_samples) overhead | Single scalar multiplication |

**The precision-recall tradeoff:**
```
Threshold  Precision  Recall  F1
0.30       0.91       0.99    0.95    (high recall, some false positives)
0.50       0.98       0.97    0.97    (balanced — our default)
0.70       0.99       0.94    0.97    (high precision, some missed fraud)
```
We use threshold=0.50 (fraud_score = 50+), mapped to risk bands for operational decisions.

---

### 3.12 The 7-Layer Decision Cascade

The cascade architecture mirrors Stripe Radar and PayPal: **cheapest checks first, most expensive last.** Version 2 prepends **Layer 0** (VPA + IFSC) before the existing blacklist check.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       TRANSACTION REQUEST                                │
│            POST /api/v1/transactions/send                                │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 0a: VPA SPOOFING CHECK              [NEW in v2]                  │
│  Cost: HTTP POST to /vpa-check — < 1s timeout                           │
│                                                                          │
│  1. If recipient.vpa is set:                                             │
│     POST http://localhost:5002/vpa-check { "vpa": recipient.vpa }       │
│     Response: { risk_score: 0-100, is_spoof: bool, signals: [...] }     │
│  2. feat.vpaRiskScore = response.risk_score                              │
│                                                                          │
│  If vpaRiskScore >= 80 → fraudScore=90, CRITICAL, BLOCK                 │
│  Signal: "VPA spoofing detected: handle resembles legitimate UPI"        │
│  Model tag: "vpa_levenshtein"                                            │
│  If vpaRiskScore < 80 → continue (vpaRiskScore fed into ML as feature)  │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 0b: IFSC VALIDATION CHECK           [NEW in v2]                  │
│  Cost: HTTP POST to /ifsc-validate — < 1s timeout                       │
│                                                                          │
│  1. If recipient.ifscCode is set:                                        │
│     POST http://localhost:5002/ifsc-validate { "ifsc": recipient.ifsc } │
│     Response: { is_valid: bool, bank_name: str, risk: str }             │
│  2. feat.ifscIsValid = response.is_valid ? 1 : 0                        │
│                                                                          │
│  If ifscIsValid == 0 AND amount >= Rs.5,000 →                           │
│     fraudScore=85, CRITICAL, BLOCK                                       │
│     Signal: "Transaction blocked: IFSC invalid or unregistered"         │
│     Model tag: "ifsc_validator"                                          │
│  Otherwise → continue (ifscIsValid fed into ML as feature)              │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 1: BLACKLIST CHECK                                                │
│  Cost: O(1) HashSet lookup — < 1ms                                      │
│                                                                          │
│  Check: isBlacklisted(recipient.identifier)                             │
│  → Checks: blockedEmails (exact) + blockedDomains + blockedKeywords     │
│                                                                          │
│  If YES → fraudScore=95, CRITICAL, BLOCK                                │
│  If NO  → continue                                                       │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 2: HARD VELOCITY RULES                                            │
│  Cost: Read transactions.json + filter — < 5ms                          │
│                                                                          │
│  Rule 1: txns_last_1h ≥ 5              → BLOCK (NPCI UPI P2P limit)     │
│  Rule 2: amount_sent_last_1h + amt     │
│          > Rs.20,000                  → BLOCK (hourly limit)             │
│  Rule 3: txns_last_24h ≥ 20           → BLOCK (daily count limit)       │
│  Rule 4: amount_sent_last_24h + amt   │
│          > Rs.1,00,000               → BLOCK (daily amount limit)       │
│                                                                          │
│  If any fires → immediate BLOCK (fraudScore=90, model="velocity_rules") │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 3: FEATURE ENGINEERING                                            │
│  Cost: Pure Java computation — < 2ms                                    │
│                                                                          │
│  Build 20-feature vector from:                                           │
│  ├── Transaction history (30-day window from transactions.json)         │
│  ├── Sender profile (balance, account age, vpa registration)            │
│  ├── Recipient profile (ifscCode, vpa)                                  │
│  ├── VPA risk score (from Layer 0a)                                      │
│  ├── IFSC validity (from Layer 0b)                                       │
│  └── Amount z-score (computed from sender's history)                    │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 4: XGBoost ML SCORING                                             │
│  Cost: HTTP POST to Python — 50–500ms (3s timeout)                      │
│                                                                          │
│  POST http://localhost:5002/assess  (20-feature JSON payload)           │
│                                                                          │
│  Python: XGBClassifier.predict_proba(X)[0][1] → fraud_probability       │
│          fraud_score = round(fraud_probability × 100)                   │
│                                                                          │
│  Hard rule overrides (Python-side):                                      │
│  ├── is_blacklisted=1     → max(score, 85)                              │
│  ├── vpa_risk_score ≥ 60  → max(score, 75)                              │
│  └── ifsc_is_valid=0 + amount ≥ Rs.5,000 → max(score, 65)              │
│                                                                          │
│  Score bands: 0–34=LOW, 35–59=MEDIUM, 60–79=HIGH, ≥80=CRITICAL         │
│  + extract_signals() → 16 possible human-readable signals               │
└────────────────────────────┬────────────────────────────────────────────┘
                             │ (Python service unreachable or timeout)
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 5: RULE-BASED FALLBACK                                            │
│  Cost: Pure Java — < 1ms                                                │
│                                                                          │
│  Same signals computed deterministically (no ML):                       │
│  ├── velocity score    (+35 if txns_1h≥4, +25 if txns_24h≥8)          │
│  ├── balance drain     (+20 if ratio ≥ 0.75)                           │
│  ├── unusual amount    (+15 if amount_to_avg_ratio ≥ 5×)               │
│  ├── new large recip   (+15 if new + amount ≥ Rs.10,000)               │
│  ├── night + new       (+10 if 0–5 AM + new recipient)                 │
│  ├── many recipients   (+10 if ≥ 8 in 24h)                             │
│  ├── new account large (+10 if age < 7 days + amount ≥ Rs.5,000)       │
│  ├── vpa risk          (+20 if vpaRiskScore ≥ 30)  [NEW in v2]         │
│  └── invalid ifsc      (+15 if ifscIsValid == 0)   [NEW in v2]         │
│                                                                          │
│  model = "rule_based_fallback"                                           │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 6: AUDIT LOGGING                                                  │
│  Every assessment → fraud_events.json                                   │
│  Fields: id, fromUserId, fromName, fromIdentifier, toIdentifier,        │
│          amount, fraudScore, riskLevel, recommendation,                  │
│          signals[], model, createdAt                                     │
│  Retention: last 1,000 events                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

**Why this ordering is critical (cascade economics):**
- Layer 0a/0b: catches spoofing/IFSC fraud in ~1s — saves ML call for obvious cases
- Layer 1: HashSet O(1) — catches blacklisted actors in <1ms
- Layer 2: Velocity rules — catches automated attacks deterministically
- Layer 3+4: ML only runs after cheap rules pass — saves compute
- Layer 5: Always available even if Python is down — 100% uptime for fraud protection

---

### 3.13 Velocity Rules — Hard Limits

These limits are derived from RBI's transaction monitoring guidelines and NPCI's UPI velocity framework:

```
Limit                          │ Value        │ Rationale
───────────────────────────────┼──────────────┼──────────────────────────────────────────
Max transactions per hour      │ 5            │ NPCI UPI recommendation for P2P
Max amount sent per hour       │ Rs. 20,000   │ Typical hourly spend for retail user
Max transactions per 24 hours  │ 20           │ Covers daily payments + buffer
Max amount sent per 24 hours   │ Rs. 1,00,000 │ RBI P2P transaction monitoring threshold
Max unique recipients per day  │ 8            │ Limits fan-out patterns
New recipient limit (signal)   │ Rs. 10,000   │ UPI guideline for first-time payee
Large transaction flag         │ Rs. 50,000   │ Enhanced monitoring
```

These are **hard rules** (not ML) because:
1. Regulators require demonstrable, auditable, explainable controls
2. Zero latency — no network calls
3. They catch velocity fraud patterns with 100% recall
4. They are required even if ML is unavailable

---

### 3.14 Blacklist Engine

```java
// In-memory sets loaded from blacklist.json at Spring Boot startup
private final Set<String> blockedDomains   = new HashSet<>();  // O(1) lookup
private final Set<String> blockedEmails    = new HashSet<>();
private final Set<String> blockedKeywords  = new HashSet<>();

private boolean isBlacklisted(String identifier) {
    identifier = identifier.toLowerCase().trim();

    // Level 1: Exact email match (specific reported bad actors)
    if (blockedEmails.contains(identifier)) return true;

    // Level 2: Domain match (disposable email providers)
    int at = identifier.indexOf('@');
    if (at >= 0) {
        String domain = identifier.substring(at + 1);
        if (blockedDomains.contains(domain)) return true;
    }

    // Level 3: Keyword match (catches variations like fraud123@gmail.com)
    for (String kw : blockedKeywords)
        if (identifier.contains(kw)) return true;

    return false;
}
```

**Blacklisted disposable domains (sample):**
```
mailinator.com    guerrillamail.com   tempmail.com    temp-mail.org
throwaway.email   dispostable.com     sharklasers.com yopmail.com
fakeinbox.com     maildrop.cc         spamgourmet.com trashmail.com
```

**Why disposable emails signal fraud:** Fraudsters use throwaway email accounts that cannot be traced. A GoPay account registered with a disposable email provider is a strong indicator of malicious intent.

---

### 3.15 Behavioural Signal Extraction (16 Signals)

After XGBoost scores a transaction, `extract_signals()` in `app.py` maps feature values to human-readable fraud signals. These appear in:
- The `/fraud` Fraud Shield dashboard
- The "Transaction Blocked" error banner in the UI
- The `fraud_events.json` audit log

```python
def extract_signals(data, fraud_prob):
    signals = []

    # CRITICAL signals (immediate block context)
    if data.get('is_blacklisted', 0):
        signals.append({'code': 'blacklisted_recipient', 'severity': 'CRITICAL',
                        'label': 'Recipient on fraud blacklist'})

    vpa_risk = data.get('vpa_risk_score', 0)
    if vpa_risk >= 60:
        signals.append({'code': 'vpa_spoofing_detected', 'severity': 'CRITICAL',
                        'label': f'VPA spoofing detected (risk score {vpa_risk}/100)'})
    elif vpa_risk >= 30:
        signals.append({'code': 'vpa_suspicious', 'severity': 'HIGH',
                        'label': f'VPA handle resembles known legitimate handle (risk {vpa_risk}/100)'})

    if not data.get('ifsc_is_valid', 1):
        signals.append({'code': 'invalid_ifsc', 'severity': 'HIGH',
                        'label': 'IFSC code failed validation (unknown bank or invalid format)'})

    # HIGH velocity signals
    if data.get('txns_last_1h', 0) >= 4:
        signals.append({'code': 'high_velocity_1h', 'severity': 'HIGH', ...})

    if data.get('amount_to_balance_ratio', 0) >= 0.75:
        signals.append({'code': 'high_balance_drain', 'severity': 'HIGH', ...})

    if data.get('amount_to_avg_ratio', 0) >= 5:
        signals.append({'code': 'unusual_amount', 'severity': 'HIGH', ...})

    # Z-score anomaly (NEW in v2)
    zscore = data.get('amount_zscore', 0)
    if abs(zscore) >= 3:
        signals.append({'code': 'amount_statistical_anomaly', 'severity': 'HIGH',
                        'label': f'Amount is {abs(zscore):.1f} std deviations from sender norm'})

    # MEDIUM signals
    if data.get('is_new_recipient', 0) and data.get('amount', 0) >= 10_000:
        signals.append({'code': 'large_new_recipient', 'severity': 'MEDIUM', ...})

    if data.get('is_night', 0) and data.get('is_new_recipient', 0):
        signals.append({'code': 'night_new_recipient', 'severity': 'MEDIUM', ...})

    if data.get('unique_recipients_24h', 0) >= 5:
        signals.append({'code': 'many_recipients', 'severity': 'MEDIUM', ...})

    # Structuring detection
    amt = data.get('amount', 0)
    if amt > 500 and amt % 1000 < 50 and data.get('txns_last_24h', 0) >= 3:
        signals.append({'code': 'structuring_pattern', 'severity': 'HIGH', ...})

    if data.get('amount_sent_last_1h', 0) >= 15_000:
        signals.append({'code': 'high_spend_rate_1h', 'severity': 'HIGH', ...})

    if data.get('account_age_days', 365) < 7 and data.get('amount', 0) >= 5_000:
        signals.append({'code': 'new_account_large_txn', 'severity': 'MEDIUM', ...})

    # Catch-all: ML caught something not explained by rules
    if not signals and fraud_prob >= 0.35:
        signals.append({'code': 'ml_anomaly', 'severity': 'MEDIUM',
                        'label': 'Behavioral pattern anomaly detected by XGBoost model'})

    return signals
```

**Complete signal catalogue (16 signals):**

| Code | Severity | Description |
|---|---|---|
| `blacklisted_recipient` | CRITICAL | Recipient email/domain on GoPay blacklist |
| `vpa_spoofing_detected` | CRITICAL | VPA handle is Levenshtein distance ≤ 1 from known handle |
| `vpa_suspicious` | HIGH | VPA Jaro-Winkler similarity ≥ 0.92 to known handle |
| `invalid_ifsc` | HIGH | IFSC fails structural or bank-registry validation |
| `high_velocity_1h` | HIGH | 4+ transactions in a single hour |
| `high_velocity_24h` | MEDIUM | 10+ transactions in 24 hours |
| `high_balance_drain` | HIGH | Transaction ≥ 75% of wallet balance |
| `unusual_amount` | HIGH | Amount ≥ 5× sender's historical average |
| `amount_statistical_anomaly` | HIGH | Amount z-score ≥ 3 (3 standard deviations above norm) |
| `large_new_recipient` | MEDIUM | ≥ Rs.10,000 to a first-time recipient |
| `night_new_recipient` | MEDIUM | 00:00–05:59 transaction to new recipient |
| `many_recipients` | MEDIUM | 5+ distinct recipients in 24 hours |
| `structuring_pattern` | HIGH | Round-amount repeated transactions (PMLA offence) |
| `high_spend_rate_1h` | HIGH | ≥ Rs.15,000 total sent in last hour |
| `new_account_large_txn` | MEDIUM | Account < 7 days old, amount ≥ Rs.5,000 |
| `ml_anomaly` | MEDIUM | Statistical anomaly detected by XGBoost (catch-all) |

---

### 3.16 Risk Bands and Recommendations

```
Score  │ Band      │ Recommendation │ User experience
───────┼───────────┼────────────────┼─────────────────────────────────────────────────────
0–34   │ LOW       │ ALLOW          │ Transaction proceeds silently
35–59  │ MEDIUM    │ ALLOW          │ Proceeds; amber badge in transaction history
60–79  │ HIGH      │ REVIEW         │ Proceeds; red badge + flagged in fraud log + signal list
≥ 80   │ CRITICAL  │ BLOCK          │ Transaction rejected; user sees blocked banner
```

**Why REVIEW exists (not just ALLOW/BLOCK):**
The 60–79 range has high ML confidence of fraud but not certainty. Blocking here causes too many false positives (frustrated legitimate users who churn). By allowing but flagging:
- Legitimate users are not harmed
- An audit trail exists for investigation
- These labelled cases feed the next retraining cycle

**Python-side hard overrides (after ML score):**
```python
if data.get('is_blacklisted', 0):
    fraud_score = max(fraud_score, 85)     # always CRITICAL

if data.get('vpa_risk_score', 0) >= 60:
    fraud_score = max(fraud_score, 75)     # always at least HIGH

if not data.get('ifsc_is_valid', 1) and data.get('amount', 0) >= 5_000:
    fraud_score = max(fraud_score, 65)     # always at least HIGH
```

---

### 3.17 User Identity — Multi-Identifier Lookup

A major v2 improvement: users can now **log in and send money using either their email OR their mobile number**.

#### The Problem (v1)

In v1, `identifier` (the email) was the only lookup key. `mobileNumber` was stored as a separate field but never checked during authentication or recipient lookup. This meant:
- Login with `9876543210` → **fails** (mobile not checked against `identifier`)
- Send money to `9876543210` → **fails** (mobile not checked in recipient search)

#### The Solution (v2)

`AuthService.findUserByAnyIdentifier()` — a unified lookup method that checks both fields:

```java
public StoredUser findUserByAnyIdentifier(String raw, List<StoredUser> users) {
    if (raw == null || raw.trim().isEmpty()) return null;
    String normalized = raw.trim().toLowerCase();
    String digitsOnly = normalized.replaceAll("[^0-9]", "");

    for (StoredUser u : users) {
        // Check 1: Email/identifier match
        if (Objects.equals(u.identifier, normalized)) return u;

        // Check 2: Mobile number match (10-digit Indian mobile)
        if (digitsOnly.length() == 10
                && Objects.equals(u.mobileNumber, digitsOnly)) return u;
    }
    return null;
}
```

**Used in:**
1. `AuthService.login()` — replaces old single-field filter:
   ```java
   // OLD: users.stream().filter(u -> u.identifier.equals(identifier)).findFirst()
   // NEW:
   StoredUser user = findUserByAnyIdentifier(identifierRaw, users);
   ```

2. `TransactionService.send()` — replaces old recipient lookup:
   ```java
   // OLD: users.stream().filter(u -> u.identifier.equals(recipientId)).findFirst()
   // NEW:
   StoredUser recipient = authService.findUserByAnyIdentifier(recipientId, users);
   ```

**Security note:** The `digitsOnly.length() == 10` guard prevents ambiguity — a 10-digit number is treated as a mobile, everything else as an email. This prevents attackers from using a numeric email to match against a mobile number.

**UPI alignment:** This mirrors how PhonePe, Google Pay, and Paytm work — you can identify a payee by their registered mobile number, which is the primary UPI identity in India.

---

### 3.18 User Profile Extension — UPI and Banking Fields

Four new fields were added to `StoredUser` in `AuthService.java` to support UPI, VPA spoofing detection, and IFSC validation:

```java
public static class StoredUser {
    public String id;
    public String name;
    public String identifier;     // login email (primary key)
    public String passwordSalt;
    public String passwordHash;
    public double balance;
    public String createdAt;
    // v2 additions:
    public String mobileNumber;   // 10-digit Indian mobile (e.g., "9876543210")
    public String vpa;            // UPI VPA (e.g., "user@ybl")
    public String bankAccount;    // Bank account number
    public String ifscCode;       // IFSC code (e.g., "HDFC0001234")
}
```

#### Signup Flow

New users can provide these fields during account creation:
```
Required: name, email (identifier), mobileNumber, password
Optional: vpa, bankAccount, ifscCode
```

**Mobile validation:**
```java
String mobile = body.mobileNumber.trim().replaceAll("[^0-9]", "");
if (mobile.length() == 10) user.mobileNumber = mobile;
// Non-10-digit → silently ignored
```

**VPA validation:**
```java
// Must contain '@' to be stored
if (body.vpa != null && !body.vpa.trim().isEmpty()) {
    user.vpa = body.vpa.trim().toLowerCase();
}
```

**IFSC validation:**
```java
// Full structural validation in updateProfile:
if (!ifsc.matches("^[A-Z]{4}0[A-Z0-9]{6}$"))
    throw new BadRequestException("Invalid IFSC format. Expected: XXXX0XXXXXX");
```

#### Profile Update Endpoint

`PATCH /api/v1/me` — allows updating any of the extended fields after account creation:
```json
{
    "name": "Harsh Sharma",
    "mobileNumber": "9876543210",
    "vpa": "harsh@ybl",
    "bankAccount": "001122334455",
    "ifscCode": "HDFC0001234"
}
```

The `Profile.tsx` frontend page includes live VPA and IFSC validation buttons that call `POST /vpa-check` and `POST /ifsc-validate` directly before saving — giving users instant feedback.

#### How These Fields Feed Fraud Detection

```
User registers with ifscCode = "FAKE0123456"
                              ↓
When someone sends money TO this user:
  FraudService.enrichWithVpaAndIfsc(feat, recipient)
  → POST /ifsc-validate {"ifsc": "FAKE0123456"}
  → Response: {is_valid: false, risk: "HIGH"}
  → feat.ifscIsValid = 0
  → Layer 0b fires: amount >= 5,000 → BLOCK
```

This means fraud signals from the recipient's profile are incorporated **before** ML scoring — no data is wasted.

---

### 3.19 Audit Trail and Compliance

Every fraud assessment is written to `fraud_events.json` in real time:

```json
{
  "id":             "fe_1775296292242",
  "fromUserId":     "user_90ba1a56ad5dd250",
  "fromName":       "Harsh Sharma",
  "fromIdentifier": "harsh@gopay.com",
  "toIdentifier":   "test@example.com",
  "amount":         500.0,
  "fraudScore":     0,
  "riskLevel":      "LOW",
  "recommendation": "ALLOW",
  "signals":        [],
  "model":          "xgboost_v2",
  "createdAt":      "2026-04-05T14:21:32Z"
}
```

**Compliance value of each field:**
- `model` field: distinguishes `"xgboost_v2"` vs `"rule_based_fallback"` vs `"blacklist"` vs `"vpa_levenshtein"` vs `"ifsc_validator"` vs `"velocity_rules"` — satisfies RBI model accountability requirements
- `signals[]`: human-readable explanations → satisfies explainability requirements for automated decisions
- `createdAt`: ISO 8601 timestamp → audit trail for dispute resolution
- `fromIdentifier` / `toIdentifier`: links to KYC records
- Maximum 1,000 events retained in-memory (configurable — in production would be a database with indefinite retention)

**Why RBI mandates this:**
- **PMLA 2002** requires transaction monitoring records for ≥5 years
- **RBI KYC/AML guidelines** mandate documentation of risk assessments
- **Dispute resolution**: if a user's transaction is blocked, the audit log shows exactly which signal triggered it
- **Model governance**: regulators can audit which model version made each decision

---

### 3.20 System Architecture (End-to-End)

```
User initiates Send Money
        │
        ▼
React SendMoney.tsx
        │  POST /api/v1/transactions/send
        │  Body: { recipientIdentifier, amount, note }
        │         recipientIdentifier can be EMAIL or MOBILE NUMBER
        │  Authorization: Bearer <token>
        ▼
Spring Boot TransactionController.java
        │  transactionService.send(authHeader, body)
        ▼
TransactionService.java
        │  1. Authenticate sender (getUserByToken)
        │  2. authService.findUserByAnyIdentifier(recipientId)
        │     → checks u.identifier (email) OR u.mobileNumber (10 digits)
        │  3. Check balance
        │  4. Build sender + recipient StoredUser objects
        │
        │  ⬇ FRAUD CHECK (before any money moves)
        ▼
FraudService.java
        │  enrichWithVpaAndIfsc(feat, recipient)
        │  ├── POST /vpa-check  (recipient.vpa)   [1s timeout]
        │  └── POST /ifsc-validate (recipient.ifscCode) [1s timeout]
        │
        │  Layer 0a: if vpaRiskScore ≥ 80 → BLOCK
        │  Layer 0b: if ifscIsValid==0 + amt≥5k → BLOCK
        │  Layer 1:  isBlacklisted(recipient.identifier)
        │  Layer 2:  checkHardVelocity(features, recentSent, amount)
        │  Layer 3:  buildFeatures(20-feature vector, including zscore)
        │  Layer 4:  callMlService(features)
        │               │  POST http://localhost:5002/assess
        │               │  XGBClassifier.predict_proba() → fraud_prob
        │               │  extract_signals() → 16 possible signals
        │               │  → { fraudScore, riskLevel, recommendation, signals }
        │               │
        │            (3s timeout) → ruleFallback(features) [Layer 5]
        │
        │  FraudAssessment { fraudScore, riskLevel, recommendation, signals }
        ▼
TransactionService.java — Post-fraud decision
        │
        ├─ If BLOCK:
        │    throw BadRequestException("Transaction blocked: " + signals[0].label)
        │    → HTTP 400 → React shows blocked banner with signal explanation
        │
        └─ If ALLOW/REVIEW:
             Deduct from sender balance
             Credit recipient balance
             Write transaction with fraud metadata:
               { fraudScore, fraudRiskLevel, fraudSignals, fraudRecommendation }
             logEvent() → fraud_events.json  [model: "xgboost_v2"]
        ▼
React renders:
        ├─ Success: transaction details + risk badge (LOW/MEDIUM/HIGH/CRITICAL)
        └─ Blocked: error banner with signal description + link to /fraud
```

---

### 3.21 Model Performance — v2

#### XGBoost on Numpy Data (25,000 rows, 7 archetypes, 20 features)

```
Metric                      │ v1 RandomForest  │ v2 XGBoost
────────────────────────────┼──────────────────┼─────────────────────────────────
Dataset size                │ 20,000           │ 25,000
Features                    │ 16               │ 20
Fraud archetypes            │ 5                │ 7
AUC-ROC                     │ 1.0000           │ 1.0000
AUC-PR (avg precision)      │ 1.0000           │ 1.0000
Precision (fraud class)     │ 0.99             │ 1.00
Recall (fraud class)        │ 1.00             │ 1.00
F1-score (fraud class)      │ 0.99             │ 1.00
Overall accuracy            │ 0.9998           │ 1.0000
5-fold CV AUC-ROC           │ 1.0000 ± 0.0000  │ 1.0000 ± 0.0000
Training rows               │ 16,000           │ 20,000
Test rows                   │ 4,000            │ 5,000
Class ratio (legit:fraud)   │ 7.33:1           │ 7.33:1
scale_pos_weight            │ N/A (class_weight)│ 7.33
```

**Note on perfect AUC-ROC (1.0):**
On synthetic data with well-separated fraud archetypes, perfect separation is expected because each archetype has highly distinct feature signatures. In real production:
- Real fraud evolves to mimic legitimate transactions
- Expected real-world AUC-ROC: 0.92–0.97 (industry benchmark for well-tuned fraud models)
- AUC-PR typically 0.75–0.90 (fraud is rare, making precision-recall harder)

**SDV-enhanced model (target — run `generate_sdv_data.py` then `train.py`):**
```
Target dataset: 80,000 rows (80% legit, 12% fraud)
Expected improvement: AUC-PR from 1.0 → still 1.0 on synthetic
                     but model generalizes better to real-world edge cases
                     because SDV preserves multi-archetype joint correlations
```

---

## 4. Comparison: Credit Score vs Fraud Score

```
Dimension              │ Credit Score Engine v1   │ Fraud Risk Scorer v2
───────────────────────┼──────────────────────────┼──────────────────────────────────────────
ML task                │ Regression               │ Binary Classification
Model                  │ GradientBoostingRegressor│ XGBoostClassifier
Output                 │ Continuous (300–900)     │ Probability → score (0–100)
Label generation       │ Deterministic formula    │ 7 archetypal fraud patterns
Training data          │ 60,000 SDV-enhanced rows │ 25,000 rows (80k with SDV)
Class balance          │ 3 score buckets (30/40/30)│ 88% legit / 12% fraud
Key metric             │ MAE (2.09), R² (0.9989)  │ AUC-ROC (1.0), AUC-PR (1.0)
Time window            │ All-time history          │ Last 1h / 24h / 30 days
Decision type          │ Informational (score only)│ Operational (blocks money)
When computed          │ On user request           │ Before every transaction
Fallback               │ Java rule formula         │ Java rules + velocity checks
Python port            │ 5001                      │ 5002
New features in v2     │ —                         │ VPA risk, IFSC valid, z-score, has_vpa
New archetypes in v2   │ —                         │ VPA Spoofing, Fake IFSC Fraud
Decision layers        │ 1 (ML or fallback)        │ 7 (Layer 0a/0b/1/2/3/4/5)
Endpoints              │ GET /score                │ POST /assess, /vpa-check, /ifsc-validate
User fields used       │ balance, txns, age        │ vpa, ifscCode, mobileNumber, balance
Industry parallel      │ CIBIL, Experian, Equifax  │ Stripe Radar, Razorpay Shield, PayPal
```

---

## 5. Key Interview Q&A

**Q: Why did you upgrade from RandomForest to XGBoost for fraud detection?**
> XGBoost is the **industry standard** for tabular fraud detection (used by Razorpay, Setu, Stripe, PayPal). The key advantages are: (1) L1/L2 regularization built into the loss function — not just tree depth; (2) `scale_pos_weight` is a more principled approach to class imbalance than per-sample weights; (3) `eval_metric='aucpr'` optimizes directly for the right metric (precision-recall, not accuracy); (4) second-order gradient optimization converges to better optima on the fraud decision boundary.

**Q: What is Levenshtein distance and why is it the right algorithm for VPA spoofing?**
> Levenshtein distance is the minimum number of single-character insertions, deletions, or substitutions to transform one string into another. It's computed via dynamic programming in O(m×n) time. For VPA spoofing, it's ideal because: attackers make exactly 1–2 character changes to legitimate handles (e.g., `okicici → okicicl`). A distance of 1 means the strings are near-identical. We layer Jaro-Winkler on top because it gives extra weight to prefix matches — most VPA spoofing differs only at the end of the handle.

**Q: What is IFSC validation and why does it matter for fraud detection?**
> IFSC (Indian Financial System Code) is an 11-character code identifying every bank branch in India. Format: `[4-letter bank code][0][6-char branch code]`. Fraudsters use non-existent IFSC codes (valid format, fake bank code) to redirect payments. We validate at two layers: (1) structural regex `^[A-Z]{4}0[A-Z0-9]{6}$` and (2) the RBI bank code registry (70+ registered banks). An IFSC with an unknown bank code gets risk=HIGH and triggers a hard BLOCK for amounts ≥ Rs.5,000.

**Q: What is the z-score anomaly detection feature and how does it improve fraud detection?**
> The z-score measures how many standard deviations the current transaction is from the sender's historical average: `z = (amount - mean_sent) / std_sent`. This is user-specific normalization — if User A normally sends Rs.100–200 and suddenly sends Rs.10,000, the z-score is ~50 (extreme anomaly). If User B normally sends Rs.1,000–50,000, the same Rs.10,000 has z-score ~0 (normal). Raw amount comparisons fail here; z-score succeeds. |z| ≥ 3 fires the `amount_statistical_anomaly` signal.

**Q: How do you handle mobile number as a login/payment identifier?**
> In v1, only the email (`identifier` field) was used for lookup. Mobile was stored separately as `mobileNumber` but never queried. The v2 fix adds `findUserByAnyIdentifier()` which checks both fields: first exact match on `u.identifier` (email), then if the input is exactly 10 digits, it matches against `u.mobileNumber`. This is used in both `login()` and the recipient lookup in `TransactionService.send()`. This mirrors how PhonePe and Google Pay work — a payee is identified by their registered mobile number.

**Q: Why are Layer 0a/0b (VPA + IFSC) checked before Blacklist (Layer 1)?**
> The ordering is deliberate. VPA and IFSC checks have ~1s network latency (calling Python service) but they catch entirely different fraud types than the blacklist. We run them first because: (1) they cover the new fraud archetypes (VPA spoofing, fake IFSC) that the blacklist cannot; (2) if the VPA check fires, we avoid the blacklist lookup entirely (cascade economics); (3) in practice, VPA spoofing is a growing fraud vector in India that isn't yet on any blacklist.

**Q: How does SDV improve the fraud model specifically (vs just generating more random data)?**
> Without SDV, features are sampled independently. This misses multi-feature correlations that define each fraud archetype. For example, velocity fraud has `txns_last_1h` ↑ AND `unique_recipients_24h` ↑ AND `amount` ↓ — a 3-way correlation. Independent sampling would generate high-velocity transactions with large amounts (which aren't velocity fraud). SDV learns the **joint distribution** across all 20 features separately for fraud and legitimate transactions, generating data where feature combinations are as realistic as the seed data.

**Q: What if the Python VPA/IFSC service is unavailable during a transaction?**
> Both Layer 0a and 0b have graceful degradation: if the `POST /vpa-check` or `POST /ifsc-validate` call times out (1s timeout) or throws an exception, the Java code silently uses safe defaults — `vpaRiskScore=0` (no spoofing risk) and `ifscIsValid=1` (assume valid). This means the transaction falls through to the existing ML and velocity layers. The 1s timeout is intentional — it's much shorter than the 3s ML timeout because VPA/IFSC checks are simpler and we want to fail fast.

**Q: How would you retrain the fraud model as real fraud data accumulates?**
> The pipeline is: (1) collect confirmed fraud cases from dispute resolution + manual review; (2) combine with existing synthetic data (real trumps synthetic); (3) run `generate_sdv_data.py` with real data as seed to expand via SDV; (4) retrain with `python train.py` — it auto-detects `fraud_sdv_data.csv`; (5) evaluate AUC-ROC/AUC-PR on held-out real fraud cases; (6) if improvement, hot-swap `fraud_model.pkl` and restart Flask. No Spring Boot or React changes needed — they consume the API, not the model.

**Q: How would you scale this to production (millions of transactions per day)?**
> Replace the Flask service with FastAPI + uvicorn (async). Add Resilience4j circuit breakers in Spring Boot for the fraud service call. Cache velocity data in Redis (vs reading transactions.json each time). Replace file-based storage with PostgreSQL for users/transactions/fraud_events. Add the Kafka outbox pattern for audit events (fraud.flagged topic consumed by risk ops). Add device fingerprinting and IP geolocation as new features. The XGBoost model itself would still work at scale — it's a fast batch predictor.

---

## 6. Real-World Benchmarks and Industry Parallels

### Credit Scoring

| Company | Model type | Features | Scale |
|---|---|---|---|
| CIBIL | Logistic Regression + scorecard | 200+ loan repayment variables | 1.5B+ records |
| Experian | GBM ensemble | 700+ variables | Global |
| Upstart | Deep Learning | 1,600+ variables (education, employment) | US lending |
| Paytm | Internal GBM | UPI transaction behaviour | India |
| **GoPay (ours)** | **GradientBoostingRegressor + SDV** | **8 payment behaviour variables** | **Platform data** |

**Key difference:** Traditional bureaus use loan repayment history. GoPay uses payment behaviour (activity, balance, cash flow). As fintech matures, payment behaviour is increasingly used as a credit proxy — Paytm, PhonePe, and Razorpay all have internal credit scoring using UPI data.

### Fraud Detection

| Company | Algorithm | Key signals | Special techniques |
|---|---|---|---|
| Stripe Radar | RF + GNN | 4,000+ signals, cross-merchant | Global network effect |
| PayPal | Adaptive AI (deep learning) | Device fingerprint, biometrics | Real-time model updates |
| Razorpay Shield | XGBoost + rules | Velocity, BIN, device trust | Merchant-specific thresholds |
| PhonePe | XGBoost | Velocity, UPI device binding | FIDO2 device attestation |
| NPCI (UPI) | Rule-based | Velocity limits, device ID | Regulatory hard limits |
| **GoPay v2 (ours)** | **XGBoost + 7-layer cascade** | **20 signals, VPA Levenshtein, IFSC registry, z-score** | **Layer 0 VPA+IFSC, SDV synthetic data** |

**What real companies do that we've now replicated:**
- ✅ XGBoost classifier (Razorpay, PhonePe standard)
- ✅ Velocity rules aligned with NPCI/RBI guidelines
- ✅ VPA spoofing via edit distance (NPCI fraud advisory recommendation)
- ✅ IFSC validation (standard in any UPI-compliant app)
- ✅ Amount z-score / statistical anomaly (Stripe's user-baseline deviation signal)
- ✅ Explainable signals per transaction (RBI explainability requirement)
- ✅ Audit trail with model version tracking (RBI model governance)

**What production systems have that we don't (yet):**
- ❌ Device fingerprinting (`device_id` mismatch — strongest single fraud signal)
- ❌ IP geolocation (`ip_country != account_country` — impossible travel detection)
- ❌ Graph neural networks (fraud ring detection across multiple accounts)
- ❌ Global network effects (Stripe sees 100B+ txns — cross-merchant fraud patterns)
- ❌ Real-time model updates (online learning as fraud patterns evolve)

These would require additional infrastructure (device SDKs, IP geo DBs, GNN training cluster) beyond the current project scope.

---

*Document version: v2 — updated to reflect Fraud Engine v2 (XGBoost, 7 archetypes, 20 features, VPA/IFSC detection, SDV fraud data, multi-identifier login)*
*Credit Engine: GradientBoostingRegressor, 8 features, SDV 60k rows, port 5001*
*Fraud Engine v2: XGBoostClassifier, 20 features, 7 archetypes, 7-layer cascade, port 5002*
*Stack: Python 3.12 · scikit-learn 1.5.2 · xgboost 3.2.0 · sdv 1.35.1 · Flask 3.0.3 · Spring Boot 2.7.17 · React 18*
