"""
GoPay Fraud Engine — SDV-Enhanced Synthetic Data Generator
============================================================
Uses Synthetic Data Vault (SDV) to generate a large, statistically rich
fraud training dataset that preserves real correlations between all 20 features.

Why SDV for fraud detection?
  The fraud patterns (ATO, velocity, structuring, mule, blacklist, VPA spoof,
  IFSC fraud) have complex multi-feature signatures. SDV learns the JOINT
  distribution so that generated samples maintain realistic feature combinations
  (e.g., high txns_last_1h co-occurs with low account_age in velocity fraud).

New features added vs original 16:
  - vpa_risk_score     : 0-100 VPA spoofing risk (Levenshtein-based)
  - ifsc_is_valid      : 1 if IFSC structurally valid + known bank, else 0
  - amount_zscore      : (amount - mean_sent) / std_sent — deviation from norm
  - sender_has_vpa     : 1 if sender registered a UPI VPA, else 0

Usage:
  python generate_sdv_data.py                          # gaussian, 80k rows
  python generate_sdv_data.py --model ctgan --rows 100000
  python generate_sdv_data.py --model auto             # auto-selects best

Output:
  fraud_sdv_data.csv           — used automatically by train.py
  fraud_sdv_quality_report.txt — statistical quality assessment
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

np.random.seed(42)

try:
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer
    SDV_OK = True
except ImportError:
    SDV_OK = False
    print("[ERROR] SDV not installed. Run: pip install sdv")
    sys.exit(1)

# ── Feature columns (must match train.py and app.py exactly) ──────────────
FEATURES = [
    'amount', 'amount_to_balance_ratio', 'amount_to_avg_ratio',
    'txns_last_1h', 'txns_last_24h',
    'amount_sent_last_1h', 'amount_sent_last_24h',
    'unique_recipients_24h', 'is_new_recipient',
    'hour_of_day', 'is_night', 'is_weekend',
    'account_age_days', 'is_round_amount', 'is_blacklisted',
    'balance_after_ratio',
    # New in v2:
    'vpa_risk_score',
    'ifsc_is_valid',
    'amount_zscore',
    'sender_has_vpa',
]


# ─────────────────────────────────────────────────────────────────────────────
# Seed data generation (mirrors train.py patterns + new features)
# ─────────────────────────────────────────────────────────────────────────────
def gen_legitimate(n):
    records = []
    for _ in range(n):
        balance       = np.random.lognormal(9.5, 0.8)
        avg_txn       = np.random.lognormal(7.0, 0.7)
        mean_sent     = avg_txn * np.random.uniform(8, 25)
        std_sent      = mean_sent * np.random.uniform(0.3, 0.7)
        amount        = max(10, min(np.random.lognormal(7.2, 0.6), 99_000))

        txns_1h       = np.random.choice([0,0,0,1,1,2], p=[0.4,0.2,0.2,0.1,0.07,0.03])
        txns_24h      = min(int(np.random.lognormal(1.0, 0.8)), 15)
        sent_1h       = txns_1h * avg_txn * np.random.uniform(0.5, 1.2)
        sent_24h      = txns_24h * avg_txn * np.random.uniform(0.6, 1.0)
        recipients    = max(1, int(txns_24h * np.random.uniform(0.4, 0.9)))
        is_new_recip  = int(np.random.random() < 0.25)
        hour_p        = [0.01,0.01,0.01,0.01,0.01,0.02,0.04,0.06,0.08,0.08,
                         0.07,0.07,0.07,0.07,0.06,0.06,0.06,0.06,0.05,0.05,
                         0.04,0.03,0.02,0.01]
        hour_p        = [p/sum(hour_p) for p in hour_p]
        hour          = int(np.random.choice(range(24), p=hour_p))
        acct_age      = int(np.random.lognormal(4.5, 1.0))
        is_round      = int(amount % 1000 == 0 and np.random.random() < 0.1)
        bal_after     = max(0, balance - amount) / max(balance, 1)
        zscore        = (amount - mean_sent) / max(std_sent, 1)

        # New features: legit users have known VPAs and valid IFSCs
        vpa_risk      = int(np.random.choice([0,0,0,0,5,10], p=[0.5,0.2,0.1,0.1,0.05,0.05]))
        ifsc_valid    = int(np.random.random() < 0.95)
        has_vpa       = int(np.random.random() < 0.70)

        records.append({
            'amount': amount, 'amount_to_balance_ratio': amount/max(balance,1),
            'amount_to_avg_ratio': amount/max(avg_txn,1),
            'txns_last_1h': txns_1h, 'txns_last_24h': txns_24h,
            'amount_sent_last_1h': sent_1h, 'amount_sent_last_24h': sent_24h,
            'unique_recipients_24h': recipients, 'is_new_recipient': is_new_recip,
            'hour_of_day': hour, 'is_night': int(hour<6), 'is_weekend': int(np.random.random()<0.29),
            'account_age_days': acct_age, 'is_round_amount': is_round,
            'is_blacklisted': 0, 'balance_after_ratio': bal_after,
            'vpa_risk_score': vpa_risk, 'ifsc_is_valid': ifsc_valid,
            'amount_zscore': zscore, 'sender_has_vpa': has_vpa,
            'label': 0,
        })
    return records


def gen_fraud(n_fraud):
    records = []
    n_per   = n_fraud // 7   # 7 patterns now (was 5)

    # Pattern 1: Account Takeover (ATO)
    for _ in range(n_per):
        balance  = np.random.lognormal(9.5, 0.8)
        amount   = balance * np.random.uniform(0.6, 0.95)
        avg_txn  = np.random.lognormal(6.5, 0.5)
        mean_sent = avg_txn * 5
        hour     = np.random.choice([0,1,2,3,4,22,23])
        records.append({
            'amount': amount, 'amount_to_balance_ratio': amount/max(balance,1),
            'amount_to_avg_ratio': amount/max(avg_txn,1),
            'txns_last_1h': np.random.randint(0,2), 'txns_last_24h': np.random.randint(1,4),
            'amount_sent_last_1h': amount, 'amount_sent_last_24h': amount,
            'unique_recipients_24h': 1, 'is_new_recipient': 1,
            'hour_of_day': hour, 'is_night': int(hour<6), 'is_weekend': int(np.random.random()<0.5),
            'account_age_days': np.random.randint(1,60), 'is_round_amount': int(amount%1000<50),
            'is_blacklisted': int(np.random.random()<0.2), 'balance_after_ratio': max(0,balance-amount)/max(balance,1),
            'vpa_risk_score': int(np.random.choice([0,0,20,40,60], p=[0.3,0.2,0.2,0.2,0.1])),
            'ifsc_is_valid': int(np.random.random()<0.5),
            'amount_zscore': (amount - mean_sent) / max(mean_sent*0.5, 1),
            'sender_has_vpa': int(np.random.random()<0.4),
            'label': 1,
        })

    # Pattern 2: Velocity Fraud / Card Testing
    for _ in range(n_per):
        balance  = np.random.lognormal(8.5, 0.7)
        amount   = np.random.uniform(1, 500)
        avg_txn  = np.random.lognormal(5.5, 0.5)
        txns_1h  = np.random.randint(5, 15)
        hour     = int(np.random.uniform(0,24))
        records.append({
            'amount': amount, 'amount_to_balance_ratio': amount/max(balance,1),
            'amount_to_avg_ratio': amount/max(avg_txn,1),
            'txns_last_1h': txns_1h, 'txns_last_24h': txns_1h+np.random.randint(0,5),
            'amount_sent_last_1h': amount*txns_1h, 'amount_sent_last_24h': amount*txns_1h*1.2,
            'unique_recipients_24h': np.random.randint(3,10), 'is_new_recipient': int(np.random.random()<0.7),
            'hour_of_day': hour, 'is_night': int(hour<6), 'is_weekend': int(np.random.random()<0.4),
            'account_age_days': np.random.randint(1,30), 'is_round_amount': 0,
            'is_blacklisted': 0, 'balance_after_ratio': max(0,balance-amount)/max(balance,1),
            'vpa_risk_score': int(np.random.choice([0,10,20], p=[0.6,0.2,0.2])),
            'ifsc_is_valid': int(np.random.random()<0.7),
            'amount_zscore': -2.0 + np.random.normal(0, 0.5),
            'sender_has_vpa': int(np.random.random()<0.5),
            'label': 1,
        })

    # Pattern 3: Structuring / Smurfing
    for _ in range(n_per):
        balance   = np.random.lognormal(10.5, 0.5)
        threshold = np.random.choice([10_000, 20_000, 50_000])
        amount    = threshold - np.random.uniform(100, 999)
        avg_txn   = np.random.lognormal(7.0, 0.6)
        txns_24h  = np.random.randint(3, 8)
        hour      = int(np.random.uniform(9,18))
        records.append({
            'amount': amount, 'amount_to_balance_ratio': amount/max(balance,1),
            'amount_to_avg_ratio': amount/max(avg_txn,1),
            'txns_last_1h': np.random.randint(0,3), 'txns_last_24h': txns_24h,
            'amount_sent_last_1h': amount*np.random.uniform(0.5,2),
            'amount_sent_last_24h': amount*txns_24h*0.8,
            'unique_recipients_24h': np.random.randint(2,6), 'is_new_recipient': int(np.random.random()<0.5),
            'hour_of_day': hour, 'is_night': 0, 'is_weekend': int(np.random.random()<0.2),
            'account_age_days': np.random.randint(10,180), 'is_round_amount': 0,
            'is_blacklisted': 0, 'balance_after_ratio': max(0,balance-amount)/max(balance,1),
            'vpa_risk_score': int(np.random.choice([0,5,10], p=[0.7,0.2,0.1])),
            'ifsc_is_valid': int(np.random.random()<0.85),
            'amount_zscore': (amount - avg_txn*10) / max(avg_txn*3, 1),
            'sender_has_vpa': int(np.random.random()<0.6),
            'label': 1,
        })

    # Pattern 4: Money Mule
    for _ in range(n_per):
        balance = np.random.lognormal(11.0, 0.5)
        avg_txn = np.random.lognormal(6.0, 0.5)
        amount  = balance * np.random.uniform(0.7, 0.99)
        txns_1h = np.random.randint(1,3)
        hour    = int(np.random.uniform(8,20))
        records.append({
            'amount': amount, 'amount_to_balance_ratio': amount/max(balance,1),
            'amount_to_avg_ratio': amount/max(avg_txn,1),
            'txns_last_1h': txns_1h, 'txns_last_24h': txns_1h+np.random.randint(0,3),
            'amount_sent_last_1h': amount, 'amount_sent_last_24h': amount,
            'unique_recipients_24h': np.random.randint(1,3), 'is_new_recipient': int(np.random.random()<0.8),
            'hour_of_day': hour, 'is_night': int(hour<6), 'is_weekend': int(np.random.random()<0.4),
            'account_age_days': np.random.randint(1,45), 'is_round_amount': int(np.random.random()<0.4),
            'is_blacklisted': int(np.random.random()<0.1),
            'balance_after_ratio': max(0,balance-amount)/max(balance,1),
            'vpa_risk_score': int(np.random.choice([0,10,30], p=[0.5,0.3,0.2])),
            'ifsc_is_valid': int(np.random.random()<0.6),
            'amount_zscore': (amount - avg_txn*10) / max(avg_txn*3, 1),
            'sender_has_vpa': int(np.random.random()<0.3),
            'label': 1,
        })

    # Pattern 5: Blacklisted / Known Bad Actor
    for _ in range(n_per):
        balance = np.random.lognormal(8.5, 1.0)
        avg_txn = np.random.lognormal(6.5, 0.7)
        amount  = min(np.random.lognormal(8.0, 0.8), 99_000)
        hour    = int(np.random.uniform(0,24))
        records.append({
            'amount': amount, 'amount_to_balance_ratio': amount/max(balance,1),
            'amount_to_avg_ratio': amount/max(avg_txn,1),
            'txns_last_1h': np.random.randint(0,3), 'txns_last_24h': np.random.randint(1,8),
            'amount_sent_last_1h': amount*np.random.uniform(0.5,1.5),
            'amount_sent_last_24h': amount*np.random.uniform(1.0,2.5),
            'unique_recipients_24h': np.random.randint(1,5), 'is_new_recipient': int(np.random.random()<0.6),
            'hour_of_day': hour, 'is_night': int(hour<6), 'is_weekend': int(np.random.random()<0.4),
            'account_age_days': np.random.randint(1,365), 'is_round_amount': int(np.random.random()<0.3),
            'is_blacklisted': 1,
            'balance_after_ratio': max(0,balance-amount)/max(balance,1),
            'vpa_risk_score': int(np.random.choice([10,40,70,90], p=[0.2,0.3,0.3,0.2])),
            'ifsc_is_valid': int(np.random.random()<0.3),
            'amount_zscore': (amount - avg_txn*10) / max(avg_txn*3, 1),
            'sender_has_vpa': int(np.random.random()<0.5),
            'label': 1,
        })

    # Pattern 6: VPA Spoofing (new)
    for _ in range(n_per):
        balance = np.random.lognormal(9.0, 0.7)
        avg_txn = np.random.lognormal(7.0, 0.6)
        amount  = min(np.random.lognormal(8.5, 0.7), 99_000)
        hour    = int(np.random.uniform(0, 24))
        records.append({
            'amount': amount, 'amount_to_balance_ratio': amount/max(balance,1),
            'amount_to_avg_ratio': amount/max(avg_txn,1),
            'txns_last_1h': np.random.randint(0,2), 'txns_last_24h': np.random.randint(1,5),
            'amount_sent_last_1h': amount*np.random.uniform(0.5,1.2),
            'amount_sent_last_24h': amount*np.random.uniform(0.8,1.5),
            'unique_recipients_24h': np.random.randint(1,3),
            'is_new_recipient': 1,                  # VPA spoof always new recipient
            'hour_of_day': hour, 'is_night': int(hour<6), 'is_weekend': int(np.random.random()<0.4),
            'account_age_days': np.random.randint(1,90),
            'is_round_amount': int(np.random.random()<0.2),
            'is_blacklisted': int(np.random.random()<0.15),
            'balance_after_ratio': max(0,balance-amount)/max(balance,1),
            'vpa_risk_score': int(np.random.choice([60,70,80,90,100], p=[0.2,0.2,0.3,0.2,0.1])),
            'ifsc_is_valid': int(np.random.random()<0.4),
            'amount_zscore': (amount - avg_txn*8) / max(avg_txn*2, 1),
            'sender_has_vpa': int(np.random.random()<0.6),
            'label': 1,
        })

    # Pattern 7: Fake IFSC / Bank Fraud (new)
    remaining = n_fraud - 6 * n_per
    for _ in range(remaining):
        balance = np.random.lognormal(9.0, 0.8)
        avg_txn = np.random.lognormal(7.0, 0.6)
        amount  = min(np.random.lognormal(9.0, 0.6), 99_000)
        hour    = int(np.random.uniform(6, 22))
        records.append({
            'amount': amount, 'amount_to_balance_ratio': amount/max(balance,1),
            'amount_to_avg_ratio': amount/max(avg_txn,1),
            'txns_last_1h': np.random.randint(0,2), 'txns_last_24h': np.random.randint(1,4),
            'amount_sent_last_1h': amount*np.random.uniform(0.3,0.8),
            'amount_sent_last_24h': amount*np.random.uniform(0.5,1.5),
            'unique_recipients_24h': np.random.randint(1,3),
            'is_new_recipient': 1,
            'hour_of_day': hour, 'is_night': 0, 'is_weekend': int(np.random.random()<0.3),
            'account_age_days': np.random.randint(1,120),
            'is_round_amount': int(np.random.random()<0.3),
            'is_blacklisted': int(np.random.random()<0.1),
            'balance_after_ratio': max(0,balance-amount)/max(balance,1),
            'vpa_risk_score': int(np.random.choice([10,30,50], p=[0.3,0.4,0.3])),
            'ifsc_is_valid': 0,                     # invalid IFSC is the defining signal
            'amount_zscore': (amount - avg_txn*8) / max(avg_txn*2, 1),
            'sender_has_vpa': int(np.random.random()<0.5),
            'label': 1,
        })

    return records


def build_seed_dataframe(n_total, fraud_ratio=0.12):
    n_fraud   = int(n_total * fraud_ratio)
    n_legit   = n_total - n_fraud
    data      = gen_legitimate(n_legit) + gen_fraud(n_fraud)
    np.random.shuffle(data)
    df        = pd.DataFrame(data)
    print(f"  Seed: {len(df)} rows | Fraud: {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SDV helpers
# ─────────────────────────────────────────────────────────────────────────────
def build_metadata(df):
    meta = SingleTableMetadata()
    meta.detect_from_dataframe(df)
    meta.update_column('label', sdtype='categorical')
    binary_cols = ['is_new_recipient','is_night','is_weekend','is_round_amount',
                   'is_blacklisted','ifsc_is_valid','sender_has_vpa']
    for col in binary_cols:
        meta.update_column(col, sdtype='categorical')
    return meta


def postprocess(df):
    df = df.copy()
    df['amount']                 = df['amount'].clip(lower=1)
    df['amount_to_balance_ratio']= df['amount_to_balance_ratio'].clip(lower=0)
    df['amount_to_avg_ratio']    = df['amount_to_avg_ratio'].clip(lower=0)
    df['txns_last_1h']           = df['txns_last_1h'].clip(lower=0).round()
    df['txns_last_24h']          = df['txns_last_24h'].clip(lower=0).round()
    df['amount_sent_last_1h']    = df['amount_sent_last_1h'].clip(lower=0)
    df['amount_sent_last_24h']   = df['amount_sent_last_24h'].clip(lower=0)
    df['unique_recipients_24h']  = df['unique_recipients_24h'].clip(lower=1).round()
    df['hour_of_day']            = df['hour_of_day'].clip(0, 23).round()
    df['account_age_days']       = df['account_age_days'].clip(lower=1).round()
    df['vpa_risk_score']         = df['vpa_risk_score'].clip(0, 100).round()
    df['balance_after_ratio']    = df['balance_after_ratio'].clip(0, 1)
    for col in ['is_new_recipient','is_night','is_weekend','is_round_amount',
                'is_blacklisted','ifsc_is_valid','sender_has_vpa','label']:
        df[col] = df[col].round().clip(0, 1).astype(int)
    return df


def quality_report(seed_df, synth_df, path):
    lines = ["="*68, "SDV Fraud Data Quality Report", "="*68,
             f"\nSeed rows     : {len(seed_df):,}",
             f"Generated rows: {len(synth_df):,}"]

    fraud_seed  = seed_df['label'].mean() * 100
    fraud_synth = synth_df['label'].mean() * 100
    lines.append(f"\nFraud rate  seed={fraud_seed:.1f}%  gen={fraud_synth:.1f}%")

    lines.append("\n--- Feature means (seed vs generated) ---\n")
    drifts = []
    for col in FEATURES:
        if col not in seed_df.columns:
            continue
        s, g = seed_df[col].mean(), synth_df[col].mean()
        d = abs(s-g)/max(abs(s),1)
        drifts.append(d)
        flag = "OK" if d < 0.20 else "DRIFT"
        lines.append(f"  {col:<30} seed={s:>9.3f}  gen={g:>9.3f}  [{flag}]")

    quality = max(0.0, 1.0 - float(np.mean(drifts)))
    lines += ["\n--- Quality score ---\n", f"  Mean drift     : {float(np.mean(drifts)):.3f}",
              f"  Quality score  : {quality:.3f}  (target >= 0.65)"]

    report = "\n".join(lines)
    print(report)
    with open(path, 'w') as f:
        f.write(report)
    return quality


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',     default='auto', choices=['auto','gaussian','ctgan'])
    parser.add_argument('--seed-rows', type=int, default=5_000)
    parser.add_argument('--rows',      type=int, default=80_000)
    parser.add_argument('--output',    default='fraud_sdv_data.csv')
    args = parser.parse_args()

    print("="*60)
    print("GoPay Fraud Engine — SDV Data Generator")
    print("="*60)
    print(f"  Model     : {args.model}")
    print(f"  Seed rows : {args.seed_rows:,}")
    print(f"  Target    : {args.rows:,}")

    print(f"\n[1/4] Generating {args.seed_rows:,} seed transactions ...")
    seed_df = build_seed_dataframe(args.seed_rows)

    print("\n[2/4] Building SDV metadata ...")
    meta = build_metadata(seed_df)

    candidates = ['gaussian', 'ctgan'] if args.model == 'auto' else [args.model]
    best_score, best_df = -1.0, None

    for idx, mname in enumerate(candidates, 1):
        print(f"\n[3.{idx}] Fitting {mname} synthesizer ...")
        if mname == 'gaussian':
            synth = GaussianCopulaSynthesizer(meta, enforce_min_max_values=True)
        else:
            synth = CTGANSynthesizer(meta, epochs=300, verbose=True)

        synth.fit(seed_df)
        print(f"[4.{idx}] Generating {args.rows:,} rows ...")
        raw      = synth.sample(num_rows=args.rows)
        clean    = postprocess(raw).dropna()
        qpath    = f"fraud_sdv_quality_{mname}.txt"
        score    = quality_report(seed_df[FEATURES + ['label']], clean[FEATURES + ['label']], qpath)
        print(f"  {mname} quality score: {score:.3f}")

        if score > best_score:
            best_score = score
            best_df    = clean

        if args.model == 'auto' and score >= 0.65:
            print("  Quality threshold met; stopping.")
            break

    output_cols = FEATURES + ['label']
    final_df    = best_df[output_cols]
    final_df.to_csv(args.output, index=False)
    print(f"\nSaved {len(final_df):,} rows -> {args.output}")
    print(f"Quality: {best_score:.3f}")
    print("\nNext: run `python train.py` to train XGBoost on SDV-enhanced data.")


if __name__ == '__main__':
    main()
