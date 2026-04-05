"""
GoPay Credit Engine — SDV-Enhanced Synthetic Data Generator
=============================================================
Uses the Synthetic Data Vault (SDV) library to generate a large, statistically
rich training dataset that preserves REAL correlations between financial features.

Why SDV over plain numpy sampling?
───────────────────────────────────
Plain numpy (our original train.py) generates features INDEPENDENTLY per profile
type. This means a "high credit" profile might get high balance but zero
transactions — an unrealistic combination. SDV learns the JOINT distribution
of all features together and preserves natural correlations, e.g.:
  • High wallet_balance  ↔  more total_transactions
  • Older accounts       ↔  lower txn_frequency_per_week variance
  • More total_received  ↔  more avg_transaction_amount

SDV Synthesizers supported:
  --model gaussian   GaussianCopula  — fast, statistical, great for continuous data
  --model ctgan      CTGAN           — GAN-based, best quality, slower (~10 min)
  --model tvae       TVAE            — VAE-based, good balance of speed and quality

Usage:
  python generate_sdv_data.py                          # GaussianCopula, 50k rows
  python generate_sdv_data.py --model ctgan            # CTGAN, 50k rows
  python generate_sdv_data.py --model tvae --rows 80000
  python generate_sdv_data.py --seed-rows 5000 --rows 100000

Output:
  credit_sdv_data.csv      — generated dataset (loaded automatically by train.py)
  sdv_synthesizer.pkl      — saved synthesizer (re-use without re-fitting)
  sdv_quality_report.txt   — statistical similarity report
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib

np.random.seed(42)

# ── SDV imports (check version compatibility) ──────────────────────────────
try:
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import (
        GaussianCopulaSynthesizer,
        CTGANSynthesizer,
        TVAESynthesizer,
    )
    SDV_OK = True
except ImportError:
    SDV_OK = False
    print("[ERROR] SDV not installed. Run: pip install sdv")
    sys.exit(1)

# ── Feature columns (must match train.py and app.py) ─────────────────────
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

# ── Scoring formula (identical to train.py — single source of truth) ──────
def compute_score(row):
    """
    Deterministic credit scoring formula.
    Produces a label in [300, 900] for each generated feature vector.
    The ML model is trained to approximate this relationship.
    """
    balance  = max(row['wallet_balance'], 0)
    txns     = max(row['total_transactions'], 0)
    sent     = max(row['total_sent'], 0)
    received = max(row['total_received'], 0)
    age      = max(row['account_age_days'], 1)
    gap      = max(row['days_since_last_txn'], 0)

    b_score  = min(1.0, balance / 75_000)
    a_score  = min(1.0, txns / 200)

    total_flow = sent + received
    nf_score   = (received / total_flow) if total_flow > 0 else 0.5

    r_score    = max(0.0, 1.0 - gap / 60) if txns > 0 else 0.0
    age_score  = min(1.0, age / 1095)

    raw = (b_score * 0.30 + a_score * 0.25 + nf_score * 0.25
           + r_score * 0.10 + age_score * 0.10)

    return int(round(300 + raw * 600))


# ── Seed data generation (same distributions as train.py) ─────────────────
def generate_profile(profile_type):
    """Generate one synthetic user profile using parameterised distributions."""
    if profile_type == 'high':
        balance        = np.random.lognormal(10.0, 0.8)
        txns           = int(np.random.lognormal(4.0, 0.7))
        received       = np.random.lognormal(10.5, 0.7)
        sent_ratio     = np.random.uniform(0.2, 0.7)
        account_age    = int(np.random.uniform(180, 1200))
        days_gap       = int(np.random.uniform(0, 7))

    elif profile_type == 'medium':
        balance        = np.random.lognormal(8.5, 0.9)
        txns           = int(np.random.lognormal(2.5, 0.8))
        received       = np.random.lognormal(9.0, 0.8)
        sent_ratio     = np.random.uniform(0.4, 0.9)
        account_age    = int(np.random.uniform(30, 365))
        days_gap       = int(np.random.uniform(5, 30))

    else:  # low
        balance        = np.random.lognormal(6.5, 1.0)
        txns           = int(np.random.lognormal(1.0, 1.0))
        received       = np.random.lognormal(7.5, 1.0)
        sent_ratio     = np.random.uniform(0.7, 1.2)
        account_age    = int(np.random.uniform(1, 120))
        days_gap       = int(np.random.uniform(14, 90))

    txns       = max(0, txns)
    balance    = max(0.0, balance)
    received   = max(0.0, received)
    sent       = max(0.0, received * sent_ratio)
    avg_txn    = (sent + received) / max(1, txns * 2) if txns > 0 else 0.0
    weeks      = max(1, account_age / 7)
    freq       = txns / weeks
    gap_actual = days_gap if txns > 0 else account_age

    return {
        'wallet_balance':         balance,
        'total_transactions':     float(txns),
        'total_sent':             sent,
        'total_received':         received,
        'avg_transaction_amount': avg_txn,
        'account_age_days':       float(account_age),
        'days_since_last_txn':    float(gap_actual),
        'txn_frequency_per_week': freq,
        'profile_type':           profile_type,
    }


def build_seed_dataframe(n_seed):
    """Generate n_seed profiles across all three credit tiers."""
    n_high   = int(n_seed * 0.30)
    n_medium = int(n_seed * 0.40)
    n_low    = n_seed - n_high - n_medium

    records = []
    for ptype, count in [('high', n_high), ('medium', n_medium), ('low', n_low)]:
        records.extend([generate_profile(ptype) for _ in range(count)])

    df = pd.DataFrame(records)
    df['credit_score'] = df.apply(compute_score, axis=1)
    return df


# ── SDV Metadata configuration ────────────────────────────────────────────
def build_metadata(df):
    """
    Detect schema from seed dataframe and fine-tune column types.
    Specifying sdtype='categorical' for profile_type helps SDV treat
    it as a discrete variable rather than a continuous one.
    """
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)

    # Override profile_type as categorical (SDV may detect it as text/id)
    metadata.update_column('profile_type',  sdtype='categorical')

    # Ensure all numeric features are correctly typed
    for col in FEATURE_COLS:
        metadata.update_column(col, sdtype='numerical')

    metadata.update_column('credit_score', sdtype='numerical')
    return metadata


# ── Synthesizer factory ───────────────────────────────────────────────────
def build_synthesizer(model_name, metadata):
    """Instantiate the requested SDV synthesizer with sensible defaults."""

    if model_name == 'gaussian':
        print("  Synthesizer : GaussianCopulaSynthesizer")
        print("  Speed       : Fast (~30 seconds)")
        print("  Approach    : Classical statistics — Gaussian copula to model")
        print("                feature dependencies, per-column marginal distributions.")
        return GaussianCopulaSynthesizer(
            metadata,
            enforce_min_max_values=True,
            enforce_rounding=False,
        )

    if model_name == 'ctgan':
        print("  Synthesizer : CTGANSynthesizer  (GAN-based deep learning)")
        print("  Speed       : Slow — ~10 minutes on CPU (best quality)")
        print("  Approach    : Conditional Tabular GAN. Learns the full joint")
        print("                distribution; handles non-normal data excellently.")
        return CTGANSynthesizer(
            metadata,
            epochs=500,
            batch_size=500,
            generator_dim=(256, 256),
            discriminator_dim=(256, 256),
            enforce_rounding=False,
            enforce_min_max_values=True,
            verbose=True,
        )

    if model_name == 'tvae':
        print("  Synthesizer : TVAESynthesizer  (Variational Autoencoder)")
        print("  Speed       : Medium — ~5 minutes on CPU")
        print("  Approach    : Tabular VAE. Good balance of quality and speed.")
        return TVAESynthesizer(
            metadata,
            epochs=300,
            batch_size=500,
            compress_dims=(256, 256),
            decompress_dims=(256, 256),
            enforce_rounding=False,
            enforce_min_max_values=True,
        )

    raise ValueError(f"Unknown model: {model_name}. Choose gaussian / ctgan / tvae")


# ── Post-processing: clip generated values to realistic ranges ─────────────
def postprocess(df):
    """
    Clip generated feature values to realistic bounds.
    SDV may occasionally generate values slightly outside the training range;
    these clips ensure clean, usable training data.
    """
    df = df.copy()
    df['wallet_balance']          = df['wallet_balance'].clip(lower=0)
    df['total_transactions']      = df['total_transactions'].clip(lower=0).round()
    df['total_sent']              = df['total_sent'].clip(lower=0)
    df['total_received']          = df['total_received'].clip(lower=0)
    df['avg_transaction_amount']  = df['avg_transaction_amount'].clip(lower=0)
    df['account_age_days']        = df['account_age_days'].clip(lower=1).round()
    df['days_since_last_txn']     = df['days_since_last_txn'].clip(lower=0)
    df['txn_frequency_per_week']  = df['txn_frequency_per_week'].clip(lower=0)
    return df


# ── Statistical quality report ────────────────────────────────────────────
def quality_report(seed_df, synth_df, output_path):
    """
    Compare key statistics between seed data and SDV-generated data.
    Checks: mean, std, min, max, and Pearson correlation for each feature.
    A high-quality synthesizer should produce similar statistics.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("SDV Data Quality Report")
    lines.append("=" * 70)
    lines.append(f"\nSeed rows     : {len(seed_df):,}")
    lines.append(f"Generated rows: {len(synth_df):,}")
    lines.append("\n--- Per-feature statistics (seed vs generated) ---\n")

    mean_drifts = []
    std_drifts = []
    for col in FEATURE_COLS + ['credit_score']:
        s = seed_df[col]
        g = synth_df[col]
        mean_drift = abs(s.mean()-g.mean()) / max(abs(s.mean()), 1)
        std_drift = abs(s.std()-g.std()) / max(abs(s.std()), 1)
        mean_drifts.append(mean_drift)
        std_drifts.append(std_drift)
        lines.append(f"{col}")
        lines.append(f"  mean  seed={s.mean():>12.2f}   gen={g.mean():>12.2f}   "
                     f"diff={mean_drift*100:5.1f}%")
        lines.append(f"  std   seed={s.std():>12.2f}   gen={g.std():>12.2f}   "
                     f"diff={std_drift*100:5.1f}%")
        lines.append(f"  min   seed={s.min():>12.2f}   gen={g.min():>12.2f}")
        lines.append(f"  max   seed={s.max():>12.2f}   gen={g.max():>12.2f}")

    lines.append("\n--- Pairwise Pearson correlations (seed vs generated) ---\n")
    cols_to_check = ['wallet_balance', 'total_transactions', 'total_received',
                     'account_age_days', 'credit_score']
    seed_corr  = seed_df[cols_to_check].corr()
    synth_corr = synth_df[cols_to_check].corr()

    corr_drifts = []
    for i, c1 in enumerate(cols_to_check):
        for c2 in cols_to_check[i+1:]:
            sr = seed_corr.loc[c1, c2]
            gr = synth_corr.loc[c1, c2]
            corr_drifts.append(abs(sr - gr))
            match = "OK" if abs(sr - gr) < 0.15 else "DRIFT"
            lines.append(f"  {c1:30s} × {c2:30s}  "
                         f"seed={sr:+.3f}  gen={gr:+.3f}  [{match}]")

    lines.append("\n--- Credit score distribution ---\n")
    score_col = synth_df['credit_score']
    lines.append(f"  300-500 (Poor)     : {((score_col>=300)&(score_col<500)).sum():,}")
    lines.append(f"  500-580 (Fair-)    : {((score_col>=500)&(score_col<580)).sum():,}")
    lines.append(f"  580-670 (Fair+)    : {((score_col>=580)&(score_col<670)).sum():,}")
    lines.append(f"  670-740 (Good)     : {((score_col>=670)&(score_col<740)).sum():,}")
    lines.append(f"  740-800 (V.Good)   : {((score_col>=740)&(score_col<800)).sum():,}")
    lines.append(f"  800-900 (Excellent): {((score_col>=800)&(score_col<=900)).sum():,}")

    # Compact quality score in [0,1], higher is better.
    # Penalises large mean/std drifts and broken correlations.
    mean_component = max(0.0, 1.0 - min(1.0, float(np.mean(mean_drifts))))
    std_component = max(0.0, 1.0 - min(1.0, float(np.mean(std_drifts))))
    corr_component = max(0.0, 1.0 - min(1.0, float(np.mean(corr_drifts) / 0.5)))
    quality_score = 0.45 * mean_component + 0.25 * std_component + 0.30 * corr_component

    lines.append("\n--- Overall quality score ---\n")
    lines.append(f"  mean component : {mean_component:.3f}")
    lines.append(f"  std component  : {std_component:.3f}")
    lines.append(f"  corr component : {corr_component:.3f}")
    lines.append(f"  quality score  : {quality_score:.3f}  (target >= 0.65)")

    report = "\n".join(lines)
    print(report)
    with open(output_path, 'w') as f:
        f.write(report)
    print(f"\nQuality report saved -> {output_path}")
    return quality_score


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Generate SDV-enhanced credit training data')
    parser.add_argument('--model',     default='auto',
                        choices=['auto', 'gaussian', 'ctgan', 'tvae'],
                        help='SDV synthesizer (default: auto = gaussian fallback to tvae)')
    parser.add_argument('--seed-rows', type=int, default=4_000,
                        help='Number of seed profiles to fit SDV on (default: 4000)')
    parser.add_argument('--rows',      type=int, default=50_000,
                        help='Number of rows to generate (default: 50000)')
    parser.add_argument('--output',    default='credit_sdv_data.csv',
                        help='Output CSV path (default: credit_sdv_data.csv)')
    parser.add_argument('--save-synth', action='store_true',
                        help='Save the fitted synthesizer to sdv_synthesizer.pkl')
    args = parser.parse_args()

    print("=" * 60)
    print("GoPay Credit Engine — SDV Data Generator")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  SDV model   : {args.model}")
    print(f"  Seed rows   : {args.seed_rows:,}")
    print(f"  Target rows : {args.rows:,}")
    print(f"  Output      : {args.output}")

    # ── Step 1: Build seed dataset ────────────────────────────────────────
    print(f"\n[1/5] Generating {args.seed_rows:,} seed profiles ...")
    seed_df = build_seed_dataframe(args.seed_rows)
    print(f"      Score range: {seed_df['credit_score'].min()} – {seed_df['credit_score'].max()}")
    print(f"      Score mean : {seed_df['credit_score'].mean():.1f}")
    print(f"      Columns    : {list(seed_df.columns)}")

    # ── Step 2: Build metadata ────────────────────────────────────────────
    print("\n[2/5] Building SDV metadata from seed dataframe ...")
    metadata = build_metadata(seed_df)
    print("      Metadata detected and column types refined.")

    # ── Step 3 & 4: Fit and sample (with auto fallback) ───────────────────
    candidate_models = ['gaussian', 'tvae'] if args.model == 'auto' else [args.model]
    output_cols = FEATURE_COLS + ['credit_score']
    seed_with_score = seed_df[output_cols].copy()

    best_quality = -1.0
    best_model = None
    best_df = None
    best_synth = None

    for idx, model_name in enumerate(candidate_models, start=1):
        print(f"\n[3.{idx}] Fitting synthesizer ({model_name}) on {args.seed_rows:,} seed profiles ...")
        synthesizer = build_synthesizer(model_name, metadata)
        synthesizer.fit(seed_df)
        print("      Fitting complete.")

        print(f"[4.{idx}] Generating {args.rows:,} synthetic rows ...")
        synth_raw = synthesizer.sample(num_rows=args.rows)
        synth_df = postprocess(synth_raw)
        synth_df['credit_score'] = synth_df.apply(compute_score, axis=1)
        candidate_df = synth_df[output_cols].copy().dropna()

        report_path = f"sdv_quality_report_{model_name}.txt"
        print(f"[5.{idx}] Generating quality report ({model_name}) ...")
        score = quality_report(seed_with_score, candidate_df, report_path)
        print(f"      {model_name} quality score: {score:.3f}")

        if score > best_quality:
            best_quality = score
            best_model = model_name
            best_df = candidate_df
            best_synth = synthesizer

        if args.model == 'auto' and score >= 0.65:
            print(f"      Quality threshold met with {model_name}; stopping auto search.")
            break

    final_df = best_df
    final_df.to_csv(args.output, index=False)
    print(f"\nSelected model: {best_model} (quality={best_quality:.3f})")
    print(f"Saved {len(final_df):,} rows -> {args.output}")

    # Keep a canonical report filename for train workflow docs.
    final_report = f"sdv_quality_report_{best_model}.txt"
    with open(final_report, 'r') as src, open('sdv_quality_report.txt', 'w') as dst:
        dst.write(src.read())

    if args.save_synth and best_synth is not None:
        best_synth.save('sdv_synthesizer.pkl')
        print("Synthesizer saved -> sdv_synthesizer.pkl")

    print("\n" + "=" * 60)
    print("Done!")
    print(f"  Selected model : {best_model}")
    print(f"  Generated data : {args.output}  ({len(final_df):,} rows)")
    print(f"  Quality report : sdv_quality_report.txt")
    print("\nNext step: run `python train.py` to train on the SDV-enhanced dataset.")
    print("=" * 60)


if __name__ == '__main__':
    main()
