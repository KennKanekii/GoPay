"""
GoPay Fraud Detection Engine — Flask API
==========================================
Loads the trained RandomForestClassifier and provides a real-time
fraud scoring endpoint. Spring Boot calls this before every transaction.

Port: 5002

Endpoints:
    GET  /health           → service health + model status
    POST /assess           → score a transaction, return risk + signals
    GET  /signals          → human-readable signal catalogue (for docs/UI)
"""

import os
import sys
import json
import numpy as np
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'fraud_model.pkl')

try:
    artifact  = joblib.load(MODEL_PATH)
    MODEL     = artifact['model']
    FEATURES  = artifact['features']
    MODEL_OK  = True
    print(f"[fraud-engine] Model loaded  |  Features: {len(FEATURES)}")
except Exception as e:
    MODEL_OK = False
    print(f"[fraud-engine] WARNING: model not found — {e}", file=sys.stderr)
    print("[fraud-engine] Run `python train.py` first.", file=sys.stderr)

# ---------------------------------------------------------------------------
# Risk band definitions (mirrors Spring Boot FraudService)
# ---------------------------------------------------------------------------
RISK_BANDS = [
    (80, 'CRITICAL', 'BLOCK',  '#dc2626'),
    (60, 'HIGH',     'REVIEW', '#ea580c'),
    (35, 'MEDIUM',   'ALLOW',  '#d97706'),
    (0,  'LOW',      'ALLOW',  '#16a34a'),
]

def score_to_band(score):
    for threshold, band, recommendation, colour in RISK_BANDS:
        if score >= threshold:
            return band, recommendation, colour
    return 'LOW', 'ALLOW', '#16a34a'

# ---------------------------------------------------------------------------
# Signal detection — maps feature values to human-readable fraud signals
# ---------------------------------------------------------------------------
def extract_signals(data, fraud_prob):
    signals = []

    if data.get('is_blacklisted', 0):
        signals.append({'code': 'blacklisted_recipient',   'severity': 'CRITICAL',
                        'label': 'Recipient on fraud blacklist'})

    if data.get('txns_last_1h', 0) >= 4:
        signals.append({'code': 'high_velocity_1h',        'severity': 'HIGH',
                        'label': f"High velocity: {data['txns_last_1h']} transactions in 1 hour"})

    if data.get('txns_last_24h', 0) >= 10:
        signals.append({'code': 'high_velocity_24h',       'severity': 'MEDIUM',
                        'label': f"High velocity: {data['txns_last_24h']} transactions in 24 hours"})

    if data.get('amount_to_balance_ratio', 0) >= 0.75:
        pct = round(data['amount_to_balance_ratio'] * 100, 1)
        signals.append({'code': 'high_balance_drain',      'severity': 'HIGH',
                        'label': f"Transaction drains {pct}% of wallet balance"})

    if data.get('amount_to_avg_ratio', 0) >= 5:
        ratio = round(data['amount_to_avg_ratio'], 1)
        signals.append({'code': 'unusual_amount',          'severity': 'HIGH',
                        'label': f"Amount is {ratio}x the sender's historical average"})

    if data.get('is_new_recipient', 0) and data.get('amount', 0) >= 10_000:
        signals.append({'code': 'large_new_recipient',     'severity': 'MEDIUM',
                        'label': 'Large amount to a first-time recipient'})

    if data.get('is_night', 0) and data.get('is_new_recipient', 0):
        signals.append({'code': 'night_new_recipient',     'severity': 'MEDIUM',
                        'label': 'Late-night transaction to new recipient'})

    if data.get('unique_recipients_24h', 0) >= 5:
        signals.append({'code': 'many_recipients',         'severity': 'MEDIUM',
                        'label': f"{data['unique_recipients_24h']} unique recipients in 24 hours"})

    amt = data.get('amount', 0)
    if amt > 500 and amt % 1000 < 50 and data.get('txns_last_24h', 0) >= 3:
        signals.append({'code': 'structuring_pattern',     'severity': 'HIGH',
                        'label': 'Repeated round-amount transactions (structuring pattern)'})

    if data.get('amount_sent_last_1h', 0) >= 15_000:
        signals.append({'code': 'high_spend_rate_1h',      'severity': 'HIGH',
                        'label': f"High spend rate: Rs.{data['amount_sent_last_1h']:.0f} in last hour"})

    if data.get('account_age_days', 365) < 7 and data.get('amount', 0) >= 5_000:
        signals.append({'code': 'new_account_large_txn',   'severity': 'MEDIUM',
                        'label': 'New account (<7 days) making a large transaction'})

    if not signals and fraud_prob >= 0.35:
        signals.append({'code': 'ml_anomaly',              'severity': 'MEDIUM',
                        'label': 'Behavioral pattern anomaly detected by ML model'})

    return signals


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'ok': True, 'service': 'fraud-engine',
                    'model': 'loaded' if MODEL_OK else 'missing',
                    'port': 5002})


@app.route('/signals', methods=['GET'])
def signal_catalogue():
    """Developer reference — all possible signals the engine can emit."""
    return jsonify([
        {'code': 'blacklisted_recipient',   'severity': 'CRITICAL', 'description': 'Recipient domain/email is on the GoPay blacklist'},
        {'code': 'high_velocity_1h',        'severity': 'HIGH',     'description': '4+ transactions in a single hour'},
        {'code': 'high_velocity_24h',       'severity': 'MEDIUM',   'description': '10+ transactions in 24 hours'},
        {'code': 'high_balance_drain',      'severity': 'HIGH',     'description': 'Transaction >= 75% of wallet balance'},
        {'code': 'unusual_amount',          'severity': 'HIGH',     'description': 'Amount >= 5x the sender historical average'},
        {'code': 'large_new_recipient',     'severity': 'MEDIUM',   'description': '>= Rs.10,000 to a first-time recipient'},
        {'code': 'night_new_recipient',     'severity': 'MEDIUM',   'description': 'Late-night (00:00-05:59) transaction to new recipient'},
        {'code': 'many_recipients',         'severity': 'MEDIUM',   'description': '5+ distinct recipients in 24 hours'},
        {'code': 'structuring_pattern',     'severity': 'HIGH',     'description': 'Round-amount repeated transactions (smurfing/structuring)'},
        {'code': 'high_spend_rate_1h',      'severity': 'HIGH',     'description': '>= Rs.15,000 total sent within the last hour'},
        {'code': 'new_account_large_txn',   'severity': 'MEDIUM',   'description': 'Account < 7 days old, transaction >= Rs.5,000'},
        {'code': 'ml_anomaly',              'severity': 'MEDIUM',   'description': 'Statistical anomaly detected by ML model (unexplained deviation)'},
    ])


@app.route('/assess', methods=['POST'])
def assess():
    if not MODEL_OK:
        return jsonify({'ok': False, 'error': 'Model not loaded. Run python train.py first.'}), 503

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({'ok': False, 'error': 'JSON body required.'}), 400

    try:
        x = np.array([[data.get(f, 0) for f in FEATURES]], dtype=float)
    except Exception as e:
        return jsonify({'ok': False, 'error': f'Invalid feature vector: {e}'}), 400

    fraud_prob  = float(MODEL.predict_proba(x)[0][1])
    fraud_score = int(round(fraud_prob * 100))
    fraud_score = max(0, min(100, fraud_score))

    # Hard rule: blacklisted = always CRITICAL, regardless of ML score
    if data.get('is_blacklisted', 0):
        fraud_score = max(fraud_score, 85)

    risk_band, recommendation, colour = score_to_band(fraud_score)
    signals = extract_signals(data, fraud_prob)

    return jsonify({
        'ok':             True,
        'fraudScore':     fraud_score,
        'fraudProb':      round(fraud_prob, 4),
        'riskLevel':      risk_band,
        'recommendation': recommendation,
        'colour':         colour,
        'signals':        signals,
        'model':          'random_forest_v1',
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5002))
    print(f"[fraud-engine] Starting on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
