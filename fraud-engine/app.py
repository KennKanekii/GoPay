"""
GoPay Fraud Detection Engine — Flask API (v2)
==============================================
Port: 5002

Endpoints:
    GET  /health           → service health
    POST /assess           → score transaction (20 features, XGBoost)
    GET  /signals          → signal catalogue
    POST /vpa-check        → VPA spoofing detection (Levenshtein + Jaro-Winkler)
    POST /ifsc-validate    → IFSC structural + bank registry validation
"""

import os
import sys
import numpy as np
import joblib
from flask import Flask, request, jsonify

import vpa_detector
import ifsc_validator

app     = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Load XGBoost model
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'fraud_model.pkl')

try:
    artifact    = joblib.load(MODEL_PATH)
    MODEL       = artifact['model']
    FEATURES    = artifact['features']
    MODEL_NAME  = artifact.get('model_name', 'xgboost_v2')
    MODEL_OK    = True
    print(f"[fraud-engine] Model loaded: {MODEL_NAME}  |  Features: {len(FEATURES)}")
except Exception as e:
    MODEL_OK   = False
    MODEL_NAME = 'not_loaded'
    FEATURES   = []
    print(f"[fraud-engine] WARNING: model not loaded — {e}", file=sys.stderr)
    print("[fraud-engine] Run `python train.py` first.", file=sys.stderr)

# ─────────────────────────────────────────────────────────────────────────────
# Risk band thresholds
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# Signal extraction — human-readable audit trail for every assessment
# ─────────────────────────────────────────────────────────────────────────────
def extract_signals(data, fraud_prob):
    signals = []

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

    if data.get('txns_last_1h', 0) >= 4:
        signals.append({'code': 'high_velocity_1h', 'severity': 'HIGH',
                        'label': f"High velocity: {data['txns_last_1h']} transactions in 1 hour"})

    if data.get('txns_last_24h', 0) >= 10:
        signals.append({'code': 'high_velocity_24h', 'severity': 'MEDIUM',
                        'label': f"High velocity: {data['txns_last_24h']} transactions in 24 hours"})

    if data.get('amount_to_balance_ratio', 0) >= 0.75:
        pct = round(data['amount_to_balance_ratio'] * 100, 1)
        signals.append({'code': 'high_balance_drain', 'severity': 'HIGH',
                        'label': f'Transaction drains {pct}% of wallet balance'})

    if data.get('amount_to_avg_ratio', 0) >= 5:
        ratio = round(data['amount_to_avg_ratio'], 1)
        signals.append({'code': 'unusual_amount', 'severity': 'HIGH',
                        'label': f"Amount is {ratio}x the sender's historical average"})

    zscore = data.get('amount_zscore', 0)
    if abs(zscore) >= 3:
        signals.append({'code': 'amount_statistical_anomaly', 'severity': 'HIGH',
                        'label': f'Amount is {abs(zscore):.1f} standard deviations from sender norm (z-score anomaly)'})

    if data.get('is_new_recipient', 0) and data.get('amount', 0) >= 10_000:
        signals.append({'code': 'large_new_recipient', 'severity': 'MEDIUM',
                        'label': 'Large amount to a first-time recipient'})

    if data.get('is_night', 0) and data.get('is_new_recipient', 0):
        signals.append({'code': 'night_new_recipient', 'severity': 'MEDIUM',
                        'label': 'Late-night transaction to new recipient'})

    if data.get('unique_recipients_24h', 0) >= 5:
        signals.append({'code': 'many_recipients', 'severity': 'MEDIUM',
                        'label': f"{data['unique_recipients_24h']} unique recipients in 24 hours"})

    amt = data.get('amount', 0)
    if amt > 500 and amt % 1000 < 50 and data.get('txns_last_24h', 0) >= 3:
        signals.append({'code': 'structuring_pattern', 'severity': 'HIGH',
                        'label': 'Repeated round-amount transactions (structuring/smurfing)'})

    if data.get('amount_sent_last_1h', 0) >= 15_000:
        signals.append({'code': 'high_spend_rate_1h', 'severity': 'HIGH',
                        'label': f"High spend rate: Rs.{data['amount_sent_last_1h']:.0f} in last hour"})

    if data.get('account_age_days', 365) < 7 and data.get('amount', 0) >= 5_000:
        signals.append({'code': 'new_account_large_txn', 'severity': 'MEDIUM',
                        'label': 'New account (<7 days) making a large transaction'})

    if not signals and fraud_prob >= 0.35:
        signals.append({'code': 'ml_anomaly', 'severity': 'MEDIUM',
                        'label': 'Behavioral pattern anomaly detected by XGBoost model'})

    return signals


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'ok': True, 'service': 'fraud-engine',
                    'model': MODEL_NAME if MODEL_OK else 'missing',
                    'features': len(FEATURES), 'port': 5002})


@app.route('/signals', methods=['GET'])
def signal_catalogue():
    return jsonify([
        {'code': 'blacklisted_recipient',      'severity': 'CRITICAL', 'description': 'Recipient on fraud blacklist'},
        {'code': 'vpa_spoofing_detected',      'severity': 'CRITICAL', 'description': 'VPA handle is a lookalike of a known legitimate handle (edit-distance ≤ 1)'},
        {'code': 'vpa_suspicious',             'severity': 'HIGH',     'description': 'VPA handle has high Jaro-Winkler similarity to a known handle'},
        {'code': 'invalid_ifsc',               'severity': 'HIGH',     'description': 'IFSC code failed structural or bank-registry validation'},
        {'code': 'high_velocity_1h',           'severity': 'HIGH',     'description': '4+ transactions in a single hour'},
        {'code': 'high_velocity_24h',          'severity': 'MEDIUM',   'description': '10+ transactions in 24 hours'},
        {'code': 'high_balance_drain',         'severity': 'HIGH',     'description': 'Transaction drains >= 75% of wallet balance'},
        {'code': 'unusual_amount',             'severity': 'HIGH',     'description': 'Amount is >= 5x the sender historical average'},
        {'code': 'amount_statistical_anomaly', 'severity': 'HIGH',     'description': 'Amount deviates > 3 standard deviations from sender norm (z-score)'},
        {'code': 'large_new_recipient',        'severity': 'MEDIUM',   'description': '>= Rs.10,000 to a first-time recipient'},
        {'code': 'night_new_recipient',        'severity': 'MEDIUM',   'description': 'Late-night (00:00–05:59) transaction to new recipient'},
        {'code': 'many_recipients',            'severity': 'MEDIUM',   'description': '5+ distinct recipients in 24 hours'},
        {'code': 'structuring_pattern',        'severity': 'HIGH',     'description': 'Round-amount repeated transactions (smurfing/structuring)'},
        {'code': 'high_spend_rate_1h',         'severity': 'HIGH',     'description': '>= Rs.15,000 total sent within the last hour'},
        {'code': 'new_account_large_txn',      'severity': 'MEDIUM',   'description': 'Account < 7 days old, transaction >= Rs.5,000'},
        {'code': 'ml_anomaly',                 'severity': 'MEDIUM',   'description': 'Statistical anomaly detected by XGBoost (unexplained deviation)'},
    ])


@app.route('/assess', methods=['POST'])
def assess():
    if not MODEL_OK:
        return jsonify({'ok': False, 'error': 'Model not loaded. Run python train.py first.'}), 503

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({'ok': False, 'error': 'JSON body required.'}), 400

    try:
        x = np.array([[float(data.get(f, 0)) for f in FEATURES]])
    except Exception as e:
        return jsonify({'ok': False, 'error': f'Invalid feature vector: {e}'}), 400

    fraud_prob  = float(MODEL.predict_proba(x)[0][1])
    fraud_score = int(round(fraud_prob * 100))
    fraud_score = max(0, min(100, fraud_score))

    # Hard rules override ML score
    if data.get('is_blacklisted', 0):
        fraud_score = max(fraud_score, 85)
    if data.get('vpa_risk_score', 0) >= 60:
        fraud_score = max(fraud_score, 75)
    if not data.get('ifsc_is_valid', 1) and data.get('amount', 0) >= 5_000:
        fraud_score = max(fraud_score, 65)

    risk_band, recommendation, colour = score_to_band(fraud_score)
    signals = extract_signals(data, fraud_prob)

    return jsonify({
        'ok': True, 'fraudScore': fraud_score, 'fraudProb': round(fraud_prob, 4),
        'riskLevel': risk_band, 'recommendation': recommendation,
        'colour': colour, 'signals': signals, 'model': MODEL_NAME,
    })


@app.route('/vpa-check', methods=['POST'])
def vpa_check():
    """
    Detect VPA (UPI handle) spoofing using Levenshtein distance and Jaro-Winkler similarity.
    Body: { "vpa": "username@bankhandle" }
    """
    data = request.get_json(silent=True) or {}
    vpa  = data.get('vpa', '')
    result = vpa_detector.check(vpa)
    return jsonify({'ok': True, **result})


@app.route('/ifsc-validate', methods=['POST'])
def ifsc_validate():
    """
    Validate an IFSC code structurally and against the RBI bank registry.
    Body: { "ifsc": "HDFC0001234" }
    """
    data = request.get_json(silent=True) or {}
    ifsc = data.get('ifsc', '')
    result = ifsc_validator.validate(ifsc)
    return jsonify({'ok': True, **result})


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5002))
    print(f"[fraud-engine] Starting on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
