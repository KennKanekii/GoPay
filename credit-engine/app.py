"""
GoPay Credit Engine — Flask Scoring API
=========================================
Loads the trained GradientBoostingRegressor and exposes a lightweight
REST API that Spring Boot calls to score a user.

Start:
    python app.py          (default port 5001)
    PORT=5001 python app.py

Endpoints:
    GET  /health           → { "ok": true, "model": "loaded" }
    POST /score            → credit score + breakdown
"""

import os
import sys
import numpy as np
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load model at startup
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')

try:
    artifact   = joblib.load(MODEL_PATH)
    MODEL      = artifact['model']
    FEATURES   = artifact['features']
    MODEL_OK   = True
    print(f"[credit-engine] Model loaded from {MODEL_PATH}")
    print(f"[credit-engine] Features: {FEATURES}")
except Exception as e:
    MODEL_OK = False
    print(f"[credit-engine] WARNING: could not load model — {e}", file=sys.stderr)
    print(f"[credit-engine] Run `python train.py` first, then restart.", file=sys.stderr)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BANDS = [
    (800, 'EXCELLENT', '#16a34a', 'Exceptional credit health. Eligible for the best loan rates.'),
    (740, 'VERY_GOOD',  '#4ade80', 'Very strong credit. Most lenders will offer favourable terms.'),
    (670, 'GOOD',       '#86efac', 'Good credit standing. Eligible for most standard loan products.'),
    (580, 'FAIR',       '#fbbf24', 'Fair credit. Some lenders may require higher interest rates.'),
    (300, 'POOR',       '#ef4444', 'Poor credit. Work on increasing your balance and transaction activity.'),
]

def score_to_band(score):
    for threshold, band, colour, tip in BANDS:
        if score >= threshold:
            return band, colour, tip
    return 'POOR', '#ef4444', BANDS[-1][3]


def compute_factor_contributions(features_dict):
    """
    Return individual factor scores (0-100) that explain the prediction
    in human-readable terms. These are rule-based decompositions used for the
    breakdown panel — they are *not* the ML model internals.
    """
    balance     = features_dict.get('wallet_balance', 0)
    txns        = features_dict.get('total_transactions', 0)
    sent        = features_dict.get('total_sent', 0)
    received    = features_dict.get('total_received', 0)
    age_days    = features_dict.get('account_age_days', 0)
    days_gap    = features_dict.get('days_since_last_txn', 999)

    b_factor  = round(min(100, balance / 75_000 * 100), 1)
    a_factor  = round(min(100, txns / 200 * 100), 1)
    nf_factor = round(received / max(1, sent + received) * 100, 1) if (sent + received) > 0 else 50.0
    r_factor  = round(max(0, 100 - days_gap / 60 * 100), 1) if txns > 0 else 0.0
    age_factor = round(min(100, age_days / 1095 * 100), 1)

    return {
        'balanceFactor':     b_factor,
        'activityFactor':    a_factor,
        'netFlowFactor':     nf_factor,
        'recencyFactor':     r_factor,
        'accountAgeFactor':  age_factor,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'ok': True, 'model': 'loaded' if MODEL_OK else 'missing'})


@app.route('/score', methods=['POST'])
def score():
    if not MODEL_OK:
        return jsonify({'ok': False, 'error': 'Model not loaded. Run python train.py first.'}), 503

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({'ok': False, 'error': 'JSON body required.'}), 400

    # Build feature vector in the correct order
    try:
        x = np.array([[data.get(f, 0) for f in FEATURES]], dtype=float)
    except Exception as e:
        return jsonify({'ok': False, 'error': f'Invalid features: {e}'}), 400

    # Predict
    raw_score = float(MODEL.predict(x)[0])
    score_val = int(np.clip(round(raw_score), 300, 900))

    band, colour, tip = score_to_band(score_val)
    breakdown = compute_factor_contributions(data)

    return jsonify({
        'ok':        True,
        'score':     score_val,
        'riskBand':  band,
        'colour':    colour,
        'tip':       tip,
        'breakdown': breakdown,
        'model':     'ml_gradient_boosting',
        'features':  FEATURES,
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f"[credit-engine] Starting on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
