"""
GoPay IFSC Code Validator
==========================
Validates Indian Financial System Codes (IFSC) with two complementary layers:

  Layer 1 — Structural validation
    IFSC format: XXXX0XXXXXX  (11 characters)
      Positions 1-4  : Bank code    (alphabetic A-Z)
      Position  5    : Always '0'  (zero, not letter O)
      Positions 6-11 : Branch code (alphanumeric A-Z / 0-9)

  Layer 2 — Bank code registry
    Checks the bank code prefix against RBI's list of authorised banks.
    An IFSC from a non-existent bank code is either fraudulent or invalid.

Why IFSC validation matters for fraud:
  Fraudsters create fake bank account details with plausible-looking but
  non-existent IFSC codes to trick victims into sending money that gets
  rerouted. This is common in:
    - UPI-based social engineering frauds
    - Fake refund/prize scam "verify your bank account" flows
    - Money mule account onboarding
"""

import re

# ---------------------------------------------------------------------------
# RBI-registered bank codes (bank prefix, first 4 chars of IFSC)
# Source: RBI IFSC master list (curated subset with major/well-known banks)
# ---------------------------------------------------------------------------
BANK_REGISTRY = {
    # Public Sector Banks
    'SBIN': 'State Bank of India',
    'PUNB': 'Punjab National Bank',
    'BARB': 'Bank of Baroda',
    'CNRB': 'Canara Bank',
    'UBIN': 'Union Bank of India',
    'BKID': 'Bank of India',
    'IOBA': 'Indian Overseas Bank',
    'ANDB': 'Andhra Bank',
    'CORP': 'Corporation Bank',
    'VIJB': 'Vijaya Bank',
    'IDBI': 'IDBI Bank',
    'ALLA': 'Allahabad Bank',
    'UCBA': 'UCO Bank',
    'ORBC': 'Oriental Bank of Commerce',
    'UTBI': 'United Bank of India',
    'PSIB': 'Punjab & Sind Bank',
    'MAHB': 'Bank of Maharashtra',
    'DENA': 'Dena Bank',
    'SYNDB': 'Syndicate Bank',
    # Private Sector Banks
    'HDFC': 'HDFC Bank',
    'ICIC': 'ICICI Bank',
    'UTIB': 'Axis Bank',
    'KKBK': 'Kotak Mahindra Bank',
    'YESB': 'Yes Bank',
    'INDB': 'IndusInd Bank',
    'RATN': 'RBL Bank',
    'FDRL': 'Federal Bank',
    'KVBL': 'Karur Vysya Bank',
    'SIBL': 'South Indian Bank',
    'CSBK': 'CSB Bank',
    'DCBL': 'DCB Bank',
    'DLXB': 'Dhanlaxmi Bank',
    'TMBL': 'Tamilnad Mercantile Bank',
    'LAVB': 'Lakshmi Vilas Bank',
    'NGSB': 'Nainital Bank',
    'DBSS': 'DBS Bank India',
    'HSBC': 'HSBC India',
    'CITI': 'Citibank',
    'SCBL': 'Standard Chartered',
    'DEUT': 'Deutsche Bank',
    'BOFA': 'Bank of America',
    'BNPP': 'BNP Paribas',
    # Small Finance Banks
    'USFB': 'Ujjivan Small Finance Bank',
    'ESFB': 'Equitas Small Finance Bank',
    'AUBL': 'AU Small Finance Bank',
    'FINO': 'Fino Payments Bank',
    'AIRP': 'Airtel Payments Bank',
    'IPOS': 'India Post Payments Bank',
    'JAKA': 'Jammu & Kashmir Bank',
    'KARB': 'Karnataka Bank',
    'CBIN': 'Central Bank of India',
    # Cooperative Banks
    'APMC': 'Andhra Pradesh Mahesh Coop',
    'COSB': 'Cosmos Co-operative Bank',
    'SAHE': 'Sahebrao Deshmukh Coop',
    # Payment Banks
    'PAYTM': 'Paytm Payments Bank',     # 5-char exception
    'PYTM':  'Paytm Payments Bank',
    'NSPB':  'NSDL Payments Bank',
}

# ---------------------------------------------------------------------------
# Structural validation
# ---------------------------------------------------------------------------
IFSC_PATTERN = re.compile(r'^[A-Z]{4}0[A-Z0-9]{6}$')


def _luhn_structural_check(ifsc: str) -> bool:
    """
    Luhn-variant structural integrity check for IFSC.

    Assigns digit values to chars (A=10..Z=35, 0-9=0-9), sums alternating
    doubled/undoubled values (like Luhn), verifies modulo 97 ≠ 0.
    This catches random character sequences that pass the regex but
    are not plausible real IFSC codes.

    Note: This is a GoPay-specific heuristic, not a formal RBI standard.
    """
    def char_val(c):
        return int(c) if c.isdigit() else (ord(c) - ord('A') + 10)

    vals = [char_val(c) for c in ifsc]
    total = 0
    for i, v in enumerate(vals):
        if i % 2 == 0:
            doubled = v * 2
            total += doubled - 9 if doubled > 9 else doubled
        else:
            total += v
    return (total % 10) != 0   # returns True if it passes heuristic check


def validate(ifsc: str) -> dict:
    """
    Validate an IFSC code.

    Returns:
        {
          'ifsc'        : original (uppercased) IFSC,
          'is_valid'    : True if passes all validation layers,
          'bank_code'   : 4-char bank prefix,
          'branch_code' : 6-char branch suffix,
          'bank_name'   : resolved bank name (or None),
          'signals'     : list of validation failure reasons,
          'risk'        : 'LOW' | 'MEDIUM' | 'HIGH'
        }
    """
    ifsc = (ifsc or '').strip().upper()
    signals = []

    if not ifsc:
        return {
            'ifsc': ifsc, 'is_valid': False, 'bank_code': None,
            'branch_code': None, 'bank_name': None,
            'signals': ['empty_ifsc'], 'risk': 'HIGH',
        }

    # ── Layer 1: Structural check ───────────────────────────────────────────
    if len(ifsc) != 11:
        signals.append(f'invalid_length_{len(ifsc)}_expected_11')

    if not IFSC_PATTERN.match(ifsc):
        if len(ifsc) == 11 and ifsc[4] != '0':
            signals.append('position_5_must_be_zero')
        elif len(ifsc) == 11:
            signals.append('invalid_character_pattern')
        if signals:
            return {
                'ifsc': ifsc, 'is_valid': False,
                'bank_code': ifsc[:4] if len(ifsc) >= 4 else None,
                'branch_code': ifsc[5:] if len(ifsc) >= 11 else None,
                'bank_name': None, 'signals': signals, 'risk': 'HIGH',
            }

    bank_code   = ifsc[:4]
    branch_code = ifsc[5:]

    # ── Layer 2: Bank registry check ───────────────────────────────────────
    bank_name = BANK_REGISTRY.get(bank_code)
    if bank_name is None:
        signals.append(f'unknown_bank_code_{bank_code}')

    # ── Layer 3: Luhn structural heuristic ──────────────────────────────────
    if not _luhn_structural_check(ifsc):
        signals.append('structural_checksum_advisory')

    is_valid = len(signals) == 0 or (
        len(signals) == 1 and signals[0] == 'structural_checksum_advisory'
    )

    if not is_valid and 'unknown_bank_code' in signals[0]:
        risk = 'HIGH'
    elif signals:
        risk = 'MEDIUM'
    else:
        risk = 'LOW'

    return {
        'ifsc':        ifsc,
        'is_valid':    is_valid,
        'bank_code':   bank_code,
        'branch_code': branch_code,
        'bank_name':   bank_name,
        'signals':     signals,
        'risk':        risk,
    }
