"""
Transaction Classifier
=======================
Rule-based classification with confidence scoring.

WHY rules-only (no LLM fallback in this PoC):
  - Rules are fast, free, deterministic, and auditable.
  - When an underwriter asks "why is this SALARY?", you point to a rule.
  - LLM classification is a black box — bad for regulated finance.
  - Rules cover ~80% of Indian banking transactions accurately.
  - The remaining ~20% get flagged for manual review — that's fine for a PoC.

  Alternative considered: LLM fallback for low-confidence transactions.
  Why deferred: adds complexity, API cost, and latency. Marked as
  production upgrade.

PRODUCTION UPGRADES:
  - LLM fallback for transactions below confidence threshold
  - ML classifier trained on labeled transaction data
  - Per-bank rule sets (SBI narrations differ from HDFC)
  - User feedback loop: manual corrections retrain the classifier
"""

from __future__ import annotations
import csv
import re
import logging
from pathlib import Path

import config
from models import Transaction

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# TAXONOMY — (regex_pattern, level2_category, confidence, is_essential, is_fixed)
#
# WHY these categories:
#   - They match standard Indian banking transaction patterns
#   - Ordered by specificity: more specific patterns first
#   - Confidence reflects how reliable the pattern match is
#     (ATM WDL is nearly 100% reliable, IMPS P2P is ~75% because
#      IMPS could also be a merchant payment)
# ═══════════════════════════════════════════════════════════════════════

RULES: list[tuple[str, str, float, bool, bool]] = [
    # High confidence — unambiguous patterns
    (r"ATM\s*W[DR]L|ATM\s*CASH",                "ATM_WITHDRAWAL",    0.98, False, False),
    (r"CREDIT\s*INTEREST|INT\.?\s*CR",           "INTEREST_INCOME",   0.97, False, False),
    (r"TREASURY\s*OFFICE|GOVT.*SALARY|PENSION",  "GOVERNMENT_CREDIT", 0.93, False, False),
    (r"\bSALARY\b|\bPAYROLL\b|\bSAL\s*CR\b",    "SALARY",            0.95, False, False),
    (r"\bEMI\b|LOAN\s*(?:REPAY|INSTAL)",         "EMI",               0.93, True,  True),
    (r"BANK\s*CHARGE|SERVICE\s*CHARGE|MIN\s*BAL", "BANK_CHARGES",     0.95, False, False),
    (r"REFUND|CASHBACK|REVERSAL",                "REFUND",            0.88, False, False),

    # Medium confidence — merchant/utility patterns
    (r"ELECTRIC|WATER\s*BILL|GAS\s*BILL|BESCOM",  "UTILITY",         0.90, True,  True),
    (r"INSURANCE|LIC|PREMIUM",                     "INSURANCE",       0.90, True,  True),
    (r"\bRENT\b|HOUSE\s*RENT",                     "RENT",            0.90, True,  True),
    (r"TUITION|SCHOOL|COLLEGE|EDUCATION\s*FEE",    "EDUCATION",       0.88, True,  False),
    (r"HOSPITAL|MEDICAL|PHARMACY|APOLLO",          "HEALTHCARE",      0.88, True,  False),
    (r"PETROL|DIESEL|HPCL|BPCL|IOCL",             "FUEL",            0.88, True,  False),
    (r"SWIGGY|ZOMATO|RESTAURANT|DOMINOS",          "FOOD_DINING",     0.85, False, False),
    (r"GROCERY|BIGBASKET|DMART|BLINKIT|ZEPTO",     "GROCERY",         0.85, True,  False),
    (r"AMAZON|FLIPKART|MYNTRA|SHOPPING",           "SHOPPING",        0.82, False, False),
    (r"NETFLIX|HOTSTAR|SPOTIFY|YOUTUBE\s*PREM",    "ENTERTAINMENT",   0.85, False, False),
    (r"IRCTC|MAKEMYTRIP|OLA|UBER",                "TRAVEL",          0.85, False, False),
    (r"MUTUAL\s*FUND|SIP|ZERODHA|GROWW",           "INVESTMENT",      0.85, False, False),
    (r"recharge|AIRTEL|JIO|VODAFONE",              "RECHARGE",        0.85, True,  False),

    # Lower confidence — generic transfer patterns
    # WHY these are last: IMPS/UPI could be P2P or merchant.
    # "ONE97 COMMUNICATIONS" is Paytm's legal entity name.
    (r"PAYTM|ONE97\s*COMMUNIC|PHONEPE|GOOGLE\s*PAY", "CARD_PAYMENT", 0.78, False, False),
    (r"UPI/CR/",                                   "P2P_RECEIVED",    0.75, False, False),
    (r"IMPS.*INB|BY\s*TRANSFER.*INB.*IMPS",        "P2P_RECEIVED",    0.72, False, False),
    (r"NEFT.*CR|BY\s*TRANSFER.*NEFT",              "CREDIT_OTHER",    0.60, False, False),
]


# ═══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

def classify_transactions(
    transactions: list[Transaction],
    custom_taxonomy_path: Path | None = None,
) -> list[Transaction]:
    """
    Classify each transaction in-place and return the list.

    Important classification rules (from assignment):
      - Not every credit is salary — P2P receipts classified separately
      - Merchant refunds are REFUND, not income
      - Narration text drives classification, not just amount
    """
    # Load custom rules if provided (CSV override)
    custom_rules = _load_custom_taxonomy(custom_taxonomy_path) if custom_taxonomy_path else []

    for txn in transactions:
        # Level 1 is straightforward
        txn.level1 = "CREDIT" if txn.credit else "DEBIT"

        # Level 2: try custom rules first, then defaults
        matched = False

        for keyword, category, conf in custom_rules:
            if keyword.upper() in txn.description.upper():
                txn.level2 = category
                txn.confidence = conf
                matched = True
                break

        if not matched:
            matched = _apply_rules(txn)

        # Fallback for unmatched
        if not matched:
            txn.level2 = "CREDIT_OTHER" if txn.level1 == "CREDIT" else "DEBIT_OTHER"
            txn.confidence = 0.30
            txn.method = "RULE"

        # Flag for review if low confidence or unknown counterparty
        if txn.confidence < config.CONFIDENCE_THRESHOLD:
            txn.needs_review = True
        if txn.counterparty == "UNKNOWN":
            txn.needs_review = True

    review_count = sum(1 for t in transactions if t.needs_review)
    logger.info(f"Classified {len(transactions)} transactions, {review_count} flagged for review")
    return transactions


def _apply_rules(txn: Transaction) -> bool:
    """Try each rule against the transaction description."""
    desc = txn.description.upper()

    for pattern, category, conf, is_essential, is_fixed in RULES:
        if re.search(pattern, desc, re.IGNORECASE):
            # Sanity: don't assign credit-only categories to debits
            if txn.level1 == "DEBIT" and category in _CREDIT_CATEGORIES:
                continue
            if txn.level1 == "CREDIT" and category in _DEBIT_CATEGORIES:
                continue

            txn.level2 = category
            txn.confidence = conf
            txn.method = "RULE"
            txn.is_essential = is_essential
            txn.is_fixed = is_fixed
            return True

    return False


# ═══════════════════════════════════════════════════════════════════════
# CUSTOM TAXONOMY LOADER
# ═══════════════════════════════════════════════════════════════════════

def _load_custom_taxonomy(path: Path) -> list[tuple[str, str, float]]:
    """
    CSV format: keyword, level2_category, confidence
    This lets users extend/override the default rules without code changes.
    """
    rules = []
    try:
        with open(path) as f:
            for row in csv.DictReader(f):
                rules.append((
                    row["keyword"],
                    row["level2_category"],
                    float(row.get("confidence", 0.90)),
                ))
    except Exception as e:
        logger.warning(f"Failed to load custom taxonomy: {e}")
    return rules


# Category sets for sanity checking
_CREDIT_CATEGORIES = {
    "SALARY", "GOVERNMENT_CREDIT", "INVESTMENT_INCOME", "LOAN_DISBURSAL",
    "REFUND", "P2P_RECEIVED", "INTEREST_INCOME", "REMITTANCE_IN", "CREDIT_OTHER",
}
_DEBIT_CATEGORIES = {
    "EMI", "UTILITY", "GROCERY", "FOOD_DINING", "FUEL", "EDUCATION",
    "HEALTHCARE", "INSURANCE", "RENT", "INVESTMENT", "ENTERTAINMENT",
    "SHOPPING", "TRAVEL", "P2P_SENT", "ATM_WITHDRAWAL", "BANK_CHARGES",
    "CARD_PAYMENT", "RECHARGE", "REMITTANCE_OUT", "TAX", "DEBIT_OTHER",
}
