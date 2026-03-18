"""
Credit Risk Scorer
===================
Computes a composite credit risk score (0–1000) from weighted components.

IMPLEMENTED (4 of 6 — the ones with highest underwriting signal):
  1. Income Stability  (30%) — most predictive of repayment ability
  2. Liquidity         (25%) — cash buffer = safety net
  3. Banking Behaviour (25%) — bounces/penalties = red flags
  4. Expense Mgmt      (20%) — spending discipline

NOT IMPLEMENTED (acknowledged — would add in production):
  5. Debt Service — needs EMI pattern detection across 6+ months.
     With only 2–3 months of data, FOIR calculation is unreliable.
     Marked as future work with clear reasoning.
  6. Fraud Indicators — balance arithmetic check is done in validation,
     but circular transaction detection needs transaction graph analysis.
     Simplified version included.

WHY these weights:
  - Income Stability at 30%: A person with stable income is the best
    credit candidate. This is the #1 signal in any underwriting model.
  - Liquidity at 25%: Cash buffer means they can absorb shocks.
  - Banking Behaviour at 25%: Bounces and penalties directly indicate
    payment discipline. One bounce is noise, three is a pattern.
  - Expense Management at 20%: Spending >90% of income is risky.

PRODUCTION UPGRADES:
  - All 6 components with weights calibrated against loan performance data
  - Debt Service: proper EMI detection with recurring payment analysis
  - Fraud: balance arithmetic validation, circular txn graph analysis,
    structuring detection (many cash txns just below ₹50K threshold)
  - Time-series analysis for income/expense trends over 12+ months
"""

from __future__ import annotations
import logging
from collections import defaultdict
from statistics import mean, stdev

import config
from models import Transaction, CreditRiskSummary, RiskComponentScore

logger = logging.getLogger(__name__)


def compute_risk(
    transactions: list[Transaction],
    customer_name: str = "",
    account_number: str = "",
) -> CreditRiskSummary:
    """Compute full credit risk summary."""

    # Ensure month is set on all transactions (defensive)
    for t in transactions:
        if not t.month:
            t.month = t.txn_date.strftime("%Y-%m")

    if not transactions:
        return CreditRiskSummary(
            customer_name=customer_name,
            account_number=account_number,
            rationale=["No transactions to analyze"],
        )

    period = f"{transactions[0].txn_date} to {transactions[-1].txn_date}"
    total_credits = sum(t.credit or 0 for t in transactions)
    total_debits = sum(t.debit or 0 for t in transactions)

    # Compute each component
    income_comp = _score_income(transactions)
    liquidity_comp = _score_liquidity(transactions)
    banking_comp = _score_banking(transactions)
    expense_comp = _score_expenses(transactions, income_comp.details.get("avg_monthly_income", 0))

    components = [income_comp, liquidity_comp, banking_comp, expense_comp]

    # Weighted composite
    composite = sum(c.score * c.weight for c in components)
    composite = int(round(max(0, min(1000, composite))))

    # Rating band
    rating = "Poor"
    for low, high, band in config.RATING_BANDS:
        if low <= composite <= high:
            rating = band
            break

    # Decision
    decision, rationale = _decide(composite, rating, components, transactions)

    return CreditRiskSummary(
        customer_name=customer_name,
        account_number=account_number,
        analysis_period=period,
        components=components,
        composite_score=composite,
        rating_band=rating,
        decision=decision,
        rationale=rationale,
        total_credits=round(total_credits, 2),
        total_debits=round(total_debits, 2),
        net_cashflow=round(total_credits - total_debits, 2),
        transaction_count=len(transactions),
    )


# ═══════════════════════════════════════════════════════════════════════
# COMPONENT SCORERS
# ═══════════════════════════════════════════════════════════════════════

def _score_income(txns: list[Transaction]) -> RiskComponentScore:
    """
    Income Stability: measures regularity and diversity of income.

    Logic:
      - Group income transactions by month
      - Calculate coefficient of variation (lower = more stable)
      - Count distinct income sources (salary, govt, interest, etc.)
      - Detect growth/decline trend
    """
    income_cats = {"SALARY", "GOVERNMENT_CREDIT", "INVESTMENT_INCOME", "INTEREST_INCOME", "REMITTANCE_IN"}
    monthly: dict[str, float] = defaultdict(float)
    sources: set[str] = set()

    for t in txns:
        if t.level1 == "CREDIT" and t.level2 in income_cats:
            monthly[t.month] += t.credit or 0
            sources.add(t.level2)

    if not monthly:
        return RiskComponentScore(
            name="Income Stability", weight=config.RISK_WEIGHTS["income_stability"],
            score=200, details={"avg_monthly_income": 0, "reason": "No identifiable income"},
        )

    totals = [monthly[m] for m in sorted(monthly.keys())]
    avg_income = mean(totals)
    source_count = len(sources)

    # Regularity: 1 - CV (coefficient of variation)
    regularity = 1.0
    if len(totals) >= 2 and avg_income > 0:
        regularity = max(0, 1 - stdev(totals) / avg_income)

    # Trend
    trend = "STABLE"
    if len(totals) >= 2:
        if totals[-1] > totals[0] * 1.05:
            trend = "GROWING"
        elif totals[-1] < totals[0] * 0.95:
            trend = "DECLINING"

    # Score
    score = 400
    score += regularity * 250       # up to +250 for stable income
    score += min(source_count * 75, 150)  # up to +150 for diversity
    score += 50 if trend == "GROWING" else (-50 if trend == "DECLINING" else 0)
    score += 50  # has income at all

    return RiskComponentScore(
        name="Income Stability", weight=config.RISK_WEIGHTS["income_stability"],
        score=round(max(0, min(1000, score))),
        details={
            "avg_monthly_income": round(avg_income, 2),
            "months_analyzed": len(totals),
            "regularity": round(regularity, 3),
            "source_count": source_count,
            "trend": trend,
        },
    )


def _score_liquidity(txns: list[Transaction]) -> RiskComponentScore:
    """
    Liquidity: are there sufficient cash reserves?

    Logic:
      - Track end-of-day balance (last balance per date)
      - Score based on average and minimum balance
      - Penalize negative balance days
    """
    eod_balances: dict[str, float] = {}
    for t in txns:
        if t.balance is not None:
            eod_balances[str(t.txn_date)] = t.balance

    if not eod_balances:
        return RiskComponentScore(
            name="Liquidity", weight=config.RISK_WEIGHTS["liquidity"],
            score=300, details={"reason": "No balance data available"},
        )

    balances = list(eod_balances.values())
    avg_bal = mean(balances)
    min_bal = min(balances)
    neg_days = sum(1 for b in balances if b < 0)

    score = 500
    if avg_bal >= 25000:
        score += 250
    elif avg_bal >= 10000:
        score += 150
    elif avg_bal >= 5000:
        score += 50
    else:
        score -= 100

    if min_bal >= 5000:
        score += 100
    elif min_bal < 0:
        score -= 200

    score -= neg_days * 75

    return RiskComponentScore(
        name="Liquidity", weight=config.RISK_WEIGHTS["liquidity"],
        score=round(max(0, min(1000, score))),
        details={
            "avg_eod_balance": round(avg_bal, 2),
            "min_eod_balance": round(min_bal, 2),
            "negative_balance_days": neg_days,
        },
    )


def _score_banking(txns: list[Transaction]) -> RiskComponentScore:
    """
    Banking Behaviour: bounces, penalties, and overdraft.

    WHY this matters: A bounce is the strongest negative signal short of
    fraud. Banks report bounces to CIBIL. Even one is a yellow flag.
    """
    bounce_keywords = ["BOUNCE", "DISHON", "RETURN", "UNPAID"]
    bounce_count = sum(
        1 for t in txns
        if any(k in t.description.upper() for k in bounce_keywords)
    )

    penalty_amount = sum(
        t.debit or 0 for t in txns
        if t.level2 == "BANK_CHARGES"
    )

    od_days = sum(1 for t in txns if t.balance is not None and t.balance < 0)

    score = 850  # start high — clean record is the norm
    score -= bounce_count * 200      # bounces are very bad
    score -= min(penalty_amount / 50, 200)  # penalties matter
    score -= od_days * 75

    return RiskComponentScore(
        name="Banking Behaviour", weight=config.RISK_WEIGHTS["banking_behaviour"],
        score=round(max(0, min(1000, score))),
        details={
            "bounce_count": bounce_count,
            "penalty_charges": round(penalty_amount, 2),
            "overdraft_days": od_days,
        },
    )


def _score_expenses(txns: list[Transaction], avg_monthly_income: float) -> RiskComponentScore:
    """
    Expense Management: essential vs discretionary spending.

    WHY this matters: Someone spending 95% of income on discretionary
    items has no buffer for loan repayment.
    """
    debits = [t for t in txns if t.level1 == "DEBIT"]
    total_expense = sum(t.debit or 0 for t in debits)

    if total_expense == 0:
        return RiskComponentScore(
            name="Expense Management", weight=config.RISK_WEIGHTS["expense_management"],
            score=700, details={"reason": "No expenses recorded"},
        )

    essential = sum(t.debit or 0 for t in debits if t.is_essential)
    ess_ratio = essential / total_expense
    disc_ratio = 1 - ess_ratio

    # Monthly expense trend
    monthly_exp: dict[str, float] = defaultdict(float)
    for t in debits:
        monthly_exp[t.month] += t.debit or 0

    months = sorted(monthly_exp.keys())
    trend = "STABLE"
    if len(months) >= 2:
        vals = [monthly_exp[m] for m in months]
        if vals[-1] > vals[0] * 1.10:
            trend = "INCREASING"
        elif vals[-1] < vals[0] * 0.90:
            trend = "DECREASING"

    score = 650
    if ess_ratio >= 0.3:    # healthy essential spending
        score += 100
    if disc_ratio > 0.7:    # too much discretionary
        score -= 150
    if avg_monthly_income > 0:
        months_count = max(len(months), 1)
        monthly_avg_exp = total_expense / months_count
        if monthly_avg_exp > avg_monthly_income * 0.90:
            score -= 200     # spending nearly all income
    if trend == "INCREASING":
        score -= 50

    return RiskComponentScore(
        name="Expense Management", weight=config.RISK_WEIGHTS["expense_management"],
        score=round(max(0, min(1000, score))),
        details={
            "essential_ratio": round(ess_ratio, 3),
            "discretionary_ratio": round(disc_ratio, 3),
            "expense_trend": trend,
        },
    )


# ═══════════════════════════════════════════════════════════════════════
# DECISION ENGINE
# ═══════════════════════════════════════════════════════════════════════

def _decide(
    composite: int,
    rating: str,
    components: list[RiskComponentScore],
    txns: list[Transaction],
) -> tuple[str, list[str]]:
    """
    Generate final decision + rationale.

    Hard-decline triggers override the score.
    This is a simplified version — production would have more rules.
    """
    rationale = []

    # Hard decline: too many bounces
    banking = next((c for c in components if c.name == "Banking Behaviour"), None)
    if banking and banking.details.get("bounce_count", 0) >= 3:
        rationale.append("CRITICAL: 3+ payment bounces — high default risk")
        return "DECLINE", rationale

    income = next((c for c in components if c.name == "Income Stability"), None)

    if composite >= 700:
        rationale.append(f"Strong profile (score: {composite}, {rating})")
        if income:
            rationale.append(f"Avg monthly income: ₹{income.details.get('avg_monthly_income', 0):,.0f}")
        return "APPROVE", rationale

    elif composite >= 500:
        rationale.append(f"Moderate profile (score: {composite}, {rating})")
        if income and income.details.get("regularity", 1) < 0.7:
            rationale.append("Income regularity below threshold — shorter tenure recommended")
        if banking and banking.details.get("bounce_count", 0) > 0:
            rationale.append("Bounces on record — monitor closely")
        return "APPROVE_WITH_CONDITIONS", rationale

    elif composite >= 350:
        rationale.append(f"Weak profile (score: {composite}, {rating}) — manual review needed")
        return "REFER", rationale

    else:
        rationale.append(f"Poor profile (score: {composite}, {rating})")
        if income and income.details.get("avg_monthly_income", 0) == 0:
            rationale.append("No identifiable income source")
        return "DECLINE", rationale
