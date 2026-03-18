"""
Data models for the Bank Statement Analyzer.
All models in one file — keeps things simple for a PoC.
"""

from __future__ import annotations
from datetime import date
from typing import Optional
from pydantic import BaseModel, Field


# ── Account Metadata ───────────────────────────────────────────────────

class AccountMeta(BaseModel):
    account_number: str = ""
    account_holder: str = ""
    bank_name: str = ""
    branch: str = ""
    ifsc_code: str = ""
    currency: str = "INR"
    statement_from: Optional[date] = None
    statement_to: Optional[date] = None
    opening_balance: Optional[float] = None


# ── Single Transaction ─────────────────────────────────────────────────

class Transaction(BaseModel):
    txn_date: date
    value_date: Optional[date] = None
    description: str
    ref_no: Optional[str] = ""
    debit: Optional[float] = None
    credit: Optional[float] = None
    balance: Optional[float] = None
    counterparty: str = "UNKNOWN"
    payment_channel: str = "OTHER"  # ATM, NEFT, UPI, IMPS, CARD, CHEQUE, etc.
    currency: str = "INR"
    month: str = ""  # YYYY-MM for grouping

    # Classification (filled by classifier)
    level1: str = ""         # CREDIT or DEBIT
    level2: str = ""         # Granular category
    confidence: float = 0.0
    method: str = "RULE"     # RULE, LLM, MANUAL
    needs_review: bool = False
    is_essential: bool = False
    is_fixed: bool = False


# ── Risk Scoring ───────────────────────────────────────────────────────

class RiskComponentScore(BaseModel):
    name: str
    weight: float
    score: float = 0.0  # 0-1000
    details: dict = Field(default_factory=dict)


class CreditRiskSummary(BaseModel):
    customer_name: str = ""
    account_number: str = ""
    analysis_period: str = ""

    components: list[RiskComponentScore] = Field(default_factory=list)
    composite_score: int = 0          # 0-1000
    rating_band: str = "Poor"         # Excellent/Good/Fair/Below Average/Poor
    decision: str = "REFER"           # APPROVE / APPROVE_WITH_CONDITIONS / DECLINE / REFER
    rationale: list[str] = Field(default_factory=list)

    total_credits: float = 0.0
    total_debits: float = 0.0
    net_cashflow: float = 0.0
    transaction_count: int = 0


# ── Full Pipeline Output ───────────────────────────────────────────────

class AnalysisResult(BaseModel):
    account: AccountMeta
    transactions: list[Transaction]
    risk_summary: CreditRiskSummary
    warnings: list[str] = Field(default_factory=list)
    page_count: int = 0
