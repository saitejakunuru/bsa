"""
Configuration — all tunables in one place.

Design decision: Single config file, not scattered env vars.
Makes it easy for the interviewer to see what's configurable.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# LLM — Using Groq for fast, free inference
# Why Groq: blazing fast inference (~10x faster than OpenAI/Anthropic),
# generous free tier, runs open-source models (Llama, Mixtral).
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

# Pipeline
BATCH_SIZE = 1      # Pages per LLM call. Why 3? See README.
CONFIDENCE_THRESHOLD = 0.7

# Risk scoring weights — must sum to 1.0
# Decision: income_stability highest because it's the strongest
# predictor of repayment. Fraud at 15% but has hard-decline overrides.
RISK_WEIGHTS = {
    "income_stability": 0.30,
    "liquidity": 0.25,
    "expense_management": 0.20,
    "banking_behaviour": 0.25,
}
# NOTE: Debt Service and Fraud Indicators are simplified in this PoC.
# In production, all 6 components would have calibrated weights.

# Rating bands
RATING_BANDS = [
    (800, 1000, "Excellent"),
    (650, 799,  "Good"),
    (500, 649,  "Fair"),
    (350, 499,  "Below Average"),
    (0,   349,  "Poor"),
]
