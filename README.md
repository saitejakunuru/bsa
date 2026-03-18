# Bank Statement Analyzer (BSA) — Credit Risk Lens

AI-powered bank statement analysis for credit underwriting. Extracts transactions from PDF statements, classifies them, and produces a credit risk assessment.

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env        # add your GROQ_API_KEY
python main.py              # → http://localhost:8000
```

Upload a bank statement PDF and click **Analyze**.

---

## Architecture

```
PDF Upload → pdfplumber (text) → Groq LLM (structuring) → Rule Classifier → Risk Scorer → JSON + UI
```

```
bsa/
├── main.py          # FastAPI server + web UI (single file)
├── extractor.py     # PDF → text → LLM → structured transactions
├── classifier.py    # Rule-based transaction classification
├── scorer.py        # Credit risk scoring engine
├── models.py        # All Pydantic data models
├── config.py        # Configuration
├── requirements.txt
├── .env.example
├── output/          # Generated sample output
└── test_data/       # Test bank statement
```

**8 source files. ~700 lines of Python.** Intentionally flat structure — nested packages with `__init__.py` files are overhead for a PoC.

---

## Design Decisions & Reasoning

### Extraction: pdfplumber + LLM (not pure regex, not pure LLM)

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| **Pure regex/tabula** | Fast, free, deterministic | Breaks on format changes, needs per-bank templates | ❌ Doesn't scale |
| **Pure LLM (send PDF to vision)** | Handles any format | Expensive, slow, wastes tokens on layout | ❌ Overkill |
| **pdfplumber + LLM (chosen)** | Free text extraction + LLM handles messy narrations | Needs API key | ✅ Best balance |

The assignment explicitly says "convert to processable format before passing to LLM" — so we extract text first (pdfplumber), then send that text to Groq (Llama 3.3 70B) for structured parsing.

**Production upgrade:** Per-bank template parsers (regex) for known formats. LLM only for unknown formats. This cuts API cost by 90%.

### Why Groq?

Groq runs inference on custom LPU (Language Processing Unit) hardware — roughly 10x faster than standard GPU-based providers. For a PoC where you're iterating quickly and testing many statements, this speed matters. The free tier gives 30 requests/minute on Llama 3.3 70B, which is plenty. The API is OpenAI-compatible, so swapping to GPT-4o or Claude in production is a one-line config change.

### Batching: ~3 Pages Per LLM Call

**Why 3?** A bank statement page is ~500-800 tokens. 3 pages ≈ 2K-2.5K tokens input. With the prompt and output, each call is ~5K tokens total — well within limits and free on Groq. A 100-page statement becomes ~33 API calls instead of one massive call.

**Why not send everything at once?** A 100-page statement would be ~80K tokens. Quality degrades on long contexts (hallucination risk increases), and you lose the ability to parallelize. Batching also means one failed call doesn't kill the whole extraction.

**Production upgrade:** Async parallel batch processing with `asyncio.gather()`.

### Classification: Rules Only (No LLM Fallback)

**Why rules first?** Three reasons:
1. **Speed** — regex match is microseconds vs. seconds for an API call
2. **Cost** — zero API cost for ~80% of transactions
3. **Auditability** — when an underwriter asks "why is this SALARY?", you point to rule #4. LLM classification is a black box.

**Why no LLM fallback in this PoC?** Adds complexity (batch management, error handling, cost tracking) for marginal gain. The 20% unmatched transactions get flagged for manual review — that's acceptable for a PoC.

**Production upgrade:** LLM fallback for low-confidence transactions. Eventually, a fine-tuned classifier trained on labeled transaction data.

### Risk Scoring: 4 of 6 Components

I implemented the 4 components with the highest underwriting signal and documented why the other 2 are deferred:

| Component | Weight | Status | Reasoning |
|-----------|--------|--------|-----------|
| Income Stability | 30% | ✅ Implemented | #1 predictor of repayment ability |
| Liquidity | 25% | ✅ Implemented | Cash buffer = shock absorber |
| Banking Behaviour | 25% | ✅ Implemented | Bounces directly predict default |
| Expense Management | 20% | ✅ Implemented | Spending discipline matters |
| Debt Service | — | ⏳ Simplified | Needs 6+ months for reliable FOIR. With 2-3 months, EMI detection is noisy |
| Fraud Indicators | — | ⏳ Simplified | Circular transaction detection needs graph analysis. Balance arithmetic check is included in extraction validation |

**Why not implement all 6?** Better to do 4 well than 6 poorly. The assignment says "focus on ones with highest underwriting impact" — that's what I did.

### Technology Choices

| Choice | Why | Alternative Considered |
|--------|-----|----------------------|
| FastAPI | Auto-docs, async, Pydantic integration | Flask — simpler but no auto-docs |
| pdfplumber | Best Python table/text extraction | PyPDF2 (worse tables), Tabula (Java dep) |
| Groq + Llama 3.3 70B | ~10x faster inference, free tier, strong JSON output | Claude/GPT-4o — better quality but slower and expensive. Groq is ideal for PoC iteration speed |
| Pydantic v2 | Validation + serialization in one | dataclasses — no validation |
| Inline HTML UI | Zero build step, iterate fast | React — overkill for PoC |

---

## What I'd Do Differently in Production

1. **Async pipeline** — Celery + Redis for long documents. Webhook callback instead of synchronous response.
2. **Per-bank parsers** — regex templates for SBI, HDFC, ICICI etc. LLM only for unknown formats. Cuts API cost 90%.
3. **ML classifier** — Train on labeled transaction data. Rule engine as fallback.
4. **Score calibration** — Current thresholds are heuristic. Production needs calibration against actual loan performance (default rates by score band).
5. **OCR pipeline** — Tesseract for scanned PDFs. Not implemented because most Indian bank statements are digital.
6. **Multi-account handling** — Current PoC assumes one account per file. Production needs account-number-based separation.
7. **Audit trail** — Store every extraction, classification decision, and confidence score for regulatory compliance.

---

## Classification Rules

The classifier uses 24 regex-based rules covering standard Indian banking patterns:

**High confidence (>0.90):** ATM_WITHDRAWAL, INTEREST_INCOME, GOVERNMENT_CREDIT, SALARY, EMI, BANK_CHARGES

**Medium confidence (0.80-0.90):** UTILITY, INSURANCE, RENT, EDUCATION, HEALTHCARE, FUEL, FOOD_DINING, GROCERY

**Lower confidence (0.60-0.78):** CARD_PAYMENT (Paytm/PhonePe), P2P_RECEIVED (UPI/IMPS), generic NEFT credits

Transactions below 0.7 confidence are flagged for manual review. Custom taxonomy CSV can be uploaded to override/extend rules.

### Important Classification Rules (from assignment requirements)
- **P2P ≠ Salary**: IMPS/UPI credits from individuals → P2P_RECEIVED, not SALARY
- **Refunds ≠ Income**: Merchant refunds → REFUND, not income category
- **ONE97 = Paytm**: ONE97 COMMUNICATIONS is Paytm's entity → CARD_PAYMENT, not P2P
- **Treasury = Government**: NEFT from Treasury Office → GOVERNMENT_CREDIT (salary/pension)

---

## Test Data

**Included:** SBI savings account statement, Sep–Nov 2019, 3 pages, 37 transactions.

**Patterns observed:**
- Regular government salary (₹23,850/month via NEFT from Treasury Office)
- Heavy Paytm usage (multiple ₹2,000 card transactions)
- ATM withdrawals at Makrana and Phulera
- P2P transfers via IMPS/UPI
- One interest credit

**Limitations:**
- Only digital PDF tested (no scanned/OCR)
- No EMI/loan payments in test data
- No bounces or penalties
- Short period (3 months) — risk scoring more reliable with 6-12 months

---

## Assumptions

1. **Currency:** INR unless stated otherwise in statement header
2. **Treasury Office = Government salary/pension** (standard in India)
3. **ONE97 COMMUNICATIONS = Paytm merchant payments** (not P2P)
4. **Single account per file** in this PoC
5. **Chronological order** expected within pages
6. **Confidence 0.7** as review threshold — would calibrate with labeled data in production

---

## API

```bash
# Health check
curl localhost:8000/api/health

# Analyze
curl -X POST localhost:8000/api/analyze -F "file=@test_data/bank-statement.pdf"

# Auto-generated docs
open localhost:8000/docs
```
