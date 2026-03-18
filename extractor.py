"""
Statement Extractor
====================
PDF/Image → raw text (pdfplumber) → structured transactions (Groq LLM)

Design decisions documented inline. This is the most important module.

WHY this approach:
  - pdfplumber for text extraction: free, fast, no API cost.
    Alternative considered: sending PDF directly to a vision API.
    Rejected because: expensive, slow, and the assignment explicitly asks
    to "convert to processable format before passing to an LLM."

  - LLM for structuring: bank statement formats vary across banks.
    Writing per-bank regex parsers doesn't scale. The LLM handles messy
    narration text, date format variations, and layout differences.

  - Groq for inference: ~10x faster than OpenAI/Anthropic, generous free
    tier, runs Llama 3.3 70B which is strong at structured output.
    Alternative considered: Anthropic Claude — better quality but slower
    and more expensive. Groq is ideal for a PoC where speed matters.

  - Batching at ~3 pages: keeps each LLM call under ~6K tokens.
    A 100-page statement would be ~200K tokens in one call — too expensive
    and risks quality degradation on long contexts.

PRODUCTION UPGRADES:
  - Per-bank template parsers (regex) with LLM only for unknown formats
  - Async batch processing with Celery for long documents
  - Tesseract OCR fallback for scanned PDFs (not implemented in PoC)
  - Retry logic with exponential backoff on API failures
"""

from __future__ import annotations
import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pdfplumber
from groq import Groq

import config
from models import AccountMeta, Transaction

logger = logging.getLogger(__name__)

# ── Minimum text to consider a page "readable" ────────────────────────
MIN_TEXT_CHARS = 40


# ════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ════════════════════════════════════════════════════════════════════════

def extract_from_file(file_path: str | Path) -> tuple[AccountMeta, list[Transaction], int, list[str]]:
    """
    Full extraction pipeline:
      1. PDF → page texts (pdfplumber)
      2. Page texts → batches
      3. Each batch → Groq LLM → structured JSON → Transaction objects

    Returns: (account_meta, transactions, page_count, warnings)
    """
    file_path = Path(file_path)
    warnings: list[str] = []

    # Step 1: Extract raw text from PDF
    page_texts = _pdf_to_text(file_path, warnings)
    page_count = len(page_texts)
    logger.info(f"Extracted text from {page_count} pages")
    print(page_texts)
    
    if not page_texts:
        raise ValueError("Could not extract any text from the file. Is it a scanned PDF?")

    # Step 2: Batch pages
    batches = [
        page_texts[i:i + config.BATCH_SIZE]
        for i in range(0, len(page_texts), config.BATCH_SIZE)
    ]
    logger.info(f"Created {len(batches)} batch(es) of ~{config.BATCH_SIZE} pages")

    # Step 3: Send to LLM for structured extraction
    client = _get_llm_client()
    account_meta = AccountMeta()
    all_transactions: list[Transaction] = []

    for batch_idx, batch in enumerate(batches):
        batch_text = "\n\n--- PAGE BREAK ---\n\n".join(batch)

        try:
            meta, txns = _extract_batch(client, batch_text, batch_idx, account_meta)

            # Inherit metadata from first batch (later pages may not have headers)
            if batch_idx == 0:
                account_meta = meta

            # Enrich transactions with account-level info
            for t in txns:
                t.currency = account_meta.currency
                t.month = t.txn_date.strftime("%Y-%m")

            all_transactions.extend(txns)
            logger.info(f"Batch {batch_idx}: {len(txns)} transactions extracted")

        except Exception as e:
            warnings.append(f"Batch {batch_idx} failed: {str(e)}")
            logger.error(f"Batch {batch_idx} extraction error: {e}")

    # Step 4: Basic validation — check date order
    for i in range(1, len(all_transactions)):
        if all_transactions[i].txn_date < all_transactions[i-1].txn_date:
            warnings.append(
                f"Date order issue: {all_transactions[i-1].txn_date} → {all_transactions[i].txn_date}"
            )
            break  # warn once, don't spam

    logger.info(f"Extraction complete: {len(all_transactions)} transactions, {len(warnings)} warnings")
    return account_meta, all_transactions, page_count, warnings


# ════════════════════════════════════════════════════════════════════════
# INTERNAL — PDF TEXT EXTRACTION
# ════════════════════════════════════════════════════════════════════════

def _pdf_to_text(file_path: Path, warnings: list[str]) -> list[str]:
    """
    Extract text from each page using pdfplumber.

    Why pdfplumber over alternatives:
      - PyPDF2/pypdf: worse at table extraction, loses layout
      - Tabula: Java dependency, overkill for text extraction
      - pdfplumber: best balance of accuracy + simplicity for Python
    """
    pages = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if len(text.strip()) < MIN_TEXT_CHARS:
                warnings.append(f"Page {i+1}: very little text extracted (possibly scanned)")
                # PRODUCTION: fall back to pytesseract OCR here
            pages.append(text)
    return pages


# ════════════════════════════════════════════════════════════════════════
# INTERNAL — LLM EXTRACTION (Groq)
# ════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a bank statement parser. Extract structured data from bank statement text.

Rules:
1. Extract EVERY transaction — never skip rows.
2. Dates: YYYY-MM-DD format.
3. Amounts: plain numbers, no commas or currency symbols.
4. Counterparty is MANDATORY — infer from narration:
   - IMPS/NEFT/UPI: extract person/entity name
   - Card payments: extract merchant (e.g., "Paytm", "Amazon")
   - ATM: "ATM" + location
   - Interest: "BANK"
   - If unidentifiable: "UNKNOWN"
5. payment_channel: one of IMPS, NEFT, UPI, CARD, ATM, CHEQUE, RTGS, INTEREST, OTHER
6. For null fields use null.

Respond with valid JSON only. No markdown fences, no commentary."""


def _build_prompt(batch_text: str, batch_idx: int, meta: AccountMeta) -> str:
    """Build extraction prompt for a batch."""

    meta_hint = ""
    if meta.account_number:
        meta_hint = f"""
Known context (inherit if not on this page):
- Account: {meta.account_number} | Bank: {meta.bank_name} | Currency: {meta.currency}
"""

    return f"""Extract all data from this bank statement text (batch {batch_idx}).
{meta_hint}
STATEMENT TEXT:
---
{batch_text}
---

Return JSON:
{{
  "account_meta": {{
    "account_number": "string or null",
    "account_holder": "string or null",
    "bank_name": "string or null",
    "branch": "string or null",
    "ifsc_code": "string or null",
    "currency": "INR",
    "statement_from": "YYYY-MM-DD or null",
    "statement_to": "YYYY-MM-DD or null",
    "opening_balance": number or null
  }},
  "transactions": [
    {{
      "txn_date": "YYYY-MM-DD",
      "value_date": "YYYY-MM-DD or null",
      "description": "raw narration",
      "ref_no": "string",
      "debit": number or null,
      "credit": number or null,
      "balance": number or null,
      "counterparty": "extracted name",
      "payment_channel": "IMPS|NEFT|UPI|CARD|ATM|INTEREST|OTHER"
    }}
  ]
}}"""


def _extract_batch(
    client: Groq,
    batch_text: str,
    batch_idx: int,
    existing_meta: AccountMeta,
) -> tuple[AccountMeta, list[Transaction]]:
    """Send one batch to Groq LLM, parse response into models."""

    # Groq uses OpenAI-compatible chat completions API
    messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_prompt(batch_text, batch_idx, existing_meta)},
        ]

        # Try JSON mode first, fall back to raw mode if Groq rejects it
    try:
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=8192,
            response_format={"type": "json_object"},
        )
        raw_text = response.choices[0].message.content
        raw = _parse_json(raw_text)
    except Exception as e:
        logger.warning(f"Batch {batch_idx}: JSON mode failed ({e}), retrying without it")
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=8192,
        )
        raw_text = response.choices[0].message.content
        raw = _parse_json(raw_text)
    # Parse account metadata (only from first batch usually)
    meta_raw = raw.get("account_meta", {})
    meta = AccountMeta(
        account_number=meta_raw.get("account_number") or existing_meta.account_number,
        account_holder=meta_raw.get("account_holder") or existing_meta.account_holder,
        bank_name=meta_raw.get("bank_name") or existing_meta.bank_name,
        branch=meta_raw.get("branch") or existing_meta.branch,
        ifsc_code=meta_raw.get("ifsc_code") or existing_meta.ifsc_code,
        currency=meta_raw.get("currency") or "INR",
        statement_from=_parse_date(meta_raw.get("statement_from")) or existing_meta.statement_from,
        statement_to=_parse_date(meta_raw.get("statement_to")) or existing_meta.statement_to,
        opening_balance=_to_float(meta_raw.get("opening_balance")) or existing_meta.opening_balance,
    )

    # Parse transactions
    transactions = []
    for t in raw.get("transactions", []):
        txn = Transaction(
            txn_date=_parse_date(t.get("txn_date")) or date.today(),
            value_date=_parse_date(t.get("value_date")),
            description=t.get("description", ""),
            ref_no=t.get("ref_no", ""),
            debit=_to_float(t.get("debit")),
            credit=_to_float(t.get("credit")),
            balance=_to_float(t.get("balance")),
            counterparty=t.get("counterparty", "UNKNOWN"),
            payment_channel=t.get("payment_channel", "OTHER"),
        )
        transactions.append(txn)

    return meta, transactions


# ════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════

def _get_llm_client() -> Groq:
    """
    Create Groq client.

    Why Groq over Anthropic/OpenAI:
      - Speed: Groq runs on custom LPU hardware, ~10x faster inference
      - Cost: generous free tier (30 req/min on Llama 3.3 70B)
      - Quality: Llama 3.3 70B is strong at structured JSON output
      - API: OpenAI-compatible, easy to swap to other providers later
    """
    if not config.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set. Add it to .env file.")
    return Groq(api_key=config.GROQ_API_KEY)


def _parse_json(text: str) -> dict:
    """Parse LLM response, stripping markdown fences if present."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())


def _parse_date(val: Any) -> date | None:
    if not val:
        return None
    try:
        return datetime.strptime(str(val), "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _to_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None
