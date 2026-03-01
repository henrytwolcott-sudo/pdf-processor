"""
Configuration & Schemas for Document Intelligence Pipeline
All tuneable parameters, prompts, document schemas, and CSV mappings live here.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()

# ── Runtime settings ────────────────────────────────────────────────
@dataclass
class PipelineConfig:
    """Single source of truth for every tuneable knob."""

    # Directories
    input_dir: str = "./input_pdfs"
    output_dir: str = "./output"
    temp_dir: str = "./output/temp_pages"

    # PDF processing
    glyph_ratio_threshold: float = 0.30
    min_text_length: int = 50
    ocr_dpi: int = 300
    vision_fallback_dpi: int = 200
    force_ocr: bool = False

    # Claude API
    api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    model_classify: str = "claude-sonnet-4-20250514"
    model_extract: str = "claude-sonnet-4-20250514"
    max_tokens_classify: int = 300
    max_tokens_extract: int = 1500
    confidence_threshold: float = 0.70

    # Concurrency (production scaling)
    max_workers: int = 4
    api_rate_limit_rpm: int = 60
    batch_size: int = 20
    max_retries: int = 3
    retry_backoff: float = 2.0

    # Logging
    log_level: str = "INFO"
    log_file: str = "./output/pipeline.log"


# ── Document schemas ────────────────────────────────────────────────
# Each schema defines: extraction prompt fields, required fields for
# validation, CSV column order, and the output filename.

DOCUMENT_SCHEMAS: Dict[str, dict] = {
    "INVOICE": {
        "extraction_prompt": """INVOICE Fields:
- invoice_id (string): Unique invoice identifier
- invoice_date (date): YYYY-MM-DD
- client_name (string): Company/individual being invoiced
- vendor_name (string): Company issuing invoice
- total_amount (float): Total amount due
- currency (string): 3-letter ISO code
- invoice_status (string): Paid|Unpaid|Partial|Unknown
- payment_method (string): Payment method or null
- description (string): Services/goods description
- file_source (string): Original PDF filename""",
        "required_fields": ["invoice_date", "total_amount", "currency"],
        "csv_columns": [
            "invoice_id", "invoice_date", "client_name", "vendor_name",
            "total_amount", "currency", "invoice_status", "payment_method",
            "description", "file_source", "page_source", "pdf_type",
            "extraction_method", "processing_timestamp",
        ],
        "csv_file": "invoices.csv",
    },
    "RECEIPT": {
        "extraction_prompt": """RECEIPT Fields:
- receipt_id (string): Transaction/receipt number
- receipt_date (date): YYYY-MM-DD
- vendor_name (string): Business name
- total_amount (float): Amount paid
- currency (string): 3-letter ISO code
- payment_method (string): Cash|Card|Wallet|Other
- items (string): Comma-separated purchased items
- category (string): Ride|Food|Office|Retail|Other
- file_source (string): Original PDF filename""",
        "required_fields": ["receipt_date", "total_amount", "currency"],
        "csv_columns": [
            "receipt_id", "receipt_date", "vendor_name", "total_amount",
            "currency", "payment_method", "items", "category", "file_source",
            "page_source", "pdf_type", "extraction_method", "processing_timestamp",
        ],
        "csv_file": "receipts.csv",
    },
    "SUBSCRIPTION": {
        "extraction_prompt": """SUBSCRIPTION Fields:
- subscription_id (string): Subscription account number
- invoice_date (date): YYYY-MM-DD
- service_name (string): Subscription service name
- billed_to_email (string): Email address on account
- subscription_amount (float): Charge per cycle
- currency (string): 3-letter ISO code
- payment_method (string): Payment method
- billing_cycle (string): Monthly|Annual|Quarterly|Other
- next_billing_date (date): YYYY-MM-DD or null
- file_source (string): Original PDF filename""",
        "required_fields": ["invoice_date", "subscription_amount", "currency"],
        "csv_columns": [
            "subscription_id", "invoice_date", "service_name", "billed_to_email",
            "subscription_amount", "currency", "payment_method", "billing_cycle",
            "next_billing_date", "file_source", "page_source", "pdf_type",
            "extraction_method", "processing_timestamp",
        ],
        "csv_file": "subscriptions.csv",
    },
    "CONTRACT": {
        "extraction_prompt": """CONTRACT Fields:
- contract_id (string): Contract number/reference
- contract_date (date): YYYY-MM-DD execution date
- party_1_name (string): First party name
- party_2_name (string): Second party name
- contract_value (float): Monetary value or null
- contract_type (string): Service Agreement|NDA|Lease|Employment|Purchase|Other
- start_date (date): Contract start date or null
- end_date (date): Contract end date or null
- key_terms (string): 3-5 key terms separated by |
- file_source (string): Original PDF filename""",
        "required_fields": ["contract_date", "party_1_name", "party_2_name"],
        "csv_columns": [
            "contract_id", "contract_date", "party_1_name", "party_2_name",
            "contract_value", "contract_type", "start_date", "end_date",
            "key_terms", "file_source", "page_source", "pdf_type",
            "extraction_method", "processing_timestamp",
        ],
        "csv_file": "contracts.csv",
    },
    "UNCLASSIFIED": {
        "extraction_prompt": "",
        "required_fields": ["original_filename"],
        "csv_columns": [
            "original_filename", "classification_confidence", "extraction_note",
            "pdf_type", "extraction_method", "processing_timestamp",
        ],
        "csv_file": "unclassified.csv",
    },
}


# ── System prompts ──────────────────────────────────────────────────

CLASSIFICATION_SYSTEM_PROMPT = """\
You are a document classification specialist. Analyze the document and classify it.

CLASSIFICATION RULES:
- Classify as ONE of: INVOICE, RECEIPT, SUBSCRIPTION, CONTRACT, EMAIL, OTHER
- Return confidence score 0.0-1.0
- If confidence < 0.7, classify as UNCLASSIFIED

PRIORITY RULES:
- INVOICE has priority over RECEIPT.
- If the document contains "Invoice" in the title/header/body — even with
  "PAID" stamps — classify as INVOICE.
- RECEIPT must NOT contain "invoice" and must explicitly function as proof of
  payment (e.g. "Receipt", "Payment Confirmation").

INDICATORS:
- INVOICE: "Invoice", line items, due date, invoice number, amount owed
- RECEIPT: "Receipt", transaction ID, "Thank you for your purchase"
- SUBSCRIPTION: recurring charges, renewal dates, plan descriptions
- CONTRACT: legal terms, signatures, obligations, deliverables
- EMAIL: headers (From/To/Subject), conversational style
- OTHER: does not fit any above

Respond with ONLY valid JSON:
{
  "document_type": "...",
  "confidence": 0.00,
  "reasoning": "..."
}"""

EXTRACTION_SYSTEM_PROMPT = """\
You are a data extraction specialist. Extract structured fields from documents.

RULES:
- Extract ONLY fields visible in the document — NEVER hallucinate
- Use null for missing fields
- Amounts: numeric only, 2 decimals (e.g. 36.75 not $36.75)
- Currency: 3-letter ISO code (CAD, USD, EUR, GBP …)
- Dates: YYYY-MM-DD
- Trim all text and normalise whitespace

Return ONLY valid JSON:
{
  "extraction": { ... },
  "extraction_notes": {
    "missing_fields": [],
    "warnings": []
  }
}"""


# ── Valid currencies ────────────────────────────────────────────────
VALID_CURRENCIES = {
    "USD", "CAD", "EUR", "GBP", "JPY", "AUD", "CHF", "CNY", "INR",
    "MXN", "BRL", "ZAR", "SGD", "HKD", "NZD", "SEK", "NOK", "DKK",
}
