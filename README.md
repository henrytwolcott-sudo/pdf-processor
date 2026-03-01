# Document Intelligence Pipeline

AI-powered PDF classification and structured data extraction for legal document processing.

Automatically detects PDF types (scanned, text, glyph-encoded, hybrid), extracts content using the optimal method, classifies documents via Claude AI, and outputs structured data to CSV.

---

## The Problem

Legal firms process thousands of PDFs monthly — invoices, contracts, subscriptions, receipts — with manual data entry that costs **100+ hours/month**, has a **15-20% error rate**, and runs **~$12,500/month**.

## The Solution

An automated pipeline that:

1. **Detects** PDF type (5 categories) with glyph-encoded text detection
2. **Routes** to the optimal extraction method (PyMuPDF or Tesseract OCR)
3. **Classifies** document type using Claude AI
4. **Extracts** structured fields into type-specific schemas
5. **Validates** and exports to CSV

**Results:** 95%+ accuracy, 10-15 sec/page, 99.6% cost reduction.

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/youruser/document-intelligence-pipeline.git
cd document-intelligence-pipeline
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# Run
mkdir -p input_pdfs
# Drop your PDF files into input_pdfs/
python -c "
from src import DocumentPipeline, PipelineConfig
pipeline = DocumentPipeline(PipelineConfig(input_dir='./input_pdfs'))
pipeline.run()
"
```

Results appear in `output/` as CSV files (one per document type).

---

## Architecture

```
PDF Input
  │
  ▼
┌─────────────────────────────────────────┐
│  1. SPLIT                               │
│  Multi-page PDFs → single pages         │
└────────────────┬────────────────────────┘
                 ▼
┌─────────────────────────────────────────┐
│  2. DETECT PDF TYPE                     │
│                                         │
│  text_normal  → PyMuPDF    (1-2 sec)    │
│  text_glyph   → Tesseract  (5-8 sec)   │
│  scanned      → Tesseract  (5-8 sec)   │
│  hybrid       → Tesseract  (5-8 sec)   │
│  empty        → error                   │
└────────────────┬────────────────────────┘
                 ▼
┌─────────────────────────────────────────┐
│  3. AI CLASSIFICATION (Claude)          │
│  → Invoice / Receipt / Contract /       │
│    Subscription / Other                 │
└────────────────┬────────────────────────┘
                 ▼
┌─────────────────────────────────────────┐
│  4. STRUCTURED EXTRACTION (Claude)      │
│  → Type-specific schema fields          │
└────────────────┬────────────────────────┘
                 ▼
┌─────────────────────────────────────────┐
│  5. VALIDATE → CSV OUTPUT               │
│  invoices.csv | receipts.csv |          │
│  contracts.csv | subscriptions.csv      │
└─────────────────────────────────────────┘
```

### Glyph Detection

The key differentiator: many PDFs have a text layer that produces garbage when extracted (custom font encodings, symbol mappings). The pipeline measures the **alphanumeric ratio** of extracted text — if it falls below the threshold (default 0.30), the page is routed through OCR instead, preventing wasted LLM tokens.

---

## Configuration

All settings are in one place via `PipelineConfig`:

```python
from src import PipelineConfig, DocumentPipeline

cfg = PipelineConfig(
    input_dir="./my_pdfs",
    output_dir="./results",
    glyph_ratio_threshold=0.25,   # more aggressive glyph detection
    force_ocr=False,              # True to always use Tesseract
    max_workers=4,                # concurrent pages (across all PDFs)
    confidence_threshold=0.80,    # stricter classification
    max_retries=3,                # API retry attempts
)

pipeline = DocumentPipeline(cfg)
pipeline.run()
```

---

## Document Types Supported

| Type | Extracted Fields |
|------|-----------------|
| **Invoice** | invoice_id, date, client, vendor, amount, currency, status, payment_method |
| **Receipt** | receipt_id, date, vendor, amount, currency, payment_method, items, category |
| **Subscription** | subscription_id, date, service, email, amount, billing_cycle, next_date |
| **Contract** | contract_id, date, parties, value, type, start/end dates, key_terms |

Adding a new type takes ~20 minutes: add a schema to `config.py` and update the classification prompt.

---

## Project Structure

```
├── src/
│   ├── config.py           # Settings, schemas, prompts
│   ├── pdf_processor.py    # Split → Detect → Extract text
│   ├── ai_engine.py        # Claude API classify → extract
│   ├── validators.py       # Post-extraction validation
│   └── pipeline.py         # Orchestrator
│
├── notebooks/
│   ├── pipeline_demo.ipynb # Interactive walkthrough
│   └── architecture.ipynb  # Design decisions & scaling
│
├── requirements.txt
└── .env.example
```

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| PDF Processing | PyMuPDF | Fast text extraction for clean PDFs |
| OCR | Tesseract 5 | Scanned, glyph, and hybrid PDFs |
| LLM | Claude API (Anthropic) | Classification & structured extraction |
| Language | Python 3.11 | Core logic |
| Concurrency | ThreadPoolExecutor | Parallel page-level processing |

---

## Token Optimisation

| Strategy | Impact |
|----------|--------|
| Glyph detection routes bad text to OCR | Prevents garbage tokens to LLM |
| Text extraction before vision API | Text calls ~10x cheaper than image |
| Separate classify/extract with tight max_tokens | Avoids over-generation |
| Type-specific schemas | Only relevant fields per call |

---

## License

MIT
