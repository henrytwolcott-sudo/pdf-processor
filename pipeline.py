"""
Document Intelligence Pipeline — Orchestrator
Ties together PDF processing, AI classification/extraction,
validation, and CSV output into a single run() call.

Production Features
───────────────────
• Concurrent page processing with ThreadPoolExecutor
• Per-document-type CSV output with auto-initialisation
• Comprehensive statistics and audit logging
• Graceful error handling — one bad page doesn't kill the batch
• Metadata enrichment (pdf_type, extraction_method, timestamps)
"""

import csv
import logging
import shutil
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from .config import PipelineConfig, DOCUMENT_SCHEMAS
from .pdf_processor import PDFProcessor, PageResult
from .ai_engine import AIEngine
from .validators import validate_extracted_data

logger = logging.getLogger(__name__)


class DocumentPipeline:
    """
    End-to-end document processing pipeline.
    
    Usage:
        from src.config import PipelineConfig
        from src.pipeline import DocumentPipeline
        
        cfg = PipelineConfig(input_dir="./my_pdfs", max_workers=4)
        pipeline = DocumentPipeline(cfg)
        pipeline.run()
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.cfg = config or PipelineConfig()

        # Create output directories
        Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cfg.temp_dir).mkdir(parents=True, exist_ok=True)

        # Sub-components
        self.processor = PDFProcessor(self.cfg)
        self.ai = AIEngine(self.cfg)

        # Statistics
        self.stats = {
            "pages_processed": 0,
            "pages_success": 0,
            "pages_failed": 0,
            "pages_low_confidence": 0,
            "by_type": {},
            "by_method": {},
        }

        # Initialise CSV files
        self._init_csvs()

    # ── CSV management ──────────────────────────────────────────────

    def _init_csvs(self):
        """Create CSV files with headers if they don't exist yet."""
        for doc_type, schema in DOCUMENT_SCHEMAS.items():
            path = Path(self.cfg.output_dir) / schema["csv_file"]
            if not path.exists():
                with open(path, "w", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=schema["csv_columns"]).writeheader()
                logger.debug(f"Created {path.name}")

    def _write_csv(self, document_type: str, data: Dict):
        """Append one record to the appropriate CSV."""
        schema = DOCUMENT_SCHEMAS.get(document_type)
        if not schema:
            return
        path = Path(self.cfg.output_dir) / schema["csv_file"]
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=schema["csv_columns"])
            row = {col: data.get(col, "") for col in schema["csv_columns"]}
            writer.writerow(row)

    # ── Single-page pipeline ────────────────────────────────────────

    def _process_single_page(
        self, page_path: Path, original_filename: str
    ) -> bool:
        """
        Full pipeline for one page:
          detect → extract text → classify → extract fields → validate → CSV
        """
        self.stats["pages_processed"] += 1

        # ── Step 1: PDF processing ──────────────────────────────────
        page_result: PageResult = self.processor.process_page(page_path)
        pdf_type = page_result.pdf_type
        method = page_result.extraction_method or "none"

        self.stats["by_type"][pdf_type] = self.stats["by_type"].get(pdf_type, 0) + 1
        self.stats["by_method"][method] = self.stats["by_method"].get(method, 0) + 1

        if page_result.error:
            logger.error(f"Extraction failed: {page_result.error}")
            self._write_csv("UNCLASSIFIED", {
                "original_filename": original_filename,
                "classification_confidence": 0,
                "extraction_note": f"Extraction failed: {page_result.error}",
                "pdf_type": pdf_type,
                "extraction_method": method,
                "processing_timestamp": datetime.now().isoformat(),
            })
            self.stats["pages_failed"] += 1
            return False

        # ── Step 2: Prepare content for Claude ──────────────────────
        if page_result.text_content:
            content = page_result.text_content
            fmt = "TEXT"
        elif page_result.base64_image:
            content = {
                "base64_image": page_result.base64_image,
                "media_type": "image/png",
            }
            fmt = "SCANNED"
        else:
            self.stats["pages_failed"] += 1
            return False

        # ── Step 3: AI classify + extract ───────────────────────────
        result = self.ai.classify_and_extract(content, page_path.name, fmt)

        # ── Step 4: Handle low confidence ───────────────────────────
        if result["status"] == "skipped_low_confidence":
            self._write_csv("UNCLASSIFIED", {
                "original_filename": original_filename,
                "classification_confidence": result["confidence"],
                "extraction_note": f"Low confidence: {result.get('reasoning', '')}",
                "pdf_type": pdf_type,
                "extraction_method": method,
                "processing_timestamp": datetime.now().isoformat(),
            })
            self.stats["pages_low_confidence"] += 1
            return True  # counted as processed, not failed

        # ── Step 5: Validate ────────────────────────────────────────
        doc_type = result["document_type"]
        extracted = result["extracted_data"]

        # Enrich with metadata
        extracted["file_source"] = original_filename
        extracted["page_source"] = page_path.name
        extracted["pdf_type"] = pdf_type
        extracted["extraction_method"] = method
        extracted["processing_timestamp"] = datetime.now().isoformat()

        is_valid, warnings = validate_extracted_data(doc_type, extracted)
        if warnings:
            logger.warning(f"Validation warnings for {page_path.name}: {warnings}")

        # ── Step 6: Write to CSV ────────────────────────────────────
        if is_valid or len(warnings) < 3:
            self._write_csv(doc_type, extracted)
            self.stats["pages_success"] += 1
            logger.info(f"✓ {page_path.name} → {doc_type}")
            return True
        else:
            logger.error(f"Validation failed for {page_path.name}: {warnings}")
            self.stats["pages_failed"] += 1
            return False

    # ── PDF-level processing ────────────────────────────────────────

    def process_pdf(self, pdf_path: Path) -> int:
        """
        Process one PDF file: split into pages, then process each page.
        Returns the number of successfully processed pages.
        """
        logger.info(f"{'─' * 60}")
        logger.info(f"Processing: {pdf_path.name}")

        pages = self.processor.split_pdf(pdf_path)
        logger.info(f"  {len(pages)} page(s)")

        successes = 0
        for i, page in enumerate(pages, 1):
            logger.info(f"  Page {i}/{len(pages)}")
            if self._process_single_page(page, pdf_path.name):
                successes += 1

        return successes

    # ── Directory-level processing ──────────────────────────────────

    def run(self, input_dir: Optional[str] = None) -> Dict:
        """
        Process every PDF in the input directory.
        
        Uses ThreadPoolExecutor for concurrent PDF-level processing.
        Each PDF is processed sequentially internally (pages depend on
        the split step) but multiple PDFs run in parallel.
        
        Returns:
            Summary statistics dict
        """
        src = Path(input_dir or self.cfg.input_dir)
        if not src.exists():
            raise FileNotFoundError(f"Input directory not found: {src}")

        pdfs = sorted(src.glob("*.pdf"))
        logger.info(f"Found {len(pdfs)} PDF(s) in {src}")

        if self.cfg.max_workers <= 1 or len(pdfs) <= 1:
            # Sequential processing
            for pdf in pdfs:
                self.process_pdf(pdf)
        else:
            # Concurrent processing
            with ThreadPoolExecutor(max_workers=self.cfg.max_workers) as pool:
                futures = {pool.submit(self.process_pdf, p): p for p in pdfs}
                for future in as_completed(futures):
                    pdf = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        logger.error(f"Failed to process {pdf.name}: {exc}")

        # Clean up temp pages
        temp = Path(self.cfg.temp_dir)
        if temp.exists():
            shutil.rmtree(temp, ignore_errors=True)

        self._print_summary()
        return self.stats

    # ── Reporting ───────────────────────────────────────────────────

    def _print_summary(self):
        s = self.stats
        ai = self.ai.get_stats()
        pdf = self.processor.get_stats()

        report = f"""
{'═' * 60}
  PIPELINE SUMMARY
{'═' * 60}
  Pages processed:     {s['pages_processed']}
  ├─ Successful:       {s['pages_success']}
  ├─ Low confidence:   {s['pages_low_confidence']}
  └─ Failed:           {s['pages_failed']}

  PDF Types:           {dict(sorted(s['by_type'].items()))}
  Extraction Methods:  {dict(sorted(s['by_method'].items()))}
  API Calls:           {ai['api_calls']}  (errors: {ai['api_errors']})

  Output directory:    {self.cfg.output_dir}
{'═' * 60}"""
        logger.info(report)

        # Log CSV row counts
        for schema in DOCUMENT_SCHEMAS.values():
            path = Path(self.cfg.output_dir) / schema["csv_file"]
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    rows = sum(1 for _ in f) - 1
                if rows > 0:
                    logger.info(f"  {path.name}: {rows} record(s)")
