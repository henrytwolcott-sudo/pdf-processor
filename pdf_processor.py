"""
PDF Processing Engine
Handles the entire pre-AI layer: split → detect type → extract text.

PDF Type Detection & Routing Strategy
──────────────────────────────────────
 Type          │ Detection                              │ Extraction
 ──────────────┼────────────────────────────────────────┼──────────────
 scanned       │ No text layer, has images              │ Tesseract OCR
 text_normal   │ Clean selectable text, high α-ratio    │ PyMuPDF (fast)
 text_glyph    │ Text layer exists but α-ratio < 0.30   │ Tesseract OCR
 hybrid        │ Both text and image layers present      │ Tesseract OCR
 empty         │ No text, no images                      │ → error

The glyph detection is the key differentiator: many PDFs appear to have
extractable text but actually use custom font encodings that produce
garbage characters. By measuring the alphanumeric ratio we catch these
and route them through OCR instead, saving tokens on the AI call.
"""

import re
import io
import base64
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

from .config import PipelineConfig

logger = logging.getLogger(__name__)


# ── Data containers ─────────────────────────────────────────────────

@dataclass
class PDFAnalysis:
    """Result of PDF type detection."""
    pdf_type: str                       # scanned | text_normal | text_glyph | hybrid | empty
    has_text: bool = False
    has_images: bool = False
    text_length: int = 0
    image_count: int = 0
    alphanumeric_ratio: float = 0.0
    readable_words: int = 0
    sample_text: str = ""
    error: str = ""


@dataclass
class PageResult:
    """Output from processing a single page."""
    source_file: str
    pdf_type: str
    analysis: PDFAnalysis
    extraction_method: Optional[str] = None
    text_content: Optional[str] = None
    base64_image: Optional[str] = None
    error: Optional[str] = None
    notes: List[str] = field(default_factory=list)


# ── Processor ───────────────────────────────────────────────────────

class PDFProcessor:
    """
    Core PDF processing engine.
    
    Responsibilities:
    1. Split multi-page PDFs into single pages
    2. Detect PDF type (5 categories) with glyph analysis
    3. Route to the optimal extraction method
    4. Provide a vision-API fallback (base64 image) if text extraction fails
    
    All configuration is injected via PipelineConfig, making this class
    easy to test and tune without touching business logic.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.cfg = config or PipelineConfig()
        self.stats = {
            "scanned": 0, "text_normal": 0, "text_glyph": 0,
            "hybrid": 0, "empty": 0, "errors": 0,
        }

    # ── 1. Split ────────────────────────────────────────────────────

    def split_pdf(self, pdf_path: Path) -> List[Path]:
        """
        Split a multi-page PDF into individual single-page PDFs.
        
        Single-page inputs are returned as-is (no temp file created).
        Output files are written to cfg.temp_dir for easy cleanup.
        """
        doc = fitz.open(pdf_path)
        num_pages = len(doc)

        if num_pages <= 1:
            doc.close()
            logger.debug(f"{pdf_path.name}: single page — no split needed")
            return [pdf_path]

        temp = Path(self.cfg.temp_dir)
        temp.mkdir(parents=True, exist_ok=True)

        pages: List[Path] = []
        for i in range(num_pages):
            out = temp / f"{pdf_path.stem}_page_{i + 1}.pdf"
            new_doc = fitz.open()
            new_doc.insert_pdf(doc, from_page=i, to_page=i)
            new_doc.save(out)
            new_doc.close()
            pages.append(out)

        doc.close()
        logger.info(f"{pdf_path.name}: split into {num_pages} pages")
        return pages

    # ── 2. Detect ───────────────────────────────────────────────────

    def detect_type(self, pdf_path: Path) -> PDFAnalysis:
        """
        Classify a single-page PDF into one of five categories.
        
        The glyph-detection algorithm works by computing the ratio of
        standard alphanumeric characters to total characters. A low ratio
        (< glyph_ratio_threshold) indicates custom/symbol font encodings
        that look like text to PyMuPDF but produce nonsense for an LLM.
        """
        try:
            doc = fitz.open(pdf_path)
            if len(doc) == 0:
                doc.close()
                return PDFAnalysis(pdf_type="empty", error="zero pages")

            page = doc[0]
            raw_text = page.get_text().strip()
            images = page.get_images(full=True)
            doc.close()

            has_text = len(raw_text) > 0
            has_images = len(images) > 0

            # ── Determine type ──────────────────────────────────────
            if not has_text and not has_images:
                pdf_type = "empty"
                analysis = PDFAnalysis(pdf_type=pdf_type)

            elif not has_text and has_images:
                pdf_type = "scanned"
                analysis = PDFAnalysis(
                    pdf_type=pdf_type, has_images=True,
                    image_count=len(images),
                )

            elif has_text and has_images:
                pdf_type = "hybrid"
                analysis = PDFAnalysis(
                    pdf_type=pdf_type, has_text=True, has_images=True,
                    text_length=len(raw_text), image_count=len(images),
                )

            else:  # has_text only — need glyph check
                clean = len(re.findall(r"[A-Za-z0-9\s]", raw_text))
                total = max(len(raw_text), 1)
                ratio = clean / total
                words = len(re.findall(r"\b[A-Za-z]{2,}\b", raw_text))

                if ratio < self.cfg.glyph_ratio_threshold or words < 5:
                    pdf_type = "text_glyph"
                    logger.info(
                        f"Glyph detected: ratio={ratio:.2f}, words={words}"
                    )
                else:
                    pdf_type = "text_normal"

                analysis = PDFAnalysis(
                    pdf_type=pdf_type, has_text=True,
                    text_length=len(raw_text),
                    alphanumeric_ratio=round(ratio, 3),
                    readable_words=words,
                    sample_text=raw_text[:120],
                )

            self.stats[pdf_type] = self.stats.get(pdf_type, 0) + 1
            logger.info(f"{pdf_path.name} → {pdf_type.upper()}")
            return analysis

        except Exception as exc:
            self.stats["errors"] += 1
            logger.error(f"Detection failed for {pdf_path}: {exc}")
            return PDFAnalysis(pdf_type="empty", error=str(exc))

    # ── 3. Extract ──────────────────────────────────────────────────

    def extract_text_pymupdf(self, pdf_path: Path) -> Optional[str]:
        """Fast direct text extraction for clean PDFs (1-2 sec)."""
        try:
            doc = fitz.open(pdf_path)
            text = doc[0].get_text().strip() if len(doc) else None
            doc.close()
            text = re.sub(r"\s+", " ", text) if text else None
            if text:
                logger.debug(f"PyMuPDF: {len(text)} chars from {pdf_path.name}")
            return text
        except Exception as exc:
            logger.error(f"PyMuPDF failed on {pdf_path}: {exc}")
            return None

    def extract_text_tesseract(self, pdf_path: Path) -> Optional[str]:
        """OCR extraction for scanned / glyph / hybrid PDFs (5-8 sec)."""
        try:
            doc = fitz.open(pdf_path)
            page = doc[0]
            zoom = self.cfg.ocr_dpi / 72
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            doc.close()

            logger.info(f"Tesseract OCR @ {self.cfg.ocr_dpi} DPI → {pdf_path.name}")
            text = pytesseract.image_to_string(img, lang="eng").strip()
            text = re.sub(r"\s+", " ", text) if text else None
            if text:
                logger.debug(f"Tesseract: {len(text)} chars from {pdf_path.name}")
            return text
        except Exception as exc:
            logger.error(f"Tesseract failed on {pdf_path}: {exc}")
            return None

    def convert_to_base64(self, pdf_path: Path) -> Optional[str]:
        """Vision-API fallback: render page as PNG → base64 string."""
        try:
            doc = fitz.open(pdf_path)
            page = doc[0]
            zoom = self.cfg.vision_fallback_dpi / 72
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            b64 = base64.standard_b64encode(pix.tobytes("png")).decode()
            doc.close()
            return b64
        except Exception as exc:
            logger.error(f"Base64 conversion failed for {pdf_path}: {exc}")
            return None

    # ── 4. Unified workflow ─────────────────────────────────────────

    def process_page(self, pdf_path: Path) -> PageResult:
        """
        Full single-page workflow: detect → extract → fallback.
        
        Routing table:
          scanned    → Tesseract
          text_normal→ PyMuPDF
          text_glyph → Tesseract
          hybrid     → Tesseract
          empty      → error
          force_ocr  → Tesseract (overrides all)
        
        If the chosen method returns empty text, we fall back to a
        base64 image so the Claude Vision API can still process it.
        """
        analysis = self.detect_type(pdf_path)
        result = PageResult(
            source_file=pdf_path.name,
            pdf_type=analysis.pdf_type,
            analysis=analysis,
        )

        if analysis.pdf_type == "empty":
            result.error = analysis.error or "empty PDF"
            return result

        # ── Select extraction method ────────────────────────────────
        text: Optional[str] = None
        route = {
            "scanned":     ("tesseract",  self.extract_text_tesseract),
            "text_normal":  ("pymupdf",    self.extract_text_pymupdf),
            "text_glyph":  ("tesseract",  self.extract_text_tesseract),
            "hybrid":      ("tesseract",  self.extract_text_tesseract),
        }

        if self.cfg.force_ocr:
            method_name, extractor = "tesseract_forced", self.extract_text_tesseract
            result.notes.append("Force-OCR enabled")
        else:
            method_name, extractor = route.get(
                analysis.pdf_type, ("tesseract", self.extract_text_tesseract)
            )

        result.extraction_method = method_name
        text = extractor(pdf_path)

        # ── Text extraction succeeded ───────────────────────────────
        if text and len(text.strip()) > 0:
            result.text_content = text
            result.notes.append(f"Extracted {len(text)} chars via {method_name}")
            return result

        # ── Fallback: base64 image for Claude Vision API ────────────
        result.notes.append(
            f"{method_name} returned empty — falling back to vision API"
        )
        b64 = self.convert_to_base64(pdf_path)
        if b64:
            result.base64_image = b64
            result.extraction_method = f"{method_name}→vision_fallback"
            result.notes.append("Converted to base64 for vision API")
        else:
            result.error = "All extraction methods failed"

        return result

    # ── Diagnostics ─────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, int]:
        return dict(self.stats)

    def print_stats(self) -> None:
        total = sum(v for k, v in self.stats.items() if k != "errors")
        if total == 0:
            logger.info("No pages processed yet.")
            return
        logger.info("─── PDF Type Statistics ───")
        for k, v in self.stats.items():
            pct = f"{100 * v / max(total, 1):.1f}%" if k != "errors" else ""
            logger.info(f"  {k:14s}: {v:4d}  {pct}")
        logger.info(f"  {'TOTAL':14s}: {total:4d}")
