"""
Document Intelligence Pipeline
──────────────────────────────
AI-powered PDF classification and structured data extraction.

Quick start:
    from src import DocumentPipeline, PipelineConfig

    pipeline = DocumentPipeline(PipelineConfig(input_dir="./pdfs"))
    pipeline.run()
"""

from .config import PipelineConfig, DOCUMENT_SCHEMAS
from .pdf_processor import PDFProcessor, PageResult, PDFAnalysis
from .ai_engine import AIEngine, ClassificationResult
from .pipeline import DocumentPipeline
from .validators import validate_extracted_data

__all__ = [
    "PipelineConfig",
    "DOCUMENT_SCHEMAS",
    "PDFProcessor",
    "PageResult",
    "PDFAnalysis",
    "AIEngine",
    "ClassificationResult",
    "DocumentPipeline",
    "validate_extracted_data",
]
