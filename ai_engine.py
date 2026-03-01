"""
AI Classification & Extraction Engine
Wraps the Anthropic Claude API for document classification and
structured data extraction. Supports both text and vision (base64) inputs.

Token Optimisation Strategy
───────────────────────────
The pipeline minimises token usage by:
1. Routing clean-text PDFs through PyMuPDF first (no image tokens)
2. Only sending base64 images when text extraction fails
3. Using separate classify → extract calls with tight max_tokens
4. Keeping system prompts concise and re-usable across calls

Retry logic uses exponential backoff for transient API errors.
"""

import json
import re
import time
import logging
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass

from anthropic import Anthropic, APIError, RateLimitError

from .config import (
    PipelineConfig,
    DOCUMENT_SCHEMAS,
    CLASSIFICATION_SYSTEM_PROMPT,
    EXTRACTION_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    document_type: str
    confidence: float
    reasoning: str


class AIEngine:
    """
    Claude API wrapper for document intelligence.
    
    Two-stage pipeline per page:
      1. classify()  → document type + confidence
      2. extract()   → structured fields per schema
    
    Both stages accept either plain text (cheap, fast) or a
    base64 image dict (for scanned/glyph PDFs that need vision).
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.cfg = config or PipelineConfig()
        if not self.cfg.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. Add it to your .env file."
            )
        self.client = Anthropic(api_key=self.cfg.api_key)
        self.api_calls = 0
        self.api_errors = 0

    # ── Internal helpers ────────────────────────────────────────────

    def _build_messages(
        self,
        prompt_text: str,
        content: Union[str, Dict],
        format_type: str,
    ) -> list:
        """Build Claude message payload for text or vision input."""
        fmt = format_type.strip().upper()

        if fmt == "TEXT":
            return [{"role": "user", "content": f"{prompt_text}\n\nDocument Content:\n{content}"}]

        elif fmt == "SCANNED":
            return [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": content.get("media_type", "image/png"),
                            "data": content["base64_image"],
                        },
                    },
                    {"type": "text", "text": prompt_text},
                ],
            }]
        else:
            raise ValueError(f"Unknown format_type: {format_type}")

    def _call_api(
        self, model: str, max_tokens: int, system: str, messages: list
    ) -> str:
        """Call Claude with retry logic for rate limits and transient errors."""
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                self.api_calls += 1
                resp = self.client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=messages,
                )
                return resp.content[0].text

            except RateLimitError:
                wait = self.cfg.retry_backoff ** attempt
                logger.warning(f"Rate limited — retrying in {wait:.0f}s (attempt {attempt})")
                time.sleep(wait)

            except APIError as exc:
                self.api_errors += 1
                if attempt < self.cfg.max_retries:
                    wait = self.cfg.retry_backoff ** attempt
                    logger.warning(f"API error: {exc} — retrying in {wait:.0f}s")
                    time.sleep(wait)
                else:
                    raise

        raise RuntimeError("Max retries exceeded")

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Extract first JSON object from response text."""
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise json.JSONDecodeError("No JSON found", text, 0)

    # ── Public API ──────────────────────────────────────────────────

    def classify(
        self,
        content: Union[str, Dict],
        filename: str,
        format_type: str = "TEXT",
    ) -> ClassificationResult:
        """
        Classify a document page.
        
        Args:
            content: extracted text (str) or {"base64_image": ..., "media_type": ...}
            filename: original filename for context
            format_type: "TEXT" or "SCANNED"
        
        Returns:
            ClassificationResult with type, confidence, reasoning
        """
        prompt = (
            f"Classify this document.\n"
            f"Filename: {filename}\n"
            f"Respond with JSON only."
        )
        messages = self._build_messages(prompt, content, format_type)

        try:
            raw = self._call_api(
                self.cfg.model_classify,
                self.cfg.max_tokens_classify,
                CLASSIFICATION_SYSTEM_PROMPT,
                messages,
            )
            data = self._parse_json(raw)
            return ClassificationResult(
                document_type=data.get("document_type", "OTHER"),
                confidence=float(data.get("confidence", 0.0)),
                reasoning=data.get("reasoning", ""),
            )
        except (json.JSONDecodeError, KeyError) as exc:
            logger.error(f"Classification parse error: {exc}")
            return ClassificationResult("OTHER", 0.0, f"Parse error: {exc}")
        except Exception as exc:
            logger.error(f"Classification API error: {exc}")
            return ClassificationResult("OTHER", 0.0, f"API error: {exc}")

    def extract(
        self,
        document_type: str,
        content: Union[str, Dict],
        filename: str,
        format_type: str = "TEXT",
    ) -> Dict:
        """
        Extract structured fields from a classified document.
        
        Uses the type-specific schema from config to tell Claude
        exactly which fields to extract — keeping token usage low.
        
        Returns:
            Dict of extracted fields (includes file_source)
        """
        schema = DOCUMENT_SCHEMAS.get(document_type)
        if not schema or not schema["extraction_prompt"]:
            logger.warning(f"No schema for type: {document_type}")
            return {"file_source": filename, "error": "no schema"}

        prompt = (
            f"Extract structured data from this {document_type} document.\n\n"
            f"{schema['extraction_prompt']}\n\n"
            f"Filename: {filename}\n"
            f"Return ONLY valid JSON in the specified format."
        )
        messages = self._build_messages(prompt, content, format_type)

        try:
            raw = self._call_api(
                self.cfg.model_extract,
                self.cfg.max_tokens_extract,
                EXTRACTION_SYSTEM_PROMPT,
                messages,
            )
            data = self._parse_json(raw)
            extracted = data.get("extraction", data)  # handle both nesting styles
            extracted["file_source"] = filename
            if "extraction_notes" in data:
                extracted["_notes"] = data["extraction_notes"]
            return extracted

        except (json.JSONDecodeError, KeyError) as exc:
            logger.error(f"Extraction parse error: {exc}")
            return {"file_source": filename, "error": f"Parse error: {exc}"}
        except Exception as exc:
            logger.error(f"Extraction API error: {exc}")
            return {"file_source": filename, "error": f"API error: {exc}"}

    def classify_and_extract(
        self,
        content: Union[str, Dict],
        filename: str,
        format_type: str = "TEXT",
    ) -> Dict:
        """
        Convenience method: classify → extract → return combined result.
        
        This is the main entry point used by the pipeline for each page.
        """
        fmt = format_type.strip().upper()

        # ── Step 1: Classify ────────────────────────────────────────
        cls = self.classify(content, filename, fmt)
        logger.info(f"Classified {filename} → {cls.document_type} ({cls.confidence:.0%})")

        if cls.confidence < self.cfg.confidence_threshold:
            logger.warning(f"Low confidence ({cls.confidence:.0%}) — skipping extraction")
            return {
                "filename": filename,
                "format": fmt,
                "document_type": cls.document_type,
                "confidence": cls.confidence,
                "reasoning": cls.reasoning,
                "status": "skipped_low_confidence",
            }

        # ── Step 2: Extract ─────────────────────────────────────────
        extracted = self.extract(cls.document_type, content, filename, fmt)
        logger.info(f"Extracted {len(extracted)} fields from {filename}")

        return {
            "filename": filename,
            "format": fmt,
            "document_type": cls.document_type,
            "confidence": cls.confidence,
            "extracted_data": extracted,
            "status": "success",
        }

    # ── Diagnostics ─────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, int]:
        return {"api_calls": self.api_calls, "api_errors": self.api_errors}
