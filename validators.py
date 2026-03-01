"""
Data Validation Utilities
Post-extraction validation for dates, amounts, currencies, and required fields.
"""

import re
from typing import Dict, List, Tuple

from .config import DOCUMENT_SCHEMAS, VALID_CURRENCIES


def validate_date(value) -> bool:
    """Check YYYY-MM-DD format."""
    if not value:
        return True  # null is acceptable
    return bool(re.match(r"^\d{4}-\d{2}-\d{2}$", str(value)))


def validate_amount(value) -> bool:
    """Check that amount is numeric."""
    if value is None:
        return True
    try:
        float(str(value).replace("$", "").replace(",", "").strip())
        return True
    except (ValueError, TypeError):
        return False


def validate_currency(value) -> bool:
    """Check ISO 3-letter currency code."""
    if not value:
        return True
    return str(value).upper() in VALID_CURRENCIES


def validate_extracted_data(document_type: str, data: Dict) -> Tuple[bool, List[str]]:
    """
    Full validation pass against schema.
    
    Returns (is_valid, list_of_warnings).
    A few warnings are acceptable; too many indicate a bad extraction.
    """
    warnings: List[str] = []
    schema = DOCUMENT_SCHEMAS.get(document_type)

    if not schema:
        return False, [f"Unknown document type: {document_type}"]

    # Required fields
    for field in schema["required_fields"]:
        val = data.get(field)
        if val is None or val == "":
            warnings.append(f"Missing required: {field}")

    # Type-specific checks
    for key, value in data.items():
        if key.startswith("_"):
            continue
        if key.endswith("_date") and value:
            if not validate_date(value):
                warnings.append(f"Bad date format in {key}: {value}")
        if any(k in key for k in ("amount", "total", "value")) and value:
            if not validate_amount(value):
                warnings.append(f"Non-numeric value in {key}: {value}")
        if key == "currency" and value:
            if not validate_currency(value):
                warnings.append(f"Invalid currency: {value}")

    return len(warnings) == 0, warnings
