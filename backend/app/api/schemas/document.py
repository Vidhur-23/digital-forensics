"""Pydantic response models for the Phase 1 screening endpoint.

The shape is designed to be directly consumable by later phases:
* every field carries a ``bbox`` (Phase 3 forensics needs regions),
* MRZ values are kept separate from visual values (Phase 2 rules will compare
  ``fields.date_of_birth`` against ``mrz.fields.date_of_birth``, etc.).
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from app.ocr.schemas import BBox
from app.rules.schemas import RuleResults


class ImageInfo(BaseModel):
    width: int
    height: int


class FieldValue(BaseModel):
    value: str
    confidence: float
    bbox: Optional[BBox] = None
    source: str = "visual"  # "visual" or "mrz"


class MRZFieldsOut(BaseModel):
    document_type: str = ""
    issuing_country: str = ""
    surname: str = ""
    given_names: str = ""
    document_number: str = ""
    nationality: str = ""
    date_of_birth: str = ""
    sex: str = ""
    expiry_date: str = ""


class MRZInfo(BaseModel):
    detected: bool = False
    format: Optional[str] = None
    text: str = ""
    bbox: Optional[BBox] = None
    fields: MRZFieldsOut = Field(default_factory=MRZFieldsOut)


class OCRWordOut(BaseModel):
    text: str
    confidence: float
    bbox: BBox


class OCRInfo(BaseModel):
    confidence: float  # mean word confidence
    word_count: int
    # Full word list retained for downstream visualisation / forensics.
    words: List[OCRWordOut] = Field(default_factory=list)


class ScreeningResponse(BaseModel):
    document_type: str
    document_type_confidence: float
    image: ImageInfo
    fields: Dict[str, FieldValue]
    mrz: MRZInfo
    ocr: OCRInfo
    # Phase 2: deterministic rule findings. Optional so a bare Phase 1 result
    # (no rules run yet) is still a valid response object.
    rules: Optional[RuleResults] = None
