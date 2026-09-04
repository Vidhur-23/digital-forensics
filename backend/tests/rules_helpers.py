"""Helpers for Phase 2 (Rules Engine) tests.

Builds valid TD3 MRZ lines (correct ICAO check digits) and assembles
:class:`ScreeningResponse` objects with controllable visual + MRZ fields, so
each rule can be tested against a precise, synthetic scenario. No real identity
data is used.
"""
from __future__ import annotations

from typing import Dict, Optional

from app.api.schemas.document import (
    FieldValue,
    ImageInfo,
    MRZFieldsOut,
    MRZInfo,
    OCRInfo,
    ScreeningResponse,
)
from app.rules.mrz import compute_check_digit


def build_td3(
    doc_no: str,
    nationality: str,
    dob: str,       # raw YYMMDD
    sex: str,
    expiry: str,    # raw YYMMDD
    surname: str,
    given: str,
    optional: str = "",
    issuing: Optional[str] = None,
):
    """Return two valid TD3 lines (44 chars each) with correct check digits."""
    issuing = issuing or nationality
    name_field = f"{surname}<<{given.replace(' ', '<')}"
    name_field = (name_field + "<" * 39)[:39]
    line1 = "P<" + issuing + name_field

    docn = (doc_no + "<" * 9)[:9]
    opt = (optional + "<" * 14)[:14]
    docchk = compute_check_digit(docn)
    dobchk = compute_check_digit(dob)
    expchk = compute_check_digit(expiry)
    optchk = compute_check_digit(opt)
    body = docn + docchk + nationality + dob + dobchk + sex + expiry + expchk + opt + optchk
    composite = body[0:10] + body[13:20] + body[21:43]
    line2 = body + compute_check_digit(composite)
    return line1, line2


def make_mrz(
    doc_no: str,
    nationality: str,
    dob: str,
    sex: str,
    expiry: str,
    surname: str,
    given: str,
    optional: str = "",
    detected: bool = True,
) -> MRZInfo:
    line1, line2 = build_td3(doc_no, nationality, dob, sex, expiry, surname, given, optional)
    return MRZInfo(
        detected=detected,
        format="TD3",
        text=f"{line1}\n{line2}",
        bbox=[0, 900, 1000, 980],
        fields=MRZFieldsOut(
            document_type="P",
            issuing_country=nationality,
            surname=surname,
            given_names=given,
            document_number=doc_no,
            nationality=nationality,
            date_of_birth=dob,
            sex=sex,
            expiry_date=expiry,
        ),
    )


def field(value: str, source: str = "visual") -> FieldValue:
    return FieldValue(value=value, confidence=0.95, bbox=[0, 0, 100, 30], source=source)


def make_response(
    fields: Dict[str, FieldValue],
    mrz: Optional[MRZInfo] = None,
    document_type: str = "passport",
) -> ScreeningResponse:
    return ScreeningResponse(
        document_type=document_type,
        document_type_confidence=0.95,
        image=ImageInfo(width=1000, height=1000),
        fields=fields,
        mrz=mrz or MRZInfo(detected=False),
        ocr=OCRInfo(confidence=0.95, word_count=10, words=[]),
    )


def genuine_response() -> ScreeningResponse:
    """A clean, internally-consistent, non-expired synthetic passport.

    All visual fields agree with the MRZ, dates are ordered, and the document
    number matches a mock trusted record -> every rule should PASS.
    """
    mrz = make_mrz(
        doc_no="L898902C3",
        nationality="UTO",
        dob="740812",
        sex="F",
        expiry="300101",  # 2030-01-01, not expired
        surname="ERIKSSON",
        given="ANNA MARIA",
    )
    fields = {
        "name": field("ANNA MARIA ERIKSSON"),
        "date_of_birth": field("12 AUG 1974"),
        "document_number": field("L898902C3"),
        "nationality": field("UTO"),
        "issue_date": field("01 JAN 2020"),
        "expiry_date": field("01 JAN 2030"),
    }
    return make_response(fields, mrz)
