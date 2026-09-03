"""Image decoding + preprocessing (valid and invalid input)."""
from __future__ import annotations

import pytest

from app.document.preprocessing import (
    ImageDecodeError,
    decode_image,
    load_document,
)
from tests.conftest import make_image_bytes


def test_decode_valid_image_exposes_dimensions():
    doc = load_document(make_image_bytes(width=800, height=600))
    assert doc.width == 800
    assert doc.height == 600
    # Original preserved in colour; OCR variant is single-channel.
    assert doc.original.ndim == 3
    assert doc.ocr_image.ndim == 2


def test_decode_invalid_bytes_raises():
    with pytest.raises(ImageDecodeError):
        decode_image(b"this is definitely not an image")


def test_decode_empty_bytes_raises():
    with pytest.raises(ImageDecodeError):
        decode_image(b"")
