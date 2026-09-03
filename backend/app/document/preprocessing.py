"""Image decoding and conservative preprocessing for OCR.

Design notes:
* We keep the ORIGINAL decoded image untouched for later visualisation /
  forensics (Phase 3 needs pixel-accurate regions).
* OCR runs on a grayscale + contrast-normalised copy at the SAME resolution as
  the original, so every OCR bounding box maps 1:1 back onto original pixels.
  (No resize by default → no coordinate rescaling to get wrong.)
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


class ImageDecodeError(ValueError):
    """Raised when uploaded bytes cannot be decoded into an image."""


@dataclass
class DocumentImage:
    """A decoded document image plus an OCR-ready variant.

    ``original`` is BGR (as decoded by OpenCV); ``ocr_image`` is a single-channel
    processed image aligned to the original resolution.
    """

    original: np.ndarray  # HxWx3, BGR
    ocr_image: np.ndarray  # HxW, grayscale/processed

    @property
    def width(self) -> int:
        return int(self.original.shape[1])

    @property
    def height(self) -> int:
        return int(self.original.shape[0])


def decode_image(data: bytes) -> np.ndarray:
    """Decode raw upload bytes into a BGR image, or raise ImageDecodeError."""
    if not data:
        raise ImageDecodeError("Empty file: no image data received.")

    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise ImageDecodeError(
            "Could not decode image. The file is not a readable image "
            "(supported: JPEG, PNG, BMP, TIFF, WEBP)."
        )
    return img


def preprocess_for_ocr(original: np.ndarray) -> np.ndarray:
    """Conservative enhancement that reliably helps Tesseract.

    Grayscale → mild denoise → CLAHE contrast equalisation. No binarisation and
    no resize, so results stay robust across document styles and bboxes stay in
    original coordinates.
    """
    if original.ndim == 3:
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    else:
        gray = original

    # Light edge-preserving denoise.
    denoised = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)

    # Local contrast improvement — helps faded / unevenly lit scans.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    return enhanced


def load_document(data: bytes) -> DocumentImage:
    """Decode + preprocess in one step."""
    original = decode_image(data)
    ocr_image = preprocess_for_ocr(original)
    return DocumentImage(original=original, ocr_image=ocr_image)
