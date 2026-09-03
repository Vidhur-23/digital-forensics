"""OCR engine abstraction.

The rest of the application depends only on :class:`OCREngine` and the schemas
in :mod:`app.ocr.schemas` — never on ``pytesseract`` directly. That keeps the
OCR backend swappable (e.g. EasyOCR/PaddleOCR) in a later phase without touching
the pipeline, extraction, or API layers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from app.ocr.schemas import OCRResult, OCRWord


class OCRError(RuntimeError):
    """Raised when OCR cannot be performed (e.g. backend unavailable)."""


class OCREngine(ABC):
    """Contract every OCR backend must satisfy."""

    @abstractmethod
    def extract(self, image: np.ndarray) -> OCRResult:
        """Run OCR on a single image and return words with bbox + confidence."""

    def is_available(self) -> bool:  # pragma: no cover - trivial default
        return True


class TesseractOCREngine(OCREngine):
    """Tesseract-backed engine via ``pytesseract`` (the repo's chosen OCR).

    Uses ``image_to_data`` so we get per-word text, confidence and bounding
    boxes rather than one flat string.
    """

    def __init__(self, lang: str = "eng", tesseract_cmd: str | None = None):
        self.lang = lang
        # Imported lazily so the module (and tests) load even if the wrapper or
        # the tesseract binary is absent.
        import pytesseract

        self._pytesseract = pytesseract
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def is_available(self) -> bool:
        try:
            self._pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def extract(self, image: np.ndarray) -> OCRResult:
        from pytesseract import Output

        try:
            data = self._pytesseract.image_to_data(
                image, lang=self.lang, output_type=Output.DICT
            )
        except Exception as exc:  # tesseract binary missing / failed
            raise OCRError(
                "Tesseract OCR failed. Is the 'tesseract' binary installed and "
                "on PATH? (e.g. `sudo dnf install tesseract`)."
            ) from exc

        words: list[OCRWord] = []
        line_key_to_id: dict[tuple, int] = {}
        n = len(data["text"])
        for i in range(n):
            text = (data["text"][i] or "").strip()
            if not text:
                continue
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = -1.0
            if conf < 0:  # -1 marks non-text layout boxes
                continue

            x, y, w, h = (
                int(data["left"][i]),
                int(data["top"][i]),
                int(data["width"][i]),
                int(data["height"][i]),
            )
            # A line is uniquely identified by block/paragraph/line indices.
            key = (
                data["block_num"][i],
                data["par_num"][i],
                data["line_num"][i],
            )
            line_id = line_key_to_id.setdefault(key, len(line_key_to_id))

            words.append(
                OCRWord(
                    text=text,
                    confidence=max(0.0, min(1.0, conf / 100.0)),
                    bbox=[x, y, x + w, y + h],
                    line_id=line_id,
                )
            )

        return OCRResult(words=words)
