"""Phase 3 forensic integration adapter.

This module does **not** implement forensics. The forensic algorithms live in
the teammate's project, migrated into this forensics package alongside this
adapter (``app/forensics/{pipeline,signals,utils,results,...}``); its
orchestrator is ``pipeline/forensics_engine.py::run_forensics``. That function
is the source of truth and is treated as such here — its modules use bare
top-level imports (``from signals ... import``), so they are loaded by putting
the forensics directory itself on ``sys.path`` rather than by rewriting them.

This adapter's only jobs are:

1. make ``run_forensics`` importable (put the forensics directory on
   ``sys.path`` once, then import it lazily so the heavy CV imports happen a
   single time and are reused across requests),
2. call it with the ORIGINAL decoded image (BGR ``np.ndarray``) — the same
   pixels, coordinate system and resolution the OCR/field bboxes use,
3. translate the engine's ``dict`` Evidence Object into the project's typed
   :class:`ForensicResults`,
4. isolate failures: if the engine cannot run at all, return an explicit
   ``UNAVAILABLE`` result — never a silent "clean / no tampering" result.

The implemented signal layers (background texture, photo boundary, colour
consistency, ELA) operate directly on image pixels and localise the portrait
region themselves; none of them consume OCR text regions, so there is no OCR
data to feed in and OCR is *not* re-run inside the forensic layer.
"""
from __future__ import annotations

import sys
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from app.api.schemas.evidence import (
    ForensicFinding,
    ForensicResults,
    ForensicStatus,
    ForensicSummary,
)

# The forensic layer is vendored into this package, so its root is simply this
# module's own directory (it holds ``pipeline/``, ``signals/``, ``results/`` …).
# No configuration/indirection is needed to locate it.
FORENSICS_DIR = Path(__file__).resolve().parent


class ForensicService(ABC):
    """Interface the pipeline depends on (injectable for tests)."""

    @abstractmethod
    def analyze(
        self, image_bgr: np.ndarray, document_id: Optional[str] = None
    ) -> ForensicResults:
        """Run forensics on an original-resolution BGR image.

        Implementations must not raise: a forensic failure is reported as an
        ``UNAVAILABLE`` :class:`ForensicResults`, so the surrounding analysis
        pipeline keeps working.
        """
        raise NotImplementedError


class ForensicLayerService(ForensicService):
    """Adapter over ``forensic_layer.pipeline.forensics_engine.run_forensics``."""

    def __init__(self, forensic_layer_path: Optional[Path] = None):
        # Defaults to this package's own directory; still overridable (tests use
        # a bogus path to exercise the unavailable-engine path).
        self._path = Path(forensic_layer_path or FORENSICS_DIR)
        self._run_forensics: Optional[Callable[..., Dict[str, Any]]] = None
        self._import_error: Optional[str] = None

    # -- lazy import (once) --------------------------------------------------
    def _ensure_engine(self) -> Optional[Callable[..., Dict[str, Any]]]:
        """Import ``run_forensics`` on first use; cache the result/failure."""
        if self._run_forensics is not None or self._import_error is not None:
            return self._run_forensics

        if not self._path.exists():
            self._import_error = f"Forensic layer not found at '{self._path}'."
            return None

        try:
            # The engine's own module also inserts this on import; doing it here
            # first is what makes ``import pipeline.forensics_engine`` resolve.
            if str(self._path) not in sys.path:
                sys.path.insert(0, str(self._path))
            from pipeline.forensics_engine import run_forensics  # type: ignore

            self._run_forensics = run_forensics
        except Exception as exc:  # pragma: no cover - defensive import guard
            self._import_error = f"Could not import forensic engine: {exc!r}"
        return self._run_forensics

    # -- public API ----------------------------------------------------------
    def analyze(
        self, image_bgr: np.ndarray, document_id: Optional[str] = None
    ) -> ForensicResults:
        run_forensics = self._ensure_engine()
        if run_forensics is None:
            return ForensicResults.unavailable(
                self._import_error or "Forensic engine unavailable."
            )

        if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
            return ForensicResults.unavailable("No image available for forensics.")

        doc_id = document_id or f"upload_{uuid.uuid4().hex[:12]}"
        try:
            # save=False → do not persist per-request evidence JSON to the
            # teammate's results/ directory.
            evidence = run_forensics(image_bgr, document_id=doc_id, save=False)
        except Exception as exc:  # engine ran but crashed as a whole
            return ForensicResults.unavailable(f"Forensic engine error: {exc!r}")

        return _evidence_to_results(evidence, image_bgr.shape)


# ---------------------------------------------------------------------------
# Evidence dict -> typed ForensicResults
# ---------------------------------------------------------------------------
def _evidence_to_results(
    evidence: Dict[str, Any], image_shape: tuple
) -> ForensicResults:
    height, width = int(image_shape[0]), int(image_shape[1])

    findings: List[ForensicFinding] = [
        _map_finding(f, width, height) for f in evidence.get("findings", [])
    ]

    summary = ForensicSummary(
        layers_requested=int(evidence.get("layers_requested", len(findings))),
        layers_completed=int(evidence.get("layers_completed", 0)),
        layers_failed=int(evidence.get("layers_failed", 0)),
        flagged_count=len(evidence.get("flagged_findings", [])),
        all_passed=bool(evidence.get("all_passed", False)),
    )

    return ForensicResults(
        status=ForensicStatus.COMPLETED,
        engine_version=evidence.get("engine_version"),
        findings=findings,
        summary=summary,
        elapsed_seconds=evidence.get("elapsed_seconds"),
    )


def _map_finding(f: Dict[str, Any], width: int, height: int) -> ForensicFinding:
    raw_metric = dict(f.get("raw_metric", {}))
    finding_type = str(f.get("finding_type", "unknown"))

    return ForensicFinding(
        layer_name=str(f.get("layer_name", "unknown")),
        finding_type=finding_type,
        passed=bool(f.get("passed", False)),
        score=float(f.get("score", 0.0)),
        severity=str(f.get("severity", "low")),
        confidence=float(f.get("confidence", 0.0)),
        region=_extract_region(raw_metric, width, height),
        signals=[finding_type],
        raw_metric=raw_metric,
        message=str(f.get("explanation", "")),
        status=str(f.get("_engine_meta", {}).get("status", "ok")),
    )


def _extract_region(
    raw_metric: Dict[str, Any], width: int, height: int
) -> Optional[List[int]]:
    """Convert the engine's ``[x, y, w, h]`` region into a clamped ``[x1, y1,
    x2, y2]`` bbox in the original image's coordinate system (the same
    convention the frontend uses for OCR/field bboxes)."""
    region = raw_metric.get("region")
    if not isinstance(region, dict):
        return None
    bbox = region.get("bbox")
    if not bbox or len(bbox) != 4:
        return None

    x, y, w, h = (int(v) for v in bbox)
    x1 = max(0, min(x, width))
    y1 = max(0, min(y, height))
    x2 = max(0, min(x + w, width))
    y2 = max(0, min(y + h, height))
    return [x1, y1, x2, y2]
