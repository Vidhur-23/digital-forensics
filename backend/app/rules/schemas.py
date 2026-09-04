"""Standardised rule-result schema (Phase 2).

Every deterministic check in the Rules Engine returns a :class:`RuleFinding`
with the same shape, so the pipeline and API can treat findings uniformly and
later phases (evidence fusion, risk scoring — NOT implemented here) can consume
them without knowing which rule produced them.

Design contract (see the task spec):
* A finding is **evidence, not a verdict**. A ``FAIL`` never means "the document
  is fake"; it means a specific deterministic check did not pass and warrants
  review.
* Every finding must explain itself: a human-readable ``message`` plus an
  ``evidence`` dict carrying the concrete values that were compared.
"""
from __future__ import annotations

from collections import Counter
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RuleStatus(str, Enum):
    """Outcome of a single rule."""

    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class RuleSeverity(str, Enum):
    """How much a non-passing result should weigh in later review."""

    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RuleFinding(BaseModel):
    """A single standardised rule result.

    ``field`` is the document field the rule is about (``date_of_birth``,
    ``document_number`` ...) or ``None`` for document-level checks.
    ``evidence`` holds the raw values behind the message so a reviewer can see
    *why* without re-running anything.
    """

    rule_id: str
    category: str  # required_field | format | date | mrz | consistency | passport | trusted_record
    status: RuleStatus
    severity: RuleSeverity
    field: Optional[str] = None
    message: str
    evidence: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def make(
        cls,
        rule_id: str,
        category: str,
        status: RuleStatus,
        severity: RuleSeverity,
        message: str,
        field: Optional[str] = None,
        evidence: Optional[Dict[str, Any]] = None,
    ) -> "RuleFinding":
        return cls(
            rule_id=rule_id,
            category=category,
            status=status,
            severity=severity,
            field=field,
            message=message,
            evidence=evidence or {},
        )


class RuleSummary(BaseModel):
    """Aggregate counts over a set of findings.

    Deliberately NOT a risk score or a verdict — just tallies so callers can see
    at a glance how many checks passed/failed. Interpretation is a later phase.
    """

    total: int = 0
    by_status: Dict[str, int] = Field(default_factory=dict)
    by_severity: Dict[str, int] = Field(default_factory=dict)
    # Convenience flag: did any check outright FAIL? (still not a verdict).
    has_failures: bool = False

    @classmethod
    def from_findings(cls, findings: List[RuleFinding]) -> "RuleSummary":
        status_counts = Counter(f.status.value for f in findings)
        severity_counts = Counter(
            f.severity.value for f in findings if f.status != RuleStatus.PASS
        )
        return cls(
            total=len(findings),
            by_status=dict(status_counts),
            by_severity=dict(severity_counts),
            has_failures=status_counts.get(RuleStatus.FAIL.value, 0) > 0,
        )


class RuleResults(BaseModel):
    """The Rules Engine's output: all findings plus a non-verdict summary."""

    findings: List[RuleFinding] = Field(default_factory=list)
    summary: RuleSummary = Field(default_factory=RuleSummary)

    @classmethod
    def from_findings(cls, findings: List[RuleFinding]) -> "RuleResults":
        return cls(findings=findings, summary=RuleSummary.from_findings(findings))
