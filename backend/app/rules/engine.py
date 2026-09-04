"""Rules Engine — orchestration only (Phase 2).

Responsibilities (and nothing more):

1. receive the structured Phase 1 result (:class:`ScreeningResponse`),
2. determine the document type,
3. select the applicable rule set for that type,
4. execute each rule group,
5. collect all findings into a :class:`RuleResults`.

The engine deliberately contains **no validation logic** — every check lives in
its own module. It also must not do OCR, preprocessing, detection, forensics,
face verification, risk scoring or evidence fusion; those belong to Phase 1 or
to later phases.

Extensibility: a new document type is supported by adding an entry to
``_RULE_SETS`` mapping the type to the list of ``check_*`` callables that apply
to it — no changes to the passport path or the engine flow.
"""
from __future__ import annotations

from typing import Callable, Dict, List

from app.api.schemas.document import ScreeningResponse
from app.rules import common, consistency, dates, driving_license, mrz, passport, trusted_record
from app.rules.schemas import RuleFinding, RuleResults

# A rule group is a callable: ScreeningResponse -> List[RuleFinding].
RuleGroup = Callable[[ScreeningResponse], List[RuleFinding]]

# Document-type-agnostic groups, always run.
_COMMON_GROUPS: List[RuleGroup] = [
    common.check_required_fields,
    common.check_field_formats,
    dates.check_dates,
    mrz.check_mrz,
    consistency.check_consistency,
    trusted_record.check_trusted_record,
]

# Per-type additional groups. Extend for new document types.
_RULE_SETS: Dict[str, List[RuleGroup]] = {
    "passport": [passport.check_passport],
    "driving_license": [driving_license.check_driving_license],
}


class RulesEngine:
    """Deterministic rule orchestrator over a Phase 1 screening result."""

    def select_rule_groups(self, document_type: str) -> List[RuleGroup]:
        """Common groups plus the type-specific set (passport default)."""
        type_groups = _RULE_SETS.get(document_type, _RULE_SETS["passport"])
        return [*_COMMON_GROUPS, *type_groups]

    def evaluate(self, result: ScreeningResponse) -> RuleResults:
        findings: List[RuleFinding] = []
        for group in self.select_rule_groups(result.document_type):
            findings.extend(group(result))
        return RuleResults.from_findings(findings)
