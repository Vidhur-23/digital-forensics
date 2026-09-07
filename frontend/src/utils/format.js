// Presentation helpers — formatting only, NO business logic.
// The backend owns every score/level/decision; these functions just map the
// backend's actual string/number values onto CSS classes and display text.

// Map a status-ish token to a semantic tone used for colour styling.
// Accepts the actual backend vocabularies (risk levels, rule statuses,
// forensic/biometric statuses, LLM status).
export function toneFor(value) {
  const v = String(value ?? "").toUpperCase();
  switch (v) {
    // good / clear
    case "LOW":
    case "PASS":
    case "MATCH":
    case "COMPLETED":
    case "GOOD":
    case "NO_ADDITIONAL_ACTION":
    case "AGREE":
      return "good";
    // caution
    case "MEDIUM":
    case "WARNING":
    case "ACCEPTABLE":
    case "MANUAL_REVIEW":
    case "PARTIAL":
    case "AMBIGUOUS":
      return "warn";
    // concern
    case "HIGH":
    case "CRITICAL":
    case "FAIL":
    case "MISMATCH":
    case "ENHANCED_VERIFICATION":
    case "ESCALATE":
    case "DISAGREE":
    case "POOR":
      return "bad";
    // not-run / unknown
    case "UNAVAILABLE":
    case "NO_FACE":
    case "INSUFFICIENT_QUALITY":
    case "NOT_APPLICABLE":
      return "muted";
    default:
      return "neutral";
  }
}

// Polarity from the fused evidence items.
export function polarityTone(polarity) {
  const v = String(polarity ?? "").toUpperCase();
  if (v === "CONCERN") return "bad";
  if (v === "REASSURING") return "good";
  return "muted";
}

export function severityTone(severity) {
  const v = String(severity ?? "").toUpperCase();
  if (v === "CRITICAL" || v === "HIGH") return "bad";
  if (v === "MEDIUM") return "warn";
  if (v === "LOW") return "neutral";
  return "muted";
}

// Turn ENUM_LIKE_TOKENS into "Enum like tokens" for display.
export function humanize(token) {
  if (token === null || token === undefined) return "";
  return String(token)
    .replace(/[_-]+/g, " ")
    .toLowerCase()
    .replace(/^\w/, (c) => c.toUpperCase());
}

// Percentage from a 0..1 value, defensively.
export function pct(value, digits = 0) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "—";
  }
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

// A plain number, defensively.
export function num(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "—";
  }
  return Number(value).toFixed(digits);
}
