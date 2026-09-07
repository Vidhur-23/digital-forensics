import Badge from "./Badge.jsx";
import { severityTone, humanize, num } from "../utils/format.js";

// Renders one raw forensic layer finding (result.forensics.findings[]).
export default function ForensicFinding({ finding }) {
  if (!finding) return null;
  // `passed === false` is a flagged anomaly; a per-layer error is status "error".
  const isError = finding.status === "error";
  const tone = isError ? "muted" : finding.passed ? "good" : "bad";

  return (
    <div className={`card tone-${tone}`}>
      <div className="row spread">
        <strong>{humanize(finding.layer_name)}</strong>
        <div className="row" style={{ gap: 6 }}>
          {finding.severity && (
            <Badge
              value={finding.severity}
              tone={severityTone(finding.severity)}
            />
          )}
          <Badge
            value={finding.passed ? "PASS" : isError ? "UNAVAILABLE" : "FAIL"}
            label={finding.passed ? "clean" : isError ? "error" : "anomaly"}
          />
        </div>
      </div>

      <div className="faint-text" style={{ marginTop: 4 }}>
        {humanize(finding.finding_type)}
      </div>

      {finding.message && (
        <p className="muted-text" style={{ margin: "8px 0 0" }}>
          {finding.message}
        </p>
      )}

      <div className="row faint-text" style={{ marginTop: 8, gap: 14 }}>
        {finding.score != null && <span>score {num(finding.score)}</span>}
        {finding.confidence != null && (
          <span>confidence {num(finding.confidence)}</span>
        )}
        {Array.isArray(finding.region) && finding.region.length === 4 && (
          <span className="mono">
            region [{finding.region.join(", ")}]
          </span>
        )}
      </div>
    </div>
  );
}
