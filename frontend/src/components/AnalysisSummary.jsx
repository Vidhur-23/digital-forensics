import Panel from "./Panel.jsx";
import Badge from "./Badge.jsx";
import RiskIndicator from "./RiskIndicator.jsx";
import { humanize } from "../utils/format.js";

// Top-level summary: overall risk + headline explanation + evidence counts.
// Everything is read from result.intelligence (backend-owned).
export default function AnalysisSummary({ result }) {
  const intel = result?.intelligence;
  const risk = intel?.risk;
  const explanation = intel?.explanation;
  const fused = intel?.fused_evidence;

  return (
    <Panel
      title="Overall assessment"
      right={
        result?.document_type ? (
          <span className="pill">{humanize(result.document_type)}</span>
        ) : null
      }
    >
      <div className="stack" style={{ gap: 16 }}>
        {risk ? (
          <RiskIndicator risk={risk} />
        ) : (
          <div className="empty">
            Intelligence assessment was not produced for this analysis.
          </div>
        )}

        {explanation?.primary_concern && (
          <div className="banner tone-warn">
            <div className="faint-text" style={{ marginBottom: 2 }}>
              Primary concern
            </div>
            {explanation.primary_concern}
          </div>
        )}

        {fused && (
          <div className="row">
            <Badge
              tone="bad"
              dot
              label={`${fused.concern_count ?? 0} concern`}
              value="concern"
            />
            <Badge
              tone="good"
              dot
              label={`${fused.reassuring_count ?? 0} reassuring`}
              value="reassuring"
            />
            <Badge
              tone="muted"
              dot
              label={`${fused.neutral_count ?? 0} neutral`}
              value="neutral"
            />
            {(fused.sources_unavailable ?? []).length > 0 && (
              <span className="pill">
                unavailable: {fused.sources_unavailable.join(", ")}
              </span>
            )}
          </div>
        )}
      </div>
    </Panel>
  );
}
