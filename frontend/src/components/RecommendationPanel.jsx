import Panel from "./Panel.jsx";
import Badge from "./Badge.jsx";
import { humanize } from "../utils/format.js";

// Renders result.intelligence.recommendation. The officer remains the final
// decision-maker; this is a review-oriented suggestion from the backend.
export default function RecommendationPanel({ recommendation }) {
  if (!recommendation) return null;
  const reasons = recommendation.reasons ?? [];
  const targets = recommendation.review_targets ?? [];

  return (
    <Panel
      title="Recommended action"
      right={<Badge value={recommendation.action} />}
    >
      <div className="stack" style={{ gap: 12 }}>
        <div className="row">
          <strong style={{ fontSize: 16 }}>
            {humanize(recommendation.action)}
          </strong>
        </div>

        {reasons.length > 0 ? (
          <div>
            <div className="faint-text" style={{ marginBottom: 4 }}>
              Why
            </div>
            <ul className="list-plain">
              {reasons.map((r, i) => (
                <li key={i}>{r}</li>
              ))}
            </ul>
          </div>
        ) : (
          <div className="empty">No specific reasons provided.</div>
        )}

        {targets.length > 0 && (
          <div>
            <div className="faint-text" style={{ marginBottom: 4 }}>
              Where to look
            </div>
            <div className="row">
              {targets.map((t, i) => (
                <span className="pill" key={i}>
                  {t.source}
                  {t.field ? ` · ${t.field}` : ""}
                </span>
              ))}
            </div>
          </div>
        )}

        <p className="disclaimer">
          Advisory only. The human officer remains the final decision-maker.
        </p>
      </div>
    </Panel>
  );
}
