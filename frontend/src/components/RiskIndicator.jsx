import { toneFor } from "../utils/format.js";

// Renders the backend Intelligence Layer risk assessment. Pure display — the
// level and score come straight from result.intelligence.risk.
export default function RiskIndicator({ risk }) {
  if (!risk) {
    return <div className="empty">No risk assessment available.</div>;
  }
  const level = risk.level ?? "—";
  const score = Number.isFinite(Number(risk.score)) ? Number(risk.score) : 0;
  const tone = toneFor(level);

  return (
    <div className="stack" style={{ gap: 12 }}>
      <div className={`risk-indicator tone-${tone}`}>
        <div className="risk-gauge" style={{ "--pct": score }}>
          <div style={{ textAlign: "center" }}>
            <div className="score">{risk.score ?? "—"}</div>
            <div className="of">/ 100</div>
          </div>
        </div>
        <div className="risk-meta">
          <div className="level">{level} risk</div>
          <div className="faint-text">
            {risk.score_kind ? risk.score_kind.replace(/_/g, " ") : "decision support"} score
          </div>
        </div>
      </div>

      {risk.rationale && <p className="muted-text" style={{ margin: 0 }}>{risk.rationale}</p>}

      {typeof risk.corroboration_bonus === "number" &&
        risk.corroboration_bonus > 0 && (
          <div className="pill">
            + {risk.corroboration_bonus} corroboration weight
          </div>
        )}

      {risk.disclaimer && <p className="disclaimer">{risk.disclaimer}</p>}
    </div>
  );
}
