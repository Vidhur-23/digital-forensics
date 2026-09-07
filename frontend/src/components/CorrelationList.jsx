import Badge from "./Badge.jsx";
import { humanize } from "../utils/format.js";

// Renders result.intelligence.fused_evidence.correlations[] — the cross-engine
// corroboration / contradiction / consistency relationships the backend found.
function correlationTone(type) {
  const t = String(type ?? "").toUpperCase();
  if (t === "CORROBORATION") return "bad";
  if (t === "CONTRADICTION") return "warn";
  if (t === "CONSISTENCY") return "good";
  return "muted";
}

export default function CorrelationList({ correlations }) {
  const list = correlations ?? [];
  if (list.length === 0) {
    return <div className="empty">No cross-engine correlations were identified.</div>;
  }
  return (
    <div className="stack" style={{ gap: 10 }}>
      {list.map((c, i) => {
        const tone = correlationTone(c.type);
        return (
          <div className={`card tone-${tone}`} key={i}>
            <div className="row spread">
              <Badge value={c.type} tone={tone} label={humanize(c.type)} />
              <div className="row" style={{ gap: 6 }}>
                {(c.sources ?? []).map((s, j) => (
                  <span className="pill" key={j}>
                    {s}
                  </span>
                ))}
              </div>
            </div>
            <p className="muted-text" style={{ margin: "8px 0 0" }}>
              {c.description}
            </p>
            {c.field && (
              <div className="faint-text" style={{ marginTop: 4 }}>
                field: <span className="mono">{c.field}</span>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
