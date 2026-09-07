import Badge from "./Badge.jsx";
import { polarityTone, severityTone, humanize, num } from "../utils/format.js";

// Renders a single normalised fused-evidence item
// (result.intelligence.fused_evidence.items[]). Source-agnostic by design.
export default function FindingCard({ item }) {
  if (!item) return null;
  const tone = polarityTone(item.polarity);

  return (
    <div className={`card tone-${tone}`}>
      <div className="row spread">
        <div className="row" style={{ gap: 8 }}>
          <span className="pill">{item.source}</span>
          {item.field && <span className="pill">{item.field}</span>}
        </div>
        <div className="row" style={{ gap: 6 }}>
          <Badge value={item.severity} tone={severityTone(item.severity)} />
          <Badge value={item.polarity} tone={tone} dot={false} />
        </div>
      </div>

      <div style={{ marginTop: 8 }}>
        <strong>{humanize(item.finding_type)}</strong>
        {item.message && (
          <p className="muted-text" style={{ margin: "4px 0 0" }}>
            {item.message}
          </p>
        )}
      </div>

      <div className="row faint-text" style={{ marginTop: 8, gap: 14 }}>
        {item.confidence != null && (
          <span>confidence {num(item.confidence)}</span>
        )}
        {item.weight != null && item.weight > 0 && (
          <span>weight {num(item.weight, 1)}</span>
        )}
        {Array.isArray(item.signals) &&
          item.signals.map((s, i) => (
            <span className="mono" key={i}>
              #{s}
            </span>
          ))}
      </div>
    </div>
  );
}
