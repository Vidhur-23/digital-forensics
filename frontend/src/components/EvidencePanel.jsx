import Panel from "./Panel.jsx";
import FindingCard from "./FindingCard.jsx";
import CorrelationList from "./CorrelationList.jsx";

// The fused evidence view: cross-engine correlations first (the headline of the
// Intelligence Layer), then every normalised evidence item grouped by polarity.
export default function EvidencePanel({ fused }) {
  if (!fused) return null;
  const items = fused.items ?? [];

  // Sort concern items to the top, by weight.
  const order = { CONCERN: 0, REASSURING: 1, NEUTRAL: 2 };
  const sorted = [...items].sort((a, b) => {
    const pa = order[a.polarity] ?? 3;
    const pb = order[b.polarity] ?? 3;
    if (pa !== pb) return pa - pb;
    return (b.weight ?? 0) - (a.weight ?? 0);
  });

  return (
    <Panel
      title="Fused evidence"
      note="Normalised findings from every engine, plus the cross-engine relationships the Intelligence Layer identified."
    >
      <div className="stack" style={{ gap: 16 }}>
        <div>
          <div className="faint-text" style={{ marginBottom: 8 }}>
            Cross-engine correlations
          </div>
          <CorrelationList correlations={fused.correlations} />
        </div>

        <div>
          <div className="faint-text" style={{ marginBottom: 8 }}>
            Evidence items ({sorted.length})
          </div>
          {sorted.length === 0 ? (
            <div className="empty">No evidence items were produced.</div>
          ) : (
            <div className="stack" style={{ gap: 0 }}>
              {sorted.map((item) => (
                <FindingCard key={item.id} item={item} />
              ))}
            </div>
          )}
        </div>
      </div>
    </Panel>
  );
}
