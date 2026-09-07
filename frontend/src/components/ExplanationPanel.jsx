import Panel from "./Panel.jsx";

// Renders result.intelligence.explanation — the deterministic, evidence-traceable
// reasoning (available even when the LLM is not).
export default function ExplanationPanel({ explanation }) {
  if (!explanation) return null;
  const {
    overall_assessment,
    corroborating_evidence = [],
    counter_evidence = [],
    reasoning = [],
  } = explanation;

  return (
    <Panel
      title="Explanation"
      note="Deterministic reasoning — every point traces back to a real engine finding."
    >
      <div className="stack" style={{ gap: 14 }}>
        {overall_assessment && (
          <div className="banner tone-neutral">{overall_assessment}</div>
        )}

        {reasoning.length > 0 && (
          <div>
            <div className="faint-text" style={{ marginBottom: 4 }}>
              Reasoning
            </div>
            <ul className="list-plain">
              {reasoning.map((r, i) => (
                <li key={i}>{r}</li>
              ))}
            </ul>
          </div>
        )}

        {corroborating_evidence.length > 0 && (
          <div>
            <div className="faint-text" style={{ marginBottom: 4 }}>
              Corroborating evidence
            </div>
            <ul className="list-plain">
              {corroborating_evidence.map((r, i) => (
                <li key={i}>{r}</li>
              ))}
            </ul>
          </div>
        )}

        {counter_evidence.length > 0 && (
          <div>
            <div className="faint-text" style={{ marginBottom: 4 }}>
              Counter / mixed evidence
            </div>
            <ul className="list-plain">
              {counter_evidence.map((r, i) => (
                <li key={i}>{r}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </Panel>
  );
}
