import Panel from "./Panel.jsx";
import Badge from "./Badge.jsx";
import { toneFor } from "../utils/format.js";

// Renders result.intelligence.llm — the advisory LLM interpretation. When the
// LLM is UNAVAILABLE the panel clearly says so; the deterministic assessment
// still stands (shown elsewhere). The LLM never overrides backend facts.
export default function LlmPanel({ llm }) {
  if (!llm) return null;
  const unavailable = llm.status !== "COMPLETED";

  return (
    <Panel
      title="LLM interpretation"
      right={
        <div className="row" style={{ gap: 6 }}>
          {llm.model && <span className="pill">{llm.model}</span>}
          <Badge value={llm.status} />
        </div>
      }
    >
      {unavailable ? (
        <div className="banner tone-muted">
          <strong>LLM explanation unavailable.</strong>
          <div className="faint-text" style={{ marginTop: 4 }}>
            {llm.error || "The advisory LLM did not run."} The deterministic risk
            assessment and recommendation above remain authoritative.
          </div>
        </div>
      ) : (
        <div className="stack" style={{ gap: 14 }}>
          {llm.summary && <p style={{ margin: 0 }}>{llm.summary}</p>}

          {llm.agreement_with_risk && (
            <div className="row">
              <span className="faint-text">Agreement with deterministic risk:</span>
              <Badge
                value={llm.agreement_with_risk}
                tone={toneFor(llm.agreement_with_risk)}
              />
            </div>
          )}

          {Array.isArray(llm.key_findings) && llm.key_findings.length > 0 && (
            <div>
              <div className="faint-text" style={{ marginBottom: 4 }}>
                Key findings
              </div>
              <div className="stack" style={{ gap: 8 }}>
                {llm.key_findings.map((k, i) => (
                  <div className="card tone-neutral" key={i}>
                    <div className="row spread">
                      <span className="pill">{k.source}</span>
                      <Badge value={k.importance} />
                    </div>
                    <p className="muted-text" style={{ margin: "6px 0 0" }}>
                      {k.finding}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

          <LlmList label="Corroboration" items={llm.corroboration} />
          <LlmList label="Contradictions" items={llm.contradictions} />
          <LlmList label="Uncertainties" items={llm.uncertainties} />

          {llm.suggested_action && (
            <div className="faint-text">
              Advisory suggested action: {llm.suggested_action} (does not override
              the recommendation above).
            </div>
          )}
          <p className="disclaimer">
            The LLM interprets the engine evidence; it cannot invent findings or
            override deterministic results.
          </p>
        </div>
      )}
    </Panel>
  );
}

function LlmList({ label, items }) {
  const list = items ?? [];
  if (list.length === 0) return null;
  return (
    <div>
      <div className="faint-text" style={{ marginBottom: 4 }}>
        {label}
      </div>
      <ul className="list-plain">
        {list.map((r, i) => (
          <li key={i}>{r}</li>
        ))}
      </ul>
    </div>
  );
}
