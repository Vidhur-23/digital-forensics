import Panel from "./Panel.jsx";
import Badge from "./Badge.jsx";

// Shows the availability/outcome of each analysis stage using the ACTUAL status
// values the backend returns. Each engine may be present, unavailable, or
// report a stage-specific status — handled defensively.
function statusOf(result) {
  const rules = result?.rules;
  const forensics = result?.forensics;
  const biometrics = result?.biometrics;
  const intelligence = result?.intelligence;

  return [
    {
      name: "Rules",
      // Rules has no top-level status enum; presence + failure flag describe it.
      value: rules
        ? rules?.summary?.has_failures
          ? "FAIL"
          : "COMPLETED"
        : "UNAVAILABLE",
      detail: rules
        ? `${rules.findings?.length ?? 0} findings`
        : "not run",
    },
    {
      name: "Forensics",
      value: forensics?.status ?? "UNAVAILABLE",
      detail:
        forensics?.status === "COMPLETED"
          ? `${forensics?.summary?.flagged_count ?? 0} flagged`
          : forensics?.error ?? "not run",
    },
    {
      name: "Biometrics",
      value: biometrics?.status ?? "UNAVAILABLE",
      detail:
        biometrics?.similarity != null
          ? `sim ${Number(biometrics.similarity).toFixed(2)}`
          : biometrics?.message
          ? biometrics.message.slice(0, 40)
          : "not compared",
    },
    {
      name: "Intelligence",
      value: intelligence ? "COMPLETED" : "UNAVAILABLE",
      detail: intelligence?.risk?.level
        ? `${intelligence.risk.level} risk`
        : "not run",
    },
    {
      name: "LLM explanation",
      value: intelligence?.llm?.status ?? "UNAVAILABLE",
      detail: intelligence?.llm?.model || "advisory",
    },
  ];
}

export default function EngineStatus({ result }) {
  const engines = statusOf(result);
  return (
    <Panel title="Engine status">
      <div className="engine-grid">
        {engines.map((e) => (
          <div className="engine-chip" key={e.name}>
            <div className="name">{e.name}</div>
            <Badge value={e.value} />
            <div className="faint-text" style={{ marginTop: 6 }}>
              {e.detail}
            </div>
          </div>
        ))}
      </div>
    </Panel>
  );
}
