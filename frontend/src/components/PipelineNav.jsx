import { useEffect, useState } from "react";

// Ordered pipeline stages, matching the backend flow:
//   Document/OCR -> Rules -> Forensics -> Biometrics -> Intelligence
const STAGES = [
  { key: "ocr", name: "OCR & Fields" },
  { key: "rules", name: "Rules" },
  { key: "forensics", name: "Forensics" },
  { key: "biometrics", name: "Biometrics" },
  { key: "intelligence", name: "Intelligence" },
];

// Derive each stage's completed status from the ACTUAL backend response.
// Returns { tone, label }. Everything read here comes straight from the JSON.
function stageStatus(result, key) {
  if (key === "ocr") {
    return result?.ocr
      ? { tone: "good", label: `${result.ocr.word_count ?? 0} words` }
      : { tone: "muted", label: "n/a" };
  }
  if (key === "rules") {
    if (!result?.rules) return { tone: "muted", label: "n/a" };
    return result.rules.summary?.has_failures
      ? { tone: "warn", label: "failures" }
      : { tone: "good", label: "passed" };
  }
  if (key === "forensics") {
    const s = result?.forensics?.status;
    if (s !== "COMPLETED") return { tone: "muted", label: "unavailable" };
    const flagged = result.forensics.summary?.flagged_count ?? 0;
    return flagged > 0
      ? { tone: "warn", label: `${flagged} flagged` }
      : { tone: "good", label: "clean" };
  }
  if (key === "biometrics") {
    const s = result?.biometrics?.status;
    if (s === "MATCH") return { tone: "good", label: "match" };
    if (s === "MISMATCH") return { tone: "bad", label: "mismatch" };
    return { tone: "muted", label: (s || "n/a").toLowerCase().replace(/_/g, " ") };
  }
  if (key === "intelligence") {
    const level = result?.intelligence?.risk?.level;
    if (!level) return { tone: "muted", label: "n/a" };
    const tone =
      level === "HIGH" ? "bad" : level === "MEDIUM" ? "warn" : "good";
    return { tone, label: `${level} risk` };
  }
  return { tone: "muted", label: "" };
}

export default function PipelineNav({ result, loading }) {
  // During a run, a highlight sweeps across the stages so each one visibly
  // "updates". Once the response arrives, stages settle to their real status.
  const [step, setStep] = useState(0);

  useEffect(() => {
    if (!loading) return;
    setStep(0);
    const id = setInterval(() => {
      setStep((s) => (s + 1) % STAGES.length);
    }, 560);
    return () => clearInterval(id);
  }, [loading]);

  return (
    <nav className="pipeline-nav" aria-label="Pipeline stages">
      {STAGES.map((stage, i) => {
        let cls = "pstage";
        let statusLabel = "pending";
        let tone = "muted";

        if (loading) {
          // Every stage shows as processing; the current one pulses.
          cls += i === step ? " processing active" : " processing";
          statusLabel = i === step ? "processing…" : "queued";
          tone = "";
        } else if (result) {
          const st = stageStatus(result, stage.key);
          cls += " done";
          tone = st.tone;
          statusLabel = st.label;
        } else {
          cls += " pending";
        }

        return (
          <div
            className={`${cls} ${tone ? `tone-${tone}` : ""}`}
            key={stage.key}
          >
            <span className="pnum">{i + 1}</span>
            <span className="pmeta">
              <span className="pname">{stage.name}</span>
              <span className="pstatus">{statusLabel}</span>
            </span>
          </div>
        );
      })}
    </nav>
  );
}
