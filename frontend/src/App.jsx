import { useState } from "react";
import { screenDocument } from "./api/client.js";
import AppShell from "./components/AppShell.jsx";
import PipelineNav from "./components/PipelineNav.jsx";
import UploadPanel from "./components/UploadPanel.jsx";
import EngineStatus from "./components/EngineStatus.jsx";
import AnalysisSummary from "./components/AnalysisSummary.jsx";
import RecommendationPanel from "./components/RecommendationPanel.jsx";
import ExplanationPanel from "./components/ExplanationPanel.jsx";
import EvidencePanel from "./components/EvidencePanel.jsx";
import LlmPanel from "./components/LlmPanel.jsx";
import DocumentViewer from "./components/DocumentViewer.jsx";
import BiometricComparison from "./components/BiometricComparison.jsx";
import ForensicFinding from "./components/ForensicFinding.jsx";
import FieldTable from "./components/FieldTable.jsx";
import Panel from "./components/Panel.jsx";

export default function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  // Object URLs of the uploaded images, for the viewer / biometric panel.
  const [docSrc, setDocSrc] = useState(null);
  const [refSrc, setRefSrc] = useState(null);

  async function handleAnalyze(documentFile, referenceFile) {
    setLoading(true);
    setError(null);
    // Keep local previews of exactly what was sent.
    setDocSrc(documentFile ? URL.createObjectURL(documentFile) : null);
    setRefSrc(referenceFile ? URL.createObjectURL(referenceFile) : null);
    try {
      const json = await screenDocument(documentFile, referenceFile);
      setResult(json);
    } catch (e) {
      setError(e.message || "Analysis failed.");
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  // Defensive slices — any stage may be absent/unavailable.
  const forensics = result?.forensics;
  const forensicFindings = forensics?.findings ?? [];
  const intel = result?.intelligence;

  return (
    <AppShell nav={<PipelineNav result={result} loading={loading} />}>
      <div className="left-col">
        <UploadPanel onAnalyze={handleAnalyze} loading={loading} />
        {result && <EngineStatus result={result} />}
        {error && (
          <Panel title="Error">
            <div className="banner tone-bad">{error}</div>
          </Panel>
        )}
      </div>

      <div className="right-col">
        {!result && !error && (
          <Panel title="Getting started">
            <p className="muted-text" style={{ marginTop: 0 }}>
              Upload a document image (and optionally a reference face) and run
              the analysis. The results below are rendered directly from the
              backend screening response — the frontend performs no scoring of
              its own.
            </p>
          </Panel>
        )}

        {result && (
          <>
            <AnalysisSummary result={result} />

            {intel?.recommendation && (
              <RecommendationPanel recommendation={intel.recommendation} />
            )}

            {docSrc && <DocumentViewer imgSrc={docSrc} result={result} />}

            {intel?.explanation && (
              <ExplanationPanel explanation={intel.explanation} />
            )}

            {intel?.fused_evidence && (
              <EvidencePanel fused={intel.fused_evidence} />
            )}

            {intel?.llm && <LlmPanel llm={intel.llm} />}

            {result.biometrics && (
              <BiometricComparison
                biometrics={result.biometrics}
                documentImgSrc={docSrc}
                referenceImgSrc={refSrc}
              />
            )}

            {forensics && (
              <Panel
                title="Forensic findings"
                right={
                  <span className="pill">
                    {forensics.status}
                    {forensics.engine_version
                      ? ` · ${forensics.engine_version}`
                      : ""}
                  </span>
                }
              >
                {forensics.status !== "COMPLETED" ? (
                  <div className="banner tone-muted">
                    {forensics.error ||
                      "Forensic analysis was unavailable — treat as unknown, not clean."}
                  </div>
                ) : forensicFindings.length === 0 ? (
                  <div className="empty">No forensic findings reported.</div>
                ) : (
                  <div className="stack" style={{ gap: 0 }}>
                    {forensicFindings.map((f, i) => (
                      <ForensicFinding key={i} finding={f} />
                    ))}
                  </div>
                )}
              </Panel>
            )}

            <FieldTable fields={result.fields} mrz={result.mrz} />
          </>
        )}
      </div>
    </AppShell>
  );
}
