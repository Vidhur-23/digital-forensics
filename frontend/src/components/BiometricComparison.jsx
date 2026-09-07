import Panel from "./Panel.jsx";
import Badge from "./Badge.jsx";
import { toneFor, humanize, num } from "../utils/format.js";

function FaceCell({ title, face, imgSrc }) {
  const detected = face?.detected;
  return (
    <div className="bio-face">
      <div className="frame">
        {imgSrc ? (
          <img src={imgSrc} alt={title} />
        ) : detected ? (
          "face detected"
        ) : (
          "no face"
        )}
      </div>
      <div className="faint-text" style={{ marginTop: 6 }}>
        {title}
      </div>
      <div className="row" style={{ justifyContent: "center", marginTop: 4 }}>
        {face?.quality && (
          <Badge value={face.quality} label={humanize(face.quality)} />
        )}
      </div>
    </div>
  );
}

// Renders result.biometrics (Phase 4). Similarity is shown against the
// backend's threshold — never recomputed. Non-comparable states
// (NO_FACE / INSUFFICIENT_QUALITY / AMBIGUOUS / UNAVAILABLE) are shown as-is.
export default function BiometricComparison({ biometrics, documentImgSrc, referenceImgSrc }) {
  if (!biometrics) return null;

  const status = biometrics.status;
  const tone = toneFor(status);
  const similarity = biometrics.similarity;
  const threshold = biometrics.threshold;
  const compared = similarity != null && threshold != null;
  // Position on the bar (clamped to 0..1) for display only.
  const fillPct = compared ? Math.max(0, Math.min(1, Number(similarity))) * 100 : 0;
  const threshPct = threshold != null ? Math.max(0, Math.min(1, Number(threshold))) * 100 : 50;

  return (
    <Panel title="Biometric comparison" right={<Badge value={status} />}>
      <div className="stack" style={{ gap: 14 }}>
        <div className="bio-faces">
          <FaceCell
            title="Document face"
            face={biometrics.document_face}
            imgSrc={documentImgSrc}
          />
          <div className="bio-vs">vs</div>
          <FaceCell
            title="Reference face"
            face={biometrics.reference_face}
            imgSrc={referenceImgSrc}
          />
        </div>

        {compared ? (
          <div className={`tone-${tone}`}>
            <div className="row spread faint-text">
              <span>similarity {num(similarity)}</span>
              <span>threshold {num(threshold)}</span>
            </div>
            <div className="sim-bar">
              <div className="fill" style={{ width: `${fillPct}%` }} />
              <div className="thresh" style={{ left: `${threshPct}%` }} />
            </div>
            {biometrics.decision_strength && (
              <div className="faint-text" style={{ marginTop: 10 }}>
                decision strength: {biometrics.decision_strength}
              </div>
            )}
          </div>
        ) : (
          <div className="banner tone-muted">
            {biometrics.message ||
              "Faces were not compared (no reference, no face, or insufficient quality)."}
          </div>
        )}

        {biometrics.message && compared && (
          <p className="muted-text" style={{ margin: 0 }}>
            {biometrics.message}
          </p>
        )}
        {biometrics.model && (
          <div className="faint-text">model: {biometrics.model}</div>
        )}
      </div>
    </Panel>
  );
}
