import { useState } from "react";
import Panel from "./Panel.jsx";

// Draws the uploaded document with bounding-box overlays taken directly from the
// backend JSON. Boxes are [x1,y1,x2,y2] in ORIGINAL image pixels, so they are
// positioned as percentages of result.image.{width,height} — correct at any
// rendered scale. No coordinates are computed here beyond that scaling.
function boxStyle(bbox, imgW, imgH) {
  if (!Array.isArray(bbox) || bbox.length !== 4 || !imgW || !imgH) return null;
  const [x1, y1, x2, y2] = bbox;
  return {
    left: `${(x1 / imgW) * 100}%`,
    top: `${(y1 / imgH) * 100}%`,
    width: `${((x2 - x1) / imgW) * 100}%`,
    height: `${((y2 - y1) / imgH) * 100}%`,
  };
}

export default function DocumentViewer({ imgSrc, result }) {
  const [show, setShow] = useState({ fields: true, forensics: true, face: true });
  const imgW = result?.image?.width;
  const imgH = result?.image?.height;

  const fieldBoxes = Object.entries(result?.fields ?? [])
    .filter(([, fv]) => Array.isArray(fv?.bbox))
    .map(([name, fv]) => ({ label: name.replace(/_/g, " "), bbox: fv.bbox }));

  const forensicBoxes = (result?.forensics?.findings ?? [])
    .filter((f) => Array.isArray(f?.region) && !f.passed)
    .map((f) => ({ label: f.finding_type?.replace(/_/g, " ") ?? "anomaly", bbox: f.region }));

  const faceBoxes = ["document_face"]
    .map((k) => result?.biometrics?.[k])
    .filter((fe) => Array.isArray(fe?.bbox))
    .map((fe) => ({ label: "face", bbox: fe.bbox }));

  if (!imgSrc) {
    return (
      <Panel title="Document">
        <div className="empty">No document image to display.</div>
      </Panel>
    );
  }

  return (
    <Panel
      title="Document viewer"
      right={
        <div className="viewer-toggles" style={{ margin: 0 }}>
          <Toggle
            label={`Fields (${fieldBoxes.length})`}
            checked={show.fields}
            onChange={(v) => setShow((s) => ({ ...s, fields: v }))}
          />
          <Toggle
            label={`Forensic (${forensicBoxes.length})`}
            checked={show.forensics}
            onChange={(v) => setShow((s) => ({ ...s, forensics: v }))}
          />
          <Toggle
            label={`Face (${faceBoxes.length})`}
            checked={show.face}
            onChange={(v) => setShow((s) => ({ ...s, face: v }))}
          />
        </div>
      }
    >
      <div style={{ textAlign: "center" }}>
        <div className="viewer-wrap">
          <img src={imgSrc} alt="uploaded document" />

          {show.fields &&
            fieldBoxes.map((b, i) => (
              <Box key={`f${i}`} b={b} imgW={imgW} imgH={imgH} tone="neutral" />
            ))}
          {show.forensics &&
            forensicBoxes.map((b, i) => (
              <Box key={`x${i}`} b={b} imgW={imgW} imgH={imgH} tone="bad" />
            ))}
          {show.face &&
            faceBoxes.map((b, i) => (
              <Box key={`p${i}`} b={b} imgW={imgW} imgH={imgH} tone="good" />
            ))}
        </div>
      </div>
    </Panel>
  );
}

function Box({ b, imgW, imgH, tone }) {
  const style = boxStyle(b.bbox, imgW, imgH);
  if (!style) return null;
  return (
    <div className={`overlay-box tone-${tone}`} style={style}>
      <span className="lbl">{b.label}</span>
    </div>
  );
}

function Toggle({ label, checked, onChange }) {
  return (
    <label className="toggle">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
      />
      {label}
    </label>
  );
}
