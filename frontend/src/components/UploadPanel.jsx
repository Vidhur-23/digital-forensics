import { useRef, useState } from "react";
import Panel from "./Panel.jsx";

function FilePicker({ id, label, hint, file, onPick }) {
  const inputRef = useRef(null);
  const [drag, setDrag] = useState(false);

  const preview = file ? URL.createObjectURL(file) : null;

  return (
    <div className="stack" style={{ gap: 8 }}>
      <label className="faint-text" htmlFor={id}>
        {label}
      </label>
      <div
        className={
          "dropzone" + (drag ? " drag" : "") + (file ? " has-file" : "")
        }
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          const f = e.dataTransfer.files?.[0];
          if (f) onPick(f);
        }}
      >
        {file ? (
          <div className="stack" style={{ gap: 8 }}>
            <img className="thumb" src={preview} alt={label} />
            <span className="faint-text">{file.name}</span>
          </div>
        ) : (
          <div className="stack" style={{ gap: 4 }}>
            <span>Click or drop an image</span>
            <span className="faint-text">{hint}</span>
          </div>
        )}
        <input
          id={id}
          ref={inputRef}
          type="file"
          accept="image/*"
          hidden
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) onPick(f);
          }}
        />
      </div>
    </div>
  );
}

export default function UploadPanel({ onAnalyze, loading }) {
  const [documentFile, setDocumentFile] = useState(null);
  const [referenceFile, setReferenceFile] = useState(null);

  return (
    <Panel title="Upload document">
      <div className="stack" style={{ gap: 16 }}>
        <FilePicker
          id="doc-input"
          label="Document image (required)"
          hint="Passport bio page / ID"
          file={documentFile}
          onPick={setDocumentFile}
        />
        <FilePicker
          id="ref-input"
          label="Reference face (optional — enables biometrics)"
          hint="A live / reference face photo"
          file={referenceFile}
          onPick={setReferenceFile}
        />

        <button
          className="btn primary"
          disabled={!documentFile || loading}
          onClick={() => onAnalyze(documentFile, referenceFile)}
        >
          {loading ? (
            <span className="row" style={{ gap: 8, justifyContent: "center" }}>
              <span className="spinner" /> Analyzing…
            </span>
          ) : (
            "Run analysis"
          )}
        </button>
        <p className="disclaimer">
          Sends the image(s) to the backend screening pipeline
          (POST&nbsp;/api/screen). All risk, forensic, biometric and
          recommendation logic runs server-side.
        </p>
      </div>
    </Panel>
  );
}
