import Panel from "./Panel.jsx";
import { pct } from "../utils/format.js";

// Renders the Phase-1 extracted fields (result.fields) and the parsed MRZ
// fields (result.mrz.fields). Read-only view of the actual backend values.
export default function FieldTable({ fields, mrz }) {
  const fieldEntries = fields ? Object.entries(fields) : [];
  const mrzFields = mrz?.fields ?? {};
  const mrzEntries = Object.entries(mrzFields).filter(([, v]) => v);

  return (
    <Panel
      title="Extracted fields"
      right={
        mrz?.detected ? (
          <span className="pill">MRZ detected</span>
        ) : (
          <span className="pill">no MRZ</span>
        )
      }
    >
      <div className="stack" style={{ gap: 16 }}>
        <div>
          <div className="faint-text" style={{ marginBottom: 6 }}>
            Visual / document fields
          </div>
          {fieldEntries.length === 0 ? (
            <div className="empty">No fields extracted.</div>
          ) : (
            <table className="table">
              <thead>
                <tr>
                  <th>Field</th>
                  <th>Value</th>
                  <th>Source</th>
                  <th>Conf.</th>
                </tr>
              </thead>
              <tbody>
                {fieldEntries.map(([name, fv]) => (
                  <tr key={name}>
                    <td>{name.replace(/_/g, " ")}</td>
                    <td className="val">{fv?.value || "—"}</td>
                    <td>
                      <span className="pill">{fv?.source ?? "—"}</span>
                    </td>
                    <td>{pct(fv?.confidence)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        {mrzEntries.length > 0 && (
          <div>
            <div className="faint-text" style={{ marginBottom: 6 }}>
              MRZ fields
            </div>
            <table className="table">
              <tbody>
                {mrzEntries.map(([name, value]) => (
                  <tr key={name}>
                    <td>{name.replace(/_/g, " ")}</td>
                    <td className="val">{String(value)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </Panel>
  );
}
