// API client — plain JavaScript, no TypeScript.
//
// Talks to the ACTUAL FastAPI backend endpoint:
//   POST /api/screen   (multipart/form-data)
//     - document        : required image file (the document/passport)
//     - reference_face   : optional image file (live/reference face for Phase 4)
//
// Returns the backend's ScreeningResponse JSON verbatim. The frontend never
// re-computes risk/forensics/biometrics — it only renders what the backend
// Intelligence Layer produced.

// Same-origin path; the Vite dev server proxies /api -> FastAPI (see
// vite.config.js). Override with VITE_API_BASE for a non-proxied deployment.
const API_BASE = import.meta.env.VITE_API_BASE || "";

export async function screenDocument(documentFile, referenceFaceFile) {
  if (!documentFile) {
    throw new Error("A document image is required.");
  }

  const formData = new FormData();
  formData.append("document", documentFile);
  if (referenceFaceFile) {
    formData.append("reference_face", referenceFaceFile);
  }

  let response;
  try {
    response = await fetch(`${API_BASE}/api/screen`, {
      method: "POST",
      body: formData,
    });
  } catch (networkErr) {
    throw new Error(
      `Could not reach the backend at ${API_BASE || window.location.origin}` +
        `/api/screen. Is the FastAPI server running? (${networkErr.message})`
    );
  }

  if (!response.ok) {
    // FastAPI returns { detail: "..." } for HTTPException; be defensive.
    let detail = `Analysis failed (HTTP ${response.status}).`;
    try {
      const body = await response.json();
      if (body && body.detail) {
        detail =
          typeof body.detail === "string"
            ? body.detail
            : JSON.stringify(body.detail);
      }
    } catch (_) {
      /* non-JSON error body — keep the generic message */
    }
    throw new Error(detail);
  }

  return response.json();
}
