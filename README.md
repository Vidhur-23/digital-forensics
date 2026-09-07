# SIH26188 — Document Screening & Digital Forensics

An identity-document screening system that ingests a scanned/photographed
document (passports today), reads it, and produces **structured, explainable
evidence** about its authenticity — never a black-box "fraud / not fraud"
verdict. A FastAPI backend runs a five-phase analysis pipeline; a React UI lets
an officer upload a document, inspect every finding on the image, and act on a
transparent recommendation.

> **Design stance:** every phase after capture is **evidence-only** and
> **failure-isolated**. A later stage can never overturn an earlier one, and a
> crash in any stage degrades to an explicit `UNAVAILABLE` result instead of
> destroying the analysis or silently reporting "clean". The final judgement is
> always left to a human reviewer.

---

## The pipeline

Each upload flows through the stages below (`backend/app/pipeline/pipeline.py`).
Phases 2–5 consume the structured output of earlier phases; they never re-run
OCR or the CV backends.

| Phase | Stage | What it does | Key modules |
|-------|-------|--------------|-------------|
| **1** | Capture & Extraction | Decode + preprocess the image, run OCR, detect the MRZ, classify the document type, extract passport fields (visual + MRZ). | `document/`, `ocr/` |
| **2** | Rules Engine | Deterministic, stateless checks over the extracted fields — date validity, cross-field consistency, MRZ check digits, passport-format rules, trusted-record comparison. | `rules/` |
| **3** | Forensic Layer | Pixel-level manipulation evidence on the original image — Error Level Analysis, background texture, photo-boundary and colour-consistency signals — each localised to a region. | `forensics/` |
| **4** | Biometric Verification | Compares the document portrait against an optional reference/live face using InsightFace (`buffalo_l` → ArcFace `w600k_r50`, 512-D embeddings, cosine similarity). | `biometrics/` |
| **5** | Intelligence Layer | Fuses the Phase 2–4 evidence, scores risk transparently, builds an evidence-traceable explanation, adds an **advisory** LLM interpretation (off by default), and produces a human-review recommendation. | `intelligence/` |

```
 upload (bytes)                         optional reference face
      │                                          │
      ▼                                          │
 [1] decode → OCR → MRZ → classify → extract     │
      │                                          │
      ▼                                          │
 [2] Rules Engine  (deterministic findings)      │
      │                                          │
      ▼                                          ▼
 [3] Forensic Layer            [4] Biometric verification
      │                                          │
      └──────────────┬───────────────────────────┘
                     ▼
        [5] fuse → risk → explanation → (LLM) → recommendation
                     │
                     ▼
            ScreeningResponse (JSON)  →  persisted  →  UI
```

---

## Tech stack

- **Backend:** Python 3.14, FastAPI, Pydantic v2, SQLAlchemy 2.
- **CV / ML:** OpenCV, NumPy, SciPy, scikit-image, InsightFace + ONNX Runtime,
  Torch/torchvision/timm, Tesseract (via `pytesseract`), PassportEye (MRZ).
- **Intelligence LLM (optional):** Anthropic SDK — advisory only, disabled by
  default; the whole pipeline runs fully offline without it.
- **Persistence:** SQLite for local dev, PostgreSQL for production (same models,
  only `APP_DATABASE_URL` changes); analysed images are kept in an evidence store.
- **Frontend:** React 18 + Vite (JavaScript/JSX).

---

## Project layout

```
digital-forensics/
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI app (app.main:app)
│   │   ├── config.py          # settings, loaded from backend/.env (APP_ prefix)
│   │   ├── pipeline/          # phase orchestration
│   │   ├── document/          # decode, preprocess, classify, regions, quality
│   │   ├── ocr/               # Tesseract engine, MRZ, field extraction
│   │   ├── rules/             # Phase 2 deterministic rules
│   │   ├── forensics/         # Phase 3 forensic signal layers + adapter
│   │   ├── biometrics/        # Phase 4 face detection / embeddings / verify
│   │   ├── intelligence/      # Phase 5 fusion, risk, explanation, LLM, recommend
│   │   ├── api/               # routes + response/request schemas
│   │   └── database/          # engine, ORM models, repository
│   ├── tests/                 # pytest suite (~120 tests)
│   ├── requirements.txt
│   └── .env.example
└── frontend/                  # React + Vite UI
    └── src/
```

---

## Getting started

### Prerequisites

- Python 3.14 and `pip`
- **Tesseract OCR** installed on the system (`tesseract` on `PATH`, or set
  `APP_TESSERACT_CMD`) — Fedora: `sudo dnf install tesseract`
- Node.js 18+ (for the frontend)
- PostgreSQL 13+ *(only for production persistence; dev uses SQLite with zero setup)*

### Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # then edit as needed (see Configuration below)

uvicorn app.main:app --reload
```

The API is now at **http://localhost:8000**:

- Interactive docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

> The import path is `app.main:app` and uvicorn must be launched from the
> `backend/` directory so the `app` package resolves.

### Frontend

```bash
cd frontend
npm install
npm run dev          # Vite dev server (proxies to the backend API)
```

---

## Configuration

All settings live in `backend/.env` and are read with the `APP_` prefix
(`backend/app/config.py`). Start from `.env.example`. Key values:

| Variable | Purpose | Default |
|----------|---------|---------|
| `APP_DATABASE_URL` | Persistence backend. SQLite for dev, Postgres for prod. | `sqlite:///./screening.db` |
| `APP_EVIDENCE_DIR` | Where analysed images are stored. | `./evidence_store` |
| `APP_PERSIST_ANALYSES` | Save every analysis for later retrieval. | `true` |
| `APP_DB_AUTO_CREATE` | Create tables on startup (dev convenience; use migrations in prod). | `true` |
| `APP_MAX_UPLOAD_BYTES` | Reject oversized uploads. | `20 MiB` |
| `APP_TESSERACT_CMD` | Explicit path to the `tesseract` binary (else `PATH`). | *(empty)* |
| `APP_LLM_ENABLED` | Turn on the advisory Phase-5 LLM explanation. | `false` |
| `APP_LLM_PROVIDER` / `APP_LLM_MODEL` | LLM provider + model when enabled. | `anthropic` / `claude-opus-5` |
| `APP_LLM_API_KEY` | Credential — **leave empty** and let the SDK read `ANTHROPIC_API_KEY` from the environment. Never commit a real key. | *(empty)* |
| `APP_LLM_BASE_URL` | Override the LLM endpoint (proxy/gateway). Leave empty for the default Anthropic API. | *(empty)* |

> **Security:** never commit a real API key. `backend/.env` is git-ignored; keep
> secrets out of it and supply them via environment variables at runtime.

### Using PostgreSQL

```bash
# create a role + database (as the postgres superuser)
sudo -u postgres psql -c "CREATE ROLE screening LOGIN PASSWORD '<password>';"
sudo -u postgres psql -c "CREATE DATABASE screening OWNER screening;"

pip install psycopg2-binary      # ensure the driver is in your venv
```

Then point the app at it:

```
APP_DATABASE_URL=postgresql+psycopg2://screening:<password>@localhost:5432/screening
```

With `APP_DB_AUTO_CREATE=true` the tables are created on first startup.

---

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Liveness + whether the OCR backend is ready. |
| `POST` | `/api/screen` | Run the full pipeline on an uploaded document image. Accepts `document` (required) and `reference_face` (optional, enables Phase 4). Returns the complete `ScreeningResponse` JSON. |
| `GET`  | `/api/analyses` | Recent analyses as light summary rows (review queue). |
| `GET`  | `/api/analyses/{id}` | One stored analysis, including the full original response JSON. |

Example:

```bash
curl -X POST http://localhost:8000/api/screen \
  -F "document=@passport.jpg" \
  -F "reference_face=@selfie.jpg"
```

`/api/screen` returns HTTP `415` for an unsupported content type, `413` for an
oversized upload, `422` for an undecodable image, and `503` if the OCR backend
is unavailable. Persistence is best-effort: a storage failure is logged and the
analysis is still returned.

---

## Testing

```bash
cd backend
source venv/bin/activate
pytest
```

The suite (~120 tests) covers OCR/extraction, MRZ, each rule family, and the
forensic, biometric, and intelligence APIs — including the failure-isolation
paths (engine-unavailable, undecodable reference image, LLM outage) that must
degrade to `UNAVAILABLE` rather than crash.

---

## Key principles

- **Evidence, not verdicts.** Phases 2–5 emit findings, scores, and
  explanations for a human to weigh; nothing auto-decides fraud.
- **Deterministic first.** Rules, forensics, fusion, risk, and the
  recommendation are all deterministic and reproducible. The LLM is advisory and
  optional — its absence never weakens a result.
- **Failure isolation.** Each stage is guarded; a failure surfaces as an
  explicit `UNAVAILABLE`, never as a silent "clean" result, and never destroys
  earlier phases' output.
- **Traceable regions.** Forensic and field findings carry image-coordinate
  bounding boxes so the UI can show *where* on the document each concern lives.
