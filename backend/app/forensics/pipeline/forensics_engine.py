"""
╔══════════════════════════════════════════════════════════════╗
║           FORENSICS ENGINE — Evidence Orchestrator           ║
╠══════════════════════════════════════════════════════════════╣
║  Runs all implemented forensic signal layers on a single     ║
║  passport image and produces a unified Evidence Object.      ║
║                                                              ║
║  Downstream consumers (Evidence Fusion, Risk Scoring, LLM    ║
║  Explanation) receive this Evidence Object — they are NOT     ║
║  implemented here.                                           ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import time
import traceback
from pathlib import Path
from datetime import datetime, timezone

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Resolve project root so we can import signals.* regardless of where
# the script is invoked from.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Import the evaluate_* entry-points from each calibrated layer.
# ---------------------------------------------------------------------------
from signals.background_texture import evaluate_background_texture   # Layer 1
from signals.photo_boundary      import evaluate_photo_boundary      # Layer 2
from signals.color_consistency   import evaluate_color_consistency    # Layer 3
from signals.ela                 import evaluate_ela                  # Layer 4

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
RESULTS_DIR   = PROJECT_ROOT / "results"
EVIDENCE_DIR  = RESULTS_DIR / "evidence"

# ---------------------------------------------------------------------------
# Layer registry — easy to extend when new layers arrive.
# Each entry: (human_name, callable, kwargs)
# ---------------------------------------------------------------------------
LAYER_REGISTRY = [
    ("Layer 1: Background Consistency", evaluate_background_texture,  {}),
    ("Layer 2: Photo Boundary",          evaluate_photo_boundary,      {}),
    ("Layer 3: Color Consistency",       evaluate_color_consistency,    {}),
    ("Layer 4: Error Level Analysis",    evaluate_ela,                  {}),
]


# ═══════════════════════════════════════════════════════════════
#  Core orchestrator
# ═══════════════════════════════════════════════════════════════
def run_forensics(image_input, document_id: str = None, save: bool = True) -> dict:
    """
    Runs every registered forensic layer on *image_input* and returns
    a unified Evidence Object.

    Parameters
    ----------
    image_input : str | Path | np.ndarray
        File path to the passport image **or** a pre-loaded BGR array.
    document_id : str, optional
        Human-readable identifier for the document (e.g. 'aze_passport_00_real_3_2').
        Auto-derived from the filename when *image_input* is a path.
    save : bool
        If True the Evidence Object is written to results/evidence/.

    Returns
    -------
    dict  —  The Evidence Object with the following top-level keys:
        document_id       : str
        image_source      : str | None
        timestamp_utc     : str (ISO-8601)
        engine_version    : str
        layers_requested  : int
        layers_completed  : int
        layers_failed     : int
        all_passed        : bool
        findings          : list[dict]   — one entry per layer
        errors            : list[dict]   — one entry per failed layer
        elapsed_seconds   : float
    """

    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Resolve document_id & image source path
    # ------------------------------------------------------------------
    image_source = None
    if isinstance(image_input, (str, Path)):
        image_source = str(image_input)
        if document_id is None:
            document_id = Path(image_input).stem
    if document_id is None:
        document_id = f"unknown_{int(time.time())}"

    # ------------------------------------------------------------------
    # 2. Run every layer and collect findings
    # ------------------------------------------------------------------
    findings: list[dict] = []
    errors:   list[dict] = []

    for layer_name, evaluate_fn, kwargs in LAYER_REGISTRY:
        layer_start = time.perf_counter()
        try:
            result = evaluate_fn(image_input, **kwargs)

            # Ensure standard schema keys are present for downstream safety
            if "finding_type" not in result:
                result["finding_type"] = "normal" if result.get("passed", True) else "anomaly"
            if "severity" not in result:
                score_val = result.get("score", 0.0)
                result["severity"] = "low" if score_val < 0.3 else ("medium" if score_val < 0.7 else "high")
            if "confidence" not in result:
                result["confidence"] = 0.5
            if "explanation" not in result:
                result["explanation"] = f"{layer_name} evaluation completed."
            if "raw_metric" not in result:
                result["raw_metric"] = {}

            result["_engine_meta"] = {
                "elapsed_ms": round((time.perf_counter() - layer_start) * 1000, 2),
                "status": "ok",
            }
            findings.append(result)
        except Exception:
            tb = traceback.format_exc()
            error_entry = {
                "layer_name": layer_name,
                "status": "error",
                "traceback": tb,
                "elapsed_ms": round((time.perf_counter() - layer_start) * 1000, 2),
            }
            errors.append(error_entry)
            # Also add a stub finding so the downstream schema stays uniform
            findings.append({
                "layer_name": layer_name,
                "finding_type": "processing_error",
                "score": 1.0,
                "confidence": 0.0,
                "severity": "high",
                "passed": False,
                "raw_metric": {},
                "explanation": f"Layer crashed: {tb.splitlines()[-1]}",
                "_engine_meta": {
                    "elapsed_ms": error_entry["elapsed_ms"],
                    "status": "error",
                },
            })

    # ------------------------------------------------------------------
    # 3. Aggregate into the Evidence Object
    # ------------------------------------------------------------------
    layers_completed = sum(1 for f in findings if f.get("_engine_meta", {}).get("status") == "ok")
    layers_failed    = len(errors)
    all_passed       = all(f.get("passed", False) for f in findings)

    # ------------------------------------------------------------------
    # 3b. Build flagged_findings — only layers that did NOT pass.
    #     This is the primary input for the downstream LLM explainer.
    # ------------------------------------------------------------------
    flagged_findings = []
    anomaly_summary  = []

    for f in findings:
        if not f.get("passed", True):
            # Strip engine metadata from the LLM-facing copy
            flagged_copy = {k: v for k, v in f.items() if k != "_engine_meta"}
            flagged_findings.append(flagged_copy)

            # Build a concise anomaly entry the LLM can reason over
            anomaly_entry = {
                "layer":       f.get("layer_name", "unknown"),
                "severity":    f.get("severity", "unknown"),
                "score":       f.get("score", None),
                "confidence":  f.get("confidence", None),
                "explanation": f.get("explanation", ""),
                "raw_metric":  f.get("raw_metric", {}),
            }
            anomaly_summary.append(anomaly_entry)

    evidence = {
        "document_id":      document_id,
        "image_source":     image_source,
        "timestamp_utc":    datetime.now(timezone.utc).isoformat(),
        "engine_version":   "0.1.0",
        "layers_requested": len(LAYER_REGISTRY),
        "layers_completed": layers_completed,
        "layers_failed":    layers_failed,
        "all_passed":       bool(all_passed),
        "findings":         findings,
        "flagged_findings": flagged_findings,
        "anomaly_summary":  anomaly_summary,
        "errors":           errors,
        "elapsed_seconds":  round(time.perf_counter() - t_start, 4),
    }

    # ------------------------------------------------------------------
    # 4. Persist to disk
    # ------------------------------------------------------------------
    if save:
        os.makedirs(EVIDENCE_DIR, exist_ok=True)
        out_path = EVIDENCE_DIR / f"{document_id}_evidence.json"
        with open(out_path, "w") as fh:
            json.dump(evidence, fh, indent=2, default=str)
        print(f"[Forensics Engine] Evidence saved → {out_path}")

    return evidence


# ═══════════════════════════════════════════════════════════════
#  Batch helper (optional convenience)
# ═══════════════════════════════════════════════════════════════
def run_batch(image_dir: str | Path, save: bool = True) -> list[dict]:
    """
    Run forensics on every image in a directory.
    Returns a list of Evidence Objects.
    """
    image_dir = Path(image_dir)
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in valid_ext
    )
    results = []
    for i, img_path in enumerate(image_paths, 1):
        print(f"\n[Forensics Engine] Processing [{i}/{len(image_paths)}]: {img_path.name}")
        ev = run_forensics(str(img_path), save=save)
        results.append(ev)
    return results


# ═══════════════════════════════════════════════════════════════
#  Pretty-print helper
# ═══════════════════════════════════════════════════════════════
def print_evidence_summary(evidence: dict):
    """Human-readable console summary of an Evidence Object."""
    sep = "─" * 60
    print(f"\n{'═' * 60}")
    print(f"  EVIDENCE SUMMARY — {evidence['document_id']}")
    print(f"{'═' * 60}")
    print(f"  Timestamp   : {evidence['timestamp_utc']}")
    print(f"  Layers      : {evidence['layers_completed']}/{evidence['layers_requested']} ok, {evidence['layers_failed']} failed")
    print(f"  All passed  : {'✅ YES' if evidence['all_passed'] else '❌ NO'}")
    print(f"  Elapsed     : {evidence['elapsed_seconds']:.3f}s")
    print(sep)

    for finding in evidence["findings"]:
        status = "✅" if finding.get("passed") else "❌"
        score  = finding.get("score", "?")
        layer  = finding.get("layer_name", "unknown")
        sev    = finding.get("severity", "-")
        conf   = finding.get("confidence", "-")
        expl   = finding.get("explanation", "")
        meta_status = finding.get("_engine_meta", {}).get("status", "?")
        meta_ms     = finding.get("_engine_meta", {}).get("elapsed_ms", "?")

        print(f"  {status} {layer}")
        print(f"     score={score}  severity={sev}  confidence={conf}  [{meta_status} {meta_ms}ms]")
        print(f"     → {expl}")
        print(sep)

    if evidence["errors"]:
        print("  ⚠️  ERRORS:")
        for err in evidence["errors"]:
            print(f"     • {err['layer_name']}: {err['traceback'].splitlines()[-1]}")
        print(sep)


# ═══════════════════════════════════════════════════════════════
#  CLI entrypoint — run on sample real & fake images
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Forensics Engine — run all forensic layers on a passport image."
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=None,
        help="Path to a single passport image. If omitted, runs a demo on 1 real + 1 fake from dataset/.",
    )
    parser.add_argument(
        "--batch-dir",
        default=None,
        help="Directory of images to process in batch mode.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not persist evidence JSON to disk.",
    )
    args = parser.parse_args()

    REALS_DIR = PROJECT_ROOT / "dataset" / "reals"
    FAKES_DIR = PROJECT_ROOT / "dataset" / "fakes"

    if args.batch_dir:
        # --- Batch mode ---
        all_ev = run_batch(args.batch_dir, save=not args.no_save)
        print(f"\n{'═' * 60}")
        print(f"  BATCH COMPLETE — {len(all_ev)} images processed")
        total_passed = sum(1 for e in all_ev if e["all_passed"])
        total_failed = len(all_ev) - total_passed
        print(f"  Passed: {total_passed}  |  Flagged: {total_failed}")
        print(f"{'═' * 60}")

    elif args.image:
        # --- Single image mode ---
        ev = run_forensics(args.image, save=not args.no_save)
        print_evidence_summary(ev)

    else:
        # --- Demo mode: 1 real + 1 fake ---
        print("=" * 60)
        print("  FORENSICS ENGINE — Demo Run")
        print("=" * 60)

        valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

        real_files = sorted(
            f for f in os.listdir(REALS_DIR)
            if Path(f).suffix.lower() in valid_ext
        ) if REALS_DIR.exists() else []

        fake_files = sorted(
            f for f in os.listdir(FAKES_DIR)
            if Path(f).suffix.lower() in valid_ext
        ) if FAKES_DIR.exists() else []

        if real_files:
            sample_real = str(REALS_DIR / real_files[0])
            print(f"\n▶ Running on REAL sample: {Path(sample_real).name}")
            ev_real = run_forensics(sample_real)
            print_evidence_summary(ev_real)

        if fake_files:
            sample_fake = str(FAKES_DIR / fake_files[0])
            print(f"\n▶ Running on FAKE sample: {Path(sample_fake).name}")
            ev_fake = run_forensics(sample_fake)
            print_evidence_summary(ev_fake)

        if not real_files and not fake_files:
            print("  No images found in dataset/reals/ or dataset/fakes/.")
            print("  Usage: python -m pipeline.forensics_engine <image_path>")
