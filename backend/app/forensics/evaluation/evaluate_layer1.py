"""
Independent Evaluation Harness for Layer 1
==========================================
Evaluates the locked Layer 1 forensic signal detector against held-out, unseen test data.

Forensic Evaluation Principles:
  1. Locked Artifacts: Uses frozen reference and thresholds (v2.1.0); never alters artifacts.
  2. Strict Partitioning: Evaluates the 15 held-out genuine test IDs and their corresponding fake attacks.
  3. Label-Blind Execution: Runs inference without providing ground-truth labels.
  4. Controlled Splicing Experiment: Tests localized portrait-only modification to verify Layer 1
     functions specifically as a background consistency detector (not a general classifier).
  5. Honest Reporting: Results are reported specifically as performance on the held-out dataset,
     with no unwarranted claims of real-world 100% generalization.
"""

from __future__ import annotations

import os
import sys
import re
import json
import glob
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from signals.background_texture import (
    ALGORITHM_VERSION,
    LAYER_NAME,
    evaluate_background_texture,
    load_calibration_artifacts,
    REALS_DIR,
    RESULTS_DIR,
    SPLIT_PATH,
    BACKGROUND_PATCH
)

FAKES_DIR = Path(__file__).resolve().parent.parent / 'dataset' / 'fakes'
REPORT_PATH = RESULTS_DIR / 'layer1_evaluation_report.json'


def get_base_id(filepath: str | Path) -> str | None:
    match = re.search(r'passport_(\d+)', os.path.basename(filepath))
    return match.group(1) if match else None


def run_evaluation() -> dict:
    """
    Executes the independent evaluation pipeline and returns the metrics dictionary.
    """
    print("=" * 70)
    print("INDEPENDENT EVALUATION — LAYER 1 (FROZEN v2.1.0)")
    print("=" * 70)

    # 1. Verify locked artifacts
    ref_int, ref_lbp, meta = load_calibration_artifacts()
    print(f"[Evaluation] Loaded locked Layer 1 artifacts (v{meta.get('algorithm_version')}).")
    print(f"[Evaluation] Calibration type: {meta.get('calibration_type')} | Locked: {meta.get('artifacts_locked')}")

    if not SPLIT_PATH.exists():
        raise FileNotFoundError(f"Dataset split file missing at {SPLIT_PATH}")

    with open(SPLIT_PATH, "r") as sf:
        split_info = json.load(sf)

    cal_base_ids = set(split_info["calibration_base_ids"])
    val_base_ids = set(split_info["validation_base_ids"])
    test_base_ids = set(split_info["test_base_ids"])

    # 2. Map dataset files
    real_files = sorted(glob.glob(str(REALS_DIR / "*.jpg")))
    fake_files = sorted(glob.glob(str(FAKES_DIR / "*.jpg")))

    reals_by_base = {get_base_id(f): f for f in real_files if get_base_id(f)}
    fakes_by_base = defaultdict(list)
    for f in fake_files:
        bid = get_base_id(f)
        if bid:
            fakes_by_base[bid].append(f)

    # 3. Evaluate Validation Set (15 Unseen Genuine Base IDs)
    val_results = []
    for bid in sorted(val_base_ids):
        fpath = reals_by_base.get(bid)
        if fpath:
            res = evaluate_background_texture(fpath)
            dist = res.get("raw_metric", {}).get("distance", res.get("distance"))
            val_results.append({
                "base_id": bid,
                "file": os.path.basename(fpath),
                "passed": res["passed"],
                "score": res["score"],
                "distance": dist
            })

    val_passed = sum(1 for r in val_results if r["passed"])
    val_accuracy = (val_passed / len(val_results)) if val_results else 0.0

    # 4. Evaluate Held-Out Final Test Set
    test_reals = [reals_by_base[bid] for bid in sorted(test_base_ids) if bid in reals_by_base]
    test_fakes = [f for bid in sorted(test_base_ids) for f in fakes_by_base.get(bid, [])]

    print(f"\n[Evaluation] Held-Out Test Set: {len(test_reals)} unseen genuine, {len(test_fakes)} unseen fakes.")

    test_eval_reals = []
    tn, fp = 0, 0
    for rpath in test_reals:
        res = evaluate_background_texture(rpath)
        dist = res.get("raw_metric", {}).get("distance", res.get("distance"))
        verdict = res.get("raw_metric", {}).get("verdict", res.get("verdict"))
        test_eval_reals.append({
            "base_id": get_base_id(rpath),
            "file": os.path.basename(rpath),
            "passed": res["passed"],
            "score": res["score"],
            "distance": dist,
            "verdict": verdict
        })
        if res["passed"]:
            tn += 1
        else:
            fp += 1

    test_eval_fakes = []
    tp, fn = 0, 0
    for fpath in test_fakes:
        res = evaluate_background_texture(fpath)
        dist = res.get("raw_metric", {}).get("distance", res.get("distance"))
        verdict = res.get("raw_metric", {}).get("verdict", res.get("verdict"))
        test_eval_fakes.append({
            "base_id": get_base_id(fpath),
            "file": os.path.basename(fpath),
            "passed": res["passed"],
            "score": res["score"],
            "distance": dist,
            "verdict": verdict,
            "explanation": res["explanation"]
        })
        if not res["passed"]:
            tp += 1
        else:
            fn += 1

    total_test = len(test_reals) + len(test_fakes)
    accuracy = (tp + tn) / total_test if total_test > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    # 5. Full Attack Corpus Evaluation (All 126 fakes)
    full_tp, full_fn = 0, 0
    full_fake_dists = []
    for fpath in fake_files:
        res = evaluate_background_texture(fpath)
        if not res["passed"]:
            full_tp += 1
        else:
            full_fn += 1
        dist = res.get("raw_metric", {}).get("distance", res.get("distance"))
        if dist is not None:
            full_fake_dists.append(dist)

    # 6. Controlled Experiment: Pure Localized Portrait Splice
    # Purpose: Verify whether Layer 1 is a background detector or a general classifier.
    # When ONLY the portrait is modified and the background is 100% untouched,
    # Layer 1 SHOULD pass (CONSISTENT_BACKGROUND), as the background is genuinely authentic.
    controlled_experiment = None
    if len(test_reals) >= 2:
        img_donor = cv2.imread(test_reals[0])
        img_source = cv2.imread(test_reals[1])
        h_d, w_d = img_donor.shape[:2]

        # Portrait box on AZE passport
        px, py, pw, ph = int(0.08 * w_d), int(0.35 * h_d), int(0.28 * w_d), int(0.45 * h_d)
        spliced = img_donor.copy()
        spliced[py:py+ph, px:px+pw] = img_source[py:py+ph, px:px+pw]

        res_splice = evaluate_background_texture(spliced)
        dist_splice = res_splice.get("raw_metric", {}).get("distance", res_splice.get("distance"))
        verdict_splice = res_splice.get("raw_metric", {}).get("verdict", res_splice.get("verdict"))
        controlled_experiment = {
            "description": "Synthetic localized photo-splice with 100% untouched background",
            "donor_base_id": get_base_id(test_reals[0]),
            "source_base_id": get_base_id(test_reals[1]),
            "layer1_verdict": verdict_splice,
            "layer1_passed": res_splice["passed"],
            "layer1_score": res_splice["score"],
            "layer1_distance": dist_splice,
            "interpretation": (
                "Layer 1 passed the untouched background, correctly functioning as a "
                "background-consistency detector. Upstream portrait alterations are designated "
                "to Layer 2 (Photo Boundary) and Layer 4 (ELA)."
            ) if res_splice["passed"] else "Layer 1 flagged background alteration."
        }

    # 7. Print Formatted Report
    print("\n" + "=" * 70)
    print("HELD-OUT TEST SET CONFUSION MATRIX (Unseen Base IDs)")
    print("=" * 70)
    print(f"                      Predicted Genuine    Predicted Fake")
    print(f"Actual Genuine ({len(test_reals):<3})      TN: {tn:<6}        FP: {fp:<6}")
    print(f"Actual Fake    ({len(test_fakes):<3})      FN: {fn:<6}        TP: {tp:<6}")
    print("-" * 70)
    print(f"Total Held-Out Evaluated : {total_test}")
    print(f"Accuracy                 : {accuracy * 100:.2f}%")
    print(f"Sensitivity / Recall     : {recall * 100:.2f}% (Fake Detection Rate)")
    print(f"Specificity              : {specificity * 100:.2f}% (Genuine Pass Rate)")
    print(f"Precision                : {precision * 100:.2f}%")
    print(f"F1-Score                 : {f1:.4f}")
    print(f"False Positive Rate (FPR): {fpr * 100:.2f}%")
    print(f"False Negative Rate (FNR): {fnr * 100:.2f}%")

    print("\n" + "=" * 70)
    print("FULL ATTACK CORPUS PERFORMANCE (126 fakes)")
    print("=" * 70)
    print(f"Total Fakes Tested       : {len(fake_files)}")
    print(f"Detected (TP)            : {full_tp}")
    print(f"Missed (FN)              : {full_fn}")
    print(f"Detection Rate           : {(full_tp / len(fake_files)) * 100:.2f}%")
    if full_fake_dists:
        print(f"Fake Distance Range      : [{min(full_fake_dists):.4f}, {max(full_fake_dists):.4f}] (Mean: {np.mean(full_fake_dists):.4f})")
        print(f"Calibrated Threshold     : {meta.get('metrics', {}).get('intensity_distribution', {}).get('threshold')}")

    if controlled_experiment:
        print("\n" + "=" * 70)
        print("CONTROLLED SPLICING EXPERIMENT (Portrait-Only Modification)")
        print("=" * 70)
        print(f"Untouched Background Result: {controlled_experiment['layer1_verdict']} (Passed: {controlled_experiment['layer1_passed']})")
        print(f"Forensic Insight: {controlled_experiment['interpretation']}")

    # 8. Save JSON Report
    report = {
        "evaluation_timestamp_utc": str(Path(__file__).stat().st_mtime),
        "algorithm_version": ALGORITHM_VERSION,
        "layer_name": LAYER_NAME,
        "dataset_split": {
            "calibration_base_count": len(cal_base_ids),
            "validation_base_count": len(val_base_ids),
            "test_base_count": len(test_base_ids)
        },
        "validation_set": {
            "count": len(val_results),
            "passed": val_passed,
            "accuracy": round(val_accuracy, 4)
        },
        "held_out_test_set": {
            "sample_counts": {
                "genuine": len(test_reals),
                "fake": len(test_fakes),
                "total": total_test
            },
            "confusion_matrix": {
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
                "true_positive": tp
            },
            "metrics": {
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall_sensitivity": round(recall, 4),
                "specificity": round(specificity, 4),
                "f1_score": round(f1, 4),
                "false_positive_rate": round(fpr, 4),
                "false_negative_rate": round(fnr, 4)
            },
            "methodology_note": (
                "Performance on the current held-out evaluation dataset (synthetic template-based). "
                "Real-world physical scans with diverse camera sensors, lighting, and wear will exhibit "
                "natural genuine variance and lower separation margins."
            )
        },
        "full_attack_corpus": {
            "total_fakes": len(fake_files),
            "detected": full_tp,
            "missed": full_fn,
            "detection_rate": round(full_tp / len(fake_files), 4) if fake_files else 0.0
        },
        "controlled_experiment": controlled_experiment
    }

    with open(REPORT_PATH, "w") as rf:
        json.dump(report, rf, indent=2)

    print(f"\n[Evaluation] Comprehensive report persisted to {REPORT_PATH}")
    return report


if __name__ == "__main__":
    run_evaluation()
