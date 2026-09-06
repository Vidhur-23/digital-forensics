"""
Layer 3: Color Consistency Analysis Signal Module
=================================================
Forensic statistical evaluation of document color distribution.
Extracts 6D LAB color signatures (mean & std across L, A, B channels)
and measures Mahalanobis distance from the genuine document centroid
using regularized covariance matrix inversion.
"""

import os
import json
import logging
from pathlib import Path
import cv2
import numpy as np

# ---------- SHARED UTILITIES ----------
from utils.file_grouping import get_image_files, save_embedding

# ---------- PATH CONFIG ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REALS_DIR = PROJECT_ROOT / 'dataset' / 'reals'
FAKES_DIR = PROJECT_ROOT / 'dataset' / 'fakes'
RESULTS_DIR = PROJECT_ROOT / 'results'
EMBEDDINGS_DIR = RESULTS_DIR / 'embeddings'
BASELINE_PATH = RESULTS_DIR / 'layer3_color_baseline.json'

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("color_consistency")


# ---------- CORE COLOR MEASUREMENT ----------
def extract_color_signature(image: np.ndarray) -> np.ndarray:
    """
    Converts image to LAB color space and extracts a 6-dimensional statistical vector:
    [L_mean, A_mean, B_mean, L_std, A_std, B_std]
    """
    if image is None or image.size == 0:
        return np.zeros(6, dtype=np.float64)

    bgr = image if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    l_channel = lab[:, :, 0].astype(np.float64)
    a_channel = lab[:, :, 1].astype(np.float64)
    b_channel = lab[:, :, 2].astype(np.float64)

    return np.array([
        float(np.mean(l_channel)),
        float(np.mean(a_channel)),
        float(np.mean(b_channel)),
        float(np.std(l_channel)),
        float(np.std(a_channel)),
        float(np.std(b_channel))
    ], dtype=np.float64)


def mahalanobis_distance(vec: np.ndarray, mean_vec: np.ndarray, cov_inv: np.ndarray) -> float:
    """
    Computes standard Mahalanobis distance from vector to centroid using inverse covariance matrix.
    sqrt((vec - mean) @ cov_inv @ (vec - mean).T)
    """
    diff = np.array(vec, dtype=np.float64) - np.array(mean_vec, dtype=np.float64)
    dist_sq = float(np.dot(np.dot(diff, cov_inv), diff.T))
    return float(np.sqrt(max(0.0, dist_sq)))


# ---------- CALIBRATION & TRAINING ----------
def build_genuine_baseline(reals_dir=REALS_DIR, save=True) -> dict:
    """
    Builds the genuine statistical color baseline from all images in dataset/reals/.
    Computes mean vector, regularized inverse covariance, and 95th percentile threshold.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    real_files = get_image_files(reals_dir)

    signatures = []
    for f in real_files:
        img_path = os.path.join(reals_dir, f)
        img = cv2.imread(img_path)
        if img is not None:
            signatures.append(extract_color_signature(img))

    if len(signatures) < 5:
        raise ValueError(f"Insufficient genuine samples to build baseline (found {len(signatures)}).")

    sig_matrix = np.array(signatures, dtype=np.float64)  # shape (N, 6)
    mean_vec = np.mean(sig_matrix, axis=0)  # shape (6,)
    cov_matrix = np.cov(sig_matrix, rowvar=False)  # shape (6, 6)

    # Regularize covariance matrix with small epsilon to ensure stability
    cov_reg = cov_matrix + 1e-6 * np.eye(6)
    cov_inv = np.linalg.pinv(cov_reg)

    # Compute distances for all genuine samples
    genuine_distances = [
        mahalanobis_distance(sig, mean_vec, cov_inv)
        for sig in signatures
    ]

    gen_dist_mean = float(np.mean(genuine_distances))
    gen_dist_std = float(np.std(genuine_distances))
    gen_dist_p95 = float(np.percentile(genuine_distances, 95))
    threshold = gen_dist_p95

    baseline = {
        "layer_name": "Layer 3: Color Consistency Analysis",
        "n_samples": len(signatures),
        "mean_vec": [round(float(x), 6) for x in mean_vec],
        "cov_inv": [[round(float(val), 8) for val in row] for row in cov_inv],
        "genuine_dist_mean": round(gen_dist_mean, 4),
        "genuine_dist_std": round(gen_dist_std, 4),
        "genuine_dist_p95": round(gen_dist_p95, 4),
        "threshold": round(threshold, 4)
    }

    if save:
        with open(BASELINE_PATH, "w") as f:
            json.dump(baseline, f, indent=4)
        print(f"[Layer 3 Baseline] Saved color baseline to {BASELINE_PATH}")

    return baseline


# ---------- SINGLE-IMAGE INFERENCE API ----------
def evaluate_color_consistency(image_input, baseline: dict = None) -> dict:
    """
    Evaluates whole-image LAB color consistency against the calibrated genuine baseline.
    Returns standard forensic schema dictionary.
    """
    # 1. Load baseline
    if baseline is None:
        if os.path.exists(BASELINE_PATH):
            with open(BASELINE_PATH, "r") as f:
                baseline = json.load(f)
        else:
            baseline = build_genuine_baseline()

    mean_vec = np.array(baseline["mean_vec"], dtype=np.float64)
    cov_inv = np.array(baseline["cov_inv"], dtype=np.float64)
    threshold = float(baseline.get("threshold", baseline.get("genuine_dist_p95", 3.0)))

    # 2. Extract Document ID and load image
    document_id = "unknown_document"
    if isinstance(image_input, (str, Path)):
        document_id = Path(image_input).stem
        img = cv2.imread(str(image_input))
        if img is None:
            return {
                "layer_name": "Layer 3: Color Consistency Analysis",
                "finding_type": "processing_error",
                "score": 1.0,
                "confidence": 0.0,
                "severity": "high",
                "passed": False,
                "raw_metric": {
                    "mahalanobis_distance": 0.0,
                    "threshold": threshold,
                    "color_signature": []
                },
                "explanation": f"Could not read image from {image_input}."
            }
    elif isinstance(image_input, np.ndarray):
        img = image_input
    else:
        return {
            "layer_name": "Layer 3: Color Consistency Analysis",
            "finding_type": "processing_error",
            "score": 1.0,
            "confidence": 0.0,
            "severity": "high",
            "passed": False,
            "raw_metric": {
                "mahalanobis_distance": 0.0,
                "threshold": threshold,
                "color_signature": []
            },
            "explanation": "Invalid input type. Expected file path or numpy image."
        }

    # 3. Extract signature and compute Mahalanobis distance
    sig = extract_color_signature(img)
    distance = mahalanobis_distance(sig, mean_vec, cov_inv)

    # 4. Scoring & Metrics
    score = min(distance / (threshold * 1.5), 1.0) if threshold > 0 else 1.0
    passed = distance <= threshold
    severity = "low" if score < 0.3 else ("medium" if score < 0.7 else "high")
    confidence = min(1.0, abs(distance - threshold) / (threshold + 1e-9))

    # 5. Deterministic explanation
    if passed:
        explanation = f"Color signature Mahalanobis distance is {distance:.2f} (threshold {threshold:.2f}) — consistent with genuine baseline."
    else:
        explanation = f"Color signature Mahalanobis distance is {distance:.2f} (threshold {threshold:.2f}) — deviates significantly from genuine baseline."

    # 6. Save embedding
    embedding_data = {
        "document_id": document_id,
        "mahalanobis_distance": round(distance, 4),
        "threshold": round(threshold, 4),
        "color_signature": [round(float(x), 4) for x in sig]
    }
    save_embedding(document_id, embedding_data, layer_slug="layer3_color_consistency")

    return {
        "layer_name": "Layer 3: Color Consistency Analysis",
        "finding_type": "color_consistency_anomaly",
        "score": round(float(score), 4),
        "confidence": round(float(confidence), 4),
        "severity": severity,
        "passed": bool(passed),
        "raw_metric": {
            "mahalanobis_distance": round(float(distance), 4),
            "threshold": round(float(threshold), 4),
            "color_signature": [round(float(x), 4) for x in sig]
        },
        "explanation": explanation
    }


# ---------- MAIN DIAGNOSTIC PIPELINE ----------
if __name__ == '__main__':
    print("=" * 50)
    print("CALIBRATING AND TESTING LAYER 3 (COLOR CONSISTENCY)")
    print("=" * 50)

    # 1. Build genuine baseline
    baseline = build_genuine_baseline()
    mean_vec = np.array(baseline["mean_vec"], dtype=np.float64)
    cov_inv = np.array(baseline["cov_inv"], dtype=np.float64)
    threshold = baseline["threshold"]

    # 2. Evaluate all real and fake files to compute diagnostic separation
    real_files = get_image_files(REALS_DIR)
    fake_files = get_image_files(FAKES_DIR)

    genuine_dists = []
    for f in real_files:
        img = cv2.imread(str(REALS_DIR / f))
        if img is not None:
            sig = extract_color_signature(img)
            genuine_dists.append(mahalanobis_distance(sig, mean_vec, cov_inv))

    fake_dists = []
    for f in fake_files:
        img = cv2.imread(str(FAKES_DIR / f))
        if img is not None:
            sig = extract_color_signature(img)
            fake_dists.append(mahalanobis_distance(sig, mean_vec, cov_inv))

    gen_mean = float(np.mean(genuine_dists)) if genuine_dists else 0.0
    gen_std = float(np.std(genuine_dists)) if genuine_dists else 0.0
    gen_max = float(np.max(genuine_dists)) if genuine_dists else 0.0

    fake_mean = float(np.mean(fake_dists)) if fake_dists else 0.0
    fake_std = float(np.std(fake_dists)) if fake_dists else 0.0
    fake_max = float(np.max(fake_dists)) if fake_dists else 0.0

    separation = fake_mean - gen_mean
    fp_rate = float(np.mean(np.array(genuine_dists) > threshold) * 100.0) if genuine_dists else 0.0
    fn_rate = float(np.mean(np.array(fake_dists) <= threshold) * 100.0) if fake_dists else 0.0

    print("\n" + "=" * 50)
    print("LAYER 3 (COLOR CONSISTENCY) DIAGNOSTIC REPORT")
    print("=" * 50)
    print(f"Genuine Samples: {len(genuine_dists)} | Fake Samples: {len(fake_dists)}")
    print(f"Genuine Mahalanobis Distances: mean={gen_mean:.4f}, std={gen_std:.4f}, max={gen_max:.4f}, 95th-pct={baseline['genuine_dist_p95']:.4f}")
    print(f"Fake Mahalanobis Distances:    mean={fake_mean:.4f}, std={fake_std:.4f}, max={fake_max:.4f}")
    print(f"Calibrated Threshold (p95):   {threshold:.4f}")
    print(f"Separation (Fake - Genuine):   {separation:.4f}")
    print(f"False Positive Rate (Genuine failing): {fp_rate:.2f}%")
    print(f"False Negative Rate (Fakes passing):   {fn_rate:.2f}%")

    if separation > gen_std and fn_rate < 50.0:
        print("\nVERDICT: Layer 3 shows USABLE separation between genuine and fake.")
    else:
        print("\nVERDICT: Layer 3 shows weak/no separation — whole-image color distribution alone does not reliably separate fakes from reals in this dataset.")
    print("=" * 50)

    # 3. Test on sample real and fake image
    if real_files:
        sample_real_path = str(REALS_DIR / real_files[0])
        doc_id_real = Path(sample_real_path).stem
        finding_real = evaluate_color_consistency(sample_real_path, baseline=baseline)
        print(f"\nReal Sample Finding ({doc_id_real}):")
        print(json.dumps(finding_real, indent=4))
        print(f"Embedding saved: results/embeddings/{doc_id_real}_layer3_color_consistency.json")

    if fake_files:
        sample_fake_path = str(FAKES_DIR / fake_files[0])
        doc_id_fake = Path(sample_fake_path).stem
        finding_fake = evaluate_color_consistency(sample_fake_path, baseline=baseline)
        print(f"\nFake Sample Finding ({doc_id_fake}):")
        print(json.dumps(finding_fake, indent=4))
        print(f"Embedding saved: results/embeddings/{doc_id_fake}_layer3_color_consistency.json")

    print("=" * 50)
