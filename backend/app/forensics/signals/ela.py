"""
Layer 4: Error Level Analysis (ELA) Signal Module
=================================================
Forensic detection of digital resaving and compression anomalies.
Resaves the input image at a specified JPEG quality and computes the
distribution of pixel-level differences against a genuine baseline
compression error band [threshold_min, threshold_max].
"""

import os
import json
import logging
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------- SHARED UTILITIES ----------
from utils.file_grouping import get_image_files, save_embedding
from utils.photo_localization import locate_photo_region

# ---------- PATH CONFIG ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REALS_DIR = PROJECT_ROOT / 'dataset' / 'reals'
FAKES_DIR = PROJECT_ROOT / 'dataset' / 'fakes'
RESULTS_DIR = PROJECT_ROOT / 'results'
EMBEDDINGS_DIR = RESULTS_DIR / 'embeddings'
METADATA_PATH = RESULTS_DIR / 'layer4_ela_metadata.json'
PLOT_PATH = RESULTS_DIR / 'layer4_ela_distribution.png'
SAMPLE_MAP_PATH = RESULTS_DIR / 'layer4_sample_ela_map.jpg'

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("ela")


# ---------- CORE ELA COMPUTATION ----------
def compute_ela_map(image: np.ndarray, quality: int = 90, scale: float = 15.0) -> tuple:
    """
    Computes Error Level Analysis (ELA) map by resaving at a specified JPEG quality
    and calculating the absolute pixel difference between original and recompressed.

    Returns:
        gray_diff (np.ndarray): Single-channel raw error values.
        amplified_map (np.ndarray): 3-channel visual error heatmap scaled for human/audit inspection.
    """
    if image is None or image.size == 0:
        return np.zeros((10, 10), dtype=np.uint8), np.zeros((10, 10, 3), dtype=np.uint8)

    bgr = image if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Encode to JPEG in memory buffer at target quality
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, enc = cv2.imencode('.jpg', bgr, encode_param)
    if not success:
        return np.zeros((10, 10), dtype=np.uint8), np.zeros((10, 10, 3), dtype=np.uint8)

    # Decode recompressed image
    recompressed = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    # Calculate absolute difference
    diff_bgr = cv2.absdiff(bgr, recompressed)
    gray_diff = cv2.cvtColor(diff_bgr, cv2.COLOR_BGR2GRAY)

    # Scale difference for visualization (amplified heatmap)
    amplified_map = np.clip(diff_bgr.astype(np.float32) * scale, 0, 255).astype(np.uint8)

    return gray_diff, amplified_map


def measure_ela(image: np.ndarray, quality: int = 90) -> dict:
    """
    Extracts comprehensive ELA statistics across the image and regional components.
    """
    gray_diff, _ = compute_ela_map(image, quality=quality)

    mean_error = float(np.mean(gray_diff))
    std_error = float(np.std(gray_diff))
    max_error = int(np.max(gray_diff))
    p95_error = float(np.percentile(gray_diff, 95))
    high_err_pct = float(np.mean(gray_diff > 10) * 100.0)

    result = {
        "mean_error": round(mean_error, 4),
        "std_error": round(std_error, 4),
        "max_error": max_error,
        "p95_error": round(p95_error, 4),
        "high_error_pct": round(high_err_pct, 4)
    }

    # Optional regional discrepancy if photo bbox is detected
    if locate_photo_region is not None:
        bbox = locate_photo_region(image)
        if bbox is not None:
            x, y, w, h = bbox
            photo_diff = gray_diff[y:y+h, x:x+w]
            mask = np.ones(gray_diff.shape, dtype=bool)
            mask[y:y+h, x:x+w] = False
            doc_diff = gray_diff[mask]

            photo_mean = float(np.mean(photo_diff)) if photo_diff.size > 0 else 0.0
            doc_mean = float(np.mean(doc_diff)) if doc_diff.size > 0 else 0.0
            ratio = float(photo_mean / (doc_mean + 1e-6))

            result["photo_mean_error"] = round(photo_mean, 4)
            result["doc_mean_error"] = round(doc_mean, 4)
            result["photo_doc_ratio"] = round(ratio, 4)

    return result


# ---------- CALIBRATION & TRAINING ----------
def calibrate_ela(reals_dir=REALS_DIR, fakes_dir=FAKES_DIR, save_artifacts=True) -> dict:
    """
    Calibrates genuine ELA error level baseline across real passports and verifies against fakes.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    real_files = get_image_files(reals_dir)
    fake_files = get_image_files(fakes_dir)

    print(f"[Layer 4 ELA Calibration] Processing {len(real_files)} real images, {len(fake_files)} fake images.")

    genuine_means = []
    genuine_stds = []
    for f in real_files:
        img = cv2.imread(os.path.join(reals_dir, f))
        if img is not None:
            m = measure_ela(img)
            genuine_means.append(m["mean_error"])
            genuine_stds.append(m["std_error"])

    fake_means = []
    fake_stds = []
    for f in fake_files:
        img = cv2.imread(os.path.join(fakes_dir, f))
        if img is not None:
            m = measure_ela(img)
            fake_means.append(m["mean_error"])
            fake_stds.append(m["std_error"])

    if len(genuine_means) < 5:
        raise ValueError(f"Insufficient genuine samples for ELA calibration (found {len(genuine_means)}).")

    gen_mean = float(np.mean(genuine_means))
    gen_std = float(np.std(genuine_means))
    gen_min = float(np.min(genuine_means))
    gen_max = float(np.max(genuine_means))

    fake_mean = float(np.mean(fake_means)) if fake_means else 0.0
    fake_std = float(np.std(fake_means)) if fake_means else 0.0
    fake_min = float(np.min(fake_means)) if fake_means else 0.0
    fake_max = float(np.max(fake_means)) if fake_means else 0.0

    # Calibrated genuine boundary: authentic passports cluster tightly around gen_mean
    # We define a genuine acceptance band with a 4-sigma boundary or min/max guard
    threshold_min = max(0.0, gen_mean - 4 * gen_std)
    threshold_max = gen_mean + 4 * gen_std

    separation = gen_mean - fake_mean
    fp_rate = float(np.mean((np.array(genuine_means) < threshold_min) | (np.array(genuine_means) > threshold_max)) * 100.0)
    fn_rate = float(np.mean((np.array(fake_means) >= threshold_min) & (np.array(fake_means) <= threshold_max)) * 100.0) if fake_means else 0.0

    metadata = {
        "layer_name": "Layer 4: Error Level Analysis (ELA)",
        "genuine_samples_count": len(genuine_means),
        "fake_samples_count": len(fake_means),
        "genuine_mean_error": round(gen_mean, 4),
        "genuine_std_error": round(gen_std, 4),
        "genuine_min_error": round(gen_min, 4),
        "genuine_max_error": round(gen_max, 4),
        "fake_mean_error": round(fake_mean, 4),
        "fake_std_error": round(fake_std, 4),
        "fake_min_error": round(fake_min, 4),
        "fake_max_error": round(fake_max, 4),
        "calibrated_threshold_min": round(threshold_min, 4),
        "calibrated_threshold_max": round(threshold_max, 4),
        "separation": round(separation, 4),
        "false_positive_rate": round(fp_rate, 2),
        "false_negative_rate": round(fn_rate, 2)
    }

    # Print Diagnostic Report
    print("\n" + "=" * 50)
    print("LAYER 4 (ERROR LEVEL ANALYSIS) DIAGNOSTIC REPORT")
    print("=" * 50)
    print(f"Genuine Samples: {len(genuine_means)} | Fake Samples: {len(fake_means)}")
    print(f"Genuine ELA Mean Error: mean={gen_mean:.4f}, std={gen_std:.4f}, min={gen_min:.4f}, max={gen_max:.4f}")
    print(f"Fake ELA Mean Error:    mean={fake_mean:.4f}, std={fake_std:.4f}, min={fake_min:.4f}, max={fake_max:.4f}")
    print(f"Calibrated Genuine Band: [{threshold_min:.4f}  <--->  {threshold_max:.4f}]")
    print(f"Separation (Genuine - Fake): {separation:.4f}")
    print(f"False Positive Rate (Genuine failing band): {fp_rate:.2f}%")
    print(f"False Negative Rate (Fakes slipping into band): {fn_rate:.2f}%")

    if separation > (gen_std * 3) and fn_rate < 10.0:
        print("\nVERDICT: Layer 4 (ELA) shows EXCEPTIONAL separation between genuine and fake.")
    else:
        print("\nVERDICT: Layer 4 (ELA) shows weak/no separation in this dataset.")
    print("=" * 50)

    if save_artifacts:
        with open(METADATA_PATH, "w") as mf:
            json.dump(metadata, mf, indent=4)

        # Plot distribution
        plt.figure(figsize=(8, 5))
        plt.hist(genuine_means, bins=15, alpha=0.6, label=f'Genuine ({len(genuine_means)})', color='green')
        if fake_means:
            plt.hist(fake_means, bins=15, alpha=0.6, label=f'Fake ({len(fake_means)})', color='red')
        plt.axvline(threshold_min, color='black', linestyle='--', label=f'Threshold Band [{threshold_min:.2f}, {threshold_max:.2f}]')
        plt.axvline(threshold_max, color='black', linestyle='--')
        plt.xlabel('Mean ELA Error at Q90')
        plt.ylabel('Document Count')
        plt.title('Layer 4: Error Level Analysis (ELA) Distribution')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.savefig(PLOT_PATH, bbox_inches='tight', dpi=150)
        plt.close()

        # Save sample visual amplified ELA map for sample real and fake
        if real_files:
            img_real = cv2.imread(os.path.join(reals_dir, real_files[0]))
            if img_real is not None:
                _, amp_map = compute_ela_map(img_real)
                cv2.imwrite(str(SAMPLE_MAP_PATH), amp_map)

        print(f"\n[Layer 4 ELA Calibration] Metadata saved to {METADATA_PATH}")
        print(f"[Layer 4 ELA Calibration] Plot saved to {PLOT_PATH}")
        print(f"[Layer 4 ELA Calibration] Sample ELA visual map saved to {SAMPLE_MAP_PATH}")

    return metadata


# ---------- SINGLE-IMAGE INFERENCE API ----------
def evaluate_ela(image_input, threshold_bounds: tuple = None) -> dict:
    """
    Evaluates Error Level Analysis (ELA) recompression discrepancy on a single passport image.
    Returns standard forensic schema dictionary.
    """
    # 1. Load thresholds
    if threshold_bounds is None:
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, "r") as mf:
                meta = json.load(mf)
                t_min = float(meta.get("calibrated_threshold_min", 1.31))
                t_max = float(meta.get("calibrated_threshold_max", 1.42))
                gen_mean = float(meta.get("genuine_mean_error", 1.366))
        else:
            meta = calibrate_ela()
            t_min = float(meta["calibrated_threshold_min"])
            t_max = float(meta["calibrated_threshold_max"])
            gen_mean = float(meta["genuine_mean_error"])
    else:
        t_min, t_max = threshold_bounds
        gen_mean = (t_min + t_max) / 2.0

    # 2. Extract Document ID and load image
    document_id = "unknown_document"
    if isinstance(image_input, (str, Path)):
        document_id = Path(image_input).stem
        img = cv2.imread(str(image_input))
        if img is None:
            return {
                "layer_name": "Layer 4: Error Level Analysis (ELA)",
                "finding_type": "processing_error",
                "score": 1.0,
                "confidence": 0.0,
                "severity": "high",
                "passed": False,
                "raw_metric": {
                    "mean_error": 0.0,
                    "threshold_min": t_min,
                    "threshold_max": t_max
                },
                "explanation": f"Could not read image from {image_input}."
            }
    elif isinstance(image_input, np.ndarray):
        img = image_input
    else:
        return {
            "layer_name": "Layer 4: Error Level Analysis (ELA)",
            "finding_type": "processing_error",
            "score": 1.0,
            "confidence": 0.0,
            "severity": "high",
            "passed": False,
            "raw_metric": {
                "mean_error": 0.0,
                "threshold_min": t_min,
                "threshold_max": t_max
            },
            "explanation": "Invalid input type. Expected file path or numpy image."
        }

    # 3. Measure ELA metrics
    measurement = measure_ela(img)
    mean_err = measurement["mean_error"]

    # 4. Check if within genuine band
    passed = bool(t_min <= mean_err <= t_max)

    # 5. Anomaly distance & scoring
    if passed:
        dev = abs(mean_err - gen_mean)
        max_allowed_dev = max(abs(t_max - gen_mean), abs(t_min - gen_mean), 0.01)
        score = min(dev / (max_allowed_dev * 1.5), 0.4)
        severity = "low"
        confidence = min(1.0, min(abs(mean_err - t_min), abs(mean_err - t_max)) / 0.1)
    else:
        # Distance outside the valid band
        dist_out = t_min - mean_err if mean_err < t_min else mean_err - t_max
        score = min(0.5 + (dist_out / 0.3) * 0.5, 1.0)
        severity = "high" if score >= 0.7 else "medium"
        confidence = min(1.0, dist_out / 0.1)

    # 6. Deterministic explanation
    if passed:
        explanation = f"Error Level Analysis error ({mean_err:.3f}) matches genuine document compression profile (band [{t_min:.2f} - {t_max:.2f}])."
    else:
        if mean_err < t_min:
            explanation = f"Error Level Analysis error ({mean_err:.3f}) is unnaturally low compared to genuine baseline ([{t_min:.2f} - {t_max:.2f}]) — indicates digital resaving/synthetic generation."
        else:
            explanation = f"Error Level Analysis error ({mean_err:.3f}) exceeds genuine baseline ([{t_min:.2f} - {t_max:.2f}]) — indicates high-frequency manipulation or localized resaving."

    # 7. Save embedding
    embedding_data = {
        "document_id": document_id,
        "mean_error": mean_err,
        "std_error": measurement["std_error"],
        "max_error": measurement["max_error"],
        "p95_error": measurement["p95_error"],
        "high_error_pct": measurement["high_error_pct"],
        "threshold_min": t_min,
        "threshold_max": t_max
    }
    if "photo_doc_ratio" in measurement:
        embedding_data["photo_doc_ratio"] = measurement["photo_doc_ratio"]

    save_embedding(document_id, embedding_data, layer_slug="layer4_ela")

    return {
        "layer_name": "Layer 4: Error Level Analysis (ELA)",
        "finding_type": "compression_artifact_anomaly",
        "score": round(float(score), 4),
        "confidence": round(float(confidence), 4),
        "severity": severity,
        "passed": bool(passed),
        "raw_metric": {
            "mean_error": round(mean_err, 4),
            "std_error": measurement["std_error"],
            "max_error": measurement["max_error"],
            "p95_error": measurement["p95_error"],
            "high_error_pct": measurement["high_error_pct"],
            "threshold_min": round(t_min, 4),
            "threshold_max": round(t_max, 4)
        },
        "explanation": explanation
    }


# ---------- MAIN PIPELINE ----------
if __name__ == '__main__':
    print("=" * 50)
    print("CALIBRATING AND TESTING LAYER 4 (ERROR LEVEL ANALYSIS)")
    print("=" * 50)

    meta = calibrate_ela()

    print("\nCalibrated Metadata:")
    print(json.dumps(meta, indent=2))

    real_files = get_image_files(REALS_DIR)
    fake_files = get_image_files(FAKES_DIR)

    if real_files:
        sample_real_path = str(REALS_DIR / real_files[0])
        doc_id_real = Path(sample_real_path).stem
        finding_real = evaluate_ela(sample_real_path)
        print(f"\nReal Sample Finding ({doc_id_real}):")
        print(json.dumps(finding_real, indent=4))
        print(f"Embedding saved: results/embeddings/{doc_id_real}_layer4_ela.json")

    if fake_files:
        sample_fake_path = str(FAKES_DIR / fake_files[0])
        doc_id_fake = Path(sample_fake_path).stem
        finding_fake = evaluate_ela(sample_fake_path)
        print(f"\nFake Sample Finding ({doc_id_fake}):")
        print(json.dumps(finding_fake, indent=4))
        print(f"Embedding saved: results/embeddings/{doc_id_fake}_layer4_ela.json")

    print("=" * 50)
