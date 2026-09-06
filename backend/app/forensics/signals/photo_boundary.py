"""
Layer 2: Photo Boundary & Seam Straightness Analysis Signal Module
==================================================================
Forensic detection of digital photo splicing and perimeter paste lines.
Analyzes the bounding boundary around localized portrait photographs for
abnormally long, straight edge segments using Canny edge detection and
probabilistic Hough transform lines.
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

# Re-export for backward compatibility
__all__ = ["locate_photo_region", "evaluate_photo_boundary", "calibrate_layer2"]

# ---------- PATH CONFIG ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REALS_DIR = PROJECT_ROOT / 'dataset' / 'reals'
FAKES_DIR = PROJECT_ROOT / 'dataset' / 'fakes'
RESULTS_DIR = PROJECT_ROOT / 'results'
EMBEDDINGS_DIR = RESULTS_DIR / 'embeddings'
METADATA_PATH = RESULTS_DIR / 'layer2_metadata.json'
PLOT_PATH = RESULTS_DIR / 'layer2_photo_boundary.png'
VISUALIZATION_PATH = RESULTS_DIR / 'layer2_boundary_visualization.jpg'

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("photo_boundary")


# ---------- CORE BOUNDARY MEASUREMENT ----------
def measure_boundary_straightness(image: np.ndarray, photo_bbox: tuple, padding: int = 15) -> dict:
    """
    Crops the photo region with padding and measures long straight boundary lines via HoughLinesP.
    Returns dict: {"long_straight_line_count": int, "total_lines_detected": int, "lines": list}
    """
    if image is None or photo_bbox is None:
        return {"long_straight_line_count": 0, "total_lines_detected": 0, "lines": []}

    img_h, img_w = image.shape[:2]
    x, y, w, h = photo_bbox

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img_w, x + w + padding)
    y2 = min(img_h, y + h + padding)

    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        return {"long_straight_line_count": 0, "total_lines_detected": 0, "lines": []}

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY) if len(cropped.shape) == 3 else cropped
    edges = cv2.Canny(gray, 50, 150)

    min_line_length = 0.7 * w
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=50,
                            minLineLength=min_line_length, maxLineGap=10)

    if lines is None or len(lines) == 0:
        return {"long_straight_line_count": 0, "total_lines_detected": 0, "lines": []}

    lines = lines.reshape(-1, 4)
    total_lines_detected = len(lines)
    long_straight_lines = []
    long_threshold_len = 0.8 * w

    for lx1, ly1, lx2, ly2 in lines:
        length = float(np.sqrt((lx2 - lx1) ** 2 + (ly2 - ly1) ** 2))
        if length > long_threshold_len:
            # Store line coordinates mapped back to full image coordinate space
            long_straight_lines.append({
                "x1": int(lx1 + x1),
                "y1": int(ly1 + y1),
                "x2": int(lx2 + x1),
                "y2": int(ly2 + y1),
                "length": float(length)
            })

    return {
        "long_straight_line_count": len(long_straight_lines),
        "total_lines_detected": total_lines_detected,
        "lines": long_straight_lines
    }


# ---------- CALIBRATION & TRAINING ----------
def calibrate_layer2(reals_dir=REALS_DIR, fakes_dir=FAKES_DIR, save_artifacts=True) -> tuple:
    """
    Calibrates Layer 2 threshold using genuine passport images.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    real_files = get_image_files(reals_dir)
    fake_files = get_image_files(fakes_dir)

    print(f"[Layer 2 Calibration] Processing {len(real_files)} real images, {len(fake_files)} fake images.")

    genuine_counts = []
    for f in real_files:
        img_path = os.path.join(reals_dir, f)
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"Could not read image: {img_path}")
            continue
        bbox = locate_photo_region(img)
        if bbox is None:
            logger.warning(f"Photo region not found in genuine image: {f}")
            continue
        meas = measure_boundary_straightness(img, bbox)
        genuine_counts.append(meas["long_straight_line_count"])

    fake_counts = []
    for f in fake_files:
        img_path = os.path.join(fakes_dir, f)
        img = cv2.imread(img_path)
        if img is None:
            continue
        bbox = locate_photo_region(img)
        if bbox is None:
            continue
        meas = measure_boundary_straightness(img, bbox)
        fake_counts.append(meas["long_straight_line_count"])

    if len(genuine_counts) < 5:
        raise ValueError(f"Insufficient genuine samples for Layer 2 calibration (found {len(genuine_counts)}).")

    gen_mean = float(np.mean(genuine_counts))
    gen_std = float(np.std(genuine_counts))
    gen_max = int(np.max(genuine_counts))

    fake_mean = float(np.mean(fake_counts)) if len(fake_counts) > 0 else 0.0
    fake_std = float(np.std(fake_counts)) if len(fake_counts) > 0 else 0.0
    fake_max = int(np.max(fake_counts)) if len(fake_counts) > 0 else 0

    calibrated_threshold = float(max(gen_mean + 3 * gen_std, gen_max * 1.5, 1.0))
    separation = float(fake_mean - gen_mean)

    genuine_arr = np.array(genuine_counts)
    fake_arr = np.array(fake_counts) if len(fake_counts) > 0 else np.array([])

    genuine_zero_pct = float(np.mean(genuine_arr == 0) * 100.0) if len(genuine_arr) > 0 else 0.0
    fake_zero_pct = float(np.mean(fake_arr == 0) * 100.0) if len(fake_arr) > 0 else 0.0

    false_positive_rate = float(np.mean(genuine_arr > calibrated_threshold) * 100.0) if len(genuine_arr) > 0 else 0.0
    false_negative_rate = float(np.mean(fake_arr <= calibrated_threshold) * 100.0) if len(fake_arr) > 0 else 0.0

    metadata = {
        "layer_name": "Layer 2: Photo Boundary Analysis",
        "genuine_samples_count": len(genuine_counts),
        "fake_samples_count": len(fake_counts),
        "genuine_count_mean": gen_mean,
        "genuine_count_std": gen_std,
        "genuine_count_max": gen_max,
        "fake_count_mean": fake_mean,
        "fake_count_std": fake_std,
        "fake_count_max": fake_max,
        "calibrated_threshold": calibrated_threshold,
        "separation": separation,
        "genuine_zero_pct": genuine_zero_pct,
        "fake_zero_pct": fake_zero_pct,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate
    }

    # Print Diagnostic Report
    print("\n" + "=" * 50)
    print("LAYER 2 DIAGNOSTIC REPORT")
    print("=" * 50)
    print(f"Genuine Samples: {len(genuine_counts)} | Fake Samples: {len(fake_counts)}")
    print(f"Genuine Long Edges: mean={gen_mean:.4f}, std={gen_std:.4f}, max={gen_max}")
    print(f"Fake Long Edges:    mean={fake_mean:.4f}, std={fake_std:.4f}, max={fake_max}")
    print(f"Separation (Fake - Genuine): {separation:.4f}")
    print(f"Genuine Images with 0 long edges: {genuine_zero_pct:.2f}%")
    print(f"Fake Images with 0 long edges:    {fake_zero_pct:.2f}%")
    print(f"False Positive Rate (Genuine failing threshold {calibrated_threshold:.2f}): {false_positive_rate:.2f}%")
    print(f"False Negative Rate (Fakes passing threshold {calibrated_threshold:.2f}):    {false_negative_rate:.2f}%")

    if separation > gen_std and false_negative_rate < 50.0:
        print("\nVERDICT: Layer 2 shows usable separation between genuine and fake.")
    else:
        print("\nVERDICT: Layer 2 shows weak/no separation — long-straight-edge counting alone does not reliably distinguish genuine from fake photos in this dataset.")
    print("=" * 50)

    if save_artifacts:
        with open(METADATA_PATH, "w") as mf:
            json.dump(metadata, mf, indent=4)

        # Generate and save histogram plot
        plt.figure(figsize=(8, 5))
        bins = np.arange(0, max(max(genuine_counts or [0]), max(fake_counts or [0]), int(calibrated_threshold) + 2) + 1.5) - 0.5
        plt.hist(genuine_counts, bins=bins, alpha=0.6, label=f'Genuine ({len(genuine_counts)})', color='green', rwidth=0.8)
        if len(fake_counts) > 0:
            plt.hist(fake_counts, bins=bins, alpha=0.6, label=f'Fake ({len(fake_counts)})', color='red', rwidth=0.8)
        plt.axvline(calibrated_threshold, color='black', linestyle='--', label=f'Threshold ({calibrated_threshold:.2f})')
        plt.xlabel('Long Straight Edge Count')
        plt.ylabel('Document Count')
        plt.title('Layer 2: Photo Boundary Seam Detection')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.savefig(PLOT_PATH, bbox_inches='tight', dpi=150)
        plt.close()

        print(f"\n[Layer 2 Calibration] Metadata saved to {METADATA_PATH}")
        print(f"[Layer 2 Calibration] Plot saved to {PLOT_PATH}")

    return calibrated_threshold, metadata


# ---------- SINGLE-IMAGE INFERENCE API ----------
def evaluate_photo_boundary(image_input, threshold: float = None) -> dict:
    """
    Evaluates photo boundary straightness on a single passport image.
    Returns standard forensic schema dictionary.
    """
    # 1. Load threshold
    if threshold is None:
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, "r") as mf:
                meta = json.load(mf)
                threshold = float(meta.get("calibrated_threshold", 1.0))
        else:
            threshold, _ = calibrate_layer2()

    # 2. Extract Document ID and load image
    document_id = "unknown_document"
    if isinstance(image_input, (str, Path)):
        document_id = Path(image_input).stem
        img = cv2.imread(str(image_input))
        if img is None:
            return {
                "layer_name": "Layer 2: Photo Boundary Analysis",
                "finding_type": "processing_error",
                "score": 1.0,
                "confidence": 0.0,
                "severity": "high",
                "passed": False,
                "raw_metric": {
                    "long_straight_line_count": 0,
                    "total_lines_detected": 0,
                    "threshold": threshold,
                    "region": {
                        "bbox": None,
                        "description": "Portrait photo boundary and surrounding seam area"
                    }
                },
                "explanation": f"Could not read image from {image_input}."
            }
    elif isinstance(image_input, np.ndarray):
        img = image_input
    else:
        return {
            "layer_name": "Layer 2: Photo Boundary Analysis",
            "finding_type": "processing_error",
            "score": 1.0,
            "confidence": 0.0,
            "severity": "high",
            "passed": False,
            "raw_metric": {
                "long_straight_line_count": 0,
                "total_lines_detected": 0,
                "threshold": threshold,
                "region": {
                    "bbox": None,
                    "description": "Portrait photo boundary and surrounding seam area"
                }
            },
            "explanation": "Invalid input type. Expected file path or numpy image."
        }

    # 3. Locate photo region
    photo_bbox = locate_photo_region(img)
    if photo_bbox is None:
        return {
            "layer_name": "Layer 2: Photo Boundary Analysis",
            "finding_type": "processing_error",
            "score": 1.0,
            "confidence": 0.0,
            "severity": "high",
            "passed": False,
            "raw_metric": {
                "long_straight_line_count": 0,
                "total_lines_detected": 0,
                "threshold": threshold,
                "region": {
                    "bbox": None,
                    "description": "Portrait photo boundary and surrounding seam area"
                }
            },
            "explanation": "Could not locate photo region for boundary analysis."
        }

    # 4. Measure boundary straightness
    measurement = measure_boundary_straightness(img, photo_bbox)
    line_count = measurement["long_straight_line_count"]

    # 5. Save embedding
    clean_embedding = {
        "document_id": document_id,
        "long_straight_line_count": line_count,
        "total_lines_detected": measurement.get("total_lines_detected", 0),
        "lines": measurement.get("lines", [])
    }
    save_embedding(document_id, clean_embedding, layer_slug="layer2_photo_boundary")

    # 6. Scoring & Metrics
    score = min(line_count / (threshold * 2.0), 1.0) if threshold > 0 else 1.0
    passed = line_count <= threshold
    severity = "low" if score < 0.3 else ("medium" if score < 0.7 else "high")
    confidence = min(1.0, abs(line_count - threshold) / (threshold + 1e-9))

    # 7. Deterministic explanation
    if passed:
        explanation = f"Photo boundary shows {line_count} long straight edge(s), within the normal range for genuine documents (threshold {threshold})."
    else:
        explanation = f"Photo boundary shows {line_count} long straight edge(s), exceeding the normal range for genuine documents (threshold {threshold}) — consistent with a possible pasted or spliced photo region."

    return {
        "layer_name": "Layer 2: Photo Boundary Analysis",
        "finding_type": "photo_boundary_anomaly",
        "score": round(float(score), 4),
        "confidence": round(float(confidence), 4),
        "severity": severity,
        "passed": bool(passed),
        "raw_metric": {
            "long_straight_line_count": int(line_count),
            "total_lines_detected": int(measurement["total_lines_detected"]),
            "threshold": float(threshold),
            "region": {
                "bbox": list(photo_bbox),
                "description": "Portrait photo boundary and surrounding seam area"
            }
        },
        "explanation": explanation
    }


# ---------- VISUALIZATION HELPER ----------
def save_boundary_visualization(image_path: str, output_path: str = str(VISUALIZATION_PATH)):
    """
    Saves a visualization showing the detected photo_bbox (red rectangle) and any detected long straight lines (yellow).
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return
    bbox = locate_photo_region(img)
    preview = img.copy()

    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red bbox

        meas = measure_boundary_straightness(img, bbox)
        for l in meas.get("lines", []):
            cv2.line(preview, (l["x1"], l["y1"]), (l["x2"], l["y2"]), (0, 255, 255), 2)  # Yellow lines

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, preview)


# ---------- MAIN PIPELINE ----------
if __name__ == '__main__':
    print("=" * 50)
    print("CALIBRATING AND TESTING LAYER 2 (PHOTO BOUNDARY)")
    print("=" * 50)

    thresh, meta = calibrate_layer2()

    print("\nCalibrated Metadata:")
    print(json.dumps(meta, indent=2))

    real_files = get_image_files(REALS_DIR)
    fake_files = get_image_files(FAKES_DIR)

    if real_files:
        sample_real_path = str(REALS_DIR / real_files[0])
        doc_id_real = Path(sample_real_path).stem
        save_boundary_visualization(sample_real_path)
        print(f"\nSaved boundary visualization to: {VISUALIZATION_PATH}")

        finding_real = evaluate_photo_boundary(sample_real_path)
        print(f"\nReal Sample Finding ({doc_id_real}):")
        print(json.dumps(finding_real, indent=4))
        print(f"Embedding saved: results/embeddings/{doc_id_real}_layer2_photo_boundary.json")

    if fake_files:
        sample_fake_path = None
        for f in fake_files:
            p = str(FAKES_DIR / f)
            img = cv2.imread(p)
            if img is not None and locate_photo_region(img) is not None:
                sample_fake_path = p
                break
        if sample_fake_path is None:
            sample_fake_path = str(FAKES_DIR / fake_files[0])

        doc_id_fake = Path(sample_fake_path).stem
        finding_fake = evaluate_photo_boundary(sample_fake_path)
        print(f"\nFake Sample Finding ({doc_id_fake}):")
        print(json.dumps(finding_fake, indent=4))
        print(f"Embedding saved: results/embeddings/{doc_id_fake}_layer2_photo_boundary.json")

    print("=" * 50)
