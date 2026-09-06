"""
Layer 1: Background Intensity, Micro-Texture & Structural Consistency Signal Module
===================================================================================
Forensic evaluation of passport security background patterns (guilloche,
micro-printing, decorative borders) through multi-feature analysis:
  1. Intensity distribution consistency (Weighted 1D Wasserstein distance)
  2. Spatial micro-texture consistency (Local Binary Patterns & Chi-square distance)
  3. Structural edge/gradient consistency (Sobel gradient magnitude & variance)

Follows a strict one-class genuine-only calibration architecture:
  - Reference baselines and decision thresholds are calibrated SOLELY on genuine documents.
  - Fake images are NEVER accessed, loaded, or used during calibration or thresholding.
  - Calibration artifacts are versioned and locked.
  - Inference is strictly label-blind and NEVER silently re-calibrates.
"""

from __future__ import annotations

import os
import sys
import re
import json
import argparse
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

# ---------- VERSIONING & TERMINOLOGY ----------
ALGORITHM_VERSION = "2.1.0"
METRIC_VERSION = "2.1.0"
LAYER_NAME = "Layer 1: Background Intensity, Micro-Texture & Structural Consistency"

# ---------- PATH CONFIG ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REALS_DIR = PROJECT_ROOT / 'dataset' / 'reals'
FAKES_DIR = PROJECT_ROOT / 'dataset' / 'fakes'
RESULTS_DIR = PROJECT_ROOT / 'results'

from utils.file_grouping import get_image_files

SPLIT_PATH = RESULTS_DIR / 'dataset_split.json'
REFERENCE_INTENSITY_HIST_PATH = RESULTS_DIR / 'layer1_reference_intensity_histogram.npy'
REFERENCE_LBP_HIST_PATH = RESULTS_DIR / 'layer1_reference_lbp_histogram.npy'
METADATA_PATH = RESULTS_DIR / 'layer1_metadata.json'
PLOT_PATH = RESULTS_DIR / 'layer1_calibration_plots.png'

# Compatibility alias
REFERENCE_HIST_PATH = REFERENCE_INTENSITY_HIST_PATH

# Background patch: (x_pct, y_pct, w_pct, h_pct)
# Verified against AZE passport top decorative border strip
BACKGROUND_PATCH = (0.05, 0.01, 0.85, 0.045)

# Predefined physical guard-band floors for scanner/sensor noise tolerance
PREDEFINED_GUARD_BAND_FLOORS = {
    "intensity": 0.45,
    "lbp": 0.10,
    "gradient": 1.50
}


# ---------- CORE MATHEMATICAL METRICS ----------
def validate_histogram(hist: np.ndarray, expected_bins: int = 256) -> np.ndarray:
    """
    Validates and normalizes a 1D probability mass histogram.
    Raises ValueError on malformed histograms (NaN, Inf, negative, wrong size, zero sum).
    """
    if hist is None:
        raise ValueError("Histogram cannot be None.")
    if not isinstance(hist, np.ndarray):
        hist = np.asarray(hist, dtype=np.float64)
    if hist.ndim != 1 or len(hist) != expected_bins:
        raise ValueError(f"Histogram must be 1-dimensional with {expected_bins} bins (got shape {hist.shape}).")
    if not np.all(np.isfinite(hist)):
        raise ValueError("Histogram contains NaN or Inf values.")
    if np.any(hist < 0):
        raise ValueError("Histogram contains negative frequencies.")
    
    total = float(np.sum(hist))
    if total <= 0:
        raise ValueError("Histogram has zero total probability mass.")
    
    return hist / total


def histogram_wasserstein_distance(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    """
    Computes true 1D Earth Mover's (Wasserstein-1) Distance between two 256-bin histograms.
    Treats bin indices (0..255) as sample coordinates and histogram values as weights:
        W_1(u, v) = scipy.stats.wasserstein_distance(bins, bins, u_weights=hist_a, v_weights=hist_b)
    """
    h_a = validate_histogram(hist_a, 256)
    h_b = validate_histogram(hist_b, 256)
    bins = np.arange(256, dtype=np.float64)
    return float(wasserstein_distance(bins, bins, u_weights=h_a, v_weights=h_b))


def chi_square_distance(hist_a: np.ndarray, hist_b: np.ndarray, eps: float = 1e-10) -> float:
    """
    Computes Chi-Square distance between two categorical/texture histograms (e.g. LBP).
        D_{chi^2}(P, Q) = 0.5 * sum( (P_i - Q_i)^2 / (P_i + Q_i + eps) )
    Range: [0.0, 1.0] for normalized probability distributions.
    """
    h_a = validate_histogram(hist_a, len(hist_a))
    h_b = validate_histogram(hist_b, len(hist_b))
    return float(0.5 * np.sum(((h_a - h_b) ** 2) / (h_a + h_b + eps)))


# ---------- FEATURE EXTRACTION & ROI ----------
def extract_roi(image: np.ndarray, patch_coords: tuple = BACKGROUND_PATCH) -> tuple[np.ndarray | None, dict]:
    """
    Extracts and boundary-clamps the specified background region of interest.
    Returns (cropped_roi, roi_metadata).
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return None, {
            "status": "invalid_image",
            "error": "Image is None or empty array."
        }
    
    if image.ndim not in (2, 3):
        return None, {
            "status": "invalid_dimensions",
            "error": f"Image must be 2D or 3D array (got ndim={image.ndim})."
        }

    h_img, w_img = image.shape[:2]
    if h_img < 50 or w_img < 50:
        return None, {
            "status": "image_too_small",
            "error": f"Image dimensions ({w_img}x{h_img}) are below minimum 50x50 threshold."
        }

    x_pct, y_pct, w_pct, h_pct = patch_coords
    x = max(0, min(int(x_pct * w_img), w_img - 1))
    y = max(0, min(int(y_pct * h_img), h_img - 1))
    w = max(1, min(int(w_pct * w_img), w_img - x))
    h = max(1, min(int(h_pct * h_img), h_img - y))

    if w < 10 or h < 5:
        return None, {
            "status": "roi_too_small",
            "error": f"Computed ROI dimensions ({w}x{h}) are too small for analysis."
        }

    roi = image[y:y+h, x:x+w]
    roi_info = {
        "status": "ok",
        "x": int(x),
        "y": int(y),
        "width": int(w),
        "height": int(h),
        "pixel_count": int(w * h),
        "template": "aze_passport"
    }
    return roi, roi_info


def assess_roi_quality(roi_gray: np.ndarray) -> dict:
    """
    Assesses ROI image quality: contrast, saturation/clipping, and sharpness.
    Returns quality diagnostic dictionary with 'status' in ['acceptable', 'insufficient_quality', 'unusable'].
    Clearly distinguishes an unusable image from an anomalous forensic pattern.
    """
    if roi_gray is None or roi_gray.size == 0:
        return {
            "status": "unusable",
            "contrast": 0.0,
            "clipping_pct": 100.0,
            "blur_score": 0.0,
            "error": "ROI is empty."
        }

    pixel_count = int(roi_gray.size)
    contrast = float(np.std(roi_gray))
    
    # Clipping / saturation: percentage of pixels that are 0 or 255
    clipped_count = int(np.sum((roi_gray <= 1) | (roi_gray >= 254)))
    clipping_pct = float((clipped_count / pixel_count) * 100.0)

    # Sharpness via variance of Laplacian
    blur_score = float(cv2.Laplacian(roi_gray, cv2.CV_64F).var())

    if pixel_count < 200 or contrast < 0.1 or clipping_pct > 85.0:
        status = "unusable"
    elif contrast < 1.0 or clipping_pct > 35.0 or blur_score < 1.5:
        status = "insufficient_quality"
    else:
        status = "acceptable"

    return {
        "status": status,
        "contrast": round(contrast, 4),
        "clipping_pct": round(clipping_pct, 4),
        "blur_score": round(blur_score, 4),
        "pixel_count": pixel_count
    }


def extract_intensity_histogram(roi_gray: np.ndarray) -> np.ndarray | None:
    """Extracts a 256-bin normalized intensity distribution histogram from a grayscale ROI."""
    if roi_gray is None or roi_gray.size == 0:
        return None
    hist, _ = np.histogram(roi_gray, bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    total = float(hist.sum())
    if total <= 0:
        return None
    return hist / total


def compute_lbp_histogram(roi_gray: np.ndarray) -> np.ndarray | None:
    """
    Computes a deterministic 256-bin Local Binary Pattern (LBP) histogram using pure NumPy.
    Extracts micro-texture characteristics using an 8-neighborhood invariant to monotonic illumination shifts.
    """
    if roi_gray is None or roi_gray.size == 0:
        return None
    
    h, w = roi_gray.shape[:2]
    if h < 3 or w < 3:
        return None

    center = roi_gray[1:h-1, 1:w-1]
    neighbors = [
        roi_gray[0:h-2, 0:w-2],  # Top-left (dr=-1, dc=-1)
        roi_gray[0:h-2, 1:w-1],  # Top (dr=-1, dc=0)
        roi_gray[0:h-2, 2:w],    # Top-right (dr=-1, dc=1)
        roi_gray[1:h-1, 2:w],    # Right (dr=0, dc=1)
        roi_gray[2:h,   2:w],    # Bottom-right (dr=1, dc=1)
        roi_gray[2:h,   1:w-1],  # Bottom (dr=1, dc=0)
        roi_gray[2:h,   0:w-2],  # Bottom-left (dr=1, dc=-1)
        roi_gray[1:h-1, 0:w-2],  # Left (dr=0, dc=-1)
    ]

    lbp = np.zeros(center.shape, dtype=np.uint16)
    for i, n in enumerate(neighbors):
        lbp += ((n >= center).astype(np.uint16) << i)

    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    total = float(hist.sum())
    if total <= 0:
        return None
    return hist / total


def extract_gradient_features(roi_gray: np.ndarray) -> dict:
    """
    Extracts structural edge & gradient statistics using 3x3 Sobel filters.
    Quantifies fine line pattern fidelity (e.g. security printing sharpness).

    Note: `gradient_mean` is the primary feature evaluated against calibrated thresholds,
    while `gradient_std`, `gradient_p95`, and `edge_density` provide diagnostic evidence.
    """
    if roi_gray is None or roi_gray.size == 0:
        return {
            "gradient_mean": 0.0,
            "gradient_std": 0.0,
            "gradient_p95": 0.0,
            "edge_density": 0.0
        }

    sobel_x = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(roi_gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    grad_mean = float(np.mean(magnitude))
    grad_std = float(np.std(magnitude))
    grad_p95 = float(np.percentile(magnitude, 95))
    edge_density = float(np.mean(magnitude > 30.0))

    return {
        "gradient_mean": round(grad_mean, 4),
        "gradient_std": round(grad_std, 4),
        "gradient_p95": round(grad_p95, 4),
        "edge_density": round(edge_density, 4)
    }



# ---------- DATASET SPLIT MANAGEMENT ----------
def load_or_create_dataset_split(reals_dir: str | Path = REALS_DIR,
                                 split_path: str | Path = SPLIT_PATH) -> dict:
    """
    Loads existing 70/15/15 dataset partition or creates a reproducible partition
    grouped by Base Passport ID (seed=42).
    """
    if os.path.exists(split_path):
        with open(split_path, 'r') as f:
            return json.load(f)

    real_files = get_image_files(reals_dir)
    base_ids = sorted(list(set(
        re.search(r'passport_(\d+)', f).group(1) for f in real_files if re.search(r'passport_(\d+)', f)
    )))

    import random
    rng = random.Random(42)
    shuffled = base_ids.copy()
    rng.shuffle(shuffled)

    # 70 calibration, 15 validation, 15 test
    n = len(shuffled)
    n_cal = int(0.70 * n)
    n_val = int(0.15 * n)

    cal_ids = sorted(shuffled[:n_cal])
    val_ids = sorted(shuffled[n_cal:n_cal + n_val])
    test_ids = sorted(shuffled[n_cal + n_val:])

    split_data = {
        "split_version": "1.0",
        "seed": 42,
        "sample_counts": {
            "calibration_genuine": len(cal_ids),
            "validation_genuine": len(val_ids),
            "final_test_genuine": len(test_ids),
            "total_genuine": len(base_ids)
        },
        "calibration_base_ids": cal_ids,
        "validation_base_ids": val_ids,
        "test_base_ids": test_ids
    }

    os.makedirs(os.path.dirname(split_path), exist_ok=True)
    with open(split_path, 'w') as f:
        json.dump(split_data, f, indent=2)

    return split_data


# ---------- GENUINE-ONLY CALIBRATION ----------
def calibrate_layer1(reals_dir: str | Path = REALS_DIR,
                     split_path: str | Path = SPLIT_PATH,
                     save_artifacts: bool = True) -> tuple[np.ndarray, dict]:
    """
    Calibrates Layer 1 reference baselines and thresholds using GENUINE IMAGES ONLY.

    Strict Separation Principle:
      - Accepts ONLY genuine images (`reals_dir`).
      - Fake images are completely prohibited from calibration.
      - References and thresholds are frozen and locked (v2.1.0).

    Uses a 70/15/15 Base-ID partition:
      - 70 Genuine Base IDs: Build reference model & derive tolerance limits.
      - 15 Genuine Base IDs: Independent validation check to verify generalization.
      - 15 Genuine Base IDs: Reserved for final held-out test (never touched here).
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    split_info = load_or_create_dataset_split(reals_dir, split_path)
    cal_base_ids = set(split_info["calibration_base_ids"])
    val_base_ids = set(split_info["validation_base_ids"])

    real_files = get_image_files(reals_dir)
    print(f"[Layer 1 Calibration] Discovered {len(real_files)} genuine passport images.")
    print(f"[Layer 1 Calibration] Partition: {len(cal_base_ids)} calibration IDs, {len(val_base_ids)} validation IDs.")

    # 1. Feature extraction for 70 calibration genuine images
    cal_intensity = []
    cal_lbp = []
    cal_grads = []

    for f in real_files:
        m = re.search(r'passport_(\d+)', f)
        if not m or m.group(1) not in cal_base_ids:
            continue

        p = os.path.join(reals_dir, f)
        img = cv2.imread(p)
        if img is None:
            continue
        roi, roi_info = extract_roi(img, BACKGROUND_PATCH)
        if roi is None:
            continue
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        q = assess_roi_quality(roi_gray)
        if q["status"] == "unusable":
            continue

        h_int = extract_intensity_histogram(roi_gray)
        h_lbp = compute_lbp_histogram(roi_gray)
        g_feat = extract_gradient_features(roi_gray)

        if h_int is not None and h_lbp is not None:
            cal_intensity.append(h_int)
            cal_lbp.append(h_lbp)
            cal_grads.append(g_feat)

    if len(cal_intensity) < 5:
        raise ValueError(f"Insufficient genuine calibration images (found {len(cal_intensity)}, min 5 required).")

    cal_intensity_arr = np.array(cal_intensity)
    cal_lbp_arr = np.array(cal_lbp)
    n_cal = len(cal_intensity)

    # 2. Build the Genuine Reference Models
    ref_intensity = np.mean(cal_intensity_arr, axis=0)
    ref_intensity = ref_intensity / ref_intensity.sum()

    ref_lbp = np.mean(cal_lbp_arr, axis=0)
    ref_lbp = ref_lbp / ref_lbp.sum()

    grad_means = [g["gradient_mean"] for g in cal_grads]
    ref_grad_mean = float(np.mean(grad_means))
    ref_grad_std = float(np.std(grad_means))

    # 3. Diagnostic Leave-One-Out (LOO) on Calibration Data
    # Purpose: Quantify natural intra-genuine variation without self-comparison bias.
    loo_int_dists = []
    loo_lbp_dists = []
    loo_grad_diffs = []

    for i in range(n_cal):
        subset_int = np.delete(cal_intensity_arr, i, axis=0)
        loo_ref_int = np.mean(subset_int, axis=0)
        loo_int_dists.append(histogram_wasserstein_distance(cal_intensity_arr[i], loo_ref_int))

        subset_lbp = np.delete(cal_lbp_arr, i, axis=0)
        loo_ref_lbp = np.mean(subset_lbp, axis=0)
        loo_lbp_dists.append(chi_square_distance(cal_lbp_arr[i], loo_ref_lbp))

        subset_grad = [grad_means[j] for j in range(n_cal) if j != i]
        loo_ref_grad = np.mean(subset_grad)
        loo_grad_diffs.append(abs(grad_means[i] - loo_ref_grad))

    loo_int_dists = np.array(loo_int_dists)
    loo_lbp_dists = np.array(loo_lbp_dists)
    loo_grad_diffs = np.array(loo_grad_diffs)

    # 4. Genuine-Only Threshold Derivation
    # Answers: "How much variation can occur naturally among genuine samples?"
    # Uses robust tolerance limits with explicit, documented guard-band floors.
    emp_int_limit = float(np.percentile(loo_int_dists, 99) * 1.5)
    thresh_intensity = float(max(emp_int_limit, PREDEFINED_GUARD_BAND_FLOORS["intensity"]))

    emp_lbp_limit = float(np.percentile(loo_lbp_dists, 99) * 1.5)
    thresh_lbp = float(max(emp_lbp_limit, PREDEFINED_GUARD_BAND_FLOORS["lbp"]))

    emp_grad_limit = float(np.percentile(loo_grad_diffs, 99) * 1.5)
    thresh_gradient = float(max(emp_grad_limit, PREDEFINED_GUARD_BAND_FLOORS["gradient"]))

    # 5. Validation Check on Held-Out Validation Set (15 Genuine Base IDs)
    val_int_dists = []
    val_lbp_dists = []
    val_grad_diffs = []
    val_passed_count = 0
    val_total = 0

    for f in real_files:
        m = re.search(r'passport_(\d+)', f)
        if not m or m.group(1) not in val_base_ids:
            continue

        p = os.path.join(reals_dir, f)
        img = cv2.imread(p)
        if img is None:
            continue
        roi, _ = extract_roi(img, BACKGROUND_PATCH)
        if roi is None:
            continue
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        h_int = extract_intensity_histogram(roi_gray)
        h_lbp = compute_lbp_histogram(roi_gray)
        g_feat = extract_gradient_features(roi_gray)

        if h_int is not None and h_lbp is not None:
            d_i = histogram_wasserstein_distance(h_int, ref_intensity)
            d_l = chi_square_distance(h_lbp, ref_lbp)
            d_g = abs(g_feat["gradient_mean"] - ref_grad_mean)

            val_int_dists.append(d_i)
            val_lbp_dists.append(d_l)
            val_grad_diffs.append(d_g)
            val_total += 1

            if (d_i <= thresh_intensity) and (d_l <= thresh_lbp) and (d_g <= thresh_gradient):
                val_passed_count += 1

    val_pass_rate = (val_passed_count / val_total) if val_total > 0 else 0.0
    print(f"[Layer 1 Calibration] Genuine Validation Check: {val_passed_count}/{val_total} passed ({val_pass_rate * 100:.1f}%).")

    # 6. Lock and Save Artifacts
    metadata = {
        "layer_name": LAYER_NAME,
        "algorithm_version": ALGORITHM_VERSION,
        "metric_version": METRIC_VERSION,
        "calibration_type": "genuine_only",
        "fake_data_used_for_calibration": False,
        "artifacts_locked": True,
        "patch_coords": BACKGROUND_PATCH,
        "sample_counts": {
            "genuine_calibration_count": n_cal,
            "genuine_validation_count": val_total,
            "final_test_count": len(split_info["test_base_ids"])
        },
        "metrics": {
            "intensity_distribution": {
                "metric": "weighted_wasserstein",
                "threshold": round(thresh_intensity, 4),
                "threshold_method": "genuine_only_robust_tolerance",
                "threshold_source": "calibration_genuine_only",
                "empirical_limit": round(emp_int_limit, 6),
                "guard_band_floor": PREDEFINED_GUARD_BAND_FLOORS["intensity"],
                "genuine_loo_mean": round(float(np.mean(loo_int_dists)), 6),
                "genuine_loo_max": round(float(np.max(loo_int_dists)), 6)
            },
            "spatial_texture": {
                "metric": "lbp_chi_square",
                "threshold": round(thresh_lbp, 4),
                "threshold_method": "genuine_only_robust_tolerance",
                "threshold_source": "calibration_genuine_only",
                "empirical_limit": round(emp_lbp_limit, 6),
                "guard_band_floor": PREDEFINED_GUARD_BAND_FLOORS["lbp"],
                "genuine_loo_mean": round(float(np.mean(loo_lbp_dists)), 6),
                "genuine_loo_max": round(float(np.max(loo_lbp_dists)), 6)
            },
            "structural_consistency": {
                "metric": "sobel_gradient_magnitude_diff",
                "primary_feature": "gradient_mean",
                "diagnostic_features": ["gradient_std", "gradient_p95", "edge_density"],
                "threshold": round(thresh_gradient, 4),
                "threshold_method": "genuine_only_robust_tolerance",
                "threshold_source": "calibration_genuine_only",
                "reference_mean": round(ref_grad_mean, 4),
                "reference_std": round(ref_grad_std, 4),
                "empirical_limit": round(emp_grad_limit, 6),
                "guard_band_floor": PREDEFINED_GUARD_BAND_FLOORS["gradient"]
            }
        },
        "validation_results": {
            "sample_count": val_total,
            "pass_rate": round(val_pass_rate, 4),
            "max_intensity_dist": round(float(np.max(val_int_dists)), 6) if val_int_dists else 0.0,
            "max_lbp_dist": round(float(np.max(val_lbp_dists)), 6) if val_lbp_dists else 0.0,
            "max_grad_diff": round(float(np.max(val_grad_diffs)), 6) if val_grad_diffs else 0.0
        },
        "dataset_characteristics": {
            "template_controlled": True,
            "zero_variance_handled": True
        }
    }

    if save_artifacts:
        np.save(REFERENCE_INTENSITY_HIST_PATH, ref_intensity)
        np.save(REFERENCE_LBP_HIST_PATH, ref_lbp)
        with open(METADATA_PATH, "w") as mf:
            json.dump(metadata, mf, indent=4)

        # Plot genuine calibration & validation distributions
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        
        # Panel 1: Intensity Wasserstein
        axes[0].hist(loo_int_dists, bins=10, alpha=0.7, color='green', label=f'Calibration LOO (n={n_cal})')
        if val_int_dists:
            axes[0].scatter(val_int_dists, [1]*len(val_int_dists), color='blue', marker='x', s=40, label=f'Validation (n={val_total})')
        axes[0].axvline(thresh_intensity, color='black', linestyle='--', label=f'Threshold={thresh_intensity:.2f}')
        axes[0].set_title("Intensity Distribution (Wasserstein-1)")
        axes[0].set_xlabel("Distance")
        axes[0].set_ylabel("Count")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, linestyle=':', alpha=0.6)

        # Panel 2: LBP Chi-Square
        axes[1].hist(loo_lbp_dists, bins=10, alpha=0.7, color='green', label=f'Calibration LOO (n={n_cal})')
        if val_lbp_dists:
            axes[1].scatter(val_lbp_dists, [1]*len(val_lbp_dists), color='blue', marker='x', s=40, label=f'Validation (n={val_total})')
        axes[1].axvline(thresh_lbp, color='black', linestyle='--', label=f'Threshold={thresh_lbp:.2f}')
        axes[1].set_title("Micro-Texture (LBP Chi-Square)")
        axes[1].set_xlabel("Distance")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, linestyle=':', alpha=0.6)

        # Panel 3: Gradient Diff
        axes[2].hist(loo_grad_diffs, bins=10, alpha=0.7, color='green', label=f'Calibration LOO (n={n_cal})')
        if val_grad_diffs:
            axes[2].scatter(val_grad_diffs, [1]*len(val_grad_diffs), color='blue', marker='x', s=40, label=f'Validation (n={val_total})')
        axes[2].axvline(thresh_gradient, color='black', linestyle='--', label=f'Threshold={thresh_gradient:.2f}')
        axes[2].set_title("Structural Edge Difference")
        axes[2].set_xlabel("Abs Difference")
        axes[2].legend(fontsize=8)
        axes[2].grid(True, linestyle=':', alpha=0.6)

        plt.suptitle("Layer 1: Genuine-Only Calibration & Validation Diagnostics", fontsize=12, y=1.02)
        plt.tight_layout()
        plt.savefig(PLOT_PATH, bbox_inches='tight', dpi=150)
        plt.close()

        print(f"[Layer 1 Calibration] Reference saved to {REFERENCE_INTENSITY_HIST_PATH}")
        print(f"[Layer 1 Calibration] LBP saved to {REFERENCE_LBP_HIST_PATH}")
        print(f"[Layer 1 Calibration] Metadata saved to {METADATA_PATH}")
        print(f"[Layer 1 Calibration] Diagnostic plots saved to {PLOT_PATH}")

    return ref_intensity, metadata


# ---------- ARTIFACT MANAGEMENT & LOCKING ----------
def load_calibration_artifacts(artifacts_dir: str | Path = RESULTS_DIR) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Loads locked calibration artifacts.

    Strict Invariant:
      NEVER silently calls calibrate_layer1() if artifacts are missing or stale.
      Raises FileNotFoundError or RuntimeError requiring explicit calibration.
    """
    artifacts_path = Path(artifacts_dir)
    int_path = artifacts_path / 'layer1_reference_intensity_histogram.npy'
    lbp_path = artifacts_path / 'layer1_reference_lbp_histogram.npy'
    meta_path = artifacts_path / 'layer1_metadata.json'

    if not (int_path.exists() and lbp_path.exists() and meta_path.exists()):
        raise FileNotFoundError(
            f"Locked Layer 1 calibration artifacts not found in '{artifacts_path}'. "
            f"Please run explicit calibration first: python3 signals/background_texture.py --calibrate"
        )

    with open(meta_path, "r") as mf:
        metadata = json.load(mf)

    if metadata.get("algorithm_version") != ALGORITHM_VERSION:
        raise RuntimeError(
            f"Stale Layer 1 artifact version ({metadata.get('algorithm_version')}). "
            f"Expected locked version v{ALGORITHM_VERSION}. "
            f"Please re-calibrate explicitly: python3 signals/background_texture.py --calibrate"
        )

    if not metadata.get("artifacts_locked", False):
        raise RuntimeError("Layer 1 calibration artifacts are not marked as locked.")

    ref_intensity = np.load(int_path)
    ref_lbp = np.load(lbp_path)

    return ref_intensity, ref_lbp, metadata


# ---------- LABEL-BLIND INFERENCE API ----------
def evaluate_background_texture(image_input,
                                reference_hist: np.ndarray | None = None,
                                threshold: float | None = None) -> dict:
    """
    Evaluates a single passport image against the locked genuine background baseline.

    Forensic Invariants:
      - Strictly label-blind: Never knows or uses ground-truth labels.
      - Never triggers re-calibration during inference.
      - Returns concise finding stopping at 'explanation'.

    Parameters:
        image_input: Either a file path (str, Path) or an OpenCV BGR image (np.ndarray).
        reference_hist: Optional pre-loaded reference intensity histogram (for testing).
        threshold: Optional custom threshold for intensity distance.

    Returns:
        dict: Forensic finding object with pass/fail, anomaly scores, and explanation.
    """
    # 1. Load Locked Calibration Artifacts
    try:
        ref_int, ref_lbp, meta = load_calibration_artifacts()
    except Exception as e:
        return {
            "layer_name": LAYER_NAME,
            "finding_type": "calibration_missing",
            "score": 1.0,
            "confidence": 0.0,
            "severity": "high",
            "passed": False,
            "raw_metric": {
                "status": "error",
                "verdict": "ERROR",
                "distance": None,
                "threshold": None,
                "error": f"Calibration artifacts unavailable: {str(e)}"
            },
            "explanation": f"Calibration artifacts unavailable: {str(e)}"
        }

    if reference_hist is not None:
        ref_int = reference_hist

    metrics_meta = meta.get("metrics", {})
    thresh_int = float(threshold if threshold is not None else metrics_meta.get("intensity_distribution", {}).get("threshold", PREDEFINED_GUARD_BAND_FLOORS["intensity"]))
    thresh_lbp = float(metrics_meta.get("spatial_texture", {}).get("threshold", PREDEFINED_GUARD_BAND_FLOORS["lbp"]))
    thresh_grad = float(metrics_meta.get("structural_consistency", {}).get("threshold", PREDEFINED_GUARD_BAND_FLOORS["gradient"]))
    ref_grad_mean = float(metrics_meta.get("structural_consistency", {}).get("reference_mean", 18.416))

    # 2. Ingest image input
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
        if img is None:
            return {
                "layer_name": LAYER_NAME,
                "finding_type": "invalid_input",
                "score": 1.0,
                "confidence": 0.0,
                "severity": "high",
                "passed": False,
                "raw_metric": {
                    "status": "error",
                    "verdict": "ERROR",
                    "distance": None,
                    "threshold": thresh_int,
                    "error": f"Could not read image from path: {image_input}"
                },
                "explanation": f"Could not read image from path: {image_input}"
            }
    elif isinstance(image_input, np.ndarray):
        img = image_input
    else:
        return {
            "layer_name": LAYER_NAME,
            "finding_type": "invalid_input",
            "score": 1.0,
            "confidence": 0.0,
            "severity": "high",
            "passed": False,
            "raw_metric": {
                "status": "error",
                "verdict": "ERROR",
                "distance": None,
                "threshold": thresh_int,
                "error": f"Invalid input type '{type(image_input)}'. Expected str, Path, or np.ndarray."
            },
            "explanation": f"Invalid input type '{type(image_input)}'. Expected str, Path, or np.ndarray."
        }

    # 3. Extract ROI
    roi, roi_info = extract_roi(img, BACKGROUND_PATCH)
    if roi is None:
        return {
            "layer_name": LAYER_NAME,
            "finding_type": "roi_extraction_failure",
            "score": 1.0,
            "confidence": 0.0,
            "severity": "high",
            "passed": False,
            "raw_metric": {
                "status": "error",
                "verdict": "ERROR",
                "distance": None,
                "threshold": thresh_int,
                "error": roi_info.get("error", "Failed to extract background ROI.")
            },
            "explanation": roi_info.get("error", "Failed to extract background ROI.")
        }

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi

    # 4. Assess Image & ROI Quality Gate
    quality = assess_roi_quality(roi_gray)
    if quality["status"] == "unusable":
        return {
            "layer_name": LAYER_NAME,
            "finding_type": "insufficient_image_quality",
            "score": 1.0,
            "confidence": 0.0,
            "severity": "high",
            "passed": False,
            "raw_metric": {
                "status": "error",
                "verdict": "UNUSABLE_QUALITY",
                "distance": None,
                "threshold": thresh_int,
                "contrast": quality.get("contrast"),
                "clipping_pct": quality.get("clipping_pct")
            },
            "explanation": f"ROI quality is unusable (contrast={quality['contrast']}, clipping={quality['clipping_pct']}%)."
        }

    # 5. Extract Multi-Feature Representations
    doc_intensity = extract_intensity_histogram(roi_gray)
    doc_lbp = compute_lbp_histogram(roi_gray)
    doc_grad = extract_gradient_features(roi_gray)

    if doc_intensity is None or doc_lbp is None:
        return {
            "layer_name": LAYER_NAME,
            "finding_type": "feature_extraction_failure",
            "score": 1.0,
            "confidence": 0.0,
            "severity": "high",
            "passed": False,
            "raw_metric": {
                "status": "error",
                "verdict": "ERROR",
                "distance": None,
                "threshold": thresh_int,
                "error": "Failed to compute feature histograms from ROI."
            },
            "explanation": "Failed to compute feature histograms from ROI."
        }

    # 6. Compute Discrepancy Distances
    d_intensity = histogram_wasserstein_distance(doc_intensity, ref_int)
    d_lbp = chi_square_distance(doc_lbp, ref_lbp)
    d_grad = abs(doc_grad["gradient_mean"] - ref_grad_mean)

    # Sub-scores normalized to [0.0, 1.0] (score >= 0.5 indicates anomaly)
    s_intensity = min(1.0, d_intensity / (thresh_int * 2.0)) if thresh_int > 0 else 1.0
    s_lbp = min(1.0, d_lbp / (thresh_lbp * 2.0)) if thresh_lbp > 0 else 1.0
    s_grad = min(1.0, d_grad / (thresh_grad * 2.0)) if thresh_grad > 0 else 1.0

    passed_int = d_intensity <= thresh_int
    passed_lbp = d_lbp <= thresh_lbp
    passed_grad = d_grad <= thresh_grad

    # Conservative composite decision: all features must pass
    passed = bool(passed_int and passed_lbp and passed_grad)
    composite_score = round(float(max(s_intensity, s_lbp, s_grad)), 4)

    # Forensic explanations
    failed_reasons = []
    if not passed_int:
        failed_reasons.append(f"intensity distribution mismatch (Wasserstein {d_intensity:.4f} > {thresh_int:.4f})")
    if not passed_lbp:
        failed_reasons.append(f"spatial micro-texture alteration (LBP Chi2 {d_lbp:.4f} > {thresh_lbp:.4f})")
    if not passed_grad:
        failed_reasons.append(f"structural sharpness deviation (GradDiff {d_grad:.2f} > {thresh_grad:.2f})")

    if passed:
        explanation = (
            f"Background security patterns are consistent with genuine baseline "
            f"(Intensity W1={d_intensity:.4f} <= {thresh_int:.4f}, LBP Chi2={d_lbp:.4f} <= {thresh_lbp:.4f})."
        )
        finding_type = "normal"
        verdict = "CONSISTENT_BACKGROUND"
    else:
        explanation = f"Background pattern anomalies detected: {'; '.join(failed_reasons)}."
        finding_type = "background_consistency_anomaly"
        verdict = "ANOMALOUS_BACKGROUND"

    # Severity and confidence estimation
    if composite_score < 0.35:
        severity = "low"
    elif composite_score < 0.70:
        severity = "medium"
    else:
        severity = "high"

    base_confidence = min(1.0, abs(d_intensity - thresh_int) / (thresh_int + 1e-9))
    if quality["status"] == "insufficient_quality":
        confidence = round(float(base_confidence * 0.5), 4)
    else:
        confidence = round(float(base_confidence), 4)

    return {
        "layer_name": LAYER_NAME,
        "finding_type": finding_type,
        "score": composite_score,
        "confidence": confidence,
        "severity": severity,
        "passed": passed,
        "raw_metric": {
            "status": "ok",
            "verdict": verdict,
            "distance": round(d_intensity, 6),
            "threshold": round(thresh_int, 6),
            "d_intensity": round(d_intensity, 6),
            "d_lbp": round(d_lbp, 6),
            "d_grad": round(d_grad, 4),
            "threshold_intensity": round(thresh_int, 6),
            "threshold_lbp": round(thresh_lbp, 6),
            "threshold_grad": round(thresh_grad, 4)
        },
        "explanation": explanation
    }


# ---------- CLI ENTRY POINT ----------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Layer 1: Background Intensity & Texture Consistency")
    parser.add_argument('--calibrate', action='store_true', help="Run genuine-only calibration and lock artifacts")
    args = parser.parse_args()

    print("=" * 50)
    print("CALIBRATING AND TESTING LAYER 1 (BACKGROUND TEXTURE)")
    print("=" * 50)

    if args.calibrate or not METADATA_PATH.exists():
        ref, meta = calibrate_layer1()
        print("\nCalibrated Metadata:")
        print(json.dumps(meta, indent=2))
    else:
        print(f"\nLocked calibration artifacts found at {METADATA_PATH}.")
        with open(METADATA_PATH, "r") as mf:
            meta = json.load(mf)
        print("\nCalibrated Metadata:")
        print(json.dumps(meta, indent=2))

    real_files = get_image_files(REALS_DIR)
    fake_files = get_image_files(FAKES_DIR)

    if real_files:
        sample_real_path = str(REALS_DIR / real_files[0])
        finding_real = evaluate_background_texture(sample_real_path)
        print(f"\nReal Sample Finding ({Path(sample_real_path).stem}):")
        print(json.dumps(finding_real, indent=4))

    if fake_files:
        sample_fake_path = str(FAKES_DIR / fake_files[0])
        finding_fake = evaluate_background_texture(sample_fake_path)
        print(f"\nFake Sample Finding ({Path(sample_fake_path).stem}):")
        print(json.dumps(finding_fake, indent=4))

    print("=" * 50)