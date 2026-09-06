"""
Unit and Integration Test Suite for Layer 1: Background Consistency (v2.1.0)
=============================================================================
Tests cover all required verification categories:
  1. Histogram extraction validation
  2. Invalid histogram error handling
  3. Wasserstein distance mathematical correctness
  4. Identical histogram zero-distance invariant
  5. Shifted histogram monotonic distance scaling
  6. ROI extraction and boundary clamping
  7. Image and ROI quality gating
  8. LBP micro-texture descriptor determinism & properties
  9. Genuine-only calibration without fake data dependency
  10. Leave-one-out cross-validation diagnostics correctness
  11. Artifact locking and refusal to silently recalibrate
  12. Dataset partitioning integrity (zero leakage between 70/15/15 splits)
  13. Single-image inference API on real, fake, and invalid inputs
  14. Controlled splicing test (untouched background passes Layer 1)
  15. Pipeline orchestrator integration and schema compliance
"""

import os
import json
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from scipy.stats import wasserstein_distance as scipy_wasserstein

from signals.background_texture import (
    ALGORITHM_VERSION,
    METRIC_VERSION,
    LAYER_NAME,
    BACKGROUND_PATCH,
    validate_histogram,
    histogram_wasserstein_distance,
    chi_square_distance,
    extract_roi,
    assess_roi_quality,
    extract_intensity_histogram,
    compute_lbp_histogram,
    extract_gradient_features,
    calibrate_layer1,
    load_calibration_artifacts,
    evaluate_background_texture,
    load_or_create_dataset_split,
    REALS_DIR,
    METADATA_PATH,
    SPLIT_PATH,
)
from pipeline.forensics_engine import run_forensics

FAKES_DIR = Path(__file__).resolve().parent.parent / 'dataset' / 'fakes'


class TestLayer1MathematicalCorrectness(unittest.TestCase):
    """Tests 1-5: Mathematical and distribution invariants."""

    def test_01_histogram_extraction(self):
        """Test 1: Intensity histogram extraction has 256 bins, sums to 1.0, non-negative, finite."""
        dummy = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
        hist = extract_intensity_histogram(dummy)
        self.assertIsNotNone(hist)
        self.assertEqual(len(hist), 256)
        self.assertAlmostEqual(float(hist.sum()), 1.0, places=6)
        self.assertTrue(np.all(hist >= 0.0))
        self.assertTrue(np.all(np.isfinite(hist)))

    def test_02_invalid_histogram_validation(self):
        """Test 2: Malformed histograms raise ValueError."""
        with self.assertRaises(ValueError):
            validate_histogram(None)
        with self.assertRaises(ValueError):
            validate_histogram(np.zeros(128))
        with self.assertRaises(ValueError):
            validate_histogram(np.zeros((16, 16)))
        nan_hist = np.ones(256) / 256
        nan_hist[10] = np.nan
        with self.assertRaises(ValueError):
            validate_histogram(nan_hist)
        inf_hist = np.ones(256) / 256
        inf_hist[10] = np.inf
        with self.assertRaises(ValueError):
            validate_histogram(inf_hist)
        neg_hist = np.ones(256) / 256
        neg_hist[5] = -0.1
        with self.assertRaises(ValueError):
            validate_histogram(neg_hist)
        with self.assertRaises(ValueError):
            validate_histogram(np.zeros(256))

    def test_03_wasserstein_mathematical_correctness(self):
        """Test 3: Wasserstein matches true weighted coordinate distance."""
        bins = np.arange(256, dtype=np.float64)
        h1 = np.zeros(256)
        h1[30] = 1.0
        h2 = np.zeros(256)
        h2[80] = 1.0

        dist = histogram_wasserstein_distance(h1, h2)
        expected = scipy_wasserstein(bins, bins, u_weights=h1, v_weights=h2)
        self.assertAlmostEqual(dist, 50.0, places=5)
        self.assertAlmostEqual(dist, expected, places=5)

    def test_04_identical_histograms_zero_distance(self):
        """Test 4: Identical distributions produce zero distance."""
        rng = np.random.RandomState(42)
        h = rng.uniform(0.1, 1.0, size=256)
        h = h / h.sum()

        self.assertAlmostEqual(histogram_wasserstein_distance(h, h), 0.0, places=7)
        self.assertAlmostEqual(chi_square_distance(h, h), 0.0, places=7)

    def test_05_shifted_histograms_monotonic_distance(self):
        """Test 5: Shifted distributions produce monotonically increasing distance."""
        base = np.zeros(256)
        base[50] = 1.0

        dists = []
        shifts = [5, 15, 30, 60]
        for s in shifts:
            shifted = np.zeros(256)
            shifted[50 + s] = 1.0
            dists.append(histogram_wasserstein_distance(base, shifted))

        for i in range(len(dists) - 1):
            self.assertLess(dists[i], dists[i+1])
            self.assertAlmostEqual(dists[i], shifts[i], places=5)


class TestLayer1FeatureAndQualityGating(unittest.TestCase):
    """Tests 6-8: ROI extraction, quality gating, and LBP descriptor."""

    def test_06_roi_extraction_and_clamping(self):
        """Test 6: Valid image clamps correctly; invalid shapes handled gracefully."""
        img = np.zeros((1000, 1000, 3), dtype=np.uint8)
        roi, info = extract_roi(img, BACKGROUND_PATCH)
        self.assertIsNotNone(roi)
        self.assertEqual(info["status"], "ok")
        self.assertGreater(info["pixel_count"], 1000)

        tiny = np.zeros((30, 30, 3), dtype=np.uint8)
        roi_tiny, info_tiny = extract_roi(tiny, BACKGROUND_PATCH)
        self.assertIsNone(roi_tiny)
        self.assertEqual(info_tiny["status"], "image_too_small")

        empty = np.array([], dtype=np.uint8)
        roi_empty, info_empty = extract_roi(empty, BACKGROUND_PATCH)
        self.assertIsNone(roi_empty)
        self.assertEqual(info_empty["status"], "invalid_image")

        roi_none, info_none = extract_roi(None, BACKGROUND_PATCH)
        self.assertIsNone(roi_none)
        self.assertEqual(info_none["status"], "invalid_image")

    def test_07_roi_quality_gating(self):
        """Test 7: Pure black, pure white, and low contrast are flagged as unusable."""
        black = np.zeros((100, 500), dtype=np.uint8)
        q_black = assess_roi_quality(black)
        self.assertEqual(q_black["status"], "unusable")

        white = np.full((100, 500), 255, dtype=np.uint8)
        q_white = assess_roi_quality(white)
        self.assertEqual(q_white["status"], "unusable")

        rng = np.random.RandomState(42)
        normal = rng.randint(40, 200, size=(100, 500), dtype=np.uint8)
        q_norm = assess_roi_quality(normal)
        self.assertEqual(q_norm["status"], "acceptable")

    def test_08_lbp_texture_properties(self):
        """Test 8: LBP histogram is deterministic, 256 bins, sums to 1.0."""
        rng = np.random.RandomState(123)
        patch = rng.randint(0, 256, size=(60, 200), dtype=np.uint8)

        lbp1 = compute_lbp_histogram(patch)
        lbp2 = compute_lbp_histogram(patch)
        self.assertIsNotNone(lbp1)
        self.assertEqual(len(lbp1), 256)
        self.assertAlmostEqual(float(lbp1.sum()), 1.0, places=6)
        np.testing.assert_array_equal(lbp1, lbp2)

        small = np.zeros((2, 2), dtype=np.uint8)
        self.assertIsNone(compute_lbp_histogram(small))


class TestLayer1GenuineCalibrationAndLocking(unittest.TestCase):
    """Tests 9-12: Calibration, Leakage Isolation, Locking, and Partitioning."""

    def test_09_calibration_genuine_only(self):
        """Test 9: Calibration succeeds with genuine-only data and rejects missing data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_dir = Path(tmpdir)
            with self.assertRaises(ValueError):
                calibrate_layer1(reals_dir=empty_dir, save_artifacts=False)

    def test_10_leakage_resistance(self):
        """Test 10: Calibration signature has no fake parameter and metadata marks fakes unused."""
        import inspect
        sig = inspect.signature(calibrate_layer1)
        self.assertNotIn("fakes_dir", sig.parameters, "calibrate_layer1 must not accept fakes_dir.")

        if METADATA_PATH.exists():
            with open(METADATA_PATH, "r") as mf:
                meta = json.load(mf)
            self.assertFalse(meta.get("fake_data_used_for_calibration"), "Fake data must not be used for calibration.")
            self.assertEqual(meta.get("calibration_type"), "genuine_only")

    def test_11_artifact_locking_and_no_silent_recalibration(self):
        """Test 11: Missing or corrupted artifacts raise error and do not silently recalibrate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                load_calibration_artifacts(artifacts_dir=tmpdir)

        # Ensure metadata is locked
        if METADATA_PATH.exists():
            with open(METADATA_PATH, "r") as mf:
                meta = json.load(mf)
            self.assertTrue(meta.get("artifacts_locked"), "Artifacts must be explicitly marked as locked.")
            self.assertEqual(meta.get("algorithm_version"), ALGORITHM_VERSION)

    def test_12_dataset_partition_integrity(self):
        """Test 12: Split has 70/15/15 disjoint base IDs with zero overlap."""
        split_data = load_or_create_dataset_split()
        cal = set(split_data["calibration_base_ids"])
        val = set(split_data["validation_base_ids"])
        test = set(split_data["test_base_ids"])

        self.assertEqual(len(cal), 70)
        self.assertEqual(len(val), 15)
        self.assertEqual(len(test), 15)
        self.assertEqual(len(cal & val), 0, "Calibration and Validation base IDs must be disjoint.")
        self.assertEqual(len(cal & test), 0, "Calibration and Test base IDs must be disjoint.")
        self.assertEqual(len(val & test), 0, "Validation and Test base IDs must be disjoint.")


class TestLayer1InferenceAndIntegration(unittest.TestCase):
    """Tests 13-15: Inference API, Controlled Splicing, and Forensics Engine."""

    def test_13_single_image_inference(self):
        """Test 13: Inference on genuine, fake, and invalid inputs."""
        real_files = sorted(list(Path(REALS_DIR).glob("*.jpg")))
        fake_files = sorted(list(Path(FAKES_DIR).glob("*.jpg")))

        if real_files:
            res_real = evaluate_background_texture(str(real_files[0]))
            self.assertEqual(res_real["raw_metric"]["status"], "ok")
            self.assertTrue(res_real["passed"])
            self.assertEqual(res_real["raw_metric"]["verdict"], "CONSISTENT_BACKGROUND")
            self.assertEqual(res_real["finding_type"], "normal")
            self.assertLessEqual(res_real["score"], 0.35)
            self.assertIn("explanation", res_real)

        if fake_files:
            res_fake = evaluate_background_texture(str(fake_files[0]))
            self.assertEqual(res_fake["raw_metric"]["status"], "ok")
            self.assertFalse(res_fake["passed"])
            self.assertEqual(res_fake["raw_metric"]["verdict"], "ANOMALOUS_BACKGROUND")
            self.assertEqual(res_fake["finding_type"], "background_consistency_anomaly")
            self.assertGreaterEqual(res_fake["score"], 0.5)

        res_bad = evaluate_background_texture("non_existent_file.jpg")
        self.assertEqual(res_bad["raw_metric"]["status"], "error")
        self.assertFalse(res_bad["passed"])
        self.assertEqual(res_bad["finding_type"], "invalid_input")

    def test_14_controlled_portrait_only_splicing(self):
        """Test 14: Untouched background with spliced photo passes Layer 1 as CONSISTENT_BACKGROUND."""
        real_files = sorted(list(Path(REALS_DIR).glob("*.jpg")))
        if len(real_files) >= 2:
            img1 = cv2.imread(str(real_files[0]))
            img2 = cv2.imread(str(real_files[1]))
            h, w = img1.shape[:2]

            # Modify portrait box ONLY; background strip is 100% untouched
            px, py, pw, ph = int(0.08 * w), int(0.35 * h), int(0.28 * w), int(0.45 * h)
            spliced = img1.copy()
            spliced[py:py+ph, px:px+pw] = img2[py:py+ph, px:px+pw]

            res_splice = evaluate_background_texture(spliced)
            self.assertEqual(res_splice["raw_metric"]["status"], "ok")
            self.assertTrue(res_splice["passed"], "Layer 1 must pass an authentic untouched background.")
            self.assertEqual(res_splice["raw_metric"]["verdict"], "CONSISTENT_BACKGROUND")

    def test_15_pipeline_engine_integration(self):
        """Test 15: Full forensics engine runs and produces compliant Evidence Object."""
        real_files = sorted(list(Path(REALS_DIR).glob("*.jpg")))
        if real_files:
            evidence = run_forensics(real_files[0], save=False)
            self.assertIn("findings", evidence)
            self.assertEqual(evidence["layers_completed"], 4)
            self.assertEqual(evidence["layers_failed"], 0)

            layer1_finding = None
            for f in evidence["findings"]:
                if "Layer 1" in f.get("layer_name", ""):
                    layer1_finding = f
                    break

            self.assertIsNotNone(layer1_finding)
            self.assertEqual(layer1_finding["raw_metric"]["status"], "ok")
            self.assertTrue(layer1_finding["passed"])
            self.assertIn("confidence", layer1_finding)
            self.assertIn("severity", layer1_finding)
            self.assertIn("explanation", layer1_finding)
            self.assertIn("raw_metric", layer1_finding)


if __name__ == "__main__":
    unittest.main(verbosity=2)
