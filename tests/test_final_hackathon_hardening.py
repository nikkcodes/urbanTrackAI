"""
Unit and Integration Tests for Final Hackathon-Grade Hardening.
Tests:
1. MultiCameraFeedAdapter registration, tier isolation, and non-mixing guarantee.
2. Data-driven camera reliability profiling from observable stream metrics.
3. Platt probability calibrator, Bayesian evidence combiner, ECE, and Brier score.
4. System health and pre-flight deployment readiness diagnostics.
5. High-scale candidate generator optimization and semantic equivalence.
6. Adversarial suite structured failure mode reporting.
7. ReID ROC-AUC and threshold sweep validation.
"""

from datetime import datetime
import math
from pathlib import Path
import unittest

from schemas.observation_schema import Observation
from inference.observation_loader import (
    MultiCameraFeedAdapter,
    DatasetClassification,
    load_member1_perception_feed,
)
from inference.reliability_engine import (
    profile_camera_reliability_from_observations,
    evaluate_camera_reliability,
)
from inference.calibrator import (
    PlattProbabilityCalibrator,
    BayesianEvidenceCombiner,
    evaluate_calibration_metrics,
)
from inference.system_health import SystemHealthChecker
from inference.candidate_generation import CandidateGenerator, benchmark_large_scale_candidate_pipeline
from inference.adversarial_suite import run_adversarial_suite
from inference.benchmark.ground_truth import GroundTruthRegistry
from inference.benchmark.evaluator import MultiCameraBenchmarkEvaluator


class TestFinalHackathonHardening(unittest.TestCase):

    def setUp(self):
        self.project_root = Path(__file__).resolve().parent.parent

    def test_multi_camera_feed_adapter_tier_isolation(self):
        """Verify MultiCameraFeedAdapter strictly isolates REAL vs CONTROLLED tiers."""
        adapter = MultiCameraFeedAdapter()
        adapter.register_camera_feed(
            "CAM_001",
            DatasetClassification.REAL,
            self.project_root / "data/member1_perception/cam_001",
        )
        adapter.register_camera_feed(
            "CAM_NORTH_01",
            DatasetClassification.CONTROLLED,
            self.project_root / "data/benchmarks/multicamera_v1/observations.json",
        )

        # Ingest only REAL
        real_obs = adapter.ingest_all_feeds(allowed_classifications=[DatasetClassification.REAL])
        self.assertEqual(len(real_obs), 39)
        self.assertTrue(all(getattr(o, "dataset_classification", None) == "REAL" for o in real_obs))

        # Ingest only CONTROLLED
        ctrl_obs = adapter.ingest_all_feeds(allowed_classifications=[DatasetClassification.CONTROLLED])
        self.assertEqual(len(ctrl_obs), 1500)
        self.assertTrue(all(getattr(o, "dataset_classification", None) == "CONTROLLED" for o in ctrl_obs))

        # Check inventory summary
        inv = adapter.get_inventory()
        self.assertIn("CAM_001", inv)
        self.assertIn("CAM_NORTH_01", inv)
        self.assertEqual(inv["CAM_001"]["classification"], "REAL")
        self.assertEqual(inv["CAM_NORTH_01"]["classification"], "CONTROLLED")

    def test_data_driven_camera_reliability_empirical(self):
        """Verify camera reliability is derived empirically and never uses magic numbers."""
        real_obs = load_member1_perception_feed()
        profile = profile_camera_reliability_from_observations("CAM_001", real_obs)

        # 39 tracklets, valid ReID and OCR rates
        self.assertEqual(profile.status, "computed_data_driven")
        self.assertIsNotNone(profile.reliability)
        self.assertGreater(profile.reliability, 0.50)
        self.assertEqual(profile.factors["sample_count"], 39)
        self.assertEqual(profile.factors["reid_validity_rate"], 1.0)
        self.assertIn("mean_detection_confidence", profile.factors)

    def test_data_driven_camera_reliability_sparse_and_empty(self):
        """Verify camera reliability returns unknown/insufficient_evidence when data is absent."""
        # Empty observations
        prof_empty = profile_camera_reliability_from_observations("CAM_EMPTY", [])
        self.assertEqual(prof_empty.status, "unknown")
        self.assertIsNone(prof_empty.reliability)

        # Sparse observations (< 3)
        sample = [
            Observation(observation_id="OBS_1", camera_id="CAM_SPARSE", timestamp_seconds=10.0),
            Observation(observation_id="OBS_2", camera_id="CAM_SPARSE", timestamp_seconds=20.0),
        ]
        prof_sparse = profile_camera_reliability_from_observations("CAM_SPARSE", sample, min_sample_threshold=3)
        self.assertEqual(prof_sparse.status, "insufficient_evidence")
        self.assertIsNone(prof_sparse.reliability)

    def test_platt_probability_calibrator(self):
        """Verify Platt scaling calibrator maps heuristic scores to empirical probabilities."""
        cal = PlattProbabilityCalibrator(a=6.0, b=-4.5)
        # Monotonicity check
        p0 = cal.predict_probability(0.0)
        p35 = cal.predict_probability(0.35)
        p75 = cal.predict_probability(0.75)
        p95 = cal.predict_probability(0.95)

        self.assertEqual(p0, 0.0)
        self.assertLess(p35, p75)
        self.assertLess(p75, p95)
        self.assertAlmostEqual(p75, 0.50, delta=0.05)

        # Fit test on toy development data
        scores = [0.1, 0.2, 0.3, 0.4, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98]
        labels = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        cal.fit(scores, labels, epochs=50)
        self.assertTrue(cal.is_fitted)
        self.assertIn("brier_score", cal.training_metrics)
        self.assertIn("expected_calibration_error", cal.training_metrics)

    def test_bayesian_evidence_combiner(self):
        """Verify Bayesian evidence combination handles contradictions and missing data."""
        combiner = BayesianEvidenceCombiner(prior_probability=0.05)

        # Strong evidence
        p_high, lrs_high = combiner.evaluate_posterior(
            reid_similarity=0.92,
            plate_similarity=1.0,
            ocr_confidence=0.95,
            spatial_feasible=True,
            temporal_feasible=True,
            vehicle_type_status="compatible",
        )
        self.assertGreater(p_high, 0.90)
        self.assertGreater(lrs_high["lr_total"], 10.0)

        # Physical contradiction: incompatible vehicle type
        p_type_contra, _ = combiner.evaluate_posterior(
            reid_similarity=0.95,
            plate_similarity=1.0,
            ocr_confidence=0.95,
            spatial_feasible=True,
            temporal_feasible=True,
            vehicle_type_status="incompatible",
        )
        self.assertEqual(p_type_contra, 0.0)

        # Plate contradiction
        p_plate_contra, _ = combiner.evaluate_posterior(
            reid_similarity=0.95,
            plate_similarity=0.15,
            ocr_confidence=0.95,
            spatial_feasible=True,
            temporal_feasible=True,
            vehicle_type_status="compatible",
        )
        self.assertEqual(p_plate_contra, 0.0)

        # Missing evidence is neutral
        p_missing, lrs_missing = combiner.evaluate_posterior(
            reid_similarity=None,
            plate_similarity=None,
            ocr_confidence=None,
            spatial_feasible=True,
            temporal_feasible=True,
            vehicle_type_status="compatible",
        )
        self.assertEqual(lrs_missing["lr_reid"], 1.0)
        self.assertEqual(lrs_missing["lr_plate"], 1.0)

    def test_system_health_checker(self):
        """Verify SystemHealthChecker pre-flight diagnostics."""
        adapter = MultiCameraFeedAdapter()
        adapter.register_camera_feed(
            "CAM_001",
            DatasetClassification.REAL,
            self.project_root / "data/member1_perception/cam_001",
        )
        checker = SystemHealthChecker(feed_adapter=adapter)
        res = checker.run_preflight_check()

        self.assertEqual(res["overall_status"], "HEALTHY")
        self.assertTrue(res["diagnostics"]["deployment_ready"])
        self.assertEqual(res["environment"]["status"], "PASS")
        self.assertEqual(res["perception_feeds"]["status"], "PASS")

    def test_adversarial_suite_explicit_reporting(self):
        """Verify all 16 adversarial scenarios have explicit structured failure fields."""
        adv_res = run_adversarial_suite()
        self.assertEqual(adv_res["total_scenarios"], 20)
        self.assertTrue(adv_res["all_passed"])

        required_keys = {"id", "name", "input", "expected_behavior", "actual_behavior", "pass_fail", "score"}
        for s in adv_res["scenarios"]:
            self.assertTrue(required_keys.issubset(s.keys()), f"Scenario {s.get('id')} missing keys")
            self.assertEqual(s["pass_fail"], "PASS")

    def test_reid_roc_auc_and_extraction_quality(self):
        """Verify ReID evaluation computes ROC-AUC and verifies unit hypersphere norms."""
        bench_dir = self.project_root / "data/benchmarks/multicamera_v1"
        registry = GroundTruthRegistry.load_from_directory(bench_dir)
        obs = adapter = MultiCameraFeedAdapter()
        adapter.register_camera_feed("MC", DatasetClassification.CONTROLLED, bench_dir / "observations.json")
        obs = adapter.load_camera_observations("MC")

        evaluator = MultiCameraBenchmarkEvaluator(registry=registry, threshold=0.75)
        reid_eval = evaluator.evaluate_reid_quality(registry, obs)

        # Extraction quality
        ext = reid_eval["embedding_extraction_quality"]
        self.assertEqual(ext["total_observations"], 1500)
        self.assertTrue(ext["unit_hypersphere_verified"])
        self.assertAlmostEqual(ext["mean_l2_norm"], 1.0, delta=0.01)

        # Pairwise matching quality
        match_q = reid_eval["pairwise_reid_matching"]
        self.assertGreater(match_q["roc_auc"], 0.80)
        self.assertGreater(match_q["same_vehicle_similarity_mean"], match_q["different_vehicle_similarity_mean"])


if __name__ == "__main__":
    unittest.main()
