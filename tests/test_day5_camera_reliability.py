"""
Unit and Integration Test Suite for Day 5: Camera Reliability + Uncertainty Propagation.

Validates:
1. CameraReliability, ObservationReliability, IdentityMatchReliability, TrajectoryReliability schemas.
2. Reliability evaluation functions (camera, observation, identity, trajectory).
3. The 10 concrete scenarios from data/synthetic/day5_reliability_scenarios.json.
4. Monotonicity: Lower sensor quality strictly non-increasing downstream reliability.
5. Strict separation of concepts: vehicle_weight != reliability != probability.
6. Preservation of Day 4 sparse inference and unobserved intermediate nodes.
7. Member 3 adapter compliance and metadata propagation.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from typing import Any, Dict

from inference.identity_fusion import match_observations
from inference.member3_adapter import (
    adapt_sparse_gap_to_normalized,
    adapt_trajectory_segment_to_normalized,
    adapt_trajectories_to_batch_payload,
    adapt_vehicle_trajectory_to_normalized,
)
from inference.reliability_engine import (
    evaluate_camera_reliability,
    evaluate_identity_uncertainty,
    evaluate_observation_reliability,
    propagate_trajectory_uncertainty,
)
from inference.road_graph import RoadGraph
from inference.sparse_engine import infer_sparse_gap
from inference.trajectory_engine import (
    reconstruct_identity_trajectory,
    reconstruct_trajectory_segment,
)
from schemas.gap_schema import SparseObservationGap
from schemas.normalized_trajectory_schema import NormalizedTrajectory
from schemas.observation_schema import Observation
from schemas.reliability_schema import (
    CameraReliability,
    IdentityMatchReliability,
    ObservationReliability,
    TrajectoryReliability,
)
from schemas.trajectory_schema import CandidateRoute, TrajectorySegment, VehicleTrajectory


class TestDay5CameraReliability(unittest.TestCase):
    """Day 5 test suite for camera reliability, observation quality, and uncertainty propagation."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load synthetic road graph and test scenarios."""
        cls.repo_root = Path(__file__).parent.parent
        cls.network_path = cls.repo_root / "data" / "synthetic" / "city_network.json"
        cls.scenarios_path = cls.repo_root / "data" / "synthetic" / "day5_reliability_scenarios.json"

        cls.graph = RoadGraph.from_json_file(str(cls.network_path))

        # Map test cameras to Member 3 junctions
        cls.graph.camera_associations["CAM_J01"] = "J01"
        cls.graph.camera_associations["CAM_J02"] = "J02"
        cls.graph.camera_associations["CAM_J03"] = "J03"
        cls.graph.camera_associations["CAM_J04"] = "J04"
        cls.graph.camera_associations["CAM_J05"] = "J05"
        cls.graph.camera_associations["CAM_J06"] = "J06"
        cls.graph.camera_associations["CAM_J07"] = "J07"
        cls.graph.camera_associations["CAM_J08"] = "J08"
        cls.graph.camera_associations["CAM_J10"] = "J10"
        cls.graph.camera_associations["CAM_J11"] = "J11"
        cls.graph.camera_associations["CAM_J12"] = "J12"

        with open(cls.scenarios_path, "r", encoding="utf-8") as f:
            cls.fixture = json.load(f)

    # -------------------------------------------------------------------------
    # 1. Schema & Model Initialization Tests
    # -------------------------------------------------------------------------
    def test_01_camera_reliability_schema_serialization(self) -> None:
        """Verify CameraReliability dataclass and round-trip serialization."""
        cr = CameraReliability(
            camera_id="CAM_TEST_01",
            overall_reliability=0.92,
            uptime_score=0.98,
            calibration_score=0.95,
            environmental_score=0.85,
            confidence_interval=(0.88, 0.96),
            metadata={"maintenance_status": "ok"},
        )
        d = cr.to_dict()
        self.assertEqual(d["camera_id"], "CAM_TEST_01")
        self.assertEqual(d["overall_reliability"], 0.92)
        self.assertEqual(d["uptime_score"], 0.98)

        restored = CameraReliability.from_dict(d)
        self.assertEqual(restored.camera_id, cr.camera_id)
        self.assertEqual(restored.overall_reliability, cr.overall_reliability)

    def test_02_observation_reliability_schema_serialization(self) -> None:
        """Verify ObservationReliability dataclass and round-trip serialization."""
        obs_rel = ObservationReliability(
            observation_id="OBS_TEST_01",
            camera_id="CAM_TEST_01",
            camera_reliability=0.90,
            detection_confidence=0.95,
            ocr_reliability=0.92,
            reid_reliability=0.88,
            overall_reliability=0.91,
            uncertainty=0.09,
            degradation_factors=["minor_glare"],
        )
        d = obs_rel.to_dict()
        self.assertEqual(d["observation_id"], "OBS_TEST_01")
        self.assertEqual(d["overall_reliability"], 0.91)
        self.assertEqual(d["uncertainty"], 0.09)

        restored = ObservationReliability.from_dict(d)
        self.assertEqual(restored.overall_reliability, 0.91)
        self.assertEqual(restored.uncertainty, 0.09)

    # -------------------------------------------------------------------------
    # 2. Reliability Engine Core Functions
    # -------------------------------------------------------------------------
    def test_03_evaluate_camera_reliability_computation(self) -> None:
        """Verify evaluate_camera_reliability handles perfect, degraded, and missing inputs."""
        # High quality camera
        high_cam = evaluate_camera_reliability(
            camera_id="CAM_HI",
            uptime=0.99,
            calibration_score=0.95,
            weather_degradation=0.0,
        )
        self.assertGreaterEqual(high_cam.overall_reliability, 0.90)

        # Degraded camera (heavy rain / dirty lens)
        deg_cam = evaluate_camera_reliability(
            camera_id="CAM_LOW",
            uptime=0.60,
            calibration_score=0.50,
            weather_degradation=0.80,
        )
        self.assertLess(deg_cam.overall_reliability, 0.50)
        self.assertIn("severe_environmental_degradation", deg_cam.metadata.get("flags", []))

    def test_04_evaluate_observation_reliability_missing_data_rule(self) -> None:
        """Verify missing camera reliability or embeddings defaults to neutral/penalized (never 1.0)."""
        obs_dict = {
            "observation_id": "OBS_MISSING_DATA",
            "camera_id": "CAM_UNKNOWN",
            "confidence": 0.90,
            # No camera_reliability, no plate text, no reid embedding
        }
        rel = evaluate_observation_reliability(obs_dict)
        # MUST NEVER be 1.0
        self.assertLessEqual(rel.overall_reliability, 0.70)
        self.assertGreater(rel.uncertainty, 0.30)
        self.assertIn("missing_camera_reliability", rel.degradation_factors)

    # -------------------------------------------------------------------------
    # 3. Ten Specific Scenarios (Cases 1 to 10)
    # -------------------------------------------------------------------------
    def test_05_case_01_baseline_high_trust(self) -> None:
        """CASE 1: High-reliability camera + high-confidence observation (baseline high trust)."""
        obs = {
            "observation_id": "OBS_C01",
            "camera_id": "CAM_HIGH",
            "camera_reliability": 0.95,
            "confidence": 0.95,
            "plate_text": "DL01AB1234",
            "plate_confidence": 0.96,
            "reid_embedding": [0.5] * 128,
        }
        rel = evaluate_observation_reliability(obs)
        self.assertGreaterEqual(rel.overall_reliability, 0.85)
        self.assertLessEqual(rel.uncertainty, 0.15)

    def test_06_case_02_degraded_camera(self) -> None:
        """CASE 2: Degraded camera (blur/occlusion/weather) with lowered camera reliability."""
        obs = {
            "observation_id": "OBS_C02",
            "camera_id": "CAM_DEG",
            "camera_reliability": 0.35,
            "confidence": 0.85,
            "plate_text": "HR26CD5678",
            "plate_confidence": 0.85,
            "reid_embedding": [0.4] * 128,
        }
        rel = evaluate_observation_reliability(obs)
        self.assertLessEqual(rel.overall_reliability, 0.60)
        self.assertGreaterEqual(rel.uncertainty, 0.40)
        self.assertIn("degraded_camera_sensor", rel.degradation_factors)

    def test_07_case_03_ocr_plate_ambiguity(self) -> None:
        """CASE 3: OCR plate ambiguity (low confidence + ambiguous reading)."""
        obs = {
            "observation_id": "OBS_C03",
            "camera_id": "CAM_GOOD",
            "camera_reliability": 0.90,
            "confidence": 0.90,
            "plate_text": "DL01AB123?",
            "plate_confidence": 0.42,
            "alternative_plates": ["DL01AB1238", "DL01AB1233"],
        }
        rel = evaluate_observation_reliability(obs)
        self.assertLessEqual(rel.overall_reliability, 0.70)
        self.assertIn("low_ocr_confidence", rel.degradation_factors)

    def test_08_case_04_reid_similarity_degradation(self) -> None:
        """CASE 4: Re-ID embedding degradation / low cosine similarity in pairwise match."""
        obs_a = {
            "observation_id": "OBS_C04_A",
            "camera_id": "CAM_A",
            "timestamp_seconds": 1000.0,
            "camera_reliability": 0.90,
            "reid_embedding": [1.0, 0.0, 0.0],
        }
        obs_b = {
            "observation_id": "OBS_C04_B",
            "camera_id": "CAM_B",
            "timestamp_seconds": 1120.0,
            "camera_reliability": 0.90,
            "reid_embedding": [0.4, 0.9, 0.0],  # Low cosine similarity (~0.40)
        }
        match_res = match_observations(obs_a, obs_b)
        self.assertIn("uncertainty", match_res)
        self.assertGreaterEqual(match_res["uncertainty"], 0.40)

    def test_09_case_05_camera_mismatch_calibration_disparity(self) -> None:
        """CASE 5: Camera mismatch / calibration disparity between endpoints (CAM_A=0.95, CAM_B=0.30)."""
        obs_a = {
            "observation_id": "OBS_C05_A",
            "camera_id": "CAM_J01",
            "timestamp_seconds": 1000.0,
            "node_id": "J01",
            "camera_reliability": 0.95,
            "plate_text": "DL01AB1234",
            "plate_confidence": 0.95,
        }
        obs_b = {
            "observation_id": "OBS_C05_B",
            "camera_id": "CAM_J02",
            "timestamp_seconds": 1150.0,
            "node_id": "J02",
            "camera_reliability": 0.30,  # Severely degraded second camera
            "plate_text": "DL01AB1234",
            "plate_confidence": 0.80,
        }
        segment = reconstruct_trajectory_segment(
            identity_id="TRK_C05",
            start_obs=obs_a,
            end_obs=obs_b,
            road_graph=self.graph,
        )
        self.assertIsNotNone(segment.reliability)
        # Overall reliability must be bottlenecked by degraded CAM_B
        self.assertLessEqual(segment.reliability["overall_reliability"], 0.65)
        self.assertGreaterEqual(segment.uncertainty["overall_uncertainty"], 0.35)

    def test_10_case_06_sparse_gap_reliable_vs_degraded_endpoints(self) -> None:
        """CASE 6: Sparse gap with reliable endpoints vs degraded endpoints across J01 -> J08 (450s)."""
        # Reliable endpoints
        gap_rel = infer_sparse_gap(
            identity_id="TRK_C06_REL",
            obs_a={
                "observation_id": "OBS_REL_A",
                "camera_id": "CAM_J01",
                "timestamp_seconds": 1000.0,
                "node_id": "J01",
                "camera_reliability": 0.95,
                "plate_confidence": 0.95,
            },
            obs_b={
                "observation_id": "OBS_REL_B",
                "camera_id": "CAM_J08",
                "timestamp_seconds": 1450.0,
                "node_id": "J08",
                "camera_reliability": 0.92,
                "plate_confidence": 0.92,
            },
            road_graph=self.graph,
        )

        # Degraded endpoints
        gap_deg = infer_sparse_gap(
            identity_id="TRK_C06_DEG",
            obs_a={
                "observation_id": "OBS_DEG_A",
                "camera_id": "CAM_J01",
                "timestamp_seconds": 1000.0,
                "node_id": "J01",
                "camera_reliability": 0.40,
                "plate_confidence": 0.50,
            },
            obs_b={
                "observation_id": "OBS_DEG_B",
                "camera_id": "CAM_J08",
                "timestamp_seconds": 1450.0,
                "node_id": "J08",
                "camera_reliability": 0.35,
                "plate_confidence": 0.45,
            },
            road_graph=self.graph,
        )

        # Reliable gap must have strictly higher reliability than degraded gap
        self.assertGreater(gap_rel.reliability["overall_reliability"], gap_deg.reliability["overall_reliability"])
        self.assertLess(gap_rel.uncertainty["overall_uncertainty"], gap_deg.uncertainty["overall_uncertainty"])

    def test_11_case_07_route_entropy_plus_degraded_sensor(self) -> None:
        """CASE 7: High-uncertainty trajectory with multiple routes (J01->J08, 650s) + degraded camera."""
        gap = infer_sparse_gap(
            identity_id="TRK_C07",
            obs_a={
                "observation_id": "OBS_C07_A",
                "camera_id": "CAM_J01",
                "timestamp_seconds": 1000.0,
                "node_id": "J01",
                "camera_reliability": 0.45,
            },
            obs_b={
                "observation_id": "OBS_C07_B",
                "camera_id": "CAM_J08",
                "timestamp_seconds": 1650.0,  # Large window: 3+ candidate corridors
                "node_id": "J08",
                "camera_reliability": 0.45,
            },
            road_graph=self.graph,
        )
        self.assertGreaterEqual(len(gap.candidate_routes), 3)
        self.assertIsNotNone(gap.uncertainty)
        # Uncertainty includes route entropy and sensor penalty
        self.assertGreaterEqual(gap.uncertainty["overall_uncertainty"], 0.45)
        self.assertIn("route_entropy", gap.uncertainty)

    def test_12_case_08_missing_metadata_never_assumes_perfect_trust(self) -> None:
        """CASE 8: Missing metadata handling: None of the fields assume 1.0."""
        obs = {
            "observation_id": "OBS_C08",
            "camera_id": "CAM_UNKNOWN",
            "timestamp_seconds": 1000.0,
            # No reliability or embedding fields
        }
        rel = evaluate_observation_reliability(obs)
        self.assertLessEqual(rel.overall_reliability, 0.65)
        self.assertNotEqual(rel.overall_reliability, 1.0)
        self.assertNotEqual(rel.camera_reliability, 1.0)

    def test_13_case_09_extreme_degradation_untrusted_observation(self) -> None:
        """CASE 9: Extreme degradation / untrusted observation (fails threshold or high uncertainty)."""
        obs = {
            "observation_id": "OBS_C09",
            "camera_id": "CAM_BROKEN",
            "camera_reliability": 0.12,
            "confidence": 0.10,
            "plate_text": "???",
            "plate_confidence": 0.10,
            "reid_embedding": None,
        }
        rel = evaluate_observation_reliability(obs)
        self.assertLessEqual(rel.overall_reliability, 0.25)
        self.assertGreaterEqual(rel.uncertainty, 0.75)
        self.assertIn("severe_detection_degradation", rel.degradation_factors)

    def test_14_case_10_downstream_route_demand_monotonicity(self) -> None:
        """
        CASE 10: Downstream route demand monotonicity under uncertainty.
        vehicle_weight remains strictly unchanged (demand/PCU).
        route_demand = vehicle_weight * route_prob is unchanged.
        uncertainty is cleanly communicated in metadata.
        """
        # Create gap with vehicle_weight = 2.5 (PCU for heavy commercial vehicle)
        gap = infer_sparse_gap(
            identity_id="TRK_C10",
            obs_a={
                "observation_id": "OBS_C10_A",
                "camera_id": "CAM_J01",
                "timestamp_seconds": 1000.0,
                "node_id": "J01",
                "camera_reliability": 0.30,  # Low reliability
            },
            obs_b={
                "observation_id": "OBS_C10_B",
                "camera_id": "CAM_J08",
                "timestamp_seconds": 1450.0,
                "node_id": "J08",
                "camera_reliability": 0.35,
            },
            road_graph=self.graph,
        )

        norm_traj = adapt_sparse_gap_to_normalized(gap, vehicle_weight=2.5)

        # 1. vehicle_weight must be strictly 2.5
        self.assertEqual(norm_traj.vehicle_weight, 2.5)

        # 2. Demand calculations
        demands = norm_traj.calculate_route_demands()
        demand_sum = sum(d["route_demand"] for d in demands)
        self.assertAlmostEqual(demand_sum, 2.5, places=2)

        # 3. Route probabilities still sum to 1.0
        prob_sum = sum(cr.probability for cr in norm_traj.candidate_routes)
        self.assertAlmostEqual(prob_sum, 1.0, places=5)

        # 4. Uncertainty and reliability are present in metadata, NOT polluting vehicle_weight
        self.assertIn("reliability", norm_traj.metadata)
        self.assertIn("uncertainty", norm_traj.metadata)
        self.assertNotEqual(norm_traj.metadata["reliability"], norm_traj.vehicle_weight)

    # -------------------------------------------------------------------------
    # 4. Monotonicity & Mathematical Soundness Tests
    # -------------------------------------------------------------------------
    def test_15_monotonicity_camera_to_trajectory(self) -> None:
        """Verify that strictly decreasing camera reliability monotonically decreases trajectory reliability."""
        reliabilities = [0.95, 0.80, 0.60, 0.40, 0.20]
        traj_reliabilities = []

        for r in reliabilities:
            gap = infer_sparse_gap(
                identity_id=f"TRK_MONO_{r}",
                obs_a={"observation_id": "A", "camera_id": "CAM_J01", "timestamp_seconds": 1000.0, "node_id": "J01", "camera_reliability": r},
                obs_b={"observation_id": "B", "camera_id": "CAM_J08", "timestamp_seconds": 1450.0, "node_id": "J08", "camera_reliability": r},
                road_graph=self.graph,
            )
            traj_reliabilities.append(gap.reliability["overall_reliability"])

        # Check non-increasing
        for i in range(len(traj_reliabilities) - 1):
            self.assertGreaterEqual(
                traj_reliabilities[i],
                traj_reliabilities[i + 1],
                f"Reliability failed monotonicity at step {i}: {traj_reliabilities[i]} < {traj_reliabilities[i+1]}",
            )

    # -------------------------------------------------------------------------
    # 5. Separation of Concepts Tests
    # -------------------------------------------------------------------------
    def test_16_separation_of_concepts(self) -> None:
        """
        Verify complete decoupling of:
        - vehicle_weight: physical PCU / demand multiplier (e.g. 1.0, 2.5).
        - camera_reliability: sensor trustworthiness.
        - observation_reliability: detection quality.
        - candidate route probability: relative estimated likelihood (sum == 1.0).
        """
        gap = infer_sparse_gap(
            identity_id="TRK_SEP",
            obs_a={"observation_id": "A", "camera_id": "CAM_J01", "timestamp_seconds": 1000.0, "node_id": "J01", "camera_reliability": 0.40},
            obs_b={"observation_id": "B", "camera_id": "CAM_J08", "timestamp_seconds": 1450.0, "node_id": "J08", "camera_reliability": 0.40},
            road_graph=self.graph,
        )
        norm = adapt_sparse_gap_to_normalized(gap, vehicle_weight=3.0)

        self.assertEqual(norm.vehicle_weight, 3.0)
        self.assertNotEqual(norm.metadata["reliability"]["overall_reliability"], norm.vehicle_weight)

        prob_sum = sum(cr.probability for cr in norm.candidate_routes)
        self.assertAlmostEqual(prob_sum, 1.0, places=5)

    # -------------------------------------------------------------------------
    # 6. Backward Compatibility with Day 4 Unobserved Intermediate Nodes
    # -------------------------------------------------------------------------
    def test_17_day4_unobserved_intermediate_nodes_preserved(self) -> None:
        """Verify that Day 4 unobserved_intermediate_nodes are fully preserved alongside Day 5 reliability."""
        gap = infer_sparse_gap(
            identity_id="TRK_DAY4_PRESERVE",
            obs_a={"observation_id": "A", "camera_id": "CAM_J01", "timestamp_seconds": 1000.0, "node_id": "J01", "camera_reliability": 0.85},
            obs_b={"observation_id": "B", "camera_id": "CAM_J08", "timestamp_seconds": 1450.0, "node_id": "J08", "camera_reliability": 0.85},
            road_graph=self.graph,
        )

        # Day 4 properties intact
        self.assertTrue(len(gap.unobserved_intermediate_nodes) > 0)
        self.assertNotIn("J01", gap.unobserved_intermediate_nodes)
        self.assertNotIn("J08", gap.unobserved_intermediate_nodes)

        # Route-level unobserved nodes
        for r in gap.candidate_routes:
            if r.feasible:
                self.assertEqual(r.unobserved_intermediate_nodes, list(r.nodes[1:-1]))
                # Day 5 route-level reliability and uncertainty present
                self.assertIsNotNone(r.reliability)
                self.assertIsNotNone(r.uncertainty)

        # Member 3 adapted representation has both Day 4 intermediate nodes and Day 5 reliability
        norm = adapt_sparse_gap_to_normalized(gap)
        self.assertIn("unobserved_intermediate_nodes", norm.metadata)
        self.assertIn("reliability", norm.metadata)
        self.assertIn("uncertainty", norm.metadata)

    # -------------------------------------------------------------------------
    # 7. Audit 8: Controlled Comparison (Observation Quality vs Identity Completeness)
    # -------------------------------------------------------------------------
    def test_18_audit8_controlled_comparison_observation_vs_identity(self) -> None:
        """
        Audit 8: Controlled comparison where underlying physical observation remains the same.
        Case A: reliable camera (0.95), strong detection (0.90), no ReID, no plate.
        Case B: degraded camera (0.35), same detection (0.90), same missing ReID, same missing plate.
        Case C: same reliable camera (0.95), same detection (0.90), but WITH ReID and plate.

        Validates:
        1. Case B has lower observation trust than Case A because the camera is degraded.
        2. Changing 'ReID available' to 'ReID unavailable' (Case C vs Case A) does NOT degrade observation_quality.
        3. Missing identity evidence is recorded under missing_evidence, NOT in degradation_factors.
        4. Distinction is visible in structured output and human-readable explanation.
        """
        obs_a = {
            "observation_id": "OBS_CTRL_A",
            "camera_id": "CAM_HIGH",
            "camera_reliability": 0.95,
            "confidence": 0.90,
            # No plate, no reid
        }
        obs_b = {
            "observation_id": "OBS_CTRL_B",
            "camera_id": "CAM_LOW",
            "camera_reliability": 0.35,
            "confidence": 0.90,
            # Same missing plate, same missing reid
        }
        obs_c = {
            "observation_id": "OBS_CTRL_C",
            "camera_id": "CAM_HIGH",
            "camera_reliability": 0.95,
            "confidence": 0.90,
            "plate_text": "DL01AB1234",
            "plate_confidence": 0.95,
            "reid_embedding": [0.5] * 128,
        }

        rel_a = evaluate_observation_reliability(obs_a)
        rel_b = evaluate_observation_reliability(obs_b)
        rel_c = evaluate_observation_reliability(obs_c)

        # 1. Observation quality is degraded in Case B compared to Case A due to sensor degradation
        self.assertGreater(rel_a.observation_quality, rel_b.observation_quality)
        self.assertGreater(rel_a.overall_reliability, rel_b.overall_reliability)
        self.assertIn("degraded_camera_sensor", rel_b.degradation_factors)

        # 2. Crucial: Physical observation_quality between Case A and Case C is IDENTICAL
        # Simply removing ReID/plate does NOT alter physical sensor observation quality
        self.assertAlmostEqual(rel_a.observation_quality, rel_c.observation_quality, places=4)

        # 3. Identity evidence completeness is cleanly separated
        self.assertEqual(rel_a.identity_evidence_quality, 0.0)
        self.assertGreaterEqual(rel_c.identity_evidence_quality, 0.95)

        # 4. Missing evidence vs degradation factors separation
        self.assertIn("appearance_embedding", rel_a.missing_evidence)
        self.assertIn("plate_evidence", rel_a.missing_evidence)
        self.assertNotIn("missing_appearance_embedding", rel_a.degradation_factors)
        self.assertNotIn("missing_plate_evidence", rel_a.degradation_factors)

        # 5. Explanations clearly distinguish physical quality from identity evidence
        self.assertIn("physical observation quality is strong", rel_a.explanation)
        self.assertIn("identity evidence is incomplete", rel_a.explanation)
        self.assertIn("physical observation quality is degraded", rel_b.explanation)

    # -------------------------------------------------------------------------
    # 8. Audit 9: Comprehensive Monotonicity Suite
    # -------------------------------------------------------------------------
    def test_19_audit9_comprehensive_monotonicity_suite(self) -> None:
        """
        Audit 9: Validated through controlled monotonicity tests across 5 key dimensions:
        1. Lower camera reliability -> non-increasing observation reliability.
        2. Lower observation reliability -> non-increasing trajectory reliability.
        3. More missing intermediate observations -> non-decreasing uncertainty.
        4. More route competition (entropy) -> non-decreasing uncertainty.
        5. Changing vehicle_weight -> zero change to reliability or uncertainty.
        """
        # Dim 1: Camera -> Observation reliability monotonicity
        cam_levels = [0.95, 0.85, 0.65, 0.45, 0.25, 0.10]
        obs_rels = []
        for c in cam_levels:
            r = evaluate_observation_reliability({
                "observation_id": f"OBS_MONO_{c}",
                "camera_id": "CAM_TEST",
                "camera_reliability": c,
                "confidence": 0.85,
            })
            obs_rels.append(r.overall_reliability)

        for i in range(len(obs_rels) - 1):
            self.assertGreaterEqual(
                obs_rels[i], obs_rels[i + 1],
                f"Camera->Obs monotonicity failed: {obs_rels[i]} < {obs_rels[i+1]}"
            )

        # Dim 2: Observation -> Trajectory reliability monotonicity
        traj_rels = []
        for c in cam_levels:
            gap = infer_sparse_gap(
                identity_id=f"TRK_MONO_{c}",
                obs_a={"observation_id": "A", "camera_id": "CAM_J01", "timestamp_seconds": 1000.0, "node_id": "J01", "camera_reliability": c},
                obs_b={"observation_id": "B", "camera_id": "CAM_J08", "timestamp_seconds": 1450.0, "node_id": "J08", "camera_reliability": c},
                road_graph=self.graph,
            )
            traj_rels.append(gap.reliability["overall_reliability"])

        for i in range(len(traj_rels) - 1):
            self.assertGreaterEqual(
                traj_rels[i], traj_rels[i + 1],
                f"Obs->Traj monotonicity failed: {traj_rels[i]} < {traj_rels[i+1]}"
            )

        # Dim 3: Missing intermediate observations penalty monotonicity
        gap_0 = propagate_trajectory_uncertainty(
            "T0",
            evaluate_observation_reliability({"observation_id": "A", "camera_id": "C1", "camera_reliability": 0.85}),
            evaluate_observation_reliability({"observation_id": "B", "camera_id": "C2", "camera_reliability": 0.85}),
            [],
            is_gap=False,
            unobserved_intermediate_nodes=[],
            gap_duration_seconds=0.0,
        )
        gap_1 = propagate_trajectory_uncertainty(
            "T1",
            evaluate_observation_reliability({"observation_id": "A", "camera_id": "C1", "camera_reliability": 0.85}),
            evaluate_observation_reliability({"observation_id": "B", "camera_id": "C2", "camera_reliability": 0.85}),
            [],
            is_gap=True,
            unobserved_intermediate_nodes=["J02"],
            gap_duration_seconds=120.0,
        )
        gap_3 = propagate_trajectory_uncertainty(
            "T3",
            evaluate_observation_reliability({"observation_id": "A", "camera_id": "C1", "camera_reliability": 0.85}),
            evaluate_observation_reliability({"observation_id": "B", "camera_id": "C2", "camera_reliability": 0.85}),
            [],
            is_gap=True,
            unobserved_intermediate_nodes=["J02", "J03", "J04"],
            gap_duration_seconds=360.0,
        )
        self.assertLessEqual(gap_0.overall_uncertainty, gap_1.overall_uncertainty)
        self.assertLessEqual(gap_1.overall_uncertainty, gap_3.overall_uncertainty)

        # Dim 4: Route competition / entropy monotonicity
        class MockRoute:
            def __init__(self, p: float, sc: float = 1.0) -> None:
                self.feasible = True
                self.estimated_likelihood = p
                self.raw_score = sc

        res_single = propagate_trajectory_uncertainty(
            "T_SINGLE",
            evaluate_observation_reliability({"observation_id": "A", "camera_id": "C1", "camera_reliability": 0.85}),
            evaluate_observation_reliability({"observation_id": "B", "camera_id": "C2", "camera_reliability": 0.85}),
            [MockRoute(1.0)],
        )
        res_competing = propagate_trajectory_uncertainty(
            "T_COMPETE",
            evaluate_observation_reliability({"observation_id": "A", "camera_id": "C1", "camera_reliability": 0.85}),
            evaluate_observation_reliability({"observation_id": "B", "camera_id": "C2", "camera_reliability": 0.85}),
            [MockRoute(0.51), MockRoute(0.49)],
        )
        self.assertLess(res_single.overall_uncertainty, res_competing.overall_uncertainty)

        # Dim 5: Vehicle weight variation has ZERO effect on reliability and uncertainty
        gap_base = infer_sparse_gap(
            identity_id="TRK_WEIGHT_TEST",
            obs_a={"observation_id": "A", "camera_id": "CAM_J01", "timestamp_seconds": 1000.0, "node_id": "J01", "camera_reliability": 0.70},
            obs_b={"observation_id": "B", "camera_id": "CAM_J08", "timestamp_seconds": 1450.0, "node_id": "J08", "camera_reliability": 0.70},
            road_graph=self.graph,
        )
        for w in [0.5, 1.0, 2.5, 5.0]:
            adapted = adapt_sparse_gap_to_normalized(gap_base, vehicle_weight=w)
            self.assertEqual(adapted.vehicle_weight, w)
            self.assertEqual(
                adapted.metadata["reliability"]["overall_reliability"],
                gap_base.reliability["overall_reliability"],
            )
            self.assertEqual(
                adapted.metadata["uncertainty"]["overall_uncertainty"],
                gap_base.uncertainty["overall_uncertainty"],
            )

    # -------------------------------------------------------------------------
    # 9. Audit 10 & 11: Semantics, Metadata, and Explanations
    # -------------------------------------------------------------------------
    def test_20_audit10_11_semantics_and_explanations(self) -> None:
        """
        Audit 10 & 11: Validate semantic precision and distinct explanations.
        - Route probability metadata clarifies relative estimated likelihood.
        - Route reliability metadata clarifies evidence trustworthiness.
        - Route uncertainty metadata clarifies 1.0 - reliability (not 1.0 - probability).
        """
        gap = infer_sparse_gap(
            identity_id="TRK_SEM_TEST",
            obs_a={"observation_id": "A", "camera_id": "CAM_J01", "timestamp_seconds": 1000.0, "node_id": "J01", "camera_reliability": 0.85},
            obs_b={"observation_id": "B", "camera_id": "CAM_J08", "timestamp_seconds": 1450.0, "node_id": "J08", "camera_reliability": 0.85},
            road_graph=self.graph,
        )
        norm = adapt_sparse_gap_to_normalized(gap)

        # Check candidate route metadata semantics
        for cr in norm.candidate_routes:
            meta = cr.metadata
            self.assertIn("probability_semantics", meta)
            self.assertEqual(meta["probability_semantics"], "relative_estimated_likelihood_among_feasible_routes")
            self.assertIn("reliability_semantics", meta)
            self.assertEqual(meta["reliability_semantics"], "evidence_trustworthiness_supporting_corridor_inference")
            self.assertIn("uncertainty_semantics", meta)
            self.assertEqual(meta["uncertainty_semantics"], "evidence_uncertainty (1.0 - reliability)")
            # Route probability and route reliability are distinct numeric values
            self.assertNotEqual(cr.probability, meta.get("reliability"))


if __name__ == "__main__":
    unittest.main()
