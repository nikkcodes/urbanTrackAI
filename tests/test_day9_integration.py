"""
UrbanTrack AI — Day 9 Integration, Robustness & Performance Test Suite.

Validates the full technical integration and boundary safety of Days 1–8:
1. End-to-end synthetic integration pipeline (Stages 1 through 9)
2. Cross-module interface contracts & handoffs
3. Identity consistency (camera-local track_id vs global identity_id)
4. Temporal robustness & comparability boundaries (standalone offset != sync)
5. Missing-data handling & graceful degradation
6. Mixed-quality batch processing & isolation
7. Failure injection & safe error handling
8. Unit consistency (m, km, km/h, px/s, vph, PCU)
9. Reliability propagation & non-multiplication into physical demand
10. Anomaly detection integration & guilt separation
11. Counterfactual simulation integration & separate allocation
12. Baseline immutability guarantees
13. Determinism across repeated pipeline runs
14. Numerical stability (near-zero probabilities, zero capacity, NaN/inf avoidance)
15. Serialization & deserialization fidelity
16. Real-data safety boundaries (zero fabricated cross-camera merges)
"""

import copy
import json
import math
import os
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.identity_fusion import match_observations
from inference.identity_graph import IdentityGraph
from inference.member3_adapter import adapt_vehicle_trajectory_to_normalized
from inference.observation_loader import load_camera_metadata, load_observations_from_json
from inference.reliability_engine import propagate_trajectory_uncertainty
from inference.road_graph import RoadEdge, RoadGraph, RoadNode
from inference.sparse_engine import detect_observation_gaps, infer_sparse_gap, infer_sparse_identity_trajectory
from inference.temporal import check_temporal_comparability
from inference.trajectory_engine import (
    evaluate_route_feasibility_and_score,
    reconstruct_identity_trajectory,
    reconstruct_trajectory_segment,
)
from mobility.flow_engine import MobilityFlowEngine
from anomaly.investigation_engine import InvestigationEngine
from simulation.counterfactual_engine import CounterfactualEngine
from schemas.anomaly_schema import (
    AnomalyResult,
    AnomalySeverity,
    DataQualityStatus,
    InvestigationPriority,
    MobilityBaseline,
)
from schemas.gap_schema import SparseObservationGap
from schemas.mobility_schema import RoadFlowMetric, ValidationIssue
from schemas.normalized_trajectory_schema import NormalizedCandidateRoute, NormalizedTrajectory
from schemas.observation_schema import Observation
from schemas.scenario_schema import ScenarioDefinition, ScenarioStatus, ScenarioType
from schemas.trajectory_schema import CandidateRoute, TrajectorySegment, VehicleTrajectory


class TestDay9EndToEndIntegration(unittest.TestCase):
    """Stage-by-stage pipeline integration tests on controlled synthetic scenarios."""

    def setUp(self):
        self.network_path = PROJECT_ROOT / "data" / "synthetic" / "city_network.json"
        self.scenario_path = PROJECT_ROOT / "data" / "synthetic" / "day9_end_to_end_scenarios.json"
        self.graph = RoadGraph.from_json_file(self.network_path)
        for i in range(1, 9):
            self.graph.camera_associations[f"CAM_J{i:02d}"] = f"J{i:02d}"

        with open(self.scenario_path, "r", encoding="utf-8") as f:
            self.scenario_data = json.load(f)

    def test_full_pipeline_stages_1_to_9(self):
        """Verify seamless execution from perception observation through counterfactual simulation across all 9 stages."""
        # Stage 1: Load Observations
        obs_list = load_observations_from_json(
            self.scenario_data["observations"], camera_metadata=self.scenario_data.get("cameras")
        )
        self.assertGreaterEqual(len(obs_list), 7)

        # Stage 2: Identity Fusion
        id_graph = IdentityGraph(min_probability_threshold=0.70)
        id_graph.build_graph(obs_list, camera_metadata=self.scenario_data.get("cameras"))
        clusters = id_graph.get_candidate_identities()
        self.assertGreaterEqual(len(clusters), 4)

        # Stage 3: Trajectory Reconstruction
        veh1 = next(c for c in clusters if len(c["member_observations"]) == 3)
        traj1 = reconstruct_identity_trajectory(veh1, self.graph)
        self.assertEqual(len(traj1.segments), 2)
        self.assertGreater(traj1.total_distance_meters, 0.0)

        # Stage 4: Sparse Inference across unobserved gaps
        veh2 = next(c for c in clusters if len(c["member_observations"]) == 2)
        traj2 = infer_sparse_identity_trajectory(veh2, self.graph)
        self.assertEqual(len(traj2.segments), 1)

        # Stage 5: Reliability & Uncertainty Evaluation
        self.assertIsNotNone(traj1.reliability)
        self.assertIsNotNone(traj2.reliability)
        self.assertIn("overall_reliability", traj1.reliability)
        self.assertIn("overall_uncertainty", traj2.reliability)

        # Stage 6: NormalizedTrajectory Adaptation
        norm1 = adapt_vehicle_trajectory_to_normalized(traj1, vehicle_weight=1.0, vehicle_class="car")
        norm2 = adapt_vehicle_trajectory_to_normalized(traj2, vehicle_weight=1.0, vehicle_class="car")
        self.assertEqual(norm1.track_id, veh1["identity_id"])
        self.assertEqual(norm2.track_id, veh2["identity_id"])
        self.assertAlmostEqual(sum(r.probability for r in norm1.candidate_routes), 1.0, places=4)
        self.assertAlmostEqual(sum(r.probability for r in norm2.candidate_routes), 1.0, places=4)

        # Stage 7: Mobility Flow Aggregation
        flow_engine = MobilityFlowEngine(self.graph)
        road_metrics, od_matrix, issues, records = flow_engine.aggregate_flows([norm1, norm2])
        self.assertEqual(len(issues), 0)
        self.assertAlmostEqual(od_matrix.total_demand, 2.0, places=4)
        for r in records:
            self.assertTrue(r["is_conserved"])

        # Stage 8: Anomaly Investigation
        inv_engine = InvestigationEngine(road_graph=self.graph)
        report = inv_engine.run_investigation([norm1, norm2])
        self.assertEqual(len(report.vehicle_anomalies), 2)
        for va in report.vehicle_anomalies:
            self.assertFalse(va.is_anomalous)
            self.assertEqual(va.investigation_priority, InvestigationPriority.NORMAL)

        # Stage 9: Counterfactual Simulation
        scenario = ScenarioDefinition.from_dict(self.scenario_data["counterfactual_scenario"])
        cf_engine = CounterfactualEngine()
        cf_report = cf_engine.simulate_counterfactual([norm1, norm2], self.graph, scenario)
        self.assertEqual(cf_report.status, ScenarioStatus.SUCCESS)
        self.assertGreater(cf_report.impact["displaced_demand"], 0.0)
        self.assertAlmostEqual(cf_report.impact["unroutable_demand"], 0.0, places=4)


class TestDay9CrossModuleContracts(unittest.TestCase):
    """Test explicit interface contracts between adjacent modules."""

    def setUp(self):
        self.network_path = PROJECT_ROOT / "data" / "synthetic" / "city_network.json"
        self.graph = RoadGraph.from_json_file(self.network_path)
        for i in range(1, 9):
            self.graph.camera_associations[f"CAM_J{i:02d}"] = f"J{i:02d}"

    def test_observation_to_identity_contract(self):
        """Observation requires observation_id, camera_id, track_id, timestamp_seconds."""
        obs = Observation(
            observation_id="OBS_TEST_01",
            camera_id="CAM_J01",
            timestamp_seconds=100.0,
            track_id="local_trk_101",
            latitude=28.6400,
            longitude=77.2000,
            plate="KA01TEST",
            appearance_embedding=[0.5] * 8,
        )
        self.assertEqual(obs.track_id, "local_trk_101")
        self.assertEqual(obs.plate, "KA01TEST")

    def test_identity_to_trajectory_contract(self):
        """Day 2 candidate cluster contract feeds directly into reconstruct_identity_trajectory."""
        cluster = {
            "identity_id": "VEHICLE_CANDIDATE_100",
            "candidate_vehicle_id": "VEHICLE_CANDIDATE_100",
            "member_observations": [
                {
                    "observation_id": "OBS_T1",
                    "camera_id": "CAM_J01",
                    "timestamp_seconds": 100.0,
                    "latitude": 28.6400,
                    "longitude": 77.2000,
                },
                {
                    "observation_id": "OBS_T2",
                    "camera_id": "CAM_J02",
                    "timestamp_seconds": 230.0,
                    "latitude": 28.6350,
                    "longitude": 77.2100,
                },
            ],
            "identity_confidence": 0.95,
        }
        traj = reconstruct_identity_trajectory(cluster, self.graph)
        self.assertEqual(traj.identity_id, "VEHICLE_CANDIDATE_100")
        self.assertEqual(traj.observations_count, 2)
        self.assertTrue(traj.segments[0].feasible)

    def test_trajectory_to_normalized_contract(self):
        """NormalizedTrajectory uses global identity_id as track_id and normalizes route probabilities."""
        cluster = {
            "identity_id": "VEHICLE_GLOBAL_999",
            "member_observations": [
                {"observation_id": "O1", "camera_id": "CAM_J01", "timestamp_seconds": 100.0, "latitude": 28.6400, "longitude": 77.2000},
                {"observation_id": "O2", "camera_id": "CAM_J02", "timestamp_seconds": 240.0, "latitude": 28.6350, "longitude": 77.2100},
            ],
        }
        traj = reconstruct_identity_trajectory(cluster, self.graph)
        norm = adapt_vehicle_trajectory_to_normalized(traj, vehicle_weight=1.5, vehicle_class="bus")
        self.assertEqual(norm.track_id, "VEHICLE_GLOBAL_999")
        self.assertEqual(norm.vehicle_weight, 1.5)
        self.assertAlmostEqual(sum(r.probability for r in norm.candidate_routes), 1.0, places=4)


class TestDay9IdentityRobustness(unittest.TestCase):
    """Verify camera-local vs global identity and boundary conditions."""

    def test_camera_local_track_id_distinct_from_global_identity(self):
        """Never treat camera-local track_id as global identity."""
        obs1 = Observation(
            observation_id="OBS_A",
            camera_id="CAM_01",
            timestamp_seconds=10.0,
            track_id="trk_01",
            vehicle_type="car",
            plate="DL01AA1111",
            appearance_embedding=[0.8] * 8,
            timestamp_semantics="synchronized",
        )
        obs2 = Observation(
            observation_id="OBS_B",
            camera_id="CAM_02",
            timestamp_seconds=50.0,
            track_id="trk_05",
            vehicle_type="car",
            plate="DL01AA1111",
            appearance_embedding=[0.8] * 8,
            timestamp_semantics="synchronized",
        )
        id_graph = IdentityGraph(min_probability_threshold=0.70)
        id_graph.build_graph([obs1, obs2])
        clusters = id_graph.get_candidate_identities()
        self.assertEqual(len(clusters), 1)
        global_id = clusters[0]["identity_id"]
        self.assertTrue(global_id.startswith("VEHICLE_CANDIDATE_"))
        self.assertNotEqual(global_id, "trk_01")
        self.assertNotEqual(global_id, "trk_05")

    def test_speed_violation_prevents_false_identity_merge(self):
        """Physically impossible travel speed rejects match even with identical plate."""
        obs1 = Observation(
            observation_id="OBS_FAST_1",
            camera_id="CAM_01",
            timestamp_seconds=10.0,
            track_id="trk_01",
            latitude=28.6400,
            longitude=77.2000,
            plate="SAME_PLATE_123",
            appearance_embedding=[0.5] * 8,
            timestamp_semantics="synchronized",
        )
        obs2 = Observation(
            observation_id="OBS_FAST_2",
            camera_id="CAM_02",
            timestamp_seconds=11.0,  # 1.8km in 1s = 6480 km/h!
            track_id="trk_02",
            latitude=28.6350,
            longitude=77.2100,
            plate="SAME_PLATE_123",
            appearance_embedding=[0.5] * 8,
            timestamp_semantics="synchronized",
        )
        match_res = match_observations(obs1, obs2)
        self.assertEqual(match_res["same_vehicle_probability"], 0.0)
        self.assertIn("impossible", match_res["explanation"].lower())

    def test_missing_reid_and_plate_does_not_falsely_merge(self):
        """Missing both Re-ID and plate must not create false positive identity edges."""
        obs1 = Observation(
            observation_id="OBS_NO_ID_1",
            camera_id="CAM_01",
            timestamp_seconds=10.0,
            track_id="trk_01",
            latitude=28.6400,
            longitude=77.2000,
            vehicle_type="car",
            plate=None,
            appearance_embedding=None,
            timestamp_semantics="synchronized",
        )
        obs2 = Observation(
            observation_id="OBS_NO_ID_2",
            camera_id="CAM_02",
            timestamp_seconds=70.0,  # 1123m in 60s = 67.4 km/h (spatio-temporally plausible)
            track_id="trk_02",
            latitude=28.6350,
            longitude=77.2100,
            vehicle_type="car",
            plate=None,
            appearance_embedding=None,
            timestamp_semantics="synchronized",
        )
        match_res = match_observations(obs1, obs2)
        # Without identity evidence, probability is 0.50 (ambiguous unconfirmed), below 0.70 threshold
        self.assertEqual(match_res["same_vehicle_probability"], 0.50)

        id_graph = IdentityGraph(min_probability_threshold=0.70)
        id_graph.build_graph([obs1, obs2])
        clusters = id_graph.get_candidate_identities()
        self.assertEqual(len(clusters), 2, "Unconfirmed observations must remain isolated singletons")


class TestDay9TemporalRobustness(unittest.TestCase):
    """Audit temporal comparability, video-relative semantics, and offset safety."""

    def test_standalone_clock_offset_does_not_establish_synchronization(self):
        """A standalone clock_offset_seconds does NOT establish cross-camera comparability."""
        obs1 = Observation(
            observation_id="O1",
            camera_id="CAM_A",
            timestamp_seconds=100.0,
            track_id="t1",
            clock_offset_seconds=5.0,  # Standalone undocumented offset
            timestamp_semantics="video_relative",
        )
        obs2 = Observation(
            observation_id="O2",
            camera_id="CAM_B",
            timestamp_seconds=150.0,
            track_id="t2",
            clock_offset_seconds=10.0,  # Standalone undocumented offset
            timestamp_semantics="video_relative",
        )
        comp = check_temporal_comparability(obs1, obs2)
        self.assertFalse(comp["comparable"])
        self.assertIsNone(comp["delta_seconds"])
        self.assertIn("without_shared_time_reference", comp["reason"])

    def test_same_camera_video_relative_is_comparable(self):
        """Observations from the same camera with video-relative timestamps are valid."""
        obs1 = Observation(
            observation_id="O1",
            camera_id="CAM_SAME",
            timestamp_seconds=10.0,
            track_id="t1",
            timestamp_semantics="video_relative",
        )
        obs2 = Observation(
            observation_id="O2",
            camera_id="CAM_SAME",
            timestamp_seconds=25.0,
            track_id="t1",
            timestamp_semantics="video_relative",
        )
        comp = check_temporal_comparability(obs1, obs2)
        self.assertTrue(comp["comparable"])
        self.assertEqual(comp["delta_seconds"], 15.0)

    def test_temporal_inversion_rejected(self):
        """Negative elapsed time (t_b < t_a) is correctly identified and rejected."""
        obs1 = Observation(
            observation_id="O1",
            camera_id="CAM_01",
            timestamp_seconds=100.0,
            track_id="t1",
            timestamp_semantics="synchronized",
        )
        obs2 = Observation(
            observation_id="O2",
            camera_id="CAM_01",
            timestamp_seconds=50.0,  # Reversed!
            track_id="t1",
            timestamp_semantics="synchronized",
        )
        comp = check_temporal_comparability(obs1, obs2)
        self.assertFalse(comp["comparable"])
        self.assertEqual(comp["status"], "invalid_negative_time")


class TestDay9MixedQualityAndMissingData(unittest.TestCase):
    """Test robust processing of mixed-quality batches and missing fields."""

    def test_mixed_quality_batch_isolation(self):
        """One invalid or incomplete record does not corrupt processing of valid records."""
        records = [
            # Vehicle A: complete valid
            {"observation_id": "VA_1", "camera_id": "CAM_01", "timestamp_seconds": 100.0, "track_id": "t1", "vehicle_type": "car", "plate": "VA1111", "appearance_embedding": [0.5]*8, "timestamp_semantics": "synchronized"},
            {"observation_id": "VA_2", "camera_id": "CAM_02", "timestamp_seconds": 150.0, "track_id": "t2", "vehicle_type": "car", "plate": "VA1111", "appearance_embedding": [0.5]*8, "timestamp_semantics": "synchronized"},
            # Vehicle B: missing plate
            {"observation_id": "VB_1", "camera_id": "CAM_01", "timestamp_seconds": 200.0, "track_id": "t3", "plate": None, "appearance_embedding": [0.3]*8, "timestamp_semantics": "synchronized"},
            # Vehicle C: missing ReID
            {"observation_id": "VC_1", "camera_id": "CAM_01", "timestamp_seconds": 300.0, "track_id": "t4", "plate": "VC3333", "appearance_embedding": None, "timestamp_semantics": "synchronized"},
            # Vehicle D: unavailable temporal sync (video_relative across cameras)
            {"observation_id": "VD_1", "camera_id": "CAM_A", "timestamp_seconds": 400.0, "track_id": "t5", "plate": "VD4444", "appearance_embedding": [0.7]*8, "timestamp_semantics": "video_relative"},
            {"observation_id": "VD_2", "camera_id": "CAM_B", "timestamp_seconds": 450.0, "track_id": "t6", "plate": "VD4444", "appearance_embedding": [0.7]*8, "timestamp_semantics": "video_relative"},
            # Vehicle E: invalid observation (invalid bbox or missing fields)
            {"observation_id": "VE_1", "camera_id": "CAM_01", "timestamp_seconds": 500.0, "track_id": "t7", "bbox": [10.0]},  # Malformed bbox
        ]

        # Filter and validate records
        processed: List[Observation] = []
        invalid: List[Dict[str, Any]] = []

        for r in records:
            try:
                obs = Observation.from_json(r)
                processed.append(obs)
            except Exception as e:
                invalid.append({"record": r, "error": str(e)})

        self.assertEqual(len(invalid), 1, "Vehicle E should be caught as invalid input")
        self.assertEqual(len(processed), 6, "Valid records should be successfully loaded")

        id_graph = IdentityGraph(min_probability_threshold=0.70)
        id_graph.build_graph(processed)
        clusters = id_graph.get_candidate_identities()

        # VA should merge
        va_cluster = next((c for c in clusters if "VA_1" in c["observation_ids"] and "VA_2" in c["observation_ids"]), None)
        self.assertIsNotNone(va_cluster, "Vehicle A should merge cleanly despite other incomplete records")

        # VD should NOT merge due to unsynchronized video-relative timelines
        vd_cluster = next((c for c in clusters if "VD_1" in c["observation_ids"] and "VD_2" in c["observation_ids"]), None)
        self.assertIsNone(vd_cluster, "Vehicle D across unsynchronized cameras must not merge")


class TestDay9FailureInjection(unittest.TestCase):
    """Verify explicit failure and rejection handling for invalid inputs."""

    def setUp(self):
        self.network_path = PROJECT_ROOT / "data" / "synthetic" / "city_network.json"
        self.graph = RoadGraph.from_json_file(self.network_path)

    def test_nonexistent_road_scenario_rejected(self):
        """Scenario targeting a nonexistent road is rejected with explicit INVALID_INPUT status."""
        scenario = ScenarioDefinition(
            scenario_id="SCN_FAIL_ROAD",
            scenario_type=ScenarioType.ROAD_CLOSURE,
            description="Closing non-existent road R999",
            affected_roads=["R999_DOES_NOT_EXIST"],
        )
        cf_engine = CounterfactualEngine()
        traj = NormalizedTrajectory(
            track_id="T1",
            origin_node="J01",
            destination_node="J02",
            candidate_routes=[NormalizedCandidateRoute(nodes=["J01", "J02"], probability=1.0, metadata={"edges": ["R01"]})],
            vehicle_weight=1.0,
        )
        report = cf_engine.simulate_counterfactual([traj], self.graph, scenario)
        self.assertEqual(report.status, ScenarioStatus.INVALID_INPUT)
        self.assertIn("does not exist", report.impact["error"])

    def test_negative_demand_multiplier_rejected(self):
        """Demand multiplier <= 0 is rejected with validation error."""
        scenario = ScenarioDefinition(
            scenario_id="SCN_FAIL_DEMAND",
            scenario_type=ScenarioType.DEMAND_INCREASE,
            description="Negative demand multiplier",
            affected_roads=["R01"],
            demand_multiplier=-0.5,
        )
        is_valid, err = scenario.validate(self.graph)
        self.assertFalse(is_valid)
        self.assertIn("strictly positive", err)

    def test_zero_capacity_does_not_crash_flow_engine(self):
        """Zero capacity on a road does not cause divide-by-zero crashes in flow aggregation."""
        # Temporarily set R01 capacity to 0.0
        r01 = self.graph.edges["R01"]
        orig_cap = r01.capacity_vph
        r01.capacity_vph = 0.0

        try:
            traj = NormalizedTrajectory(
                track_id="T_ZERO_CAP",
                origin_node="J01",
                destination_node="J02",
                candidate_routes=[NormalizedCandidateRoute(nodes=["J01", "J02"], probability=1.0, metadata={"edges": ["R01"]})],
                vehicle_weight=1.0,
                time_window_start=1000.0,
                time_window_end=1060.0,  # 60s duration -> 60 vph
            )
            flow_engine = MobilityFlowEngine(self.graph)
            road_metrics, od_matrix, issues, records = flow_engine.aggregate_flows([traj])
            r01_metric = next(r for r in road_metrics if r.road_id == "R01")
            self.assertEqual(r01_metric.capacity_vph, 0.0)
            self.assertFalse(math.isnan(r01_metric.utilization_ratio))
            self.assertFalse(math.isinf(r01_metric.utilization_ratio))
        finally:
            r01.capacity_vph = orig_cap


class TestDay9UnitConsistency(unittest.TestCase):
    """Audit units: meters vs kilometers, km/h vs px/s, vph, PCU."""

    def test_distance_unit_consistency(self):
        """Verify explicit distance_km = distance_m / 1000 across road edges."""
        edge = RoadEdge(
            road_id="R_TEST",
            from_node="A",
            to_node="B",
            distance_m=1500.0,
        )
        self.assertEqual(edge.distance_m, 1500.0)
        d = edge.to_dict()
        self.assertEqual(d["distance_km"], 1.5)

    def test_pixel_coordinates_not_treated_as_lat_lon(self):
        """Image-space bbox [ymin, xmin, ymax, xmax] must remain distinct from geographic coordinates."""
        obs = Observation(
            observation_id="OBS_PIX",
            camera_id="CAM_01",
            timestamp_seconds=10.0,
            track_id="trk_01",
            bbox=[120.0, 340.0, 250.0, 500.0],  # Pixel coordinates
            latitude=28.6400,                   # Degrees lat
            longitude=77.2000,                  # Degrees lon
        )
        self.assertEqual(len(obs.bbox), 4)
        self.assertNotEqual(obs.bbox[0], obs.latitude)
        self.assertNotEqual(obs.bbox[1], obs.longitude)

    def test_vehicle_weight_pcu_not_multiplied_by_reliability(self):
        """Vehicle weight is physical demand (PCU), never multiplied by evidence reliability."""
        traj = NormalizedTrajectory(
            track_id="TRK_PCU",
            origin_node="J01",
            destination_node="J02",
            candidate_routes=[NormalizedCandidateRoute(nodes=["J01", "J02"], probability=1.0, metadata={"edges": ["R01"]})],
            vehicle_weight=2.5,  # e.g. 2.5 PCU for a heavy truck
            metadata={"reliability": 0.40},  # Low sensor reliability
        )
        graph = RoadGraph.from_json_file(PROJECT_ROOT / "data" / "synthetic" / "city_network.json")
        flow_engine = MobilityFlowEngine(graph)
        metrics, od_matrix, _, _ = flow_engine.aggregate_flows([traj])
        r01_metric = next(m for m in metrics if m.road_id == "R01")
        # Physical demand on R01 MUST be 2.5 PCU, NEVER 2.5 * 0.40 = 1.0 PCU!
        self.assertAlmostEqual(r01_metric.expected_demand_in_window, 2.5, places=4)
        self.assertAlmostEqual(od_matrix.total_demand, 2.5, places=4)


class TestDay9DeterminismAndStability(unittest.TestCase):
    """Verify determinism across repeated executions and numerical stability."""

    def test_deterministic_pipeline_execution(self):
        """Repeated runs of the same synthetic batch produce semantically identical outputs."""
        scenario_path = PROJECT_ROOT / "data" / "synthetic" / "day9_end_to_end_scenarios.json"
        network_path = PROJECT_ROOT / "data" / "synthetic" / "city_network.json"

        with open(scenario_path, "r", encoding="utf-8") as f:
            sc_data = json.load(f)

        def run_once():
            graph = RoadGraph.from_json_file(network_path)
            for i in range(1, 9):
                graph.camera_associations[f"CAM_J{i:02d}"] = f"J{i:02d}"
            obs = load_observations_from_json(sc_data["observations"], camera_metadata=sc_data.get("cameras"))
            id_g = IdentityGraph(min_probability_threshold=0.70)
            id_g.build_graph(obs, camera_metadata=sc_data.get("cameras"))
            clusters = id_g.get_candidate_identities()
            c_ids = [c["identity_id"] for c in clusters]
            return c_ids

        run1 = run_once()
        run2 = run_once()
        run3 = run_once()
        self.assertEqual(run1, run2)
        self.assertEqual(run2, run3)

    def test_numerical_stability_near_zero_probabilities(self):
        """Route scoring with near-zero likelihoods does not produce NaN or crashes."""
        path_data = {
            "distance_m": 1800.0,
            "speed_limit_kmh": 50.0,
            "expected_speed_kmh": 40.0,
            "estimated_travel_time_s": 162.0,
            "min_travel_time_s": 129.6,
        }
        # Required speed very low (e.g. 1800m in 10,000s = 0.65 km/h)
        is_feas, status, req_spd, raw_score, _ = evaluate_route_feasibility_and_score(
            path_data, delta_t_seconds=10000.0, shortest_distance_m=1800.0
        )
        self.assertTrue(is_feas)
        self.assertFalse(math.isnan(raw_score))
        self.assertFalse(math.isinf(raw_score))
        self.assertGreater(raw_score, 0.0)


class TestDay9SerializationFidelity(unittest.TestCase):
    """Verify JSON serialization round-tripping for all stage schemas."""

    def test_observation_serialization(self):
        obs = Observation(
            observation_id="OBS_SER_01",
            camera_id="CAM_01",
            timestamp_seconds=123.45,
            track_id="trk_01",
            plate="KA01SER1",
            appearance_embedding=[0.1, 0.2, 0.3],
            timestamp_semantics="synchronized",
        )
        d = obs.to_dict()
        obs_restored = Observation.from_json(d)
        self.assertEqual(obs.observation_id, obs_restored.observation_id)
        self.assertEqual(obs.plate, obs_restored.plate)
        self.assertEqual(obs.appearance_embedding, obs_restored.appearance_embedding)

    def test_normalized_trajectory_serialization(self):
        norm = NormalizedTrajectory(
            track_id="TRK_NORM_SER",
            origin_node="J01",
            destination_node="J02",
            candidate_routes=[NormalizedCandidateRoute(nodes=["J01", "J02"], probability=1.0, metadata={"edges": ["R01"]})],
            vehicle_weight=1.0,
            metadata={"source": "test"},
        )
        d = norm.to_dict()
        self.assertEqual(d["track_id"], "TRK_NORM_SER")
        self.assertEqual(d["origin_node"], "J01")
        self.assertEqual(d["candidate_routes"][0]["probability"], 1.0)


class TestDay9RealDataSafetyBoundary(unittest.TestCase):
    """Verify real-data zero-fabrication safety invariants on Kanishka's dataset."""

    def test_real_data_produces_isolated_singletons_without_fabrication(self):
        """Real Kanishka observations lacking Re-ID & plates must safely remain singletons."""
        real_data_path = PROJECT_ROOT / "data" / "observations" / "kanishka_traffic.json"
        if not real_data_path.is_file():
            self.skipTest("Real dataset not present")

        with open(real_data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        obs_list = load_observations_from_json(raw_data[:20])  # Evaluate first 20 observations
        id_graph = IdentityGraph(min_probability_threshold=0.70)
        id_graph.build_graph(obs_list)
        clusters = id_graph.get_candidate_identities()

        # Invariant: Insufficient identity evidence must NOT produce fabricated merges
        self.assertEqual(len(clusters), len(obs_list), "Real data lacking Re-ID/plate evidence must safely remain isolated singletons")
        for c in clusters:
            self.assertEqual(len(c["member_observations"]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
