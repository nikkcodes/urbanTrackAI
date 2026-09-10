"""
UrbanTrack AI — Day 10 Final System Hardening & Hackathon Readiness Test Suite.

Validates the full system hardening objectives:
1. Complete demo scenario execution and validity
2. End-to-end evidence traceability
3. Anomaly explainability integrity (strictly non-moralistic/non-criminal)
4. Robustness catalog (20 categories A through T):
   A: Missing plate
   B: Missing Re-ID embedding
   C: Missing GPS coordinates
   D: Missing camera reliability
   E: Unsynchronized timestamps
   F: Standalone clock offset without shared reference
   G: Invalid timestamp
   H: Negative temporal difference
   I: Physically impossible speed
   J: Invalid road ID
   K: Closed road
   L: No feasible alternative route
   M: Zero road capacity (safe division by zero)
   N: Missing anomaly baseline (insufficient evidence)
   O: Invalid trajectory
   P: Empty observation batch
   Q: Single observation
   R: Ambiguous route set
   S: Mixed-quality batch
   T: Low-reliability evidence
5. Physical demand semantics & conservation (reliability NEVER multiplied into demand)
6. Temporal safety cases 1 to 6
7. Identity safety (camera-local track_id != global identity_id)
8. Bidirectional serialization roundtrips (Observation, NormalizedTrajectory, AnomalyResult, CounterfactualReport)
9. Baseline immutability during simulation
10. Multi-run semantic determinism (within 1e-6 tolerance)
11. Multi-batch performance profiling
12. Real Kanishka data safety invariant (zero fabrication of cross-camera links)
"""

import copy
from datetime import datetime, timezone
import json
import math
import os
import sys
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.identity_fusion import match_observations
from inference.identity_graph import IdentityGraph
from inference.member3_adapter import adapt_vehicle_trajectory_to_normalized
from inference.observation_loader import load_observations_from_json
from inference.road_graph import RoadGraph, RoadEdge
from inference.sparse_engine import detect_observation_gaps, infer_sparse_identity_trajectory
from inference.temporal import check_temporal_comparability, temporal_feasibility
from inference.trajectory_engine import reconstruct_identity_trajectory, reconstruct_trajectory_segment
from mobility.flow_engine import MobilityFlowEngine
from anomaly.investigation_engine import InvestigationEngine
from simulation.counterfactual_engine import CounterfactualEngine
from schemas.anomaly_schema import (
    AnomalyEvidence,
    AnomalyResult,
    AnomalySeverity,
    DataQualityStatus,
    InvestigationPriority,
    MobilityBaseline,
    SignalCategory,
    SignalSeverity,
)
from schemas.normalized_trajectory_schema import (
    InvalidRouteError,
    NormalizedCandidateRoute,
    NormalizedTrajectory,
    ProbabilityValidationError,
)
from schemas.observation_schema import Observation
from schemas.scenario_schema import (
    CounterfactualReport,
    ScenarioDefinition,
    ScenarioStatus,
    ScenarioType,
)


class TestDay10FinalReadiness(unittest.TestCase):
    """
    Day 10 Final System Hardening Test Suite for UrbanTrack AI.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.network_path = PROJECT_ROOT / "data" / "synthetic" / "city_network.json"
        cls.scenario_path = PROJECT_ROOT / "data" / "synthetic" / "day9_end_to_end_scenarios.json"
        cls.real_feed_path = PROJECT_ROOT / "data" / "observations" / "kanishka_traffic.json"

        cls.road_graph = RoadGraph.from_json_file(cls.network_path)
        for i in range(1, 9):
            cls.road_graph.camera_associations[f"CAM_J{i:02d}"] = f"J{i:02d}"

        with open(cls.scenario_path, "r", encoding="utf-8") as f:
            cls.scenario_data = json.load(f)

    # =========================================================================
    # THEME 1: COMPLETE DEMO SCENARIO EXECUTION & VALIDITY
    # =========================================================================
    def test_demo_scenario_execution_and_validity(self):
        """Verify that the full 9-stage pipeline executes cleanly from demo data."""
        observations = load_observations_from_json(
            self.scenario_data["observations"], camera_metadata=self.scenario_data.get("cameras")
        )
        self.assertGreaterEqual(len(observations), 7)

        id_graph = IdentityGraph(min_probability_threshold=0.70)
        id_graph.build_graph(observations, camera_metadata=self.scenario_data.get("cameras"))
        candidate_identities = id_graph.get_candidate_identities()
        self.assertGreaterEqual(len(candidate_identities), 4)

        veh1_cluster = next((c for c in candidate_identities if len(c["member_observations"]) == 3), None)
        veh2_cluster = next((c for c in candidate_identities if len(c["member_observations"]) == 2), None)
        self.assertIsNotNone(veh1_cluster)
        self.assertIsNotNone(veh2_cluster)

        veh1_traj = reconstruct_identity_trajectory(veh1_cluster, self.road_graph)
        veh2_traj = infer_sparse_identity_trajectory(veh2_cluster, self.road_graph)

        norm1 = adapt_vehicle_trajectory_to_normalized(veh1_traj, vehicle_weight=1.0)
        norm2 = adapt_vehicle_trajectory_to_normalized(veh2_traj, vehicle_weight=1.0)

        flow_engine = MobilityFlowEngine(self.road_graph)
        road_metrics, od_matrix, flow_issues, records = flow_engine.aggregate_flows([norm1, norm2])
        self.assertEqual(len(flow_issues), 0)
        self.assertAlmostEqual(od_matrix.total_demand, 2.0, places=4)

        inv_engine = InvestigationEngine(road_graph=self.road_graph)
        rep = inv_engine.run_investigation([norm1, norm2])
        self.assertEqual(len(rep.vehicle_anomalies), 2)

        scenario = ScenarioDefinition.from_dict(self.scenario_data["counterfactual_scenario"])
        cf_engine = CounterfactualEngine()
        cf_rep = cf_engine.simulate_counterfactual([norm1, norm2], self.road_graph, scenario)
        self.assertEqual(cf_rep.status, ScenarioStatus.SUCCESS)

    # =========================================================================
    # THEME 2: END-TO-END TRACEABILITY
    # =========================================================================
    def test_end_to_end_traceability(self):
        """Verify strict chain of evidence from Observation IDs to counterfactual impact."""
        observations = load_observations_from_json(
            self.scenario_data["observations"], camera_metadata=self.scenario_data.get("cameras")
        )
        obs_map = {o.observation_id: o for o in observations}
        self.assertIn("OBS_VEH1_CAM1", obs_map)
        self.assertIn("OBS_VEH1_CAM2", obs_map)
        self.assertIn("OBS_VEH1_CAM3", obs_map)

        # Track IDs are camera-local
        trk1 = obs_map["OBS_VEH1_CAM1"].track_id
        trk2 = obs_map["OBS_VEH1_CAM2"].track_id
        self.assertNotEqual(trk1, trk2)

        # Pairwise evidence fusion produces high probability
        match_res = match_observations(obs_map["OBS_VEH1_CAM1"], obs_map["OBS_VEH1_CAM2"], camera_metadata=self.scenario_data.get("cameras"))
        self.assertGreater(match_res["same_vehicle_probability"], 0.90)

        # Global identity cluster
        id_graph = IdentityGraph(min_probability_threshold=0.70)
        id_graph.build_graph(observations, camera_metadata=self.scenario_data.get("cameras"))
        veh1_cluster = next(c for c in id_graph.get_candidate_identities() if len(c["member_observations"]) == 3)
        global_id = veh1_cluster["identity_id"]
        self.assertTrue(global_id.startswith("VEHICLE_CANDIDATE_"))

        # Trajectory reflects global identity
        traj = reconstruct_identity_trajectory(veh1_cluster, self.road_graph)
        self.assertEqual(traj.identity_id, global_id)

        # NormalizedTrajectory preserves global ID and demand weight
        norm = adapt_vehicle_trajectory_to_normalized(traj, vehicle_weight=1.0)
        self.assertEqual(norm.track_id, global_id)
        self.assertEqual(norm.vehicle_weight, 1.0)

    # =========================================================================
    # THEME 3: EXPLAINABILITY INTEGRITY (NON-MORALISTIC LANGUAGE)
    # =========================================================================
    def test_anomaly_explainability_non_moralistic(self):
        """Verify anomaly explanations use objective behavioral/physical terms, never moralistic/criminal words."""
        forbidden_words = ["guilt", "criminal", "stolen", "malicious", "offender", "suspicious person", "felon", "culprit"]

        # Test compliant vehicle
        norm = NormalizedTrajectory(
            track_id="COMPLIANT_001",
            origin_node="J01",
            destination_node="J03",
            candidate_routes=[NormalizedCandidateRoute(nodes=["J01", "J02", "J03"], probability=1.0)],
            vehicle_weight=1.0,
        )
        engine = InvestigationEngine(road_graph=self.road_graph)
        res = engine.investigate_trajectory(norm)
        for word in forbidden_words:
            self.assertNotIn(word, res.explanation.lower(), f"Forbidden word '{word}' found in explanation")

        # Test with baseline speed inconsistency
        base = MobilityBaseline(source="configured", expected_travel_times={("J01", "J03"): 300.0})
        engine_base = InvestigationEngine(baseline=base, road_graph=self.road_graph)
        res_fast = engine_base.investigate_trajectory(norm)
        for word in forbidden_words:
            self.assertNotIn(word, res_fast.explanation.lower(), f"Forbidden word '{word}' found in explanation")

    # =========================================================================
    # THEME 4: ROBUSTNESS CATALOG (CATEGORIES A THROUGH T)
    # =========================================================================
    def test_robustness_cat_a_missing_plate(self):
        """Cat A: Missing license plate remains unavailable, not assumed matching or mismatching."""
        obs_a = Observation(camera_id="CAM_J01", track_id="t1", frame_id=1, timestamp_seconds=100.0, vehicle_type="car", plate=None, appearance_embedding=[0.5, 0.5])
        obs_b = Observation(camera_id="CAM_J02", track_id="t2", frame_id=2, timestamp_seconds=140.0, vehicle_type="car", plate=None, appearance_embedding=[0.5, 0.5])
        res = match_observations(obs_a, obs_b, camera_metadata=self.scenario_data.get("cameras"))
        self.assertIsNone(res["evidence"]["plate_similarity"])
        self.assertIn("Appearance similarity", res["explanation"])

    def test_robustness_cat_b_missing_reid(self):
        """Cat B: Missing Re-ID embedding remains missing/unavailable, never fabricated."""
        obs_a = Observation(camera_id="CAM_J01", track_id="t1", frame_id=1, timestamp_seconds=100.0, vehicle_type="car", appearance_embedding=None, plate="KA01AB1234")
        obs_b = Observation(camera_id="CAM_J02", track_id="t2", frame_id=2, timestamp_seconds=140.0, vehicle_type="car", appearance_embedding=None, plate="KA01AB1234")
        res = match_observations(obs_a, obs_b, camera_metadata=self.scenario_data.get("cameras"))
        self.assertIsNone(res["evidence"]["appearance_similarity"])
        self.assertEqual(res["evidence"]["appearance_status"], "missing")

    def test_robustness_cat_c_missing_gps(self):
        """Cat C: Missing GPS coordinates handled safely without crashing."""
        obs_a = Observation(camera_id="CAM_UNKNOWN_1", track_id="t1", frame_id=1, timestamp_seconds=100.0, vehicle_type="car", latitude=None, longitude=None)
        obs_b = Observation(camera_id="CAM_UNKNOWN_2", track_id="t2", frame_id=2, timestamp_seconds=140.0, vehicle_type="car", latitude=None, longitude=None)
        res = match_observations(obs_a, obs_b)
        self.assertIsNotNone(res)
        self.assertEqual(res["evidence"]["geographic_distance_meters"], 0.0)

    def test_robustness_cat_d_missing_camera_reliability(self):
        """Cat D: Missing camera reliability falls back gracefully without crashing."""
        obs = Observation(camera_id="CAM_J01", track_id="t1", frame_id=1, timestamp_seconds=100.0, vehicle_type="car", camera_reliability=None)
        from inference.reliability_engine import evaluate_observation_reliability
        rel = evaluate_observation_reliability(obs)
        self.assertIsNotNone(rel)
        self.assertGreater(rel.overall_reliability, 0.0)

    def test_robustness_cat_e_unsynchronized_timestamps(self):
        """Cat E: Independent unsynchronized timestamps yield unavailable comparability."""
        obs_a = Observation(camera_id="CAM_J01", track_id="t1", frame_id=1, timestamp_seconds=10.0, timestamp_semantics="unsynchronized")
        obs_b = Observation(camera_id="CAM_J02", track_id="t2", frame_id=2, timestamp_seconds=20.0, timestamp_semantics="unsynchronized")
        t_comp = check_temporal_comparability(obs_a, obs_b)
        self.assertFalse(t_comp["comparable"])
        self.assertEqual(t_comp["status"], "unavailable")

    def test_robustness_cat_f_standalone_clock_offset_without_shared_reference(self):
        """Cat F: Standalone clock offset without shared reference does NOT establish comparability."""
        obs_a = Observation(camera_id="CAM_J01", track_id="t1", frame_id=1, timestamp_seconds=10.0, clock_offset_seconds=5.0, timestamp_semantics="video_relative", time_reference_id=None)
        obs_b = Observation(camera_id="CAM_J02", track_id="t2", frame_id=2, timestamp_seconds=20.0, clock_offset_seconds=5.0, timestamp_semantics="video_relative", time_reference_id=None)
        t_comp = check_temporal_comparability(obs_a, obs_b)
        self.assertFalse(t_comp["comparable"])
        self.assertEqual(t_comp["status"], "unavailable")
        self.assertIn("without", t_comp["reason"])

    def test_robustness_cat_g_invalid_timestamp(self):
        """Cat G: Invalid timestamp string raises explicit ValueError rather than unhandled crash."""
        bad_data = {
            "camera_id": "CAM_J01",
            "timestamp": "invalid_date_format_xyz",
        }
        with self.assertRaises(ValueError):
            Observation.from_dict(bad_data)

    def test_robustness_cat_h_negative_temporal_difference(self):
        """Cat H: Negative temporal difference (tb < ta) rejected as temporal inversion."""
        obs_a = Observation(camera_id="CAM_J01", track_id="t1", frame_id=1, timestamp_seconds=200.0, timestamp_semantics="synchronized", time_reference_id="sync")
        obs_b = Observation(camera_id="CAM_J02", track_id="t2", frame_id=2, timestamp_seconds=100.0, timestamp_semantics="synchronized", time_reference_id="sync")
        t_comp = check_temporal_comparability(obs_a, obs_b)
        self.assertFalse(t_comp["comparable"])
        self.assertEqual(t_comp["status"], "invalid_negative_time")

    def test_robustness_cat_i_physically_impossible_speed(self):
        """Cat I: Required travel speed > 120 km/h rejected as physically impossible."""
        obs_a = Observation(camera_id="CAM_J01", track_id="t1", frame_id=1, timestamp_seconds=100.0, latitude=28.64, longitude=77.20, timestamp_semantics="synchronized", time_reference_id="sync")
        obs_b = Observation(camera_id="CAM_J02", track_id="t2", frame_id=2, timestamp_seconds=101.0, latitude=28.64, longitude=77.25, timestamp_semantics="synchronized", time_reference_id="sync")
        res = match_observations(obs_a, obs_b)
        self.assertEqual(res["same_vehicle_probability"], 0.0)
        self.assertIn("Physically impossible", res["explanation"])

    def test_robustness_cat_j_invalid_road_id(self):
        """Cat J: Invalid road ID in scenario definition rejected during scenario validation."""
        scenario = ScenarioDefinition(
            scenario_id="SCN_INVALID_ROAD",
            scenario_type=ScenarioType.ROAD_CLOSURE,
            description="Testing nonexistent road rejection",
            affected_roads=["NON_EXISTENT_ROAD_999"],
        )
        is_valid, err = scenario.validate(self.road_graph)
        self.assertFalse(is_valid)
        self.assertIn("NON_EXISTENT_ROAD_999", err)

    def test_robustness_cat_k_closed_road(self):
        """Cat K: Closed road is avoided during path discovery."""
        rg = copy.deepcopy(self.road_graph)
        rg.close_road("R01")
        paths = rg.find_candidate_paths("J01", "J02", max_paths=3)
        for p in paths:
            self.assertNotIn("R01", p["edges"])

    def test_robustness_cat_l_no_feasible_alternative_route(self):
        """Cat L: No feasible alternative route reports explicit unroutable demand without fabrication."""
        norm = NormalizedTrajectory(
            track_id="TRK_ISOLATED",
            origin_node="J01",
            destination_node="J04",
            candidate_routes=[NormalizedCandidateRoute(nodes=["J01", "J04"], probability=1.0, metadata={"edges": ["R07"]})],
            vehicle_weight=1.0,
        )
        rg = copy.deepcopy(self.road_graph)
        # Close all egress roads from J01: R01, R07, R14
        rg.close_road("R01")
        rg.close_road("R07")
        rg.close_road("R14")

        scenario = ScenarioDefinition(
            scenario_id="SCN_CUTOFF",
            scenario_type=ScenarioType.ROAD_CLOSURE,
            description="Isolate J01",
            affected_roads=["R07"],
        )
        cf_engine = CounterfactualEngine()
        rep = cf_engine.simulate_counterfactual([norm], rg, scenario)
        self.assertGreater(rep.impact["unroutable_demand"], 0.0)

    def test_robustness_cat_m_zero_road_capacity(self):
        """Cat M: Zero road capacity handled safely without division by zero."""
        rg = copy.deepcopy(self.road_graph)
        rg.edges["R01"].capacity_vph = 0.0
        norm = NormalizedTrajectory(
            track_id="TRK_01",
            origin_node="J01",
            destination_node="J02",
            candidate_routes=[NormalizedCandidateRoute(nodes=["J01", "J02"], probability=1.0, metadata={"edges": ["R01"]})],
            vehicle_weight=1.0,
        )
        flow_engine = MobilityFlowEngine(rg)
        road_metrics, _, _, _ = flow_engine.aggregate_flows([norm], time_window={"duration_seconds": 3600.0, "start": 0.0, "end": 3600.0})
        m01 = next(m for m in road_metrics if m.road_id == "R01")
        self.assertEqual(m01.capacity_vph, 0.0)
        self.assertFalse(math.isinf(m01.utilization_ratio))
        self.assertFalse(math.isnan(m01.utilization_ratio))
        self.assertEqual(m01.utilization_ratio, 0.0)  # Safe zero rather than div by zero or Inf

    def test_robustness_cat_n_missing_anomaly_baseline(self):
        """Cat N: Missing anomaly baseline results in INSUFFICIENT_EVIDENCE, not an anomaly."""
        norm = NormalizedTrajectory(
            track_id="TRK_NO_BASE",
            origin_node="J01",
            destination_node="J03",
            candidate_routes=[NormalizedCandidateRoute(nodes=["J01", "J02", "J03"], probability=1.0)],
            vehicle_weight=1.0,
        )
        inv = InvestigationEngine(baseline=MobilityBaseline(source="unavailable"), road_graph=self.road_graph)
        res = inv.investigate_trajectory(norm)
        self.assertEqual(res.severity, AnomalySeverity.INSUFFICIENT_EVIDENCE)
        self.assertFalse(res.is_anomalous)

    def test_robustness_cat_o_invalid_trajectory(self):
        """Cat O: Invalid trajectory (e.g. probability sum mismatch) raises explicit validation error."""
        with self.assertRaises(ProbabilityValidationError):
            NormalizedTrajectory(
                track_id="TRK_BAD_PROB",
                origin_node="J01",
                destination_node="J02",
                candidate_routes=[NormalizedCandidateRoute(nodes=["J01", "J02"], probability=0.5)],
                vehicle_weight=1.0,
            )

    def test_robustness_cat_p_empty_observation_batch(self):
        """Cat P: Empty observation batch handled safely."""
        graph = IdentityGraph()
        graph.build_graph([])
        self.assertEqual(len(graph.get_candidate_identities()), 0)

        flow_engine = MobilityFlowEngine(self.road_graph)
        road_metrics, od_matrix, issues, _ = flow_engine.aggregate_flows([])
        self.assertEqual(od_matrix.total_demand, 0.0)

    def test_robustness_cat_q_single_observation(self):
        """Cat Q: Single observation yields exactly 1 singleton identity."""
        obs = Observation(camera_id="CAM_J01", track_id="t1", frame_id=1, timestamp_seconds=100.0, vehicle_type="car")
        graph = IdentityGraph()
        graph.build_graph([obs])
        candidates = graph.get_candidate_identities()
        self.assertEqual(len(candidates), 1)
        self.assertEqual(len(candidates[0]["member_observations"]), 1)

    def test_robustness_cat_r_ambiguous_route_set(self):
        """Cat R: Ambiguous route set preserves uncertainty with is_ambiguous=True."""
        # Create segment where two paths have nearly identical score
        obs_a = Observation(camera_id="CAM_J01", track_id="t1", frame_id=1, timestamp_seconds=1000.0, vehicle_type="car", latitude=28.64, longitude=77.20, timestamp_semantics="synchronized", time_reference_id="sync")
        obs_b = Observation(camera_id="CAM_J05", track_id="t2", frame_id=2, timestamp_seconds=1256.0, vehicle_type="car", latitude=28.62, longitude=77.218, timestamp_semantics="synchronized", time_reference_id="sync")
        seg = reconstruct_trajectory_segment(obs_a, obs_b, self.road_graph, config={"ambiguity_threshold": 0.15})
        self.assertTrue(seg.is_ambiguous)
        self.assertIsNotNone(seg.ambiguity_reason)

    def test_robustness_cat_s_mixed_quality_batch(self):
        """Cat S: Mixed-quality batch processes valid pairs and isolates invalid ones safely."""
        obs_valid1 = Observation(camera_id="CAM_J01", track_id="t1", frame_id=1, timestamp_seconds=1000.0, vehicle_type="car", plate="KA01AB1234", appearance_embedding=[0.8, 0.4], timestamp_semantics="synchronized", time_reference_id="sync")
        obs_valid2 = Observation(camera_id="CAM_J02", track_id="t2", frame_id=2, timestamp_seconds=1130.0, vehicle_type="car", plate="KA01AB1234", appearance_embedding=[0.8, 0.4], timestamp_semantics="synchronized", time_reference_id="sync")
        obs_broken = Observation(camera_id="CAM_J03", track_id="t3", frame_id=3, timestamp_seconds=900.0, vehicle_type="car", plate=None, appearance_embedding=None, timestamp_semantics="unsynchronized")

        graph = IdentityGraph(min_probability_threshold=0.70)
        graph.build_graph([obs_valid1, obs_valid2, obs_broken])
        clusters = graph.get_candidate_identities()
        multi = [c for c in clusters if len(c["member_observations"]) == 2]
        single = [c for c in clusters if len(c["member_observations"]) == 1]
        self.assertEqual(len(multi), 1)
        self.assertEqual(len(single), 1)

    def test_robustness_cat_t_low_reliability_evidence(self):
        """Cat T: Low reliability evidence elevates uncertainty but is NEVER multiplied into demand."""
        obs_low = Observation(camera_id="CAM_J01", track_id="t1", frame_id=1, timestamp_seconds=1000.0, vehicle_type="car", camera_reliability=0.15)
        from inference.reliability_engine import evaluate_observation_reliability
        rel = evaluate_observation_reliability(obs_low)
        self.assertLess(rel.overall_reliability, 0.35)

        norm = NormalizedTrajectory(
            track_id="TRK_LOW_REL",
            origin_node="J01",
            destination_node="J02",
            candidate_routes=[NormalizedCandidateRoute(nodes=["J01", "J02"], probability=1.0, metadata={"edges": ["R01"]})],
            vehicle_weight=1.0,
            metadata={"reliability": rel.overall_reliability},
        )
        flow_engine = MobilityFlowEngine(self.road_graph)
        road_metrics, od_matrix, _, _ = flow_engine.aggregate_flows([norm])
        m01 = next(m for m in road_metrics if m.road_id == "R01")
        # Physical demand must be exactly 1.0 PCU, not 0.15!
        self.assertAlmostEqual(m01.expected_demand_in_window, 1.0, places=4)

    # =========================================================================
    # THEME 5: DEMAND SEMANTICS & CONSERVATION
    # =========================================================================
    def test_demand_semantics_and_conservation(self):
        """Verify demand = vehicle_weight * route_allocation, never multiplied by reliability, and conserved."""
        norm = NormalizedTrajectory(
            track_id="TRK_DEMAND_TEST",
            origin_node="J01",
            destination_node="J05",
            candidate_routes=[
                NormalizedCandidateRoute(nodes=["J01", "J02", "J05"], probability=0.60, metadata={"edges": ["R01", "R08"]}),
                NormalizedCandidateRoute(nodes=["J01", "J04", "J05"], probability=0.40, metadata={"edges": ["R07", "R18"]}),
            ],
            vehicle_weight=2.5,  # Heavy vehicle / bus PCU
        )
        demands = norm.calculate_route_demands()
        self.assertAlmostEqual(demands[0]["route_demand"], 2.5 * 0.60, places=4)
        self.assertAlmostEqual(demands[1]["route_demand"], 2.5 * 0.40, places=4)
        self.assertAlmostEqual(sum(d["route_demand"] for d in demands), 2.5, places=4)

    # =========================================================================
    # THEME 6: TEMPORAL SAFETY (CASES 1 TO 6)
    # =========================================================================
    def test_temporal_safety_cases_1_to_6(self):
        """Explicitly test all 6 temporal comparability cases."""
        # Case 1: Same camera video-relative timestamps -> comparable
        obs1 = Observation(camera_id="CAM_J01", track_id="t1", frame_id=1, timestamp_seconds=10.0, timestamp_semantics="video_relative")
        obs2 = Observation(camera_id="CAM_J01", track_id="t1", frame_id=2, timestamp_seconds=25.0, timestamp_semantics="video_relative")
        c1 = check_temporal_comparability(obs1, obs2)
        self.assertTrue(c1["comparable"])
        self.assertEqual(c1["delta_seconds"], 15.0)

        # Case 2: Cross-camera synchronized timestamps with shared reference -> comparable
        obs3 = Observation(camera_id="CAM_J01", track_id="t1", frame_id=1, timestamp_seconds=100.0, timestamp_semantics="synchronized", time_reference_id="metro_gps")
        obs4 = Observation(camera_id="CAM_J02", track_id="t2", frame_id=2, timestamp_seconds=140.0, timestamp_semantics="synchronized", time_reference_id="metro_gps")
        c2 = check_temporal_comparability(obs3, obs4)
        self.assertTrue(c2["comparable"])
        self.assertEqual(c2["delta_seconds"], 40.0)

        # Case 3: Cross-camera video-relative timestamps without shared reference -> unavailable
        obs5 = Observation(camera_id="CAM_J01", track_id="t1", frame_id=1, timestamp_seconds=100.0, timestamp_semantics="video_relative", time_reference_id=None)
        obs6 = Observation(camera_id="CAM_J02", track_id="t2", frame_id=2, timestamp_seconds=140.0, timestamp_semantics="video_relative", time_reference_id=None)
        c3 = check_temporal_comparability(obs5, obs6)
        self.assertFalse(c3["comparable"])

        # Case 4: Standalone clock_offset_seconds without shared reference -> unavailable
        obs7 = Observation(camera_id="CAM_J01", track_id="t1", frame_id=1, timestamp_seconds=100.0, clock_offset_seconds=5.0, timestamp_semantics="video_relative", time_reference_id=None)
        obs8 = Observation(camera_id="CAM_J02", track_id="t2", frame_id=2, timestamp_seconds=140.0, clock_offset_seconds=10.0, timestamp_semantics="video_relative", time_reference_id=None)
        c4 = check_temporal_comparability(obs7, obs8)
        self.assertFalse(c4["comparable"])

        # Case 5: Mismatched time references -> unavailable
        obs9 = Observation(camera_id="CAM_J01", track_id="t1", frame_id=1, timestamp_seconds=100.0, timestamp_semantics="synchronized", time_reference_id="metro_gps")
        obs10 = Observation(camera_id="CAM_J02", track_id="t2", frame_id=2, timestamp_seconds=140.0, timestamp_semantics="synchronized", time_reference_id="local_ntp")
        c5 = check_temporal_comparability(obs9, obs10)
        self.assertFalse(c5["comparable"])

        # Case 6: Negative resulting time difference -> invalid_negative_time
        obs11 = Observation(camera_id="CAM_J01", track_id="t1", frame_id=1, timestamp_seconds=200.0, timestamp_semantics="synchronized", time_reference_id="metro_gps")
        obs12 = Observation(camera_id="CAM_J02", track_id="t2", frame_id=2, timestamp_seconds=100.0, timestamp_semantics="synchronized", time_reference_id="metro_gps")
        c6 = check_temporal_comparability(obs11, obs12)
        self.assertFalse(c6["comparable"])
        self.assertEqual(c6["status"], "invalid_negative_time")

    # =========================================================================
    # THEME 7: IDENTITY SAFETY
    # =========================================================================
    def test_identity_safety_local_vs_global(self):
        """Verify camera-local track_id != global identity_id and impossible speed prevents merge."""
        # Vehicle with identical plate but impossible speed across cameras
        obs_a = Observation(camera_id="CAM_J01", track_id="trk_c1", frame_id=1, timestamp_seconds=1000.0, vehicle_type="car", plate="MH12XY9999", latitude=28.64, longitude=77.20, timestamp_semantics="synchronized", time_reference_id="sync")
        obs_b = Observation(camera_id="CAM_J02", track_id="trk_c2", frame_id=2, timestamp_seconds=1005.0, vehicle_type="car", plate="MH12XY9999", latitude=28.635, longitude=77.21, timestamp_semantics="synchronized", time_reference_id="sync")
        res = match_observations(obs_a, obs_b)
        self.assertEqual(res["same_vehicle_probability"], 0.0)

        graph = IdentityGraph(min_probability_threshold=0.70)
        graph.build_graph([obs_a, obs_b])
        candidates = graph.get_candidate_identities()
        # Must produce 2 separate singletons, NEVER merge
        self.assertEqual(len(candidates), 2)

    # =========================================================================
    # THEME 8: BIDIRECTIONAL SERIALIZATION ROUNDTRIPS
    # =========================================================================
    def test_serialization_roundtrip_observation(self):
        """Verify Observation lossless roundtrip serialization."""
        obs = Observation(
            observation_id="OBS_TEST_01",
            camera_id="CAM_J01",
            track_id="trk_01",
            frame_id=10,
            timestamp_seconds=1050.5,
            timestamp=datetime(2026, 9, 10, 10, 17, 30, tzinfo=timezone.utc),
            vehicle_type="car",
            detection_confidence=0.95,
            bbox=[100.0, 150.0, 250.0, 300.0],
            latitude=28.64,
            longitude=77.20,
            plate="KA01AB1234",
            plate_confidence=0.98,
            appearance_embedding=[0.1, 0.2, 0.3],
            camera_reliability=0.92,
            time_reference_id="metro_gps",
            timestamp_semantics="synchronized",
        )
        data = obs.to_dict()
        restored = Observation.from_dict(data)
        self.assertEqual(obs.observation_id, restored.observation_id)
        self.assertEqual(obs.plate, restored.plate)
        self.assertEqual(obs.appearance_embedding, restored.appearance_embedding)
        self.assertEqual(obs.timestamp_semantics, restored.timestamp_semantics)

    def test_serialization_roundtrip_normalized_trajectory(self):
        """Verify NormalizedTrajectory lossless roundtrip serialization."""
        route = NormalizedCandidateRoute(nodes=["J01", "J02", "J03"], probability=1.0, metadata={"distance_m": 4000.0})
        traj = NormalizedTrajectory(
            track_id="VEH_001",
            origin_node="J01",
            destination_node="J03",
            candidate_routes=[route],
            vehicle_weight=1.0,
            time_window_start=1000.0,
            time_window_end=1262.0,
            metadata={"overall_confidence": 0.99},
        )
        data = traj.to_dict()
        restored = NormalizedTrajectory.from_dict(data)
        self.assertEqual(traj.track_id, restored.track_id)
        self.assertEqual(traj.origin_node, restored.origin_node)
        self.assertEqual(len(restored.candidate_routes), 1)
        self.assertAlmostEqual(restored.candidate_routes[0].probability, 1.0, places=6)

    def test_serialization_roundtrip_anomaly_result(self):
        """Verify AnomalyResult lossless roundtrip serialization."""
        evidence = AnomalyEvidence(
            signal_type="speed_inconsistency",
            signal_category=SignalCategory.PHYSICAL_INCONSISTENCY,
            measured_value=135.0,
            threshold=120.0,
            signal_score=0.85,
            severity=SignalSeverity.ELEVATED,
            available=True,
            explanation="Speed exceeds 120 km/h limit",
        )
        result = AnomalyResult(
            anomaly_id="ANOM_001",
            entity_type="vehicle",
            entity_id="VEH_001",
            data_quality_status=DataQualityStatus.VALID,
            anomaly_types=["speed_inconsistency"],
            overall_score=0.85,
            severity=AnomalySeverity.INVESTIGATE,
            evidence=[evidence],
            reliability=0.90,
            uncertainty=0.10,
            investigation_priority=InvestigationPriority.HIGH_PRIORITY,
            explanation="Vehicle exceeded physical road network speed limit.",
        )
        data = result.to_dict()
        restored = AnomalyResult.from_dict(data)
        self.assertEqual(result.anomaly_id, restored.anomaly_id)
        self.assertEqual(result.severity, restored.severity)
        self.assertEqual(result.investigation_priority, restored.investigation_priority)
        self.assertEqual(len(restored.evidence), 1)
        self.assertEqual(restored.evidence[0].signal_category, SignalCategory.PHYSICAL_INCONSISTENCY)

    def test_serialization_roundtrip_counterfactual_report(self):
        """Verify CounterfactualReport lossless roundtrip serialization."""
        report = CounterfactualReport(
            scenario_id="SCN_TEST_01",
            scenario_type="ROAD_CLOSURE",
            status=ScenarioStatus.SUCCESS,
            baseline={"total_demand": 2.0},
            counterfactual={"total_demand": 2.0},
            impact={"displaced_demand": 0.45, "unroutable_demand": 0.0},
            assumptions=["Hypothetical closure of R07"],
            uncertainties=["Demand reallocates via J02"],
        )
        data = report.to_dict()
        restored = CounterfactualReport.from_dict(data)
        self.assertEqual(report.scenario_id, restored.scenario_id)
        self.assertEqual(report.status, restored.status)
        self.assertAlmostEqual(restored.impact["displaced_demand"], 0.45, places=4)

    # =========================================================================
    # THEME 9: BASELINE IMMUTABILITY
    # =========================================================================
    def test_baseline_immutability_during_simulation(self):
        """Verify RoadGraph and NormalizedTrajectory remain strictly unmutated after simulation."""
        norm = NormalizedTrajectory(
            track_id="TRK_IMMUTABLE",
            origin_node="J01",
            destination_node="J05",
            candidate_routes=[
                NormalizedCandidateRoute(nodes=["J01", "J02", "J05"], probability=0.5503, metadata={"edges": ["R01", "R08"]}),
                NormalizedCandidateRoute(nodes=["J01", "J04", "J05"], probability=0.4497, metadata={"edges": ["R07", "R18"]}),
            ],
            vehicle_weight=1.0,
        )
        edges_before = {rid: copy.deepcopy(e.to_dict()) for rid, e in self.road_graph.edges.items()}
        probs_before = [r.probability for r in norm.candidate_routes]

        scenario = ScenarioDefinition.from_dict(self.scenario_data["counterfactual_scenario"])
        cf_engine = CounterfactualEngine()
        cf_rep = cf_engine.simulate_counterfactual([norm], self.road_graph, scenario)

        edges_after = {rid: e.to_dict() for rid, e in self.road_graph.edges.items()}
        probs_after = [r.probability for r in norm.candidate_routes]

        self.assertEqual(edges_before, edges_after, "RoadGraph edges were mutated!")
        self.assertEqual(probs_before, probs_after, "Baseline candidate route probabilities were mutated!")

    # =========================================================================
    # THEME 10: MULTI-RUN SEMANTIC DETERMINISM
    # =========================================================================
    def test_multi_run_semantic_determinism(self):
        """Verify repeated pipeline executions produce semantically identical results within tested numerical tolerance (1e-6)."""
        observations = load_observations_from_json(
            self.scenario_data["observations"], camera_metadata=self.scenario_data.get("cameras")
        )

        results = []
        for run_idx in range(5):
            id_graph = IdentityGraph(min_probability_threshold=0.70)
            id_graph.build_graph(observations, camera_metadata=self.scenario_data.get("cameras"))
            candidates = id_graph.get_candidate_identities()
            veh1 = next(c for c in candidates if len(c["member_observations"]) == 3)
            traj1 = reconstruct_identity_trajectory(veh1, self.road_graph)
            results.append({
                "cluster_count": len(candidates),
                "veh1_confidence": veh1["identity_confidence"],
                "traj1_distance": traj1.total_distance_meters,
                "traj1_time": traj1.total_time_seconds,
            })

        for i in range(1, len(results)):
            self.assertEqual(results[i]["cluster_count"], results[0]["cluster_count"])
            self.assertAlmostEqual(results[i]["veh1_confidence"], results[0]["veh1_confidence"], places=6)
            self.assertAlmostEqual(results[i]["traj1_distance"], results[0]["traj1_distance"], places=6)
            self.assertAlmostEqual(results[i]["traj1_time"], results[0]["traj1_time"], places=6)

    # =========================================================================
    # THEME 11: MULTI-BATCH PERFORMANCE PROFILING
    # =========================================================================
    def test_performance_scaling_multiple_batch_sizes(self):
        """Profile pipeline performance across small, medium, and larger synthetic observation batches."""
        base_obs = load_observations_from_json(
            self.scenario_data["observations"], camera_metadata=self.scenario_data.get("cameras")
        )
        batch_sizes = [7, 14, 28]

        timings = {}
        for size in batch_sizes:
            # Replicate observations with shifted IDs to simulate scaling batches
            batch = []
            for mult in range(size // len(base_obs) + 1):
                for o in base_obs:
                    if len(batch) >= size:
                        break
                    o_dict = o.to_dict()
                    o_dict["observation_id"] = f"{o.observation_id}_m{mult}"
                    batch.append(Observation.from_dict(o_dict))

            t0 = time.perf_counter()
            id_graph = IdentityGraph(min_probability_threshold=0.70)
            id_graph.build_graph(batch, camera_metadata=self.scenario_data.get("cameras"))
            clusters = id_graph.get_candidate_identities()
            dur_ms = (time.perf_counter() - t0) * 1000.0
            timings[size] = dur_ms

        # Document honest timing: verify execution completes in sub-second time
        for size, t_ms in timings.items():
            self.assertLess(t_ms, 1000.0, f"Batch size {size} took {t_ms:.2f} ms")

    # =========================================================================
    # THEME 12: REAL DATA SAFETY INVARIANT
    # =========================================================================
    def test_real_data_safety_invariant(self):
        """Verify real Kanishka data invariant: missing fields remain None and unsupported links are never fabricated."""
        if not self.real_feed_path.is_file():
            self.skipTest("Real Kanishka data file not present.")

        with open(self.real_feed_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # Inspect first 50 observations
        sample_raw = raw[:50]
        camera_meta = {"traffic": {"latitude": 17.3850, "longitude": 78.4867}}
        observations = load_observations_from_json(sample_raw, camera_metadata=camera_meta)

        for obs in observations:
            self.assertIsNone(obs.appearance_embedding, "Re-ID embedding should NOT be fabricated")
            self.assertIsNone(obs.plate, "Plate should NOT be fabricated")

        graph = IdentityGraph(min_probability_threshold=0.70)
        graph.build_graph(observations, camera_metadata=camera_meta)
        clusters = graph.get_candidate_identities()

        # Invariant: unsupported cross-camera identities must NOT be fabricated
        # All members should remain singletons because no cross-camera links exist in the traffic feed
        for c in clusters:
            self.assertEqual(len(c["cameras_visited"]), 1, "Cross-camera identity was falsely fabricated on real feed!")


if __name__ == "__main__":
    unittest.main(verbosity=2)
