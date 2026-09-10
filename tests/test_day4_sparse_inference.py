"""
Test Suite for Day 4: Sparse / Missing-Camera Trajectory Inference.

Validates the Gap Model, candidate hidden route generation, temporal feasibility,
impossible route rejection, closed-road handling, one-way directional constraints,
uncertainty preservation, ambiguity surfacing, multi-observation gaps, strict
prevention of observation fabrication, and Member 3 NormalizedTrajectory compatibility.
"""

import json
import math
from pathlib import Path
import unittest

from inference.member3_adapter import (
    adapt_sparse_gap_to_normalized,
    adapt_trajectories_to_batch_payload,
    adapt_trajectory_segment_to_normalized,
    adapt_vehicle_trajectory_to_normalized,
)
from inference.road_graph import RoadGraph
from inference.sparse_engine import (
    detect_observation_gaps,
    infer_sparse_gap,
    infer_sparse_identity_trajectory,
)
from schemas.gap_schema import SparseObservationGap
from schemas.normalized_trajectory_schema import NormalizedTrajectory


class TestDay4SparseInference(unittest.TestCase):
    """Day 4 test suite verifying all 10 required cases and contract integrity."""

    def setUp(self):
        """Set up test environment using Member 3's shared city network."""
        self.network_path = Path("data/synthetic/city_network.json")
        self.assertTrue(self.network_path.is_file(), f"Missing city_network.json at {self.network_path}")
        self.graph = RoadGraph.from_json_file(self.network_path)

        # Map test cameras to Member 3 junctions
        self.graph.camera_associations["CAM_J01"] = "J01"
        self.graph.camera_associations["CAM_J02"] = "J02"
        self.graph.camera_associations["CAM_J03"] = "J03"
        self.graph.camera_associations["CAM_J04"] = "J04"
        self.graph.camera_associations["CAM_J05"] = "J05"
        self.graph.camera_associations["CAM_J06"] = "J06"
        self.graph.camera_associations["CAM_J07"] = "J07"
        self.graph.camera_associations["CAM_J08"] = "J08"
        self.graph.camera_associations["CAM_J10"] = "J10"
        self.graph.camera_associations["CAM_J11"] = "J11"
        self.graph.camera_associations["CAM_J12"] = "J12"

        # Load Day 4 test scenarios fixture
        fixture_path = Path("data/synthetic/day4_sparse_scenarios.json")
        self.assertTrue(fixture_path.is_file(), f"Missing day4_sparse_scenarios.json at {fixture_path}")
        with open(fixture_path, "r", encoding="utf-8") as f:
            self.fixture = json.load(f)

    # =========================================================================
    # 1. GAP MODEL & DETECTION
    # =========================================================================

    def test_01_gap_model_initialization_and_serialization(self):
        """Verify SparseObservationGap attributes and serialization."""
        obs_a = {"observation_id": "o1", "camera_id": "CAM_J01", "timestamp_seconds": 1000.0}
        obs_b = {"observation_id": "o2", "camera_id": "CAM_J08", "timestamp_seconds": 1450.0}

        gap = infer_sparse_gap(obs_a, obs_b, self.graph, identity_id="TRK_01")
        self.assertIsInstance(gap, SparseObservationGap)
        self.assertEqual(gap.gap_id, "gap_o1->o2")
        self.assertEqual(gap.identity_id, "TRK_01")
        self.assertEqual(gap.start_node_id, "J01")
        self.assertEqual(gap.end_node_id, "J08")
        self.assertEqual(gap.gap_duration_seconds, 450.0)
        self.assertTrue(gap.feasible)
        self.assertIn("J04", gap.unobserved_intermediate_nodes)

        # Check dictionary serialization
        gap_dict = gap.to_dict()
        self.assertEqual(gap_dict["gap_id"], "gap_o1->o2")
        self.assertEqual(gap_dict["gap_duration_seconds"], 450.0)
        self.assertIn("candidate_routes", gap_dict)
        self.assertIn("unobserved_intermediate_nodes", gap_dict)

    def test_02_detect_observation_gaps_classification(self):
        """Verify detect_observation_gaps distinguishes direct single-hop vs unobserved gaps."""
        # 3 observations: J01 -> J02 (direct R01) -> J08 (gap via unobserved J05)
        obs_seq = [
            {"observation_id": "o1", "camera_id": "CAM_J01", "timestamp_seconds": 1000.0},
            {"observation_id": "o2", "camera_id": "CAM_J02", "timestamp_seconds": 1150.0},
            {"observation_id": "o3", "camera_id": "CAM_J08", "timestamp_seconds": 1500.0},
        ]

        analysis = detect_observation_gaps(obs_seq, self.graph)
        self.assertEqual(len(analysis), 2)

        # Interval 1: J01 -> J02 (single road R01)
        self.assertFalse(analysis[0]["is_gap"])
        self.assertEqual(analysis[0]["classification"], "direct_adjacent_corridor")

        # Interval 2: J02 -> J08 (requires J05 intermediate junction)
        self.assertTrue(analysis[1]["is_gap"])
        self.assertEqual(analysis[1]["classification"], "missing_intermediate_observations")

    # =========================================================================
    # 2. REQUIRED 10 DAY-4 SCENARIOS (CASES 1 THROUGH 10)
    # =========================================================================

    def test_03_case_1_multiple_feasible_hidden_routes(self):
        """CASE 1: Two observations with multiple feasible hidden routes (J01 -> J08, 450s)."""
        obs_a = {"observation_id": "o1", "camera_id": "CAM_J01", "timestamp_seconds": 1000.0}
        obs_b = {"observation_id": "o2", "camera_id": "CAM_J08", "timestamp_seconds": 1450.0}

        gap = infer_sparse_gap(obs_a, obs_b, self.graph, identity_id="TRK_CASE_01", max_paths=5)
        self.assertEqual(gap.status, "success")
        self.assertTrue(gap.feasible)

        feasible_routes = [r for r in gap.candidate_routes if r.feasible]
        self.assertGreaterEqual(len(feasible_routes), 2)

        # Route probabilities must sum to 1.0
        prob_sum = sum(r.estimated_likelihood for r in feasible_routes)
        self.assertAlmostEqual(prob_sum, 1.0, places=4)

        # All candidate routes must strictly use Member 3 J and R identifiers
        for r in gap.candidate_routes:
            self.assertEqual(r.nodes[0], "J01")
            self.assertEqual(r.nodes[-1], "J08")
            for node in r.nodes:
                self.assertTrue(node.startswith("J"), f"Expected J junction, got {node}")
            for edge in r.edges:
                self.assertTrue(edge.startswith("R"), f"Expected R road, got {edge}")

        # Ambiguity must be surfaced
        self.assertTrue(gap.is_ambiguous)
        self.assertIsNotNone(gap.ambiguity_reason)

    def test_04_case_2_one_feasible_several_impossible_routes(self):
        """CASE 2: One feasible route and several impossible routes (J01 -> J08, 320s)."""
        # Distance to J08: Shortest route via J04 is 3900m (req speed: 43.9 km/h, limit 40 km/h -> within 1.25 tol)
        # Longer route via J05 is 5500m (req speed: 61.9 km/h, limit 45 km/h -> exceeds limit!)
        obs_a = {"observation_id": "o1", "camera_id": "CAM_J01", "timestamp_seconds": 1000.0}
        obs_b = {"observation_id": "o2", "camera_id": "CAM_J08", "timestamp_seconds": 1320.0}

        gap = infer_sparse_gap(obs_a, obs_b, self.graph, identity_id="TRK_CASE_02", max_paths=5)
        self.assertEqual(gap.status, "success")
        self.assertTrue(gap.feasible)

        feasible_routes = [r for r in gap.candidate_routes if r.feasible]
        infeasible_routes = [r for r in gap.candidate_routes if not r.feasible]

        self.assertEqual(len(feasible_routes), 1)
        self.assertGreaterEqual(len(infeasible_routes), 1)

        # The sole feasible route receives 100% likelihood
        self.assertAlmostEqual(feasible_routes[0].estimated_likelihood, 1.0, places=4)
        self.assertFalse(gap.is_ambiguous)

        # Infeasible routes must preserve explainable rejection reason
        for inf in infeasible_routes:
            self.assertIn("speed", inf.explanation.lower())

    def test_05_case_3_closed_road_removes_plausible_route(self):
        """CASE 3: A closed road (R11) removes an otherwise plausible route."""
        obs_a = {"observation_id": "o1", "camera_id": "CAM_J01", "timestamp_seconds": 1000.0}
        obs_b = {"observation_id": "o2", "camera_id": "CAM_J08", "timestamp_seconds": 1450.0}

        # Initially R11 is open
        gap_before = infer_sparse_gap(obs_a, obs_b, self.graph, max_paths=5)
        routes_with_r11 = [r for r in gap_before.candidate_routes if "R11" in r.edges]
        self.assertGreaterEqual(len(routes_with_r11), 1)

        # Dynamically close R11 (J04 -> J08)
        self.graph.close_road("R11")
        try:
            gap_after = infer_sparse_gap(obs_a, obs_b, self.graph, max_paths=5)
            self.assertEqual(gap_after.status, "success")
            for r in gap_after.candidate_routes:
                self.assertNotIn("R11", r.edges, "Closed road R11 must not appear in candidate routes")
        finally:
            self.graph.restore_road("R11")

    def test_06_case_4_one_way_restriction_removes_route(self):
        """CASE 4: A one-way restriction removes reverse traversal (J03 -> J02)."""
        # Road R02 is directed: J02 -> J03. No direct road exists from J03 -> J02.
        obs_a = {"observation_id": "o1", "camera_id": "CAM_J03", "timestamp_seconds": 1000.0}
        obs_b = {"observation_id": "o2", "camera_id": "CAM_J02", "timestamp_seconds": 1150.0}

        gap = infer_sparse_gap(obs_a, obs_b, self.graph, max_paths=5)
        # Verify no candidate path direct through R02 in reverse
        for r in gap.candidate_routes:
            self.assertNotIn("R02", r.edges)

    def test_07_case_5_short_time_window_eliminates_long_corridors(self):
        """CASE 5: Very short time window (350s) makes 5.5km routes impossible."""
        obs_a = {"observation_id": "o1", "camera_id": "CAM_J01", "timestamp_seconds": 1000.0}
        obs_b = {"observation_id": "o2", "camera_id": "CAM_J08", "timestamp_seconds": 1350.0}

        gap = infer_sparse_gap(obs_a, obs_b, self.graph, max_paths=5)
        self.assertEqual(gap.status, "success")

        # 3.9 km route requires 40.1 km/h (feasible)
        # 5.5 km route requires 56.6 km/h (impossible, speed limit 45 km/h)
        for r in gap.candidate_routes:
            if r.distance_meters > 5000:
                self.assertFalse(r.feasible)
                self.assertIn("speed", r.explanation.lower())

    def test_08_case_6_large_time_window_allows_multiple_routes(self):
        """CASE 6: Large time window (650s) allows multiple routes with uncertainty preserved."""
        obs_a = {"observation_id": "o1", "camera_id": "CAM_J01", "timestamp_seconds": 1000.0}
        obs_b = {"observation_id": "o2", "camera_id": "CAM_J08", "timestamp_seconds": 1650.0}

        gap = infer_sparse_gap(obs_a, obs_b, self.graph, max_paths=5)
        self.assertEqual(gap.status, "success")
        feasible = [r for r in gap.candidate_routes if r.feasible]
        self.assertGreaterEqual(len(feasible), 3)

        # Relative likelihoods sum to 1.0
        self.assertAlmostEqual(sum(r.estimated_likelihood for r in feasible), 1.0, places=4)

    def test_09_case_7_multi_observation_one_missing_intermediate_segment(self):
        """CASE 7: Multiple observations with one missing intermediate segment (J01 -> J02 -> [gap] -> J08)."""
        obs1 = {"observation_id": "o1", "camera_id": "CAM_J01", "timestamp_seconds": 1000.0}
        obs2 = {"observation_id": "o2", "camera_id": "CAM_J02", "timestamp_seconds": 1150.0}
        obs3 = {"observation_id": "o3", "camera_id": "CAM_J08", "timestamp_seconds": 1500.0}

        ident = {
            "identity_id": "TRK_CASE_07",
            "observations": [obs1, obs2, obs3],
        }

        traj = infer_sparse_identity_trajectory(ident, self.graph)
        self.assertEqual(traj.observations_count, 3)
        self.assertEqual(len(traj.segments), 2)
        self.assertEqual(traj.gaps_count, 1)

        # Segment 1: J01 -> J02 (direct single-hop R01)
        self.assertFalse(traj.segments[0].is_gap)

        # Segment 2: J02 -> J08 (gap through unobserved J05)
        self.assertTrue(traj.segments[1].is_gap)
        self.assertEqual(traj.segments[1].gap_state, "missing_intermediate_observations")

        # Full journey route is composed
        self.assertIn("R01", traj.complete_route_edges)
        self.assertIn("R08", traj.complete_route_edges)
        self.assertIn("R09", traj.complete_route_edges)

    def test_10_case_8_multiple_missing_intermediate_observations_across_city(self):
        """CASE 8: Multiple missing intermediate observations across city (J01 -> J12, 800s)."""
        obs_a = {"observation_id": "o1", "camera_id": "CAM_J01", "timestamp_seconds": 1000.0}
        obs_b = {"observation_id": "o2", "camera_id": "CAM_J12", "timestamp_seconds": 1800.0}

        gap = infer_sparse_gap(obs_a, obs_b, self.graph, max_paths=5)
        self.assertEqual(gap.status, "success")
        self.assertTrue(gap.feasible)
        self.assertGreaterEqual(len(gap.unobserved_intermediate_nodes), 3)

        # Unobserved junctions should include central hubs like J08, J05, J10, etc.
        for node in gap.unobserved_intermediate_nodes:
            self.assertIn(node, self.graph.nodes)

    def test_11_case_9_no_feasible_route(self):
        """CASE 9: No feasible route due to extreme required speed (J01 -> J12 in 30s)."""
        obs_a = {"observation_id": "o1", "camera_id": "CAM_J01", "timestamp_seconds": 1000.0}
        obs_b = {"observation_id": "o2", "camera_id": "CAM_J12", "timestamp_seconds": 1030.0}

        gap = infer_sparse_gap(obs_a, obs_b, self.graph, max_paths=5)
        self.assertEqual(gap.status, "infeasible")
        self.assertFalse(gap.feasible)
        self.assertEqual(gap.gap_state, "no_feasible_route")
        self.assertIn("physically infeasible", gap.ambiguity_reason)

    def test_12_case_10_invalid_or_non_positive_time_interval(self):
        """CASE 10: Invalid/non-positive time interval (delta_t <= 0). Fails safely."""
        # Reversed timestamps (delta_t = -50s)
        obs_a = {"observation_id": "o1", "camera_id": "CAM_J01", "timestamp_seconds": 1000.0}
        obs_b = {"observation_id": "o2", "camera_id": "CAM_J08", "timestamp_seconds": 950.0}

        gap_rev = infer_sparse_gap(obs_a, obs_b, self.graph)
        self.assertEqual(gap_rev.status, "temporal_inversion")
        self.assertFalse(gap_rev.feasible)
        self.assertEqual(gap_rev.gap_state, "invalid_time_interval")

        # Simultaneous distinct locations (delta_t = 0.0s)
        obs_c = {"observation_id": "o3", "camera_id": "CAM_J08", "timestamp_seconds": 1000.0}
        gap_sim = infer_sparse_gap(obs_a, obs_c, self.graph)
        self.assertEqual(gap_sim.status, "infeasible")
        self.assertFalse(gap_sim.feasible)
        self.assertEqual(gap_sim.gap_state, "invalid_time_interval")

    # =========================================================================
    # 3. CRITICAL PRINCIPLE: ZERO OBSERVATION FABRICATION
    # =========================================================================

    def test_13_zero_observation_fabrication(self):
        """Verify that inferring hidden routes NEVER fabricates synthetic Observation records."""
        # 2 sightings with 2 missing intermediate cameras (J04 unobserved)
        obs_list = [
            {"observation_id": "obs_orig", "camera_id": "CAM_J01", "timestamp_seconds": 1000.0},
            {"observation_id": "obs_dest", "camera_id": "CAM_J08", "timestamp_seconds": 1450.0},
        ]
        ident = {
            "identity_id": "TRK_ZERO_FABRICATION",
            "observations": obs_list,
        }

        traj = infer_sparse_identity_trajectory(ident, self.graph)

        # Observations count must strictly equal 2! No fake observations added.
        self.assertEqual(traj.observations_count, 2)
        self.assertEqual(len(traj.cameras_visited), 2)
        self.assertEqual(traj.cameras_visited, ["CAM_J01", "CAM_J08"])
        self.assertNotIn("CAM_J04", traj.cameras_visited)
        self.assertNotIn("CAM_J02", traj.cameras_visited)

    # =========================================================================
    # 4. MEMBER 3 CONTRACT COMPATIBILITY & DEMAND WEIGHTING
    # =========================================================================

    def test_14_adapt_sparse_gap_to_normalized_trajectory(self):
        """Verify adapting SparseObservationGap into NormalizedTrajectory conforming to Member 3."""
        obs_a = {"observation_id": "o1", "camera_id": "CAM_J01", "timestamp_seconds": 1000.0}
        obs_b = {"observation_id": "o2", "camera_id": "CAM_J08", "timestamp_seconds": 1450.0}

        gap = infer_sparse_gap(obs_a, obs_b, self.graph, identity_id="TRK_M3_GAP", max_paths=5)
        norm_traj = adapt_sparse_gap_to_normalized(gap, vehicle_weight=1.5, vehicle_class="truck")

        # Verify top-level contract fields
        self.assertEqual(norm_traj.track_id, "TRK_M3_GAP")
        self.assertEqual(norm_traj.origin_node, "J01")
        self.assertEqual(norm_traj.destination_node, "J08")
        self.assertEqual(norm_traj.vehicle_weight, 1.5)
        self.assertEqual(norm_traj.time_window_start, 1000.0)
        self.assertEqual(norm_traj.time_window_end, 1450.0)

        # Verify probabilities sum to 1.0 within 1e-6
        prob_sum = sum(r.probability for r in norm_traj.candidate_routes)
        self.assertAlmostEqual(prob_sum, 1.0, places=6)

        # Verify Day 4 gap metadata injection
        self.assertTrue(norm_traj.metadata["gap_detected"])
        self.assertEqual(norm_traj.metadata["gap_duration_seconds"], 450.0)
        self.assertEqual(norm_traj.metadata["observed_endpoints"], ["CAM_J01", "CAM_J08"])
        self.assertTrue(norm_traj.metadata["inferred_segment"])
        self.assertEqual(norm_traj.metadata["gap_state"], "missing_intermediate_observations")
        self.assertEqual(norm_traj.metadata["observations_used"], ["o1", "o2"])
        self.assertEqual(norm_traj.metadata["vehicle_class"], "truck")

        # Verify downstream route demand calculation: W * P
        demands = norm_traj.calculate_route_demands()
        total_demand = sum(d["route_demand"] for d in demands)
        self.assertAlmostEqual(total_demand, 1.5, places=4)

    def test_15_batch_serialization_with_sparse_gaps(self):
        """Verify adapt_trajectories_to_batch_payload serializes SparseObservationGap instances."""
        obs_a = {"observation_id": "o1", "camera_id": "CAM_J01", "timestamp_seconds": 1000.0}
        obs_b = {"observation_id": "o2", "camera_id": "CAM_J08", "timestamp_seconds": 1450.0}

        gap = infer_sparse_gap(obs_a, obs_b, self.graph, identity_id="TRK_BATCH_GAP")
        payload = adapt_trajectories_to_batch_payload([gap], default_weight=1.0)

        self.assertEqual(payload["trajectories_count"], 1)
        traj_entry = payload["trajectories"][0]
        self.assertEqual(traj_entry["track_id"], "TRK_BATCH_GAP")
        self.assertTrue(traj_entry["metadata"]["gap_detected"])
        self.assertEqual(traj_entry["metadata"]["gap_duration_seconds"], 450.0)

    # =========================================================================
    # 5. CORRECTIVE VALIDATION: UNOBSERVED INTERMEDIATE NODES (TESTS 1 - 7)
    # =========================================================================

    def test_16_corrective_test_1_single_candidate_route(self):
        """TEST 1: Single candidate route reports internal nodes exactly matching that route."""
        # J01 to J08 with tight delta_t = 310s (only shortest path J01->J04->J08 is feasible)
        obs_a = {"observation_id": "o1", "camera_id": "CAM_J01", "timestamp_seconds": 1000.0}
        obs_b = {"observation_id": "o2", "camera_id": "CAM_J08", "timestamp_seconds": 1310.0}

        gap = infer_sparse_gap(obs_a, obs_b, self.graph, identity_id="TRK_TEST1")
        feasible_routes = [r for r in gap.candidate_routes if r.feasible]
        self.assertEqual(len(feasible_routes), 1)

        single_route = feasible_routes[0]
        self.assertEqual(single_route.nodes, ["J01", "J04", "J08"])
        self.assertEqual(single_route.unobserved_intermediate_nodes, ["J04"])
        self.assertEqual(gap.unobserved_intermediate_nodes, ["J04"])
        self.assertEqual(gap.most_likely_intermediate_nodes, ["J04"])
        self.assertEqual(gap.common_intermediate_nodes, ["J04"])

        norm = adapt_sparse_gap_to_normalized(gap)
        self.assertEqual(norm.candidate_routes[0].metadata["unobserved_intermediate_nodes"], ["J04"])

    def test_17_corrective_test_2_multiple_candidate_routes_route_specific_nodes(self):
        """TEST 2: Multiple candidate routes each report only their own route-specific intermediate nodes."""
        obs_a = {"observation_id": "o1", "camera_id": "CAM_J01", "timestamp_seconds": 1000.0}
        obs_b = {"observation_id": "o2", "camera_id": "CAM_J08", "timestamp_seconds": 1450.0}

        gap = infer_sparse_gap(obs_a, obs_b, self.graph, identity_id="TRK_TEST2", max_paths=5)
        feasible_routes = [r for r in gap.candidate_routes if r.feasible]
        self.assertGreaterEqual(len(feasible_routes), 3)

        # Map by route nodes
        routes_by_nodes = {tuple(r.nodes): r for r in feasible_routes}

        # Route 1: J01 -> J04 -> J08
        r1 = routes_by_nodes.get(("J01", "J04", "J08"))
        self.assertIsNotNone(r1)
        self.assertEqual(r1.unobserved_intermediate_nodes, ["J04"])
        self.assertNotIn("J02", r1.unobserved_intermediate_nodes)
        self.assertNotIn("J05", r1.unobserved_intermediate_nodes)

        # Route 2: J01 -> J02 -> J05 -> J08
        r2 = routes_by_nodes.get(("J01", "J02", "J05", "J08"))
        self.assertIsNotNone(r2)
        self.assertEqual(r2.unobserved_intermediate_nodes, ["J02", "J05"])
        self.assertNotIn("J04", r2.unobserved_intermediate_nodes)

        # Route 3: J01 -> J04 -> J05 -> J08
        r3 = routes_by_nodes.get(("J01", "J04", "J05", "J08"))
        self.assertIsNotNone(r3)
        self.assertEqual(r3.unobserved_intermediate_nodes, ["J04", "J05"])
        self.assertNotIn("J02", r3.unobserved_intermediate_nodes)

        # Check NormalizedCandidateRoute metadata
        norm = adapt_sparse_gap_to_normalized(gap)
        norm_map = {tuple(cr.nodes): cr for cr in norm.candidate_routes}
        self.assertEqual(norm_map[("J01", "J04", "J08")].metadata["unobserved_intermediate_nodes"], ["J04"])
        self.assertEqual(norm_map[("J01", "J02", "J05", "J08")].metadata["unobserved_intermediate_nodes"], ["J02", "J05"])
        self.assertEqual(norm_map[("J01", "J04", "J05", "J08")].metadata["unobserved_intermediate_nodes"], ["J04", "J05"])

    def test_18_corrective_test_3_infeasible_search_nodes_excluded(self):
        """TEST 3: Nodes explored during search but NOT in accepted candidate routes are NOT reported."""
        obs_a = {"observation_id": "o1", "camera_id": "CAM_J01", "timestamp_seconds": 1000.0}
        obs_b = {"observation_id": "o2", "camera_id": "CAM_J08", "timestamp_seconds": 1450.0}

        # Find raw candidate paths from graph
        raw_paths = self.graph.find_candidate_paths("J01", "J08", max_paths=5)
        raw_explored_nodes = set()
        for p in raw_paths:
            raw_explored_nodes.update(p["nodes"][1:-1])

        # J06 was visited in raw paths 4 & 5 (7500m length requiring 60 km/h)
        self.assertIn("J06", raw_explored_nodes)

        gap = infer_sparse_gap(obs_a, obs_b, self.graph, identity_id="TRK_TEST3", max_paths=5)

        # Infeasible routes must have status != 'feasible'
        infeasible_routes = [r for r in gap.candidate_routes if not r.feasible]
        self.assertTrue(any("J06" in r.nodes for r in infeasible_routes))

        # BUT accepted candidate routes and the gap's unobserved intermediate nodes must NOT contain J06
        feasible_routes = [r for r in gap.candidate_routes if r.feasible]
        for r in feasible_routes:
            self.assertNotIn("J06", r.unobserved_intermediate_nodes)

        self.assertNotIn("J06", gap.unobserved_intermediate_nodes)
        self.assertEqual(gap.unobserved_intermediate_nodes, ["J02", "J04", "J05"])

        norm = adapt_sparse_gap_to_normalized(gap)
        self.assertNotIn("J06", norm.metadata["unobserved_intermediate_nodes"])

    def test_19_corrective_test_4_origin_and_destination_never_intermediate(self):
        """TEST 4: Origin and destination must never be reported as intermediate nodes."""
        obs_a = {"observation_id": "o1", "camera_id": "CAM_J01", "timestamp_seconds": 1000.0}
        obs_b = {"observation_id": "o2", "camera_id": "CAM_J08", "timestamp_seconds": 1450.0}

        gap = infer_sparse_gap(obs_a, obs_b, self.graph, identity_id="TRK_TEST4", max_paths=5)
        self.assertNotIn("J01", gap.unobserved_intermediate_nodes)
        self.assertNotIn("J08", gap.unobserved_intermediate_nodes)

        for r in gap.candidate_routes:
            self.assertNotIn("J01", r.unobserved_intermediate_nodes)
            self.assertNotIn("J08", r.unobserved_intermediate_nodes)

        # Single-hop adjacent edge (J01 -> J02)
        obs_c = {"observation_id": "o3", "camera_id": "CAM_J02", "timestamp_seconds": 1150.0}
        gap_adj = infer_sparse_gap(obs_a, obs_c, self.graph, identity_id="TRK_TEST4_ADJ")
        self.assertEqual(gap_adj.unobserved_intermediate_nodes, [])
        for r in gap_adj.candidate_routes:
            self.assertEqual(r.unobserved_intermediate_nodes, [])

    def test_20_corrective_test_5_closed_road_exclusion(self):
        """TEST 5: Closed roads exclude routes and their nodes are not reported."""
        # Create road graph and close R11 (J04 -> J08)
        closed_graph = RoadGraph.from_json_file(self.network_path)
        closed_graph.camera_associations["CAM_J01"] = "J01"
        closed_graph.camera_associations["CAM_J08"] = "J08"

        self.assertIn("R11", closed_graph.edges)
        closed_graph.close_road("R11")

        obs_a = {"observation_id": "o1", "camera_id": "CAM_J01", "timestamp_seconds": 1000.0}
        obs_b = {"observation_id": "o2", "camera_id": "CAM_J08", "timestamp_seconds": 1500.0}

        gap = infer_sparse_gap(obs_a, obs_b, closed_graph, identity_id="TRK_TEST5", max_paths=5)
        feasible_routes = [r for r in gap.candidate_routes if r.feasible]

        # Route J01 -> J04 -> J08 must be completely excluded
        for r in feasible_routes:
            self.assertNotIn("R11", r.edges)

        # All surviving candidate routes go via J05
        for r in feasible_routes:
            self.assertIn("J05", r.unobserved_intermediate_nodes)

    def test_21_corrective_test_6_ambiguous_trajectory_preservation(self):
        """TEST 6: Ambiguous trajectory preserves all candidate routes and route-specific intermediate nodes."""
        obs_a = {"observation_id": "o1", "camera_id": "CAM_J01", "timestamp_seconds": 1000.0}
        obs_b = {"observation_id": "o2", "camera_id": "CAM_J08", "timestamp_seconds": 1450.0}

        gap = infer_sparse_gap(obs_a, obs_b, self.graph, identity_id="TRK_TEST6", max_paths=5)
        self.assertTrue(gap.is_ambiguous)
        self.assertIsNotNone(gap.ambiguity_reason)

        feasible_routes = [r for r in gap.candidate_routes if r.feasible]
        self.assertGreaterEqual(len(feasible_routes), 2)

        # Check that candidate routes are NOT collapsed and each maintains distinct intermediate nodes
        nodes_sets = [set(r.unobserved_intermediate_nodes) for r in feasible_routes]
        self.assertNotEqual(nodes_sets[0], nodes_sets[1])

        # Adapt to NormalizedTrajectory
        norm = adapt_sparse_gap_to_normalized(gap)
        self.assertTrue(norm.metadata["is_ambiguous"])
        self.assertAlmostEqual(sum(cr.probability for cr in norm.candidate_routes), 1.0, places=6)

    def test_22_corrective_test_7_day4_scenario_regression(self):
        """TEST 7: Regression test for Day 4 scenario ensuring J06 never appears in feasible routes or union."""
        case1 = self.fixture["scenarios"][0]
        obs_a = case1["obs_a"]
        obs_b = case1["obs_b"]

        gap = infer_sparse_gap(
            obs_a,
            obs_b,
            self.graph,
            identity_id=case1["identity_id"],
            max_paths=5,
        )

        feasible_routes = [r for r in gap.candidate_routes if r.feasible]
        self.assertEqual(len(feasible_routes), 3)

        # Check exactly the 3 feasible routes
        expected_routes_nodes = [
            ["J01", "J04", "J08"],
            ["J01", "J02", "J05", "J08"],
            ["J01", "J04", "J05", "J08"],
        ]
        actual_routes_nodes = [r.nodes for r in feasible_routes]
        for exp in expected_routes_nodes:
            self.assertIn(exp, actual_routes_nodes)

        # Verify J06 does NOT appear anywhere among feasible routes
        for r in feasible_routes:
            self.assertNotIn("J06", r.nodes)
            self.assertNotIn("J06", r.unobserved_intermediate_nodes)

        # Verify unobserved_intermediate_nodes semantics
        self.assertEqual(gap.unobserved_intermediate_nodes, ["J02", "J04", "J05"])
        self.assertNotIn("J06", gap.unobserved_intermediate_nodes)

        # General assertion: every node in gap.unobserved_intermediate_nodes must be in at least one route
        union_check = set()
        for r in feasible_routes:
            union_check.update(r.unobserved_intermediate_nodes)
        self.assertEqual(set(gap.unobserved_intermediate_nodes), union_check)

        # Check most likely and common
        self.assertEqual(gap.most_likely_intermediate_nodes, ["J04"])
        self.assertEqual(gap.common_intermediate_nodes, [])

        # Check metadata
        self.assertEqual(gap.metadata["intermediate_node_semantics"], "union_of_feasible_candidate_routes")
        self.assertEqual(gap.metadata["most_likely_route_intermediate_nodes"], ["J04"])
        self.assertEqual(gap.metadata["common_intermediate_nodes"], [])


if __name__ == "__main__":
    unittest.main()

