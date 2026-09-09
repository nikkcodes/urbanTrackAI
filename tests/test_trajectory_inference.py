"""
Comprehensive Day 3 Trajectory Reconstruction Test Suite.
Verifies all 12 Edge Cases (CASE A through CASE L), candidate route generation,
feasibility checking, scoring, uncertainty preservation, and multi-observation chaining.
"""

import os
import unittest

from inference.road_graph import RoadEdge, RoadGraph, RoadNode
from inference.trajectory_engine import (
    reconstruct_identity_trajectory,
    reconstruct_trajectory_segment,
)
from schemas.observation_schema import Observation


class TestTrajectoryInference(unittest.TestCase):
    """Test suite covering edge cases A through L for Day 3 trajectory inference."""

    def setUp(self):
        """Build a controlled synthetic road graph for deterministic testing."""
        self.graph = RoadGraph(metadata={"name": "test_urban_network"})

        # Nodes:
        # N1 (cam_01), N2 (cam_02), N3 (cam_03), N4 (cam_04), N5 (isolated)
        self.graph.add_node(RoadNode(node_id="N1", name="Junction 1", latitude=17.3850, longitude=78.4867))
        self.graph.add_node(RoadNode(node_id="N2", name="Junction 2", latitude=17.3880, longitude=78.4867))
        self.graph.add_node(RoadNode(node_id="N3", name="Junction 3", latitude=17.3850, longitude=78.4900))
        self.graph.add_node(RoadNode(node_id="N4", name="Junction 4", latitude=17.3880, longitude=78.4900))
        self.graph.add_node(RoadNode(node_id="N5", name="Isolated Junction", latitude=17.4000, longitude=78.5000))

        # Roads:
        # Route 1 (Urban Arterial): N1 -> N2 -> N4 (350m + 350m = 700m, speed limit 50 km/h, expected 35 km/h)
        self.graph.add_edge(RoadEdge(
            road_id="road_12", name="West St", from_node="N1", to_node="N2",
            distance_m=350.0, speed_limit_kmh=50.0, expected_speed_kmh=35.0, one_way=True
        ))
        self.graph.add_edge(RoadEdge(
            road_id="road_24", name="North St", from_node="N2", to_node="N4",
            distance_m=350.0, speed_limit_kmh=50.0, expected_speed_kmh=35.0, one_way=True
        ))

        # Route 2 (Bypass Expressway): N1 -> N3 -> N4 (400m + 600m = 1000m, speed limit 100 km/h, expected 80 km/h)
        self.graph.add_edge(RoadEdge(
            road_id="road_13", name="South Bypass", from_node="N1", to_node="N3",
            distance_m=400.0, speed_limit_kmh=100.0, expected_speed_kmh=80.0, one_way=True
        ))
        self.graph.add_edge(RoadEdge(
            road_id="road_34", name="East Expressway", from_node="N3", to_node="N4",
            distance_m=600.0, speed_limit_kmh=100.0, expected_speed_kmh=80.0, one_way=True
        ))

        # Single corridor branch: N4 -> N2 (one-way reverse)
        self.graph.add_edge(RoadEdge(
            road_id="road_42", name="One Way Ave", from_node="N4", to_node="N2",
            distance_m=350.0, speed_limit_kmh=40.0, expected_speed_kmh=30.0, one_way=True
        ))

        # Explicit camera associations
        self.graph.camera_associations["cam_01"] = "N1"
        self.graph.camera_associations["cam_02"] = "N2"
        self.graph.camera_associations["cam_03"] = "N3"
        self.graph.camera_associations["cam_04"] = "N4"
        self.graph.camera_associations["cam_isolated"] = "N5"

    def test_case_a_one_clear_feasible_route(self):
        """CASE A: Exactly one feasible route exists between start and end."""
        # N1 -> N2: only road_12 connects them directly in this graph
        obs_a = {"camera_id": "cam_01", "timestamp": 100.0, "latitude": 17.3850, "longitude": 78.4867}
        obs_b = {"camera_id": "cam_02", "timestamp": 135.0, "latitude": 17.3880, "longitude": 78.4867}

        seg = reconstruct_trajectory_segment(obs_a, obs_b, self.graph, max_paths=3)
        self.assertTrue(seg.feasible)
        self.assertEqual(len(seg.candidate_routes), 1)
        self.assertEqual(seg.most_likely_route, ["road_12"])
        self.assertAlmostEqual(seg.confidence, 1.0, places=2)
        self.assertFalse(seg.is_ambiguous)
        # Required speed: 350m / 35s = 10 m/s = 36 km/h, well under 50 km/h speed limit
        self.assertAlmostEqual(seg.candidate_routes[0].required_speed_kmh, 36.0, places=1)

    def test_case_b_multiple_feasible_routes(self):
        """CASE B: Multiple plausible alternative routes are discovered and preserved."""
        # N1 -> N4 has two distinct corridors:
        # Corridor 1: road_12 -> road_24 (700m)
        # Corridor 2: road_13 -> road_34 (1000m)
        # Observed time: 60s
        # Required speed for Route 1: 700 / 60 = 11.67 m/s = 42.0 km/h (limit 50) -> feasible
        # Required speed for Route 2: 1000 / 60 = 16.67 m/s = 60.0 km/h (limit 100) -> feasible
        obs_a = {"camera_id": "cam_01", "timestamp": 100.0, "latitude": 17.3850, "longitude": 78.4867}
        obs_b = {"camera_id": "cam_04", "timestamp": 160.0, "latitude": 17.3880, "longitude": 78.4900}

        seg = reconstruct_trajectory_segment(obs_a, obs_b, self.graph, max_paths=5)
        self.assertTrue(seg.feasible)
        self.assertEqual(len(seg.candidate_routes), 2)
        # Both routes must be preserved in candidate_routes
        routes = [r.route for r in seg.candidate_routes]
        self.assertIn(["road_12", "road_24"], routes)
        self.assertIn(["road_13", "road_34"], routes)
        # All estimated likelihoods sum to 1.0
        total_p = sum(r.estimated_likelihood for r in seg.candidate_routes)
        self.assertAlmostEqual(total_p, 1.0, places=2)

    def test_case_c_shortest_route_infeasible_alternative_feasible(self):
        """CASE C: Shortest distance route is not temporally feasible, but longer alternative is."""
        # N1 -> N4 in delta_t = 40 seconds.
        # Corridor 1 (700m): required speed = 700 / 40 = 17.5 m/s = 63 km/h. Exceeds speed limit 50 km/h!
        # Corridor 2 (1000m, Expressway): required speed = 1000 / 40 = 25 m/s = 90 km/h. Below limit 100 km/h!
        obs_a = {"camera_id": "cam_01", "timestamp": 100.0, "latitude": 17.3850, "longitude": 78.4867}
        obs_b = {"camera_id": "cam_04", "timestamp": 140.0, "latitude": 17.3880, "longitude": 78.4900}

        seg = reconstruct_trajectory_segment(obs_a, obs_b, self.graph, max_paths=5)
        self.assertTrue(seg.feasible)
        # Most likely route must NOT be the shortest path because Corridor 1 is infeasible / heavily penalized
        self.assertEqual(seg.most_likely_route, ["road_13", "road_34"])

        # Check that Corridor 1 is flagged infeasible (speed 63 > 50)
        c1 = next(r for r in seg.candidate_routes if r.route == ["road_12", "road_24"])
        self.assertFalse(c1.feasible)

        # Check that Expressway Corridor 2 is feasible (speed 90 <= 100)
        c2 = next(r for r in seg.candidate_routes if r.route == ["road_13", "road_34"])
        self.assertTrue(c2.feasible)

    def test_case_d_all_routes_physically_infeasible(self):
        """CASE D: All candidate routes are physically impossible due to extreme speed."""
        # N1 -> N4 in delta_t = 5 seconds.
        # Required speeds: 700m / 5s = 504 km/h; 1000m / 5s = 720 km/h.
        obs_a = {"camera_id": "cam_01", "timestamp": 100.0, "latitude": 17.3850, "longitude": 78.4867}
        obs_b = {"camera_id": "cam_04", "timestamp": 105.0, "latitude": 17.3880, "longitude": 78.4900}

        seg = reconstruct_trajectory_segment(obs_a, obs_b, self.graph, max_paths=5)
        self.assertFalse(seg.feasible)
        self.assertEqual(seg.confidence, 0.0)
        # All candidates must have feasible=False
        for r in seg.candidate_routes:
            self.assertFalse(r.feasible)

    def test_case_e_same_camera_at_different_times(self):
        """CASE E: Same camera observation at different times (stationary vehicle or loitering)."""
        obs_a = {"camera_id": "cam_01", "timestamp": 100.0, "latitude": 17.3850, "longitude": 78.4867}
        obs_b = {"camera_id": "cam_01", "timestamp": 180.0, "latitude": 17.3850, "longitude": 78.4867}

        seg = reconstruct_trajectory_segment(obs_a, obs_b, self.graph)
        self.assertTrue(seg.feasible)
        self.assertEqual(seg.most_likely_route, [])
        self.assertEqual(seg.travel_distance_m, 0.0)
        self.assertIn("Stationary", seg.evidence_explanation)

    def test_case_f_reversed_timestamps(self):
        """CASE F: Observation timestamps are reversed (delta_t < 0). Fails safely."""
        obs_a = {"camera_id": "cam_01", "timestamp": 200.0, "latitude": 17.3850, "longitude": 78.4867}
        obs_b = {"camera_id": "cam_04", "timestamp": 100.0, "latitude": 17.3880, "longitude": 78.4900}

        seg = reconstruct_trajectory_segment(obs_a, obs_b, self.graph)
        self.assertFalse(seg.feasible)
        self.assertEqual(seg.confidence, 0.0)
        self.assertIn("Negative time difference", seg.evidence_explanation)

    def test_case_g_unassociated_camera(self):
        """CASE G: Camera cannot be associated with the road graph. Fails safely, no route fabricated."""
        obs_a = {"camera_id": "cam_unmapped_far", "timestamp": 100.0, "latitude": 28.6139, "longitude": 77.2090}
        obs_b = {"camera_id": "cam_04", "timestamp": 150.0, "latitude": 17.3880, "longitude": 78.4900}

        seg = reconstruct_trajectory_segment(obs_a, obs_b, self.graph)
        self.assertFalse(seg.feasible)
        self.assertEqual(seg.confidence, 0.0)
        self.assertEqual(len(seg.candidate_routes), 0)
        self.assertIn("could not be associated", seg.evidence_explanation)

    def test_case_h_no_path_exists_between_nodes(self):
        """CASE H: Start and end cameras exist in disconnected components or one-way traps."""
        # N4 -> N1: One-way road_42 goes N4 -> N2, but no road goes from N2 to N1!
        obs_a = {"camera_id": "cam_04", "timestamp": 100.0, "latitude": 17.3880, "longitude": 78.4900}
        obs_b = {"camera_id": "cam_01", "timestamp": 150.0, "latitude": 17.3850, "longitude": 78.4867}

        seg = reconstruct_trajectory_segment(obs_a, obs_b, self.graph)
        self.assertFalse(seg.feasible)
        self.assertEqual(len(seg.candidate_routes), 0)
        self.assertIn("No connected", seg.evidence_explanation)

        # Isolated junction N5
        obs_isolated = {"camera_id": "cam_isolated", "timestamp": 150.0, "latitude": 17.4000, "longitude": 78.5000}
        seg_iso = reconstruct_trajectory_segment(obs_a, obs_isolated, self.graph)
        self.assertFalse(seg_iso.feasible)

    def test_case_i_very_small_time_difference_unrealistic_speed(self):
        """CASE I: Very small time difference delta_t = 0.5s for 700m travel."""
        obs_a = {"camera_id": "cam_01", "timestamp": 100.0, "latitude": 17.3850, "longitude": 78.4867}
        obs_b = {"camera_id": "cam_04", "timestamp": 100.5, "latitude": 17.3880, "longitude": 78.4900}

        seg = reconstruct_trajectory_segment(obs_a, obs_b, self.graph)
        self.assertFalse(seg.feasible)
        self.assertEqual(seg.confidence, 0.0)
        for r in seg.candidate_routes:
            self.assertFalse(r.feasible)

    def test_case_j_two_routes_nearly_identical_feasibility_preserves_uncertainty(self):
        """CASE J: Two routes have nearly identical feasibility; uncertainty must be preserved."""
        # Create symmetric graph: N1 -> NA -> N4 (500m, 50km/h) and N1 -> NB -> N4 (500m, 50km/h)
        sym_graph = RoadGraph(metadata={"name": "symmetric_network"})
        sym_graph.add_node(RoadNode(node_id="N1", name="J1", latitude=17.3850, longitude=78.4867))
        sym_graph.add_node(RoadNode(node_id="NA", name="JA", latitude=17.3870, longitude=78.4850))
        sym_graph.add_node(RoadNode(node_id="NB", name="JB", latitude=17.3870, longitude=78.4880))
        sym_graph.add_node(RoadNode(node_id="N4", name="J4", latitude=17.3890, longitude=78.4867))

        sym_graph.add_edge(RoadEdge(road_id="R_1A", from_node="N1", to_node="NA", distance_m=250.0, speed_limit_kmh=50.0, expected_speed_kmh=40.0))
        sym_graph.add_edge(RoadEdge(road_id="R_A4", from_node="NA", to_node="N4", distance_m=250.0, speed_limit_kmh=50.0, expected_speed_kmh=40.0))
        sym_graph.add_edge(RoadEdge(road_id="R_1B", from_node="N1", to_node="NB", distance_m=250.0, speed_limit_kmh=50.0, expected_speed_kmh=40.0))
        sym_graph.add_edge(RoadEdge(road_id="R_B4", from_node="NB", to_node="N4", distance_m=250.0, speed_limit_kmh=50.0, expected_speed_kmh=40.0))

        sym_graph.camera_associations["cam_01"] = "N1"
        sym_graph.camera_associations["cam_04"] = "N4"

        obs_a = {"camera_id": "cam_01", "timestamp": 100.0, "latitude": 17.3850, "longitude": 78.4867}
        obs_b = {"camera_id": "cam_04", "timestamp": 145.0, "latitude": 17.3890, "longitude": 78.4867}

        seg = reconstruct_trajectory_segment(obs_a, obs_b, sym_graph, max_paths=5)
        self.assertTrue(seg.feasible)
        self.assertEqual(len(seg.candidate_routes), 2)

        p1 = seg.candidate_routes[0].estimated_likelihood
        p2 = seg.candidate_routes[1].estimated_likelihood

        # Likelihoods must be approximately 0.50 each (symmetric)
        self.assertAlmostEqual(p1, 0.50, delta=0.05)
        self.assertAlmostEqual(p2, 0.50, delta=0.05)
        # CRITICAL UNCERTAINTY PRINCIPLE:
        # Confidence must NOT be artificially inflated to 0.99!
        self.assertLess(seg.confidence, 0.70)
        self.assertTrue(seg.is_ambiguous)
        self.assertIn("ambiguous", seg.evidence_explanation.lower())

    def test_case_k_multi_observation_trajectory(self):
        """CASE K: Identity contains multiple sequential observations (A -> B -> C -> D)."""
        # Chain: cam_01 (N1) -> cam_02 (N2) -> cam_04 (N4)
        identity_data = {
            "identity_id": "VEHICLE_CANDIDATE_001",
            "observations": [
                {"camera_id": "cam_01", "timestamp": 100.0, "latitude": 17.3850, "longitude": 78.4867},
                {"camera_id": "cam_02", "timestamp": 135.0, "latitude": 17.3880, "longitude": 78.4867},
                {"camera_id": "cam_04", "timestamp": 170.0, "latitude": 17.3880, "longitude": 78.4900},
            ]
        }

        traj = reconstruct_identity_trajectory(identity_data, self.graph)
        self.assertEqual(traj.identity_id, "VEHICLE_CANDIDATE_001")
        self.assertEqual(len(traj.segments), 2)
        self.assertTrue(traj.feasible)

        # Full route should seamlessly chain: [road_12] + [road_24] = [road_12, road_24]
        self.assertEqual(traj.full_route, ["road_12", "road_24"])
        self.assertAlmostEqual(traj.total_distance_m, 700.0, places=1)
        self.assertAlmostEqual(traj.total_duration_s, 70.0, places=1)
        self.assertGreater(traj.overall_confidence, 0.8)

    def test_case_l_missing_optional_observation_fields(self):
        """CASE L: Observation dictionaries lack optional fields (e.g. None vehicle_type or None coords)."""
        # Coordinates missing from dict, but camera_id has explicit mapping
        obs_a = {"camera_id": "cam_01", "timestamp": 100.0}
        obs_b = {"camera_id": "cam_02", "timestamp": 135.0}

        seg = reconstruct_trajectory_segment(obs_a, obs_b, self.graph)
        self.assertTrue(seg.feasible)
        self.assertEqual(seg.most_likely_route, ["road_12"])
        self.assertAlmostEqual(seg.travel_distance_m, 350.0, places=1)


if __name__ == "__main__":
    unittest.main()
