"""
Member 3 Integration Contract & Spatial Network Test Suite.

Validates that Day 3 Trajectory Reconstruction cleanly and natively integrates
with Member 3's real spatial graph (data/synthetic/city_network.json) and produces
trajectories conforming to the downstream NormalizedTrajectory interface contract.

Contract Verified:
NormalizedTrajectory
├── track_id
├── origin_node
├── destination_node
├── vehicle_weight
├── candidate_routes[]
│   ├── nodes
│   └── probability
└── time_window
    ├── start
    └── end
"""

import math
import os
from pathlib import Path
import unittest

from inference.member3_adapter import (
    adapt_trajectories_to_batch_payload,
    adapt_trajectory_segment_to_normalized,
    adapt_vehicle_trajectory_to_normalized,
)
from inference.road_graph import RoadEdge, RoadGraph, RoadNode
from inference.trajectory_engine import (
    reconstruct_identity_trajectory,
    reconstruct_trajectory_segment,
)
from schemas.normalized_trajectory_schema import (
    InvalidRouteError,
    NormalizedCandidateRoute,
    NormalizedTrajectory,
    ProbabilityValidationError,
)
from schemas.trajectory_schema import CandidateRoute, TrajectorySegment, VehicleTrajectory


class TestMember3Integration(unittest.TestCase):
    """
    Comprehensive test suite covering:
    1. Member 3 city_network.json loading, directed edges, units, and closed roads.
    2. Camera spatial snapping and observation mapping.
    3. Day-3 trajectory reconstruction on city_network.json using J01..J14 & R01..R28.
    4. NormalizedTrajectory contract, probability distribution, and demand weighting (W * P).
    5. Multi-observation trajectory corridors and Member 3 payload compatibility.
    """

    def setUp(self):
        """Set up test environment with Member 3's city_network.json."""
        self.network_path = Path("data/synthetic/city_network.json")
        self.assertTrue(self.network_path.is_file(), f"Missing city_network.json at {self.network_path}")
        self.graph = RoadGraph.from_json_file(self.network_path)

        # Associate test cameras with Member 3 junctions
        # J01 (lat: 28.6400, lon: 77.2000) - North Gate Terminal
        # J02 (lat: 28.6350, lon: 77.2100) - North Junction
        # J05 (lat: 28.6200, lon: 77.2180) - Midtown Circle
        # J08 (lat: 28.6100, lon: 77.2100) - Central Square
        # J10 (lat: 28.6000, lon: 77.2000) - South Boulevard
        # J12 (lat: 28.5900, lon: 77.2100) - South Hub Terminal
        self.graph.camera_associations["CAM_J01"] = "J01"
        self.graph.camera_associations["CAM_J02"] = "J02"
        self.graph.camera_associations["CAM_J05"] = "J05"
        self.graph.camera_associations["CAM_J08"] = "J08"
        self.graph.camera_associations["CAM_J10"] = "J10"
        self.graph.camera_associations["CAM_J12"] = "J12"

    # =========================================================================
    # 1. MEMBER 3 GRAPH LOADING & UNIT CONVERSION TESTS
    # =========================================================================

    def test_01_member3_graph_loads_all_nodes_and_roads(self):
        """Verify city_network.json loads exactly 14 junctions (J01..J14) and 28 roads (R01..R28)."""
        self.assertEqual(len(self.graph.nodes), 14)
        self.assertEqual(len(self.graph.edges), 28)

        # Check all J01..J14 are present
        for i in range(1, 15):
            node_id = f"J{i:02d}"
            self.assertIn(node_id, self.graph.nodes)
            node = self.graph.nodes[node_id]
            self.assertTrue(28.5 <= node.latitude <= 28.7)
            self.assertTrue(77.1 <= node.longitude <= 77.3)

        # Check all R01..R28 are present
        for i in range(1, 29):
            road_id = f"R{i:02d}"
            self.assertIn(road_id, self.graph.edges)

    def test_02_distance_units_conversion_km_to_meters(self):
        """Verify distance_km in Member 3's graph is correctly converted to distance_m (x1000)."""
        # R01: J01 -> J02, distance_km = 1.8 -> distance_m = 1800.0
        r01 = self.graph.edges["R01"]
        self.assertAlmostEqual(r01.distance_km, 1.8, places=3)
        self.assertAlmostEqual(r01.distance_m, 1800.0, places=1)

        # R16: J07 -> J14, distance_km = 3.2 -> distance_m = 3200.0
        r16 = self.graph.edges["R16"]
        self.assertAlmostEqual(r16.distance_km, 3.2, places=3)
        self.assertAlmostEqual(r16.distance_m, 3200.0, places=1)

    def test_03_speed_limit_units_and_free_flow_speed(self):
        """Verify speed limits (kmph) and free-flow expected speeds from city_network.json."""
        # R01: speed_limit_kmph = 50.0, free_flow_time_min = 2.16 -> 1.8 / (2.16 / 60) = 50.0 km/h
        r01 = self.graph.edges["R01"]
        self.assertEqual(r01.speed_limit_kmh, 50.0)
        self.assertAlmostEqual(r01.expected_speed_kmh, 50.0, places=1)

        # R16: speed_limit_kmph = 80.0
        r16 = self.graph.edges["R16"]
        self.assertEqual(r16.speed_limit_kmh, 80.0)

    def test_04_directed_roads_behavior(self):
        """Verify directed edges are respected: R01 connects J01 -> J02; reverse requires R28."""
        # Forward path from J01 to J02 exists
        dist_fwd = self.graph.find_shortest_distance("J01", "J02")
        self.assertIsNotNone(dist_fwd)
        self.assertAlmostEqual(dist_fwd, 1800.0, places=1)

        # Reverse path from J02 to J01 uses R28 (1800m)
        dist_rev = self.graph.find_shortest_distance("J02", "J01")
        self.assertIsNotNone(dist_rev)
        self.assertAlmostEqual(dist_rev, 1800.0, places=1)

        # One-way road R02 connects J02 -> J03, but no direct reverse road exists
        neighbors_j03 = [dest for dest, _, _, _ in self.graph.adjacency.get("J03", [])]
        self.assertNotIn("J02", neighbors_j03)

    def test_05_closed_road_exclusion(self):
        """Verify that closing a road excludes it from active routing and candidate routes."""
        # Initially, J02 -> J05 uses R08 (1600m)
        r08 = self.graph.edges["R08"]
        self.assertFalse(r08.is_closed)
        paths_before = self.graph.find_candidate_paths("J02", "J05")
        self.assertGreaterEqual(len(paths_before), 1)
        self.assertIn("R08", paths_before[0]["edges"])

        # Dynamically close R08
        closed = self.graph.close_road("R08")
        self.assertTrue(closed)

        # R08 must now be excluded from candidate paths
        paths_after = self.graph.find_candidate_paths("J02", "J05")
        for p in paths_after:
            self.assertNotIn("R08", p["edges"])

        # Restore R08
        restored = self.graph.restore_road("R08")
        self.assertTrue(restored)
        paths_restored = self.graph.find_candidate_paths("J02", "J05")
        self.assertIn("R08", paths_restored[0]["edges"])

    # =========================================================================
    # 2. CAMERA SPATIAL MAPPING TESTS
    # =========================================================================

    def test_06_camera_spatial_snapping_to_junctions(self):
        """Verify observations snap to nearest junction within 150m or fail safely."""
        # Coordinate exact at J01 (28.6400, 77.2000)
        snapped_j01 = self.graph.associate_camera("CAM_NEW_01", latitude=28.6400, longitude=77.2000)
        self.assertEqual(snapped_j01, "J01")

        # Coordinate within 50m of J02 (28.6350, 77.2100)
        snapped_j02 = self.graph.associate_camera("CAM_NEAR_02", latitude=28.6352, longitude=77.2102)
        self.assertEqual(snapped_j02, "J02")

        # Coordinate in different city (Hyderabad 17.385, 78.486) -> exceeds 150m, fails safely
        snapped_far = self.graph.associate_camera("CAM_HYD", latitude=17.3850, longitude=78.4867)
        self.assertIsNone(snapped_far)

        # Explicit mapped camera
        snapped_explicit = self.graph.associate_camera("CAM_J08")
        self.assertEqual(snapped_explicit, "J08")

    # =========================================================================
    # 3. DAY 3 TRAJECTORY RECONSTRUCTION ON MEMBER 3 GRAPH
    # =========================================================================

    def test_07_candidate_routes_use_member3_identifiers(self):
        """Verify candidate routes generated use J01..J14 and R01..R28 identifiers."""
        # Travel J01 -> J08 (approx 5.5 km, speed limit 50 km/h -> ~400s)
        obs_a = {"camera_id": "CAM_J01", "timestamp": 1000.0}
        obs_b = {"camera_id": "CAM_J08", "timestamp": 1450.0}

        segment = reconstruct_trajectory_segment(obs_a, obs_b, self.graph, identity_id="TRK_TEST_001")
        self.assertEqual(segment.status, "success")
        self.assertGreaterEqual(len(segment.candidate_routes), 1)

        # Verify candidate routes use J01..J14 and R01..R28
        top_route = segment.candidate_routes[0]
        self.assertEqual(top_route.nodes[0], "J01")
        self.assertEqual(top_route.nodes[-1], "J08")
        for node in top_route.nodes:
            self.assertTrue(node.startswith("J"), f"Expected J identifier, got {node}")
        for edge in top_route.edges:
            self.assertTrue(edge.startswith("R"), f"Expected R identifier, got {edge}")

    def test_08_multiple_candidate_corridors_preserved_with_likelihoods(self):
        """Verify that alternative corridors between J01 and J12 are discovered and preserved."""
        # J01 -> J12: Corridor A (via Midtown J02-J05-J08-J10) vs Corridor B (via Civic J04-J08-J11)
        # Total distance ~ 9.4 km, normal driving time ~ 750s (~45 km/h)
        obs_a = {"camera_id": "CAM_J01", "timestamp": 1000.0}
        obs_b = {"camera_id": "CAM_J12", "timestamp": 1750.0}

        segment = reconstruct_trajectory_segment(obs_a, obs_b, self.graph, identity_id="TRK_TEST_002", max_paths=5)
        self.assertEqual(segment.status, "success")
        self.assertGreaterEqual(len(segment.candidate_routes), 2)

        # All candidate routes should have valid estimated likelihoods summing to 1.0
        feasible_routes = [r for r in segment.candidate_routes if r.feasible]
        self.assertGreaterEqual(len(feasible_routes), 2)
        total_likelihood = sum(r.estimated_likelihood for r in feasible_routes)
        self.assertAlmostEqual(total_likelihood, 1.0, places=4)

    def test_09_physically_impossible_speed_marked_infeasible(self):
        """Verify extreme speed travel between J01 and J12 is marked infeasible."""
        # 9.4 km in 30 seconds -> required speed > 1100 km/h
        obs_a = {"camera_id": "CAM_J01", "timestamp": 1000.0}
        obs_b = {"camera_id": "CAM_J12", "timestamp": 1030.0}

        segment = reconstruct_trajectory_segment(obs_a, obs_b, self.graph, identity_id="TRK_FAST")
        self.assertEqual(segment.status, "infeasible")
        self.assertFalse(segment.feasible)

    # =========================================================================
    # 4. NORMALIZEDTRAJECTORY CONTRACT & ADAPTER TESTS
    # =========================================================================

    def test_10_adapt_trajectory_segment_to_normalized(self):
        """Verify adapting segment into NormalizedTrajectory conforming to Member 3 contract."""
        obs_a = {"camera_id": "CAM_J01", "timestamp": 1000.0}
        obs_b = {"camera_id": "CAM_J08", "timestamp": 1450.0}

        segment = reconstruct_trajectory_segment(obs_a, obs_b, self.graph, identity_id="TRK_001")
        norm_traj = adapt_trajectory_segment_to_normalized(segment, vehicle_weight=1.0)

        # Check required Member 3 contract attributes
        self.assertEqual(norm_traj.track_id, "TRK_001")
        self.assertEqual(norm_traj.origin_node, "J01")
        self.assertEqual(norm_traj.destination_node, "J08")
        self.assertEqual(norm_traj.vehicle_weight, 1.0)
        self.assertEqual(norm_traj.time_window_start, 1000.0)
        self.assertEqual(norm_traj.time_window_end, 1450.0)
        self.assertGreaterEqual(len(norm_traj.candidate_routes), 1)

        # Check probability normalization (sum == 1.0 within 1e-6)
        total_p = sum(r.probability for r in norm_traj.candidate_routes)
        self.assertAlmostEqual(total_p, 1.0, places=6)

        # Check endpoints consistency
        for r in norm_traj.candidate_routes:
            self.assertEqual(r.nodes[0], "J01")
            self.assertEqual(r.nodes[-1], "J08")

    def test_11_route_demand_weighting_semantics(self):
        """Verify downstream route demand = vehicle_weight * route_probability."""
        routes = [
            NormalizedCandidateRoute(nodes=["J01", "J02", "J05", "J08", "J10", "J12"], probability=0.70),
            NormalizedCandidateRoute(nodes=["J01", "J04", "J08", "J11", "J12"], probability=0.30),
        ]
        traj = NormalizedTrajectory(
            track_id="TRK_001",
            origin_node="J01",
            destination_node="J12",
            candidate_routes=routes,
            vehicle_weight=1.0,
            time_window_start="08:00:00",
            time_window_end="08:15:00",
        )
        demands = traj.calculate_route_demands()
        self.assertEqual(len(demands), 2)
        # W = 1.0 -> 0.70 and 0.30
        self.assertAlmostEqual(demands[0]["route_demand"], 0.70, places=6)
        self.assertAlmostEqual(demands[1]["route_demand"], 0.30, places=6)

        # Heavy vehicle convoy: W = 2.5
        traj_heavy = NormalizedTrajectory(
            track_id="TRK_CONVOY",
            origin_node="J01",
            destination_node="J12",
            candidate_routes=routes,
            vehicle_weight=2.5,
        )
        demands_heavy = traj_heavy.calculate_route_demands()
        # 2.5 * 0.70 = 1.75; 2.5 * 0.30 = 0.75
        self.assertAlmostEqual(demands_heavy[0]["route_demand"], 1.75, places=6)
        self.assertAlmostEqual(demands_heavy[1]["route_demand"], 0.75, places=6)
        self.assertAlmostEqual(sum(d["route_demand"] for d in demands_heavy), 2.5, places=6)

    def test_12_multi_observation_journey_projection(self):
        """Verify multi-observation trajectory (J01 -> J02 -> J05 -> J08) projects full corridor."""
        # 3 legs: J01 -> J02 (1800m, ~130s), J02 -> J05 (1600m, ~120s), J05 -> J08 (2100m, ~150s)
        obs1 = {"observation_id": "o1", "camera_id": "CAM_J01", "timestamp": 1000.0}
        obs2 = {"observation_id": "o2", "camera_id": "CAM_J02", "timestamp": 1130.0}
        obs3 = {"observation_id": "o3", "camera_id": "CAM_J05", "timestamp": 1250.0}
        obs4 = {"observation_id": "o4", "camera_id": "CAM_J08", "timestamp": 1400.0}

        identity_data = {
            "identity_id": "VEHICLE_CANDIDATE_001",
            "observations": [obs1, obs2, obs3, obs4],
        }

        vehicle_traj = reconstruct_identity_trajectory(identity_data, self.graph)
        self.assertEqual(len(vehicle_traj.segments), 3)

        # Adapt full trajectory to Member 3
        norm_traj = adapt_vehicle_trajectory_to_normalized(vehicle_traj, vehicle_weight=1.0)
        self.assertEqual(norm_traj.track_id, "VEHICLE_CANDIDATE_001")
        self.assertEqual(norm_traj.origin_node, "J01")
        self.assertEqual(norm_traj.destination_node, "J08")
        self.assertEqual(norm_traj.time_window_start, 1000.0)
        self.assertEqual(norm_traj.time_window_end, 1400.0)

        # All candidate routes should span J01 to J08
        for r in norm_traj.candidate_routes:
            self.assertEqual(r.nodes[0], "J01")
            self.assertEqual(r.nodes[-1], "J08")
            self.assertIn("J02", r.nodes)
            self.assertIn("J05", r.nodes)

        # Probabilities sum to 1.0 within 1e-6
        total_p = sum(r.probability for r in norm_traj.candidate_routes)
        self.assertAlmostEqual(total_p, 1.0, places=6)

    def test_13_batch_payload_serialization_for_member3(self):
        """Verify adapt_trajectories_to_batch_payload outputs structure ingested by Member 3."""
        obs_a = {"observation_id": "o1", "camera_id": "CAM_J01", "timestamp": 1000.0}
        obs_b = {"observation_id": "o2", "camera_id": "CAM_J08", "timestamp": 1450.0}
        segment = reconstruct_trajectory_segment(obs_a, obs_b, self.graph, identity_id="TRK_001")

        payload = adapt_trajectories_to_batch_payload([segment], default_weight=1.0)
        self.assertIn("trajectories", payload)
        self.assertEqual(payload["trajectories_count"], 1)

        traj_dict = payload["trajectories"][0]
        self.assertEqual(traj_dict["track_id"], "TRK_001")
        self.assertEqual(traj_dict["origin_node"], "J01")
        self.assertEqual(traj_dict["destination_node"], "J08")
        self.assertEqual(traj_dict["vehicle_weight"], 1.0)
        self.assertIn("candidate_routes", traj_dict)
        self.assertIn("time_window", traj_dict)
        self.assertEqual(traj_dict["time_window"]["start"], 1000.0)
        self.assertEqual(traj_dict["time_window"]["end"], 1450.0)

    def test_14_schema_validation_rejections(self):
        """Verify strict validation rejects invalid routes, probabilities, and weight."""
        # Single-node route rejected
        with self.assertRaises(InvalidRouteError):
            NormalizedCandidateRoute(nodes=["J01"], probability=1.0)

        # Probability out of bounds
        with self.assertRaises(ProbabilityValidationError):
            NormalizedCandidateRoute(nodes=["J01", "J02"], probability=1.5)

        # Negative vehicle weight rejected
        valid_routes = [NormalizedCandidateRoute(nodes=["J01", "J02"], probability=1.0)]
        with self.assertRaises(ValueError):
            NormalizedTrajectory("TRK_ERR", "J01", "J02", valid_routes, vehicle_weight=-0.5)

        # Origin / destination endpoint mismatch rejected
        bad_endpoint_routes = [NormalizedCandidateRoute(nodes=["J02", "J03"], probability=1.0)]
        with self.assertRaises(InvalidRouteError):
            NormalizedTrajectory("TRK_ERR", "J01", "J03", bad_endpoint_routes)

    def test_15_member3_consumer_ingestion_and_demand_aggregation(self):
        """Verify Member 3's consumer can ingest our trajectory payload and compute route demand."""
        # Generate a realistic multi-candidate trajectory
        obs_a = {"camera_id": "CAM_J01", "timestamp": 1000.0}
        obs_b = {"camera_id": "CAM_J12", "timestamp": 1750.0}
        segment = reconstruct_trajectory_segment(obs_a, obs_b, self.graph, identity_id="TRK_M3_E2E", max_paths=3)
        norm_traj = adapt_trajectory_segment_to_normalized(segment, vehicle_weight=1.5)

        # Convert to serialized payload dictionary as passed across module boundary
        payload_dict = norm_traj.to_dict()

        # Ingestion simulation of Member 3 MockTrajectoryAdapter.adapt_one()
        ingested_track_id = str(payload_dict["track_id"])
        ingested_orig = str(payload_dict["origin_node"])
        ingested_dest = str(payload_dict["destination_node"])
        ingested_weight = float(payload_dict.get("vehicle_weight", 1.0))
        ingested_routes = payload_dict["candidate_routes"]

        self.assertEqual(ingested_track_id, "TRK_M3_E2E")
        self.assertEqual(ingested_orig, "J01")
        self.assertEqual(ingested_dest, "J12")
        self.assertEqual(ingested_weight, 1.5)
        self.assertGreaterEqual(len(ingested_routes), 2)

        # Downstream route demand calculation: route demand = vehicle_weight * route_probability
        aggregated_demand = 0.0
        for r in ingested_routes:
            self.assertIn("nodes", r)
            self.assertIn("probability", r)
            self.assertEqual(r["nodes"][0], ingested_orig)
            self.assertEqual(r["nodes"][-1], ingested_dest)
            route_demand = ingested_weight * r["probability"]
            aggregated_demand += route_demand

        self.assertAlmostEqual(aggregated_demand, 1.5, places=6)


if __name__ == "__main__":
    unittest.main()
