"""
Member 3 Integration Contract Test Suite.

Validates that Day 3 Trajectory Reconstruction outputs cleanly adhere to
the contract expected by Member 3's Mobility Graph, Flow Aggregation,
Traffic Analytics, and Decision Intelligence engines.

Contract Verified:
NormalizedTrajectory
├── track_id
├── origin_node
├── destination_node
├── vehicle_weight
├── candidate_routes[]
│   ├── nodes
│   └── probability
└── time window
"""

import math
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
from schemas.observation_schema import Observation
from schemas.trajectory_schema import CandidateRoute, TrajectorySegment, VehicleTrajectory


class TestMember3Integration(unittest.TestCase):
    """Test suite for Member 3 integration schema, adapter layer, and contract validation."""

    def setUp(self):
        """Set up road network and sample camera observations."""
        self.graph = RoadGraph(metadata={"name": "integration_test_network"})

        # Nodes
        self.graph.add_node(RoadNode(node_id="J1", name="Junction 1", latitude=17.3850, longitude=78.4867))
        self.graph.add_node(RoadNode(node_id="J2", name="Junction 2", latitude=17.3880, longitude=78.4867))
        self.graph.add_node(RoadNode(node_id="J3", name="Junction 3", latitude=17.3850, longitude=78.4900))
        self.graph.add_node(RoadNode(node_id="J4", name="Junction 4", latitude=17.3880, longitude=78.4900))

        # Roads from J1 to J4
        # Path A (via J2): J1 -> J2 -> J4 (700m)
        self.graph.add_edge(RoadEdge(road_id="r12", from_node="J1", to_node="J2", distance_m=350.0, name="Road 1-2", speed_limit_kmh=50.0, one_way=True))
        self.graph.add_edge(RoadEdge(road_id="r24", from_node="J2", to_node="J4", distance_m=350.0, name="Road 2-4", speed_limit_kmh=50.0, one_way=True))

        # Path B (via J3): J1 -> J3 -> J4 (1000m)
        self.graph.add_edge(RoadEdge(road_id="r13", from_node="J1", to_node="J3", distance_m=400.0, name="Road 1-3", speed_limit_kmh=80.0, one_way=True))
        self.graph.add_edge(RoadEdge(road_id="r34", from_node="J3", to_node="J4", distance_m=600.0, name="Road 3-4", speed_limit_kmh=80.0, one_way=True))

        # Camera associations
        self.graph.camera_associations["CAM_01"] = "J1"
        self.graph.camera_associations["CAM_02"] = "J2"
        self.graph.camera_associations["CAM_03"] = "J3"
        self.graph.camera_associations["CAM_04"] = "J4"

    # =========================================================================
    # 1. SCHEMA VALIDATION TESTS
    # =========================================================================

    def test_normalized_candidate_route_valid(self):
        """Test valid creation of NormalizedCandidateRoute."""
        route = NormalizedCandidateRoute(
            nodes=["J1", "J2", "J4"],
            probability=0.75,
            metadata={"distance_m": 700.0},
        )
        self.assertEqual(route.nodes, ["J1", "J2", "J4"])
        self.assertEqual(route.probability, 0.75)
        self.assertEqual(route.metadata["distance_m"], 700.0)

    def test_normalized_candidate_route_invalid_nodes(self):
        """Test that candidate route must contain at least 2 valid string nodes."""
        with self.assertRaises(InvalidRouteError):
            NormalizedCandidateRoute(nodes=["J1"], probability=1.0)

        with self.assertRaises(InvalidRouteError):
            NormalizedCandidateRoute(nodes=[], probability=1.0)

        with self.assertRaises(InvalidRouteError):
            NormalizedCandidateRoute(nodes=["J1", ""], probability=1.0)

        with self.assertRaises(InvalidRouteError):
            NormalizedCandidateRoute(nodes=["J1", None], probability=1.0)

    def test_normalized_candidate_route_invalid_probability(self):
        """Test probability bounds [0.0, 1.0] and finiteness."""
        with self.assertRaises(ProbabilityValidationError):
            NormalizedCandidateRoute(nodes=["J1", "J2"], probability=-0.1)

        with self.assertRaises(ProbabilityValidationError):
            NormalizedCandidateRoute(nodes=["J1", "J2"], probability=1.05)

        with self.assertRaises(ProbabilityValidationError):
            NormalizedCandidateRoute(nodes=["J1", "J2"], probability=float("nan"))

        with self.assertRaises(ProbabilityValidationError):
            NormalizedCandidateRoute(nodes=["J1", "J2"], probability=float("inf"))

    def test_normalized_trajectory_valid(self):
        """Test valid creation and properties of NormalizedTrajectory."""
        routes = [
            NormalizedCandidateRoute(nodes=["J1", "J2", "J4"], probability=0.70),
            NormalizedCandidateRoute(nodes=["J1", "J3", "J4"], probability=0.30),
        ]
        traj = NormalizedTrajectory(
            track_id="VEH_001",
            origin_node="J1",
            destination_node="J4",
            candidate_routes=routes,
            vehicle_weight=1.0,
            time_window_start="2026-03-03T10:00:00Z",
            time_window_end="2026-03-03T10:02:00Z",
        )
        self.assertEqual(traj.track_id, "VEH_001")
        self.assertEqual(traj.origin_node, "J1")
        self.assertEqual(traj.destination_node, "J4")
        self.assertEqual(traj.vehicle_weight, 1.0)
        self.assertEqual(len(traj.candidate_routes), 2)

    def test_endpoint_consistency_validation(self):
        """Verify that every candidate route must start at origin and end at destination."""
        # Start mismatch
        invalid_start_routes = [
            NormalizedCandidateRoute(nodes=["J2", "J4"], probability=1.0),
        ]
        with self.assertRaises(InvalidRouteError):
            NormalizedTrajectory(
                track_id="VEH_001",
                origin_node="J1",
                destination_node="J4",
                candidate_routes=invalid_start_routes,
            )

        # End mismatch
        invalid_end_routes = [
            NormalizedCandidateRoute(nodes=["J1", "J2"], probability=1.0),
        ]
        with self.assertRaises(InvalidRouteError):
            NormalizedTrajectory(
                track_id="VEH_001",
                origin_node="J1",
                destination_node="J4",
                candidate_routes=invalid_end_routes,
            )

    def test_probability_sum_validation(self):
        """Verify that candidate route probabilities must sum to 1.0."""
        # Sums to 0.85 (invalid)
        bad_sum_routes = [
            NormalizedCandidateRoute(nodes=["J1", "J2", "J4"], probability=0.60),
            NormalizedCandidateRoute(nodes=["J1", "J3", "J4"], probability=0.25),
        ]
        with self.assertRaises(ProbabilityValidationError):
            NormalizedTrajectory(
                track_id="VEH_001",
                origin_node="J1",
                destination_node="J4",
                candidate_routes=bad_sum_routes,
            )

        # Sums to 1.00005 (valid within 1e-4 tolerance)
        acceptable_routes = [
            NormalizedCandidateRoute(nodes=["J1", "J2", "J4"], probability=0.70002),
            NormalizedCandidateRoute(nodes=["J1", "J3", "J4"], probability=0.29999),
        ]
        traj = NormalizedTrajectory(
            track_id="VEH_001",
            origin_node="J1",
            destination_node="J4",
            candidate_routes=acceptable_routes,
        )
        self.assertIsNotNone(traj)

    def test_vehicle_weight_validation(self):
        """Verify vehicle_weight validation (non-negative, finite)."""
        routes = [NormalizedCandidateRoute(nodes=["J1", "J4"], probability=1.0)]
        with self.assertRaises(ValueError):
            NormalizedTrajectory("VEH_001", "J1", "J4", routes, vehicle_weight=-1.0)

        with self.assertRaises(ValueError):
            NormalizedTrajectory("VEH_001", "J1", "J4", routes, vehicle_weight=float("nan"))

    # =========================================================================
    # 2. DEMAND WEIGHTING TESTS (W * P)
    # =========================================================================

    def test_demand_calculation_default_weight(self):
        """Verify demand calculation with default vehicle weight (1.0)."""
        routes = [
            NormalizedCandidateRoute(nodes=["J1", "J2", "J4"], probability=0.7),
            NormalizedCandidateRoute(nodes=["J1", "J3", "J4"], probability=0.3),
        ]
        traj = NormalizedTrajectory(
            track_id="VEH_001",
            origin_node="J1",
            destination_node="J4",
            candidate_routes=routes,
            vehicle_weight=1.0,
        )
        demands = traj.calculate_route_demands()
        self.assertEqual(len(demands), 2)
        self.assertAlmostEqual(demands[0]["route_demand"], 0.7, places=4)
        self.assertAlmostEqual(demands[1]["route_demand"], 0.3, places=4)
        total_demand = sum(d["route_demand"] for d in demands)
        self.assertAlmostEqual(total_demand, 1.0, places=4)

    def test_demand_calculation_heavy_vehicle(self):
        """Verify demand calculation with custom vehicle weight (e.g. Bus/Truck = 2.5 PCU)."""
        routes = [
            NormalizedCandidateRoute(nodes=["J1", "J2", "J4"], probability=0.8),
            NormalizedCandidateRoute(nodes=["J1", "J3", "J4"], probability=0.2),
        ]
        traj = NormalizedTrajectory(
            track_id="TRUCK_001",
            origin_node="J1",
            destination_node="J4",
            candidate_routes=routes,
            vehicle_weight=2.5,
        )
        demands = traj.calculate_route_demands()
        # 2.5 * 0.8 = 2.0
        self.assertAlmostEqual(demands[0]["route_demand"], 2.0, places=4)
        # 2.5 * 0.2 = 0.5
        self.assertAlmostEqual(demands[1]["route_demand"], 0.5, places=4)
        total_demand = sum(d["route_demand"] for d in demands)
        self.assertAlmostEqual(total_demand, 2.5, places=4)

    # =========================================================================
    # 3. ADAPTER FUNCTION TESTS
    # =========================================================================

    def test_adapt_trajectory_segment_to_normalized(self):
        """Verify adapting a Day 3 TrajectorySegment into a NormalizedTrajectory."""
        obs1 = {
            "observation_id": "obs_01",
            "camera_id": "CAM_01",
            "timestamp": 100.0,
            "vehicle_type": "car",
            "plate": "KA01AB1234",
            "plate_confidence": 0.95,
        }
        # Travel 700m at ~42 km/h -> 60s
        obs2 = {
            "observation_id": "obs_02",
            "camera_id": "CAM_04",
            "timestamp": 160.0,
            "vehicle_type": "car",
            "plate": "KA01AB1234",
            "plate_confidence": 0.95,
        }

        segment = reconstruct_trajectory_segment(obs1, obs2, self.graph, identity_id="ID_001")
        self.assertEqual(segment.status, "success")
        self.assertGreater(len(segment.candidate_routes), 0)

        # Adapt to Member 3 contract
        norm_traj = adapt_trajectory_segment_to_normalized(segment, vehicle_weight=1.0)

        self.assertEqual(norm_traj.track_id, "ID_001")
        self.assertEqual(norm_traj.origin_node, "J1")
        self.assertEqual(norm_traj.destination_node, "J4")
        self.assertEqual(norm_traj.vehicle_weight, 1.0)
        self.assertGreaterEqual(len(norm_traj.candidate_routes), 1)

        # Check endpoints and probabilities
        for r in norm_traj.candidate_routes:
            self.assertEqual(r.nodes[0], "J1")
            self.assertEqual(r.nodes[-1], "J4")
            self.assertGreater(r.probability, 0.0)

        total_prob = sum(r.probability for r in norm_traj.candidate_routes)
        self.assertAlmostEqual(total_prob, 1.0, places=4)

    def test_adapt_trajectory_segment_unassociated_camera(self):
        """Verify adapter raises ValueError if segment cameras are unassociated."""
        obs1 = {
            "observation_id": "obs_01",
            "camera_id": "UNKNOWN_CAM",
            "timestamp": 100.0,
            "vehicle_type": "car",
        }
        obs2 = {
            "observation_id": "obs_02",
            "camera_id": "CAM_04",
            "timestamp": 160.0,
            "vehicle_type": "car",
        }
        segment = reconstruct_trajectory_segment(obs1, obs2, self.graph, identity_id="ID_001")
        self.assertEqual(segment.status, "unassociated_camera")

        with self.assertRaises(ValueError) as ctx:
            adapt_trajectory_segment_to_normalized(segment)
        self.assertIn("Unassociated", str(ctx.exception))

    def test_adapt_vehicle_trajectory_multi_segment(self):
        """Verify multi-observation journey (J1 -> J2 -> J4) projects complete origin-to-destination corridor."""
        obs1 = {
            "observation_id": "obs_01",
            "camera_id": "CAM_01",
            "timestamp": 100.0,
            "vehicle_type": "car",
            "bounding_box": [0, 0, 10, 10],
            "plate_number": "DL01XY9999",
        }
        # 350m at 35 km/h = 36s
        obs2 = {
            "observation_id": "obs_02",
            "camera_id": "CAM_02",
            "timestamp": 136.0,
            "vehicle_type": "car",
            "bounding_box": [0, 0, 10, 10],
            "plate_number": "DL01XY9999",
        }
        # 350m at 35 km/h = 36s
        obs3 = {
            "observation_id": "obs_03",
            "camera_id": "CAM_04",
            "timestamp": 172.0,
            "vehicle_type": "car",
            "bounding_box": [0, 0, 10, 10],
            "plate_number": "DL01XY9999",
        }

        identity_data = {
            "identity_id": "ID_MULTI",
            "observations": [obs1, obs2, obs3],
        }
        vehicle_traj = reconstruct_identity_trajectory(identity_data, self.graph)
        self.assertEqual(len(vehicle_traj.segments), 2)

        # Adapt complete trajectory to Member 3
        norm_traj = adapt_vehicle_trajectory_to_normalized(vehicle_traj, vehicle_weight=1.5)

        self.assertEqual(norm_traj.track_id, "ID_MULTI")
        self.assertEqual(norm_traj.origin_node, "J1")
        self.assertEqual(norm_traj.destination_node, "J4")
        self.assertEqual(norm_traj.vehicle_weight, 1.5)

        # All candidate routes should span J1 to J4
        for cr in norm_traj.candidate_routes:
            self.assertEqual(cr.nodes[0], "J1")
            self.assertEqual(cr.nodes[-1], "J4")
            self.assertIn("J2", cr.nodes)

        total_prob = sum(cr.probability for cr in norm_traj.candidate_routes)
        self.assertAlmostEqual(total_prob, 1.0, places=4)

    def test_adapt_trajectories_to_batch_payload(self):
        """Verify batch payload serialization for Member 3's flow aggregator."""
        obs1 = {"observation_id": "o1", "camera_id": "CAM_01", "timestamp": 100.0, "vehicle_type": "car"}
        obs2 = {"observation_id": "o2", "camera_id": "CAM_04", "timestamp": 160.0, "vehicle_type": "car"}
        seg = reconstruct_trajectory_segment(obs1, obs2, self.graph, identity_id="ID_01")

        payload = adapt_trajectories_to_batch_payload([seg], default_weight=1.0)

        self.assertIn("trajectories", payload)
        self.assertEqual(payload["trajectories_count"], 1)
        traj_dict = payload["trajectories"][0]

        # Verify contract keys expected by Member 3
        self.assertEqual(traj_dict["track_id"], "ID_01")
        self.assertEqual(traj_dict["origin_node"], "J1")
        self.assertEqual(traj_dict["destination_node"], "J4")
        self.assertEqual(traj_dict["vehicle_weight"], 1.0)
        self.assertIn("candidate_routes", traj_dict)
        self.assertIn("time_window", traj_dict)
        self.assertEqual(traj_dict["time_window"]["start"], 100.0)
        self.assertEqual(traj_dict["time_window"]["end"], 160.0)

        for route in traj_dict["candidate_routes"]:
            self.assertIn("nodes", route)
            self.assertIn("probability", route)
            self.assertIsInstance(route["nodes"], list)
            self.assertIsInstance(route["probability"], float)

    # =========================================================================
    # 4. CONTRACT SERIALIZATION & DESERIALIZATION ROUND TRIP
    # =========================================================================

    def test_serialization_round_trip(self):
        """Verify that NormalizedTrajectory serializes to dict and deserializes accurately."""
        routes = [
            NormalizedCandidateRoute(nodes=["J1", "J2", "J4"], probability=0.65, metadata={"dist": 700}),
            NormalizedCandidateRoute(nodes=["J1", "J3", "J4"], probability=0.35, metadata={"dist": 1000}),
        ]
        original = NormalizedTrajectory(
            track_id="TRIP_123",
            origin_node="J1",
            destination_node="J4",
            candidate_routes=routes,
            vehicle_weight=2.0,
            time_window_start="2026-03-03T12:00:00Z",
            time_window_end="2026-03-03T12:05:00Z",
            metadata={"driver": "test"},
        )

        d = original.to_dict()
        reconstructed = NormalizedTrajectory.from_dict(d)

        self.assertEqual(reconstructed.track_id, original.track_id)
        self.assertEqual(reconstructed.origin_node, original.origin_node)
        self.assertEqual(reconstructed.destination_node, original.destination_node)
        self.assertEqual(reconstructed.vehicle_weight, original.vehicle_weight)
        self.assertEqual(len(reconstructed.candidate_routes), len(original.candidate_routes))
        self.assertEqual(reconstructed.candidate_routes[0].nodes, original.candidate_routes[0].nodes)
        self.assertEqual(reconstructed.candidate_routes[0].probability, original.candidate_routes[0].probability)


if __name__ == "__main__":
    unittest.main()
