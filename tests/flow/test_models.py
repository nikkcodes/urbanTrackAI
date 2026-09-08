"""Unit tests for Phase 2 flow models and normalized trajectory validation."""

import unittest

from backend.flow.models import (
    CandidateRoute,
    FlowAggregationResult,
    InvalidRouteError,
    NormalizedTrajectory,
    ProbabilityValidationError,
    RoadFlow,
)


class TestCandidateRoute(unittest.TestCase):
    """Test suite for CandidateRoute model."""

    def test_valid_candidate_route(self) -> None:
        route = CandidateRoute(nodes=["J01", "J02", "J05"], probability=0.75)
        self.assertEqual(route.nodes, ["J01", "J02", "J05"])
        self.assertAlmostEqual(route.probability, 0.75)

    def test_fewer_than_two_nodes_rejected(self) -> None:
        with self.assertRaises(InvalidRouteError):
            CandidateRoute(nodes=["J01"], probability=1.0)
        with self.assertRaises(InvalidRouteError):
            CandidateRoute(nodes=[], probability=1.0)

    def test_empty_node_string_rejected(self) -> None:
        with self.assertRaises(InvalidRouteError):
            CandidateRoute(nodes=["J01", ""], probability=1.0)
        with self.assertRaises(InvalidRouteError):
            CandidateRoute(nodes=["J01", "   "], probability=1.0)

    def test_negative_probability_rejected(self) -> None:
        with self.assertRaises(ProbabilityValidationError):
            CandidateRoute(nodes=["J01", "J02"], probability=-0.05)

    def test_probability_greater_than_one_rejected(self) -> None:
        with self.assertRaises(ProbabilityValidationError):
            CandidateRoute(nodes=["J01", "J02"], probability=1.05)

    def test_non_finite_probability_rejected(self) -> None:
        with self.assertRaises(ProbabilityValidationError):
            CandidateRoute(nodes=["J01", "J02"], probability=float("nan"))
        with self.assertRaises(ProbabilityValidationError):
            CandidateRoute(nodes=["J01", "J02"], probability=float("inf"))

    def test_serialization_round_trip(self) -> None:
        route = CandidateRoute(nodes=["J01", "J02"], probability=0.4, metadata={"source": "heuristic"})
        data = route.to_dict()
        restored = CandidateRoute.from_dict(data)
        self.assertEqual(route.nodes, restored.nodes)
        self.assertAlmostEqual(route.probability, restored.probability)
        self.assertEqual(route.metadata, restored.metadata)


class TestNormalizedTrajectory(unittest.TestCase):
    """Test suite for NormalizedTrajectory model."""

    def test_valid_trajectory(self) -> None:
        r1 = CandidateRoute(nodes=["J01", "J02", "J05"], probability=0.7)
        r2 = CandidateRoute(nodes=["J01", "J04", "J05"], probability=0.3)
        traj = NormalizedTrajectory(
            track_id="T001",
            origin_node="J01",
            destination_node="J05",
            candidate_routes=[r1, r2],
            vehicle_weight=1.0,
        )
        self.assertEqual(traj.track_id, "T001")
        self.assertEqual(traj.origin_node, "J01")
        self.assertEqual(traj.destination_node, "J05")
        self.assertEqual(len(traj.candidate_routes), 2)
        self.assertAlmostEqual(traj.vehicle_weight, 1.0)

    def test_empty_track_id_rejected(self) -> None:
        r = CandidateRoute(nodes=["J01", "J02"], probability=1.0)
        with self.assertRaises(ValueError):
            NormalizedTrajectory(track_id="", origin_node="J01", destination_node="J02", candidate_routes=[r])

    def test_empty_candidate_routes_rejected(self) -> None:
        with self.assertRaises(ValueError):
            NormalizedTrajectory(track_id="T001", origin_node="J01", destination_node="J02", candidate_routes=[])

    def test_zero_or_negative_vehicle_weight_rejected(self) -> None:
        r = CandidateRoute(nodes=["J01", "J02"], probability=1.0)
        with self.assertRaises(ValueError):
            NormalizedTrajectory(track_id="T001", origin_node="J01", destination_node="J02", candidate_routes=[r], vehicle_weight=0.0)
        with self.assertRaises(ValueError):
            NormalizedTrajectory(track_id="T001", origin_node="J01", destination_node="J02", candidate_routes=[r], vehicle_weight=-1.5)

    def test_route_endpoints_mismatch_rejected(self) -> None:
        r1 = CandidateRoute(nodes=["J02", "J05"], probability=1.0)
        with self.assertRaises(InvalidRouteError):
            NormalizedTrajectory(track_id="T001", origin_node="J01", destination_node="J05", candidate_routes=[r1])

        r2 = CandidateRoute(nodes=["J01", "J04"], probability=1.0)
        with self.assertRaises(InvalidRouteError):
            NormalizedTrajectory(track_id="T001", origin_node="J01", destination_node="J05", candidate_routes=[r2])

    def test_probability_sum_deviation_rejected(self) -> None:
        # Sum = 0.90 (under 1.0)
        r1 = CandidateRoute(nodes=["J01", "J02", "J05"], probability=0.6)
        r2 = CandidateRoute(nodes=["J01", "J04", "J05"], probability=0.3)
        with self.assertRaises(ProbabilityValidationError):
            NormalizedTrajectory(track_id="T001", origin_node="J01", destination_node="J05", candidate_routes=[r1, r2])

        # Sum = 1.10 (over 1.0)
        r3 = CandidateRoute(nodes=["J01", "J02", "J05"], probability=0.7)
        r4 = CandidateRoute(nodes=["J01", "J04", "J05"], probability=0.4)
        with self.assertRaises(ProbabilityValidationError):
            NormalizedTrajectory(track_id="T001", origin_node="J01", destination_node="J05", candidate_routes=[r3, r4])

    def test_serialization_round_trip(self) -> None:
        r1 = CandidateRoute(nodes=["J01", "J02"], probability=1.0)
        traj = NormalizedTrajectory(
            track_id="T001",
            origin_node="J01",
            destination_node="J02",
            candidate_routes=[r1],
            vehicle_weight=2.0,
            time_window_start="08:00:00",
            time_window_end="08:15:00",
        )
        data = traj.to_dict()
        restored = NormalizedTrajectory.from_dict(data)
        self.assertEqual(traj.track_id, restored.track_id)
        self.assertEqual(traj.vehicle_weight, restored.vehicle_weight)
        self.assertEqual(traj.time_window_start, restored.time_window_start)
        self.assertEqual(len(restored.candidate_routes), 1)
        self.assertEqual(restored.candidate_routes[0].nodes, ["J01", "J02"])


class TestRoadFlowAndResult(unittest.TestCase):
    """Test suite for RoadFlow and FlowAggregationResult."""

    def test_valid_road_flow(self) -> None:
        rf = RoadFlow(road_id="R01", from_node="J01", to_node="J02", expected_flow=1.5, contributing_trajectories_count=2)
        self.assertEqual(rf.road_id, "R01")
        self.assertEqual(rf.expected_flow, 1.5)
        self.assertEqual(rf.contributing_trajectories_count, 2)

    def test_negative_flow_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RoadFlow(road_id="R01", from_node="J01", to_node="J02", expected_flow=-0.5)

    def test_flow_aggregation_result_helpers(self) -> None:
        rf1 = RoadFlow(road_id="R01", from_node="J01", to_node="J02", expected_flow=1.5)
        rf2 = RoadFlow(road_id="R02", from_node="J02", to_node="J03", expected_flow=0.8)
        result = FlowAggregationResult(flows={"R01": rf1, "R02": rf2})

        self.assertAlmostEqual(result.get_flow("R01"), 1.5)
        self.assertAlmostEqual(result.get_flow("R02"), 0.8)
        self.assertAlmostEqual(result.get_flow("R99"), 0.0)

        flows = result.all_flows(sorted_by_id=True)
        self.assertEqual(len(flows), 2)
        self.assertEqual(flows[0].road_id, "R01")
        self.assertEqual(flows[1].road_id, "R02")

        d = result.to_dict()
        self.assertEqual(d["total_roads_with_flow"], 2)
        self.assertIn("R01", d["flows"])


if __name__ == "__main__":
    unittest.main()
