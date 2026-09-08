"""Unit tests for ExpectedFlowAggregator.

Verifies the 14 required aggregation and validation behaviors:
1. Basic expected-flow calculation
2. Multiple trajectories sharing roads
3. Probability-weighted flow
4. Vehicle weight
5. Probability sum validation
6. Negative probability rejection
7. Probability > 1 rejection
8. Invalid node rejection
9. Invalid/non-connected route rejection
10. Node-path to road-ID mapping
11. Duplicate road within a route follows the documented rule
12. Optional time-window preservation
13. Empty trajectory input
14. Deterministic output
"""

import unittest
from pathlib import Path

from backend.flow.adapters import MockTrajectoryAdapter
from backend.flow.aggregation import ExpectedFlowAggregator
from backend.flow.models import (
    CandidateRoute,
    InvalidRouteError,
    NormalizedTrajectory,
    ProbabilityValidationError,
)
from backend.mobility.graph import MobilityGraph
from backend.mobility.models import Node, RoadSegment

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class TestExpectedFlowAggregator(unittest.TestCase):
    """Test suite verifying ExpectedFlowAggregator requirements."""

    def setUp(self) -> None:
        """Construct a controlled miniature network for exact analytical assertions.

        Network topology:
            J01 ---[R01]---> J02 ---[R02]---> J05
             |                                 ^
           [R07]                             [R18]
             v                                 |
            J04 -------------------------------+
             |
           [R11]
             v
            J08

        Also R28 goes J02 -> J01 (bidirectional support between J01 and J02 for loop testing).
        """
        self.graph = MobilityGraph(name="TestNetwork")
        for node_id in ["J01", "J02", "J04", "J05", "J08"]:
            self.graph.add_node(Node(node_id=node_id, name=f"Node {node_id}"))

        self.graph.add_road(RoadSegment(road_id="R01", from_node="J01", to_node="J02", distance_km=1.5, speed_limit_kmph=50, capacity_vph=1500))
        self.graph.add_road(RoadSegment(road_id="R02", from_node="J02", to_node="J05", distance_km=2.0, speed_limit_kmph=50, capacity_vph=1500))
        self.graph.add_road(RoadSegment(road_id="R07", from_node="J01", to_node="J04", distance_km=1.8, speed_limit_kmph=45, capacity_vph=1200))
        self.graph.add_road(RoadSegment(road_id="R18", from_node="J04", to_node="J05", distance_km=1.4, speed_limit_kmph=40, capacity_vph=1100))
        self.graph.add_road(RoadSegment(road_id="R11", from_node="J04", to_node="J08", distance_km=1.9, speed_limit_kmph=40, capacity_vph=1200))
        self.graph.add_road(RoadSegment(road_id="R28", from_node="J02", to_node="J01", distance_km=1.5, speed_limit_kmph=50, capacity_vph=1500))

        self.aggregator = ExpectedFlowAggregator(self.graph)

    # 1. Basic expected-flow calculation
    def test_01_basic_expected_flow(self) -> None:
        """Test basic single trajectory expected flow calculation."""
        route = CandidateRoute(nodes=["J01", "J02", "J05"], probability=1.0)
        traj = NormalizedTrajectory(
            track_id="T01",
            origin_node="J01",
            destination_node="J05",
            candidate_routes=[route],
            vehicle_weight=1.0,
        )

        result = self.aggregator.aggregate([traj])
        self.assertAlmostEqual(result.get_flow("R01"), 1.0)
        self.assertAlmostEqual(result.get_flow("R02"), 1.0)
        self.assertAlmostEqual(result.get_flow("R07"), 0.0)

        # Check endpoints and trajectory counts
        rf_r01 = result.get_road_flow("R01")
        self.assertIsNotNone(rf_r01)
        self.assertEqual(rf_r01.from_node, "J01")
        self.assertEqual(rf_r01.to_node, "J02")
        self.assertEqual(rf_r01.contributing_trajectories_count, 1)

    # 2. Multiple trajectories sharing roads
    def test_02_multiple_trajectories_sharing_roads(self) -> None:
        """Test multiple distinct trajectories traversing the same road segment."""
        # Trajectory 1: J01 -> J02 -> J05
        t1 = NormalizedTrajectory(
            track_id="T01",
            origin_node="J01",
            destination_node="J05",
            candidate_routes=[CandidateRoute(nodes=["J01", "J02", "J05"], probability=1.0)],
            vehicle_weight=1.0,
        )
        # Trajectory 2: J01 -> J02 (origin J01, dest J02)
        t2 = NormalizedTrajectory(
            track_id="T02",
            origin_node="J01",
            destination_node="J02",
            candidate_routes=[CandidateRoute(nodes=["J01", "J02"], probability=1.0)],
            vehicle_weight=1.0,
        )

        result = self.aggregator.aggregate([t1, t2])
        # R01 is shared by T01 and T02 -> expected flow = 1.0 + 1.0 = 2.0
        self.assertAlmostEqual(result.get_flow("R01"), 2.0)
        # R02 is only traversed by T01 -> expected flow = 1.0
        self.assertAlmostEqual(result.get_flow("R02"), 1.0)

        rf_r01 = result.get_road_flow("R01")
        self.assertEqual(rf_r01.contributing_trajectories_count, 2)

    # 3. Probability-weighted flow
    def test_03_probability_weighted_flow(self) -> None:
        """Test that probability weights split vehicle flow across routes without collapsing."""
        # T01 splits 70% Route A (J01->J02->J05) and 30% Route B (J01->J04->J05)
        rA = CandidateRoute(nodes=["J01", "J02", "J05"], probability=0.70)
        rB = CandidateRoute(nodes=["J01", "J04", "J05"], probability=0.30)
        t1 = NormalizedTrajectory(
            track_id="T01",
            origin_node="J01",
            destination_node="J05",
            candidate_routes=[rA, rB],
            vehicle_weight=1.0,
        )

        # T02 splits 80% Route A and 20% Route B
        rA2 = CandidateRoute(nodes=["J01", "J02", "J05"], probability=0.80)
        rB2 = CandidateRoute(nodes=["J01", "J04", "J05"], probability=0.20)
        t2 = NormalizedTrajectory(
            track_id="T02",
            origin_node="J01",
            destination_node="J05",
            candidate_routes=[rA2, rB2],
            vehicle_weight=1.0,
        )

        result = self.aggregator.aggregate([t1, t2])

        # Route A uses R01 and R02: expected flow = 0.70 + 0.80 = 1.50
        self.assertAlmostEqual(result.get_flow("R01"), 1.50)
        self.assertAlmostEqual(result.get_flow("R02"), 1.50)

        # Route B uses R07 and R18: expected flow = 0.30 + 0.20 = 0.50
        self.assertAlmostEqual(result.get_flow("R07"), 0.50)
        self.assertAlmostEqual(result.get_flow("R18"), 0.50)

    # 4. Vehicle weight
    def test_04_vehicle_weight(self) -> None:
        """Test that vehicle_weight scales flow contribution by weight * probability."""
        rA = CandidateRoute(nodes=["J01", "J02", "J05"], probability=0.60)
        rB = CandidateRoute(nodes=["J01", "J04", "J05"], probability=0.40)
        traj = NormalizedTrajectory(
            track_id="T_CONVOY",
            origin_node="J01",
            destination_node="J05",
            candidate_routes=[rA, rB],
            vehicle_weight=2.5,
        )

        result = self.aggregator.aggregate([traj])
        # R01: 2.5 * 0.60 = 1.50
        self.assertAlmostEqual(result.get_flow("R01"), 1.50)
        self.assertAlmostEqual(result.get_flow("R02"), 1.50)
        # R07: 2.5 * 0.40 = 1.00
        self.assertAlmostEqual(result.get_flow("R07"), 1.00)
        self.assertAlmostEqual(result.get_flow("R18"), 1.00)

    # 5. Probability sum validation
    def test_05_probability_sum_validation(self) -> None:
        """Verify that candidate route probabilities summing to != 1.0 raises validation error."""
        r1 = CandidateRoute(nodes=["J01", "J02", "J05"], probability=0.50)
        r2 = CandidateRoute(nodes=["J01", "J04", "J05"], probability=0.40)
        # Sum = 0.90
        with self.assertRaises(ProbabilityValidationError):
            NormalizedTrajectory(
                track_id="T_BAD_SUM",
                origin_node="J01",
                destination_node="J05",
                candidate_routes=[r1, r2],
            )

    # 6. Negative probability rejection
    def test_06_negative_probability_rejection(self) -> None:
        """Verify negative candidate route probability is rejected."""
        with self.assertRaises(ProbabilityValidationError):
            CandidateRoute(nodes=["J01", "J02"], probability=-0.1)

    # 7. Probability > 1 rejection
    def test_07_probability_greater_than_one_rejection(self) -> None:
        """Verify candidate route probability > 1.0 is rejected."""
        with self.assertRaises(ProbabilityValidationError):
            CandidateRoute(nodes=["J01", "J02"], probability=1.001)

    # 8. Invalid node rejection
    def test_08_invalid_node_rejection(self) -> None:
        """Verify route referencing a non-existent graph node raises an error during validation."""
        route = CandidateRoute(nodes=["J01", "J99", "J05"], probability=1.0)
        traj = NormalizedTrajectory(
            track_id="T_UNKNOWN_NODE",
            origin_node="J01",
            destination_node="J05",
            candidate_routes=[route],
        )

        with self.assertRaises(InvalidRouteError) as ctx:
            self.aggregator.aggregate([traj])
        self.assertIn("J99", str(ctx.exception))

    # 9. Invalid/non-connected route rejection
    def test_09_invalid_non_connected_route_rejection(self) -> None:
        """Verify route with unconnected graph nodes raises InvalidRouteError."""
        # J02 -> J04 has no directed road in our test network
        route = CandidateRoute(nodes=["J01", "J02", "J04", "J05"], probability=1.0)
        traj = NormalizedTrajectory(
            track_id="T_DISCONNECTED",
            origin_node="J01",
            destination_node="J05",
            candidate_routes=[route],
        )

        with self.assertRaises(InvalidRouteError) as ctx:
            self.aggregator.aggregate([traj])
        self.assertIn("J02 -> J04", str(ctx.exception))

    # 10. Node-path to road-ID mapping
    def test_10_node_path_to_road_id_mapping(self) -> None:
        """Verify node sequences map dynamically and accurately to road segment IDs."""
        route = CandidateRoute(nodes=["J01", "J04", "J08"], probability=1.0)
        road_ids = self.aggregator.map_route_to_roads(route)
        self.assertEqual(road_ids, ["R07", "R11"])

    # 11. Duplicate road within a route follows the documented rule
    def test_11_duplicate_road_within_route_follows_documented_rule(self) -> None:
        """Verify that traversing a road more than once in a route counts the road ONCE per route."""
        # Path with cycle: J01 --(R01)--> J02 --(R28)--> J01 --(R01)--> J02 --(R02)--> J05
        # R01 is traversed twice in this path!
        route = CandidateRoute(nodes=["J01", "J02", "J01", "J02", "J05"], probability=1.0)
        traj = NormalizedTrajectory(
            track_id="T_LOOP",
            origin_node="J01",
            destination_node="J05",
            candidate_routes=[route],
            vehicle_weight=1.0,
        )

        result = self.aggregator.aggregate([traj])
        # Per documented rule, R01 is counted once: expected flow = 1.0 (not 2.0)
        self.assertAlmostEqual(result.get_flow("R01"), 1.0)
        self.assertAlmostEqual(result.get_flow("R28"), 1.0)
        self.assertAlmostEqual(result.get_flow("R02"), 1.0)

    # 12. Optional time-window preservation
    def test_12_optional_time_window_preservation(self) -> None:
        """Verify time windows are preserved and grouping by time window works correctly."""
        t1 = NormalizedTrajectory(
            track_id="T01",
            origin_node="J01",
            destination_node="J05",
            candidate_routes=[CandidateRoute(nodes=["J01", "J02", "J05"], probability=1.0)],
            time_window_start="08:00:00",
            time_window_end="08:15:00",
        )
        t2 = NormalizedTrajectory(
            track_id="T02",
            origin_node="J01",
            destination_node="J05",
            candidate_routes=[CandidateRoute(nodes=["J01", "J04", "J05"], probability=1.0)],
            time_window_start="08:15:00",
            time_window_end="08:30:00",
        )

        # Aggregate individual batch with time_window set
        res1 = self.aggregator.aggregate([t1])
        self.assertEqual(res1.time_window_start, "08:00:00")
        self.assertEqual(res1.time_window_end, "08:15:00")
        self.assertEqual(res1.get_road_flow("R01").time_window_start, "08:00:00")

        # Aggregate partitioned by time window
        windowed_results = self.aggregator.aggregate_by_time_window([t1, t2])
        self.assertIn(("08:00:00", "08:15:00"), windowed_results)
        self.assertIn(("08:15:00", "08:30:00"), windowed_results)

        w1_res = windowed_results[("08:00:00", "08:15:00")]
        self.assertAlmostEqual(w1_res.get_flow("R01"), 1.0)
        self.assertAlmostEqual(w1_res.get_flow("R07"), 0.0)

        w2_res = windowed_results[("08:15:00", "08:30:00")]
        self.assertAlmostEqual(w2_res.get_flow("R01"), 0.0)
        self.assertAlmostEqual(w2_res.get_flow("R07"), 1.0)

    # 13. Empty trajectory input
    def test_13_empty_trajectory_input(self) -> None:
        """Verify empty trajectory list returns valid empty result without crashing."""
        result = self.aggregator.aggregate([])
        self.assertEqual(len(result.flows), 0)
        self.assertEqual(result.get_flow("R01"), 0.0)

        # With include_zero_flow_roads=True
        result_full = self.aggregator.aggregate([], include_zero_flow_roads=True)
        self.assertEqual(len(result_full.flows), self.graph.road_count)
        self.assertAlmostEqual(result_full.get_flow("R01"), 0.0)

    # 14. Deterministic output
    def test_14_deterministic_output(self) -> None:
        """Verify flow aggregation results are deterministic across multiple runs and shuffles."""
        t1 = NormalizedTrajectory(
            track_id="T01",
            origin_node="J01",
            destination_node="J05",
            candidate_routes=[
                CandidateRoute(nodes=["J01", "J02", "J05"], probability=0.6),
                CandidateRoute(nodes=["J01", "J04", "J05"], probability=0.4),
            ],
            vehicle_weight=1.0,
        )
        t2 = NormalizedTrajectory(
            track_id="T02",
            origin_node="J01",
            destination_node="J05",
            candidate_routes=[CandidateRoute(nodes=["J01", "J02", "J05"], probability=1.0)],
            vehicle_weight=1.5,
        )

        res1 = self.aggregator.aggregate([t1, t2])
        res2 = self.aggregator.aggregate([t2, t1])

        flows1 = [(rf.road_id, rf.expected_flow) for rf in res1.all_flows(sorted_by_id=True)]
        flows2 = [(rf.road_id, rf.expected_flow) for rf in res2.all_flows(sorted_by_id=True)]
        self.assertEqual(flows1, flows2)

    def test_synthetic_city_network_and_mock_trajectories_integration(self) -> None:
        """End-to-end test loading the full synthetic city network and mock trajectories fixture."""
        city_json = PROJECT_ROOT / "data" / "synthetic" / "city_network.json"
        mock_json = PROJECT_ROOT / "data" / "synthetic" / "mock_trajectories.json"

        city_graph = MobilityGraph.load_from_json(city_json)
        adapter = MockTrajectoryAdapter()
        trajectories = adapter.adapt(mock_json)

        city_aggregator = ExpectedFlowAggregator(city_graph)
        result = city_aggregator.aggregate(trajectories, include_zero_flow_roads=True)

        # Verify that all 28 synthetic roads are represented
        self.assertEqual(len(result.flows), city_graph.road_count)

        # Check that active roads have expected flows > 0
        self.assertGreater(result.get_flow("R01"), 0.0)
        self.assertGreater(result.get_flow("R02"), 0.0)
        self.assertGreater(result.get_flow("R13"), 0.0)

        # Check trajectory retention for Phase 4
        self.assertEqual(len(result.trajectories), len(trajectories))


if __name__ == "__main__":
    unittest.main()
