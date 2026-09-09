"""Tests for Phase 4 OD and probabilistic route demand analytics."""

import unittest

from backend.analytics.od import ODAnalyzer
from backend.analytics.models import ODMatrix, ODPairDemand
from backend.flow.aggregation import ExpectedFlowAggregator
from backend.flow.models import CandidateRoute, FlowAggregationResult, NormalizedTrajectory
from backend.mobility.graph import MobilityGraph
from backend.mobility.models import RoadSegment


class TestODAnalyzer(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = MobilityGraph(name="ODTest")
        self.graph.add_road(RoadSegment("R01", "J01", "J02", 1.0, 40.0, 1000.0))
        self.graph.add_road(RoadSegment("R02", "J02", "J10", 1.0, 40.0, 1000.0))
        self.graph.add_road(RoadSegment("R03", "J01", "J03", 1.0, 40.0, 1000.0))
        self.graph.add_road(RoadSegment("R04", "J03", "J10", 1.0, 40.0, 1000.0))
        self.analyzer = ODAnalyzer()

    def trajectory(self, track_id: str, destination: str, weight: float, routes: list[CandidateRoute], start: str = "08:00:00", end: str = "08:15:00") -> NormalizedTrajectory:
        return NormalizedTrajectory(
            track_id=track_id,
            origin_node="J01",
            destination_node=destination,
            candidate_routes=routes,
            vehicle_weight=weight,
            time_window_start=start,
            time_window_end=end,
        )

    def test_accumulated_od_demand_uses_trajectory_weight_once(self) -> None:
        trajectories = [
            self.trajectory("T1", "J10", 1.0, [CandidateRoute(["J01", "J02", "J10"], 1.0)]),
            self.trajectory("T2", "J10", 1.5, [CandidateRoute(["J01", "J02", "J10"], 1.0)]),
            self.trajectory("T3", "J03", 1.0, [CandidateRoute(["J01", "J03"], 1.0)], start="08:15:00", end="08:30:00"),
        ]
        result = self.analyzer.analyze_trajectories(trajectories)

        self.assertEqual(result.trajectory_count, 3)
        self.assertAlmostEqual(result.total_demand, 3.5)
        self.assertAlmostEqual(result.matrix.get_demand("J01", "J10", ("08:00:00", "08:15:00")), 2.5)
        self.assertAlmostEqual(result.matrix.get_demand("J01", "J03", ("08:15:00", "08:30:00")), 1.0)
        self.assertEqual(result.top_od_pairs[0].demand, 2.5)

    def test_empty_input_and_zero_demand_record(self) -> None:
        empty = self.analyzer.analyze_trajectories([])
        self.assertEqual(empty.total_demand, 0.0)
        self.assertEqual(empty.matrix.origins, ())

        zero = ODMatrix((ODPairDemand("J01", "J10", 0.0),))
        self.assertEqual(zero.total_demand, 0.0)
        self.assertEqual(zero.get_demand("J01", "J10"), 0.0)

    def test_invalid_od_and_windows_rejected(self) -> None:
        with self.assertRaises(ValueError):
            NormalizedTrajectory("T", "", "J10", [CandidateRoute(["J01", "J02", "J10"], 1.0)])
        with self.assertRaises(ValueError):
            self.analyzer.analyze_trajectories([
                self.trajectory(
                    "T",
                    "J10",
                    1.0,
                    [CandidateRoute(["J01", "J02", "J10"], 1.0)],
                    "08:15:00",
                    "08:00:00",
                )
            ])

    def test_probabilistic_route_demand_is_weighted_without_collapsing(self) -> None:
        trajectories = [
            self.trajectory(
                "T1",
                "J10",
                2.0,
                [
                    CandidateRoute(["J01", "J02", "J10"], 0.7),
                    CandidateRoute(["J01", "J03", "J10"], 0.3),
                ],
            ),
            self.trajectory("T2", "J10", 1.0, [CandidateRoute(["J01", "J02", "J10"], 1.0)]),
        ]
        flow_result = ExpectedFlowAggregator(self.graph).aggregate(trajectories)
        demands = self.analyzer.analyze_route_demand(flow_result, self.graph)
        ranked = self.analyzer.top_route_demands(demands)

        self.assertEqual(len(ranked), 2)
        self.assertAlmostEqual(ranked[0].demand, 2.4)
        self.assertAlmostEqual(ranked[1].demand, 0.6)
        self.assertEqual(ranked[0].road_ids, ("R01", "R02"))

    def test_route_probability_validation_is_reused(self) -> None:
        route = CandidateRoute(["J01", "J02", "J10"], 0.7)
        with self.assertRaises(ValueError):
            NormalizedTrajectory("T", "J01", "J10", [route])


if __name__ == "__main__":
    unittest.main()
