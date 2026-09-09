"""Tests for deterministic Phase 6 route assignment."""

import unittest

from backend.flow.models import CandidateRoute, NormalizedTrajectory
from backend.mobility.graph import MobilityGraph
from backend.mobility.models import RoadSegment
from backend.simulation.models import Scenario
from backend.simulation.routing import DeterministicRouter
from backend.simulation.scenarios import ScenarioGraphBuilder


class TestDeterministicRouter(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = MobilityGraph("RoutingTest")
        self.graph.add_road(RoadSegment("R01", "J01", "J02", 1.0, 60.0, 1000.0))
        self.graph.add_road(RoadSegment("R02", "J02", "J03", 1.0, 60.0, 1000.0))
        self.graph.add_road(RoadSegment("R03", "J01", "J04", 1.0, 30.0, 1000.0))
        self.graph.add_road(RoadSegment("R04", "J04", "J03", 1.0, 30.0, 1000.0))
        self.trajectory = NormalizedTrajectory(
            "T1", "J01", "J03",
            [CandidateRoute(["J01", "J02", "J03"], 0.7), CandidateRoute(["J01", "J04", "J03"], 0.3)],
            vehicle_weight=10.0,
        )

    def test_baseline_preserves_probability_weighted_routes(self) -> None:
        outcome = DeterministicRouter().assign_baseline([self.trajectory], self.graph)
        self.assertEqual(len(outcome.assignments), 2)
        self.assertAlmostEqual(sum(item.demand for item in outcome.assignments), 10.0)
        self.assertEqual(outcome.assignments[0].road_ids, ("R01", "R02"))

    def test_closed_routes_reroute_deterministically(self) -> None:
        baseline = DeterministicRouter().assign_baseline([self.trajectory], self.graph)
        scenario_graph = ScenarioGraphBuilder().build(self.graph, Scenario("s", "close", closed_road_ids=("R01", "R02")))
        outcome = DeterministicRouter().assign_counterfactual([self.trajectory], baseline.assignments, scenario_graph)
        self.assertEqual(len(outcome.unroutable), 0)
        self.assertEqual(sum(item.demand for item in outcome.assignments if item.route_changed), 7.0)
        self.assertEqual(sum(item.demand for item in outcome.assignments if not item.route_changed), 3.0)
        self.assertEqual(sum(item.demand for item in outcome.assignments), 10.0)

    def test_disconnected_od_is_explicitly_unroutable(self) -> None:
        baseline = DeterministicRouter().assign_baseline([self.trajectory], self.graph)
        scenario_graph = ScenarioGraphBuilder().build(self.graph, Scenario("s", "disconnect", closed_road_ids=("R01", "R02", "R03", "R04")))
        outcome = DeterministicRouter().assign_counterfactual([self.trajectory], baseline.assignments, scenario_graph)
        self.assertGreater(sum(item.demand for item in outcome.unroutable), 0.0)


if __name__ == "__main__":
    unittest.main()
