"""Tests for Phase 6 simulation orchestration."""

import unittest

from backend.flow.models import CandidateRoute, NormalizedTrajectory
from backend.mobility.graph import MobilityGraph
from backend.mobility.models import RoadSegment
from backend.simulation.engine import CounterfactualSimulationEngine
from backend.simulation.models import RecommendationClass, Scenario


class TestCounterfactualSimulationEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = MobilityGraph("EngineTest")
        self.graph.add_road(RoadSegment("R01", "J01", "J02", 1.0, 60.0, 100.0))
        self.graph.add_road(RoadSegment("R02", "J02", "J03", 1.0, 60.0, 100.0))
        self.graph.add_road(RoadSegment("R03", "J01", "J04", 1.0, 30.0, 100.0))
        self.graph.add_road(RoadSegment("R04", "J04", "J03", 1.0, 30.0, 100.0))
        self.trajectories = [NormalizedTrajectory(
            "T1", "J01", "J03",
            [CandidateRoute(["J01", "J02", "J03"], 1.0)],
            vehicle_weight=10.0,
        )]
        self.engine = CounterfactualSimulationEngine(aggregation_duration_hours=1.0)

    def test_baseline_scenario_flow_and_rerouting(self) -> None:
        result = self.engine.simulate(self.graph, self.trajectories, Scenario("close", "Close R01", closed_road_ids=("R01",)))
        impacts = {item.road_id: item for item in result.road_impacts}
        self.assertAlmostEqual(impacts["R01"].scenario_hourly_flow, 0.0)
        self.assertGreater(impacts["R03"].scenario_hourly_flow, impacts["R03"].baseline_hourly_flow)
        self.assertGreater(result.decision.rerouted_demand, 0.0)
        self.assertFalse(self.graph.is_road_closed("R01"))
        self.assertTrue(result.scenario_graph.is_road_closed("R01"))
        self.assertEqual(result.decision.recommendation_class, RecommendationClass.UNFAVORABLE)

    def test_multiple_scenarios_are_independent(self) -> None:
        results = self.engine.simulate_many(
            self.graph,
            self.trajectories,
            [Scenario("b", "Close R03", closed_road_ids=("R03",)), Scenario("a", "Close R01", closed_road_ids=("R01",))],
        )
        self.assertEqual(len(results), 2)
        self.assertFalse(self.graph.is_road_closed("R01"))
        self.assertFalse(self.graph.is_road_closed("R03"))
        self.assertNotEqual(results[0].scenario.scenario_id, results[1].scenario.scenario_id)

    def test_unroutable_demand_makes_decision_unfavorable(self) -> None:
        result = self.engine.simulate(
            self.graph,
            self.trajectories,
            Scenario("disconnect", "Disconnect", closed_road_ids=("R01", "R02", "R03", "R04")),
        )
        self.assertGreater(result.decision.unroutable_demand, 0.0)
        self.assertFalse(result.decision.feasible)
        self.assertEqual(result.decision.recommendation_class, RecommendationClass.UNFAVORABLE)


if __name__ == "__main__":
    unittest.main()
