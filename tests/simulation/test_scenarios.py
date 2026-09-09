"""Tests for isolated counterfactual graph construction."""

import unittest

from backend.mobility.graph import MobilityGraph
from backend.mobility.models import RoadSegment
from backend.simulation.models import Scenario
from backend.simulation.scenarios import ScenarioGraphBuilder


class TestScenarioGraphBuilder(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = MobilityGraph("ScenarioTest")
        self.graph.add_road(RoadSegment("R01", "J01", "J02", 1.0, 60.0, 1000.0))
        self.graph.add_road(RoadSegment("R02", "J02", "J03", 1.0, 60.0, 1000.0))

    def test_closure_does_not_mutate_baseline(self) -> None:
        scenario_graph = ScenarioGraphBuilder().build(self.graph, Scenario("s", "close", closed_road_ids=("R01",)))
        self.assertFalse(self.graph.is_road_closed("R01"))
        self.assertTrue(scenario_graph.is_road_closed("R01"))
        scenario_graph.restore_road("R01")
        self.assertFalse(self.graph.is_road_closed("R01"))

    def test_capacity_and_speed_modifications_are_isolated(self) -> None:
        scenario_graph = ScenarioGraphBuilder().build(
            self.graph,
            Scenario("s", "modify", capacity_modifications_vph={"R01": 500.0}, speed_modifications_kmph={"R01": 30.0}),
        )
        self.assertEqual(self.graph.get_road("R01").capacity_vph, 1000.0)
        self.assertEqual(scenario_graph.get_road("R01").capacity_vph, 500.0)
        self.assertEqual(scenario_graph.get_road("R01").speed_limit_kmph, 30.0)

    def test_unknown_road_rejected(self) -> None:
        with self.assertRaises(KeyError):
            ScenarioGraphBuilder().build(self.graph, Scenario("s", "bad", closed_road_ids=("R99",)))


if __name__ == "__main__":
    unittest.main()
