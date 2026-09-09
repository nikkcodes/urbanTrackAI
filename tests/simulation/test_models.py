"""Tests for Phase 6 simulation models."""

import unittest

from backend.simulation.models import RecommendationClass, Scenario


class TestScenarioModels(unittest.TestCase):
    def test_valid_scenario_serialization(self) -> None:
        scenario = Scenario(
            scenario_id="close-R01",
            name="Close R01",
            description="Test closure",
            closed_road_ids=("R01",),
            capacity_modifications_vph={"R02": 900.0},
            speed_modifications_kmph={"R03": 30.0},
        )
        data = scenario.to_dict()
        self.assertEqual(data["closed_road_ids"], ["R01"])
        self.assertEqual(data["capacity_modifications_vph"]["R02"], 900.0)
        self.assertEqual(RecommendationClass.UNFAVORABLE.value, "UNFAVORABLE")

    def test_invalid_scenario_values(self) -> None:
        with self.assertRaises(ValueError):
            Scenario("", "name")
        with self.assertRaises(ValueError):
            Scenario("s", "name", closed_road_ids=("R01", "R01"))
        with self.assertRaises(ValueError):
            Scenario("s", "name", capacity_modifications_vph={"R01": -1.0})
        with self.assertRaises(ValueError):
            Scenario("s", "name", speed_modifications_kmph={"R01": 0.0})


if __name__ == "__main__":
    unittest.main()
