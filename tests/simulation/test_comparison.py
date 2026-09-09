"""Tests for Phase 6 impact comparison."""

import unittest

from backend.simulation.comparison import ScenarioComparator
from backend.simulation.models import RecommendationClass, Scenario
from backend.traffic.models import CongestionLevel, NetworkTrafficSummary, TrafficMetric


def metric(road_id: str, flow: float, utilization: float, level: CongestionLevel, travel_time: float) -> TrafficMetric:
    return TrafficMetric(
        road_id=road_id,
        expected_flow=flow,
        hourly_flow=flow,
        capacity_vph=100.0,
        utilization_ratio=utilization,
        free_flow_time_minutes=10.0,
        estimated_travel_time_minutes=travel_time,
        congestion_score=max(0.0, (travel_time - 10.0) / travel_time),
        congestion_level=level,
    )


def summary(utilization: float, travel_time: float) -> NetworkTrafficSummary:
    return NetworkTrafficSummary(
        total_expected_flow=100.0,
        road_count=2,
        active_roads_with_flow=1,
        evaluated_roads_count=2,
        average_utilization=utilization,
        flow_weighted_utilization=utilization,
        average_travel_time_minutes=travel_time,
        flow_weighted_travel_time_minutes=travel_time,
        congestion_level_counts={"FREE": 1, "MODERATE": 0, "HEAVY": 1, "SEVERE": 0},
    )


class TestScenarioComparator(unittest.TestCase):
    def test_road_deltas_and_new_bottleneck(self) -> None:
        comparator = ScenarioComparator()
        impacts = comparator.road_impacts(
            [metric("R01", 100.0, 0.5, CongestionLevel.FREE, 10.0)],
            [metric("R01", 180.0, 1.0, CongestionLevel.HEAVY, 11.5)],
        )
        self.assertAlmostEqual(impacts[0].flow_delta, 80.0)
        self.assertAlmostEqual(impacts[0].utilization_delta, 0.5)
        self.assertTrue(impacts[0].became_new_bottleneck)

    def test_recommendation_classes(self) -> None:
        comparator = ScenarioComparator()
        scenario = Scenario("s", "test")
        impacts = comparator.road_impacts(
            [metric("R01", 100.0, 0.5, CongestionLevel.FREE, 10.0)],
            [metric("R01", 180.0, 1.0, CongestionLevel.HEAVY, 11.5)],
        )
        decision = comparator.decision(scenario, impacts, (), summary(0.5, 10.0), summary(1.0, 11.5), ())
        self.assertEqual(decision.recommendation_class, RecommendationClass.UNFAVORABLE)
        self.assertIn("increased", decision.explanation)

    def test_deterministic_empty_comparison(self) -> None:
        comparator = ScenarioComparator()
        self.assertEqual(comparator.road_impacts([], []), ())


if __name__ == "__main__":
    unittest.main()
