"""Tests for Phase 4 network intelligence and integration."""

import unittest
from pathlib import Path

from backend.analytics.network import NetworkIntelligenceAnalyzer, PriorityWeights, UrbanMobilityAnalyzer
from backend.flow.adapters import MockTrajectoryAdapter
from backend.flow.aggregation import ExpectedFlowAggregator
from backend.mobility.graph import MobilityGraph
from backend.mobility.models import RoadSegment
from backend.traffic.metrics import TrafficMetricsCalculator
from backend.traffic.models import CongestionLevel, TrafficMetric

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class TestNetworkIntelligence(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = MobilityGraph(name="NetworkTest")
        self.graph.add_road(RoadSegment("R01", "J01", "J02", 1.0, 40.0, 1000.0))
        self.graph.add_road(RoadSegment("R02", "J02", "J03", 1.0, 40.0, 1000.0))
        self.graph.add_road(RoadSegment("R03", "J01", "J03", 1.0, 40.0, 1000.0))
        self.analyzer = NetworkIntelligenceAnalyzer()

    def metric(self, road_id: str, hourly_flow: float, utilization: float = 0.2) -> TrafficMetric:
        return TrafficMetric(
            road_id=road_id,
            expected_flow=hourly_flow,
            hourly_flow=hourly_flow,
            capacity_vph=100.0,
            utilization_ratio=utilization,
            free_flow_time_minutes=10.0,
            estimated_travel_time_minutes=10.0,
            congestion_score=0.0,
            congestion_level=CongestionLevel.FREE,
        )

    def test_zero_total_flow_is_safe(self) -> None:
        result = self.analyzer.analyze([self.metric("R01", 0.0)], self.graph)
        self.assertEqual(result.total_hourly_flow, 0.0)
        self.assertEqual(result.flow_concentration_hhi, 0.0)
        self.assertEqual(result.road_flow_shares, {})

    def test_hhi_and_shares_use_hourly_flow(self) -> None:
        result = self.analyzer.analyze([self.metric("R01", 10.0), self.metric("R02", 30.0)], self.graph)
        self.assertAlmostEqual(result.road_flow_shares["R01"], 0.25)
        self.assertAlmostEqual(result.road_flow_shares["R02"], 0.75)
        self.assertAlmostEqual(sum(result.road_flow_shares.values()), 1.0)
        self.assertAlmostEqual(result.flow_concentration_hhi, 0.25 ** 2 + 0.75 ** 2)

        even = self.analyzer.analyze([self.metric("R01", 10.0), self.metric("R02", 10.0)], self.graph)
        self.assertAlmostEqual(even.flow_concentration_hhi, 0.5)
        dominant = self.analyzer.analyze([self.metric("R01", 100.0), self.metric("R02", 0.0)], self.graph)
        self.assertAlmostEqual(dominant.flow_concentration_hhi, 1.0)

    def test_structural_importance_and_registered_count(self) -> None:
        result = self.analyzer.analyze([self.metric("R01", 10.0), self.metric("R02", 10.0)], self.graph)
        self.assertEqual(result.road_count, 3)
        self.assertEqual(result.evaluated_roads_count, 2)
        self.assertIn("R01", result.structural_importance)
        self.assertIn("R02", result.structural_importance)
        self.assertGreaterEqual(result.structural_importance["R01"], 0.0)

    def test_priority_exposes_signals_and_weights_are_validated(self) -> None:
        metrics = [self.metric("R01", 10.0, 0.2), self.metric("R02", 30.0, 0.8)]
        network = self.analyzer.analyze(metrics, self.graph)
        priorities = self.analyzer.rank_priority_roads(metrics, network, limit=2)
        self.assertEqual(len(priorities), 2)
        self.assertEqual(priorities[0].road_id, "R02")
        self.assertGreaterEqual(priorities[0].flow_share, priorities[1].flow_share)
        with self.assertRaises(ValueError):
            PriorityWeights(flow_share=-1.0)

    def test_full_synthetic_phase1_to_phase4_pipeline(self) -> None:
        graph = MobilityGraph.load_from_json(PROJECT_ROOT / "data" / "synthetic" / "city_network.json")
        trajectories = MockTrajectoryAdapter().adapt(PROJECT_ROOT / "data" / "synthetic" / "mock_trajectories.json")
        flow_result = ExpectedFlowAggregator(graph).aggregate(trajectories)
        metrics = TrafficMetricsCalculator(graph=graph).calculate_metrics(flow_result)
        result = UrbanMobilityAnalyzer().analyze(flow_result, metrics, graph, top_n=5)

        self.assertEqual(result.od_analysis.trajectory_count, len(trajectories))
        self.assertGreater(result.od_analysis.total_demand, 0.0)
        self.assertGreater(len(result.route_demands), 0)
        self.assertEqual(result.network.road_count, graph.road_count)
        self.assertGreater(result.network.total_hourly_flow, 0.0)
        self.assertEqual(len(result.top_priority_roads), 5)


if __name__ == "__main__":
    unittest.main()
