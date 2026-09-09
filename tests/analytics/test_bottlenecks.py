"""Tests for deterministic Phase 4 bottleneck analysis."""

import unittest

from backend.analytics.bottlenecks import BottleneckDetector
from backend.traffic.models import CongestionLevel, TrafficMetric


class TestBottleneckDetector(unittest.TestCase):
    def metric(self, road_id: str, utilization: float, level: CongestionLevel, congestion: float = 0.0, delay: float = 0.0) -> TrafficMetric:
        return TrafficMetric(
            road_id=road_id,
            expected_flow=utilization * 100.0,
            hourly_flow=utilization * 100.0,
            capacity_vph=100.0,
            utilization_ratio=utilization,
            free_flow_time_minutes=10.0,
            estimated_travel_time_minutes=10.0 + delay,
            congestion_score=congestion,
            congestion_level=level,
        )

    def test_free_moderate_heavy_severe_levels_are_retained(self) -> None:
        metrics = [
            self.metric("FREE", 0.2, CongestionLevel.FREE),
            self.metric("MOD", 0.75, CongestionLevel.MODERATE, 0.01),
            self.metric("HEAVY", 1.0, CongestionLevel.HEAVY, 0.1, 2.0),
            self.metric("SEV", 1.2, CongestionLevel.SEVERE, 0.2, 5.0),
        ]
        results = BottleneckDetector().detect(metrics)

        self.assertEqual([item.road_id for item in results], ["SEV", "HEAVY", "MOD", "FREE"])
        self.assertEqual(results[0].severity, CongestionLevel.SEVERE.value)
        self.assertIn("utilization_at_or_above_capacity", results[0].signals)

    def test_higher_congestion_and_delay_increase_score(self) -> None:
        detector = BottleneckDetector()
        low = self.metric("LOW", 0.8, CongestionLevel.MODERATE, congestion=0.1, delay=1.0)
        high_congestion = self.metric("HIGH_C", 0.8, CongestionLevel.MODERATE, congestion=0.2, delay=1.0)
        high_delay = self.metric("HIGH_D", 0.8, CongestionLevel.MODERATE, congestion=0.1, delay=2.0)
        scores = {item.road_id: item.bottleneck_score for item in detector.detect([low, high_congestion, high_delay])}

        self.assertGreater(scores["HIGH_C"], scores["LOW"])
        self.assertGreater(scores["HIGH_D"], scores["LOW"])

    def test_empty_input_and_invalid_metric(self) -> None:
        detector = BottleneckDetector()
        self.assertEqual(detector.detect([]), ())
        with self.assertRaises(TypeError):
            detector.detect([object()])
        with self.assertRaises(ValueError):
            BottleneckDetector(utilization_weight=0.0, congestion_weight=0.0, delay_weight=0.0)

    def test_missing_hourly_flow_is_rejected(self) -> None:
        metric = TrafficMetric(
            road_id="R1",
            expected_flow=1.0,
            capacity_vph=100.0,
            utilization_ratio=0.01,
            free_flow_time_minutes=10.0,
            estimated_travel_time_minutes=10.0,
            congestion_score=0.0,
            congestion_level=CongestionLevel.FREE,
        )
        with self.assertRaises(ValueError):
            BottleneckDetector().detect([metric])


if __name__ == "__main__":
    unittest.main()
