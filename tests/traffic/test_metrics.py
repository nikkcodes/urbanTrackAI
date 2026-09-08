"""Comprehensive unit tests for Phase 3 traffic metrics and intelligence engine."""

import math
import unittest
from pathlib import Path

from backend.flow.adapters import MockTrajectoryAdapter
from backend.flow.aggregation import ExpectedFlowAggregator
from backend.flow.models import FlowAggregationResult, RoadFlow
from backend.mobility.graph import MobilityGraph
from backend.mobility.models import Node, RoadSegment
from backend.traffic.metrics import TrafficMetricsCalculator
from backend.traffic.models import (
    BPRParameters,
    CongestionLevel,
    CongestionThresholds,
    NetworkTrafficSummary,
    TrafficMetric,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class TestTrafficModels(unittest.TestCase):
    """Test validation and serialization of Phase 3 models."""

    def test_bpr_parameters_validation(self) -> None:
        """Verify BPR parameter bounds and validation."""
        params = BPRParameters(alpha=0.20, beta=4.5)
        self.assertEqual(params.alpha, 0.20)
        self.assertEqual(params.beta, 4.5)

        with self.assertRaises(ValueError):
            BPRParameters(alpha=-0.1)
        with self.assertRaises(ValueError):
            BPRParameters(beta=-1.0)
        with self.assertRaises(ValueError):
            BPRParameters(alpha=float("nan"))
        with self.assertRaises(ValueError):
            BPRParameters(beta=float("inf"))

        data = params.to_dict()
        restored = BPRParameters.from_dict(data)
        self.assertEqual(params.alpha, restored.alpha)
        self.assertEqual(params.beta, restored.beta)

    def test_congestion_thresholds_validation(self) -> None:
        """Verify threshold bounds, ordering, and classification."""
        th = CongestionThresholds(free_limit=0.60, moderate_limit=0.85, heavy_limit=1.05)
        self.assertEqual(th.free_limit, 0.60)

        # Inverted or non-increasing order rejected
        with self.assertRaises(ValueError):
            CongestionThresholds(free_limit=0.90, moderate_limit=0.80, heavy_limit=1.10)
        with self.assertRaises(ValueError):
            CongestionThresholds(free_limit=-0.1, moderate_limit=0.80, heavy_limit=1.10)
        with self.assertRaises(ValueError):
            CongestionThresholds(free_limit=0.70, moderate_limit=0.90, heavy_limit=0.90)

        # Classification tests
        self.assertEqual(th.classify(0.50), CongestionLevel.FREE)
        self.assertEqual(th.classify(0.60), CongestionLevel.MODERATE)
        self.assertEqual(th.classify(0.84), CongestionLevel.MODERATE)
        self.assertEqual(th.classify(0.85), CongestionLevel.HEAVY)
        self.assertEqual(th.classify(1.04), CongestionLevel.HEAVY)
        self.assertEqual(th.classify(1.05), CongestionLevel.SEVERE)
        self.assertEqual(th.classify(1.50), CongestionLevel.SEVERE)

    def test_traffic_metric_validation(self) -> None:
        """Verify TrafficMetric field validation."""
        metric = TrafficMetric(
            road_id="R01",
            expected_flow=100.0,
            capacity_vph=1000.0,
            utilization_ratio=0.1,
            free_flow_time_minutes=2.0,
            estimated_travel_time_minutes=2.0003,
            congestion_score=0.00015,
            congestion_level=CongestionLevel.FREE,
        )
        self.assertEqual(metric.road_id, "R01")
        self.assertEqual(metric.congestion_level, CongestionLevel.FREE)

        # Negative flow rejected
        with self.assertRaises(ValueError):
            TrafficMetric(
                road_id="R01",
                expected_flow=-5.0,
                capacity_vph=1000.0,
                utilization_ratio=0.1,
                free_flow_time_minutes=2.0,
                estimated_travel_time_minutes=2.0,
                congestion_score=0.0,
                congestion_level=CongestionLevel.FREE,
            )

        # Zero or negative capacity rejected
        with self.assertRaises(ValueError):
            TrafficMetric(
                road_id="R01",
                expected_flow=100.0,
                capacity_vph=0.0,
                utilization_ratio=0.1,
                free_flow_time_minutes=2.0,
                estimated_travel_time_minutes=2.0,
                congestion_score=0.0,
                congestion_level=CongestionLevel.FREE,
            )

        # Estimated travel time less than free flow rejected
        with self.assertRaises(ValueError):
            TrafficMetric(
                road_id="R01",
                expected_flow=100.0,
                capacity_vph=1000.0,
                utilization_ratio=0.1,
                free_flow_time_minutes=2.0,
                estimated_travel_time_minutes=1.5,
                congestion_score=0.0,
                congestion_level=CongestionLevel.FREE,
            )

        # Congestion score outside [0, 1] rejected
        with self.assertRaises(ValueError):
            TrafficMetric(
                road_id="R01",
                expected_flow=100.0,
                capacity_vph=1000.0,
                utilization_ratio=0.1,
                free_flow_time_minutes=2.0,
                estimated_travel_time_minutes=2.0,
                congestion_score=1.5,
                congestion_level=CongestionLevel.FREE,
            )

        data = metric.to_dict()
        restored = TrafficMetric.from_dict(data)
        self.assertEqual(metric.road_id, restored.road_id)
        self.assertEqual(metric.congestion_level, restored.congestion_level)


class TestTrafficMetricsCalculator(unittest.TestCase):
    """Test suite for TrafficMetricsCalculator calculations and edge cases."""

    def setUp(self) -> None:
        """Construct standard test network and calculator."""
        self.graph = MobilityGraph(name="TrafficTestNetwork")
        self.graph.add_node(Node(node_id="J01"))
        self.graph.add_node(Node(node_id="J02"))
        self.graph.add_node(Node(node_id="J03"))

        # R01: length 2 km, 60 km/h -> free flow time = 2.0 min, capacity = 1000 vph
        self.segment1 = RoadSegment(
            road_id="R01",
            from_node="J01",
            to_node="J02",
            distance_km=2.0,
            speed_limit_kmph=60.0,
            capacity_vph=1000.0,
            free_flow_time_min=2.0,
        )
        # R02: length 3 km, 60 km/h -> free flow time = 3.0 min, capacity = 1500 vph
        self.segment2 = RoadSegment(
            road_id="R02",
            from_node="J02",
            to_node="J03",
            distance_km=3.0,
            speed_limit_kmph=60.0,
            capacity_vph=1500.0,
            free_flow_time_min=3.0,
        )
        self.graph.add_road(self.segment1)
        self.graph.add_road(self.segment2)

        self.calculator = TrafficMetricsCalculator(graph=self.graph)

    # 1. Normal utilization calculation (flow below capacity)
    def test_utilization_below_capacity(self) -> None:
        """Verify normal utilization ratio calculation when V < C."""
        flow = RoadFlow(road_id="R01", from_node="J01", to_node="J02", expected_flow=500.0)
        metric = self.calculator.calculate_road_metric(flow)

        self.assertAlmostEqual(metric.expected_flow, 500.0)
        self.assertAlmostEqual(metric.capacity_vph, 1000.0)
        self.assertAlmostEqual(metric.utilization_ratio, 0.5)
        self.assertEqual(metric.congestion_level, CongestionLevel.FREE)

    # 2. Flow equal to capacity
    def test_utilization_at_capacity(self) -> None:
        """Verify utilization ratio when V = C (1.0)."""
        flow = RoadFlow(road_id="R01", from_node="J01", to_node="J02", expected_flow=1000.0)
        metric = self.calculator.calculate_road_metric(flow)

        self.assertAlmostEqual(metric.utilization_ratio, 1.0)
        # Default moderate_limit=0.90, heavy_limit=1.10 -> V/C=1.0 is HEAVY
        self.assertEqual(metric.congestion_level, CongestionLevel.HEAVY)

    # 3. Flow above capacity (unclamped)
    def test_utilization_above_capacity(self) -> None:
        """Verify flow above capacity is not arbitrarily clamped."""
        flow = RoadFlow(road_id="R01", from_node="J01", to_node="J02", expected_flow=1500.0)
        metric = self.calculator.calculate_road_metric(flow)

        self.assertAlmostEqual(metric.utilization_ratio, 1.5)
        self.assertGreater(metric.utilization_ratio, 1.0)
        self.assertEqual(metric.congestion_level, CongestionLevel.SEVERE)

    # 4. Zero capacity rejection
    def test_zero_and_invalid_capacity_handling(self) -> None:
        """Verify zero or negative capacity raises clear ValueError, preventing division by zero."""
        invalid_segment = RoadSegment(
            road_id="R_ZERO",
            from_node="J01",
            to_node="J02",
            distance_km=1.0,
            speed_limit_kmph=50.0,
            capacity_vph=100.0,
        )
        # Manually alter capacity attribute to test calculator safety
        object.__setattr__(invalid_segment, "capacity_vph", 0.0)

        flow = RoadFlow(road_id="R_ZERO", from_node="J01", to_node="J02", expected_flow=50.0)
        with self.assertRaises(ValueError) as ctx:
            self.calculator.calculate_road_metric(flow, road_segment=invalid_segment)
        self.assertIn("invalid capacity", str(ctx.exception).lower())

    # 5. Free-flow travel time baseline (V = 0)
    def test_free_flow_travel_time_baseline(self) -> None:
        """Verify that zero flow yields exactly free-flow travel time and zero congestion score."""
        flow = RoadFlow(road_id="R01", from_node="J01", to_node="J02", expected_flow=0.0)
        metric = self.calculator.calculate_road_metric(flow)

        self.assertAlmostEqual(metric.utilization_ratio, 0.0)
        self.assertAlmostEqual(metric.estimated_travel_time_minutes, metric.free_flow_time_minutes)
        self.assertAlmostEqual(metric.congestion_score, 0.0)
        self.assertEqual(metric.congestion_level, CongestionLevel.FREE)

    # 6. BPR travel time formula validation
    def test_bpr_travel_time_calculation(self) -> None:
        """Verify BPR formula: t = t0 * (1 + alpha * (V/C)^beta)."""
        # t0 = 2.0 min, C = 1000 vph, V = 1000 vph -> (V/C) = 1.0
        # t = 2.0 * (1 + 0.15 * (1.0)^4) = 2.0 * 1.15 = 2.30 min
        flow = RoadFlow(road_id="R01", from_node="J01", to_node="J02", expected_flow=1000.0)
        metric = self.calculator.calculate_road_metric(flow)

        expected_t = 2.0 * (1.0 + 0.15 * (1.0 ** 4.0))
        self.assertAlmostEqual(metric.estimated_travel_time_minutes, expected_t, places=6)

        # Expected congestion score = (t - t0) / t = (2.30 - 2.0) / 2.30 = 0.30 / 2.30
        expected_score = (expected_t - 2.0) / expected_t
        self.assertAlmostEqual(metric.congestion_score, expected_score, places=6)

    # 7. Custom alpha and beta parameters
    def test_custom_bpr_parameters(self) -> None:
        """Verify calculator respects custom alpha and beta."""
        custom_params = BPRParameters(alpha=0.25, beta=2.0)
        calc = TrafficMetricsCalculator(graph=self.graph, bpr_params=custom_params)

        # V = 500, C = 1000 -> V/C = 0.5
        # t = 2.0 * (1 + 0.25 * (0.5)^2) = 2.0 * (1 + 0.25 * 0.25) = 2.0 * 1.0625 = 2.125
        flow = RoadFlow(road_id="R01", from_node="J01", to_node="J02", expected_flow=500.0)
        metric = calc.calculate_road_metric(flow)

        expected_t = 2.0 * (1.0 + 0.25 * (0.5 ** 2.0))
        self.assertAlmostEqual(metric.estimated_travel_time_minutes, expected_t, places=6)
        self.assertEqual(metric.metadata["bpr_alpha"], 0.25)
        self.assertEqual(metric.metadata["bpr_beta"], 2.0)

    # 8. Congestion score boundaries
    def test_congestion_score_boundaries(self) -> None:
        """Verify congestion score S is bounded in [0.0, 1.0) and monotonic."""
        flows = [0.0, 100.0, 500.0, 1000.0, 2000.0, 5000.0]
        scores = []
        for v in flows:
            rf = RoadFlow(road_id="R01", from_node="J01", to_node="J02", expected_flow=v)
            m = self.calculator.calculate_road_metric(rf)
            self.assertGreaterEqual(m.congestion_score, 0.0)
            self.assertLess(m.congestion_score, 1.0)
            scores.append(m.congestion_score)

        # Check strict monotonicity
        for i in range(len(scores) - 1):
            self.assertLess(scores[i], scores[i + 1])

    # 9. Congestion level boundaries
    def test_congestion_level_boundaries(self) -> None:
        """Verify classification at exact boundaries (0.70, 0.90, 1.10)."""
        # Capacity = 1000
        # Flow = 699 -> V/C = 0.699 -> FREE
        m_free = self.calculator.calculate_road_metric(
            RoadFlow(road_id="R01", from_node="J01", to_node="J02", expected_flow=699.0)
        )
        self.assertEqual(m_free.congestion_level, CongestionLevel.FREE)

        # Flow = 700 -> V/C = 0.70 -> MODERATE
        m_mod = self.calculator.calculate_road_metric(
            RoadFlow(road_id="R01", from_node="J01", to_node="J02", expected_flow=700.0)
        )
        self.assertEqual(m_mod.congestion_level, CongestionLevel.MODERATE)

        # Flow = 900 -> V/C = 0.90 -> HEAVY
        m_heavy = self.calculator.calculate_road_metric(
            RoadFlow(road_id="R01", from_node="J01", to_node="J02", expected_flow=900.0)
        )
        self.assertEqual(m_heavy.congestion_level, CongestionLevel.HEAVY)

        # Flow = 1100 -> V/C = 1.10 -> SEVERE
        m_sev = self.calculator.calculate_road_metric(
            RoadFlow(road_id="R01", from_node="J01", to_node="J02", expected_flow=1100.0)
        )
        self.assertEqual(m_sev.congestion_level, CongestionLevel.SEVERE)

    # 10. Time-window preservation and windowed metrics
    def test_time_window_preservation(self) -> None:
        """Verify time window metadata is preserved on metrics and summaries."""
        flow = RoadFlow(
            road_id="R01",
            from_node="J01",
            to_node="J02",
            expected_flow=400.0,
            time_window_start="08:00:00",
            time_window_end="08:15:00",
        )
        metric = self.calculator.calculate_road_metric(flow)
        self.assertEqual(metric.time_window_start, "08:00:00")
        self.assertEqual(metric.time_window_end, "08:15:00")

        summary = self.calculator.summarize_network([metric])
        self.assertEqual(summary.time_window_start, "08:00:00")
        self.assertEqual(summary.time_window_end, "08:15:00")

    # 11. Network summary and flow-weighted averages
    def test_network_summary_and_flow_weighted_averages(self) -> None:
        """Verify network summary semantics and exact flow-weighted formula calculations."""
        # Road 1: V = 200, C = 1000 -> u1 = 0.2, t0 = 2.0 -> delay = 0.15*(0.2)^4 = 0.00024 -> t1 = 2.00048
        m1 = self.calculator.calculate_road_metric(
            RoadFlow(road_id="R01", from_node="J01", to_node="J02", expected_flow=200.0)
        )
        # Road 2: V = 600, C = 1500 -> u2 = 0.4, t0 = 3.0 -> delay = 0.15*(0.4)^4 = 0.00384 -> t2 = 3.01152
        m2 = self.calculator.calculate_road_metric(
            RoadFlow(road_id="R02", from_node="J02", to_node="J03", expected_flow=600.0)
        )

        summary = self.calculator.summarize_network([m1, m2])

        # Semantics checks
        self.assertEqual(summary.total_expected_flow, 800.0)
        self.assertEqual(summary.road_count, self.graph.road_count)  # 2 registered roads
        self.assertEqual(summary.active_roads_with_flow, 2)
        self.assertEqual(summary.evaluated_roads_count, 2)

        # Simple averages: (0.2 + 0.4) / 2 = 0.3
        self.assertAlmostEqual(summary.average_utilization, 0.3, places=6)
        expected_simple_time = (m1.estimated_travel_time_minutes + m2.estimated_travel_time_minutes) / 2
        self.assertAlmostEqual(summary.average_travel_time_minutes, expected_simple_time, places=6)

        # Flow-weighted utilization: (200*0.2 + 600*0.4) / 800 = (40 + 240) / 800 = 280 / 800 = 0.35
        expected_fw_util = (200.0 * 0.2 + 600.0 * 0.4) / 800.0
        self.assertAlmostEqual(summary.flow_weighted_utilization, expected_fw_util, places=6)

        # Flow-weighted travel time: sum(V * t) / sum(V)
        expected_fw_time = (200.0 * m1.estimated_travel_time_minutes + 600.0 * m2.estimated_travel_time_minutes) / 800.0
        self.assertAlmostEqual(summary.flow_weighted_travel_time_minutes, expected_fw_time, places=6)

        # Congestion level counts
        self.assertEqual(summary.congestion_level_counts[CongestionLevel.FREE.value], 2)
        self.assertEqual(summary.congestion_level_counts[CongestionLevel.MODERATE.value], 0)

    # 12. Safe behavior when total expected flow is zero
    def test_zero_total_flow_summary_safety(self) -> None:
        """Verify that when total expected flow is zero, flow-weighted values are safely defined as 0.0 without dividing by zero."""
        m1 = self.calculator.calculate_road_metric(
            RoadFlow(road_id="R01", from_node="J01", to_node="J02", expected_flow=0.0)
        )
        summary = self.calculator.summarize_network([m1])

        self.assertEqual(summary.total_expected_flow, 0.0)
        self.assertEqual(summary.active_roads_with_flow, 0)
        self.assertEqual(summary.evaluated_roads_count, 1)
        self.assertAlmostEqual(summary.average_utilization, 0.0)
        self.assertAlmostEqual(summary.flow_weighted_utilization, 0.0)
        self.assertAlmostEqual(summary.flow_weighted_travel_time_minutes, 0.0)

    # 13. Empty input handling
    def test_empty_metrics_summary(self) -> None:
        """Verify summary calculation on an empty list of metrics."""
        summary = self.calculator.summarize_network([])

        self.assertEqual(summary.total_expected_flow, 0.0)
        self.assertEqual(summary.evaluated_roads_count, 0)
        self.assertEqual(summary.active_roads_with_flow, 0)
        self.assertEqual(summary.average_utilization, 0.0)
        self.assertEqual(summary.flow_weighted_utilization, 0.0)
        self.assertEqual(summary.average_travel_time_minutes, 0.0)
        self.assertEqual(summary.flow_weighted_travel_time_minutes, 0.0)

    # 14. Invalid numeric values rejection
    def test_invalid_numeric_values_rejected(self) -> None:
        """Verify non-finite or negative flows raise ValueError."""
        with self.assertRaises(ValueError):
            self.calculator.calculate_road_metric(
                RoadFlow(road_id="R01", from_node="J01", to_node="J02", expected_flow=-10.0)
            )

        # Test NaN flow
        with self.assertRaises(ValueError):
            RoadFlow(road_id="R01", from_node="J01", to_node="J02", expected_flow=float("nan"))

    # 15. Deterministic repeated calculations
    def test_deterministic_repeated_calculations(self) -> None:
        """Verify metrics calculations yield identical outputs across runs."""
        flow = RoadFlow(road_id="R01", from_node="J01", to_node="J02", expected_flow=750.0)
        m1 = self.calculator.calculate_road_metric(flow)
        m2 = self.calculator.calculate_road_metric(flow)

        self.assertEqual(m1.to_dict(), m2.to_dict())

    def test_window_duration_normalizes_flow_before_bpr(self) -> None:
        """Verify a 15-minute count becomes hourly flow before utilization and BPR."""
        segment = RoadSegment(
            road_id="R15",
            from_node="J01",
            to_node="J02",
            distance_km=2.0,
            speed_limit_kmph=60.0,
            capacity_vph=1600.0,
            free_flow_time_min=2.0,
        )
        flow = RoadFlow(
            road_id="R15",
            from_node="J01",
            to_node="J02",
            expected_flow=2.5,
            time_window_start="2026-01-01T08:00:00",
            time_window_end="2026-01-01T08:15:00",
        )

        metric = self.calculator.calculate_road_metric(flow, road_segment=segment)

        self.assertAlmostEqual(metric.hourly_flow, 10.0)
        self.assertAlmostEqual(metric.utilization_ratio, 0.00625)
        expected_time = 2.0 * (1.0 + 0.15 * (0.00625 ** 4.0))
        self.assertAlmostEqual(metric.estimated_travel_time_minutes, expected_time, places=12)
        self.assertAlmostEqual(metric.metadata["aggregation_duration_hours"], 0.25)

    def test_supported_window_durations(self) -> None:
        """Verify 15-, 30-, and 60-minute windows produce the expected hourly rate."""
        for end_time, expected_hourly_flow in [
            ("08:15:00", 10.0),
            ("08:30:00", 5.0),
            ("09:00:00", 2.5),
        ]:
            flow = RoadFlow(
                road_id="R01",
                from_node="J01",
                to_node="J02",
                expected_flow=2.5,
                time_window_start="08:00:00",
                time_window_end=end_time,
            )
            metric = self.calculator.calculate_road_metric(flow)
            self.assertAlmostEqual(metric.hourly_flow, expected_hourly_flow)

    def test_invalid_time_windows_rejected(self) -> None:
        """Reject reversed, zero-duration, malformed, partial, and incompatible windows."""
        invalid_windows = [
            ("08:15:00", "08:00:00"),
            ("08:00:00", "08:00:00"),
            ("not-a-time", "08:15:00"),
            ("2026-01-01T08:00:00", "08:15:00"),
            ("08:00:00", None),
        ]
        for start, end in invalid_windows:
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                self.calculator.calculate_road_metric(
                    RoadFlow(
                        road_id="R01",
                        from_node="J01",
                        to_node="J02",
                        expected_flow=2.5,
                        time_window_start=start,
                        time_window_end=end,
                    )
                )

    def test_missing_window_uses_explicit_configured_duration(self) -> None:
        """A missing window uses only the calculator's documented fallback duration."""
        calculator = TrafficMetricsCalculator(graph=self.graph, default_aggregation_duration_hours=0.5)
        flow = RoadFlow(road_id="R01", from_node="J01", to_node="J02", expected_flow=2.5)

        metric = calculator.calculate_road_metric(flow)

        self.assertAlmostEqual(metric.hourly_flow, 5.0)
        self.assertAlmostEqual(metric.metadata["aggregation_duration_hours"], 0.5)

    def test_network_weighting_uses_hourly_flow_for_mixed_windows(self) -> None:
        """Verify network weighting uses compatible hourly units across different windows."""
        m1 = self.calculator.calculate_road_metric(
            RoadFlow(
                road_id="R01",
                from_node="J01",
                to_node="J02",
                expected_flow=2.5,
                time_window_start="08:00:00",
                time_window_end="08:15:00",
            )
        )
        m2 = self.calculator.calculate_road_metric(
            RoadFlow(
                road_id="R02",
                from_node="J02",
                to_node="J03",
                expected_flow=10.0,
                time_window_start="08:00:00",
                time_window_end="09:00:00",
            )
        )

        summary = self.calculator.summarize_network([m1, m2])

        self.assertAlmostEqual(summary.total_expected_flow, 12.5)
        self.assertEqual(summary.road_count, 2)
        self.assertEqual(summary.active_roads_with_flow, 2)
        self.assertEqual(summary.evaluated_roads_count, 2)
        expected_utilization = (10.0 * m1.utilization_ratio + 10.0 * m2.utilization_ratio) / 20.0
        expected_travel_time = (10.0 * m1.estimated_travel_time_minutes + 10.0 * m2.estimated_travel_time_minutes) / 20.0
        self.assertAlmostEqual(summary.flow_weighted_utilization, expected_utilization)
        self.assertAlmostEqual(summary.flow_weighted_travel_time_minutes, expected_travel_time)

    # 16. Full end-to-end integration with synthetic city network and mock trajectories
    def test_end_to_end_phase1_phase2_phase3_integration(self) -> None:
        """Verify integration: city_network.json -> mock_trajectories.json -> Phase 2 flow -> Phase 3 metrics."""
        city_json = PROJECT_ROOT / "data" / "synthetic" / "city_network.json"
        mock_json = PROJECT_ROOT / "data" / "synthetic" / "mock_trajectories.json"

        graph = MobilityGraph.load_from_json(city_json)
        adapter = MockTrajectoryAdapter()
        trajectories = adapter.adapt(mock_json)

        # Phase 2 Aggregator
        aggregator = ExpectedFlowAggregator(graph=graph)
        flow_result = aggregator.aggregate(trajectories, include_zero_flow_roads=False)

        # Phase 3 Calculator
        calculator = TrafficMetricsCalculator(graph=graph)
        metrics = calculator.calculate_metrics(flow_result)

        self.assertEqual(len(metrics), len(flow_result.flows))

        for m in metrics:
            self.assertGreater(m.capacity_vph, 0.0)
            self.assertGreater(m.free_flow_time_minutes, 0.0)
            self.assertGreaterEqual(m.estimated_travel_time_minutes, m.free_flow_time_minutes)
            self.assertGreaterEqual(m.congestion_score, 0.0)
            self.assertLess(m.congestion_score, 1.0)
            self.assertIn(m.congestion_level, [CongestionLevel.FREE, CongestionLevel.MODERATE, CongestionLevel.HEAVY, CongestionLevel.SEVERE])

        summary = calculator.summarize_network(metrics)
        self.assertEqual(summary.road_count, graph.road_count)  # 28 roads
        self.assertEqual(summary.evaluated_roads_count, len(metrics))
        self.assertGreater(summary.total_expected_flow, 0.0)
        self.assertGreater(summary.flow_weighted_travel_time_minutes, 0.0)
        self.assertGreater(summary.flow_weighted_utilization, 0.0)

        # Windowed metrics
        windowed_flows = aggregator.aggregate_by_time_window(trajectories, include_zero_flow_roads=False)
        windowed_metrics = calculator.calculate_windowed_metrics(windowed_flows)
        self.assertGreater(len(windowed_metrics), 0)


if __name__ == "__main__":
    unittest.main()
